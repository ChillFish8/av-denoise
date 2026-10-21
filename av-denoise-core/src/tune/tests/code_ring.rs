use cubecl::prelude::*;
use cubecl::server::Handle;

use super::engine::{noisy_frames, run_engine, temporal_fast};
use super::probe::small_device_limits;
use super::runtime::{TestRuntime, test_client};
use crate::bench_api::HostIo;
use crate::engine::{DevicePlane, Engine, Geometry, SampleFormat};
use crate::nlmeans::kernels::gpu_cast_codes;
use crate::nlmeans::{
    ChannelMode,
    DenoisingMode,
    MotionCompensationMode,
    NlmDenoiser,
    NlmParams,
    NlmeansAlgorithm,
    NlmeansOptions,
    PrefilterMode,
};
use crate::tune::gpu::{read_f32, sync};
use crate::tune::{EngineKernels, IntegerCodes, PipelineProfile};
use crate::{Nl4d, Nl4dOptions, Nlmeans};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const ENGINE_WIDTH: u32 = 96;
const ENGINE_HEIGHT: u32 = 64;
const RING_RADIUS: u32 = 2;
const RING_FRAMES: usize = 2 * RING_RADIUS as usize + 1;
const MOTION_RADIUS: u32 = 1;
const MOTION_FRAMES: usize = 2 * MOTION_RADIUS as usize + 1;

const EIGHT_BIT: IntegerCodes = IntegerCodes { depth: 8 };
const TEN_BIT: IntegerCodes = IntegerCodes { depth: 10 };

fn hashed_code(index: u32, seed: u32, max_code: u32) -> u32 {
    let mut hash = index.wrapping_mul(0x9E37_79B9) ^ seed.wrapping_mul(0x85EB_CA6B);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x2C1B_3C6D);
    hash ^= hash >> 12;

    hash % (max_code + 1)
}

/// `len` normalised codes, with code 0 and the top code at the first two indices.
fn code_frame(len: usize, codes: IntegerCodes, seed: u32) -> Vec<f32> {
    let max_code = codes.max_code();
    let max_value = max_code as f32;

    (0..len as u32)
        .map(|index| {
            let code = match index {
                0 => 0,
                1 => max_code,
                _ => hashed_code(index, seed, max_code),
            };
            code as f32 / max_value
        })
        .collect()
}

fn read_codes(client: &ComputeClient<TestRuntime>, handle: &Handle, len: usize) -> Vec<u16> {
    let owned = handle.clone();
    let bytes = client.read_one(owned).expect("code readback failed");
    let values = u16::from_bytes(&bytes);

    values[..len].to_vec()
}

/// Checks every code is `values` scaled to codes and rounded, as the cast rounds them.
fn assert_codes_round(values: &[f32], codes_read: &[u16], codes: IntegerCodes, label: &str) {
    let max_value = codes.max_code() as f32;

    for (index, (value, code)) in values.iter().zip(codes_read).enumerate() {
        let rounded = value * max_value + 0.5;
        let bounded = rounded.clamp(0.0, max_value);
        let expected = bounded as u16;

        assert_eq!(*code, expected, "{label}: element {index} of value {value}");
    }
}

/// Reads the input and code rings back and checks the code ring mirrors every slot.
fn assert_code_ring_mirrors_input(front: &NlmDenoiser<TestRuntime>, label: &str) {
    let client = front.compute_client();
    let (width, height, channels) = front.frame_shape();
    let ring_len = (width * height * channels.storage_count()) as usize * RING_FRAMES;
    let (code_handle, codes) = front.code_ring().expect("the code ring is enabled");

    let input_ring = front.input_ring();
    let input = read_f32(client, input_ring, ring_len).expect("input readback failed");
    let codes_read = read_codes(client, code_handle, ring_len);

    assert_codes_round(&input, &codes_read, codes, label);
}

fn ring_params(channels: ChannelMode) -> NlmParams {
    NlmParams {
        temporal_radius: RING_RADIUS,
        channels,
        ..NlmParams::default()
    }
}

fn cast_codes_back(codes: IntegerCodes) -> Vec<u16> {
    let client = test_client();
    let max_code = codes.max_code();
    let count = max_code as usize + 1;
    let max_value = max_code as f32;
    let normalised: Vec<f32> = (0..=max_code).map(|code| code as f32 / max_value).collect();
    let normalised_bytes = f32::as_bytes(&normalised);
    let source = client.create_from_slice(normalised_bytes);
    let target = client.empty(count * size_of::<u16>());
    let cube_count = CubeCount::new_1d(4);
    let cube_dim = CubeDim::new_1d(64);
    let source_handle = source.clone();
    let target_handle = target.clone();

    unsafe {
        let source_arg = ArrayArg::from_raw_parts(source_handle, count);
        let target_arg = ArrayArg::from_raw_parts(target_handle, count);

        gpu_cast_codes::launch_unchecked::<u16, TestRuntime>(
            &client,
            cube_count,
            cube_dim,
            source_arg,
            target_arg,
            max_value,
            0,
            count as u32,
            4 * 64,
        );
    }

    read_codes(&client, &target, count)
}

#[test]
fn only_the_integer_search_profile_keeps_codes() {
    let mut limits = small_device_limits();
    limits.u16_buffer = true;
    let formats = [SampleFormat::U8, SampleFormat::U16 { depth: 10 }];

    for profile in PipelineProfile::ALL {
        for format in formats {
            let codes = IntegerCodes::for_profile(profile, format, &limits);
            let expected = if profile == PipelineProfile::IntegerSearch {
                IntegerCodes::from_format(format)
            } else {
                None
            };

            assert_eq!(codes, expected, "{profile:?} {format:?}");
        }
    }

    limits.u16_buffer = false;
    let without_storage =
        IntegerCodes::for_profile(PipelineProfile::IntegerSearch, SampleFormat::U8, &limits);

    assert_eq!(without_storage, None);
}

#[test]
fn gpu_cast_codes_rounds_normalised_codes_back_to_codes() {
    for codes in [EIGHT_BIT, TEN_BIT] {
        let cast = cast_codes_back(codes);
        let max_code = codes.max_code() as u16;
        let expected: Vec<u16> = (0..=max_code).collect();

        assert_eq!(cast, expected, "{codes:?}");
    }
}

#[test]
fn gpu_cast_codes_clamps_out_of_range_samples() {
    let client = test_client();
    let samples = [-0.75f32, -0.001, 1.001, 4.0, 0.0, 1.0, 0.5, 0.25];
    let count = samples.len();
    let samples_bytes = f32::as_bytes(&samples);
    let source = client.create_from_slice(samples_bytes);
    let target = client.empty(count * size_of::<u16>());
    let cube_count = CubeCount::new_1d(1);
    let cube_dim = CubeDim::new_1d(64);
    let source_handle = source.clone();
    let target_handle = target.clone();

    unsafe {
        let source_arg = ArrayArg::from_raw_parts(source_handle, count);
        let target_arg = ArrayArg::from_raw_parts(target_handle, count);

        gpu_cast_codes::launch_unchecked::<u16, TestRuntime>(
            &client,
            cube_count,
            cube_dim,
            source_arg,
            target_arg,
            1023.0,
            0,
            count as u32,
            64,
        );
    }

    let cast = read_codes(&client, &target, count);

    assert_eq!(cast, [0, 0, 1023, 1023, 0, 1023, 512, 256]);
}

#[test]
fn the_code_ring_mirrors_every_pushed_and_duplicated_slot() {
    let client = test_client();

    for codes in [EIGHT_BIT, TEN_BIT] {
        for channels in [ChannelMode::Luma, ChannelMode::Yuv] {
            let params = ring_params(channels);
            let mut front = NlmDenoiser::<TestRuntime>::new(&client, params, WIDTH, HEIGHT);
            front.enable_code_ring(codes);

            // The first push primes two leading copies, so four pushes write every slot.
            let frame_len = (WIDTH * HEIGHT * channels.count()) as usize;
            for seed in 0..4 {
                let frame = code_frame(frame_len, codes, seed);
                front.push_frame(&frame);
            }

            let label = format!("{codes:?} {channels:?}");
            assert_code_ring_mirrors_input(&front, &label);
        }
    }
}

#[test]
fn the_code_ring_follows_a_stream_reset() {
    let client = test_client();

    for codes in [EIGHT_BIT, TEN_BIT] {
        let channels = ChannelMode::Luma;
        let params = ring_params(channels);
        let mut front = NlmDenoiser::<TestRuntime>::new(&client, params, WIDTH, HEIGHT);
        front.enable_code_ring(codes);

        let frame_len = (WIDTH * HEIGHT) as usize;
        for seed in 0..3 {
            let frame = code_frame(frame_len, codes, seed);
            front.push_frame(&frame);
        }

        front.reset_stream_state();

        for seed in 10..13 {
            let frame = code_frame(frame_len, codes, seed);
            front.push_frame(&frame);
        }

        let label = format!("{codes:?} after a reset");
        assert_code_ring_mirrors_input(&front, &label);
    }
}

#[test]
fn motion_compensation_mirrors_the_compensated_codes() {
    let client = test_client();

    for codes in [EIGHT_BIT, TEN_BIT] {
        let params = NlmParams {
            temporal_radius: MOTION_RADIUS,
            channels: ChannelMode::Luma,
            motion_compensation: MotionCompensationMode::mvtools_default(),
            ..NlmParams::default()
        };
        let mut front = NlmDenoiser::<TestRuntime>::new(&client, params, WIDTH, HEIGHT);
        front.enable_code_ring(codes);

        let frame_len = (WIDTH * HEIGHT) as usize;
        for seed in 0..4 {
            let frame = code_frame(frame_len, codes, seed);
            front.push_frame(&frame);
        }

        let output = front.denoise().expect("the frame denoises");
        assert!(output.is_some(), "{codes:?}: the window is full");

        // The centre copy and the warp of each neighbour write every slot of the compensated ring.
        let ring_len = frame_len * MOTION_FRAMES;
        let compensated = front
            .compensated_input_ring_for_test()
            .expect("motion compensation allocates the compensated ring");
        let compensated_codes = front
            .compensated_code_ring()
            .expect("motion compensation allocates the compensated code ring");

        let values = read_f32(&client, compensated, ring_len).expect("compensated readback failed");
        let codes_read = read_codes(&client, compensated_codes, ring_len);

        let label = format!("{codes:?} compensated");
        assert_codes_round(&values, &codes_read, codes, &label);
    }
}

fn geometry(channels: ChannelMode, input: SampleFormat) -> Geometry {
    Geometry {
        width: ENGINE_WIDTH,
        height: ENGINE_HEIGHT,
        channels,
        input,
        output: SampleFormat::F32,
    }
}

fn profile_kernels(profile: PipelineProfile) -> EngineKernels {
    EngineKernels {
        profile,
        ..EngineKernels::default()
    }
}

fn motion_compensated_nlmeans() -> NlmeansAlgorithm {
    let options = NlmeansOptions {
        mode: DenoisingMode::Temporal {
            radius: MOTION_RADIUS,
        },
        motion_compensation: MotionCompensationMode::mvtools_default(),
        ..NlmeansOptions::default()
    };

    NlmeansAlgorithm::Fast(options)
}

#[test]
fn integer_search_engines_build_the_code_rings_their_passes_read() {
    let client = test_client();
    let ten_bit = SampleFormat::U16 { depth: 10 };

    let luma_geometry = geometry(ChannelMode::Luma, SampleFormat::U8);
    let algorithm = motion_compensated_nlmeans();
    let nlmeans_kernels = profile_kernels(PipelineProfile::IntegerSearch);
    let nlmeans = Nlmeans::with_kernels(&client, algorithm, luma_geometry, nlmeans_kernels)
        .expect("the nlmeans engine builds");
    let nlmeans_front = nlmeans.front_for_test();
    let nlmeans_codes = nlmeans_front.code_ring().map(|(_, codes)| codes);
    let has_compensated_codes = nlmeans_front.compensated_code_ring().is_some();

    assert_eq!(nlmeans_codes, Some(EIGHT_BIT));
    assert!(has_compensated_codes);

    let yuv_geometry = geometry(ChannelMode::Yuv, ten_bit);
    let options = Nl4dOptions::default();
    let nl4d_kernels = profile_kernels(PipelineProfile::IntegerSearch);
    let nl4d =
        Nl4d::with_kernels(&client, options, yuv_geometry, nl4d_kernels).expect("the nl4d engine builds");
    let nl4d_denoiser = nl4d.denoiser_for_test();
    let nl4d_front = nl4d_denoiser.front_for_test();
    let nl4d_codes = nl4d_front.code_ring().map(|(_, codes)| codes);
    let integer_codes = nl4d_denoiser.integer_codes();
    let has_compensated_codes = nl4d_front.compensated_code_ring().is_some();

    assert_eq!(integer_codes, Some(TEN_BIT));
    assert_eq!(nl4d_codes, Some(TEN_BIT));
    assert!(!has_compensated_codes);

    sync(&client).expect("the device syncs");
}

#[test]
fn integer_search_skips_deep_and_prefiltered_sources() {
    let client = test_client();
    let twelve_bit = SampleFormat::U16 { depth: 12 };

    let deep_geometry = geometry(ChannelMode::Luma, twelve_bit);
    let options = Nl4dOptions::default();
    let nl4d_kernels = profile_kernels(PipelineProfile::IntegerSearch);
    let nl4d =
        Nl4d::with_kernels(&client, options, deep_geometry, nl4d_kernels).expect("the nl4d engine builds");
    let nl4d_denoiser = nl4d.denoiser_for_test();
    let deep_codes = nl4d_denoiser.integer_codes();
    let deep_has_ring = nl4d_denoiser.front_for_test().code_ring().is_some();

    assert_eq!(deep_codes, None);
    assert!(!deep_has_ring);

    let prefiltered = NlmeansOptions {
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 3.0,
            sigma_r: 0.02,
        },
        ..NlmeansOptions::default()
    };
    let algorithm = NlmeansAlgorithm::Fast(prefiltered);
    let eight_bit_geometry = geometry(ChannelMode::Luma, SampleFormat::U8);
    let nlmeans_kernels = profile_kernels(PipelineProfile::IntegerSearch);
    let nlmeans = Nlmeans::with_kernels(&client, algorithm, eight_bit_geometry, nlmeans_kernels)
        .expect("the nlmeans engine builds");
    let prefiltered_has_ring = nlmeans.front_for_test().code_ring().is_some();

    assert!(!prefiltered_has_ring);

    sync(&client).expect("the device syncs");
}

/// The sample at this pixel of the 10-bit plane carries stray bits above the top code.
const STRAY_PIXEL: usize = 7;

/// One plane of `pixels` codes counting up from 0 and wrapping past the top code.
fn counting_codes(pixels: usize, codes: IntegerCodes) -> Vec<u16> {
    let wrap = codes.max_code() as usize + 1;

    (0..pixels).map(|pixel| (pixel % wrap) as u16).collect()
}

/// Pushes `plane` as one luma plane in `format` and reads the code ring's first slot back.
fn ingested_codes(
    client: &ComputeClient<TestRuntime>,
    format: SampleFormat,
    plane: Handle,
    pixels: usize,
) -> (Vec<u16>, IntegerCodes) {
    let luma_geometry = geometry(ChannelMode::Luma, format);
    let options = NlmeansOptions::default();
    let algorithm = NlmeansAlgorithm::Fast(options);
    let kernels = profile_kernels(PipelineProfile::IntegerSearch);
    let mut nlmeans =
        Nlmeans::with_kernels(client, algorithm, luma_geometry, kernels).expect("the nlmeans engine builds");

    let device_plane = DevicePlane::new(&plane, ENGINE_WIDTH, ENGINE_HEIGHT);
    nlmeans.push(&[device_plane]).expect("the frame pushes");

    let (code_handle, codes) = nlmeans
        .front_for_test()
        .code_ring()
        .expect("an integer source builds a code ring");
    let codes_read = read_codes(client, code_handle, pixels);

    (codes_read, codes)
}

#[test]
fn ingested_integer_planes_land_in_the_code_ring_as_their_codes() {
    let client = test_client();
    let pixels = (ENGINE_WIDTH * ENGINE_HEIGHT) as usize;

    let eight_bit_codes = counting_codes(pixels, EIGHT_BIT);
    let eight_bit_samples: Vec<u8> = eight_bit_codes.iter().map(|&code| code as u8).collect();
    let eight_bit_plane = client.create_from_slice(&eight_bit_samples);
    let (eight_bit_read, eight_bit_kind) = ingested_codes(&client, SampleFormat::U8, eight_bit_plane, pixels);

    assert_eq!(eight_bit_kind, EIGHT_BIT);
    assert_eq!(eight_bit_read, eight_bit_codes, "8-bit codes");

    let mut ten_bit_codes = counting_codes(pixels, TEN_BIT);
    let mut ten_bit_samples = ten_bit_codes.clone();
    ten_bit_samples[STRAY_PIXEL] = 0xFFFF;
    ten_bit_codes[STRAY_PIXEL] = 1023;
    let ten_bit_bytes = u16::as_bytes(&ten_bit_samples);
    let ten_bit_plane = client.create_from_slice(ten_bit_bytes);
    let ten_bit_format = SampleFormat::U16 { depth: 10 };
    let (ten_bit_read, ten_bit_kind) = ingested_codes(&client, ten_bit_format, ten_bit_plane, pixels);

    assert_eq!(ten_bit_kind, TEN_BIT);
    assert_eq!(ten_bit_read[STRAY_PIXEL], 1023, "a stray-high-bits sample");
    assert_eq!(ten_bit_read, ten_bit_codes, "10-bit codes");
}

#[test]
fn a_float_source_builds_no_code_ring() {
    let client = test_client();
    let modes = [
        (ChannelMode::Luma, 1usize),
        (ChannelMode::Chroma, 2),
        (ChannelMode::Yuv, 3),
    ];

    for (channels, planes) in modes {
        let frames = noisy_frames(ENGINE_WIDTH, ENGINE_HEIGHT, planes, 8);
        let float_geometry = geometry(channels, SampleFormat::F32);

        let integer_kernels = profile_kernels(PipelineProfile::IntegerSearch);
        let integer_algorithm = temporal_fast();
        let mut nlmeans = Nlmeans::with_kernels(&client, integer_algorithm, float_geometry, integer_kernels)
            .expect("the nlmeans engine builds");
        let nlmeans_has_ring = nlmeans.front_for_test().code_ring().is_some();
        assert!(!nlmeans_has_ring, "{channels:?} nlmeans");

        let integer_nlmeans = run_engine(&client, &mut nlmeans, ENGINE_WIDTH, ENGINE_HEIGHT, &frames)
            .expect("the nlmeans frames denoise");

        let baseline_kernels = profile_kernels(PipelineProfile::Baseline);
        let baseline_algorithm = temporal_fast();
        let mut baseline =
            Nlmeans::with_kernels(&client, baseline_algorithm, float_geometry, baseline_kernels)
                .expect("the baseline engine builds");
        let baseline_nlmeans = run_engine(&client, &mut baseline, ENGINE_WIDTH, ENGINE_HEIGHT, &frames)
            .expect("the baseline frames denoise");

        assert!(!baseline_nlmeans.is_empty(), "{channels:?} nlmeans emits frames");
        assert_eq!(integer_nlmeans, baseline_nlmeans, "{channels:?} nlmeans");

        let integer_kernels = profile_kernels(PipelineProfile::IntegerSearch);
        let integer_options = Nl4dOptions::default();
        let mut nl4d = Nl4d::with_kernels(&client, integer_options, float_geometry, integer_kernels)
            .expect("the nl4d engine builds");
        let nl4d_denoiser = nl4d.denoiser_for_test();
        let nl4d_codes = nl4d_denoiser.integer_codes();
        let nl4d_has_ring = nl4d_denoiser.front_for_test().code_ring().is_some();
        assert_eq!(nl4d_codes, None, "{channels:?} nl4d");
        assert!(!nl4d_has_ring, "{channels:?} nl4d");

        let integer_nl4d = run_engine(&client, &mut nl4d, ENGINE_WIDTH, ENGINE_HEIGHT, &frames)
            .expect("the nl4d frames denoise");

        let f32_kernels = profile_kernels(PipelineProfile::F32Search);
        let f32_options = Nl4dOptions::default();
        let mut f32_nl4d = Nl4d::with_kernels(&client, f32_options, float_geometry, f32_kernels)
            .expect("the f32 nl4d engine builds");
        let f32_outputs = run_engine(&client, &mut f32_nl4d, ENGINE_WIDTH, ENGINE_HEIGHT, &frames)
            .expect("the f32 nl4d frames denoise");

        assert!(!f32_outputs.is_empty(), "{channels:?} nl4d emits frames");
        assert_eq!(integer_nl4d, f32_outputs, "{channels:?} nl4d");
    }
}

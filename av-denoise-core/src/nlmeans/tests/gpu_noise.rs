use cubecl::prelude::*;

use super::helpers::{make_client, seeded_unit_gaussian};
use crate::bench_api::HostIo;
use crate::collab::kernels::aggregate::weight_scale_with_g_max;
use crate::motion::sad_noise_floor;
use crate::nl4d::{Nl4dDenoiser, Nl4dParams, PsyParams};
use crate::nlmeans::dispatch::{mc_sad_noise_floor_sigma, pilot_h2_inv_norm};
use crate::nlmeans::params::sigma_eff;
use crate::nlmeans::*;
use crate::noise::tests::fold_reference::relative_gap;
use crate::noise::{
    DerivedValues,
    NoiseConsumer,
    NoiseSnapshot,
    STRENGTH_MAP_OFF,
    StrengthMapKind,
    build_spatial_offset_lut,
    spatial_offset_factor,
    spatial_offset_index,
};

// Large enough that each luma bin the ramp crosses gathers the quarters a curve needs.
const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;

/// The largest relative gap a coefficient formula may show against the host formula.
const MAX_FORMULA_GAP: f32 = 1.0e-6;

/// One scene of [grain_frames]: a vertical brightness ramp, top to bottom or flipped.
#[derive(Clone, Copy)]
struct Scene {
    flipped: bool,
    seed: u32,
}

/// `count` frames of a static ramp with a textured band, carrying fresh grain per frame whose
/// strength grows with brightness, in `channels` interleaved planes.
fn grain_frames(channels: u32, count: u32, scene: Scene) -> Vec<Vec<f32>> {
    let mut frames = Vec::with_capacity(count as usize);

    for frame_index in 0..count {
        let mut frame = vec![0.0f32; (WIDTH * HEIGHT * channels) as usize];

        for y in 0..HEIGHT {
            let fraction = y as f32 / (HEIGHT - 1) as f32;
            let position = if scene.flipped { 1.0 - fraction } else { fraction };
            let luma = 0.15 + 0.7 * position;
            let noise_std = 0.004 + 0.02 * luma;

            for x in 0..WIDTH {
                let in_band = x < WIDTH / 4;
                let wave = (x as f32 * 0.7).sin() * (y as f32 * 0.5).cos();
                let texture = if in_band { 0.08 * wave } else { 0.0 };

                for channel in 0..channels {
                    let pixel = y * WIDTH + x;
                    let seed = (scene.seed + frame_index) * channels + channel;
                    let noise = seeded_unit_gaussian(pixel, seed);
                    let sample = luma + texture + noise * noise_std;
                    frame[(pixel * channels + channel) as usize] = sample.clamp(0.0, 1.0);
                }
            }
        }

        frames.push(frame);
    }

    frames
}

fn hq_params(channels: ChannelMode, temporal_radius: u32, motion: MotionCompensationMode) -> NlmParams {
    NlmParams {
        temporal_radius,
        channels,
        prefilter: PrefilterMode::None,
        motion_compensation: motion,
        hq: Some(HqParams::default()),
        ..NlmParams::default()
    }
}

fn nl4d_params(channels: ChannelMode) -> Nl4dParams {
    let defaults = Nl4dParams::default();
    let nlm = NlmParams {
        channels,
        prefilter: PrefilterMode::None,
        ..defaults.nlm
    };

    Nl4dParams {
        nlm,
        noise_map: true,
        psy: Some(PsyParams::default()),
        ..defaults
    }
}

/// The host formulas' derived values for `median` and `low` on `front`.
fn host_formulas(front: &NlmDenoiser, median: [f32; 3], low: [f32; 3]) -> DerivedValues {
    let params = &front.params;
    let count = params.channels.count() as usize;
    let effective_sigma = sigma_eff(&median[..count], params.channels);
    let h2_inv_norm = params.h2_inv_norm_with(Some(effective_sigma));
    let input_noise_offset = params.noise_offset_with(Some(&low[..count]));

    let (pilot_h2_inv_norm, noise_offset) = match params.prefilter {
        PrefilterMode::NlmSpatial { strength_scale } => {
            let pilot = pilot_h2_inv_norm(h2_inv_norm, strength_scale);
            (pilot, 0.0)
        },
        _ => (h2_inv_norm, input_noise_offset),
    };

    let sad_floor = front.mc_ctx.as_ref().map_or(0.0, |motion_ctx| {
        let sigma = mc_sad_noise_floor_sigma(params.prefilter, median[0]);
        sad_noise_floor(motion_ctx.blksize, sigma)
    });
    let confidence_sad_floor = front
        .confidence_ctx
        .as_ref()
        .map_or(0.0, |geometry| sad_noise_floor(geometry.blksize, median[0]));

    DerivedValues {
        sad_noise_floor: sad_floor,
        confidence_sad_noise_floor: confidence_sad_floor,
        h2_inv_norm,
        pilot_h2_inv_norm,
        noise_offset,
        input_noise_offset,
        ..DerivedValues::default()
    }
}

/// Every mix of auto strength, the noise floor, the NLM pilot, motion compensation and the luma
/// or yuv plane, with temporal confidence on.
fn coefficient_cases() -> Vec<NlmParams> {
    let prefilters = [
        PrefilterMode::None,
        PrefilterMode::NlmSpatial { strength_scale: 0.8 },
    ];
    let motions = [
        MotionCompensationMode::None,
        MotionCompensationMode::mvtools_default(),
    ];
    let mut cases = Vec::new();

    for auto_strength in [true, false] {
        for noise_floor in [true, false] {
            let hq = HqParams {
                auto_strength,
                noise_floor,
                ..HqParams::default()
            };

            for prefilter in prefilters {
                for motion in motions {
                    for channels in [ChannelMode::Luma, ChannelMode::Yuv] {
                        let params = NlmParams {
                            prefilter,
                            hq: Some(hq),
                            ..hq_params(channels, 1, motion)
                        };
                        cases.push(params);
                    }
                }
            }
        }
    }

    cases
}

#[test]
fn the_coefficients_reproduce_the_host_formulas() {
    let client = make_client();
    let triples = [
        [0.01, 0.02, 0.03],
        [4.0 / 255.0, 2.5 / 255.0, 3.1 / 255.0],
        [0.2, 0.15, 0.1],
    ];
    let mut checked = 0;

    for params in coefficient_cases() {
        let front_params = params.clone();
        let front = NlmDenoiser::new(&client, front_params, 64, 64);
        let noise = front.measured_noise().expect("the front measures its noise");
        let coefficients = noise.settings().fold.coefficients;
        let count = params.channels.count() as usize;

        for median in triples {
            let low = median.map(|sigma| 0.8 * sigma);
            let snapshot = NoiseSnapshot {
                median: Some(median),
                low: Some(low),
                low_unboosted: Some(low),
                temporal_only: None,
                rho: None,
                curve: None,
                classes_present: false,
                derived: DerivedValues::default(),
            };
            let consumer = NoiseConsumer::default();
            let gpu = coefficients.derive_host(&snapshot, count, consumer, StrengthMapKind::Off);
            let host = host_formulas(&front, median, low);
            let pairs = [
                ("h2_inv_norm", gpu.h2_inv_norm, host.h2_inv_norm),
                ("pilot_h2_inv_norm", gpu.pilot_h2_inv_norm, host.pilot_h2_inv_norm),
                ("noise_offset", gpu.noise_offset, host.noise_offset),
                (
                    "input_noise_offset",
                    gpu.input_noise_offset,
                    host.input_noise_offset,
                ),
                ("sad_noise_floor", gpu.sad_noise_floor, host.sad_noise_floor),
                (
                    "confidence_sad_noise_floor",
                    gpu.confidence_sad_noise_floor,
                    host.confidence_sad_noise_floor,
                ),
            ];

            for (name, gpu_value, host_value) in pairs {
                let gap = relative_gap(gpu_value, host_value);
                assert!(
                    gap <= MAX_FORMULA_GAP,
                    "{name} is {gap:e} apart for {params:?} at {median:?}: {gpu_value:e} vs {host_value:e}"
                );
            }

            checked += 1;
        }
    }

    assert_eq!(checked, 96, "every configuration and triple is checked");
}

/// Reads back the derived values the front's kernels bind.
fn bound_derived(front: &NlmDenoiser) -> DerivedValues {
    let handle = front.derived_handle();
    let bytes = front
        .client
        .read_one(handle)
        .expect("the derived values read back");
    let words = f32::from_bytes(&bytes);

    DerivedValues::from_words(words)
}

#[test]
fn the_separable_offset_reads_the_spatial_table_entry() {
    let search_radius = 3u32;
    let radius = search_radius as i32;
    let rho = 0.4f32;
    let noise_offset = 0.0137f32;
    let table = build_spatial_offset_lut(search_radius, rho, noise_offset);
    let mut checked = 0;

    for q_y in -radius..=radius {
        for q_x in -radius..=radius {
            let offset_index = spatial_offset_index(search_radius, q_x, q_y);
            let factor = spatial_offset_factor(q_x, q_y, rho);
            let expected = noise_offset * factor;

            let entry_bits = table[offset_index as usize].to_bits();
            let expected_bits = expected.to_bits();
            assert_eq!(entry_bits, expected_bits, "candidate {q_x}, {q_y}");

            checked += 1;
        }
    }

    assert_eq!(checked, table.len(), "every candidate is checked");
}

#[test]
fn an_nl4d_pilot_strength_follows_the_gpu_state() {
    let scene = Scene {
        flipped: false,
        seed: 13,
    };
    let frames = grain_frames(1, 6, scene);
    let defaults = nl4d_params(ChannelMode::Luma);
    let strength_scale = DEFAULT_PILOT_STRENGTH_SCALE;
    let nlm = NlmParams {
        prefilter: PrefilterMode::NlmSpatial { strength_scale },
        ..defaults.nlm
    };
    let params = Nl4dParams { nlm, ..defaults };
    let client = make_client();
    let mut denoiser = Nl4dDenoiser::new(&client, params, WIDTH, HEIGHT).expect("the denoiser builds");
    let initial_front = denoiser.front_for_test();
    let initial = bound_derived(initial_front);

    for frame in &frames {
        denoiser.push_frame(frame);
        denoiser.denoise().expect("the submit runs");
    }

    let folded_front = denoiser.front_for_test();
    let folded = bound_derived(folded_front);
    let expected = pilot_h2_inv_norm(folded.h2_inv_norm, strength_scale);
    let gap = relative_gap(folded.pilot_h2_inv_norm, expected);

    assert_ne!(folded.pilot_h2_inv_norm, initial.pilot_h2_inv_norm);
    assert!(
        gap <= MAX_FORMULA_GAP,
        "pilot h2 {} against {expected}",
        folded.pilot_h2_inv_norm
    );
}

/// The radius 0 front, and a radius 2 front with motion compensation and the NLM pilot.
fn gpu_noise_cases() -> [NlmParams; 2] {
    let spatial = hq_params(ChannelMode::Luma, 0, MotionCompensationMode::None);
    let motion = MotionCompensationMode::mvtools_default();
    let prefilter = PrefilterMode::NlmSpatial {
        strength_scale: DEFAULT_PILOT_STRENGTH_SCALE,
    };
    let temporal = NlmParams {
        prefilter,
        ..hq_params(ChannelMode::Luma, 2, motion)
    };

    [spatial, temporal]
}

/// Pushes `frames` through `denoiser` and returns every output frame, the flush included.
fn nlmeans_outputs(denoiser: &mut NlmDenoiser, frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut outputs = Vec::new();

    for frame in frames {
        denoiser.push_frame(frame);

        let denoised = denoiser.denoise().expect("the submit runs");
        outputs.extend(denoised);
    }

    denoiser
        .flush(|frame| outputs.push(frame.to_vec()))
        .expect("the flush runs");

    outputs
}

fn output_bits(outputs: &[Vec<f32>]) -> Vec<Vec<u32>> {
    outputs
        .iter()
        .map(|frame| frame.iter().map(|sample| sample.to_bits()).collect())
        .collect()
}

#[test]
fn nlmeans_hq_output_is_byte_identical_across_runs() {
    let scene = Scene {
        flipped: false,
        seed: 17,
    };
    let frames = grain_frames(1, 8, scene);
    let client = make_client();

    for params in gpu_noise_cases() {
        let first_params = params.clone();
        let mut first = NlmDenoiser::new(&client, first_params, WIDTH, HEIGHT);
        let first_outputs = nlmeans_outputs(&mut first, &frames);

        let second_params = params.clone();
        let mut second = NlmDenoiser::new(&client, second_params, WIDTH, HEIGHT);
        let second_outputs = nlmeans_outputs(&mut second, &frames);

        let first_bits = output_bits(&first_outputs);
        let second_bits = output_bits(&second_outputs);
        let output_count = first_bits.len();

        assert_eq!(
            output_count,
            frames.len(),
            "every frame is emitted for {params:?}"
        );
        assert!(first_bits == second_bits, "the two runs differ for {params:?}");
    }
}

/// The sigma every pinned front in [a_pinned_front_binds_the_host_formula_values] holds.
const PINNED_SIGMA: f32 = 3.0 / 255.0;

/// The derived values the host formulas give `front` at the pinned `sigma`.
fn pinned_formulas(front: &NlmDenoiser, sigma: f32) -> DerivedValues {
    let params = &front.params;
    let consumer = front.noise_settings_for_test().fold.consumer;
    let weight = weight_scale_with_g_max(sigma, consumer.weight_g_max);
    let count = params.channels.count() as usize;
    let sigmas = [sigma; 3];
    let h2_inv_norm = params.h2_inv_norm_with(Some(sigma));
    let input_noise_offset = params.noise_offset_with(Some(&sigmas[..count]));

    let (pilot_h2_inv_norm, noise_offset) = match params.prefilter {
        PrefilterMode::NlmSpatial { strength_scale } => {
            let pilot = pilot_h2_inv_norm(h2_inv_norm, strength_scale);
            (pilot, 0.0)
        },
        _ => (h2_inv_norm, input_noise_offset),
    };

    let sad_floor = front.mc_ctx.as_ref().map_or(0.0, |motion_ctx| {
        let floor_sigma = mc_sad_noise_floor_sigma(params.prefilter, sigma);
        sad_noise_floor(motion_ctx.blksize, floor_sigma)
    });
    let confidence_sad_floor = front
        .confidence_ctx
        .as_ref()
        .map_or(0.0, |geometry| sad_noise_floor(geometry.blksize, sigma));

    DerivedValues {
        sigma: sigmas,
        weight_scale: weight,
        curve_valid: false,
        map_mode: STRENGTH_MAP_OFF,
        sad_noise_floor: sad_floor,
        confidence_sad_noise_floor: confidence_sad_floor,
        h2_inv_norm,
        pilot_h2_inv_norm,
        noise_offset,
        input_noise_offset,
        rho: 0.0,
    }
}

/// Checks every word `front` binds against [pinned_formulas], bit for bit.
fn assert_binds_pinned_formulas(front: &NlmDenoiser, what: &str) -> DerivedValues {
    let bound = bound_derived(front);
    let expected = pinned_formulas(front, PINNED_SIGMA);
    let bound_words = bound.to_words();
    let expected_words = expected.to_words();
    let bound_bits = bound_words.map(f32::to_bits);
    let expected_bits = expected_words.map(f32::to_bits);

    assert_eq!(
        bound_bits, expected_bits,
        "{what}: bound {bound:?}\nexpected {expected:?}"
    );

    bound
}

#[test]
fn a_pinned_front_binds_the_host_formula_values() {
    let scene = Scene {
        flipped: false,
        seed: 23,
    };
    let frames = grain_frames(1, 4, scene);
    let client = make_client();
    let pinned_hq = HqParams::with_sigma(PINNED_SIGMA);

    let defaults = nl4d_params(ChannelMode::Luma);
    let nl4d_hq = defaults.nlm.hq.map(|hq| HqParams {
        sigma_override: Some(PINNED_SIGMA),
        ..hq
    });
    let nl4d_nlm = NlmParams {
        hq: nl4d_hq,
        ..defaults.nlm
    };
    let nl4d = Nl4dParams {
        nlm: nl4d_nlm,
        ..defaults
    };
    let mut nl4d_denoiser = Nl4dDenoiser::new(&client, nl4d, WIDTH, HEIGHT).expect("the denoiser builds");
    for frame in &frames {
        nl4d_denoiser.push_frame(frame);
        nl4d_denoiser.denoise().expect("the submit runs");
    }

    let nl4d_front = nl4d_denoiser.front_for_test();
    let nl4d_bound = assert_binds_pinned_formulas(nl4d_front, "pinned nl4d");
    assert!(
        nl4d_bound.sad_noise_floor > 0.0,
        "the nl4d front runs motion compensation"
    );

    let prefilter = PrefilterMode::NlmSpatial {
        strength_scale: DEFAULT_PILOT_STRENGTH_SCALE,
    };
    let motion = MotionCompensationMode::mvtools_default();
    let compensated = NlmParams {
        prefilter,
        hq: Some(pinned_hq),
        ..hq_params(ChannelMode::Luma, 1, motion)
    };
    let confidence_only = NlmParams {
        hq: Some(pinned_hq),
        ..hq_params(ChannelMode::Luma, 1, MotionCompensationMode::None)
    };
    let mut bound_values = Vec::new();

    for (params, what) in [
        (compensated, "pinned nlmeans with MC and the pilot"),
        (confidence_only, "pinned nlmeans with confidence only"),
    ] {
        let mut denoiser = NlmDenoiser::new(&client, params, WIDTH, HEIGHT);
        for frame in &frames {
            denoiser.push_frame(frame);
            denoiser.denoise().expect("the submit runs");
        }

        let bound = assert_binds_pinned_formulas(&denoiser, what);
        bound_values.push(bound);
    }

    // The pilot's residual fraction is 0, so the MC front's SAD floor is 0. The nl4d front pins a
    // nonzero one.
    let compensated_bound = bound_values[0];
    let confidence_bound = bound_values[1];
    assert!(
        compensated_bound.input_noise_offset > 0.0,
        "the pilot keeps its input offset"
    );
    assert_eq!(
        compensated_bound.noise_offset, 0.0,
        "the main pass compares pilot outputs"
    );
    assert_ne!(
        compensated_bound.pilot_h2_inv_norm, compensated_bound.h2_inv_norm,
        "the pilot scales its strength"
    );
    assert!(
        confidence_bound.confidence_sad_noise_floor > 0.0,
        "the confidence-only front pins its SAD floor"
    );
    assert!(confidence_bound.weight_scale > 0.0, "the weight scale is set");
}

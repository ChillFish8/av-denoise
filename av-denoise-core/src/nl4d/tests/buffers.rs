use cubecl::prelude::*;
use cubecl::server::Handle;

use super::helpers::{
    R,
    RAMP_HEIGHT,
    RAMP_WIDTH,
    make_client,
    noisy_frames,
    ramp_clip,
    ramp_params,
    static_clip_params,
};
use crate::bench_api::HostIo;
use crate::collab::geometry::strength_map_dims;
use crate::collab::kernels::aggregate::weight_scale;
use crate::collab::kernels::transforms::dct_noise_profile;
use crate::nl4d::{Nl4dDenoiser, Nl4dParams};
use crate::nlmeans::mirror::MirroredBuffer;
use crate::nlmeans::motion::sad_noise_floor;
use crate::nlmeans::{ChannelMode, HqParams, NlmDenoiser, mc_sad_noise_floor_sigma};
use crate::noise::tests::fold_reference::{SnapshotGap, relative_gap};
use crate::noise::tests::frames::MAX_RELATIVE;
use crate::noise::{
    DerivedValues,
    NOISE_CURVE_BINS,
    NoiseSnapshot,
    STRENGTH_MAP_ALL,
    STRENGTH_MAP_LUMA,
    STRENGTH_MAP_OFF,
};

const SIZE: u32 = 64;

fn run_frames(params: Nl4dParams, width: u32, height: u32, frames: &[Vec<f32>]) -> Nl4dDenoiser<R> {
    let client = make_client();
    let mut denoiser = Nl4dDenoiser::new(&client, params, width, height).expect("valid params");

    for frame in frames {
        denoiser.push_frame(frame);
        denoiser.denoise().expect("denoise");
    }

    denoiser
}

fn run_stream(params: Nl4dParams, count: usize) -> Nl4dDenoiser<R> {
    let frames = noisy_frames(SIZE, SIZE, 1, count);
    run_frames(params, SIZE, SIZE, &frames)
}

fn read_handle<T: CubeElement>(handle: Handle) -> Vec<T> {
    let client = make_client();
    let bytes = client.read_one(handle).expect("the buffer reads back");

    T::from_bytes(&bytes).to_vec()
}

fn read_derived(denoiser: &Nl4dDenoiser<R>) -> DerivedValues {
    let front = denoiser.front_for_test();
    let handle = front.derived_handle();
    let words: Vec<f32> = read_handle(handle);

    DerivedValues::from_words(&words)
}

fn gpu_snapshot(denoiser: &Nl4dDenoiser<R>) -> NoiseSnapshot {
    denoiser
        .noise_snapshot()
        .expect("every front has a noise snapshot")
}

/// The derived values the fold's formulas give for the chains, curve and classes of `snapshot`.
fn derived_from_chains(front: &NlmDenoiser<R>, snapshot: &NoiseSnapshot) -> DerivedValues {
    let (_, _, channel_mode) = front.frame_shape();
    let channels = channel_mode.count() as usize;
    let settings = front.noise_settings_for_test();
    let coefficients = settings.fold.coefficients;
    let consumer = settings.fold.consumer;
    let map = settings.reading.map;

    coefficients.derive_host(snapshot, channels, consumer, map)
}

/// Runs the ramp clip and checks the derived curve flag and map mode against `expected` and against
/// the rules the fold applies to the GPU chains.
fn assert_curve_flag_and_map_mode(channels: ChannelMode, noise_map: bool, expected: (bool, u32)) {
    let frames = ramp_clip(channels.count());
    let params = ramp_params(channels, noise_map, 1.0);
    let denoiser = run_frames(params, RAMP_WIDTH, RAMP_HEIGHT, &frames);
    let derived = read_derived(&denoiser);
    let snapshot = gpu_snapshot(&denoiser);
    let front = denoiser.front_for_test();
    let rules = derived_from_chains(front, &snapshot);

    let actual = (derived.curve_valid, derived.map_mode);
    let fold_rules = (rules.curve_valid, rules.map_mode);
    assert_eq!(actual, expected);
    assert_eq!(actual, fold_rules);
}

fn read_words<T: CubeElement>(mirror: &MirroredBuffer) -> Vec<T> {
    let handle = mirror.handle().clone();
    read_handle(handle)
}

#[test]
fn a_pinned_sigma_binds_a_zero_curve_and_a_unit_map() {
    let params = static_clip_params(1);
    let denoiser = run_stream(params, 10);
    let front = denoiser.front_for_test();

    let curve_handle = front.noise_curve_handle();
    let map_handle = front.strength_map_handle();
    let curve: Vec<f32> = read_handle(curve_handle);
    let map: Vec<f32> = read_handle(map_handle);

    let (map_cols, map_rows) = strength_map_dims(SIZE, SIZE);
    let map_len = (map_cols * map_rows) as usize;
    let zero_curve = vec![0.0f32; NOISE_CURVE_BINS];
    let unit_map = vec![1.0f32; map_len];

    assert_eq!(curve, zero_curve);
    assert_eq!(map, unit_map);
}

#[test]
fn an_estimated_derived_buffer_holds_the_gpu_sigma_and_weight_scale() {
    let mut params = static_clip_params(1);
    params.nlm.hq = Some(HqParams::default());
    let denoiser = run_stream(params, 10);
    let front = denoiser.front_for_test();

    let sigmas = front.current_sigmas_temporal_only();
    let profile = dct_noise_profile(0.0);
    let expected_scale = weight_scale(sigmas[0], &profile);
    let derived = read_derived(&denoiser);

    let sigma_bits = derived.sigma[0].to_bits();
    let expected_sigma_bits = sigmas[0].to_bits();
    let scale_gap = relative_gap(derived.weight_scale, expected_scale);

    assert!(sigmas[0] > 0.0, "the noisy stream measures a sigma");
    assert_eq!(sigma_bits, expected_sigma_bits);
    assert!(
        scale_gap <= MAX_RELATIVE,
        "the weight scale is {scale_gap:e} apart"
    );
}

#[test]
fn the_derived_buffer_follows_the_gpu_chains() {
    for channels in [ChannelMode::Luma, ChannelMode::Chroma] {
        let frames = ramp_clip(channels.count());
        let mut sampled = 0;
        let params = ramp_params(channels, true, 1.0);
        let client = make_client();
        let mut denoiser =
            Nl4dDenoiser::<R>::new(&client, params, RAMP_WIDTH, RAMP_HEIGHT).expect("valid params");

        for (submit, frame) in frames.iter().enumerate() {
            denoiser.push_frame(frame);
            denoiser.denoise().expect("denoise");

            let snapshot = gpu_snapshot(&denoiser);
            if snapshot.temporal_only.is_some() {
                sampled += 1;
            }

            let bound = read_derived(&denoiser);
            let front = denoiser.front_for_test();
            let expected = derived_from_chains(front, &snapshot);
            let reference = NoiseSnapshot {
                derived: expected,
                ..snapshot.clone()
            };
            let gpu = NoiseSnapshot {
                derived: bound,
                ..snapshot
            };
            let (_, _, channel_mode) = front.frame_shape();
            let count = channel_mode.count() as usize;
            let gap = SnapshotGap::measure(&gpu, &reference, count);
            let structure_matches = gap.structure_matches;
            let derived_gap = gap.derived;

            assert!(
                structure_matches,
                "{channels:?} submit {submit}: flags or map mode"
            );
            assert!(
                derived_gap <= MAX_RELATIVE,
                "{channels:?} submit {submit}: derived values are {derived_gap:e} apart"
            );
        }

        assert!(sampled > 0, "{channels:?}: a submit folds a temporal sample");
    }
}

#[test]
fn the_curve_buffer_holds_the_gpu_curve() {
    let frames = ramp_clip(1);
    let params = ramp_params(ChannelMode::Luma, true, 1.0);
    let client = make_client();
    let mut denoiser =
        Nl4dDenoiser::<R>::new(&client, params, RAMP_WIDTH, RAMP_HEIGHT).expect("valid params");
    let mut curve_submits = 0;

    for frame in &frames {
        denoiser.push_frame(frame);
        denoiser.denoise().expect("denoise");

        let snapshot = gpu_snapshot(&denoiser);
        let front = denoiser.front_for_test();
        let curve_handle = front.noise_curve_handle();
        let words: Vec<f32> = read_handle(curve_handle);

        let expected = match snapshot.curve {
            Some((ratios, _)) if snapshot.derived.curve_valid => {
                curve_submits += 1;
                ratios.to_vec()
            },
            _ => vec![0.0f32; NOISE_CURVE_BINS],
        };
        assert_eq!(words, expected);
    }

    assert!(curve_submits > 0, "the ramp clip forms a curve");
}

#[test]
fn a_luma_front_with_the_noise_map_and_psy_applies_the_curve_and_the_luma_map() {
    assert_curve_flag_and_map_mode(ChannelMode::Luma, true, (true, STRENGTH_MAP_LUMA));
}

#[test]
fn a_luma_front_without_the_noise_map_applies_neither_the_curve_nor_a_map() {
    assert_curve_flag_and_map_mode(ChannelMode::Luma, false, (false, STRENGTH_MAP_OFF));
}

#[test]
fn a_chroma_front_with_the_flat_boost_maps_every_channel_without_a_curve() {
    assert_curve_flag_and_map_mode(ChannelMode::Chroma, true, (false, STRENGTH_MAP_ALL));
}

#[test]
fn the_neighbour_table_follows_the_pass_centre() {
    let params = static_clip_params(1);
    let denoiser = run_stream(params, 10);
    let front = denoiser.front_for_test();
    let table = front
        .neighbour_slots_for_test()
        .expect("a temporal window has neighbours");

    let expected = [front.ring_slot(0), front.ring_slot(2)];
    let actual: Vec<u32> = read_words(table);
    assert_eq!(actual, expected);
}

#[test]
fn a_pinned_sigma_derived_buffer_holds_the_host_values() {
    let params = static_clip_params(1);
    let pinned = params.nlm.hq.and_then(|hq| hq.sigma_override);
    let sigma = pinned.expect("the clip pins its sigma");
    let prefilter = params.nlm.prefilter;
    let denoiser = run_stream(params, 10);
    let front = denoiser.front_for_test();

    let blksize = front.motion_ctx().blksize;
    let floor_sigma = mc_sad_noise_floor_sigma(prefilter, sigma);
    let expected_floor = sad_noise_floor(blksize, floor_sigma);
    let handle = front.derived_handle();
    let client = make_client();
    let bytes = client.read_one(handle).expect("the derived buffer reads back");
    let words = f32::from_bytes(&bytes);
    let derived = DerivedValues::from_words(words);

    let floor_bits = derived.sad_noise_floor.to_bits();
    let expected_floor_bits = expected_floor.to_bits();
    let sigma_bits = derived.sigma[0].to_bits();
    let pinned_bits = sigma.to_bits();

    assert_eq!(floor_bits, expected_floor_bits);
    assert_eq!(sigma_bits, pinned_bits);
}

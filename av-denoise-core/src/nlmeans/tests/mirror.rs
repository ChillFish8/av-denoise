use cubecl::prelude::*;
use cubecl::server::Handle;

use super::helpers::{R, correlated_noisy_frame, make_client, make_noisy_gaussian_frame};
use crate::bench_api::HostIo;
use crate::nlmeans::mirror::MirroredBuffer;
use crate::nlmeans::*;
use crate::noise::tests::fold_reference::relative_gap;
use crate::noise::tests::frames::MAX_RELATIVE;
use crate::noise::{DerivedValues, build_spatial_offset_lut};

const SIZE: u32 = 64;

fn read_floats(client: &ComputeClient<R>, mirror: &MirroredBuffer) -> Vec<f32> {
    let handle = mirror.handle().clone();
    let bytes = client.read_one(handle).expect("the mirror reads back");
    f32::from_bytes(&bytes).to_vec()
}

#[test]
fn an_unchanged_update_uploads_nothing() {
    let client = make_client();
    let values = [0.5f32, 0.25, 0.125];
    let bytes = f32::as_bytes(&values);
    let mut mirror = MirroredBuffer::new(&client, bytes);

    mirror.update(&client, bytes);

    let actual = read_floats(&client, &mirror);
    assert_eq!(mirror.uploads(), 1);
    assert_eq!(actual, values);
}

#[test]
fn a_changed_update_reaches_the_buffer() {
    let client = make_client();
    let first = [0.5f32, 0.25, 0.125];
    let second = [1.0f32, 2.0, 3.0];
    let first_bytes = f32::as_bytes(&first);
    let second_bytes = f32::as_bytes(&second);
    let mut mirror = MirroredBuffer::new(&client, first_bytes);

    mirror.update(&client, second_bytes);

    let actual = read_floats(&client, &mirror);
    assert_eq!(mirror.uploads(), 2);
    assert_eq!(actual, second);
}

#[test]
#[should_panic(expected = "keeps its size")]
fn a_resized_update_is_refused() {
    let client = make_client();
    let three = [0.0f32; 3];
    let four = [0.0f32; 4];
    let three_bytes = f32::as_bytes(&three);
    let four_bytes = f32::as_bytes(&four);
    let mut mirror = MirroredBuffer::new(&client, three_bytes);

    mirror.update(&client, four_bytes);
}

fn hq_denoiser(prefilter: PrefilterMode, sigma_override: Option<f32>) -> NlmDenoiser<R> {
    let hq = HqParams {
        sigma_override,
        ..HqParams::default()
    };
    let params = NlmParams {
        temporal_radius: 1,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter,
        motion_compensation: MotionCompensationMode::None,
        hq: Some(hq),
    };
    let client = make_client();

    NlmDenoiser::new(&client, params, SIZE, SIZE)
}

fn pilot_prefilter() -> PrefilterMode {
    PrefilterMode::NlmSpatial { strength_scale: 1.0 }
}

fn push_frame_at(denoiser: &mut NlmDenoiser<R>, index: usize) {
    let base = 0.3 + index as f32 * 0.01;
    let frame = make_noisy_gaussian_frame(SIZE, SIZE, 1, base, &[0.03]);
    denoiser.push_frame(&frame);
}

fn run_frames(denoiser: &mut NlmDenoiser<R>, count: usize) {
    for index in 0..count {
        push_frame_at(denoiser, index);

        denoiser.denoise().expect("denoise");
    }
}

fn read_table(denoiser: &NlmDenoiser<R>, handle: Handle) -> Vec<f32> {
    let bytes = denoiser.client.read_one(handle).expect("the table reads back");
    f32::from_bytes(&bytes).to_vec()
}

fn spatial_table(denoiser: &NlmDenoiser<R>) -> Vec<f32> {
    let handle = denoiser.spatial_offsets_handle();
    read_table(denoiser, handle)
}

fn pilot_table(denoiser: &NlmDenoiser<R>) -> Vec<f32> {
    let handle = denoiser.pilot_offsets_handle();
    read_table(denoiser, handle)
}

fn derived_values(denoiser: &NlmDenoiser<R>) -> DerivedValues {
    let snapshot = denoiser.noise_snapshot().expect("the noise state reads back");
    snapshot.derived
}

/// Checks every entry of `actual` within [MAX_RELATIVE] of `expected`, with the centre exactly 0.
fn assert_table_close(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: the table length");

    let centre = actual.len() / 2;
    let centre_bits = actual[centre].to_bits();
    assert_eq!(centre_bits, 0, "{what}: the centre entry");

    let pairs = actual.iter().zip(expected).enumerate();
    for (index, (&gpu, &host)) in pairs {
        let gap = relative_gap(gpu, host);
        assert!(
            gap <= MAX_RELATIVE,
            "{what}: entry {index} is {gap:e} apart, {gpu:e} against {host:e}"
        );
    }
}

fn table_bits(table: &[f32]) -> Vec<u32> {
    table.iter().map(|entry| entry.to_bits()).collect()
}

#[test]
fn a_pinned_sigma_keeps_the_host_spatial_offset_table() {
    let mut denoiser = hq_denoiser(PrefilterMode::None, Some(0.02));
    let before = spatial_table(&denoiser);
    run_frames(&mut denoiser, 6);

    let after = spatial_table(&denoiser);
    let derived = derived_values(&denoiser);
    let search_radius = denoiser.params.search_radius;
    let expected = build_spatial_offset_lut(search_radius, 0.0, derived.noise_offset);
    let before_bits = table_bits(&before);
    let after_bits = table_bits(&after);
    let expected_bits = table_bits(&expected);

    assert!(derived.noise_offset > 0.0, "the pinned sigma sets a noise floor");
    assert_eq!(after_bits, before_bits, "the frames leave the table alone");
    assert_eq!(after_bits, expected_bits);
}

#[test]
fn a_pinned_sigma_keeps_the_host_pilot_offset_table() {
    let prefilter = pilot_prefilter();
    let mut denoiser = hq_denoiser(prefilter, Some(0.02));
    let before = pilot_table(&denoiser);
    run_frames(&mut denoiser, 6);

    let after = pilot_table(&denoiser);
    let derived = derived_values(&denoiser);
    let search_radius = denoiser.params.search_radius;
    let expected = build_spatial_offset_lut(search_radius, 0.0, derived.input_noise_offset);
    let before_bits = table_bits(&before);
    let after_bits = table_bits(&after);
    let expected_bits = table_bits(&expected);

    assert!(
        derived.input_noise_offset > 0.0,
        "the pinned sigma sets a noise floor"
    );
    assert_eq!(after_bits, before_bits, "the frames leave the table alone");
    assert_eq!(after_bits, expected_bits);
}

#[test]
fn an_estimated_sigma_keeps_the_spatial_offset_table_current() {
    let mut denoiser = hq_denoiser(PrefilterMode::None, None);
    let before = spatial_table(&denoiser);
    run_frames(&mut denoiser, 6);

    let actual = spatial_table(&denoiser);
    let derived = derived_values(&denoiser);
    let search_radius = denoiser.params.search_radius;
    let expected = build_spatial_offset_lut(search_radius, derived.rho, derived.noise_offset);

    assert_ne!(actual, before, "the folds rewrite the table");
    assert_table_close(&actual, &expected, "spatial table");
}

#[test]
fn a_measured_separable_front_binds_a_spatial_table_that_follows_rho() {
    let params = NlmParams {
        temporal_radius: 1,
        search_radius: 2,
        patch_radius: 9,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        hq: Some(HqParams::default()),
    };
    let client = make_client();
    let mut denoiser = NlmDenoiser::<R>::new(&client, params, SIZE, SIZE);
    let before = spatial_table(&denoiser);

    // A `[0.25, 0.5, 0.25]` blur scales the variance by 0.375 and correlates neighbours.
    let sigma_pre = (8.0 / 255.0) / 0.375f32.sqrt();
    for index in 0..6 {
        let seed = 300 + index;
        let frame = correlated_noisy_frame(SIZE, SIZE, 0.4, sigma_pre, seed);
        denoiser.push_frame(&frame);

        denoiser.denoise().expect("denoise");
    }

    let actual = spatial_table(&denoiser);
    let derived = derived_values(&denoiser);
    let search_radius = denoiser.params.search_radius;
    let expected = build_spatial_offset_lut(search_radius, derived.rho, derived.noise_offset);

    assert!(
        denoiser.use_separable,
        "a patch radius of 9 takes the separable path"
    );
    assert!(
        derived.rho > 0.1,
        "the correlated grain sets rho, got {}",
        derived.rho
    );
    assert!(derived.noise_offset > 0.0, "the noise floor is measured");
    assert_ne!(actual, before, "the folds rewrite the table");
    assert_table_close(&actual, &expected, "separable spatial table");
}

#[test]
fn an_estimated_sigma_keeps_the_pilot_offset_table_current() {
    let prefilter = pilot_prefilter();
    let mut denoiser = hq_denoiser(prefilter, None);
    let before = pilot_table(&denoiser);
    run_frames(&mut denoiser, 6);

    let actual = pilot_table(&denoiser);
    let derived = derived_values(&denoiser);
    let search_radius = denoiser.params.search_radius;
    let expected = build_spatial_offset_lut(search_radius, 0.0, derived.input_noise_offset);

    assert_ne!(actual, before, "the folds rewrite the table");
    assert_table_close(&actual, &expected, "pilot table");
}

#[test]
fn fast_temporal_nlm_never_reuploads_the_neighbour_table() {
    let params = NlmParams {
        temporal_radius: 1,
        search_radius: 2,
        patch_radius: 2,
        channels: ChannelMode::Luma,
        motion_compensation: MotionCompensationMode::None,
        hq: None,
        ..NlmParams::default()
    };
    let client = make_client();
    let mut denoiser = NlmDenoiser::<R>::new(&client, params, SIZE, SIZE);
    run_frames(&mut denoiser, 6);

    let table = denoiser
        .neighbour_slots_for_test()
        .expect("a temporal window has a neighbour-slot table");
    assert_eq!(table.uploads(), 1);
}

#[test]
fn the_confidence_pass_keeps_the_neighbour_table_on_the_window() {
    let mut denoiser = hq_denoiser(PrefilterMode::None, Some(0.02));
    run_frames(&mut denoiser, 6);

    let center_t = denoiser.params.temporal_radius;
    let expected = denoiser.window_neighbour_slots(center_t);
    let table = denoiser
        .neighbour_slots_for_test()
        .expect("a temporal window has a neighbour-slot table");
    let handle = table.handle().clone();
    let bytes = denoiser.client.read_one(handle).expect("the table reads back");
    let slots = u32::from_bytes(&bytes).to_vec();

    assert!(table.uploads() > 1);
    assert_eq!(slots, expected);
}

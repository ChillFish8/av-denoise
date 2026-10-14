use cubecl::prelude::*;

use super::helpers::{R, make_client, make_noisy_gaussian_frame};
use crate::bench_api::HostIo;
use crate::nlmeans::mirror::MirroredBuffer;
use crate::nlmeans::noise::build_spatial_offset_lut;
use crate::nlmeans::*;

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

    assert_eq!(mirror.uploads(), 1);
    assert_eq!(read_floats(&client, &mirror), values);
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

    assert_eq!(mirror.uploads(), 2);
    assert_eq!(read_floats(&client, &mirror), second);
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

        let denoised = denoiser.denoise();
        denoised.expect("denoise");
    }
}

#[test]
fn a_pinned_sigma_uploads_the_spatial_offset_table_once() {
    let mut denoiser = hq_denoiser(PrefilterMode::None, Some(0.02));
    run_frames(&mut denoiser, 6);

    let (spatial, _) = denoiser.spatial_offset_lut_for_test();
    assert_eq!(spatial.uploads(), 1);
}

#[test]
fn a_pinned_sigma_uploads_the_pilot_offset_table_once() {
    let mut denoiser = hq_denoiser(pilot_prefilter(), Some(0.02));
    run_frames(&mut denoiser, 6);

    assert_eq!(denoiser.pilot_offset_lut_for_test().uploads(), 1);
}

#[test]
fn an_estimated_sigma_keeps_the_spatial_offset_table_current() {
    let mut denoiser = hq_denoiser(PrefilterMode::None, None);
    run_frames(&mut denoiser, 6);

    let (spatial, expected) = denoiser.spatial_offset_lut_for_test();
    assert!(spatial.uploads() > 1);
    assert_eq!(read_floats(&denoiser.client, spatial), expected);
}

#[test]
fn an_estimated_sigma_keeps_the_pilot_offset_table_current() {
    let mut denoiser = hq_denoiser(pilot_prefilter(), None);
    run_frames(&mut denoiser, 6);

    // The pilot runs on push, so one more push uploads the table for the latest estimate.
    push_frame_at(&mut denoiser, 6);

    let search_radius = denoiser.params.search_radius;
    let expected = build_spatial_offset_lut(search_radius, 0.0, denoiser.input_noise_offset);
    let pilot = denoiser.pilot_offset_lut_for_test();
    assert!(pilot.uploads() > 1);
    assert_eq!(read_floats(&denoiser.client, pilot), expected);
}

use cubecl::prelude::*;

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
use crate::nl4d::denoiser::noise_curve_upload;
use crate::nl4d::{Nl4dDenoiser, Nl4dParams};
use crate::nlmeans::mirror::MirroredBuffer;
use crate::nlmeans::{ChannelMode, HqParams};

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

fn read_words<T: CubeElement>(mirror: &MirroredBuffer) -> Vec<T> {
    let client = make_client();
    let handle = mirror.handle().clone();
    let bytes = client.read_one(handle).expect("the mirror reads back");
    T::from_bytes(&bytes).to_vec()
}

#[test]
fn a_pinned_sigma_uploads_its_buffers_only_when_they_change() {
    let params = static_clip_params(1);
    let denoiser = run_stream(params, 10);

    // Built as zeros, then the pinned sigma once.
    assert_eq!(denoiser.sigma_buffer_for_test().uploads(), 2);
    // A pinned sigma has no curve and no classes, so both keep what they were built with.
    assert_eq!(denoiser.noise_curve_buffer_for_test().uploads(), 1);
    assert_eq!(denoiser.strength_map_buffer_for_test().uploads(), 1);
}

#[test]
fn an_estimated_sigma_buffer_holds_what_the_last_pass_read() {
    let mut params = static_clip_params(1);
    params.nlm.hq = Some(HqParams::default());
    let denoiser = run_stream(params, 10);
    let front = denoiser.front_for_test();

    let sigmas = front.current_sigmas_temporal_only();
    let sigma_mirror = denoiser.sigma_buffer_for_test();
    let sigma_buffer: Vec<f32> = read_words(sigma_mirror);
    assert_eq!(sigma_buffer[0].to_bits(), sigmas[0].to_bits());
}

#[test]
fn the_noise_curve_buffer_holds_what_the_last_pass_read() {
    let frames = ramp_clip(1);
    let params = ramp_params(ChannelMode::Luma, true, 1.0);
    let denoiser = run_frames(params, RAMP_WIDTH, RAMP_HEIGHT, &frames);
    let front = denoiser.front_for_test();

    let curve = front.current_noise_curve();
    assert!(curve.is_some(), "the ramp clip forms a curve");

    let curve_buffer = denoiser.noise_curve_buffer_for_test();
    assert!(curve_buffer.uploads() > 1);

    let ratios = curve.map(|curve| curve.ratios);
    let (expected_curve, _) = noise_curve_upload(ratios, true);
    let actual_curve: Vec<f32> = read_words(curve_buffer);
    assert_eq!(actual_curve, expected_curve);
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

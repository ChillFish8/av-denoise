use cubecl::prelude::*;

use super::helpers::*;
use crate::bench_api::HostIo;
use crate::nl4d::{Nl4dDenoiser, Nl4dParams};
use crate::nlmeans::denoiser::LOOKAHEAD_FRAMES;
use crate::nlmeans::*;

const PAN_WIDTH: u32 = 256;
const PAN_HEIGHT: u32 = 192;
const PAN_SHIFT: u32 = 3;
const PAN_FRAMES: usize = 12;
const PAN_SIGMA: f32 = 2.0 / 255.0;
const PAN_SEED: u32 = 7;

/// Each pan frame's sigma read along its pass's own motion search, the reference the look-ahead
/// readings match. `Direct` at radius 2 and `Chained` at radius 3 read the same values.
const PASS_TIME_SIGMAS: [f32; PAN_FRAMES] = [
    0.007815467,
    0.007805687,
    0.007863555,
    0.007839382,
    0.0077624097,
    0.007782205,
    0.007812833,
    0.0077841855,
    0.007855936,
    0.007838592,
    0.007886594,
    0.007867605,
];

fn nl4d_pan_params(temporal_radius: u32) -> Nl4dParams {
    let defaults = Nl4dParams::default();
    let motion_compensation = MotionCompensationMode::Mvtools {
        blksize: 16,
        overlap: 8,
        search_radius: 4,
        pyramid_levels: 2,
        estimation: MotionEstimation::Auto,
    };
    let nlm = NlmParams {
        channels: ChannelMode::Luma,
        motion_compensation,
        ..defaults.nlm
    };

    Nl4dParams {
        nlm,
        temporal_radius,
        ..defaults
    }
}

fn hq_params(temporal_radius: u32, motion_compensation: MotionCompensationMode) -> NlmParams {
    NlmParams {
        temporal_radius,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation,
        hq: Some(HqParams::default()),
    }
}

/// Pushes and submits every frame of `frames`, as an engine does.
fn run_nl4d(params: Nl4dParams, frames: &[Vec<f32>]) -> Nl4dDenoiser {
    let client = make_client();
    let mut denoiser =
        Nl4dDenoiser::new(&client, params, PAN_WIDTH, PAN_HEIGHT).expect("construction failed");

    for frame in frames {
        denoiser.push_frame(frame);
        denoiser.denoise().expect("denoise failed");
    }

    denoiser
}

/// The temporal sample sigma the history holds for stream frame `frame`, `None` without a sample.
fn history_sigma(front: &NlmDenoiser, frame: u32) -> Option<f32> {
    let noise = front.measured_noise().expect("the noise is measured");
    let (reading, _, _) = noise.read_history_frame(frame);

    reading.sample.map(|sample| sample.sigma[0])
}

fn assert_within(value: f32, reference: f32, tolerance: f32, label: &str) {
    let rel_err = (value - reference).abs() / reference;
    assert!(
        rel_err <= tolerance,
        "{label}: {value} vs {reference} (rel err {rel_err:.4})"
    );
}

#[test]
fn lookahead_readings_match_pass_time_readings() {
    let frames = panning_frames(PAN_WIDTH, PAN_HEIGHT, PAN_SHIFT, PAN_FRAMES, PAN_SIGMA, PAN_SEED);

    for (radius, label) in [(2, "direct"), (3, "chained")] {
        let denoiser = run_nl4d(nl4d_pan_params(radius), &frames);
        let front = denoiser.front_for_test();
        assert_eq!(front.is_chained(), radius == 3, "{label}: estimation");

        for (frame, &reference) in PASS_TIME_SIGMAS.iter().enumerate() {
            let sigma = history_sigma(front, frame as u32).expect("a pan frame has a sample");
            let frame_label = format!("{label} frame {frame}");
            assert_within(sigma, reference, 0.02, &frame_label);
        }
    }
}

#[test]
fn scene_first_frame_reads_against_the_next_frame() {
    let frames = panning_frames(PAN_WIDTH, PAN_HEIGHT, PAN_SHIFT, 2, PAN_SIGMA, PAN_SEED);
    let client = make_client();
    let params = nl4d_pan_params(2);
    let mut denoiser =
        Nl4dDenoiser::new(&client, params, PAN_WIDTH, PAN_HEIGHT).expect("construction failed");

    denoiser.push_frame(&frames[0]);
    let alone = history_sigma(denoiser.front_for_test(), 0);
    assert_eq!(alone, None, "the first frame alone has nothing to read against");

    denoiser.push_frame(&frames[1]);
    let first = history_sigma(denoiser.front_for_test(), 0).expect("the first frame has a sample");
    assert_within(first, PASS_TIME_SIGMAS[0], 0.02, "first frame");

    let raw = raw_reading_sigma(&frames[1], &frames[0], PAN_WIDTH, PAN_HEIGHT, 1)
        .expect("the raw reading has a sample");
    assert!(
        first <= 0.5 * raw[0],
        "the first frame read {first}, not well below the uncompensated {}",
        raw[0]
    );
}

#[test]
fn held_frames_do_not_enter_the_window() {
    const HELD: [usize; 2] = [3, 7];
    let frames = panning_frames(PAN_WIDTH, PAN_HEIGHT, PAN_SHIFT, 16, PAN_SIGMA, PAN_SEED);
    let mut with_held = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        with_held.push(frame.clone());
        if HELD.contains(&index) {
            with_held.push(frame.clone());
        }
    }

    let plain = run_nl4d(nl4d_pan_params(2), &frames);
    let held = run_nl4d(nl4d_pan_params(2), &with_held);

    // Each held copy follows its frame, so it lands one stream frame later per earlier copy.
    let front = held.front_for_test();
    for (copies_before, &index) in HELD.iter().enumerate() {
        let copy = (index + copies_before + 1) as u32;
        let sigma = history_sigma(front, copy);
        assert_eq!(sigma, None, "the held copy at stream frame {copy} has a sample");
    }

    let plain_snapshot = plain.noise_snapshot().expect("the noise state reads back");
    let held_snapshot = held.noise_snapshot().expect("the noise state reads back");
    let plain_sigma = plain_snapshot.derived.sigma[0];
    let held_sigma = held_snapshot.derived.sigma[0];
    assert_within(held_sigma, plain_sigma, 0.01, "derived sigma with held frames");
}

#[test]
fn motion_off_reads_uncompensated_at_push() {
    let frames = panning_frames(PAN_WIDTH, PAN_HEIGHT, 0, 4, PAN_SIGMA, PAN_SEED);
    let client = make_client();
    let params = hq_params(2, MotionCompensationMode::None);
    let mut denoiser = NlmDenoiser::new(&client, params, PAN_WIDTH, PAN_HEIGHT);

    for (index, frame) in frames.iter().enumerate() {
        denoiser.push_frame(frame);

        let frame_index = index as u32;
        let sigma = history_sigma(&denoiser, frame_index);
        let Some(previous) = index.checked_sub(1) else {
            assert_eq!(sigma, None, "the first frame has nothing to read against");
            continue;
        };

        let sigma = sigma.expect("a static frame has a sample at push");
        let reference = raw_reading_sigma(&frames[previous], frame, PAN_WIDTH, PAN_HEIGHT, 1)
            .expect("the raw reading has a sample");
        let label = format!("frame {index}");
        assert_within(sigma, reference[0], 1e-5, &label);
    }

    let spatial_params = hq_params(0, MotionCompensationMode::None);
    let mut spatial = NlmDenoiser::new(&client, spatial_params, PAN_WIDTH, PAN_HEIGHT);
    spatial.push_frame(&frames[0]);
    spatial.push_frame(&frames[1]);

    let noise = spatial.measured_noise().expect("the noise is measured");
    let (reading, _, _) = noise.read_history_frame(1);
    assert_eq!(reading.sample, None, "a spatial front reads no temporal sample");
    assert!(
        reading.immerkaer_low[0] > 0.0,
        "a spatial front still reads its Immerkær sigma at push"
    );
}

#[test]
fn released_frames_are_bit_identical_to_pushed_frames() {
    let pushes = LOOKAHEAD_FRAMES as usize + 4;
    let frames = panning_frames(PAN_WIDTH, PAN_HEIGHT, PAN_SHIFT, pushes, PAN_SIGMA, PAN_SEED);
    let client = make_client();
    let params = hq_params(2, MotionCompensationMode::mvtools_default());
    let total_frames = params.total_frames();
    let mut denoiser = NlmDenoiser::new(&client, params, PAN_WIDTH, PAN_HEIGHT);
    let depth = denoiser.lookahead_depth();
    assert_eq!(depth, LOOKAHEAD_FRAMES as usize, "a measuring front looks ahead");

    let frame_len = (PAN_WIDTH * PAN_HEIGHT) as usize;
    for (index, frame) in frames.iter().enumerate() {
        denoiser.push_frame(frame);
        denoiser.denoise().expect("denoise failed");

        let Some(released) = index.checked_sub(depth) else {
            assert_eq!(denoiser.real_pushes(), 0, "push {index} released a frame early");
            continue;
        };

        assert_eq!(denoiser.real_pushes(), released + 1, "one release per push");

        let newest_slot = denoiser.ring_slot(total_frames - 1) as usize;
        let ring = denoiser.input_ring().clone();
        let ring_bytes = client.read_one(ring).expect("the ring reads back");
        let ring_values = f32::from_bytes(&ring_bytes);
        let start = newest_slot * frame_len;
        let stored = &ring_values[start..start + frame_len];

        let stored_bits: Vec<u32> = stored.iter().map(|value| value.to_bits()).collect();
        let pushed_bits: Vec<u32> = frames[released].iter().map(|value| value.to_bits()).collect();
        assert!(
            stored_bits == pushed_bits,
            "released frame {released} differs from the pushed frame"
        );
    }
}

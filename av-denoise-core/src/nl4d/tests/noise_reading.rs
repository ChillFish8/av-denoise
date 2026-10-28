use super::helpers::{RAMP_HEIGHT, RAMP_WIDTH, make_client, ramp_clip, ramp_params};
use crate::bench_api::HostIo;
use crate::nl4d::{Nl4dDenoiser, Nl4dParams};
use crate::nlmeans::denoiser::LOOKAHEAD_FRAMES;
use crate::nlmeans::tests::helpers::{interleave_planes, panning_frames, raw_reading_sigma};
use crate::nlmeans::{ChannelMode, MotionCompensationMode, MotionEstimation, NlmParams};
use crate::noise::{BLOCK_MOVED_LANE, temporal_stats_record_len};

const PAN_WIDTH: u32 = 256;
const PAN_HEIGHT: u32 = 192;
const PAN_SHIFT: u32 = 3;
/// Twelve frames run as centres while the stream is pushed, after the noise look-ahead fills.
const PAN_FRAMES: usize = 12 + LOOKAHEAD;
/// The pushes a frame waits in the noise look-ahead before it enters the ring.
const LOOKAHEAD: usize = LOOKAHEAD_FRAMES as usize;
const PAN_SIGMA: f32 = 2.0 / 255.0;
const PAN_SEED: u32 = 7;

fn reading_params(channels: ChannelMode, motion_compensation: MotionCompensationMode) -> Nl4dParams {
    let defaults = Nl4dParams::default();
    let nlm = NlmParams {
        channels,
        motion_compensation,
        ..defaults.nlm
    };

    Nl4dParams { nlm, ..defaults }
}

fn mvtools(blksize: u32, overlap: u32) -> MotionCompensationMode {
    MotionCompensationMode::Mvtools {
        blksize,
        overlap,
        search_radius: 4,
        pyramid_levels: 2,
        estimation: MotionEstimation::Auto,
    }
}

/// Pushes and submits every frame, calling `after_pass` with the centre frame's index in the
/// stream once each submit with a full window ran its pass.
fn run_stream(
    params: Nl4dParams,
    width: u32,
    height: u32,
    frames: &[Vec<f32>],
    mut after_pass: impl FnMut(usize, &Nl4dDenoiser),
) -> Nl4dDenoiser {
    let client = make_client();
    let radius = params.temporal_radius as usize;
    let mut denoiser = Nl4dDenoiser::new(&client, params, width, height).expect("construction failed");

    for (index, frame) in frames.iter().enumerate() {
        denoiser.push_frame(frame);
        denoiser.denoise().expect("denoise failed");

        if index < 2 * radius + LOOKAHEAD {
            continue;
        }

        let centre = index - radius - LOOKAHEAD;
        after_pass(centre, &denoiser);
    }

    denoiser
}

/// Pushes and submits every frame, calling `after_push` with each frame's index in the stream
/// once the look-ahead has read it.
fn run_pushes(
    params: Nl4dParams,
    width: u32,
    height: u32,
    frames: &[Vec<f32>],
    mut after_push: impl FnMut(usize, &Nl4dDenoiser),
) {
    let client = make_client();
    let mut denoiser = Nl4dDenoiser::new(&client, params, width, height).expect("construction failed");

    for (index, frame) in frames.iter().enumerate() {
        denoiser.push_frame(frame);
        denoiser.denoise().expect("denoise failed");

        after_push(index, &denoiser);
    }
}

fn folded_temporal_sigma(denoiser: &Nl4dDenoiser) -> f32 {
    let sigmas = denoiser.front_for_test().current_sigmas_temporal_only();
    sigmas[0]
}

/// The temporal sample sigma the history holds for stream frame `frame`.
fn frame_sigma(denoiser: &Nl4dDenoiser, frame: usize) -> Option<[f32; 3]> {
    let noise = denoiser
        .front_for_test()
        .measured_noise()
        .expect("the noise is measured");
    let (reading, _, _) = noise.read_history_frame(frame as u32);

    reading.sample.map(|sample| sample.sigma)
}

/// The share of noise blocks of the frame the look-ahead read at stream frame `frame` that were
/// read along a non-zero vector.
///
/// The look-ahead keeps a frame's stats only until its slot is reused, so this reads the frame
/// just pushed.
fn moved_share(denoiser: &Nl4dDenoiser, frame: usize, stored_ch: u32) -> f32 {
    let noise = denoiser
        .front_for_test()
        .measured_noise()
        .expect("the noise is measured");
    let slot = frame as u32 % LOOKAHEAD_FRAMES;
    let (_, _, records) = noise.read_slot_stats(slot);
    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let moved_lane = (2 * stored_ch + BLOCK_MOVED_LANE) as usize;

    let blocks = records.len() / record_len;
    let moved = records
        .chunks_exact(record_len)
        .filter(|record| record[moved_lane] != 0.0)
        .count();

    moved as f32 / blocks as f32
}

/// The share of motion blocks the last pass's search gave a non-zero vector against the frame
/// before its centre, `None` before the first pass.
///
/// The last pass's centre trails the frame just pushed by the ring and look-ahead latency.
fn searched_moved_share(denoiser: &Nl4dDenoiser) -> Option<f32> {
    let snapshot = denoiser.motion_snapshot()?;
    let previous = snapshot
        .offsets
        .iter()
        .position(|&offset| offset == -1)
        .expect("the window holds the previous frame");
    let vectors = &snapshot.vectors[previous];
    let moved = vectors.iter().filter(|&&vector| vector != [0, 0]).count();

    let share = moved as f32 / vectors.len() as f32;
    Some(share)
}

fn assert_tracks_pan_sigma(params: Nl4dParams, label: &str) {
    let frames = panning_frames(PAN_WIDTH, PAN_HEIGHT, PAN_SHIFT, PAN_FRAMES, PAN_SIGMA, PAN_SEED);
    let denoiser = run_stream(params, PAN_WIDTH, PAN_HEIGHT, &frames, |_, _| {});

    let sigma = folded_temporal_sigma(&denoiser);
    let rel_err = (sigma - PAN_SIGMA).abs() / PAN_SIGMA;
    assert!(
        rel_err <= 0.10,
        "{label}: folded sigma {sigma} vs true {PAN_SIGMA} (rel err {rel_err:.3})"
    );
}

#[test]
fn pan_reading_tracks_true_noise_nl4d() {
    let params = reading_params(ChannelMode::Luma, mvtools(16, 8));
    assert_tracks_pan_sigma(params, "direct");
}

#[test]
fn pan_reading_tracks_true_noise_nl4d_chained() {
    let mut params = reading_params(ChannelMode::Luma, mvtools(16, 8));
    params.temporal_radius = 3;
    assert_tracks_pan_sigma(params, "chained");
}

#[test]
fn pan_reading_holds_for_non_default_motion_steps() {
    for (blksize, overlap) in [(8, 4), (32, 0)] {
        let params = reading_params(ChannelMode::Luma, mvtools(blksize, overlap));
        let label = format!("blksize {blksize} overlap {overlap}");
        assert_tracks_pan_sigma(params, &label);
    }
}

#[test]
fn static_stream_reading_is_unchanged() {
    let frames = panning_frames(PAN_WIDTH, PAN_HEIGHT, 0, PAN_FRAMES, PAN_SIGMA, PAN_SEED);
    let params = reading_params(ChannelMode::Luma, mvtools(16, 8));

    let mut searched_shares = Vec::new();
    let mut followed_shares = Vec::new();
    run_pushes(params, PAN_WIDTH, PAN_HEIGHT, &frames, |frame, denoiser| {
        if frame == 0 {
            return;
        }

        let reading = frame_sigma(denoiser, frame).expect("a static frame has a sample");
        let reference = raw_reading_sigma(&frames[frame - 1], &frames[frame], PAN_WIDTH, PAN_HEIGHT, 1)
            .expect("the raw reference has a sample");

        let ratio = reading[0] / reference[0];
        assert!(
            (0.97..=1.03).contains(&ratio),
            "frame {frame}: reading {} vs raw reference {} (ratio {ratio:.4})",
            reading[0],
            reference[0]
        );

        let searched = searched_moved_share(denoiser);
        let followed = moved_share(denoiser, frame, 1);
        searched_shares.extend(searched);
        followed_shares.push(followed);
    });

    eprintln!("static stream searched non-zero vector share per pass centre: {searched_shares:?}");
    eprintln!("static stream followed share per pushed frame: {followed_shares:?}");
}

#[test]
fn cut_inside_window_keeps_previous_estimate() {
    let cut_at = PAN_FRAMES / 2;
    let before = panning_frames(PAN_WIDTH, PAN_HEIGHT, PAN_SHIFT, cut_at, PAN_SIGMA, PAN_SEED);
    let after = panning_frames(PAN_WIDTH, PAN_HEIGHT, PAN_SHIFT, cut_at, PAN_SIGMA, 99);
    let frames: Vec<Vec<f32>> = before.into_iter().chain(after).collect();
    let params = reading_params(ChannelMode::Luma, mvtools(16, 8));

    run_stream(params, PAN_WIDTH, PAN_HEIGHT, &frames, |centre, denoiser| {
        let sigma = folded_temporal_sigma(denoiser);
        assert!(
            sigma >= 0.9 * PAN_SIGMA,
            "centre {centre}: folded sigma {sigma} fell below 0.9x the true {PAN_SIGMA}"
        );
    });
}

#[test]
fn chroma_denoiser_pan_reading_does_not_rise() {
    let width = PAN_WIDTH / 2;
    let height = PAN_HEIGHT / 2;
    let u_planes = panning_frames(width, height, PAN_SHIFT, PAN_FRAMES, PAN_SIGMA, 21);
    let v_planes = panning_frames(width, height, PAN_SHIFT, PAN_FRAMES, PAN_SIGMA, 22);
    let frames: Vec<Vec<f32>> = u_planes
        .iter()
        .zip(&v_planes)
        .map(|(u_plane, v_plane)| interleave_planes(&[u_plane, v_plane]))
        .collect();
    let params = reading_params(ChannelMode::Chroma, mvtools(16, 8));

    let mut compared = 0;
    run_pushes(params, width, height, &frames, |frame, denoiser| {
        if frame == 0 {
            return;
        }

        let reading = frame_sigma(denoiser, frame);
        let reference = raw_reading_sigma(&frames[frame - 1], &frames[frame], width, height, 2);
        let (Some(reading), Some(reference)) = (reading, reference) else {
            return;
        };

        let followed = moved_share(denoiser, frame, 2);
        assert!(
            followed >= 0.5,
            "frame {frame}: only {followed} of the blocks followed the pan"
        );

        for channel in 0..2 {
            assert!(
                reading[channel] <= reference[channel],
                "frame {frame} channel {channel}: reading {} rose above the raw reference {}",
                reading[channel],
                reference[channel]
            );

            // The raw reading carries the panning texture at about 12x the noise, so an inactive
            // compensation reads near 1.0x raw here.
            assert!(
                reading[channel] <= 0.25 * reference[channel],
                "frame {frame} channel {channel}: reading {} is not clearly below the raw \
                 reference {}",
                reading[channel],
                reference[channel]
            );
        }

        compared += 1;
    });

    assert!(compared > 0, "no pass had both a reading and a raw reference");
}

#[test]
fn static_flat_noise_map_clip_reads_as_raw() {
    let frames = ramp_clip(1);
    let params = ramp_params(ChannelMode::Luma, true, 1.0);

    let mut searched_shares = Vec::new();
    let mut followed_shares = Vec::new();
    run_pushes(params, RAMP_WIDTH, RAMP_HEIGHT, &frames, |frame, denoiser| {
        if frame == 0 {
            return;
        }

        let reading = frame_sigma(denoiser, frame).expect("a static frame has a sample");
        let reference = raw_reading_sigma(&frames[frame - 1], &frames[frame], RAMP_WIDTH, RAMP_HEIGHT, 1)
            .expect("the raw reference has a sample");

        let ratio = reading[0] / reference[0];
        assert!(
            (0.97..=1.03).contains(&ratio),
            "frame {frame}: reading {} vs raw reference {} (ratio {ratio:.4})",
            reading[0],
            reference[0]
        );

        let searched = searched_moved_share(denoiser);
        let followed = moved_share(denoiser, frame, 1);
        searched_shares.extend(searched);
        followed_shares.push(followed);
    });

    eprintln!("static flat clip searched non-zero vector share per pass centre: {searched_shares:?}");
    eprintln!("static flat clip followed share per pushed frame: {followed_shares:?}");
}

/// The synthetic quiet stream reads about 5% above its true sigma once settled, so the first
/// passes are held to the settled reading of the same stream run fresh.
#[test]
fn scene_start_reads_from_the_scene_itself() {
    const LOUD_SIGMA: f32 = 6.0 / 255.0;
    let loud = panning_frames(PAN_WIDTH, PAN_HEIGHT, 0, PAN_FRAMES, LOUD_SIGMA, PAN_SEED);
    let quiet = panning_frames(PAN_WIDTH, PAN_HEIGHT, 0, PAN_FRAMES, PAN_SIGMA, 31);
    let params = reading_params(ChannelMode::Luma, mvtools(16, 8));
    let radius = params.temporal_radius as usize;

    let fresh = run_stream(params.clone(), PAN_WIDTH, PAN_HEIGHT, &quiet, |_, _| {});
    let settled = folded_temporal_sigma(&fresh);

    let client = make_client();
    let mut denoiser =
        Nl4dDenoiser::new(&client, params, PAN_WIDTH, PAN_HEIGHT).expect("construction failed");

    for frame in &loud {
        denoiser.push_frame(frame);
        denoiser.denoise().expect("denoise failed");
    }

    let loud_sigma = folded_temporal_sigma(&denoiser);
    assert!(
        loud_sigma > 2.0 * PAN_SIGMA,
        "the first stream reads {loud_sigma}"
    );

    denoiser.reset_stream();

    for frame in &quiet[..=2 * radius + LOOKAHEAD] {
        denoiser.push_frame(frame);
        denoiser.denoise().expect("denoise failed");
    }

    let sigma = folded_temporal_sigma(&denoiser);
    let rel_err = (sigma - settled).abs() / settled;
    assert!(
        rel_err <= 0.05,
        "the first passes after the reset read {sigma} vs the settled {settled} (rel err {rel_err:.3})"
    );
}

/// The derived sigma of each of a panning continuation stream's first passes, beside the raw
/// reading of frame 1 against frame 0.
fn continuation_pass_sigmas(temporal_radius: u32) -> (Vec<f32>, f32) {
    let frames = panning_frames(PAN_WIDTH, PAN_HEIGHT, PAN_SHIFT, PAN_FRAMES, PAN_SIGMA, PAN_SEED);
    let mut params = reading_params(ChannelMode::Luma, mvtools(16, 8));
    params.temporal_radius = temporal_radius;
    let radius = temporal_radius as usize;
    let client = make_client();
    let mut denoiser =
        Nl4dDenoiser::new(&client, params, PAN_WIDTH, PAN_HEIGHT).expect("construction failed");
    denoiser.mark_continuation();

    let mut sigmas = Vec::new();
    for (index, frame) in frames.iter().enumerate() {
        denoiser.push_frame(frame);
        denoiser.denoise().expect("denoise failed");

        let first_passes = 2 * radius + LOOKAHEAD..=3 * radius + LOOKAHEAD;
        if !first_passes.contains(&index) {
            continue;
        }

        let snapshot = denoiser.noise_snapshot().expect("the noise state reads back");
        sigmas.push(snapshot.derived.sigma[0]);
    }

    let raw = raw_reading_sigma(&frames[0], &frames[1], PAN_WIDTH, PAN_HEIGHT, 1)
        .expect("the raw reference has a sample");

    (sigmas, raw[0])
}

#[test]
fn a_panning_continuation_folds_only_compensated_samples() {
    for temporal_radius in [2, 3] {
        let (sigmas, raw) = continuation_pass_sigmas(temporal_radius);

        assert!(!sigmas.is_empty(), "radius {temporal_radius}: no pass ran");
        for (pass, sigma) in sigmas.iter().enumerate() {
            let rel_err = (sigma - PAN_SIGMA).abs() / PAN_SIGMA;
            assert!(
                rel_err <= 0.10,
                "radius {temporal_radius} pass {pass}: derived sigma {sigma} vs true {PAN_SIGMA} \
                 (rel err {rel_err:.3})"
            );
            assert!(
                *sigma <= 0.5 * raw,
                "radius {temporal_radius} pass {pass}: derived sigma {sigma} is not well below the \
                 uncompensated reading {raw}"
            );
        }
    }
}

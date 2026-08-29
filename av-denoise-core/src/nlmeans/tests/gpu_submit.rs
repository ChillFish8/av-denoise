//! `denoise_submit_gpu` and `flush_step_gpu` skip the readback that
//! `denoise_submit` and `flush` start automatically, handing back the
//! raw GPU handle instead.
//!
//! These tests pin that the GPU-resident path produces exactly the same
//! frames, in the same order, as the readback path it is built from.

use cubecl::prelude::*;

use super::helpers::*;
use crate::nlmeans::*;

fn temporal_params(radius: u32) -> NlmParams {
    NlmParams {
        temporal_radius: radius,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        hq: None,
    }
}

/// Distinct frames so a mixed-up slot or a stale readback shows up as a
/// mismatch rather than being masked by every frame looking the same.
fn distinct_frames(w: u32, h: u32, count: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            let noise_val = 0.2 + (i as f32) * 0.05;
            make_frame_with_noisy_region(w, h, 1, 0.5, w / 2, h / 2, 3, noise_val)
        })
        .collect()
}

#[test]
fn submit_gpu_matches_submit() {
    let client = make_client();
    let w = 16;
    let h = 16;

    let mut via_readback = NlmDenoiser::<R>::new(&client, temporal_params(1), w, h);
    let mut via_gpu = NlmDenoiser::<R>::new(&client, temporal_params(1), w, h);

    for frame in distinct_frames(w, h, 5) {
        via_readback.push_frame(&frame);
        via_gpu.push_frame(&frame);

        let expected = via_readback.denoise_submit().expect("submit failed");
        let actual = via_gpu.denoise_submit_gpu().expect("submit_gpu failed");

        match (expected, actual) {
            (None, None) => {},
            (Some(pending), Some(output)) => {
                let expected_frame = pending.wait().expect("wait failed");

                let bytes = client.read_one(output.handle).expect("gpu readback failed");
                let actual_frame = f32::from_bytes(&bytes);

                assert_eq!(actual_frame.len(), expected_frame.len(), "frame length mismatch");
                for (i, (a, b)) in expected_frame.iter().zip(actual_frame.iter()).enumerate() {
                    assert!(
                        (a - b).abs() < 1e-6,
                        "pixel {i}: denoise_submit gave {a}, denoise_submit_gpu gave {b}"
                    );
                }
            },
            (a, b) => panic!(
                "denoise_submit and denoise_submit_gpu disagreed on readiness: {} vs {}",
                a.is_some(),
                b.is_some()
            ),
        }
    }
}

/// Reads a `GpuOutput` fully back to the host, the same way `flush`
/// reads it back internally, so a test can compare it against a frame
/// produced through the plain readback path.
fn read_output(client: &ComputeClient<R>, output: GpuOutput) -> Vec<f32> {
    let bytes = client.read_one(output.handle).expect("gpu readback failed");
    f32::from_bytes(&bytes).to_vec()
}

fn assert_frames_match(expected: &[Vec<f32>], actual: &[Vec<f32>]) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "flush and flush_step_gpu produced different frame counts"
    );
    for (frame_idx, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        assert_eq!(e.len(), a.len(), "frame {frame_idx}: length mismatch");
        for (i, (ev, av)) in e.iter().zip(a.iter()).enumerate() {
            assert!(
                (ev - av).abs() < 1e-6,
                "frame {frame_idx}, pixel {i}: flush gave {ev}, flush_step_gpu gave {av}"
            );
        }
    }
}

/// Runs `pushes` frames through two denoisers built from the same
/// parameters, drains one with `flush` and the other by hand-driving
/// `flush_step_gpu` against `flush_target`, and checks the two tails
/// agree frame for frame.
fn compare_flush_paths(radius: u32, pushes: usize) {
    let client = make_client();
    let w = 16;
    let h = 16;

    let mut via_flush = NlmDenoiser::<R>::new(&client, temporal_params(radius), w, h);
    let mut via_step = NlmDenoiser::<R>::new(&client, temporal_params(radius), w, h);

    for frame in distinct_frames(w, h, pushes) {
        via_flush.push_frame(&frame);
        let _ = via_flush.denoise().expect("denoise failed");

        via_step.push_frame(&frame);
        let _ = via_step.denoise_submit_gpu().expect("submit_gpu failed");
    }

    let mut expected = Vec::new();
    via_flush
        .flush(|frame| expected.push(frame.to_vec()))
        .expect("flush failed");

    let target = via_step.flush_target();
    let mut actual = Vec::new();
    while actual.len() < target {
        if let Some(output) = via_step.flush_step_gpu().expect("flush_step_gpu failed") {
            actual.push(read_output(&client, output));
        }
    }
    assert!(
        via_step.flush_step_gpu().is_ok(),
        "a further flush_step_gpu call past the target should still succeed"
    );

    assert_eq!(
        actual.len(),
        target,
        "flush_target did not match the frames actually collected"
    );
    assert_frames_match(&expected, &actual);
}

#[test]
fn flush_step_gpu_emits_the_same_count_and_frames_short_stream() {
    // Radius 2, one push: the window never fills during pushing, so the
    // whole drain happens inside the padding phase of the flush.
    compare_flush_paths(2, 1);
}

#[test]
fn flush_step_gpu_emits_the_same_count_and_frames_long_stream() {
    // Radius 2, five pushes: the window fills while pushing, so the
    // drain runs entirely through the trailing-tail phase of the flush.
    compare_flush_paths(2, 5);
}

/// Radius 2, two pushes. Neither of the two tests above exercises a
/// single drain that needs both behaviours. The short-stream case stops
/// as soon as the window finishes filling, and the long-stream case
/// never fills the window during the drain at all, since it was already
/// full before the drain began. Here the window is exactly one frame
/// short when the drain starts, and the target is two frames, so the
/// first duplicate both finishes filling the window and produces the
/// first output, then a second duplicate runs with the window already
/// full and produces the second. One drain call has to switch from one
/// behaviour to the other partway through, which is what this test
/// pins down.
#[test]
fn flush_step_gpu_emits_the_same_count_and_frames_mixed_phase_stream() {
    let radius = 2;
    let pushes = 2;

    let client = make_client();
    let w = 16;
    let h = 16;

    let mut via_flush = NlmDenoiser::<R>::new(&client, temporal_params(radius), w, h);
    let mut via_step = NlmDenoiser::<R>::new(&client, temporal_params(radius), w, h);

    let mut during_pushes_flush = 0usize;
    let mut during_pushes_step = 0usize;
    for frame in distinct_frames(w, h, pushes) {
        via_flush.push_frame(&frame);
        if via_flush.denoise().expect("denoise failed").is_some() {
            during_pushes_flush += 1;
        }

        via_step.push_frame(&frame);
        if via_step
            .denoise_submit_gpu()
            .expect("submit_gpu failed")
            .is_some()
        {
            during_pushes_step += 1;
        }
    }
    assert_eq!(
        during_pushes_flush, 0,
        "radius {radius} pushes {pushes}: the window should still be filling, so nothing \
         should come out during pushing"
    );
    assert_eq!(during_pushes_step, during_pushes_flush);

    // Confirm the drain actually straddles both phases rather than just
    // assuming it from the chosen parameters. The window has to still be
    // short when the drain starts, and filling it in has to leave at
    // least one more output for the trailing-tail phase to supply, or
    // every output would come from a single phase after all.
    let total_frames = via_step.params.total_frames() as usize;
    let target = via_step.flush_target();
    let frames_short = total_frames - via_step.frames_loaded;
    assert!(
        via_step.frames_loaded < total_frames,
        "radius {radius} pushes {pushes}: the window must still be filling when the drain \
         starts, got frames_loaded {} of {total_frames}",
        via_step.frames_loaded
    );
    assert!(
        frames_short < target,
        "radius {radius} pushes {pushes}: filling the window takes {frames_short} steps, \
         which must leave at least one more output for the target of {target}, or the drain \
         never reaches the trailing-tail phase"
    );

    let mut expected = Vec::new();
    via_flush
        .flush(|frame| expected.push(frame.to_vec()))
        .expect("flush failed");

    let mut actual = Vec::new();
    while actual.len() < target {
        if let Some(output) = via_step.flush_step_gpu().expect("flush_step_gpu failed") {
            actual.push(read_output(&client, output));
        }
    }

    assert_eq!(
        actual.len(),
        target,
        "flush_target did not match the frames actually collected"
    );
    assert_eq!(
        during_pushes_flush + expected.len(),
        pushes,
        "radius {radius} pushes {pushes}: total emissions must equal the number of real \
         frames pushed, got {during_pushes_flush} during pushing and {} from flush",
        expected.len()
    );
    assert_frames_match(&expected, &actual);
}

#[test]
fn current_sigmas_reads_zero_on_the_fast_path() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frame = make_frame_with_noisy_region(w, h, 1, 0.5, 8, 8, 3, 0.9);

    let mut denoiser = NlmDenoiser::<R>::new(&client, temporal_params(0), w, h);
    assert_eq!(denoiser.current_sigmas(), [0.0, 0.0, 0.0]);

    denoiser.push_frame(&frame);
    let _ = denoiser.denoise().expect("denoise failed");
    assert_eq!(
        denoiser.current_sigmas(),
        [0.0, 0.0, 0.0],
        "no HQ estimator runs on the fast path, so the sigma stays zero"
    );
}

#[test]
fn current_sigmas_broadcasts_a_pinned_sigma_override() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frame = make_frame_with_noisy_region(w, h, 1, 0.5, 8, 8, 3, 0.9);

    let params = NlmParams {
        hq: Some(HqParams {
            auto_strength: false,
            noise_floor: false,
            sigma_override: Some(6.0 / 255.0),
            temporal_confidence: false,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
            windowed_noise_estimation: false,
        }),
        ..temporal_params(0)
    };
    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);

    // A pinned sigma reads back immediately, before any frame is pushed.
    assert_eq!(denoiser.current_sigmas(), [6.0 / 255.0; 3]);

    denoiser.push_frame(&frame);
    let _ = denoiser.denoise().expect("denoise failed");
    assert_eq!(denoiser.current_sigmas(), [6.0 / 255.0; 3]);
}

#[test]
fn current_sigmas_matches_the_median_estimator_once_it_folds() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frame = make_frame_with_noisy_region(w, h, 1, 0.5, 8, 8, 3, 0.9);

    let params = NlmParams {
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
            temporal_confidence: false,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
            windowed_noise_estimation: false,
        }),
        ..temporal_params(0)
    };
    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);
    let _ = denoiser.denoise().expect("denoise failed");

    let expected = denoiser.noise_estimator.current().expect("seeded on first push")[0];
    assert_eq!(denoiser.current_sigmas()[0], expected);
}

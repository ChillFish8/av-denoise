use cubecl::prelude::*;

use super::helpers::*;
use crate::bench_api::{GpuOutput, HostIo};
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

/// Distinct frames, so a mixed-up slot or a stale readback shows up as a mismatch.
fn distinct_frames(width: u32, height: u32, count: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            let noise_value = 0.2 + (i as f32) * 0.05;
            make_frame_with_noisy_region(width, height, 1, 0.5, width / 2, height / 2, 3, noise_value)
        })
        .collect()
}

#[test]
fn submit_gpu_matches_submit() {
    let client = make_client();
    let width = 16;
    let height = 16;

    let readback_params = temporal_params(1);
    let gpu_params = temporal_params(1);
    let mut via_readback = NlmDenoiser::<R>::new(&client, readback_params, width, height);
    let mut via_gpu = NlmDenoiser::<R>::new(&client, gpu_params, width, height);

    let frames = distinct_frames(width, height, 5);
    for frame in frames {
        via_readback.push_frame(&frame);
        via_gpu.push_frame(&frame);

        let expected = via_readback.denoise().expect("denoise failed");
        let actual = via_gpu.denoise_submit_gpu().expect("submit_gpu failed");

        match (expected, actual) {
            (None, None) => {},
            (Some(expected_frame), Some(output)) => {
                let bytes = client.read_one(output.handle).expect("gpu readback failed");
                let actual_frame = f32::from_bytes(&bytes);

                assert_eq!(actual_frame.len(), expected_frame.len(), "frame length mismatch");

                let pairs = expected_frame.iter().zip(actual_frame.iter()).enumerate();
                for (i, (readback_value, gpu_value)) in pairs {
                    assert!(
                        (readback_value - gpu_value).abs() < 1e-6,
                        "pixel {i}: denoise gave {readback_value}, denoise_submit_gpu gave {gpu_value}"
                    );
                }
            },
            (readback, gpu) => panic!(
                "denoise and denoise_submit_gpu disagreed on readiness: {} vs {}",
                readback.is_some(),
                gpu.is_some()
            ),
        }
    }
}

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

    for (frame_idx, (expected_frame, actual_frame)) in expected.iter().zip(actual.iter()).enumerate() {
        assert_eq!(
            expected_frame.len(),
            actual_frame.len(),
            "frame {frame_idx}: length mismatch"
        );

        let pairs = expected_frame.iter().zip(actual_frame.iter()).enumerate();
        for (i, (expected_value, actual_value)) in pairs {
            assert!(
                (expected_value - actual_value).abs() < 1e-6,
                "frame {frame_idx}, pixel {i}: flush gave {expected_value}, flush_step_gpu gave {actual_value}"
            );
        }
    }
}

/// Drains one denoiser with `flush` and a twin by hand-driving `flush_step_gpu`, then checks the two
/// tails agree frame for frame.
fn compare_flush_paths(radius: u32, pushes: usize) {
    let client = make_client();
    let width = 16;
    let height = 16;

    let flush_params = temporal_params(radius);
    let step_params = temporal_params(radius);
    let mut via_flush = NlmDenoiser::<R>::new(&client, flush_params, width, height);
    let mut via_step = NlmDenoiser::<R>::new(&client, step_params, width, height);

    let frames = distinct_frames(width, height, pushes);
    for frame in frames {
        via_flush.push_frame(&frame);
        let _ = via_flush.denoise().expect("denoise failed");

        via_step.push_frame(&frame);
        let _ = via_step.denoise_submit_gpu().expect("submit_gpu failed");
    }

    let mut expected = Vec::new();
    let collect = |frame: &[f32]| {
        let samples = frame.to_vec();
        expected.push(samples);
    };
    via_flush.flush(collect).expect("flush failed");

    let target = via_step.flush_target();
    let mut actual = Vec::new();
    while actual.len() < target {
        if let Some(output) = via_step.flush_step_gpu().expect("flush_step_gpu failed") {
            let frame = read_output(&client, output);
            actual.push(frame);
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
    // One push at radius 2 never fills the window, so the whole drain runs in the padding phase.
    compare_flush_paths(2, 1);
}

#[test]
fn flush_step_gpu_emits_the_same_count_and_frames_long_stream() {
    // Five pushes at radius 2 fill the window, so the whole drain runs in the trailing-tail phase.
    compare_flush_paths(2, 5);
}

/// Pins a single drain that switches from the padding phase to the trailing-tail phase.
///
/// With radius 2 and two pushes the window is one frame short when the drain starts, and the target
/// is two frames. The first step fills the window and emits, then the second runs on a full window.
#[test]
fn flush_step_gpu_emits_the_same_count_and_frames_mixed_phase_stream() {
    let radius = 2;
    let pushes = 2;

    let client = make_client();
    let width = 16;
    let height = 16;

    let flush_params = temporal_params(radius);
    let step_params = temporal_params(radius);
    let mut via_flush = NlmDenoiser::<R>::new(&client, flush_params, width, height);
    let mut via_step = NlmDenoiser::<R>::new(&client, step_params, width, height);

    let frames = distinct_frames(width, height, pushes);
    let mut during_pushes_flush = 0usize;
    let mut during_pushes_step = 0usize;
    for frame in frames {
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

    // Checks the drain really straddles both phases instead of assuming it from the parameters.
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
    let collect = |frame: &[f32]| {
        let samples = frame.to_vec();
        expected.push(samples);
    };
    via_flush.flush(collect).expect("flush failed");

    let mut actual = Vec::new();
    while actual.len() < target {
        if let Some(output) = via_step.flush_step_gpu().expect("flush_step_gpu failed") {
            let frame = read_output(&client, output);
            actual.push(frame);
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
    let width = 16;
    let height = 16;
    let frame = make_frame_with_noisy_region(width, height, 1, 0.5, 8, 8, 3, 0.9);

    let params = temporal_params(0);
    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
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
    let width = 16;
    let height = 16;
    let frame = make_frame_with_noisy_region(width, height, 1, 0.5, 8, 8, 3, 0.9);

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
    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);

    // A pinned sigma reads back immediately, before any frame is pushed.
    assert_eq!(denoiser.current_sigmas(), [6.0 / 255.0; 3]);

    denoiser.push_frame(&frame);
    let _ = denoiser.denoise().expect("denoise failed");
    assert_eq!(denoiser.current_sigmas(), [6.0 / 255.0; 3]);
}

#[test]
fn current_sigmas_matches_the_median_estimator_once_it_folds() {
    let client = make_client();
    let width = 16;
    let height = 16;
    let frame = make_frame_with_noisy_region(width, height, 1, 0.5, 8, 8, 3, 0.9);

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
    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);
    let _ = denoiser.denoise().expect("denoise failed");

    let expected = denoiser.noise_estimator.current().expect("seeded on first push")[0];
    assert_eq!(denoiser.current_sigmas()[0], expected);
}

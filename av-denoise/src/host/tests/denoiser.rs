use av_denoise_core::{ChannelMode, DenoisingMode, EdgePadding, Nl4dOptions, NlmTuning, NlmeansOptions};

use crate::backend::Device;
use crate::backend::accelerate::Accelerator;
use crate::host::{Algorithm, DenoiserError, DenoiserOptions, HostDenoiser};

fn opts(mode: DenoisingMode) -> DenoiserOptions {
    DenoiserOptions::builder()
        .channel_mode(ChannelMode::Luma)
        .mode(mode)
        .build()
}

fn create(width: u32, height: u32, options: DenoiserOptions) -> Result<HostDenoiser, DenoiserError> {
    HostDenoiser::create(&[Accelerator::Vulkan], &Device::Default, width, height, options)
}

fn luma_denoiser(mode: DenoisingMode) -> HostDenoiser {
    let options = opts(mode);
    create(16, 16, options).expect("denoiser construction failed")
}

fn frame(width: u32, height: u32) -> Vec<u8> {
    vec![128u8; (width * height) as usize]
}

fn frame_filled(value: u8) -> Vec<u8> {
    vec![value; 16 * 16]
}

/// The single plane of a luma frame.
fn luma_plane(mut planes: Vec<Vec<u8>>) -> Vec<u8> {
    assert_eq!(planes.len(), 1, "a luma frame has one plane");
    planes.remove(0)
}

/// Flushes `denoiser`, collecting each frame's luma plane into `out`.
fn flush_luma(denoiser: &mut HostDenoiser, out: &mut Vec<Vec<u8>>) -> Result<(), DenoiserError> {
    denoiser.flush(|planes| {
        let plane = luma_plane(planes);
        out.push(plane);
    })
}

/// Pushes `count` frames of `value`, receiving whenever the queue is full.
fn push_n_with_drain(denoiser: &mut HostDenoiser, count: usize, value: u8, out: &mut Vec<Vec<u8>>) {
    let plane = frame_filled(value);

    for _ in 0..count {
        loop {
            match denoiser.push(&[&plane]) {
                Ok(()) => break,
                Err(DenoiserError::QueueFull) => {
                    let received = denoiser.recv().expect("recv ok");
                    let planes = received.expect("queue full but recv yielded none");
                    let plane = luma_plane(planes);
                    out.push(plane);
                },
                Err(error) => panic!("unexpected push error: {error:?}"),
            }
        }
    }
}

#[test]
fn spatial_denoise_roundtrip() {
    let mut denoiser = luma_denoiser(DenoisingMode::Spacial);
    assert_eq!(denoiser.selected_accelerator(), Accelerator::Vulkan);

    let plane = frame(16, 16);
    denoiser.push(&[&plane]).expect("push failed");

    let received = denoiser.recv().expect("recv failed");
    let planes = received.expect("no frame");
    let out = luma_plane(planes);
    assert_eq!(out.len(), 16 * 16);
}

#[test]
fn a_plane_that_is_not_a_whole_number_of_words_round_trips() {
    let options = opts(DenoisingMode::Spacial);
    let mut denoiser = create(13, 9, options).expect("denoiser construction failed");

    let plane = frame(13, 9);
    denoiser.push(&[&plane]).expect("push failed");

    let received = denoiser.recv().expect("recv failed");
    let planes = received.expect("no frame");
    let out = luma_plane(planes);
    assert_eq!(out.len(), 13 * 9);
}

#[test]
fn a_chroma_frame_keeps_its_u_and_v_order() {
    let options = DenoiserOptions::builder()
        .channel_mode(ChannelMode::Chroma)
        .mode(DenoisingMode::Spacial)
        .build();
    let mut denoiser = create(16, 16, options).expect("denoiser construction failed");

    let u_plane = frame_filled(60);
    let v_plane = frame_filled(190);
    denoiser.push(&[&u_plane, &v_plane]).expect("push failed");

    let received = denoiser.recv().expect("recv failed");
    let planes = received.expect("no frame");
    assert_eq!(planes.len(), 2);

    for &sample in &planes[0] {
        assert!(
            sample.abs_diff(60) <= 2,
            "U plane sample {sample}, expected about 60"
        );
    }

    for &sample in &planes[1] {
        assert!(
            sample.abs_diff(190) <= 2,
            "V plane sample {sample}, expected about 190"
        );
    }
}

#[test]
fn a_wrong_plane_count_is_rejected_before_uploading() {
    let mut denoiser = luma_denoiser(DenoisingMode::Spacial);

    let plane = frame(16, 16);
    let result = denoiser.push(&[&plane, &plane]);

    let is_plane_mismatch = matches!(
        result,
        Err(DenoiserError::Engine(av_denoise_core::Error::PlaneMismatch(_)))
    );
    assert!(is_plane_mismatch, "got {result:?}");
}

#[test]
fn a_plane_of_the_wrong_length_is_rejected() {
    let mut denoiser = luma_denoiser(DenoisingMode::Spacial);

    let plane = vec![128u8; 16 * 15];
    let result = denoiser.push(&[&plane]);

    assert!(matches!(result, Err(DenoiserError::Other(_))), "got {result:?}");
}

#[test]
fn nl4d_algorithm_round_trips_through_the_host() {
    let options = DenoiserOptions::builder()
        .channel_mode(ChannelMode::Luma)
        .mode(DenoisingMode::Temporal { radius: 2 })
        .algorithm(Algorithm::Nl4d(Nl4dOptions::default()))
        .build();
    let mut denoiser = create(16, 16, options).expect("nl4d denoiser construction failed");
    assert_eq!(denoiser.selected_accelerator(), Accelerator::Vulkan);

    let plane = frame(16, 16);
    denoiser.push(&[&plane]).expect("push failed");

    let received = denoiser.recv().expect("recv failed");
    assert!(received.is_none());

    let mut out = Vec::new();
    flush_luma(&mut denoiser, &mut out).expect("flush failed");
    assert_eq!(out.len(), 1, "expected exactly one output for one pushed frame");
    assert_eq!(out[0].len(), 16 * 16);
}

/// nl4d groups patches across neighbouring frames, so a spatial mode leaves it nothing to do.
#[test]
fn nl4d_rejects_a_spatial_denoising_mode() {
    let options = DenoiserOptions::builder()
        .channel_mode(ChannelMode::Luma)
        .mode(DenoisingMode::Spacial)
        .algorithm(Algorithm::Nl4d(Nl4dOptions::default()))
        .build();
    let result = create(16, 16, options);

    match result {
        Err(DenoiserError::Other(error)) => assert!(
            error.to_string().contains("temporal window"),
            "unexpected error message: {error}"
        ),
        Err(other) => panic!("expected DenoiserError::Other, got {other:?}"),
        Ok(_) => panic!("expected a rejection, got Ok"),
    }
}

#[test]
fn window_span_is_symmetric_for_nlmeans() {
    let options = DenoiserOptions::builder()
        .channel_mode(ChannelMode::Luma)
        .mode(DenoisingMode::Temporal { radius: 3 })
        .algorithm(Algorithm::Nlmeans(NlmeansOptions::default()))
        .build();
    let denoiser = create(16, 16, options).expect("denoiser construction failed");

    let span = denoiser.window_span();
    assert_eq!(span.behind, 3, "behind should equal the temporal radius");
    assert_eq!(span.ahead, 3, "ahead should equal the temporal radius");
    assert_eq!(span.edges, EdgePadding::Repeat);
}

#[test]
fn window_span_is_doubled_on_both_sides_for_nl4d() {
    let options = DenoiserOptions::builder()
        .channel_mode(ChannelMode::Luma)
        .mode(DenoisingMode::Temporal { radius: 3 })
        .algorithm(Algorithm::Nl4d(Nl4dOptions::default()))
        .build();
    let denoiser = create(16, 16, options).expect("nl4d denoiser construction failed");

    let span = denoiser.window_span();
    assert_eq!(span.behind, 6, "behind should equal 2 * the temporal radius");
    assert_eq!(span.ahead, 6, "ahead should equal 2 * the temporal radius");
    assert_eq!(span.edges, EdgePadding::Shifted);
}

#[test]
fn invalid_params_surface_as_error() {
    let tuning = NlmTuning {
        strength: Some(0.0),
        ..NlmTuning::default()
    };
    let algorithm = Algorithm::Nlmeans(NlmeansOptions {
        tuning,
        ..NlmeansOptions::default()
    });
    let options = DenoiserOptions::builder().algorithm(algorithm).build();
    let result = create(16, 16, options);

    match result {
        Err(DenoiserError::Engine(av_denoise_core::Error::InvalidOptions(_))) => {},
        Err(other) => panic!("expected an invalid options error, got {other:?}"),
        Ok(_) => panic!("expected validation error, got Ok"),
    }
}

#[test]
fn tiny_frame_dimensions_surface_as_error() {
    let options = opts(DenoisingMode::Spacial);
    let result = create(2, 2, options);

    match result {
        Err(DenoiserError::Engine(error)) => assert!(
            error.to_string().contains("supported minimum"),
            "unexpected error message: {error}"
        ),
        Err(other) => panic!("expected an engine error, got {other:?}"),
        Ok(_) => panic!("expected dimension validation error, got Ok"),
    }
}

#[test]
fn push_after_pending_returns_queue_full() {
    let mut denoiser = luma_denoiser(DenoisingMode::Spacial);
    let plane = frame(16, 16);

    denoiser.push(&[&plane]).unwrap();
    denoiser.push(&[&plane]).unwrap();
    let error = denoiser.push(&[&plane]).expect_err("expected QueueFull");
    assert!(matches!(error, DenoiserError::QueueFull));

    let received = denoiser.recv().unwrap();
    let planes = received.unwrap();
    let out = luma_plane(planes);
    assert_eq!(out.len(), 16 * 16);

    denoiser.push(&[&plane]).expect("push after drain failed");
}

#[test]
fn queue_full_does_not_poison() {
    let mut denoiser = luma_denoiser(DenoisingMode::Spacial);
    let plane = frame(16, 16);

    denoiser.push(&[&plane]).unwrap();
    denoiser.push(&[&plane]).unwrap();
    let error = denoiser.push(&[&plane]).expect_err("expected QueueFull");
    assert!(matches!(error, DenoiserError::QueueFull));
    assert!(!denoiser.poisoned, "QueueFull must not poison the denoiser");

    let received = denoiser.recv().unwrap();
    received.expect("recv failed after QueueFull");

    denoiser
        .push(&[&plane])
        .expect("push after QueueFull drain should succeed, not poison");
}

#[test]
fn poisoned_denoiser_refuses_every_entry_point() {
    let mut denoiser = luma_denoiser(DenoisingMode::Spacial);
    denoiser.poisoned = true;

    let plane = frame(16, 16);

    let pushed = denoiser.push(&[&plane]);
    assert!(matches!(pushed, Err(DenoiserError::Poisoned)));

    let primed = denoiser.push_priming(&[&plane]);
    assert!(matches!(primed, Err(DenoiserError::Poisoned)));

    let received = denoiser.recv();
    assert!(matches!(received, Err(DenoiserError::Poisoned)));

    let polled = denoiser.try_recv();
    assert!(matches!(polled, Err(DenoiserError::Poisoned)));

    let flushed = denoiser.flush(|_| {});
    assert!(matches!(flushed, Err(DenoiserError::Poisoned)));

    let drained = denoiser.drain_grain_chunks();
    assert!(matches!(drained, Err(DenoiserError::Poisoned)));
}

#[test]
fn a_failed_push_poisons_the_denoiser() {
    let mut denoiser = luma_denoiser(DenoisingMode::Spacial);

    let short_plane = vec![128u8; 4];
    let result = denoiser.push(&[&short_plane]);
    assert!(result.is_err());

    let plane = frame(16, 16);
    let pushed = denoiser.push(&[&plane]);
    assert!(matches!(pushed, Err(DenoiserError::Poisoned)));
}

#[test]
fn reset_stream_clears_poison() {
    let mut denoiser = luma_denoiser(DenoisingMode::Spacial);
    denoiser.poisoned = true;

    denoiser.reset_stream();
    assert!(!denoiser.poisoned, "reset_stream must clear the poison flag");

    let plane = frame(16, 16);
    denoiser
        .push(&[&plane])
        .expect("push after reset_stream should succeed");
}

#[test]
fn flush_leaves_denoiser_reusable_spatial() {
    let mut denoiser = luma_denoiser(DenoisingMode::Spacial);

    let mut batch_a = Vec::new();
    push_n_with_drain(&mut denoiser, 5, 64, &mut batch_a);
    flush_luma(&mut denoiser, &mut batch_a).expect("first flush failed");
    assert_eq!(batch_a.len(), 5);

    let received = denoiser.recv().unwrap();
    assert!(received.is_none());

    let mut batch_b = Vec::new();
    push_n_with_drain(&mut denoiser, 5, 191, &mut batch_b);
    flush_luma(&mut denoiser, &mut batch_b).expect("second flush failed");
    assert_eq!(batch_b.len(), 5);

    for &sample in batch_b.iter().flatten() {
        assert!(
            sample.abs_diff(191) < 25,
            "batch_b carried state from batch_a: {sample}"
        );
    }

    for &sample in batch_a.iter().flatten() {
        assert!(
            sample.abs_diff(64) < 25,
            "batch_a value unexpectedly drifted: {sample}"
        );
    }
}

#[test]
fn flush_leaves_denoiser_reusable_temporal() {
    let mut denoiser = luma_denoiser(DenoisingMode::Temporal { radius: 1 });

    let mut batch_a = Vec::new();
    push_n_with_drain(&mut denoiser, 5, 64, &mut batch_a);
    flush_luma(&mut denoiser, &mut batch_a).expect("first flush failed");
    assert_eq!(batch_a.len(), 5, "expected 5 frames from first batch");

    // With r=1 the window needs more than one push before anything is ready.
    let received = denoiser.recv().unwrap();
    assert!(received.is_none());

    let plane = frame_filled(191);
    denoiser.push(&[&plane]).unwrap();

    let received = denoiser.recv().unwrap();
    assert!(
        received.is_none(),
        "first push of new temporal stream should not produce output yet"
    );

    let mut batch_b = Vec::new();
    push_n_with_drain(&mut denoiser, 4, 191, &mut batch_b);
    flush_luma(&mut denoiser, &mut batch_b).expect("second flush failed");
    assert_eq!(batch_b.len(), 5, "expected 5 frames from second batch");

    for &sample in batch_b.iter().flatten() {
        assert!(
            sample.abs_diff(191) < 25,
            "batch_b carried state from batch_a: {sample}"
        );
    }
}

#[test]
fn flush_emits_exactly_n_outputs_for_small_n() {
    for count in 1..=5usize {
        let mut denoiser = luma_denoiser(DenoisingMode::Temporal { radius: 2 });

        let mut out = Vec::new();
        push_n_with_drain(&mut denoiser, count, 128, &mut out);
        flush_luma(&mut denoiser, &mut out).expect("flush failed");

        assert_eq!(
            out.len(),
            count,
            "expected {count} outputs for {count} pushes, got {}",
            out.len()
        );
    }
}

#[test]
fn dropping_a_polled_pending_frame_does_not_poison_the_device() {
    let new = || {
        let options = opts(DenoisingMode::Spacial);
        create(64, 64, options).unwrap()
    };
    let plane = frame(64, 64);

    let mut denoiser = new();
    denoiser.push(&[&plane]).unwrap();

    // Whether the readback lands on this poll depends on the GPU, and both outcomes must survive the drop.
    let _ = denoiser.try_recv().unwrap();
    drop(denoiser);

    // The staging pool is per device, so a fresh denoiser on the same device is handed the same buffers.
    let mut denoiser = new();
    for _ in 0..4 {
        denoiser.push(&[&plane]).unwrap();

        let received = denoiser
            .recv()
            .expect("readback after a dropped polled frame should not fail");
        received.expect("spatial mode emits one frame per push");
    }

    denoiser.flush(|_| {}).unwrap();
}

#[test]
fn try_recv_observes_a_landed_readback_within_a_bounded_poll() {
    // A deadline covers both a slow GPU and a fast CPU, where a poll count would not.
    const DEADLINE: std::time::Duration = std::time::Duration::from_secs(30);

    let radius = 2u32;
    let new = || {
        let options = opts(DenoisingMode::Temporal { radius });
        create(64, 64, options).unwrap()
    };

    // `radius + 1` pushes fill the window and leave exactly one readback in flight.
    let window: Vec<Vec<u8>> = (0..=radius as usize)
        .map(|index| {
            (0..64 * 64)
                .map(|pixel| ((pixel * 7 + index * 13) % 256) as u8)
                .collect()
        })
        .collect();

    let mut polled = new();
    for plane in &window {
        polled.push(&[plane]).unwrap();
    }

    let start = std::time::Instant::now();
    let mut got = None;
    let mut polls = 0;
    while start.elapsed() < DEADLINE {
        polls += 1;

        if let Some(planes) = polled.try_recv().unwrap() {
            got = Some(planes);
            break;
        }
    }

    let got = got.unwrap_or_else(|| panic!("readback never landed within {DEADLINE:?} ({polls} polls)"));

    let mut blocking = new();
    for plane in &window {
        blocking.push(&[plane]).unwrap();
    }

    let received = blocking.recv().unwrap();
    let expected = received.expect("blocking denoiser should have a frame ready");

    assert_eq!(got, expected);
}

#[test]
fn try_recv_returns_none_when_nothing_is_in_flight() {
    let mut denoiser = luma_denoiser(DenoisingMode::Temporal { radius: 2 });
    let polled = denoiser.try_recv().unwrap();
    assert_eq!(polled, None);
}

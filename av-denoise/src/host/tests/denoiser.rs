use std::time::Duration;

use av_denoise_core::tune::{TuningConfig, TuningMode, build_engine};
use av_denoise_core::{
    ChromaLayout,
    DenoisingMode,
    DevicePlane,
    EdgePadding,
    Engine,
    EngineTiming,
    GrainChunks,
    Nl4dOptions,
    NlmTuning,
    NlmeansHqOptions,
    NlmeansOptions,
    PassTimings,
    PlaneSource,
    WindowSpan,
};

use crate::host::{Algorithm, ChannelIntent, DenoiserError, DenoiserOptions, HostDenoiser, OUTPUT_SETS};

fn luma_options(mode: DenoisingMode) -> DenoiserOptions {
    DenoiserOptions::builder()
        .intent(ChannelIntent::Luma)
        .chroma(ChromaLayout::Gray)
        .mode(mode)
        .build()
}

/// Spatial options cleaning `intent` on a frame laid out as `chroma`.
fn spatial_options(intent: ChannelIntent, chroma: ChromaLayout) -> DenoiserOptions {
    DenoiserOptions::builder()
        .intent(intent)
        .chroma(chroma)
        .mode(DenoisingMode::Spacial)
        .build()
}

fn create(width: u32, height: u32, options: DenoiserOptions) -> Result<HostDenoiser, DenoiserError> {
    let accelerator = crate::testing::test_accelerator();
    let device = crate::testing::test_device();
    HostDenoiser::create(&[accelerator], &device, width, height, options)
}

fn luma_denoiser(mode: DenoisingMode) -> HostDenoiser {
    let options = luma_options(mode);
    create(16, 16, options).expect("denoiser construction failed")
}

fn frame(width: u32, height: u32) -> Vec<u8> {
    vec![128u8; (width * height) as usize]
}

fn frame_filled(value: u8) -> Vec<u8> {
    vec![value; 16 * 16]
}

fn luma_plane(mut planes: Vec<Vec<u8>>) -> Vec<u8> {
    assert_eq!(planes.len(), 1, "a luma frame has one plane");
    planes.remove(0)
}

/// Flushes `denoiser`, collecting each frame's luma plane into `outputs`.
fn flush_luma(denoiser: &mut HostDenoiser, outputs: &mut Vec<Vec<u8>>) -> Result<(), DenoiserError> {
    denoiser.flush(|planes| {
        let plane = luma_plane(planes);
        outputs.push(plane);
    })
}

/// Pushes `count` frames of `value`, receiving whenever the queue is full.
fn push_n_with_drain(denoiser: &mut HostDenoiser, count: usize, value: u8, outputs: &mut Vec<Vec<u8>>) {
    let plane = frame_filled(value);

    for _ in 0..count {
        loop {
            match denoiser.push(&[&plane]) {
                Ok(()) => break,
                Err(DenoiserError::QueueFull) => {
                    let received = denoiser.recv().expect("recv ok");
                    let planes = received.expect("queue full but recv yielded none");
                    let plane = luma_plane(planes);
                    outputs.push(plane);
                },
                Err(error) => panic!("unexpected push error: {error:?}"),
            }
        }
    }
}

#[test]
fn spatial_denoise_roundtrip() {
    let mut denoiser = luma_denoiser(DenoisingMode::Spacial);
    assert_eq!(
        denoiser.selected_accelerator(),
        crate::testing::test_accelerator()
    );

    let plane = frame(16, 16);
    denoiser.push(&[&plane]).expect("push failed");

    let received = denoiser.recv().expect("recv failed");
    let planes = received.expect("no frame");
    let output_plane = luma_plane(planes);

    assert_eq!(output_plane.len(), 16 * 16);
}

#[test]
fn a_host_keeps_what_core_needs_to_rebuild_its_engine() {
    let options = luma_options(DenoisingMode::Temporal { radius: 1 });
    let denoiser = create(16, 16, options).expect("denoiser construction failed");
    let spec = denoiser.spec();
    let config = TuningConfig {
        mode: TuningMode::Off,
        ..TuningConfig::default()
    };

    let rebuilt = build_engine(denoiser.selected(), denoiser.client(), spec, &config);
    let rebuilt = rebuilt.expect("core rebuilds the engine");

    assert_eq!(spec.geometry().width, 16);
    assert_eq!(rebuilt.window_span(), denoiser.window_span());
    assert_eq!(spec.window_span(), denoiser.window_span());
}

#[test]
fn a_plane_that_is_not_a_whole_number_of_words_round_trips() {
    let options = luma_options(DenoisingMode::Spacial);
    let mut denoiser = create(13, 9, options).expect("denoiser construction failed");

    let plane = frame(13, 9);
    denoiser.push(&[&plane]).expect("push failed");

    let received = denoiser.recv().expect("recv failed");
    let planes = received.expect("no frame");
    let output_plane = luma_plane(planes);

    assert_eq!(output_plane.len(), 13 * 9);
}

#[test]
fn a_chroma_frame_keeps_its_u_and_v_order() {
    let options = spatial_options(ChannelIntent::Chroma, ChromaLayout::Yuv444);
    let mut denoiser = create(16, 16, options).expect("denoiser construction failed");

    let y_plane = frame_filled(100);
    let u_plane = frame_filled(60);
    let v_plane = frame_filled(190);
    denoiser
        .push(&[&y_plane, &u_plane, &v_plane])
        .expect("push failed");

    let received = denoiser.recv().expect("recv failed");
    let planes = received.expect("no frame");
    assert_eq!(planes.len(), 3);
    assert_eq!(planes[0], y_plane, "uncleaned luma comes out unchanged");

    for &sample in &planes[1] {
        assert!(
            sample.abs_diff(60) <= 2,
            "U plane sample {sample}, expected about 60"
        );
    }

    for &sample in &planes[2] {
        assert!(
            sample.abs_diff(190) <= 2,
            "V plane sample {sample}, expected about 190"
        );
    }
}

/// A plane of `width` by `height` samples whose values vary, so a passthrough copy is checkable.
fn ramp(width: u32, height: u32, seed: u8) -> Vec<u8> {
    let samples = (width * height) as usize;

    (0..samples)
        .map(|index| (index as u8).wrapping_mul(7).wrapping_add(seed))
        .collect()
}

#[test]
fn every_layout_takes_each_plane_at_its_own_size() {
    let layouts = [
        (ChromaLayout::Gray, vec![(13, 9)]),
        (ChromaLayout::Yuv420, vec![(13, 9), (7, 5), (7, 5)]),
        (ChromaLayout::Yuv422, vec![(13, 9), (7, 9), (7, 9)]),
        (ChromaLayout::Yuv444, vec![(13, 9), (13, 9), (13, 9)]),
    ];

    for (chroma, dims) in layouts {
        let options = spatial_options(ChannelIntent::Luma, chroma);
        let mut denoiser = create(13, 9, options).expect("denoiser construction failed");
        let planes: Vec<Vec<u8>> = dims
            .iter()
            .enumerate()
            .map(|(index, &(width, height))| ramp(width, height, index as u8))
            .collect();
        let frame: Vec<&[u8]> = planes.iter().map(Vec::as_slice).collect();

        denoiser.push(&frame).expect("push failed");
        let received = denoiser.recv().expect("recv failed");
        let output = received.expect("no frame");

        assert_eq!(output.len(), planes.len(), "plane count for {chroma:?}");
        assert_eq!(output[0].len(), planes[0].len(), "luma length for {chroma:?}");
        assert_eq!(output[1..], planes[1..], "uncleaned chroma for {chroma:?}");
    }
}

#[test]
fn a_chroma_plane_at_the_luma_size_is_rejected_for_subsampled_layouts() {
    for chroma in [ChromaLayout::Yuv420, ChromaLayout::Yuv422] {
        let options = spatial_options(ChannelIntent::LumaChroma, chroma);
        let mut denoiser = create(16, 16, options).expect("denoiser construction failed");

        let plane = frame(16, 16);
        let result = denoiser.push(&[&plane, &plane, &plane]);

        let is_plane_mismatch = matches!(
            result,
            Err(DenoiserError::Engine(av_denoise_core::Error::PlaneMismatch(_)))
        );
        assert!(is_plane_mismatch, "{chroma:?} got {result:?}");
    }
}

#[test]
fn a_gray_frame_cannot_clean_chroma() {
    let options = spatial_options(ChannelIntent::Chroma, ChromaLayout::Gray);
    let result = create(16, 16, options);

    let is_invalid_geometry = matches!(
        result,
        Err(DenoiserError::Engine(av_denoise_core::Error::InvalidGeometry(_)))
    );
    assert!(is_invalid_geometry, "got {:?}", result.err());
}

#[test]
fn a_temporal_stream_pairs_uncleaned_planes_with_their_own_frame() {
    let options = DenoiserOptions::builder()
        .intent(ChannelIntent::Chroma)
        .chroma(ChromaLayout::Yuv420)
        .mode(DenoisingMode::Temporal { radius: 2 })
        .build();
    let mut denoiser = create(16, 16, options).expect("denoiser construction failed");
    let (chroma_width, chroma_height) = (8, 8);
    let frames: Vec<[Vec<u8>; 3]> = (0..7u8)
        .map(|index| {
            let y_plane = ramp(16, 16, index * 31);
            let u_plane = ramp(chroma_width, chroma_height, 1);
            let v_plane = ramp(chroma_width, chroma_height, 2);
            [y_plane, u_plane, v_plane]
        })
        .collect();
    let mut outputs: Vec<Vec<Vec<u8>>> = Vec::new();

    for planes in &frames {
        let frame: Vec<&[u8]> = planes.iter().map(Vec::as_slice).collect();
        loop {
            match denoiser.push(&frame) {
                Ok(()) => break,
                Err(DenoiserError::QueueFull) => {
                    let received = denoiser.recv().expect("recv");
                    outputs.extend(received);
                },
                Err(error) => panic!("unexpected push error: {error:?}"),
            }
        }
    }

    denoiser.flush(|planes| outputs.push(planes)).expect("flush");

    assert_eq!(outputs.len(), frames.len());
    for (index, (output, input)) in outputs.iter().zip(&frames).enumerate() {
        assert_eq!(output[0], input[0], "frame {index} luma came from another frame");
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

    let is_plane_mismatch = matches!(
        result,
        Err(DenoiserError::Engine(av_denoise_core::Error::PlaneMismatch(_)))
    );
    assert!(is_plane_mismatch, "got {result:?}");
}

/// An engine that returns scripted outcomes and writes nothing.
///
/// `push_outcome` receives how many pushes came before it.
struct ScriptedEngine {
    push_outcome: fn(usize) -> Result<usize, av_denoise_core::Error>,
    emit_outcome: fn() -> Result<(), av_denoise_core::Error>,
    finish_outcome: fn() -> Result<usize, av_denoise_core::Error>,
    pushes: usize,
}

impl ScriptedEngine {
    fn pushing(push_outcome: fn(usize) -> Result<usize, av_denoise_core::Error>) -> Self {
        Self {
            push_outcome,
            emit_outcome: || Ok(()),
            finish_outcome: || Ok(0),
            pushes: 0,
        }
    }
}

impl Engine for ScriptedEngine {
    fn push(&mut self, _planes: &[DevicePlane<'_>]) -> Result<usize, av_denoise_core::Error> {
        let earlier_pushes = self.pushes;
        self.pushes += 1;

        (self.push_outcome)(earlier_pushes)
    }

    fn push_from(&mut self, _source: &mut dyn PlaneSource) -> Result<usize, av_denoise_core::Error> {
        self.push(&[])
    }

    fn push_context(&mut self, _planes: &[DevicePlane<'_>]) -> Result<(), av_denoise_core::Error> {
        Ok(())
    }

    fn emit_into(&mut self, _planes: &[DevicePlane<'_>]) -> Result<(), av_denoise_core::Error> {
        (self.emit_outcome)()
    }

    fn finish(&mut self) -> Result<usize, av_denoise_core::Error> {
        (self.finish_outcome)()
    }

    fn reset(&mut self) {}

    fn window_span(&self) -> WindowSpan {
        WindowSpan {
            behind: 0,
            ahead: 0,
            edges: EdgePadding::Repeat,
        }
    }

    fn max_held_frames(&self) -> usize {
        0
    }

    fn drain_grain_chunks(&mut self) -> Result<GrainChunks, av_denoise_core::Error> {
        Ok(GrainChunks::default())
    }

    fn timing(&mut self) -> PassTimings {
        PassTimings {
            luma: Some(EngineTiming::default()),
            chroma: None,
        }
    }

    fn sample_device_time(&mut self, _every: u32) {}
}

#[test]
fn a_push_readying_more_frames_than_output_sets_fails_instead_of_overwriting() {
    let mut denoiser = luma_denoiser(DenoisingMode::Spacial);
    let engine = ScriptedEngine::pushing(|_| Ok(OUTPUT_SETS + 1));
    denoiser.engine = Box::new(engine);

    let plane = frame(16, 16);
    let result = denoiser.push(&[&plane]);

    assert!(matches!(result, Err(DenoiserError::Other(_))), "got {result:?}");
    assert!(
        denoiser.pending.is_empty(),
        "no output set should have been emitted"
    );

    let pushed = denoiser.push(&[&plane]);
    assert!(matches!(pushed, Err(DenoiserError::Poisoned)));
}

#[test]
fn a_push_filling_every_output_set_with_one_in_flight_fails_instead_of_overwriting() {
    let mut denoiser = luma_denoiser(DenoisingMode::Spacial);
    let engine = ScriptedEngine::pushing(|earlier_pushes| match earlier_pushes {
        0 => Ok(1),
        _ => Ok(OUTPUT_SETS),
    });
    denoiser.engine = Box::new(engine);

    let plane = frame(16, 16);
    denoiser.push(&[&plane]).expect("first push should land");
    assert_eq!(denoiser.pending.len(), 1);

    let result = denoiser.push(&[&plane]);

    assert!(matches!(result, Err(DenoiserError::Other(_))), "got {result:?}");
    assert_eq!(
        denoiser.pending.len(),
        1,
        "no output set should have been emitted"
    );
}

#[test]
fn a_failed_emit_during_push_poisons_the_denoiser() {
    let mut denoiser = luma_denoiser(DenoisingMode::Spacial);
    let engine = ScriptedEngine {
        emit_outcome: || Err(av_denoise_core::Error::NothingToEmit),
        ..ScriptedEngine::pushing(|_| Ok(1))
    };
    denoiser.engine = Box::new(engine);

    let plane = frame(16, 16);
    let result = denoiser.push(&[&plane]);

    let is_nothing_to_emit = matches!(
        result,
        Err(DenoiserError::Engine(av_denoise_core::Error::NothingToEmit))
    );
    assert!(is_nothing_to_emit, "got {result:?}");

    let pushed = denoiser.push(&[&plane]);
    assert!(matches!(pushed, Err(DenoiserError::Poisoned)));
}

#[test]
fn a_failed_finish_during_flush_poisons_the_denoiser() {
    let mut denoiser = luma_denoiser(DenoisingMode::Spacial);
    let engine = ScriptedEngine {
        finish_outcome: || Err(av_denoise_core::Error::OutputsPending),
        ..ScriptedEngine::pushing(|_| Ok(0))
    };
    denoiser.engine = Box::new(engine);

    let flushed = denoiser.flush(|_| {});

    let is_outputs_pending = matches!(
        flushed,
        Err(DenoiserError::Engine(av_denoise_core::Error::OutputsPending))
    );
    assert!(is_outputs_pending, "got {flushed:?}");

    let plane = frame(16, 16);
    let pushed = denoiser.push(&[&plane]);
    assert!(matches!(pushed, Err(DenoiserError::Poisoned)));
}

#[test]
fn nl4d_algorithm_round_trips_through_the_host() {
    let algorithm = Algorithm::Nl4d(Nl4dOptions::default());
    let options = DenoiserOptions::builder()
        .intent(ChannelIntent::Luma)
        .chroma(ChromaLayout::Gray)
        .mode(DenoisingMode::Temporal { radius: 2 })
        .algorithm(algorithm)
        .build();
    let mut denoiser = create(16, 16, options).expect("nl4d denoiser construction failed");
    assert_eq!(
        denoiser.selected_accelerator(),
        crate::testing::test_accelerator()
    );

    let plane = frame(16, 16);
    denoiser.push(&[&plane]).expect("push failed");

    let received = denoiser.recv().expect("recv failed");
    assert!(received.is_none());

    let mut outputs = Vec::new();
    flush_luma(&mut denoiser, &mut outputs).expect("flush failed");

    assert_eq!(
        outputs.len(),
        1,
        "expected exactly one output for one pushed frame"
    );
    assert_eq!(outputs[0].len(), 16 * 16);
}

/// nl4d groups patches across neighbouring frames, so a spatial mode leaves it nothing to do.
#[test]
fn nl4d_rejects_a_spatial_denoising_mode() {
    let algorithm = Algorithm::Nl4d(Nl4dOptions::default());
    let options = DenoiserOptions::builder()
        .intent(ChannelIntent::Luma)
        .chroma(ChromaLayout::Gray)
        .mode(DenoisingMode::Spacial)
        .algorithm(algorithm)
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
    let algorithm = Algorithm::Nlmeans(NlmeansOptions::default());
    let options = DenoiserOptions::builder()
        .intent(ChannelIntent::Luma)
        .chroma(ChromaLayout::Gray)
        .mode(DenoisingMode::Temporal { radius: 3 })
        .algorithm(algorithm)
        .build();
    let denoiser = create(16, 16, options).expect("denoiser construction failed");

    let span = denoiser.window_span();

    assert_eq!(span.behind, 3, "behind should equal the temporal radius");
    assert_eq!(span.ahead, 3, "ahead should equal the temporal radius");
    assert_eq!(span.edges, EdgePadding::Repeat);
}

#[test]
fn window_span_is_doubled_and_looks_ahead_on_both_sides_for_nl4d() {
    let algorithm = Algorithm::Nl4d(Nl4dOptions::default());
    let options = DenoiserOptions::builder()
        .intent(ChannelIntent::Luma)
        .chroma(ChromaLayout::Gray)
        .mode(DenoisingMode::Temporal { radius: 3 })
        .algorithm(algorithm)
        .build();
    let denoiser = create(16, 16, options).expect("nl4d denoiser construction failed");

    let span = denoiser.window_span();

    // Twice the radius, plus the 8 frames the noise look-ahead holds.
    assert_eq!(
        span.behind, 14,
        "behind should equal 2 * the temporal radius plus the look-ahead"
    );
    assert_eq!(
        span.ahead, 14,
        "ahead should equal 2 * the temporal radius plus the look-ahead"
    );
    assert_eq!(span.edges, EdgePadding::Shifted);
}

#[test]
fn window_span_adds_the_lookahead_for_nlmeans_hq() {
    let algorithm = Algorithm::NlmeansHq(NlmeansHqOptions::default());
    let options = DenoiserOptions::builder()
        .intent(ChannelIntent::Luma)
        .chroma(ChromaLayout::Gray)
        .mode(DenoisingMode::Temporal { radius: 3 })
        .algorithm(algorithm)
        .build();
    let denoiser = create(16, 16, options).expect("denoiser construction failed");

    let span = denoiser.window_span();

    // The radius, plus the 8 frames the noise look-ahead holds.
    assert_eq!(
        span.behind, 11,
        "behind should equal the temporal radius plus the look-ahead"
    );
    assert_eq!(
        span.ahead, 11,
        "ahead should equal the temporal radius plus the look-ahead"
    );
    assert_eq!(span.edges, EdgePadding::Repeat);
}

#[test]
fn invalid_params_surface_as_error() {
    let tuning = NlmTuning {
        strength: Some(0.0),
        ..NlmTuning::default()
    };
    let nlmeans_options = NlmeansOptions {
        tuning,
        ..NlmeansOptions::default()
    };
    let algorithm = Algorithm::Nlmeans(nlmeans_options);
    let options = DenoiserOptions::builder()
        .intent(ChannelIntent::Luma)
        .chroma(ChromaLayout::Gray)
        .algorithm(algorithm)
        .build();
    let result = create(16, 16, options);

    match result {
        Err(DenoiserError::Engine(av_denoise_core::Error::InvalidOptions(_))) => {},
        Err(other) => panic!("expected an invalid options error, got {other:?}"),
        Ok(_) => panic!("expected validation error, got Ok"),
    }
}

#[test]
fn tiny_frame_dimensions_surface_as_error() {
    let options = luma_options(DenoisingMode::Spacial);
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
    let output_plane = luma_plane(planes);

    assert_eq!(output_plane.len(), 16 * 16);

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

    let flushed = denoiser.flush(|_| {});
    assert!(matches!(flushed, Err(DenoiserError::Poisoned)));

    let drained = denoiser.drain_grain_chunks();
    assert!(matches!(drained, Err(DenoiserError::Poisoned)));
}

#[test]
fn a_gpu_failure_during_push_poisons_the_denoiser() {
    let mut denoiser = luma_denoiser(DenoisingMode::Spacial);
    let engine = ScriptedEngine::pushing(|_| {
        let failure = anyhow::anyhow!("synthetic dispatch failure");
        Err(av_denoise_core::Error::Gpu(failure))
    });
    denoiser.engine = Box::new(engine);

    let plane = frame(16, 16);
    let result = denoiser.push(&[&plane]);
    assert!(result.is_err());

    let pushed = denoiser.push(&[&plane]);
    assert!(matches!(pushed, Err(DenoiserError::Poisoned)));
}

#[test]
fn a_rejected_plane_does_not_poison_the_denoiser() {
    let mut denoiser = luma_denoiser(DenoisingMode::Spacial);

    let short_plane = vec![128u8; 4];
    let result = denoiser.push(&[&short_plane]);
    assert!(result.is_err());

    let plane = frame(16, 16);
    denoiser
        .push(&[&plane])
        .expect("a push after a rejected plane should land");
}

#[test]
fn priming_after_a_push_is_rejected_without_poisoning() {
    let mut denoiser = luma_denoiser(DenoisingMode::Temporal { radius: 1 });
    let plane = frame(16, 16);

    denoiser.push(&[&plane]).expect("first push should land");

    let primed = denoiser.push_priming(&[&plane]);
    let is_context_after_push = matches!(
        primed,
        Err(DenoiserError::Engine(av_denoise_core::Error::ContextAfterPush))
    );
    assert!(is_context_after_push, "got {primed:?}");

    denoiser
        .push(&[&plane])
        .expect("a push after a rejected priming frame should land");
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

    // With radius 1 the window needs more than one push before anything is ready.
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

        let mut outputs = Vec::new();
        push_n_with_drain(&mut denoiser, count, 128, &mut outputs);
        flush_luma(&mut denoiser, &mut outputs).expect("flush failed");

        assert_eq!(
            outputs.len(),
            count,
            "expected {count} outputs for {count} pushes, got {}",
            outputs.len()
        );
    }
}

#[test]
fn flush_sink_time_is_left_out_of_submit_time() {
    let mut denoiser = luma_denoiser(DenoisingMode::Spacial);
    let engine = ScriptedEngine {
        finish_outcome: || Ok(1),
        ..ScriptedEngine::pushing(|_| Ok(1))
    };
    denoiser.engine = Box::new(engine);

    let plane = frame(16, 16);
    denoiser.push(&[&plane]).expect("first push should land");
    denoiser.push(&[&plane]).expect("second push should land");

    let sink_sleep = Duration::from_millis(50);
    let mut flushed = 0;
    denoiser
        .flush(|_| {
            std::thread::sleep(sink_sleep);
            flushed += 1;
        })
        .expect("flush");

    let timing = denoiser.timing();
    let total_sleep = sink_sleep * flushed;

    assert_eq!(flushed, 3);
    assert_eq!(timing.frames, 3);
    assert!(
        timing.submit < total_sleep,
        "submit {:?} includes the sink's {total_sleep:?}",
        timing.submit
    );
}

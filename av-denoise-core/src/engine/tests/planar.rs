use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use cubecl::prelude::*;
use cubecl::server::Handle;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Level, Metadata, Subscriber};

use crate::engine::planar::{GRAY_WARNING, stream_warning};
use crate::engine::{
    ChromaLayout,
    ChromaSiting,
    DevicePlane,
    EdgePadding,
    Engine,
    EngineSpec,
    FramePass,
    Geometry,
    PassSpec,
    PlaneSource,
    SampleFormat,
};
use crate::error::Error;
use crate::motion::{MotionCompensationMode, MotionSearch, PairModelProducer, SharedMotion};
use crate::nl4d::Nl4dOptions;
use crate::nl4d::tests::helpers::noisy_frames;
use crate::nlmeans::tests::luma_motion::{flat_chroma, panning_luma};
use crate::nlmeans::tests::models::{PanShape, pan_clip};
use crate::nlmeans::{DenoisingMode, NlmeansAlgorithm, NlmeansHqOptions, NlmeansOptions};
use crate::noise::colour::{ColourBlocks, ColourGrid};
use crate::tune::EngineKernels;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
/// Enough frames for a radius 1 engine to emit some frames from pushes through its noise look-ahead.
const FRAMES: usize = 14;
const CONTEXT_FRAMES: usize = 4;
/// Enough frames to hold a whole reseed window, noise look-ahead included, mid-clip.
const RESEED_FRAMES: usize = 30;

type Planes = Vec<Vec<u8>>;

fn client() -> Client {
    crate::tests::compilation_cache::install_test_compilation_cache();
    crate::testing::test_client()
}

fn geometry(chroma: ChromaLayout) -> Geometry {
    Geometry {
        width: WIDTH,
        height: HEIGHT,
        chroma,
        input: SampleFormat::U8,
        output: SampleFormat::U8,
        chroma_siting: ChromaSiting::Unknown,
    }
}

fn nl4d_options() -> Nl4dOptions {
    Nl4dOptions {
        temporal_radius: 1,
        ..Nl4dOptions::default()
    }
}

fn nlmeans_algorithm() -> NlmeansAlgorithm {
    let nlm = NlmeansOptions {
        mode: DenoisingMode::Temporal { radius: 1 },
        ..NlmeansOptions::default()
    };
    let hq = NlmeansHqOptions {
        nlm,
        ..NlmeansHqOptions::default()
    };

    NlmeansAlgorithm::Hq(hq)
}

/// An nl4d and an nlmeans spec, each cleaning luma when `luma` and chroma when `chroma`.
fn specs(geometry: Geometry, luma: bool, chroma: bool) -> [EngineSpec; 2] {
    let nl4d = EngineSpec::Nl4d {
        luma: luma.then(nl4d_options),
        chroma: chroma.then(nl4d_options),
        geometry,
    };
    let nlmeans = EngineSpec::Nlmeans {
        luma: luma.then(nlmeans_algorithm),
        chroma: chroma.then(nlmeans_algorithm),
        geometry,
    };

    [nl4d, nlmeans]
}

/// HQ nlmeans at radius 1 compensating motion with `motion`.
fn nlmeans_mc_algorithm(motion: MotionCompensationMode) -> NlmeansAlgorithm {
    let nlm = NlmeansOptions {
        mode: DenoisingMode::Temporal { radius: 1 },
        motion_compensation: motion,
        ..NlmeansOptions::default()
    };
    let hq = NlmeansHqOptions {
        nlm,
        ..NlmeansHqOptions::default()
    };

    NlmeansAlgorithm::Hq(hq)
}

/// An nl4d and a motion-compensated nlmeans spec, each cleaning luma and chroma.
fn motion_specs(geometry: Geometry) -> [EngineSpec; 2] {
    let motion = MotionCompensationMode::mvtools_default();
    let nl4d = EngineSpec::Nl4d {
        luma: Some(nl4d_options()),
        chroma: Some(nl4d_options()),
        geometry,
    };
    let nlmeans = EngineSpec::Nlmeans {
        luma: Some(nlmeans_mc_algorithm(motion)),
        chroma: Some(nlmeans_mc_algorithm(motion)),
        geometry,
    };

    [nl4d, nlmeans]
}

/// `spec` with its luma pass dropped, so it cleans chroma alone.
fn without_luma(spec: EngineSpec) -> EngineSpec {
    match spec {
        EngineSpec::Nl4d { chroma, geometry, .. } => EngineSpec::Nl4d {
            luma: None,
            chroma,
            geometry,
        },
        EngineSpec::Nlmeans { chroma, geometry, .. } => EngineSpec::Nlmeans {
            luma: None,
            chroma,
            geometry,
        },
    }
}

/// [motion_specs] and the same specs cleaning chroma alone.
fn motion_and_chroma_only_specs(geometry: Geometry) -> Vec<EngineSpec> {
    let both = motion_specs(geometry);
    let chroma_only = both.map(without_luma);

    both.into_iter().chain(chroma_only).collect()
}

fn default_kernels(_pass: &PassSpec) -> EngineKernels {
    EngineKernels::default()
}

fn build(client: &Client, spec: &EngineSpec) -> Result<Box<dyn Engine>, Error> {
    spec.build(client, default_kernels)
}

/// A pass engine that reads the camera models a producer fits from the clip's luma, and the block
/// colour of the clip's frames, as the passes of a merged engine do.
struct SeededPass {
    pass: Box<dyn FramePass>,
    producer: Option<PairModelProducer>,
    colour: Option<ColourBlocks>,
    /// The clip's planes, by stream frame.
    planes: Vec<Vec<Handle>>,
}

impl SeededPass {
    /// Fits the model and measures the block colour of the frame the pass takes next.
    fn fit(&mut self) -> Result<(), Error> {
        let frame = self.pass.next_frame();
        let handles = &self.planes[frame as usize];
        let dims = plane_dims(handles.len(), true);
        let planes = as_planes(handles, &dims);

        if let Some(colour) = self.colour.as_ref() {
            colour.queue(&planes, frame);
        }

        let Some(producer) = self.producer.as_mut() else {
            return Ok(());
        };

        producer.push(&planes[0], frame).map_err(Error::Gpu)
    }
}

/// The pass of `spec` cleaning luma when `luma`, otherwise chroma, fed the models of `clip`.
fn seeded_pass(client: &Client, spec: &EngineSpec, luma: bool, clip: &[Planes]) -> SeededPass {
    let (luma_pass, chroma_pass) = spec.passes();
    let pass = if luma { luma_pass } else { chroma_pass };
    let pass = pass.expect("the spec has this pass");

    let producer = spec.pair_model_producer(client).expect("the producer builds");
    let motion = producer.as_ref().map(|producer| SharedMotion {
        ring: producer.ring().clone(),
        luma_motion: None,
    });
    let kernels = EngineKernels::default();
    let mut pass = pass
        .build_pass(client, kernels, motion, false)
        .expect("build pass");

    let colour = spec.colour_blocks(client);
    if let Some(blocks) = colour.as_ref() {
        let grid = if luma {
            ColourGrid::Luma
        } else {
            ColourGrid::Chroma
        };
        let source = blocks.source(grid);
        pass.measure_colour(source);
    }

    let planes = clip.iter().map(|frame| upload(client, frame)).collect();

    SeededPass {
        pass,
        producer,
        colour,
        planes,
    }
}

fn to_codes(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .map(|sample| (sample * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect()
}

fn deinterleave(frame: &[f32], channel: usize) -> Vec<f32> {
    frame.iter().skip(channel).step_by(2).copied().collect()
}

/// A noisy 4:2:0 clip of `FRAMES` frames as `u8` planes, with distinct luma and chroma content.
fn clip() -> Vec<Planes> {
    clip_of(FRAMES)
}

/// [clip] at `count` frames.
fn clip_of(count: usize) -> Vec<Planes> {
    let luma_frames = noisy_frames(WIDTH, HEIGHT, 1, count);
    let chroma_frames = noisy_frames(WIDTH / 2, HEIGHT / 2, 2, count);

    luma_frames
        .iter()
        .zip(&chroma_frames)
        .map(|(luma, chroma)| {
            let u_plane = deinterleave(chroma, 0);
            let v_plane = deinterleave(chroma, 1);
            vec![to_codes(luma), to_codes(&u_plane), to_codes(&v_plane)]
        })
        .collect()
}

/// A noisy 4:2:0 textured pan of `count` frames as `u8` planes, moving 2 luma pixels a frame.
fn pan_planes(count: usize) -> Vec<Planes> {
    let shape = PanShape {
        width: WIDTH,
        height: HEIGHT,
        chroma_scale: (2, 2),
        shift: (2, 0),
        sigma: 2.0 / 255.0,
    };
    let frames = pan_clip(shape, count, 5);

    frames
        .iter()
        .map(|frame| {
            let u_plane = deinterleave(&frame.chroma, 0);
            let v_plane = deinterleave(&frame.chroma, 1);
            vec![to_codes(&frame.luma), to_codes(&u_plane), to_codes(&v_plane)]
        })
        .collect()
}

/// A noisy 4:2:0 clip of `count` frames whose luma pans 4 pixels a frame over flat chroma grain.
///
/// No chroma search finds the pan, so the chroma output shows whether luma motion guided it.
fn luma_panning_over_flat_chroma(count: usize) -> Vec<Planes> {
    let sigma = 2.0 / 255.0;
    let luma_frames = panning_luma(WIDTH, HEIGHT, 4, sigma, count);
    let chroma_frames = flat_chroma(WIDTH / 2, HEIGHT / 2, sigma, count);

    luma_frames
        .iter()
        .zip(&chroma_frames)
        .map(|(luma, chroma)| {
            let u_plane = deinterleave(chroma, 0);
            let v_plane = deinterleave(chroma, 1);
            vec![to_codes(luma), to_codes(&u_plane), to_codes(&v_plane)]
        })
        .collect()
}

/// The `y` plane of each frame alone.
fn luma_of(clip: &[Planes]) -> Vec<Planes> {
    clip.iter().map(|frame| vec![frame[0].clone()]).collect()
}

/// The `u` and `v` planes of each frame.
fn chroma_of(clip: &[Planes]) -> Vec<Planes> {
    clip.iter().map(|frame| frame[1..].to_vec()).collect()
}

/// The calls a test drives, shared by the merged engine and a pass engine.
trait Stream {
    fn push(&mut self, planes: &[DevicePlane<'_>]) -> Result<usize, Error>;
    fn push_context(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error>;
    fn emit_into(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error>;
    fn finish(&mut self) -> Result<usize, Error>;
}

impl Stream for Box<dyn Engine> {
    fn push(&mut self, planes: &[DevicePlane<'_>]) -> Result<usize, Error> {
        Engine::push(self.as_mut(), planes)
    }

    fn push_context(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error> {
        Engine::push_context(self.as_mut(), planes)
    }

    fn emit_into(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error> {
        Engine::emit_into(self.as_mut(), planes)
    }

    fn finish(&mut self) -> Result<usize, Error> {
        Engine::finish(self.as_mut())
    }
}

impl Stream for SeededPass {
    fn push(&mut self, planes: &[DevicePlane<'_>]) -> Result<usize, Error> {
        self.fit()?;
        self.pass.push(planes)
    }

    fn push_context(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error> {
        self.fit()?;
        self.pass.push_context(planes)
    }

    fn emit_into(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error> {
        self.pass.emit_into(planes)
    }

    fn finish(&mut self) -> Result<usize, Error> {
        self.pass.finish()
    }
}

/// The width and height of each plane a frame of `count` planes holds.
fn plane_dims(count: usize, luma_first: bool) -> Vec<(u32, u32)> {
    let chroma = (WIDTH / 2, HEIGHT / 2);
    match (count, luma_first) {
        (1, true) => vec![(WIDTH, HEIGHT)],
        (2, false) => vec![chroma, chroma],
        (3, true) => vec![(WIDTH, HEIGHT), chroma, chroma],
        _ => panic!("no layout of {count} planes"),
    }
}

fn upload(client: &Client, frame: &Planes) -> Vec<Handle> {
    frame
        .iter()
        .map(|plane| client.create_from_slice(plane))
        .collect()
}

fn as_planes<'a>(handles: &'a [Handle], dims: &[(u32, u32)]) -> Vec<DevicePlane<'a>> {
    handles
        .iter()
        .zip(dims)
        .map(|(handle, &(width, height))| DevicePlane::new(handle, width, height))
        .collect()
}

fn emit(
    stream: &mut impl Stream,
    client: &Client,
    dims: &[(u32, u32)],
    ready: usize,
    outputs: &mut Vec<Planes>,
) {
    for _ in 0..ready {
        let handles: Vec<Handle> = dims
            .iter()
            .map(|&(width, height)| {
                let pixels = u64::from(width * height);
                let bytes = SampleFormat::U8.plane_bytes(pixels);
                client.empty(bytes as usize)
            })
            .collect();
        let planes = as_planes(&handles, dims);
        stream.emit_into(&planes).expect("emit");

        let frame = handles
            .iter()
            .map(|handle| {
                let bytes = client.read_one(handle.clone()).expect("read plane");
                bytes.to_vec()
            })
            .collect();
        outputs.push(frame);
    }
}

/// Feeds the first `context` frames as context and the rest as pushes, then the tail.
fn drive(
    stream: &mut impl Stream,
    client: &Client,
    frames: &[Planes],
    context: usize,
    luma_first: bool,
) -> Vec<Planes> {
    let dims = plane_dims(frames[0].len(), luma_first);
    let mut outputs = Vec::new();

    for (index, frame) in frames.iter().enumerate() {
        let handles = upload(client, frame);
        let planes = as_planes(&handles, &dims);

        if index < context {
            stream.push_context(&planes).expect("push_context");
            continue;
        }

        let ready = stream.push(&planes).expect("push");
        emit(stream, client, &dims, ready, &mut outputs);
    }

    let tail = stream.finish().expect("finish");
    emit(stream, client, &dims, tail, &mut outputs);

    outputs
}

#[test]
fn luma_output_is_unchanged_with_the_ring() {
    let client = client();
    let clip = clip();
    let geometry = geometry(ChromaLayout::Yuv420);

    for spec in motion_specs(geometry) {
        assert!(spec.luma_motion_ring(&client).is_some(), "{spec:?}");

        let mut luma_engine = seeded_pass(&client, &spec, true, &clip);
        let luma_frames = luma_of(&clip);
        let luma_expected = drive(&mut luma_engine, &client, &luma_frames, 0, true);

        let mut engine = build(&client, &spec).expect("build");
        let merged = drive(&mut engine, &client, &clip, 0, true);

        assert_eq!(merged.len(), FRAMES, "{spec:?}");
        assert_eq!(luma_expected.len(), FRAMES, "{spec:?}");
        for (index, frame) in merged.iter().enumerate() {
            assert_eq!(frame[0], luma_expected[index][0], "{spec:?} frame {index} luma");
        }
    }
}

#[test]
fn the_chroma_siting_leaves_the_output_unchanged() {
    let client = client();
    let clip = clip();
    let unknown = geometry(ChromaLayout::Yuv420);
    let left = Geometry {
        chroma_siting: ChromaSiting::Left,
        ..unknown
    };

    let unknown_specs = specs(unknown, true, true);
    let left_specs = specs(left, true, true);
    for (unknown_spec, left_spec) in unknown_specs.iter().zip(&left_specs) {
        let mut unknown_engine = build(&client, unknown_spec).expect("build");
        let unknown_output = drive(&mut unknown_engine, &client, &clip, 0, true);

        let mut left_engine = build(&client, left_spec).expect("build");
        let left_output = drive(&mut left_engine, &client, &clip, 0, true);

        assert_eq!(left_output, unknown_output, "{left_spec:?}");
    }
}

#[test]
fn chroma_only_without_motion_matches_its_pass_engine() {
    let client = client();
    let clip = clip();
    let geometry = geometry(ChromaLayout::Yuv420);
    let [_, spec] = specs(geometry, false, true);
    assert!(spec.luma_motion_ring(&client).is_none(), "{spec:?}");

    let mut chroma_engine = seeded_pass(&client, &spec, false, &clip);
    let chroma_frames = chroma_of(&clip);
    let expected = drive(&mut chroma_engine, &client, &chroma_frames, 0, false);

    let mut engine = build(&client, &spec).expect("build");
    let merged = drive(&mut engine, &client, &clip, 0, true);

    assert_eq!(merged.len(), expected.len(), "{spec:?}");
    for (index, frame) in merged.iter().enumerate() {
        assert_eq!(frame[1..], expected[index][..], "{spec:?} frame {index} chroma");
    }
}

#[test]
fn chroma_only_matches_the_luma_plus_chroma_chroma() {
    let client = client();
    let clip = luma_panning_over_flat_chroma(RESEED_FRAMES);
    let geometry = geometry(ChromaLayout::Yuv420);

    for spec in motion_specs(geometry) {
        let chroma_only = without_luma(spec);
        let mut both_engine = build(&client, &spec).expect("build");
        let mut engine = build(&client, &chroma_only).expect("build");

        // A context-led stream, then a scene-start stream on the same engines.
        for context in [CONTEXT_FRAMES, 0] {
            let expected = drive(&mut both_engine, &client, &clip, context, true);
            let actual = drive(&mut engine, &client, &clip, context, true);

            assert_eq!(actual.len(), expected.len(), "{chroma_only:?}");
            for (index, frame) in actual.iter().enumerate() {
                assert_eq!(
                    frame[1..],
                    expected[index][1..],
                    "{chroma_only:?} context {context} frame {index} chroma"
                );
            }
        }
    }
}

#[test]
fn chroma_only_passes_luma_through_unchanged() {
    let client = client();
    let clip = pan_planes(RESEED_FRAMES);
    let geometry = geometry(ChromaLayout::Yuv420);

    for spec in motion_specs(geometry).map(without_luma) {
        let mut engine = build(&client, &spec).expect("build");

        for context in [CONTEXT_FRAMES, 0] {
            let outputs = drive(&mut engine, &client, &clip, context, true);

            // The outputs are the clip's last frames in order.
            let first = RESEED_FRAMES - outputs.len();
            for (index, frame) in outputs.iter().enumerate() {
                let source = &clip[first + index];
                assert_eq!(
                    frame[0], source[0],
                    "{spec:?} context {context} frame {index} luma"
                );
            }
        }
    }
}

/// The output of the frame at `target` after a reseed, as the host runs one.
///
/// A shifted-edge engine takes the frames behind the target as context and pushes the rest. A
/// repeated-edge engine takes every frame but the last as context.
fn reseed_at(engine: &mut Box<dyn Engine>, client: &Client, clip: &[Planes], target: usize) -> Planes {
    let span = engine.window_span();
    let first = target - span.behind;
    let window = &clip[first..first + span.frame_count()];
    let context = match span.edges {
        EdgePadding::Shifted => span.behind,
        EdgePadding::Repeat => span.behind + span.ahead,
    };
    let dims = plane_dims(3, true);
    let mut outputs = Vec::new();

    engine.reset();
    for (index, frame) in window.iter().enumerate() {
        let handles = upload(client, frame);
        let planes = as_planes(&handles, &dims);

        if index < context {
            engine.push_context(&planes).expect("push_context");
            continue;
        }

        let ready = engine.push(&planes).expect("push");
        emit(engine, client, &dims, ready, &mut outputs);
    }

    outputs.pop().expect("a full window emits its target")
}

/// The frame a reseed targets, mid-clip with its whole window inside a clip of `frames`.
fn reseed_target(engine: &dyn Engine, frames: usize) -> usize {
    let span = engine.window_span();
    let target = frames / 2;
    let inside = target > span.behind && target + span.ahead + 1 < frames;
    assert!(inside, "test premise: {span:?} fits around frame {target}");

    target
}

#[test]
fn reseed_equals_streaming_for_luma_and_chroma() {
    let client = client();
    let clip = pan_planes(RESEED_FRAMES);
    let geometry = geometry(ChromaLayout::Yuv420);

    for spec in motion_and_chroma_only_specs(geometry) {
        let mut engine = build(&client, &spec).expect("build");
        let streamed = drive(&mut engine, &client, &clip, 0, true);
        let target = reseed_target(engine.as_ref(), clip.len());

        let reseeded = reseed_at(&mut engine, &client, &clip, target);

        assert_eq!(reseeded, streamed[target], "{spec:?}");
    }
}

#[test]
fn a_reseed_never_reads_stale_luma_motion() {
    let client = client();
    let clip = clip_of(RESEED_FRAMES);
    let other_clip: Vec<Planes> = clip.iter().rev().cloned().collect();
    let geometry = geometry(ChromaLayout::Yuv420);
    let dims = plane_dims(3, true);

    for spec in motion_and_chroma_only_specs(geometry) {
        let mut fresh = build(&client, &spec).expect("build");
        let target = reseed_target(fresh.as_ref(), clip.len());
        let expected = reseed_at(&mut fresh, &client, &clip, target);

        // A stream of other frames leaves its luma motion in the ring before the reseed.
        let mut engine = build(&client, &spec).expect("build");
        let mut outputs = Vec::new();
        for frame in &other_clip {
            let handles = upload(&client, frame);
            let planes = as_planes(&handles, &dims);
            let ready = engine.push(&planes).expect("push");
            emit(&mut engine, &client, &dims, ready, &mut outputs);
        }

        let reseeded = reseed_at(&mut engine, &client, &clip, target);

        assert_eq!(reseeded, expected, "{spec:?}");
    }
}

#[test]
fn mismatched_pass_settings_are_rejected() {
    let client = client();
    let geometry = geometry(ChromaLayout::Yuv420);
    let narrow_search = MotionSearch {
        search_radius: 2,
        ..MotionSearch::default()
    };
    let narrow_nl4d = Nl4dOptions {
        motion: narrow_search,
        ..nl4d_options()
    };
    let narrow_motion = MotionCompensationMode::from(narrow_search);
    let mismatched = [
        EngineSpec::Nl4d {
            luma: Some(nl4d_options()),
            chroma: Some(narrow_nl4d),
            geometry,
        },
        EngineSpec::Nlmeans {
            luma: Some(nlmeans_mc_algorithm(MotionCompensationMode::mvtools_default())),
            chroma: Some(nlmeans_mc_algorithm(narrow_motion)),
            geometry,
        },
        EngineSpec::Nlmeans {
            luma: Some(nlmeans_mc_algorithm(MotionCompensationMode::mvtools_default())),
            chroma: Some(nlmeans_algorithm()),
            geometry,
        },
    ];

    for spec in mismatched {
        let result = build(&client, &spec);
        let Err(Error::InvalidOptions(message)) = result else {
            panic!("{spec:?} built with mismatched motion settings");
        };

        assert!(
            message.contains("takes its motion from the luma pass"),
            "{message}"
        );
    }
}

#[test]
fn greyscale_builds_no_ring() {
    let client = client();
    let gray = geometry(ChromaLayout::Gray);
    let yuv = geometry(ChromaLayout::Yuv420);
    let [nl4d_gray, _] = specs(gray, true, false);
    let [nl4d_luma, _] = specs(yuv, true, false);
    let [nl4d_chroma, nlmeans_chroma_without_motion] = specs(yuv, false, true);
    let [nl4d_both, nlmeans_both] = motion_specs(yuv);
    let [_, nlmeans_without_motion] = specs(yuv, true, true);
    let nlmeans_chroma = without_luma(nlmeans_both);

    assert!(nl4d_gray.luma_motion_ring(&client).is_none());
    assert!(nl4d_luma.luma_motion_ring(&client).is_none());
    assert!(nlmeans_without_motion.luma_motion_ring(&client).is_none());
    assert!(nlmeans_chroma_without_motion.luma_motion_ring(&client).is_none());
    assert!(nl4d_both.luma_motion_ring(&client).is_some());
    assert!(nlmeans_both.luma_motion_ring(&client).is_some());
    assert!(nl4d_chroma.luma_motion_ring(&client).is_some());
    assert!(nlmeans_chroma.luma_motion_ring(&client).is_some());
}

/// Checks a merged engine cleaning one plane group matches its pass engine there, and that the
/// other group comes out as the source of the same frame.
///
/// It runs a context-led stream and then a scene-start stream on the same engines.
fn assert_passes_through(client: &Client, spec: &EngineSpec, clean_luma: bool) {
    let clip = clip();
    let (cleaned_frames, luma_first) = if clean_luma {
        (luma_of(&clip), true)
    } else {
        (chroma_of(&clip), false)
    };
    let mut pass_engine = seeded_pass(client, spec, clean_luma, &clip);
    let mut engine = build(client, spec).expect("build");

    for context in [CONTEXT_FRAMES, 0] {
        let expected = drive(&mut pass_engine, client, &cleaned_frames, context, luma_first);
        let merged = drive(&mut engine, client, &clip, context, true);

        assert_eq!(merged.len(), expected.len(), "{spec:?}");
        if context > 0 {
            assert!(
                merged.len() > FRAMES - context,
                "{spec:?} emitted no context frames"
            );
        }

        // The outputs are the clip's last frames in order.
        let first = FRAMES - merged.len();
        for (index, frame) in merged.iter().enumerate() {
            let source = &clip[first + index];
            if clean_luma {
                assert_eq!(frame[0], expected[index][0], "{spec:?} frame {index} luma");
                assert_eq!(frame[1..], source[1..], "{spec:?} frame {index} chroma");
            } else {
                assert_eq!(frame[0], source[0], "{spec:?} frame {index} luma");
                assert_eq!(frame[1..], expected[index][..], "{spec:?} frame {index} chroma");
            }
        }
    }
}

#[test]
fn an_uncleaned_plane_comes_out_unchanged() {
    let client = client();
    let geometry = geometry(ChromaLayout::Yuv420);

    for spec in specs(geometry, true, false) {
        assert_passes_through(&client, &spec, true);
    }

    // A chroma pass that compensates motion takes it from a luma guide, so only the one without
    // motion matches its pass engine.
    let [_, chroma_without_motion] = specs(geometry, false, true);
    assert_passes_through(&client, &chroma_without_motion, false);
}

#[test]
fn timing_lists_each_pass_the_engine_runs() {
    let client = client();
    let clip = clip();
    let geometry = geometry(ChromaLayout::Yuv420);
    let [both, _] = specs(geometry, true, true);
    let [luma_only, _] = specs(geometry, true, false);
    let [chroma_only, _] = specs(geometry, false, true);

    let mut engine = build(&client, &both).expect("build");
    engine.sample_device_time(1);
    drive(&mut engine, &client, &clip, 0, true);
    let timing = engine.timing();
    assert!(timing.luma.is_some());
    let both_chroma = timing.chroma.expect("chroma timing");

    let mut engine = build(&client, &luma_only).expect("build");
    let timing = engine.timing();
    assert!(timing.luma.is_some());
    assert!(timing.chroma.is_none());

    // The luma guide of a chroma only run cleans no plane, so its time goes to chroma.
    let mut engine = build(&client, &chroma_only).expect("build");
    engine.sample_device_time(1);
    drive(&mut engine, &client, &clip, 0, true);
    let timing = engine.timing();
    assert!(timing.luma.is_none());

    let guided_chroma = timing.chroma.expect("chroma timing");
    assert_eq!(guided_chroma.device_samples, both_chroma.device_samples);
    assert!(
        guided_chroma.device_time > both_chroma.device_time,
        "the guide's time is in chroma, got {guided_chroma:?} against {both_chroma:?}"
    );
}

#[test]
fn a_gray_engine_runs_the_luma_pass_alone() {
    let client = client();
    let clip = clip();
    let luma_frames = luma_of(&clip);
    let geometry = geometry(ChromaLayout::Gray);

    for spec in specs(geometry, true, false) {
        let (_, chroma_pass) = spec.passes();
        assert!(chroma_pass.is_none(), "{spec:?}");

        let mut luma_engine = seeded_pass(&client, &spec, true, &clip);
        let expected = drive(&mut luma_engine, &client, &luma_frames, 0, true);

        let mut engine = build(&client, &spec).expect("build");
        let merged = drive(&mut engine, &client, &luma_frames, 0, true);

        assert_eq!(merged, expected, "{spec:?}");
    }
}

#[test]
fn a_spec_without_passes_is_rejected() {
    let client = client();
    let geometry = geometry(ChromaLayout::Yuv420);

    for spec in specs(geometry, false, false) {
        let result = build(&client, &spec);
        assert!(matches!(result, Err(Error::InvalidOptions(_))), "{spec:?}");
    }
}

#[test]
fn a_chroma_pass_on_gray_is_rejected() {
    let client = client();
    let geometry = geometry(ChromaLayout::Gray);

    for spec in specs(geometry, true, true) {
        let result = build(&client, &spec);
        assert!(matches!(result, Err(Error::InvalidGeometry(_))), "{spec:?}");
    }
}

#[test]
fn a_chroma_plane_below_the_patch_size_is_rejected() {
    let client = client();
    let geometry = Geometry {
        width: 14,
        ..geometry(ChromaLayout::Yuv420)
    };
    let [spec, _] = specs(geometry, true, true);

    let (_, chroma_pass) = spec.passes();
    let chroma_pass = chroma_pass.expect("the spec cleans chroma");
    let kernels = EngineKernels::default();
    let pass_result = chroma_pass.build(&client, kernels);
    let Err(Error::InvalidGeometry(expected)) = pass_result else {
        panic!("the chroma pass engine accepted a 7 sample wide plane");
    };

    let result = build(&client, &spec);
    let Err(Error::InvalidGeometry(message)) = result else {
        panic!("the merged engine accepted a 7 sample wide chroma plane");
    };
    assert_eq!(message, expected);
}

/// One frame's planes, uploaded on request, remembering which planes each request asked for.
struct RecordingSource<'a> {
    client: &'a Client,
    frame: &'a Planes,
    requests: Vec<Range<usize>>,
}

impl PlaneSource for RecordingSource<'_> {
    fn upload(&mut self, planes: Range<usize>) -> Vec<Handle> {
        self.requests.push(planes.clone());

        self.frame[planes]
            .iter()
            .map(|plane| self.client.create_from_slice(plane))
            .collect()
    }
}

/// Pushes `frames` through [Engine::push_from], then the tail, returning the output and every
/// frame's upload requests.
fn drive_from(
    engine: &mut Box<dyn Engine>,
    client: &Client,
    frames: &[Planes],
) -> (Vec<Planes>, Vec<Vec<Range<usize>>>) {
    let dims = plane_dims(frames[0].len(), true);
    let mut outputs = Vec::new();
    let mut requests = Vec::new();

    for frame in frames {
        let mut source = RecordingSource {
            client,
            frame,
            requests: Vec::new(),
        };
        let ready = engine.push_from(&mut source).expect("push_from");
        requests.push(source.requests);
        emit(engine, client, &dims, ready, &mut outputs);
    }

    let tail = engine.finish().expect("finish");
    emit(engine, client, &dims, tail, &mut outputs);

    (outputs, requests)
}

#[test]
fn push_from_matches_push_and_uploads_luma_first_when_a_pass_needs_it() {
    let client = client();
    let clip = clip();
    let geometry = geometry(ChromaLayout::Yuv420);
    let cases = [(true, true), (true, false), (false, true)];

    for (luma, chroma) in cases {
        for spec in specs(geometry, luma, chroma) {
            let mut pushed = build(&client, &spec).expect("build");
            let expected = drive(&mut pushed, &client, &clip, 0, true);

            let mut staged = build(&client, &spec).expect("build");
            let (actual, requests) = drive_from(&mut staged, &client, &clip);

            assert_eq!(actual, expected, "{spec:?}");

            // An nl4d pass always tracks motion, so its camera model fit asks for luma first.
            let luma_first = luma || spec.is_nl4d();
            let order = if luma_first { [0..1, 1..3] } else { [1..3, 0..1] };
            for frame_requests in requests {
                assert_eq!(frame_requests, order, "{spec:?}");
            }
        }
    }
}

#[test]
fn push_from_rejects_a_plane_of_the_wrong_size_before_any_pass_moves() {
    let client = client();
    let geometry = geometry(ChromaLayout::Yuv420);
    let [spec, _] = specs(geometry, true, true);
    let mut engine = build(&client, &spec).expect("build");
    let short_luma = vec![0u8; 4];
    let frame = vec![short_luma, vec![0u8; 16], vec![0u8; 16]];
    let mut source = RecordingSource {
        client: &client,
        frame: &frame,
        requests: Vec::new(),
    };

    let result = engine.push_from(&mut source);

    assert!(matches!(result, Err(Error::PlaneMismatch(_))), "{result:?}");
    let luma_only = 0..1;
    assert_eq!(source.requests.len(), 1, "chroma is never asked for");
    assert_eq!(source.requests[0], luma_only);
}

/// Counts the warnings logged with one message.
struct WarningCounter {
    message: &'static str,
    count: Arc<AtomicUsize>,
}

/// Reads an event's message.
#[derive(Default)]
struct MessageVisitor {
    message: String,
}

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
        if field.name() == "message" {
            self.message = format!("{value:?}");
        }
    }
}

impl Subscriber for WarningCounter {
    fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
        true
    }

    fn new_span(&self, _span: &Attributes<'_>) -> Id {
        Id::from_u64(1)
    }

    fn record(&self, _span: &Id, _values: &Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event(&self, event: &Event<'_>) {
        if *event.metadata().level() != Level::WARN {
            return;
        }

        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        if visitor.message == self.message {
            self.count.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn enter(&self, _span: &Id) {}

    fn exit(&self, _span: &Id) {}
}

#[test]
fn only_gray_frames_have_a_stream_warning() {
    let gray = geometry(ChromaLayout::Gray);
    let yuv = geometry(ChromaLayout::Yuv420);

    assert_eq!(stream_warning(&gray), Some(GRAY_WARNING));
    assert_eq!(stream_warning(&yuv), None);
}

#[test]
fn a_gray_engine_warns_once_per_stream() {
    let client = client();
    let clip = clip();
    let luma_frames = luma_of(&clip);
    let geometry = geometry(ChromaLayout::Gray);
    let [spec, _] = specs(geometry, true, false);
    let count = Arc::new(AtomicUsize::new(0));
    let counter = WarningCounter {
        message: GRAY_WARNING,
        count: Arc::clone(&count),
    };

    tracing::subscriber::with_default(counter, || {
        let mut engine = build(&client, &spec).expect("build");
        let mut quiet = spec.build_quiet(&client, default_kernels).expect("build");
        assert_eq!(count.load(Ordering::SeqCst), 0, "building logs nothing");

        drive(&mut engine, &client, &luma_frames, 0, true);
        drive(&mut quiet, &client, &luma_frames, 0, true);
        assert_eq!(count.load(Ordering::SeqCst), 1, "one stream");

        drive(&mut engine, &client, &luma_frames, CONTEXT_FRAMES, true);
        assert_eq!(count.load(Ordering::SeqCst), 2, "a second stream after finish");

        let handles = upload(&client, &luma_frames[0]);
        let dims = plane_dims(1, true);
        let planes = as_planes(&handles, &dims);
        engine.push(&planes).expect("push");
        engine.reset();
        engine.push(&planes).expect("push");
        assert_eq!(count.load(Ordering::SeqCst), 4, "a stream each side of the reset");
    });
}

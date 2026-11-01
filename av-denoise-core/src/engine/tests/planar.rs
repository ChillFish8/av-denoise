use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::{
    ChromaLayout,
    DevicePlane,
    Engine,
    EngineSpec,
    Geometry,
    PassEngine,
    PassSpec,
    SampleFormat,
};
use crate::error::Error;
use crate::nl4d::Nl4dOptions;
use crate::nl4d::tests::helpers::noisy_frames;
use crate::nlmeans::{DenoisingMode, NlmeansAlgorithm, NlmeansHqOptions, NlmeansOptions};
use crate::tune::EngineKernels;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
/// Enough frames for a radius 1 engine to emit some frames from pushes through its noise look-ahead.
const FRAMES: usize = 14;
const CONTEXT_FRAMES: usize = 4;

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

fn default_kernels(_pass: &PassSpec) -> EngineKernels {
    EngineKernels::default()
}

fn build(client: &Client, spec: &EngineSpec) -> Result<Box<dyn Engine>, Error> {
    spec.build(client, default_kernels)
}

fn build_pass(client: &Client, pass: Option<PassSpec>) -> Box<dyn PassEngine> {
    let pass = pass.expect("the spec has this pass");
    let kernels = EngineKernels::default();

    pass.build(client, kernels).expect("build pass")
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
    let luma_frames = noisy_frames(WIDTH, HEIGHT, 1, FRAMES);
    let chroma_frames = noisy_frames(WIDTH / 2, HEIGHT / 2, 2, FRAMES);

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

impl Stream for Box<dyn PassEngine> {
    fn push(&mut self, planes: &[DevicePlane<'_>]) -> Result<usize, Error> {
        PassEngine::push(self.as_mut(), planes)
    }

    fn push_context(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error> {
        PassEngine::push_context(self.as_mut(), planes)
    }

    fn emit_into(&mut self, planes: &[DevicePlane<'_>]) -> Result<(), Error> {
        PassEngine::emit_into(self.as_mut(), planes)
    }

    fn finish(&mut self) -> Result<usize, Error> {
        PassEngine::finish(self.as_mut())
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
fn luma_and_chroma_match_the_two_pass_engines() {
    let client = client();
    let clip = clip();
    let geometry = geometry(ChromaLayout::Yuv420);

    for spec in specs(geometry, true, true) {
        let (luma_pass, chroma_pass) = spec.passes();
        let mut luma_engine = build_pass(&client, luma_pass);
        let mut chroma_engine = build_pass(&client, chroma_pass);
        let luma_frames = luma_of(&clip);
        let chroma_frames = chroma_of(&clip);
        let luma_expected = drive(&mut luma_engine, &client, &luma_frames, 0, true);
        let chroma_expected = drive(&mut chroma_engine, &client, &chroma_frames, 0, false);

        let mut engine = build(&client, &spec).expect("build");
        let merged = drive(&mut engine, &client, &clip, 0, true);

        assert_eq!(merged.len(), FRAMES, "{spec:?}");
        assert_eq!(luma_expected.len(), FRAMES, "{spec:?}");
        for (index, frame) in merged.iter().enumerate() {
            assert_eq!(frame[0], luma_expected[index][0], "{spec:?} frame {index} luma");
            assert_eq!(
                frame[1..],
                chroma_expected[index][..],
                "{spec:?} frame {index} chroma"
            );
        }
    }
}

/// Checks a merged engine cleaning one plane group matches its pass engine there, and that the
/// other group comes out as the source of the same frame.
///
/// It runs a context-led stream and then a scene-start stream on the same engines.
fn assert_passes_through(client: &Client, spec: &EngineSpec, clean_luma: bool) {
    let clip = clip();
    let (luma_pass, chroma_pass) = spec.passes();
    let (pass, cleaned_frames, luma_first) = if clean_luma {
        (luma_pass, luma_of(&clip), true)
    } else {
        (chroma_pass, chroma_of(&clip), false)
    };
    let mut pass_engine = build_pass(client, pass);
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

    for spec in specs(geometry, false, true) {
        assert_passes_through(&client, &spec, false);
    }
}

#[test]
fn timing_lists_each_pass_the_engine_runs() {
    let client = client();
    let clip = clip();
    let geometry = geometry(ChromaLayout::Yuv420);
    let [both, _] = specs(geometry, true, true);
    let [luma_only, _] = specs(geometry, true, false);

    let mut engine = build(&client, &both).expect("build");
    engine.sample_device_time(1);
    drive(&mut engine, &client, &clip, 0, true);
    let timing = engine.timing();
    assert!(timing.luma.is_some());
    assert!(timing.chroma.is_some());

    let mut engine = build(&client, &luma_only).expect("build");
    let timing = engine.timing();
    assert!(timing.luma.is_some());
    assert!(timing.chroma.is_none());
}

#[test]
fn a_gray_engine_runs_the_luma_pass_alone() {
    let client = client();
    let clip = clip();
    let luma_frames = luma_of(&clip);
    let geometry = geometry(ChromaLayout::Gray);

    for spec in specs(geometry, true, false) {
        let (luma_pass, chroma_pass) = spec.passes();
        assert!(chroma_pass.is_none(), "{spec:?}");

        let mut luma_engine = build_pass(&client, luma_pass);
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

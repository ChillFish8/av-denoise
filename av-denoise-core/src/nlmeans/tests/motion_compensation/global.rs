use cubecl::prelude::*;

use crate::bench_api::HostIo;
use crate::engine::{
    ChromaLayout,
    ChromaSiting,
    DevicePlane,
    Engine,
    EngineSpec,
    Geometry,
    PassSpec,
    SampleFormat,
};
use crate::motion::{MotionCompensationMode, MotionEstimation};
use crate::nl4d::Nl4dOptions;
use crate::nlmeans::denoiser::LOOKAHEAD_FRAMES;
use crate::nlmeans::tests::helpers::make_client;
use crate::nlmeans::tests::models::{
    ModelFitter,
    NeighbourFit,
    PanFrame,
    PanShape,
    assert_follows,
    assert_premise_misses,
    matched_flags,
    mean_over_blocks,
    pan_clip,
    stays_inside,
};
use crate::nlmeans::*;
use crate::tune::EngineKernels;
use crate::tune::gpu::{read_f32, read_i32};
use crate::tune::tests::fixtures::upload;

const WIDTH: u32 = 384;
const HEIGHT: u32 = 192;
/// Past the 12 px an unseeded search reaches at the default block search.
const FAST_PAN: u32 = 30;
const SIGMA: f32 = 2.0 / 255.0;
const RADIUS: u32 = 2;
/// Twelve frames leave the noise look-ahead while the stream is pushed.
const FRAMES: usize = 12 + LOOKAHEAD_FRAMES as usize;
/// Pixels a scored block keeps from the plane's edge after moving.
const MARGIN: u32 = 4;

fn mode(estimation: MotionEstimation) -> MotionCompensationMode {
    MotionCompensationMode::Mvtools {
        blksize: 16,
        overlap: 8,
        search_radius: 4,
        max_search_escalation: 4,
        pyramid_levels: 2,
        estimation,
    }
}

fn hq_params(estimation: MotionEstimation) -> NlmParams {
    NlmParams {
        temporal_radius: RADIUS,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: mode(estimation),
        hq: Some(HqParams::default()),
    }
}

fn luma_shape(shift: (u32, u32)) -> PanShape {
    PanShape {
        width: WIDTH,
        height: HEIGHT,
        chroma_scale: (2, 2),
        shift,
        sigma: SIGMA,
    }
}

/// A luma front fed every frame of `frames`, seeded from luma camera models when `seeded`.
///
/// Returns the front after its last submit and every frame it put out, the flushed tail included
/// when `flush`.
fn run_front(
    params: NlmParams,
    shape: PanShape,
    frames: &[PanFrame],
    seeded: bool,
    flush: bool,
) -> (NlmDenoiser, Vec<Vec<f32>>) {
    let client = make_client();
    let mode = params.motion_compensation;
    let mut fitter = ModelFitter::new(&client, shape.width, shape.height, mode);
    let mut denoiser = NlmDenoiser::new(&client, params, shape.width, shape.height);
    if seeded {
        let motion = fitter.motion();
        denoiser.set_shared_motion(motion);
    }

    let mut outputs = Vec::new();
    for frame in frames {
        fitter.fit(&frame.luma);
        denoiser.push_frame(&frame.luma);

        let output = denoiser.denoise().expect("denoise failed");
        outputs.extend(output);
    }

    if flush {
        denoiser
            .flush(|frame| outputs.push(frame.to_vec()))
            .expect("flush failed");
    }

    (denoiser, outputs)
}

/// The last pass's fit to each neighbour, over the blocks that stay inside both frames.
///
/// Neighbours come oldest first, and the content moves `shift` pixels left and up per frame.
fn neighbour_fits(denoiser: &NlmDenoiser, shift: (u32, u32)) -> Vec<NeighbourFit> {
    let motion_ctx = denoiser.motion_ctx();
    let blocks_x = motion_ctx.blocks_x;
    let blocks_y = motion_ctx.blocks_y;
    let blocks = (blocks_x * blocks_y) as usize;
    let conf_stride = motion_ctx.conf_stride() as usize;
    let mv_stride = motion_ctx.mv_stride() as usize;
    let blksize = motion_ctx.blksize;
    let step = motion_ctx.step;
    let (width, height, _) = denoiser.frame_shape();
    let neighbours = 2 * RADIUS as usize;

    let confidence_buf = denoiser
        .confidence_buf
        .as_ref()
        .expect("HQ temporal confidence is on");
    let mv_field_buf = denoiser.mv_field_buf.as_ref().expect("motion compensation is on");
    let confidence_words = neighbours * conf_stride;
    let mv_words = neighbours * mv_stride;
    let confidence =
        read_f32(&denoiser.client, confidence_buf, confidence_words).expect("confidence reads back");
    let vectors = read_i32(&denoiser.client, mv_field_buf, mv_words).expect("the field reads back");
    let offsets = (-(RADIUS as i32)..0).chain(1..=RADIUS as i32);

    offsets
        .enumerate()
        .map(|(neighbour, offset)| {
            let shift_x = -(shift.0 as i32) * offset;
            let shift_y = -(shift.1 as i32) * offset;
            let conf_start = neighbour * conf_stride;
            let mv_start = neighbour * mv_stride;
            let block_confidence = &confidence[conf_start..conf_start + blocks];
            let block_words = &vectors[mv_start..mv_start + 2 * blocks];
            let (block_vectors, _) = block_words.as_chunks::<2>();
            let matched = matched_flags(block_vectors, [shift_x, shift_y]);
            let inside = |col: u32, row: u32| {
                let inside_x = stays_inside(col * step, blksize, shift_x, width, MARGIN);
                let inside_y = stays_inside(row * step, blksize, shift_y, height, MARGIN);
                inside_x && inside_y
            };

            let confidence = mean_over_blocks(block_confidence, blocks_x, blocks_y, inside);
            let matched = mean_over_blocks(&matched, blocks_x, blocks_y, inside);

            NeighbourFit { confidence, matched }
        })
        .collect()
}

fn assert_tracks_fast_pan(estimation: MotionEstimation, label: &str) {
    let shape = luma_shape((FAST_PAN, 0));
    let frames = pan_clip(shape, FRAMES, 11);
    let params = hq_params(estimation);

    let (unseeded, _) = run_front(params.clone(), shape, &frames, false, false);
    let unseeded_fits = neighbour_fits(&unseeded, shape.shift);
    assert_premise_misses(&unseeded_fits, label);

    let (seeded, _) = run_front(params, shape, &frames, true, false);
    let seeded_fits = neighbour_fits(&seeded, shape.shift);
    assert_follows(&seeded_fits, label);
}

#[test]
fn nlmeans_direct_matches_a_pan_beyond_reach() {
    assert_tracks_fast_pan(MotionEstimation::Direct, "direct");
}

#[test]
fn chained_matches_a_pan_beyond_reach() {
    let chained = MotionEstimation::chained_default();

    assert_tracks_fast_pan(chained, "chained");
}

fn bits(frames: &[Vec<f32>]) -> Vec<Vec<u32>> {
    frames
        .iter()
        .map(|frame| frame.iter().map(|value| value.to_bits()).collect())
        .collect()
}

#[test]
fn a_static_stream_is_bit_identical_to_unseeded() {
    let shape = luma_shape((0, 0));
    let frames = pan_clip(shape, FRAMES, 12);
    let chained = MotionEstimation::chained_default();

    for estimation in [MotionEstimation::Direct, chained] {
        let params = hq_params(estimation);
        let (_, unseeded) = run_front(params.clone(), shape, &frames, false, true);
        let (_, seeded) = run_front(params, shape, &frames, true, true);

        assert_eq!(seeded.len(), FRAMES, "{estimation:?}: every frame comes out");
        assert!(
            bits(&seeded) == bits(&unseeded),
            "{estimation:?}: a static stream must match the unseeded output bit for bit"
        );
    }
}

fn gray_geometry() -> Geometry {
    Geometry {
        width: WIDTH,
        height: HEIGHT,
        chroma: ChromaLayout::Gray,
        input: SampleFormat::F32,
        output: SampleFormat::F32,
        chroma_siting: ChromaSiting::Unknown,
    }
}

fn default_kernels(_pass: &PassSpec) -> EngineKernels {
    EngineKernels::default()
}

/// Pushes every frame's luma through `engine` and returns what it puts out, the tail included
/// when `finish`.
fn drive(client: &Client, engine: &mut Box<dyn Engine>, frames: &[PanFrame], finish: bool) -> Vec<Vec<f32>> {
    let pixels = (WIDTH * HEIGHT) as usize;
    let output = client.empty(pixels * size_of::<f32>());
    let output_planes = [DevicePlane::new(&output, WIDTH, HEIGHT)];
    let mut outputs = Vec::new();
    let mut emit = |engine: &mut Box<dyn Engine>| {
        engine.emit_into(&output_planes).expect("emit failed");
        let frame = read_f32(client, &output, pixels).expect("output reads back");
        outputs.push(frame);
    };

    for frame in frames {
        let handle = upload(client, &frame.luma);
        let planes = [DevicePlane::new(&handle, WIDTH, HEIGHT)];
        let ready = engine.push(&planes).expect("push failed");
        for _ in 0..ready {
            emit(engine);
        }
    }

    if finish {
        let tail = engine.finish().expect("finish failed");
        for _ in 0..tail {
            emit(engine);
        }
    }

    outputs
}

#[test]
fn models_do_not_leak_across_reset() {
    let client = make_client();
    let geometry = gray_geometry();
    let hq_options = NlmeansHqOptions {
        nlm: NlmeansOptions {
            motion_compensation: mode(MotionEstimation::Direct),
            mode: DenoisingMode::Temporal { radius: RADIUS },
            ..NlmeansOptions::default()
        },
        hq: HqParams::default(),
    };
    let nlmeans = EngineSpec::Nlmeans {
        luma: Some(NlmeansAlgorithm::Hq(hq_options)),
        chroma: None,
        geometry,
    };
    let nl4d = EngineSpec::Nl4d {
        luma: Some(Nl4dOptions::default()),
        chroma: None,
        geometry,
    };

    let before_shape = luma_shape((0, 24));
    let after_shape = luma_shape((FAST_PAN, 0));
    let before = pan_clip(before_shape, FRAMES, 14);
    let after = pan_clip(after_shape, FRAMES, 15);

    for spec in [nlmeans, nl4d] {
        let mut reset_engine = spec.build(&client, default_kernels).expect("the spec builds");
        drive(&client, &mut reset_engine, &before, false);
        reset_engine.reset();
        let after_reset = drive(&client, &mut reset_engine, &after, true);

        let mut fresh_engine = spec.build(&client, default_kernels).expect("the spec builds");
        let fresh = drive(&client, &mut fresh_engine, &after, true);

        assert_eq!(after_reset.len(), FRAMES, "{spec:?}: every frame comes out");
        assert!(
            bits(&after_reset) == bits(&fresh),
            "{spec:?}: the stream after a reset must match a fresh engine bit for bit"
        );
    }
}

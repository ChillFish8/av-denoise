use super::helpers::make_client;
use crate::bench_api::HostIo;
use crate::nl4d::denoiser::CompletedRegion;
use crate::nl4d::{MotionSnapshot, Nl4dDenoiser, Nl4dParams};
use crate::nlmeans::denoiser::LOOKAHEAD_FRAMES;
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
use crate::tune::gpu::read_f32;

const WIDTH: u32 = 384;
const HEIGHT: u32 = 192;
/// Past the 12 px an unseeded search reaches at the default block search.
const FAST_PAN: u32 = 30;
const SIGMA: f32 = 2.0 / 255.0;
/// Twelve frames leave the noise look-ahead while the stream is pushed.
const FRAMES: usize = 12 + LOOKAHEAD_FRAMES as usize;
/// Pixels a scored block keeps from the plane's edge after moving.
const MARGIN: u32 = 4;

fn shape(shift: u32) -> PanShape {
    PanShape {
        width: WIDTH,
        height: HEIGHT,
        chroma_scale: (2, 2),
        shift: (shift, 0),
        sigma: SIGMA,
    }
}

/// A denoiser fed every frame of `frames`, seeded from the camera models when `seeded`.
///
/// Returns the denoiser after its last submit and every frame it put out.
fn run_stream(frames: &[PanFrame], seeded: bool) -> (Nl4dDenoiser, Vec<Vec<f32>>) {
    let client = make_client();
    let params = Nl4dParams::default();
    let mode = params.nlm.motion_compensation;
    let mut fitter = ModelFitter::new(&client, WIDTH, HEIGHT, mode);
    let mut denoiser = Nl4dDenoiser::new(&client, params, WIDTH, HEIGHT).expect("construction failed");
    if seeded {
        let motion = fitter.motion(1.0, 1.0);
        denoiser.set_shared_motion(motion);
    }

    let mut outputs = Vec::new();
    for frame in frames {
        fitter.fit(&frame.luma);
        denoiser.push_frame(&frame.luma);

        let output = denoiser.denoise().expect("denoise failed");
        outputs.extend(output);
    }

    (denoiser, outputs)
}

/// Drains the look-ahead and runs the tail passes, keeping the stream so the last pass, centred on
/// the window's newest frame, can be inspected.
fn run_tail(denoiser: &mut Nl4dDenoiser, outputs: &mut Vec<Vec<f32>>) {
    while let Some(region) = denoiser.drain_lookahead().expect("drain failed") {
        let frame = read_region(denoiser, region);
        outputs.push(frame);
    }

    let regions = denoiser.finish_passes().expect("tail passes failed");
    for region in regions {
        let frame = read_region(denoiser, region);
        outputs.push(frame);
    }
}

fn read_region(denoiser: &mut Nl4dDenoiser, region: CompletedRegion) -> Vec<f32> {
    let handle = denoiser.read_region(region);
    let pixels = (WIDTH * HEIGHT) as usize;
    let client = denoiser.compute_client();

    read_f32(client, &handle, pixels).expect("the region reads back")
}

/// Each neighbour's fit to the pan over the blocks that stay inside both frames.
///
/// `offsets` holds each neighbour's distance from the pass centre, and the content moves
/// [FAST_PAN] pixels left per frame.
fn neighbour_fits(snapshot: &MotionSnapshot, offsets: &[i32]) -> Vec<NeighbourFit> {
    let fields = snapshot.vectors.iter().zip(&snapshot.confidence);

    offsets
        .iter()
        .zip(fields)
        .map(|(&offset, (vectors, confidence))| {
            let shift_x = -(FAST_PAN as i32) * offset;
            let matched = matched_flags(vectors, [shift_x, 0]);
            let inside = |col: u32, row: u32| {
                let origin_x = col * snapshot.step;
                let origin_y = row * snapshot.step;
                let inside_x = stays_inside(origin_x, snapshot.blksize, shift_x, WIDTH, MARGIN);
                let inside_y = stays_inside(origin_y, snapshot.blksize, 0, HEIGHT, MARGIN);
                inside_x && inside_y
            };

            let confidence = mean_over_blocks(confidence, snapshot.blocks_x, snapshot.blocks_y, inside);
            let matched = mean_over_blocks(&matched, snapshot.blocks_x, snapshot.blocks_y, inside);

            NeighbourFit { confidence, matched }
        })
        .collect()
}

/// The fits of the centred pass and of the last tail pass, whose neighbours are all behind.
fn pass_fits(frames: &[PanFrame], seeded: bool) -> (Vec<NeighbourFit>, Vec<NeighbourFit>) {
    let (mut denoiser, mut outputs) = run_stream(frames, seeded);
    let radius = Nl4dParams::default().temporal_radius as i32;

    let centred = denoiser.motion_snapshot().expect("a pass ran");
    let centred_fits = neighbour_fits(&centred, &centred.offsets);

    run_tail(&mut denoiser, &mut outputs);
    let tail = denoiser.motion_snapshot().expect("a tail pass ran");
    let tail_offsets: Vec<i32> = (-2 * radius..0).collect();
    let tail_fits = neighbour_fits(&tail, &tail_offsets);

    (centred_fits, tail_fits)
}

#[test]
fn nl4d_direct_matches_a_pan_beyond_reach() {
    let frames = pan_clip(shape(FAST_PAN), FRAMES, 21);

    let (unseeded_centred, _) = pass_fits(&frames, false);
    assert_premise_misses(&unseeded_centred, "nl4d");

    let (centred, tail) = pass_fits(&frames, true);
    assert_follows(&centred, "centred pass");
    assert_follows(&tail, "tail pass");
}

fn bits(frames: &[Vec<f32>]) -> Vec<Vec<u32>> {
    frames
        .iter()
        .map(|frame| frame.iter().map(|value| value.to_bits()).collect())
        .collect()
}

#[test]
fn a_static_nl4d_stream_is_bit_identical_to_unseeded() {
    let frames = pan_clip(shape(0), FRAMES, 22);

    let (mut unseeded_denoiser, mut unseeded) = run_stream(&frames, false);
    run_tail(&mut unseeded_denoiser, &mut unseeded);
    let (mut seeded_denoiser, mut seeded) = run_stream(&frames, true);
    run_tail(&mut seeded_denoiser, &mut seeded);

    assert_eq!(seeded.len(), FRAMES, "every frame comes out");
    assert!(
        bits(&seeded) == bits(&unseeded),
        "a static stream must match the unseeded output bit for bit"
    );
}

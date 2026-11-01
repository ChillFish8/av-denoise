use crate::bench_api::HostIo;
use crate::motion::{MotionCompensationMode, MotionCtx, MotionEstimation, MotionSearch};
use crate::nlmeans::denoiser::LOOKAHEAD_FRAMES;
use crate::nlmeans::tests::helpers::{correlated_noisy_frame, make_client, noisy_field_over};
use crate::nlmeans::tests::models::{PanShape, matched_flags, mean_over_blocks, pan_clip, stays_inside};
use crate::nlmeans::*;
use crate::tune::gpu::read_i32;

const WIDTH: u32 = 384;
const HEIGHT: u32 = 192;
const SIGMA: f32 = 2.0 / 255.0;
const RADIUS: u32 = 2;
/// Twelve frames leave the noise look-ahead while the stream is pushed.
const FRAMES: usize = 12 + LOOKAHEAD_FRAMES as usize;
/// Pixels a scored block keeps from the plane's edge after moving.
const MARGIN: u32 = 4;
/// A pan whose two-frame neighbours lie past a radius 2 search's 6 px reach and inside radius 4's.
const PAN: u32 = 4;
/// The sigma of the textures the moving-object clip draws.
const TEXTURE_SIGMA: f32 = 24.0 / 255.0;
/// The side of the moving object, in pixels.
const OBJECT_SIDE: u32 = 96;
/// How far the object moves right each frame, past a radius 2 search's 6 px reach.
const OBJECT_SPEED: u32 = 7;
/// The share of object blocks a search that follows the object matches, and one that misses it
/// stays below.
const FOLLOWED: f32 = 0.9;
const MISSED: f32 = 0.5;

fn mode(
    search_radius: u32,
    max_search_escalation: u32,
    estimation: MotionEstimation,
) -> MotionCompensationMode {
    MotionCompensationMode::Mvtools {
        blksize: 16,
        overlap: 8,
        search_radius,
        max_search_escalation,
        pyramid_levels: 2,
        estimation,
    }
}

fn hq_params(motion_compensation: MotionCompensationMode) -> NlmParams {
    NlmParams {
        temporal_radius: RADIUS,
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

/// A front searching with `motion_compensation`, fed every frame of `frames` and returned after its
/// last submit.
fn run_front(motion_compensation: MotionCompensationMode, frames: &[Vec<f32>]) -> NlmDenoiser {
    let client = make_client();
    let params = hq_params(motion_compensation);
    let mut denoiser = NlmDenoiser::new(&client, params, WIDTH, HEIGHT);

    for frame in frames {
        denoiser.push_frame(frame);
        denoiser.denoise().expect("denoise failed");
    }

    denoiser
}

/// The last pass's vectors, one list of `[dx, dy]` per neighbour, oldest neighbour first.
fn neighbour_vectors(denoiser: &NlmDenoiser) -> Vec<Vec<[i32; 2]>> {
    let motion_ctx = denoiser.motion_ctx();
    let blocks = (motion_ctx.blocks_x * motion_ctx.blocks_y) as usize;
    let mv_stride = motion_ctx.mv_stride() as usize;
    let neighbours = 2 * RADIUS as usize;
    let mv_field_buf = denoiser.mv_field_buf.as_ref().expect("motion compensation is on");
    let words = neighbours * mv_stride;
    let vectors = read_i32(&denoiser.client, mv_field_buf, words).expect("the field reads back");

    (0..neighbours)
        .map(|neighbour| {
            let start = neighbour * mv_stride;
            let block_words = &vectors[start..start + 2 * blocks];
            let (block_vectors, _) = block_words.as_chunks::<2>();
            block_vectors.to_vec()
        })
        .collect()
}

/// The temporal offset of every neighbour, oldest first.
fn neighbour_offsets() -> Vec<i32> {
    (-(RADIUS as i32)..0).chain(1..=RADIUS as i32).collect()
}

fn pan_frames() -> Vec<Vec<f32>> {
    let shape = PanShape {
        width: WIDTH,
        height: HEIGHT,
        chroma_scale: (2, 2),
        shift: (PAN, 0),
        sigma: SIGMA,
    };
    let frames = pan_clip(shape, FRAMES, 21);

    frames.into_iter().map(|frame| frame.luma).collect()
}

/// Whether the block at `col` and `row` stays inside both frames under a pan of `shift_x`.
fn inside_pan(motion_ctx: &MotionCtx, col: u32, row: u32, shift_x: i32) -> bool {
    let blksize = motion_ctx.blksize;
    let step = motion_ctx.step;
    let inside_x = stays_inside(col * step, blksize, shift_x, WIDTH, MARGIN);
    let inside_y = stays_inside(row * step, blksize, 0, HEIGHT, MARGIN);

    inside_x && inside_y
}

/// Each neighbour's share of blocks that follow the pan, over the blocks inside both frames.
fn pan_matched(denoiser: &NlmDenoiser) -> Vec<f32> {
    let motion_ctx = denoiser.motion_ctx();
    let blocks_x = motion_ctx.blocks_x;
    let blocks_y = motion_ctx.blocks_y;
    let vectors = neighbour_vectors(denoiser);

    vectors
        .iter()
        .zip(neighbour_offsets())
        .map(|(block_vectors, offset)| {
            let shift_x = -(PAN as i32) * offset;
            let matched = matched_flags(block_vectors, [shift_x, 0]);
            let inside = |col: u32, row: u32| inside_pan(motion_ctx, col, row, shift_x);

            mean_over_blocks(&matched, blocks_x, blocks_y, inside)
        })
        .collect()
}

/// Each neighbour's vectors at the blocks inside both frames.
fn inside_pan_vectors(denoiser: &NlmDenoiser) -> Vec<Vec<[i32; 2]>> {
    let motion_ctx = denoiser.motion_ctx();
    let blocks_x = motion_ctx.blocks_x;
    let vectors = neighbour_vectors(denoiser);

    vectors
        .iter()
        .zip(neighbour_offsets())
        .map(|(block_vectors, offset)| {
            let shift_x = -(PAN as i32) * offset;
            let indexed = block_vectors.iter().enumerate();
            indexed
                .filter(|(block, _)| {
                    let col = *block as u32 % blocks_x;
                    let row = *block as u32 / blocks_x;
                    inside_pan(motion_ctx, col, row, shift_x)
                })
                .map(|(_, vector)| *vector)
                .collect()
        })
        .collect()
}

#[test]
fn a_small_first_window_with_escalation_matches_a_full_window_on_a_pan() {
    let frames = pan_frames();
    let direct = MotionEstimation::Direct;

    let small_mode = mode(2, 2, direct);
    let small = run_front(small_mode, &frames);
    let small_matched = pan_matched(&small);
    let misses = small_matched.iter().any(|&matched| matched < MISSED);
    assert!(
        misses,
        "test premise: a radius 2 search misses the far neighbours, got {small_matched:?}"
    );

    let full_mode = mode(4, 4, direct);
    let full = run_front(full_mode, &frames);

    let adaptive_mode = mode(2, 4, direct);
    let adaptive = run_front(adaptive_mode, &frames);

    let full_matched = pan_matched(&full);
    let adaptive_matched = pan_matched(&adaptive);

    for (neighbour, &matched) in full_matched.iter().enumerate() {
        assert!(
            matched > FOLLOWED,
            "test premise: radius 4 follows neighbour {neighbour}, got {full_matched:?}"
        );
    }

    assert_eq!(
        inside_pan_vectors(&adaptive),
        inside_pan_vectors(&full),
        "escalating from radius 2 to 4 must find the vectors radius 4 finds, matched shares \
         {adaptive_matched:?} against {full_matched:?}"
    );
}

#[test]
fn defaults_keep_every_interior_vector_of_a_pan_in_reach() {
    let frames = pan_frames();
    let direct = MotionEstimation::Direct;
    let defaults = MotionSearch {
        estimation: direct,
        ..MotionSearch::default()
    };
    let fixed_mode = mode(defaults.search_radius, defaults.search_radius, direct);
    let escalating_mode = MotionCompensationMode::from(defaults);
    let escalating = run_front(escalating_mode, &frames);
    let fixed = run_front(fixed_mode, &frames);
    let escalating_vectors = inside_pan_vectors(&escalating);
    let fixed_vectors = inside_pan_vectors(&fixed);

    assert_eq!(
        escalating_vectors, fixed_vectors,
        "escalation must leave every interior vector of a pan in reach where the fixed search put it"
    );
}

/// A static textured background with a textured square moving [OBJECT_SPEED] pixels right per
/// frame, both under fresh grain each frame.
fn object_frames() -> Vec<Vec<f32>> {
    let texture_sigma_pre = TEXTURE_SIGMA / 0.375f32.sqrt();
    let background = correlated_noisy_frame(WIDTH, HEIGHT, 0.5, texture_sigma_pre, 31);
    let object = correlated_noisy_frame(OBJECT_SIDE, OBJECT_SIDE, 0.5, texture_sigma_pre, 32);
    let object_top = (HEIGHT - OBJECT_SIDE) / 2;

    (0..FRAMES as u32)
        .map(|index| {
            let object_left = 32 + OBJECT_SPEED * index;
            let mut clean = background.clone();
            for row in 0..OBJECT_SIDE {
                for col in 0..OBJECT_SIDE {
                    let x = (object_left + col) % WIDTH;
                    let y = object_top + row;
                    clean[(y * WIDTH + x) as usize] = object[(row * OBJECT_SIDE + col) as usize];
                }
            }

            noisy_field_over(&clean, WIDTH, HEIGHT, SIGMA, 33 + index)
        })
        .collect()
}

/// The stream frame the last pass of a [run_front] centres on.
///
/// The look-ahead holds the newest frames back, and the window reaches [RADIUS] frames past its
/// centre.
fn last_centre_frame() -> u32 {
    let newest = FRAMES as u32 - 1;
    newest - LOOKAHEAD_FRAMES - RADIUS
}

/// Each neighbour's share of blocks inside the centre frame's object that follow the object.
fn object_matched(denoiser: &NlmDenoiser) -> Vec<f32> {
    let motion_ctx = denoiser.motion_ctx();
    let blocks_x = motion_ctx.blocks_x;
    let blocks_y = motion_ctx.blocks_y;
    let blksize = motion_ctx.blksize;
    let step = motion_ctx.step;
    let vectors = neighbour_vectors(denoiser);
    let object_left = 32 + OBJECT_SPEED * last_centre_frame();
    let object_top = (HEIGHT - OBJECT_SIDE) / 2;
    let in_object = |col: u32, row: u32| {
        let left = col * step;
        let top = row * step;
        let in_object_x = left >= object_left && left + blksize <= object_left + OBJECT_SIDE;
        let in_object_y = top >= object_top && top + blksize <= object_top + OBJECT_SIDE;
        in_object_x && in_object_y
    };

    vectors
        .iter()
        .zip(neighbour_offsets())
        .map(|(block_vectors, offset)| {
            let shift_x = OBJECT_SPEED as i32 * offset;
            let matched = matched_flags(block_vectors, [shift_x, 0]);

            mean_over_blocks(&matched, blocks_x, blocks_y, in_object)
        })
        .collect()
}

#[test]
fn escalation_rescues_a_fast_local_object() {
    let frames = object_frames();
    let direct = MotionEstimation::Direct;

    let fixed_mode = mode(2, 2, direct);
    let fixed = run_front(fixed_mode, &frames);
    let fixed_matched = object_matched(&fixed);
    let misses = fixed_matched.iter().all(|&matched| matched < MISSED);
    assert!(
        misses,
        "test premise: a radius 2 search misses the object, got {fixed_matched:?}"
    );

    let escalated_mode = mode(2, 8, direct);
    let escalated = run_front(escalated_mode, &frames);
    let escalated_matched = object_matched(&escalated);
    for (neighbour, &matched) in escalated_matched.iter().enumerate() {
        assert!(
            matched > FOLLOWED,
            "escalation must find the object for neighbour {neighbour}, got {escalated_matched:?}"
        );
    }
}

#[test]
fn chained_escalation_rescues_a_fast_local_object() {
    let frames = object_frames();
    let chained = MotionEstimation::chained_default();

    let fixed_mode = mode(2, 2, chained);
    let fixed = run_front(fixed_mode, &frames);
    let fixed_matched = object_matched(&fixed);
    let misses = fixed_matched.iter().all(|&matched| matched < MISSED);
    assert!(
        misses,
        "test premise: chained radius 2 searches miss the object, got {fixed_matched:?}"
    );

    let escalated_mode = mode(2, 8, chained);
    let escalated = run_front(escalated_mode, &frames);
    let escalated_matched = object_matched(&escalated);
    for (neighbour, &matched) in escalated_matched.iter().enumerate() {
        assert!(
            matched > FOLLOWED,
            "escalation must find the object for neighbour {neighbour}, got {escalated_matched:?}"
        );
    }
}

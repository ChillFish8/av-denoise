use cubecl::prelude::*;
use cubecl::terminate;

use crate::motion::global::{
    AFFINE_WORDS,
    FIT_INLIER_CAP,
    FIT_LANES,
    FIT_STAGE_GATE,
    FIT_STAGE_POLISH,
    FIT_STAGE_SOLVE,
    FIT_STAGE_WEIGH,
    FIT_THREADS,
    MAX_ROLL_TAN,
    MODEL_WORDS,
    NO_PAIR,
};

/// Composes the camera model from the centre frame to each neighbour, in the pass's own pixels.
///
/// One thread handles one neighbour, in motion-field order with past neighbours first. Its
/// `max_steps` gap entries name the model-ring slots between the centre and that neighbour,
/// listed from the centre outward. A future neighbour applies them in order. A past neighbour
/// multiplies them into the motion from the neighbour to the centre and writes its inverse.
/// A `NO_PAIR` entry, or a model whose `valid` word is zero, counts as no motion.
///
/// Models are in luma pixels and each output is `m00, m01, m10, m11, tx, ty` with
/// `x_luma = x_pass * scale` per axis.
#[cube(launch_unchecked)]
pub fn mc_compose_models(
    models: &[f32],
    gaps: &[u32],
    out: &mut [f32],
    #[comptime] neighbours: u32,
    #[comptime] max_steps: u32,
    scale_x: f32,
    scale_y: f32,
) {
    let neighbour = ABSOLUTE_POS_X;
    if neighbour >= neighbours {
        terminate!();
    }

    let past = neighbour < neighbours / 2;
    let mut acc_a = 1.0f32;
    let mut acc_b = 0.0f32;
    let mut acc_tx = 0.0f32;
    let mut acc_ty = 0.0f32;

    for step in 0..max_steps {
        let gap = gaps[(neighbour * max_steps + step) as usize];
        let mut pair_a = 1.0f32;
        let mut pair_b = 0.0f32;
        let mut pair_tx = 0.0f32;
        let mut pair_ty = 0.0f32;

        if gap != NO_PAIR {
            let base = gap * MODEL_WORDS;
            let valid = models[(base + 4) as usize];
            if valid != 0.0 {
                pair_a = models[base as usize];
                pair_b = models[(base + 1) as usize];
                pair_tx = models[(base + 2) as usize];
                pair_ty = models[(base + 3) as usize];
            }
        }

        // Future steps apply after the motion so far, past steps apply before it.
        let mut outer_a = pair_a;
        let mut outer_b = pair_b;
        let mut outer_tx = pair_tx;
        let mut outer_ty = pair_ty;
        let mut inner_a = acc_a;
        let mut inner_b = acc_b;
        let mut inner_tx = acc_tx;
        let mut inner_ty = acc_ty;

        if past {
            outer_a = acc_a;
            outer_b = acc_b;
            outer_tx = acc_tx;
            outer_ty = acc_ty;
            inner_a = pair_a;
            inner_b = pair_b;
            inner_tx = pair_tx;
            inner_ty = pair_ty;
        }

        acc_a = outer_a * inner_a - outer_b * inner_b;
        acc_b = outer_a * inner_b + outer_b * inner_a;
        acc_tx = outer_a * inner_tx - outer_b * inner_ty + outer_tx;
        acc_ty = outer_b * inner_tx + outer_a * inner_ty + outer_ty;
    }

    if past {
        let norm = acc_a * acc_a + acc_b * acc_b;
        let mut inverse_a = 1.0f32;
        let mut inverse_b = 0.0f32;
        let mut inverse_tx = 0.0f32;
        let mut inverse_ty = 0.0f32;

        if norm > 0.0 {
            let reciprocal = 1.0f32 / norm;
            inverse_a = acc_a * reciprocal;
            inverse_b = -acc_b * reciprocal;
            inverse_tx = -(inverse_a * acc_tx - inverse_b * acc_ty);
            inverse_ty = -(inverse_b * acc_tx + inverse_a * acc_ty);
        }

        acc_a = inverse_a;
        acc_b = inverse_b;
        acc_tx = inverse_tx;
        acc_ty = inverse_ty;
    }

    let base = (neighbour * AFFINE_WORDS) as usize;
    out[base] = acc_a;
    out[base + 1] = -acc_b * scale_y / scale_x;
    out[base + 2] = acc_b * scale_x / scale_y;
    out[base + 3] = acc_a;
    out[base + 4] = acc_tx / scale_x;
    out[base + 5] = acc_ty / scale_y;
}

/// Halving rounds that reduce [FIT_THREADS] partial sums to one.
const FIT_ROUNDS: u32 = 8;

const _: () = assert!(1 << FIT_ROUNDS == FIT_THREADS);

/// Weigh lane holding the base weight.
const LANE_BASE: u32 = 0;

/// Weigh lane holding the base-weighted SAD.
const LANE_BASE_SAD: u32 = 1;

/// Solve lane holding the weight.
const LANE_WEIGHT: u32 = 0;

/// Solve lane holding the weighted point's x.
const LANE_POINT_X: u32 = 1;

/// Solve lane holding the weighted point's y.
const LANE_POINT_Y: u32 = 2;

/// Solve lane holding the weighted motion's x.
const LANE_MOTION_X: u32 = 3;

/// Solve lane holding the weighted motion's y.
const LANE_MOTION_Y: u32 = 4;

/// Solve lane holding the weighted dot product of point and motion.
const LANE_DOT: u32 = 5;

/// Solve lane holding the weighted cross product of point and motion.
const LANE_CROSS: u32 = 6;

/// Solve lane holding the weighted squared distance of the point.
const LANE_SPREAD: u32 = 7;

/// Score lane holding the weighted capped squared residual of the model.
const LANE_MODEL_SCORE: u32 = 0;

/// Score lane holding the weighted capped squared motion, the residual of no motion.
const LANE_STILL_SCORE: u32 = 1;

/// Score lane holding the weight.
const LANE_SCORE_WEIGHT: u32 = 2;

/// Band word holding the first block row below the top bar, 0 without one.
const BAND_TOP: u32 = 0;

/// Band word holding the first block row of the bottom bar, the row count without one.
const BAND_BOTTOM: u32 = 1;

/// Band word holding the first block column right of the left bar, 0 without one.
const BAND_LEFT: u32 = 2;

/// Band word holding the first block column of the right bar, the column count without one.
const BAND_RIGHT: u32 = 3;

/// Fit state word holding the field's mean SAD, or 0 when it has none.
const STATE_SAD_SCALE: u32 = 0;

/// Fit state word that is 1 when the reference model words hold a model.
const STATE_REFERENCE_VALID: u32 = 1;

/// The first of the reference model's `a, b, tx, ty` state words, with `t` about the frame centre.
const STATE_REFERENCE: u32 = 2;

/// Fit state word holding the largest residual scale the solve passes use.
const STATE_SCALE_LIMIT: u32 = 6;

/// The scale limit of a fit that starts from scratch, high enough to never bind.
const UNLIMITED_SCALE: f32 = 1.0e30;

/// How much a zero vector counts in a solve, so static overlays cannot anchor a moving fit.
const ZERO_VECTOR_WEIGHT: f32 = 0.2;

/// Agreeing neighbours added to every block's count, so a block no neighbour agrees with keeps a
/// little weight.
const NEIGHBOUR_PRIOR: f32 = 0.5;

/// The least share of moving blocks, in eighths, in the row or column just inside a bar.
const BAND_EDGE_EIGHTHS: u32 = 7;

/// The most two opposite bars may differ in length, in block lines.
const BAND_PAIR_SLACK: u32 = 2;

/// The least total weight a solve needs.
const MIN_FIT_WEIGHT: f32 = 1e-3;

/// The least weighted mean squared distance of the blocks from their centroid, in luma pixels.
const MIN_FIT_SPREAD: f32 = 64.0;

/// How far below the no-motion score the model's score must land to be accepted.
const ACCEPT_MARGIN: f32 = 0.7;

/// The least weighted mean capped no-motion score, in squared luma pixels, that counts as motion.
const MIN_MOTION_SCORE: f32 = 1e-3;

/// The largest change of scale a model may carry.
const MAX_ZOOM_CHANGE: f32 = 0.15;

/// The largest shift a model may give a frame corner, in luma pixels.
const MAX_CORNER_SHIFT: f32 = 400.0;

/// Finds static letterbox and pillarbox bars in a pair field.
///
/// A bar is a run of block rows, or columns, whose vectors are all zero, starting at the frame's
/// edge and no longer than a third of the lines. The line just past it must move in almost every
/// block, and the opposite edge must hold a bar of nearly the same length. A still background
/// beside a moving layer that spans the frame can still pass for a symmetric pair of bars. One
/// workgroup writes the four band words to `bands`.
#[cube(launch_unchecked)]
pub fn mc_fit_bands(
    field: &[i32],
    bands: &mut [u32],
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
) {
    // Rows and columns 1 up to the last one fully inside the frame are counted.
    let inside_rows = comptime!(height.saturating_sub(blksize) / step);
    let inside_cols = comptime!(width.saturating_sub(blksize) / step);
    let interior_rows = comptime!(blocks_y.saturating_sub(2));
    let interior_cols = comptime!(blocks_x.saturating_sub(2));
    let last_row = comptime!(inside_rows.min(interior_rows));
    let last_col = comptime!(inside_cols.min(interior_cols));
    let counted = comptime!(last_row >= 1 && last_col >= 1);
    let mut row_moving = Shared::<[u32]>::new_slice(blocks_y as usize);
    let mut col_moving = Shared::<[u32]>::new_slice(blocks_x as usize);
    let thread_id = UNIT_POS_X;

    let mut row = thread_id;
    while row < blocks_y {
        let mut moving = 0u32;
        if row >= 1 && row <= last_row {
            for col in 1..comptime!(last_col + 1) {
                moving += block_moves(field, row * blocks_x + col);
            }
        }

        row_moving[row as usize] = moving;
        row += FIT_THREADS;
    }

    let mut col = thread_id;
    while col < blocks_x {
        let mut moving = 0u32;
        if col >= 1 && col <= last_col {
            for row in 1..comptime!(last_row + 1) {
                moving += block_moves(field, row * blocks_x + col);
            }
        }

        col_moving[col as usize] = moving;
        col += FIT_THREADS;
    }

    sync_cube();

    if thread_id != 0 {
        terminate!();
    }

    let mut top = 0u32;
    let mut bottom = 0u32;
    let mut left = 0u32;
    let mut right = 0u32;

    if counted {
        top = leading_band(&row_moving, 1, true, last_row, last_col);
        bottom = leading_band(&row_moving, last_row, false, last_row, last_col);
        left = leading_band(&col_moving, 1, true, last_col, last_row);
        right = leading_band(&col_moving, last_col, false, last_col, last_row);
    }

    let rows_paired = bands_pair(top, bottom);
    let cols_paired = bands_pair(left, right);

    // Without a bar the bounds sit outside the counted rows and columns.
    let top_end = 1 + top;
    let bottom_start = last_row + 1 - bottom;
    let left_end = 1 + left;
    let right_start = last_col + 1 - right;
    bands[BAND_TOP as usize] = select(rows_paired, top_end, 0u32);
    bands[BAND_BOTTOM as usize] = select(rows_paired, bottom_start, blocks_y);
    bands[BAND_LEFT as usize] = select(cols_paired, left_end, 0u32);
    bands[BAND_RIGHT as usize] = select(cols_paired, right_start, blocks_x);
}

/// Accumulates one stage of the camera-model fit over a pair field.
///
/// One thread handles one block and each workgroup reduces the [FIT_LANES] stage sums of its
/// blocks into `partials` in a fixed order. Each point is a block centre, `origin + blksize / 2`,
/// measured from the frame centre, and its motion is the block's vector. With `older_to_newer`
/// unset, `field` is the newer-to-older direction and each block is turned around, so every
/// stage sees the older frame's points moving into the newer frame.
#[cube(launch_unchecked)]
pub fn mc_fit_sums(
    field: &[i32],
    sad: &[f32],
    bands: &[u32],
    weights: &mut [f32],
    state: &[f32],
    models: &[f32],
    partials: &mut [f32],
    scale: f32,
    prior_slot: u32,
    #[comptime] stage: u32,
    #[comptime] has_sad: bool,
    #[comptime] older_to_newer: bool,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
) {
    let weigh = comptime!(stage == FIT_STAGE_WEIGH);
    let gate = comptime!(stage == FIT_STAGE_GATE);
    let polish = comptime!(stage == FIT_STAGE_POLISH);
    let solve = comptime!(stage == FIT_STAGE_SOLVE || stage == FIT_STAGE_POLISH);
    let blocks = comptime!(blocks_x * blocks_y);
    let scratch_len = comptime!(FIT_LANES * FIT_THREADS) as usize;
    let mut scratch = Shared::<[f32]>::new_slice(scratch_len);
    let thread_id = UNIT_POS_X;
    let block = ABSOLUTE_POS_X;

    #[unroll]
    for lane in 0..FIT_LANES {
        set_lane(&mut scratch, lane, thread_id, 0.0f32);
    }

    if block < blocks {
        let col = block % blocks_x;
        let row = block / blocks_x;
        let mvx = field[(block * 2) as usize];
        let mvy = field[(block * 2 + 1) as usize];
        let mut block_sad = 0.0f32;
        if has_sad {
            block_sad = sad[block as usize];
        }

        let half = f32::cast_from(blksize) * 0.5f32;
        let centre_x = f32::cast_from(width) * 0.5f32;
        let centre_y = f32::cast_from(height) * 0.5f32;
        let block_x = f32::cast_from(col * step) + half - centre_x;
        let block_y = f32::cast_from(row * step) + half - centre_y;
        let shift_x = f32::cast_from(mvx);
        let shift_y = f32::cast_from(mvy);

        let mut point_x = block_x;
        let mut point_y = block_y;
        let mut motion_x = shift_x;
        let mut motion_y = shift_y;
        if !older_to_newer {
            point_x = block_x + shift_x;
            point_y = block_y + shift_y;
            motion_x = -shift_x;
            motion_y = -shift_y;
        }

        let scale_sq = scale * scale;

        if weigh {
            let base = block_base_weight(
                field, bands, col, row, mvx, mvy, width, height, blksize, step, blocks_x, blocks_y,
            );
            weights[block as usize] = base;

            let base_sad = base * block_sad;
            set_lane(&mut scratch, LANE_BASE, thread_id, base);
            set_lane(&mut scratch, LANE_BASE_SAD, thread_id, base_sad);
        } else {
            let mut matched = weights[block as usize];
            if has_sad {
                let sad_scale = state[STATE_SAD_SCALE as usize];
                let safe_scale = f32::max(sad_scale, 1e-12f32);
                let ratio = block_sad / safe_scale;
                let ratio_sq = ratio * ratio;
                let sad_weight = 1.0f32 / (1.0f32 + ratio_sq * ratio_sq);
                matched *= select(sad_scale > 0.0f32, sad_weight, 1.0f32);
            }

            let reference_valid = state[STATE_REFERENCE_VALID as usize] != 0.0f32;
            let reference_a = state[STATE_REFERENCE as usize];
            let reference_b = state[(STATE_REFERENCE + 1) as usize];
            let reference_tx = state[(STATE_REFERENCE + 2) as usize];
            let reference_ty = state[(STATE_REFERENCE + 3) as usize];
            let reference_sq = residual_sq(
                reference_a,
                reference_b,
                reference_tx,
                reference_ty,
                point_x,
                point_y,
                motion_x,
                motion_y,
            );

            if solve {
                // Geman-McClure weights, `1 / (1 + (r / scale)^2)^2`, or the inliers at equal weight.
                let scale_limit = state[STATE_SCALE_LIMIT as usize];
                let limited_scale = f32::min(scale, scale_limit);
                let limited_sq = limited_scale * limited_scale;
                let damping = 1.0f32 / (1.0f32 + reference_sq / limited_sq);
                let mut residual_weight = damping * damping;
                if polish {
                    let inlier = reference_sq <= limited_sq;
                    residual_weight = select(inlier, 1.0f32, 0.0f32);
                }

                let still = mvx == 0i32 && mvy == 0i32;
                let zero_weight = select(still, ZERO_VECTOR_WEIGHT, 1.0f32);
                let fit_weight = select(reference_valid, residual_weight, 1.0f32);
                let weight = matched * zero_weight * fit_weight;
                let dot = point_x * motion_x + point_y * motion_y;
                let cross = point_x * motion_y - point_y * motion_x;
                let spread = point_x * point_x + point_y * point_y;

                set_lane(&mut scratch, LANE_WEIGHT, thread_id, weight);
                set_lane(&mut scratch, LANE_POINT_X, thread_id, weight * point_x);
                set_lane(&mut scratch, LANE_POINT_Y, thread_id, weight * point_y);
                set_lane(&mut scratch, LANE_MOTION_X, thread_id, weight * motion_x);
                set_lane(&mut scratch, LANE_MOTION_Y, thread_id, weight * motion_y);
                set_lane(&mut scratch, LANE_DOT, thread_id, weight * dot);
                set_lane(&mut scratch, LANE_CROSS, thread_id, weight * cross);
                set_lane(&mut scratch, LANE_SPREAD, thread_id, weight * spread);
            } else {
                // The gate scores the prior, which is in absolute pixels, at the absolute point.
                let mut model_sq = reference_sq;
                if gate {
                    let prior_base = prior_slot * MODEL_WORDS;
                    let prior_a = models[prior_base as usize];
                    let prior_b = models[(prior_base + 1) as usize];
                    let prior_tx = models[(prior_base + 2) as usize];
                    let prior_ty = models[(prior_base + 3) as usize];
                    let absolute_x = point_x + centre_x;
                    let absolute_y = point_y + centre_y;
                    model_sq = residual_sq(
                        prior_a, prior_b, prior_tx, prior_ty, absolute_x, absolute_y, motion_x, motion_y,
                    );
                }

                let motion_sq = motion_x * motion_x + motion_y * motion_y;
                let model_score = matched * f32::min(model_sq, scale_sq);
                let still_score = matched * f32::min(motion_sq, scale_sq);

                set_lane(&mut scratch, LANE_MODEL_SCORE, thread_id, model_score);
                set_lane(&mut scratch, LANE_STILL_SCORE, thread_id, still_score);
                set_lane(&mut scratch, LANE_SCORE_WEIGHT, thread_id, matched);
            }
        }
    }

    sync_cube();

    reduce_lanes(&mut scratch, thread_id);

    if thread_id < FIT_LANES {
        let lane = thread_id;
        let total = lane_total(&scratch, lane);
        partials[(CUBE_POS_X * FIT_LANES + lane) as usize] = total;
    }
}

/// Finishes one stage of the camera-model fit in a single workgroup.
///
/// It adds the [FIT_LANES] sums of all `num_partials` workgroups in a fixed order, then:
///
/// - weigh stores the mean SAD and clears the reference model,
/// - gate adopts the model in ring slot `prior_slot` when it is valid and would pass acceptance,
/// - solve and polish replace the reference with the closed-form least-squares similarity,
/// - score writes the reference, or identity with `valid` at 0, to ring slot `slot`.
#[cube(launch_unchecked)]
pub fn mc_fit_finish(
    partials: &[f32],
    state: &mut [f32],
    models: &mut [f32],
    num_partials: u32,
    prior_slot: u32,
    slot: u32,
    #[comptime] stage: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) {
    let weigh = comptime!(stage == FIT_STAGE_WEIGH);
    let gate = comptime!(stage == FIT_STAGE_GATE);
    let solve = comptime!(stage == FIT_STAGE_SOLVE || stage == FIT_STAGE_POLISH);
    let scratch_len = comptime!(FIT_LANES * FIT_THREADS) as usize;
    let mut scratch = Shared::<[f32]>::new_slice(scratch_len);
    let thread_id = UNIT_POS_X;

    #[unroll]
    for lane in 0..FIT_LANES {
        let mut sum = 0.0f32;
        let mut index = thread_id;
        while index < num_partials {
            sum += partials[(index * FIT_LANES + lane) as usize];
            index += FIT_THREADS;
        }

        set_lane(&mut scratch, lane, thread_id, sum);
    }

    sync_cube();

    reduce_lanes(&mut scratch, thread_id);

    if thread_id != 0 {
        terminate!();
    }

    let centre_x = f32::cast_from(width) * 0.5f32;
    let centre_y = f32::cast_from(height) * 0.5f32;

    if weigh {
        let base = lane_total(&scratch, LANE_BASE);
        let base_sad = lane_total(&scratch, LANE_BASE_SAD);
        let safe_base = f32::max(base, 1e-12f32);
        let sad_scale = select(base > 0.0f32, base_sad / safe_base, 0.0f32);

        state[STATE_SAD_SCALE as usize] = sad_scale;
        state[STATE_SCALE_LIMIT as usize] = UNLIMITED_SCALE;
        write_reference(state, 0.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32);
    } else if gate {
        let model_score = lane_total(&scratch, LANE_MODEL_SCORE);
        let still_score = lane_total(&scratch, LANE_STILL_SCORE);
        let weight = lane_total(&scratch, LANE_SCORE_WEIGHT);
        let base = prior_slot * MODEL_WORDS;
        let prior_valid = models[(base + 4) as usize] != 0.0f32;
        let passes = beats_still(model_score, still_score, weight);

        // A prior already holds the inliers, so the passes refine it at the inlier scale.
        if prior_valid && passes {
            let prior_a = models[base as usize];
            let prior_b = models[(base + 1) as usize];
            let prior_tx = models[(base + 2) as usize];
            let prior_ty = models[(base + 3) as usize];
            let centred_tx = prior_tx + prior_a * centre_x - prior_b * centre_y - centre_x;
            let centred_ty = prior_ty + prior_b * centre_x + prior_a * centre_y - centre_y;

            state[STATE_SCALE_LIMIT as usize] = FIT_INLIER_CAP;
            write_reference(state, 1.0f32, prior_a, prior_b, centred_tx, centred_ty);
        }
    } else if solve {
        let weight = lane_total(&scratch, LANE_WEIGHT);
        let sum_px = lane_total(&scratch, LANE_POINT_X);
        let sum_py = lane_total(&scratch, LANE_POINT_Y);
        let sum_dx = lane_total(&scratch, LANE_MOTION_X);
        let sum_dy = lane_total(&scratch, LANE_MOTION_Y);
        let sum_dot = lane_total(&scratch, LANE_DOT);
        let sum_cross = lane_total(&scratch, LANE_CROSS);
        let sum_spread = lane_total(&scratch, LANE_SPREAD);

        let inverse_weight = 1.0f32 / f32::max(weight, MIN_FIT_WEIGHT);
        let mean_px = sum_px * inverse_weight;
        let mean_py = sum_py * inverse_weight;
        let mean_dx = sum_dx * inverse_weight;
        let mean_dy = sum_dy * inverse_weight;
        let spread = sum_spread - weight * (mean_px * mean_px + mean_py * mean_py);
        let dot = sum_dot - weight * (mean_px * mean_dx + mean_py * mean_dy);
        let cross = sum_cross - weight * (mean_px * mean_dy - mean_py * mean_dx);

        let mut reference_valid = 0.0f32;
        let mut reference_a = 1.0f32;
        let mut reference_b = 0.0f32;
        let mut reference_tx = 0.0f32;
        let mut reference_ty = 0.0f32;

        if weight >= MIN_FIT_WEIGHT && spread > MIN_FIT_SPREAD * weight {
            let inverse_spread = 1.0f32 / spread;
            reference_valid = 1.0f32;
            reference_a = 1.0f32 + dot * inverse_spread;
            reference_b = cross * inverse_spread;
            reference_tx = mean_px + mean_dx - (reference_a * mean_px - reference_b * mean_py);
            reference_ty = mean_py + mean_dy - (reference_b * mean_px + reference_a * mean_py);
        }

        write_reference(
            state,
            reference_valid,
            reference_a,
            reference_b,
            reference_tx,
            reference_ty,
        );
    } else {
        let model_score = lane_total(&scratch, LANE_MODEL_SCORE);
        let still_score = lane_total(&scratch, LANE_STILL_SCORE);
        let weight = lane_total(&scratch, LANE_SCORE_WEIGHT);

        let reference_valid = state[STATE_REFERENCE_VALID as usize] != 0.0f32;
        let model_a = state[STATE_REFERENCE as usize];
        let model_b = state[(STATE_REFERENCE + 1) as usize];
        let centred_tx = state[(STATE_REFERENCE + 2) as usize];
        let centred_ty = state[(STATE_REFERENCE + 3) as usize];
        let model_tx = centred_tx + centre_x - (model_a * centre_x - model_b * centre_y);
        let model_ty = centred_ty + centre_y - (model_b * centre_x + model_a * centre_y);

        let passes = beats_still(model_score, still_score, weight);
        let zoom = f32::sqrt(model_a * model_a + model_b * model_b);
        let zoom_change = f32::abs(zoom - 1.0f32);
        let zoom_ok = zoom_change <= MAX_ZOOM_CHANGE;
        let roll_ok = model_a > 0.0f32 && f32::abs(model_b) <= MAX_ROLL_TAN * model_a;
        let far_x = f32::cast_from(width);
        let far_y = f32::cast_from(height);
        let corner_limit = MAX_CORNER_SHIFT * MAX_CORNER_SHIFT;
        let top_left = corner_shift_sq(model_a, model_b, model_tx, model_ty, 0.0f32, 0.0f32);
        let top_right = corner_shift_sq(model_a, model_b, model_tx, model_ty, far_x, 0.0f32);
        let bottom_left = corner_shift_sq(model_a, model_b, model_tx, model_ty, 0.0f32, far_y);
        let bottom_right = corner_shift_sq(model_a, model_b, model_tx, model_ty, far_x, far_y);
        let top_ok = top_left <= corner_limit && top_right <= corner_limit;
        let bottom_ok = bottom_left <= corner_limit && bottom_right <= corner_limit;
        let plausible = zoom_ok && roll_ok && top_ok && bottom_ok;
        let accepted = reference_valid && passes && plausible;

        let base = (slot * MODEL_WORDS) as usize;
        models[base] = select(accepted, model_a, 1.0f32);
        models[base + 1] = select(accepted, model_b, 0.0f32);
        models[base + 2] = select(accepted, model_tx, 0.0f32);
        models[base + 3] = select(accepted, model_ty, 0.0f32);
        models[base + 4] = select(accepted, 1.0f32, 0.0f32);
    }
}

/// Whether a model's score cuts the no-motion score by [ACCEPT_MARGIN] on a moving field.
#[cube]
fn beats_still(model_score: f32, still_score: f32, weight: f32) -> bool {
    let moving = weight > 0.0f32 && still_score > MIN_MOTION_SCORE * weight;
    let better = model_score <= ACCEPT_MARGIN * still_score;

    moving && better
}

/// The base weight of one block of the pair field.
///
/// Border blocks, bar blocks, blocks reaching past the frame and blocks whose match lands outside
/// it weigh nothing. The rest weigh by how many of their four neighbours agree with them.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "the weight reads the block, its vector, the bars and the whole grid shape"
)]
fn block_base_weight(
    field: &[i32],
    bands: &[u32],
    col: u32,
    row: u32,
    mvx: i32,
    mvy: i32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
) -> f32 {
    // Neighbours of a block moving with the camera differ by up to 3/10 of a step at 15 degrees.
    let tolerance = comptime!((1 + (3 * step) / 10) as i32);
    let origin_x = (col * step) as i32;
    let origin_y = (row * step) as i32;
    let size = blksize as i32;
    let interior = col >= 1 && row >= 1 && col + 1 < blocks_x && row + 1 < blocks_y;
    let inside = origin_x + size <= width as i32 && origin_y + size <= height as i32;
    let target_x = origin_x + mvx;
    let target_y = origin_y + mvy;
    let lands_x = target_x >= 0 && target_x + size <= width as i32;
    let lands_y = target_y >= 0 && target_y + size <= height as i32;
    let in_rows = row >= bands[BAND_TOP as usize] && row < bands[BAND_BOTTOM as usize];
    let in_cols = col >= bands[BAND_LEFT as usize] && col < bands[BAND_RIGHT as usize];

    let mut weight = 0.0f32;

    if interior && inside && lands_x && lands_y && in_rows && in_cols {
        let block = row * blocks_x + col;
        let left = neighbour_agrees(field, block - 1, mvx, mvy, tolerance);
        let right = neighbour_agrees(field, block + 1, mvx, mvy, tolerance);
        let up = neighbour_agrees(field, block - blocks_x, mvx, mvy, tolerance);
        let down = neighbour_agrees(field, block + blocks_x, mvx, mvy, tolerance);
        let agreeing = f32::cast_from(left + right + up + down);
        weight = (agreeing + NEIGHBOUR_PRIOR) / (4.0f32 + NEIGHBOUR_PRIOR);
    }

    weight
}

/// 1 when block `neighbour`'s vector is within `tolerance` of `(mvx, mvy)` on both axes.
#[cube]
fn neighbour_agrees(field: &[i32], neighbour: u32, mvx: i32, mvy: i32, tolerance: i32) -> u32 {
    let other_x = field[(neighbour * 2) as usize];
    let other_y = field[(neighbour * 2 + 1) as usize];
    let gap_x = other_x - mvx;
    let gap_y = other_y - mvy;
    let close_x = i32::max(gap_x, -gap_x) <= tolerance;
    let close_y = i32::max(gap_y, -gap_y) <= tolerance;

    select(close_x && close_y, 1u32, 0u32)
}

/// 1 when the block's vector is not zero.
#[cube]
fn block_moves(field: &[i32], block: u32) -> u32 {
    let mvx = field[(block * 2) as usize];
    let mvy = field[(block * 2 + 1) as usize];

    select(mvx != 0i32 || mvy != 0i32, 1u32, 0u32)
}

/// The length of the bar of still lines from line `start` inward, or 0 when it is not a bar.
///
/// `moving` holds each line's moving-block count over `width` blocks, and lines 1 to `last` are
/// counted. The walk goes up from `start` when `upward` is set and down otherwise. A bar is at most
/// a third of the lines, and the line just past it moves in at least [BAND_EDGE_EIGHTHS] eighths
/// of its blocks.
#[cube]
fn leading_band(moving: &Shared<[u32]>, start: u32, upward: bool, last: u32, width: u32) -> u32 {
    let mut steps = 0u32;
    let mut walking = true;
    while walking {
        let line = band_line(start, steps, upward);
        let still = steps < last && moving[line as usize] == 0u32;
        walking = still;
        if still {
            steps += 1;
        }
    }

    let edge = band_line(start, steps, upward);
    let edge_moving = moving[edge as usize];
    let short = steps * 3 <= last;
    let edge_ok = steps < last && edge_moving * 8 >= BAND_EDGE_EIGHTHS * width;

    select(steps > 0 && short && edge_ok, steps, 0u32)
}

/// The line `steps` lines from `start`, upward or downward.
#[cube]
fn band_line(start: u32, steps: u32, upward: bool) -> u32 {
    let mut line = start;
    if upward {
        line += steps;
    } else {
        line -= steps;
    }

    line
}

/// Whether two opposite bars are both present and within [BAND_PAIR_SLACK] lines of each other.
#[cube]
fn bands_pair(first: u32, second: u32) -> bool {
    let longer = u32::max(first, second);
    let shorter = u32::min(first, second);

    shorter > 0 && longer - shorter <= BAND_PAIR_SLACK
}

/// Writes `value` to `lane` of `thread_id`'s slot in the reduction scratch.
#[cube]
fn set_lane(scratch: &mut Shared<[f32]>, lane: u32, thread_id: u32, value: f32) {
    scratch[(lane * FIT_THREADS + thread_id) as usize] = value;
}

/// The reduced total of `lane` after [reduce_lanes].
#[cube]
fn lane_total(scratch: &Shared<[f32]>, lane: u32) -> f32 {
    scratch[(lane * FIT_THREADS) as usize]
}

/// Halves every lane's [FIT_THREADS] partial sums in `scratch` down to its first slot.
///
/// The pairing never changes, so the totals are the same bits on every run. Every thread reaches
/// every `sync_cube()`.
#[cube]
fn reduce_lanes(scratch: &mut Shared<[f32]>, thread_id: u32) {
    #[unroll]
    for round in 0..FIT_ROUNDS {
        let stride = comptime!(FIT_THREADS >> (round + 1));
        if thread_id < stride {
            #[unroll]
            for lane in 0..FIT_LANES {
                let here = (lane * FIT_THREADS + thread_id) as usize;
                let partner = here + stride as usize;
                scratch[here] += scratch[partner];
            }
        }

        sync_cube();
    }
}

/// Writes the reference model words of the fit state.
#[cube]
fn write_reference(state: &mut [f32], valid: f32, a: f32, b: f32, tx: f32, ty: f32) {
    state[STATE_REFERENCE_VALID as usize] = valid;
    state[STATE_REFERENCE as usize] = a;
    state[(STATE_REFERENCE + 1) as usize] = b;
    state[(STATE_REFERENCE + 2) as usize] = tx;
    state[(STATE_REFERENCE + 3) as usize] = ty;
}

/// The squared distance between `(motion_x, motion_y)` and the motion the model gives `(x, y)`.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "the residual reads a whole model, a point and a motion"
)]
fn residual_sq(a: f32, b: f32, tx: f32, ty: f32, x: f32, y: f32, motion_x: f32, motion_y: f32) -> f32 {
    let predicted_x = (a - 1.0f32) * x - b * y + tx;
    let predicted_y = b * x + (a - 1.0f32) * y + ty;
    let residual_x = motion_x - predicted_x;
    let residual_y = motion_y - predicted_y;

    residual_x * residual_x + residual_y * residual_y
}

/// The squared distance the model moves point `(x, y)`.
#[cube]
fn corner_shift_sq(a: f32, b: f32, tx: f32, ty: f32, x: f32, y: f32) -> f32 {
    let shift_x = a * x - b * y + tx - x;
    let shift_y = b * x + a * y + ty - y;

    shift_x * shift_x + shift_y * shift_y
}

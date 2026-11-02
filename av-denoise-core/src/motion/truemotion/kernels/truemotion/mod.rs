mod candidates;
mod score;

use cubecl::prelude::*;

pub(crate) use self::candidates::{gather_candidates, neighbourhood_median, stage_neighbourhood};
use self::score::{score_cooperative, score_serial};
use crate::motion::kernels::BlockVector;
use crate::motion::truemotion::params::MAX_REFINE_RADIUS;

/// The most distinct vectors one block scores before its refine.
///
/// They are its own vector, its neighbourhood median, its four adjacent blocks' vectors, zero, the
/// camera model's prediction, and its seed with the seeds of its four diagonal blocks.
pub const TRUEMOTION_CANDIDATES: u32 = 13;

/// Entries the shared vector list holds, enough for every candidate or one refine ring.
pub const LIST_CAPACITY: u32 = TRUEMOTION_CANDIDATES;

const REFINE_SIDE: u32 = 2 * MAX_REFINE_RADIUS + 1;
const _: () = assert!(REFINE_SIDE * REFINE_SIDE - 1 <= LIST_CAPACITY);

/// `i32` words of shared state.
///
/// They are the list length, the median's entry and vector, the best vector, the block's own vector
/// and a flag every thread reads to take the same branch.
pub const META_WORDS: u32 = 9;

/// `f32` words of shared state, the best cost, its SAD and the block's penalty per pixel.
pub const SCORE_WORDS: u32 = 3;

const META_COUNT: usize = 0;
const META_MEDIAN_ENTRY: usize = 1;
const META_MEDIAN_X: usize = 2;
const META_MEDIAN_Y: usize = 3;
const META_BEST_X: usize = 4;
const META_BEST_Y: usize = 5;
const META_OWN_X: usize = 6;
const META_OWN_Y: usize = 7;
const META_FLAG: usize = 8;

const SCORE_BEST_COST: usize = 0;
const SCORE_BEST_SAD: usize = 1;
const SCORE_LAMBDA: usize = 2;

/// The most neighbours a block has in its 3x3 neighbourhood.
const NEIGHBOURHOOD: u32 = 8;

/// `i32` words of the staged 3x3 field, two per block.
const STAGED_FIELD_WORDS: u32 = 18;

/// `i32` words the staged 3x3 field and the staged 3x3 seeds hold together.
pub const STAGED_WORDS: u32 = 2 * STAGED_FIELD_WORDS;

/// `u32` words of the staged 3x3 change flags.
pub const STAGED_FLAG_WORDS: u32 = 9;

/// The early check found no winner, so every candidate is scored.
const UNSETTLED: i32 = 0;
/// The block's own vector wins.
const OWN_WINS: i32 = 1;
/// The median wins.
const MEDIAN_WINS: i32 = 2;

/// The values every block of a truemotion pass scores with.
///
/// `lambda_base` and `lambda_noise` build the penalty per pixel of distance from the median, as
/// `lambda_base + lambda_noise * floor`. `fade_level` is where the median's excess SAD halves it,
/// with 0 turning the fade off. `thsad` and the derived word of `noise` at `sad_field` score the
/// confidence.
#[derive(CubeType, Clone, Copy)]
pub(crate) struct PassValues {
    pub(crate) lambda_base: f32,
    pub(crate) lambda_noise: f32,
    pub(crate) surcharge: f32,
    pub(crate) fade_level: f32,
    pub(crate) thsad: f32,
    pub(crate) sad_field: u32,
}

/// Where one block sits and which frames it matches.
#[derive(CubeType, Clone, Copy)]
pub(crate) struct PassBlock {
    pub(crate) block_col: u32,
    pub(crate) block_row: u32,
    pub(crate) neighbour_index: u32,
    pub(crate) neighbour_base: u32,
    pub(crate) origin_x: i32,
    pub(crate) origin_y: i32,
    pub(crate) thread_id: u32,
    pub(crate) threads: u32,
}

/// The shared buffers one block's scoring reuses.
#[derive(CubeType)]
struct ScoreScratch {
    centre_smem: Shared<[f32]>,
    list: Shared<[i32]>,
    sads: Shared<[f32]>,
    partials: Shared<[f32]>,
    row_sums: Shared<[f32]>,
}

#[cube]
fn clamp_coord(value: i32, limit: i32) -> i32 {
    let mut result = value;
    if value < 0 {
        result = 0;
    } else if value >= limit {
        result = limit - 1;
    }

    result
}

#[cube]
pub(crate) fn abs_i32(value: i32) -> i32 {
    let mut result = value;
    if value < 0 {
        result = -value;
    }

    result
}

/// Loads the block's centre tile, clamped at the frame edge, with every thread of the cube.
///
/// The centre frame starts `centre_base` values into `pyramid`.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the tile load reads"
)]
pub(crate) fn load_centre_tile(
    pyramid: &[f32],
    centre_smem: &mut Shared<[f32]>,
    centre_base: u32,
    block_origin_x: i32,
    block_origin_y: i32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
) {
    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let mut pixel_y = local_y;
    while pixel_y < blksize {
        let mut pixel_x = local_x;
        while pixel_x < blksize {
            let centre_x = clamp_coord(block_origin_x + pixel_x as i32, width as i32);
            let centre_y = clamp_coord(block_origin_y + pixel_y as i32, height as i32);
            let pixel_index = (centre_y * width as i32 + centre_x) as u32;
            centre_smem[(pixel_y * blksize + pixel_x) as usize] =
                pyramid[(centre_base + pixel_index) as usize];
            pixel_x += CUBE_DIM_X;
        }

        pixel_y += CUBE_DIM_Y;
    }
}

/// The vector of block `(col, row)` in the field that starts `base` values into `field`.
#[cube]
fn field_vector(field: &[i32], base: u32, col: u32, row: u32, #[comptime] blocks_x: u32) -> BlockVector {
    let index = (base + (row * blocks_x + col) * 2u32) as usize;

    BlockVector {
        dx: field[index],
        dy: field[index + 1],
    }
}

/// Whether block `(col, row)` lies on the grid.
#[cube]
fn on_grid(col: i32, row: i32, #[comptime] blocks_x: u32, #[comptime] blocks_y: u32) -> bool {
    col >= 0 && row >= 0 && col < blocks_x as i32 && row < blocks_y as i32
}

/// Sums the SADs of the listed vectors from entry `first` up to `count` into the scratch's SADs,
/// with the serial layout when `cooperative_threads` is 0 and the cooperative one otherwise.
///
/// Every thread must call it.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the scoring reads"
)]
fn score_entries(
    pyramid: &[f32],
    scratch: &mut ScoreScratch,
    first: u32,
    count: u32,
    block: &PassBlock,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
    #[comptime] cooperative_threads: u32,
) {
    if comptime!(cooperative_threads == 0) {
        score_serial(
            pyramid,
            &scratch.centre_smem,
            &scratch.list,
            &mut scratch.sads,
            &mut scratch.row_sums,
            first,
            count,
            block,
            width,
            height,
            blksize,
        );
    } else {
        score_cooperative(
            pyramid,
            &scratch.centre_smem,
            &scratch.list,
            &mut scratch.sads,
            &mut scratch.partials,
            first,
            count,
            block,
            width,
            height,
            blksize,
            cooperative_threads,
        );
    }
}

/// A vector's cost, its SAD plus `lambda` per pixel of distance from the median.
///
/// A vector off the median also pays `surcharge` times its SAD.
#[cube]
pub(crate) fn penalised_cost(
    sad: f32,
    mvx: i32,
    mvy: i32,
    median_x: i32,
    median_y: i32,
    lambda: f32,
    surcharge: f32,
) -> f32 {
    let deviation = abs_i32(mvx - median_x) + abs_i32(mvy - median_y);
    let mut cost = sad + lambda * deviation as f32;
    if deviation != 0i32 && surcharge > 0.0f32 {
        cost += surcharge * sad;
    }

    cost
}

/// The block's penalty per pixel of distance, faded by how poorly the median matches.
///
/// The floor is the derived word of `noise` at `sad_field` when `reads_floor` is set and it is
/// positive, and 0 otherwise. With a floor of 0 the fade reads the median's whole SAD, so on heavy
/// grain the searches that keep the unscaled penalty fade it further.
#[cube]
fn block_lambda(noise: &[f32], median_sad: f32, values: &PassValues, #[comptime] reads_floor: bool) -> f32 {
    let mut floor = 0.0f32;
    if reads_floor {
        let reading = noise[values.sad_field as usize];
        if reading > 0.0f32 {
            floor = reading;
        }
    }

    let mut lambda = values.lambda_base + values.lambda_noise * floor;

    if values.fade_level > 0.0f32 {
        let excess = f32::max(median_sad - floor, 0.0f32);
        let level_sq = values.fade_level * values.fade_level;
        let fade = level_sq / (level_sq + excess * excess);
        lambda *= fade;
    }

    lambda
}

/// Makes listed entry `entry` the best vector at `cost`, with its SAD from `sads`.
#[cube]
fn take_best(
    list: &Shared<[i32]>,
    sads: &Shared<[f32]>,
    meta: &mut Shared<[i32]>,
    scores: &mut Shared<[f32]>,
    entry: u32,
    cost: f32,
) {
    meta[META_BEST_X] = list[(entry * 2u32) as usize];
    meta[META_BEST_Y] = list[(entry * 2u32 + 1u32) as usize];
    scores[SCORE_BEST_COST] = cost;
    scores[SCORE_BEST_SAD] = sads[entry as usize];
}

/// Whether the block's own vector or its median wins before the other candidates are scored, and
/// makes the winner the best one in `meta` and `scores`.
///
/// Only the first two entries need scores. Every candidate other than the median sits at least one
/// pixel from it, so it costs at least the penalty for one pixel. The cheaper of own and median wins
/// when it costs no more than that, with a tie going to the earlier entry, as the full pick does.
/// The only candidate wins outright. Run by one thread.
#[cube]
fn settle_early(
    list: &Shared<[i32]>,
    sads: &Shared<[f32]>,
    meta: &mut Shared<[i32]>,
    scores: &mut Shared<[f32]>,
    noise: &[f32],
    values: &PassValues,
    #[comptime] reads_floor: bool,
) -> i32 {
    let count = meta[META_COUNT] as u32;
    let median_entry = meta[META_MEDIAN_ENTRY] as u32;
    let median_x = meta[META_MEDIAN_X];
    let median_y = meta[META_MEDIAN_Y];
    let median_sad = sads[median_entry as usize];
    let lambda = block_lambda(noise, median_sad, values, reads_floor);
    scores[SCORE_LAMBDA] = lambda;

    let own_cost = penalised_cost(
        sads[0],
        list[0],
        list[1],
        median_x,
        median_y,
        lambda,
        values.surcharge,
    );

    let median_cost = penalised_cost(
        median_sad,
        median_x,
        median_y,
        median_x,
        median_y,
        lambda,
        values.surcharge,
    );

    let mut outcome = UNSETTLED.runtime();
    if count == 1u32 || (median_entry == 0u32 && own_cost <= lambda) {
        outcome = OWN_WINS;
    } else if median_entry == 1u32 {
        if own_cost <= median_cost && own_cost <= lambda {
            outcome = OWN_WINS;
        } else if median_cost < own_cost && median_cost <= lambda {
            outcome = MEDIAN_WINS;
        }
    }

    if outcome == OWN_WINS {
        take_best(list, sads, meta, scores, 0u32, own_cost);
    } else if outcome == MEDIAN_WINS {
        take_best(list, sads, meta, scores, 1u32, median_cost);
    }

    outcome
}

/// Picks the lowest-cost listed vector into `meta` and `scores`, with a tie keeping the earlier one.
///
/// It costs with the penalty [settle_early] left in `scores`. Run by one thread after the SADs
/// are in.
#[cube]
fn pick_listed(
    list: &Shared<[i32]>,
    sads: &Shared<[f32]>,
    meta: &mut Shared<[i32]>,
    scores: &mut Shared<[f32]>,
    values: &PassValues,
) {
    let count = meta[META_COUNT] as u32;
    let median_x = meta[META_MEDIAN_X];
    let median_y = meta[META_MEDIAN_Y];
    let lambda = scores[SCORE_LAMBDA];

    let mut best = 0u32;
    let mut best_cost = penalised_cost(
        sads[0],
        list[0],
        list[1],
        median_x,
        median_y,
        lambda,
        values.surcharge,
    );
    let mut entry = 1u32;
    while entry < count {
        let mvx = list[(entry * 2u32) as usize];
        let mvy = list[(entry * 2u32 + 1u32) as usize];
        let sad = sads[entry as usize];
        let cost = penalised_cost(sad, mvx, mvy, median_x, median_y, lambda, values.surcharge);
        if cost < best_cost {
            best_cost = cost;
            best = entry;
        }

        entry += 1u32;
    }

    take_best(list, sads, meta, scores, best, best_cost);
}

/// Replaces the list with the ring of vectors within `refine_radius` of the best one.
///
/// Run by one thread.
#[cube]
fn list_refine_ring(list: &mut Shared<[i32]>, meta: &mut Shared<[i32]>, #[comptime] refine_radius: u32) {
    let best_x = meta[META_BEST_X];
    let best_y = meta[META_BEST_Y];
    let side = comptime!(2 * refine_radius + 1);
    let reach = comptime!(refine_radius as i32);
    let mut count = 0u32;

    for row in 0..side {
        for col in 0..side {
            let offset_x = col as i32 - reach;
            let offset_y = row as i32 - reach;
            if offset_x != 0i32 || offset_y != 0i32 {
                list[(count * 2u32) as usize] = best_x + offset_x;
                list[(count * 2u32 + 1u32) as usize] = best_y + offset_y;
                count += 1u32;
            }
        }
    }

    meta[META_COUNT] = count as i32;
}

/// Lets each refine vector replace the best one when it costs strictly less.
///
/// The ring is costed with the penalty the listed candidates used. Run by one thread.
#[cube]
fn pick_refined(
    list: &Shared<[i32]>,
    sads: &Shared<[f32]>,
    meta: &mut Shared<[i32]>,
    scores: &mut Shared<[f32]>,
    values: &PassValues,
) {
    let count = meta[META_COUNT] as u32;
    let median_x = meta[META_MEDIAN_X];
    let median_y = meta[META_MEDIAN_Y];
    let lambda = scores[SCORE_LAMBDA];

    let mut entry = 0u32;
    while entry < count {
        let mvx = list[(entry * 2u32) as usize];
        let mvy = list[(entry * 2u32 + 1u32) as usize];
        let sad = sads[entry as usize];
        let cost = penalised_cost(sad, mvx, mvy, median_x, median_y, lambda, values.surcharge);
        if cost < scores[SCORE_BEST_COST] {
            scores[SCORE_BEST_COST] = cost;
            scores[SCORE_BEST_SAD] = sad;
            meta[META_BEST_X] = mvx;
            meta[META_BEST_Y] = mvy;
        }

        entry += 1u32;
    }
}

/// A match's confidence between 0 and 1, from how far its SAD sits past the noise floor.
///
/// It is 1 up to `sad_noise_floor` and falls to 0 at `thsad` past it.
#[cube]
pub(crate) fn pass_confidence(sad: f32, sad_noise_floor: f32, thsad: f32) -> f32 {
    let mut excess = sad - sad_noise_floor;
    if excess < 0.0f32 {
        excess = 0.0f32;
    }

    let thsad_sq = thsad * thsad;
    let excess_sq = excess * excess;
    let mut confidence = (thsad_sq - excess_sq) / (thsad_sq + excess_sq);
    if confidence < 0.0f32 {
        confidence = 0.0f32;
    }

    confidence
}

/// Whether the block or any block of its 3x3 neighbourhood changed its vector in the pass that
/// wrote `changes`, which starts `base` values in.
///
/// Each thread stages one flag into `flags`, and every thread gets the same answer. Every thread
/// must call it.
#[cube]
fn neighbourhood_changed(
    changes: &[u32],
    base: u32,
    flags: &mut Shared<[u32]>,
    block: &PassBlock,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
) -> bool {
    let mut word = block.thread_id;
    while word < STAGED_FLAG_WORDS {
        let col = block.block_col as i32 + (word % 3u32) as i32 - 1i32;
        let row = block.block_row as i32 + (word / 3u32) as i32 - 1i32;

        let mut flag = 0u32;
        if on_grid(col, row, blocks_x, blocks_y) {
            let index = base + row as u32 * blocks_x + col as u32;
            flag = changes[index as usize];
        }

        flags[word as usize] = flag;
        word += block.threads;
    }

    sync_cube();

    let mut changed = false;
    for word in 0..STAGED_FLAG_WORDS {
        if flags[word as usize] != 0u32 {
            changed = true;
        }
    }

    changed
}

/// The slots one block's outputs go to.
#[derive(CubeType, Clone, Copy)]
struct PassSlots {
    mv_slot: u32,
    block_slot: u32,
}

/// Writes the best vector, and its confidence, SAD and change flag when asked. Run by one thread.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is an output, an input the confidence reads or a comptime switch"
)]
fn write_result(
    meta: &Shared<[i32]>,
    scores: &Shared<[f32]>,
    mv_out: &mut [i32],
    confidence_out: &mut [f32],
    sad_out: &mut [f32],
    changes_out: &mut [u32],
    noise: &[f32],
    values: &PassValues,
    slots: &PassSlots,
    #[comptime] write_confidence: bool,
    #[comptime] write_sad: bool,
    #[comptime] writes_changes: bool,
) {
    let mv_slot = slots.mv_slot as usize;
    let block_slot = slots.block_slot as usize;
    let best_x = meta[META_BEST_X];
    let best_y = meta[META_BEST_Y];
    let best_sad = scores[SCORE_BEST_SAD];
    mv_out[mv_slot] = best_x;
    mv_out[mv_slot + 1] = best_y;

    if write_confidence {
        let sad_noise_floor = noise[values.sad_field as usize];
        confidence_out[block_slot] = pass_confidence(best_sad, sad_noise_floor, values.thsad);
    }

    if write_sad {
        sad_out[block_slot] = best_sad;
    }

    if writes_changes {
        let kept = best_x == meta[META_OWN_X] && best_y == meta[META_OWN_Y];
        changes_out[block_slot] = select(kept, 0u32, 1u32);
    }
}

/// Runs one truemotion pass for this cube's block, scoring with `cooperative_threads` threads
/// sharing each vector, or one thread per row of a vector when it is 0.
///
/// See [mc_truemotion] for the inputs and outputs.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, value or comptime shape the pass binds"
)]
pub(crate) fn truemotion_block(
    pyramid: &[f32],
    neighbour_slots: &[u32],
    mv_in: &[i32],
    seeds: &[i32],
    predictions: &[f32],
    changes_in: &[u32],
    mv_out: &mut [i32],
    confidence_out: &mut [f32],
    sad_out: &mut [f32],
    changes_out: &mut [u32],
    noise: &[f32],
    centre_slot: u32,
    values: &PassValues,
    #[comptime] has_seeds: bool,
    #[comptime] model_candidate: bool,
    #[comptime] reads_floor: bool,
    #[comptime] write_confidence: bool,
    #[comptime] write_sad: bool,
    #[comptime] reads_changes: bool,
    #[comptime] writes_changes: bool,
    #[comptime] refine_radius: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] slot_pixels: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
    #[comptime] mv_stride: u32,
    #[comptime] conf_stride: u32,
    #[comptime] cooperative_threads: u32,
) {
    let neighbour_index = CUBE_POS_Z;
    let neighbour_slot = neighbour_slots[neighbour_index as usize];
    let block = PassBlock {
        block_col: CUBE_POS_X,
        block_row: CUBE_POS_Y,
        neighbour_index,
        neighbour_base: neighbour_slot * slot_pixels,
        origin_x: CUBE_POS_X as i32 * step as i32,
        origin_y: CUBE_POS_Y as i32 * step as i32,
        thread_id: UNIT_POS_Y * CUBE_DIM_X + UNIT_POS_X,
        threads: CUBE_DIM_X * CUBE_DIM_Y,
    };
    let block_index = block.block_row * blocks_x + block.block_col;
    let centre_base = centre_slot * slot_pixels;
    let mv_base = neighbour_index * mv_stride;
    let conf_base = neighbour_index * conf_stride;
    let slots = PassSlots {
        mv_slot: mv_base + block_index * 2u32,
        block_slot: conf_base + block_index,
    };

    let block_pixels = comptime!(blksize * blksize);
    let partial_words = comptime!(u32::max(LIST_CAPACITY * cooperative_threads, 1));
    let row_words = comptime!(
        if cooperative_threads == 0 {
            LIST_CAPACITY * blksize
        } else {
            1
        }
    );
    let mut scratch = ScoreScratch {
        centre_smem: Shared::<[f32]>::new_slice(block_pixels as usize),
        list: Shared::<[i32]>::new_slice(comptime!(2 * LIST_CAPACITY) as usize),
        sads: Shared::<[f32]>::new_slice(LIST_CAPACITY as usize),
        partials: Shared::<[f32]>::new_slice(partial_words as usize),
        row_sums: Shared::<[f32]>::new_slice(row_words as usize),
    };
    let mut meta = Shared::<[i32]>::new_slice(META_WORDS as usize);
    let mut scores = Shared::<[f32]>::new_slice(SCORE_WORDS as usize);
    let mut staged = Shared::<[i32]>::new_slice(STAGED_WORDS as usize);
    let mut flags = Shared::<[u32]>::new_slice(STAGED_FLAG_WORDS as usize);

    // A block whose neighbourhood kept every vector last pass would pick what it holds again, and
    // its confidence and SAD from that pass still stand.
    if reads_changes {
        let changed = neighbourhood_changed(changes_in, conf_base, &mut flags, &block, blocks_x, blocks_y);

        if !changed {
            if block.thread_id == 0u32 {
                let own = field_vector(mv_in, mv_base, block.block_col, block.block_row, blocks_x);
                let mv_slot = slots.mv_slot as usize;
                mv_out[mv_slot] = own.dx;
                mv_out[mv_slot + 1] = own.dy;

                if writes_changes {
                    changes_out[slots.block_slot as usize] = 0u32;
                }
            }

            // Every thread read the same flags, so the whole cube exits together.
            terminate!();
        }
    }

    stage_neighbourhood(
        mv_in,
        seeds,
        mv_base,
        &mut staged,
        &block,
        has_seeds,
        blocks_x,
        blocks_y,
    );

    load_centre_tile(
        pyramid,
        &mut scratch.centre_smem,
        centre_base,
        block.origin_x,
        block.origin_y,
        width,
        height,
        blksize,
    );

    sync_cube();

    neighbourhood_median(&staged, &mut meta, &block, blocks_x, blocks_y);

    sync_cube();

    if block.thread_id == 0u32 {
        gather_candidates(
            &staged,
            predictions,
            &mut scratch.list,
            &mut meta,
            &block,
            has_seeds,
            model_candidate,
            width,
            height,
            blksize,
            blocks_x,
            blocks_y,
        );
    }

    sync_cube();

    // Own is entry 0 and the median entry 0 or 1, so the first two entries settle most blocks.
    let listed = meta[META_COUNT] as u32;
    let head = u32::min(listed, 2u32);
    score_entries(
        pyramid,
        &mut scratch,
        0u32,
        head,
        &block,
        width,
        height,
        blksize,
        cooperative_threads,
    );

    sync_cube();

    if block.thread_id == 0u32 {
        let outcome = settle_early(
            &scratch.list,
            &scratch.sads,
            &mut meta,
            &mut scores,
            noise,
            values,
            reads_floor,
        );
        meta[META_FLAG] = outcome;
    }

    sync_cube();

    let outcome = meta[META_FLAG];
    if outcome == OWN_WINS {
        if block.thread_id == 0u32 {
            write_result(
                &meta,
                &scores,
                mv_out,
                confidence_out,
                sad_out,
                changes_out,
                noise,
                values,
                &slots,
                write_confidence,
                write_sad,
                writes_changes,
            );
        }

        // Every thread read the same flag, so the whole cube exits together.
        terminate!();
    }

    // A settled median skips the rest, and the empty range still reaches every barrier.
    let unsettled = outcome == UNSETTLED;
    let rest = select(unsettled, head, listed);
    score_entries(
        pyramid,
        &mut scratch,
        rest,
        listed,
        &block,
        width,
        height,
        blksize,
        cooperative_threads,
    );

    sync_cube();

    // The search already scored the block's own vector against everything around it, so only a
    // switch to another candidate earns a refine.
    if block.thread_id == 0u32 {
        if unsettled {
            pick_listed(&scratch.list, &scratch.sads, &mut meta, &mut scores, values);
        }

        let kept = meta[META_BEST_X] == meta[META_OWN_X] && meta[META_BEST_Y] == meta[META_OWN_Y];
        let refines = comptime!(refine_radius > 0) && !kept;
        meta[META_FLAG] = select(refines, 1i32, 0i32);

        if comptime!(refine_radius > 0) {
            list_refine_ring(&mut scratch.list, &mut meta, refine_radius);
        }
    }

    sync_cube();

    if meta[META_FLAG] == 0i32 {
        if block.thread_id == 0u32 {
            write_result(
                &meta,
                &scores,
                mv_out,
                confidence_out,
                sad_out,
                changes_out,
                noise,
                values,
                &slots,
                write_confidence,
                write_sad,
                writes_changes,
            );
        }

        // Every thread read the same flag, so the whole cube exits together.
        terminate!();
    }

    let ring = meta[META_COUNT] as u32;
    score_entries(
        pyramid,
        &mut scratch,
        0u32,
        ring,
        &block,
        width,
        height,
        blksize,
        cooperative_threads,
    );

    sync_cube();

    if block.thread_id == 0u32 {
        pick_refined(&scratch.list, &scratch.sads, &mut meta, &mut scores, values);
        write_result(
            &meta,
            &scores,
            mv_out,
            confidence_out,
            sad_out,
            changes_out,
            noise,
            values,
            &slots,
            write_confidence,
            write_sad,
            writes_changes,
        );
    }
}

/// Runs one truemotion pass, picking each block's vector from its candidates under a penalty
/// toward its neighbourhood's consensus.
///
/// Launch one cube per block per neighbour, with the neighbour on z. Each block scores the distinct
/// vectors [gather_candidates] lists from `mv_in`, `seeds` and `predictions`, one row of one vector
/// per thread at a time. Each costs its SAD plus a penalty per pixel of distance from the
/// component-wise median of the neighbouring blocks, as [PassValues] describes, and a tie keeps the
/// earlier vector. The block's own vector and the median are scored first, and when
/// [settle_early] shows one of them wins the rest are not scored. With a `refine_radius` above 0
/// and a best vector other than the block's own, the vectors within it of the best one are scored
/// too, and replace it only when cheaper.
///
/// With `writes_changes`, each block flags in `changes_out` whether it left its own vector. With
/// `reads_changes`, a block whose 3x3 neighbourhood flagged no change in `changes_in` keeps its own
/// vector unscored and leaves its confidence and SAD as they were, since it would pick the same
/// again. Both flag buffers hold one `u32` per block in the confidence layout.
///
/// Every frame is a full-resolution luma slot of `pyramid`, `slot_pixels` values apart. Neighbour
/// `n` reads the fields `n * mv_stride` values into `mv_in` and `seeds` and matches slot
/// `neighbour_slots[n]` against `centre_slot`. Its predictions are words `6 * n` onward of
/// `predictions`. It writes its vectors `n * mv_stride` values into `mv_out`. With
/// `write_confidence`, each block's confidence lands `n * conf_stride` values into
/// `confidence_out`, and with `write_sad`, its winning SAD lands at the same place in `sad_out`.
/// Unused inputs and outputs are never touched and can be placeholders.
///
/// `mv_in` and `mv_out` must be separate buffers, so the result does not depend on block order.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, value or comptime shape the kernel binds"
)]
pub fn mc_truemotion(
    pyramid: &[f32],
    neighbour_slots: &[u32],
    mv_in: &[i32],
    seeds: &[i32],
    predictions: &[f32],
    changes_in: &[u32],
    mv_out: &mut [i32],
    confidence_out: &mut [f32],
    sad_out: &mut [f32],
    changes_out: &mut [u32],
    noise: &[f32],
    centre_slot: u32,
    lambda_base: f32,
    lambda_noise: f32,
    surcharge: f32,
    fade_level: f32,
    thsad: f32,
    sad_field: u32,
    #[comptime] has_seeds: bool,
    #[comptime] model_candidate: bool,
    #[comptime] reads_floor: bool,
    #[comptime] write_confidence: bool,
    #[comptime] write_sad: bool,
    #[comptime] reads_changes: bool,
    #[comptime] writes_changes: bool,
    #[comptime] refine_radius: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] slot_pixels: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
    #[comptime] mv_stride: u32,
    #[comptime] conf_stride: u32,
) {
    let values = PassValues {
        lambda_base,
        lambda_noise,
        surcharge,
        fade_level,
        thsad,
        sad_field,
    };

    truemotion_block(
        pyramid,
        neighbour_slots,
        mv_in,
        seeds,
        predictions,
        changes_in,
        mv_out,
        confidence_out,
        sad_out,
        changes_out,
        noise,
        centre_slot,
        &values,
        has_seeds,
        model_candidate,
        reads_floor,
        write_confidence,
        write_sad,
        reads_changes,
        writes_changes,
        refine_radius,
        width,
        height,
        slot_pixels,
        blksize,
        step,
        blocks_x,
        blocks_y,
        mv_stride,
        conf_stride,
        0u32,
    );
}

use cubecl::prelude::*;
use cubecl::terminate;

use super::block_match::{
    BlockSite,
    coarse_window_best,
    fine_match,
    match_confidence,
    search_candidates,
    write_fine_result,
};
use super::candidates::{BlockVector, CandidateList, MODEL_CANDIDATE_BIAS, beats_incumbent};

/// Blocks scoring a confidence below this fail their first pass.
///
/// At 0.5 the SAD sits about 0.58 `thsad` past the noise floor.
pub const ESCALATE_BELOW_CONFIDENCE: f32 = 0.5;

/// Pixels a block's vector may sit from its closest grid neighbour's before the block fails.
///
/// The gap is the larger of the two axis differences. A block that agrees with any one of its
/// four neighbours passes, so the edge of a moving object does not fail on its own.
pub const NEIGHBOUR_DISAGREEMENT: i32 = 4;

/// The share of a search's SAD floor a wider search must win by.
///
/// It applies twice. An edge winner counts only when it beats every inner offset by this margin,
/// and a wider search replaces the kept result only when it beats it by this margin. On a flat noisy
/// block every offset scores near the floor, so a wider window wins by chance alone, and the margin
/// keeps those blocks where they are.
pub const ESCALATE_SAD_MARGIN: f32 = 0.125;

/// The threads in one failure-flag cube, one block each.
pub const FLAG_THREADS: u32 = 64;

/// The threads in one floor-measuring cube.
pub const FLOOR_THREADS: u32 = 256;

/// The most kept SADs the floor is measured from.
pub const FLOOR_SAMPLES: u32 = 4096;

/// The floor is the kept SAD one in this many counted blocks falls at or below.
pub const FLOOR_RANK_DIVISOR: u32 = 4;

/// A kept SAD at or below this is an exact match, which says nothing about the noise.
const EXACT_MATCH_SAD: f32 = 1.0e-6;

/// Blocks whose kept SAD is below this share of the median are near-flat and not ranked.
pub const FLAT_SHARE: f32 = 0.25;

/// The low bits of a kept SAD's bit pattern the floor measure drops, leaving 7 mantissa bits.
const FLOOR_BIN_SHIFT: u32 = 16;

/// The bin of positive infinity, above every finite positive float's.
const POSITIVE_INFINITY_BIN: u32 = 0x7F80;

/// The sections of the escalation state, each one `f32` per block in the confidence layout.
///
/// The kept SAD, the flag word, then the centre of the window the first pass's winner came from, on
/// x and y. A search writes its winner's edge bar to the flag word, and the flag kernel turns it
/// into 1 for a block that must search again and 0 otherwise. One SAD floor follows the sections.
pub const STATE_SECTIONS: u32 = 4;
const STATE_SAD: u32 = 0;
const STATE_FLAG: u32 = 1;
const STATE_CENTRE_X: u32 = 2;
const STATE_CENTRE_Y: u32 = 3;

/// Where the SAD floor sits in the escalation state, after the sections.
#[cube]
fn floor_slot(section: u32) -> usize {
    let slot = STATE_SECTIONS * section;
    slot as usize
}

/// The fine block match of [mc_block_match_fine](super::mc_block_match_fine) that also records what
/// escalation needs.
///
/// Every output matches [mc_block_match_fine](super::mc_block_match_fine) exactly. Each block also
/// writes its winning SAD, its winner's edge bar and the centre of the window it came from into
/// `state`, laid out as [STATE_SECTIONS] describes with sections `CUBE_COUNT_Z * conf_stride`
/// values apart.
#[cube(launch_unchecked)]
pub fn mc_block_match_fine_tracked(
    pyramid: &[f32],
    neighbour_slots: &[u32],
    mv_field: &mut [i32],
    confidence: &mut [f32],
    sad: &mut [f32],
    state: &mut [f32],
    predictions: &[f32],
    centre_slot: u32,
    #[comptime] write_confidence: bool,
    #[comptime] write_sad: bool,
    #[comptime] model_candidate: bool,
    noise: &[f32],
    sad_field: u32,
    thsad: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] slot_pixels: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] search_radius: u32,
    use_seed: u32,
    #[comptime] blocks_x: u32,
    #[comptime] mv_stride: u32,
    #[comptime] conf_stride: u32,
) {
    let found = fine_match(
        pyramid,
        neighbour_slots,
        mv_field,
        predictions,
        noise,
        centre_slot,
        model_candidate,
        write_sad,
        true,
        width,
        height,
        slot_pixels,
        blksize,
        step,
        search_radius,
        use_seed,
        blocks_x,
        mv_stride,
        conf_stride,
    );

    if found.thread_id != 0 {
        terminate!();
    }

    write_fine_result(
        mv_field,
        confidence,
        sad,
        noise,
        &found,
        write_confidence,
        write_sad,
        sad_field,
        thsad,
        conf_stride,
    );

    let section = CUBE_COUNT_Z * conf_stride;
    let block_slot = found.block_slot;
    state[(STATE_SAD * section + block_slot) as usize] = found.best.sad;
    state[(STATE_FLAG * section + block_slot) as usize] = found.best.edge_bar;
    state[(STATE_CENTRE_X * section + block_slot) as usize] = found.best.centre_dx as f32;
    state[(STATE_CENTRE_Y * section + block_slot) as usize] = found.best.centre_dy as f32;
}

/// Writes the launch's SAD floor into `state`.
///
/// With `reads_floor` set and a positive derived word of `noise` at `sad_field`, the floor is that
/// reading. Otherwise it is measured from the first pass's kept SADs. Two copies of the same content
/// differ by noise alone, so a block that found its match keeps a SAD near the floor.
///
/// The measure pools every neighbour's blocks and leaves out exact matches. It also leaves out
/// near-flat blocks, such as bars with faint compression noise, whose kept SAD is below
/// [FLAT_SHARE] of the median. The floor is the kept SAD that one in [FLOOR_RANK_DIVISOR] of the
/// remaining blocks fall at or below, so it holds while that share of the grainy blocks found their
/// match. A launch with no inexact match gets a floor of 0. Each of `neighbours` neighbours holds
/// `blocks` blocks.
///
/// The launch runs one cube of [FLOOR_THREADS] threads. It reads at most [FLOOR_SAMPLES] blocks,
/// spread evenly, then searches the top bits of their bit patterns, which sort as positive floats
/// do. Each rank is the lower edge of a bin about 1% wide, and the same on every run.
#[cube(launch_unchecked)]
pub fn mc_measure_floor(
    state: &mut [f32],
    noise: &[f32],
    sad_field: u32,
    neighbours: u32,
    #[comptime] reads_floor: bool,
    #[comptime] blocks: u32,
    #[comptime] conf_stride: u32,
) {
    let thread_id = UNIT_POS_X;
    let section = neighbours * conf_stride;
    let floor_index = floor_slot(section);

    if reads_floor {
        // Every thread reads the same word, so the whole cube exits together.
        let reading = noise[sad_field as usize];
        if reading > 0.0f32 {
            if thread_id == 0 {
                state[floor_index] = reading;
            }

            terminate!();
        }
    }

    let total = neighbours * blocks;
    let sample_step = total.div_ceil(FLOOR_SAMPLES);
    let sample_count = total.div_ceil(sample_step);

    let mut keys = Shared::<[u32]>::new_slice(FLOOR_SAMPLES as usize);
    let mut counts = Shared::<[u32]>::new_slice(FLOOR_THREADS as usize);
    let mut bounds = Shared::<[u32]>::new_slice(2usize);

    let mut sample = thread_id;
    while sample < FLOOR_SAMPLES {
        let flat = u32::min(sample, sample_count - 1) * sample_step;
        let neighbour_index = flat / blocks;
        let block = flat % blocks;
        let sad_slot = STATE_SAD * section + neighbour_index * conf_stride + block;
        let kept_sad = state[sad_slot as usize];
        let counted = sample < sample_count && kept_sad > EXACT_MATCH_SAD;
        let bin = u32::reinterpret(kept_sad) >> FLOOR_BIN_SHIFT;
        // An exact match or a spare slot gets bin 0, which every count leaves out.
        keys[sample as usize] = select(counted, bin, 0u32);
        sample += FLOOR_THREADS;
    }

    sync_cube();

    let no_limit = u32::MAX.runtime();
    let first_bin = 1u32.runtime();
    let valid = count_in_bins(&keys, &mut counts, first_bin, no_limit, thread_id);

    let median_bin = ranked_bin(&keys, &mut counts, &mut bounds, first_bin, 2u32, thread_id);
    let median_bits = median_bin << FLOOR_BIN_SHIFT;
    let median = f32::reinterpret(median_bits);
    let flat_limit = median * FLAT_SHARE;
    let flat_bin = u32::reinterpret(flat_limit) >> FLOOR_BIN_SHIFT;
    let grainy_bin = u32::max(flat_bin, 1u32);

    let floor_bin = ranked_bin(
        &keys,
        &mut counts,
        &mut bounds,
        grainy_bin,
        FLOOR_RANK_DIVISOR,
        thread_id,
    );

    if thread_id == 0 {
        let floor_bits = floor_bin << FLOOR_BIN_SHIFT;
        let ranked = f32::reinterpret(floor_bits);
        let floor = select(valid == 0u32, 0.0f32, ranked);
        state[floor_index] = floor;
    }
}

/// The lowest bin at or above `lower` that one in `divisor` of the `keys` from `lower` up fall at
/// or below.
///
/// It returns `lower` when no key reaches `lower`. Every thread of the cube must call it, and every
/// thread gets the same bin.
#[cube]
fn ranked_bin(
    keys: &Shared<[u32]>,
    counts: &mut Shared<[u32]>,
    bounds: &mut Shared<[u32]>,
    lower: u32,
    divisor: u32,
    thread_id: u32,
) -> u32 {
    let no_limit = u32::MAX.runtime();
    let valid = count_in_bins(keys, counts, lower, no_limit, thread_id);

    if thread_id == 0 {
        bounds[0] = lower;
        bounds[1] = POSITIVE_INFINITY_BIN;
    }

    sync_cube();

    let mut low = bounds[0];
    let mut high = bounds[1];
    while low < high {
        let middle = low + (high - low) / 2u32;
        let below = count_in_bins(keys, counts, lower, middle, thread_id);
        let holds_share = divisor * below >= valid;

        if thread_id == 0 {
            bounds[0] = select(holds_share, low, middle + 1u32);
            bounds[1] = select(holds_share, middle, high);
        }

        sync_cube();

        low = bounds[0];
        high = bounds[1];

        sync_cube();
    }

    low
}

/// How many `keys` lie between `lower` and `limit`, both included.
///
/// Every thread of the cube must call it, and every thread gets the same count.
#[cube]
fn count_in_bins(
    keys: &Shared<[u32]>,
    counts: &mut Shared<[u32]>,
    lower: u32,
    limit: u32,
    thread_id: u32,
) -> u32 {
    let mut count = 0u32;
    let mut sample = thread_id;

    while sample < FLOOR_SAMPLES {
        let key = keys[sample as usize];
        let within = key >= lower && key <= limit;
        count += select(within, 1u32, 0u32);
        sample += FLOOR_THREADS;
    }

    counts[thread_id as usize] = count;
    sync_cube();

    let half_threads = FLOOR_THREADS / 2u32;
    let mut stride = half_threads.runtime();
    while stride > 0u32 {
        if thread_id < stride {
            let pair_sum = counts[thread_id as usize] + counts[(thread_id + stride) as usize];
            counts[thread_id as usize] = pair_sum;
        }

        sync_cube();
        stride /= 2u32;
    }

    let total = counts[0];
    sync_cube();

    total
}

/// Marks each block that must search again in the flag section of `state`.
///
/// A block fails when its kept SAD scores a confidence below [ESCALATE_BELOW_CONFIDENCE], when its
/// winner beats its edge bar by more than [ESCALATE_SAD_MARGIN] times the floor, or when its vector
/// in `mv_field` sits more than [NEIGHBOUR_DISAGREEMENT] pixels from every grid neighbour's. The
/// confidence scores against the SAD floor in `state` and `thsad`, as the fine block match scores
/// against its noise floor.
///
/// The launch runs [FLAG_THREADS] blocks of a row per cube, with the row on y and the neighbour
/// on z. It reads `mv_field` only, so every block sees the same field.
#[cube(launch_unchecked)]
pub fn mc_flag_failures(
    mv_field: &[i32],
    state: &mut [f32],
    thsad: f32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
    #[comptime] mv_stride: u32,
    #[comptime] conf_stride: u32,
) {
    let block_col = ABSOLUTE_POS_X;
    if block_col >= blocks_x {
        terminate!();
    }

    let block_row = CUBE_POS_Y;
    let neighbour_index = CUBE_POS_Z;
    let section = CUBE_COUNT_Z * conf_stride;
    let block_slot = neighbour_index * conf_stride + block_row * blocks_x + block_col;
    let sad_slot = (STATE_SAD * section + block_slot) as usize;
    let flag_slot = (STATE_FLAG * section + block_slot) as usize;

    let floor_index = floor_slot(section);
    let sad_floor = state[floor_index];
    let kept_sad = state[sad_slot];
    let confidence = match_confidence(kept_sad, sad_floor, thsad);
    let low_confidence = confidence < ESCALATE_BELOW_CONFIDENCE;
    let margin = ESCALATE_SAD_MARGIN * sad_floor;
    let on_edge = beats_incumbent(kept_sad, state[flag_slot], margin);

    let mv_base = neighbour_index * mv_stride;
    let closest_gap = closest_neighbour_gap(mv_field, mv_base, block_col, block_row, blocks_x, blocks_y);
    let disagrees = closest_gap > NEIGHBOUR_DISAGREEMENT;

    let failed = low_confidence || on_edge || disagrees;
    state[flag_slot] = select(failed, 1.0f32, 0.0f32);
}

/// The smallest gap between a block's vector and any of its four grid neighbours'.
///
/// The gap is the larger of the two axis differences. A grid of one block has no neighbours and
/// returns 0.
#[cube]
fn closest_neighbour_gap(
    mv_field: &[i32],
    mv_base: u32,
    block_col: u32,
    block_row: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
) -> i32 {
    let own_slot = mv_base + (block_row * blocks_x + block_col) * 2;
    let own_dx = mv_field[own_slot as usize];
    let own_dy = mv_field[own_slot as usize + 1];

    // Larger than any gap, so a block without neighbours keeps it until the end.
    let no_neighbour = i32::MAX.runtime();
    let mut closest = no_neighbour;

    if block_col > 0 {
        let gap = vector_gap(mv_field, own_slot - 2, own_dx, own_dy);
        closest = i32::min(closest, gap);
    }

    if block_col + 1 < blocks_x {
        let gap = vector_gap(mv_field, own_slot + 2, own_dx, own_dy);
        closest = i32::min(closest, gap);
    }

    if block_row > 0 {
        let gap = vector_gap(mv_field, own_slot - 2 * blocks_x, own_dx, own_dy);
        closest = i32::min(closest, gap);
    }

    if block_row + 1 < blocks_y {
        let gap = vector_gap(mv_field, own_slot + 2 * blocks_x, own_dx, own_dy);
        closest = i32::min(closest, gap);
    }

    select(closest == no_neighbour, 0i32, closest)
}

/// The larger axis difference between the vector at `slot` and `(dx, dy)`.
#[cube]
fn vector_gap(mv_field: &[i32], slot: u32, dx: i32, dy: i32) -> i32 {
    let gap_x = i32::abs(mv_field[slot as usize] - dx);
    let gap_y = i32::abs(mv_field[slot as usize + 1] - dy);
    i32::max(gap_x, gap_y)
}

/// Searches each flagged block again with a window `search_radius` either side of its first pass's
/// window centre.
///
/// The window widens around the same centre as the first pass, so `search_radius` bounds the total
/// reach. With `widen_coarse` set, the block first searches the coarse level again at
/// `search_radius`, around no motion and, with `model_candidate`, the camera model's prediction, as
/// [mc_block_match_coarse](super::mc_block_match_coarse) does. The fine level then also searches
/// around that coarse winner unless it is the first window's centre. The coarse level of every
/// frame starts `level_offset` values in, `level_slot_pixels` apart, and `floor_scale` converts the
/// SAD floor from a full-resolution block to a coarse one.
///
/// Blocks whose flag word in `state` is 0 exit at once. A flagged block keeps the new winner only
/// when its SAD beats the kept SAD in `state` by more than [ESCALATE_SAD_MARGIN] times the SAD
/// floor in `state`, so ties and near-ties keep the earlier result. A kept winner replaces the
/// block's vector, its SAD in `sad` when `write_sad` is set, its confidence when `write_confidence`
/// is set, and its kept SAD in `state`. The window mean in `sad` stays the first pass's. The flag
/// word then holds the kept winner's edge bar, or 0 when nothing was kept.
///
/// Each block reads and writes only its own slots, so the launch order never changes the output.
/// Buffers, layouts and the confidence follow [mc_block_match_fine](super::mc_block_match_fine)
/// and [mc_block_match_fine_tracked].
#[cube(launch_unchecked)]
pub fn mc_escalate_search(
    pyramid: &[f32],
    neighbour_slots: &[u32],
    mv_field: &mut [i32],
    confidence: &mut [f32],
    sad: &mut [f32],
    state: &mut [f32],
    predictions: &[f32],
    centre_slot: u32,
    #[comptime] write_confidence: bool,
    #[comptime] write_sad: bool,
    #[comptime] model_candidate: bool,
    noise: &[f32],
    sad_field: u32,
    thsad: f32,
    level_offset: u32,
    level_slot_pixels: u32,
    floor_scale: f32,
    #[comptime] widen_coarse: bool,
    #[comptime] level_width: u32,
    #[comptime] level_height: u32,
    #[comptime] coarse_blksize: u32,
    #[comptime] level_scale: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] slot_pixels: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] search_radius: u32,
    #[comptime] blocks_x: u32,
    #[comptime] mv_stride: u32,
    #[comptime] conf_stride: u32,
) {
    let block_col = CUBE_POS_X;
    let block_row = CUBE_POS_Y;
    let neighbour_index = CUBE_POS_Z;

    let section = CUBE_COUNT_Z * conf_stride;
    let block_slot = neighbour_index * conf_stride + block_row * blocks_x + block_col;
    let flag_slot = (STATE_FLAG * section + block_slot) as usize;

    // Every thread reads the same word, so the whole cube exits together.
    if state[flag_slot] == 0.0f32 {
        terminate!();
    }

    let centre_dx = state[(STATE_CENTRE_X * section + block_slot) as usize] as i32;
    let centre_dy = state[(STATE_CENTRE_Y * section + block_slot) as usize] as i32;
    let floor_index = floor_slot(section);
    let sad_floor = state[floor_index];
    let margin = ESCALATE_SAD_MARGIN * sad_floor;
    let neighbour_slot = neighbour_slots[neighbour_index as usize];
    let thread_id = UNIT_POS_Y * CUBE_DIM_X + UNIT_POS_X;
    let threads = CUBE_DIM_X * CUBE_DIM_Y;

    let mut centres = CandidateList::new();
    centres.set(0u32, centre_dx, centre_dy);

    if widen_coarse {
        let coarse_site = BlockSite {
            centre_base: level_offset + centre_slot * level_slot_pixels,
            neighbour_base: level_offset + neighbour_slot * level_slot_pixels,
            origin_x: (block_col * step / level_scale) as i32,
            origin_y: (block_row * step / level_scale) as i32,
            thread_id,
            threads,
        };
        let coarse_bias = MODEL_CANDIDATE_BIAS * sad_floor * floor_scale;
        let coarse_best = coarse_search(
            pyramid,
            predictions,
            &coarse_site,
            neighbour_index,
            coarse_bias,
            model_candidate,
            level_width,
            level_height,
            coarse_blksize,
            level_scale,
            search_radius,
        );

        centres.set(1u32, coarse_best.dx, coarse_best.dy);
    }

    let site = BlockSite {
        centre_base: centre_slot * slot_pixels,
        neighbour_base: neighbour_slot * slot_pixels,
        origin_x: block_col as i32 * step as i32,
        origin_y: block_row as i32 * step as i32,
        thread_id,
        threads,
    };

    let count = comptime!(if widen_coarse { 2u32 } else { 1u32 });
    let best = search_candidates(
        pyramid,
        &centres,
        margin,
        &site,
        width,
        height,
        blksize,
        search_radius,
        count,
        false,
        true,
    );

    if site.thread_id != 0 {
        terminate!();
    }

    let sad_slot = (STATE_SAD * section + block_slot) as usize;
    let improves = beats_incumbent(best.sad, state[sad_slot], margin);

    if improves {
        let mv_slot = (neighbour_index * mv_stride + (block_row * blocks_x + block_col) * 2) as usize;
        mv_field[mv_slot] = best.dx;
        mv_field[mv_slot + 1] = best.dy;
        state[sad_slot] = best.sad;

        if write_sad {
            sad[block_slot as usize] = best.sad;
        }

        if write_confidence {
            let sad_noise_floor = noise[sad_field as usize];
            confidence[block_slot as usize] = match_confidence(best.sad, sad_noise_floor, thsad);
        }
    }

    state[flag_slot] = select(improves, best.edge_bar, 0.0f32);
}

/// Searches a block's coarse level around no motion and the camera model's prediction, and returns
/// the winner in full-resolution pixels to every thread.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the search reads"
)]
fn coarse_search(
    pyramid: &[f32],
    predictions: &[f32],
    site: &BlockSite,
    neighbour_index: u32,
    bias: f32,
    #[comptime] model_candidate: bool,
    #[comptime] level_width: u32,
    #[comptime] level_height: u32,
    #[comptime] blksize: u32,
    #[comptime] level_scale: u32,
    #[comptime] search_radius: u32,
) -> BlockVector {
    let best = coarse_window_best(
        pyramid,
        predictions,
        site,
        neighbour_index,
        bias,
        model_candidate,
        level_width,
        level_height,
        blksize,
        level_scale,
        search_radius,
    );

    // Only thread 0 holds the winner, so it hands it to the rest of the cube.
    let mut winner = Shared::<[i32]>::new_slice(2usize);
    if site.thread_id == 0 {
        winner[0] = best.dx * level_scale as i32;
        winner[1] = best.dy * level_scale as i32;
    }

    sync_cube();

    BlockVector {
        dx: winner[0],
        dy: winner[1],
    }
}

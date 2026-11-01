use cubecl::prelude::*;
use cubecl::terminate;

use super::block_match::{BlockSite, fine_match, match_confidence, search_candidates, write_fine_result};
use super::candidates::{CandidateList, beats_incumbent};
use crate::noise::DERIVED_SAD_FLOOR;

/// Blocks scoring a confidence below this fail their first pass.
///
/// At 0.5 the SAD sits about 0.58 `thsad` past the noise floor.
pub const ESCALATE_BELOW_CONFIDENCE: f32 = 0.5;

/// Pixels a block's vector may sit from its closest grid neighbour's before the block fails.
///
/// The gap is the larger of the two axis differences. A block that agrees with any one of its
/// four neighbours passes, so the edge of a moving object does not fail on its own.
pub const NEIGHBOUR_DISAGREEMENT: i32 = 4;

/// The share of the noise SAD floor a wider search must win by.
///
/// It applies twice. An edge winner counts only when it beats every inner offset by this margin,
/// and a wider search replaces the kept result only when it beats it by this margin. On a flat noisy
/// block every offset scores near the floor, so a wider window wins by chance alone, and the margin
/// keeps those blocks where they are.
pub const ESCALATE_SAD_MARGIN: f32 = 0.125;

/// The threads in one failure-flag cube, one block each.
pub const FLAG_THREADS: u32 = 64;

/// The sections of the escalation state, each one `f32` per block in the confidence layout.
///
/// The kept SAD, the flag word, then the centre of the window the first pass's winner came from, on
/// x and y. The flag word holds 1 when the search's winner sat on its window's edge, and the flag
/// kernel turns it into 1 for a block that must search again.
pub const STATE_SECTIONS: u32 = 4;
const STATE_SAD: u32 = 0;
const STATE_FLAG: u32 = 1;
const STATE_CENTRE_X: u32 = 2;
const STATE_CENTRE_Y: u32 = 3;

/// The fine block match of [mc_block_match_fine](super::mc_block_match_fine) that also records what
/// escalation needs.
///
/// Every output matches [mc_block_match_fine](super::mc_block_match_fine) exactly. Each block also
/// writes its winning SAD, whether the winner sat on its window's edge, and the centre of the window
/// it came from into `state`, laid out as [STATE_SECTIONS] describes with sections
/// `CUBE_COUNT_Z * conf_stride` values apart. An edge winner must beat the window's inner offsets
/// by [ESCALATE_SAD_MARGIN] times the derived SAD floor of `noise`.
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
    let edge_margin = ESCALATE_SAD_MARGIN * noise[DERIVED_SAD_FLOOR as usize];
    let found = fine_match(
        pyramid,
        neighbour_slots,
        mv_field,
        predictions,
        noise,
        centre_slot,
        edge_margin,
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
    let edge_word = select(found.best.edge, 1.0f32, 0.0f32);
    state[(STATE_SAD * section + block_slot) as usize] = found.best.sad;
    state[(STATE_FLAG * section + block_slot) as usize] = edge_word;
    state[(STATE_CENTRE_X * section + block_slot) as usize] = found.best.centre_dx as f32;
    state[(STATE_CENTRE_Y * section + block_slot) as usize] = found.best.centre_dy as f32;
}

/// Marks each block that must search again in the flag section of `state`.
///
/// A block fails when its kept SAD scores a confidence below [ESCALATE_BELOW_CONFIDENCE], when its
/// winner sat on its window's edge, or when its vector in `mv_field` sits more than
/// [NEIGHBOUR_DISAGREEMENT] pixels from every grid neighbour's. The confidence uses the derived
/// word of `noise` at index `sad_field` and `thsad`, as the fine block match does.
///
/// The launch runs [FLAG_THREADS] blocks of a row per cube, with the row on y and the neighbour
/// on z. It reads `mv_field` only, so every block sees the same field.
#[cube(launch_unchecked)]
pub fn mc_flag_failures(
    mv_field: &[i32],
    state: &mut [f32],
    noise: &[f32],
    sad_field: u32,
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

    let sad_noise_floor = noise[sad_field as usize];
    let confidence = match_confidence(state[sad_slot], sad_noise_floor, thsad);
    let low_confidence = confidence < ESCALATE_BELOW_CONFIDENCE;
    let on_edge = state[flag_slot] != 0.0f32;

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
/// reach. Blocks whose flag word in `state` is 0 exit at once. A flagged block keeps the new winner
/// only when its SAD beats the kept SAD in `state` by more than [ESCALATE_SAD_MARGIN] times the
/// derived SAD floor of `noise`, so ties and near-ties keep the earlier result. A kept winner
/// replaces the block's vector, its SAD and window mean in `sad` when `write_sad` is set, its
/// confidence when `write_confidence` is set, and its kept SAD in `state`. The flag word is then
/// left at 1 only for a kept winner on the new window's edge, tested with the same margin.
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
    centre_slot: u32,
    #[comptime] write_confidence: bool,
    #[comptime] write_sad: bool,
    noise: &[f32],
    sad_field: u32,
    thsad: f32,
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
    let margin = ESCALATE_SAD_MARGIN * noise[DERIVED_SAD_FLOOR as usize];
    let neighbour_slot = neighbour_slots[neighbour_index as usize];

    let site = BlockSite {
        centre_base: centre_slot * slot_pixels,
        neighbour_base: neighbour_slot * slot_pixels,
        origin_x: block_col as i32 * step as i32,
        origin_y: block_row as i32 * step as i32,
        thread_id: UNIT_POS_Y * CUBE_DIM_X + UNIT_POS_X,
        threads: CUBE_DIM_X * CUBE_DIM_Y,
    };

    let mut centres = CandidateList::new();
    centres.set(0u32, centre_dx, centre_dy);

    let best = search_candidates(
        pyramid,
        &centres,
        0.0f32,
        margin,
        &site,
        width,
        height,
        blksize,
        search_radius,
        1u32,
        write_sad,
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
            sad[(section + block_slot) as usize] = best.mean;
        }

        if write_confidence {
            let sad_noise_floor = noise[sad_field as usize];
            confidence[block_slot as usize] = match_confidence(best.sad, sad_noise_floor, thsad);
        }
    }

    let still_on_edge = improves && best.edge;
    state[flag_slot] = select(still_on_edge, 1.0f32, 0.0f32);
}

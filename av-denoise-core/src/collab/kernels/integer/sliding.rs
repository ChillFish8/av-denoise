use cubecl::prelude::*;

use super::variant::IntegerVariant;
use super::{centre_group_size, centre_window_positions};
use crate::collab::geometry::refs_along;
use crate::collab::kernels::fused::MEMBER_WORDS;
use crate::collab::kernels::group::{clamp_top_left, pack_pos_t};
use crate::collab::kernels::plane_ops::{SELF_MATCH_SSD, UNFILLED_SSD, ranks_before_int};
use crate::collab::{MAX_K, PATCH_SIZE, STEP};

/// Pixels along one side of a block of squared differences, the reference step.
pub(crate) const SLIDING_BLOCK: u32 = STEP;

// A patch is two blocks along each side, so a reference adds four block sums.
const _: () = assert!(
    PATCH_SIZE == 2 * SLIDING_BLOCK,
    "a sliding reference patch must be two blocks across"
);

/// The top-left of block `index` along one axis of a cube whose first reference is `first`.
///
/// A block sits at its reference's corner. A block past the last reference sits one block after the
/// last reference's corner.
#[cube]
fn block_origin(first: u32, index: u32, #[comptime] frame_refs: u32, #[comptime] max_pos: u32) -> u32 {
    let reference = first + index;
    let clamped = u32::min(reference, frame_refs - 1u32);
    let corner = u32::min(clamped * STEP, max_pos);
    let past_end = reference >= frame_refs;

    select(past_end, corner + SLIDING_BLOCK, corner)
}

/// Channel `c` of the code at `(x, y)` in frame `slot`, with the position clamped into the frame.
#[cube]
fn clamped_code<C: Int, N: Size>(
    codes: &Array<Vector<C, N>>,
    x: i32,
    y: i32,
    slot: u32,
    #[comptime] c: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) -> i32 {
    let last_x = comptime!(width as i32 - 1);
    let last_y = comptime!(height as i32 - 1);
    let inside_x = i32::max(x, 0i32);
    let inside_y = i32::max(y, 0i32);
    let column = i32::min(inside_x, last_x) as u32;
    let row = i32::min(inside_y, last_y) as u32;
    let index = (slot * height + row) * width + column;
    let pixel = codes[index as usize];

    i32::cast_from(pixel[c as usize])
}

/// The block at `(corner_x, corner_y)` of frame `slot`, every summed channel, row by row.
#[cube]
fn load_block<C: Int, N: Size>(
    codes: &Array<Vector<C, N>>,
    corner_x: u32,
    corner_y: u32,
    slot: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
) -> Array<i32> {
    let mut block = Array::<i32>::new(comptime!(SLIDING_BLOCK * SLIDING_BLOCK * channels) as usize);

    #[unroll]
    for py in 0..SLIDING_BLOCK {
        #[unroll]
        for px in 0..SLIDING_BLOCK {
            #[unroll]
            for c in 0..channels {
                let at = comptime!((py * SLIDING_BLOCK + px) * channels + c);
                let x = (corner_x + px) as i32;
                let y = (corner_y + py) as i32;
                block[at as usize] = clamped_code(codes, x, y, slot, c, width, height);
            }
        }
    }

    block
}

/// The SSD from `block` to the block at `(x, y)` of frame `slot`.
#[cube]
fn block_ssd<C: Int, N: Size>(
    codes: &Array<Vector<C, N>>,
    block: &Array<i32>,
    x: i32,
    y: i32,
    slot: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
) -> i32 {
    let mut sum = 0i32;

    #[unroll]
    for py in 0..SLIDING_BLOCK {
        #[unroll]
        for px in 0..SLIDING_BLOCK {
            #[unroll]
            for c in 0..channels {
                let at = comptime!((py * SLIDING_BLOCK + px) * channels + c);
                let candidate_x = x + px as i32;
                let candidate_y = y + py as i32;
                let code = clamped_code(codes, candidate_x, candidate_y, slot, c, width, height);
                let diff = block[at as usize] - code;
                sum += diff * diff;
            }
        }
    }

    sum
}

/// The SSD from the patch at `(ref_x, ref_y)` to the patch at `(x, y)` of frame `slot`, read
/// straight from the code ring.
#[cube]
fn direct_ssd<C: Int, N: Size>(
    codes: &Array<Vector<C, N>>,
    ref_x: u32,
    ref_y: u32,
    x: i32,
    y: i32,
    slot: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
) -> i32 {
    let mut sum = 0i32;

    for py in 0..PATCH_SIZE {
        for px in 0..PATCH_SIZE {
            let reference_x = (ref_x + px) as i32;
            let reference_y = (ref_y + py) as i32;
            let candidate_x = x + px as i32;
            let candidate_y = y + py as i32;

            #[unroll]
            for c in 0..channels {
                let reference = clamped_code(codes, reference_x, reference_y, slot, c, width, height);
                let candidate = clamped_code(codes, candidate_x, candidate_y, slot, c, width, height);
                let diff = reference - candidate;
                sum += diff * diff;
            }
        }
    }

    sum
}

/// Inserts `(ssd, packed)` into the lane's top-8, held in registers in
/// [ranks_before_int](crate::collab::kernels::plane_ops::ranks_before_int) order.
#[cube]
fn insert_ranked8(best_ssd: &mut Array<i32>, best_pos: &mut Array<u32>, ssd: i32, packed: u32) {
    let mut carry_ssd = ssd;
    let mut carry_pos = packed;

    #[unroll]
    for m in 0..MAX_K {
        let held_ssd = best_ssd[m as usize];
        let held_pos = best_pos[m as usize];
        let before = ranks_before_int(carry_ssd, carry_pos, held_ssd, held_pos);
        best_ssd[m as usize] = select(before, carry_ssd, held_ssd);
        best_pos[m as usize] = select(before, carry_pos, held_pos);
        carry_ssd = select(before, held_ssd, carry_ssd);
        carry_pos = select(before, held_pos, carry_pos);
    }
}

/// The integer collab centre search, offset by offset over a block of references.
///
/// A cube owns `variant.cube_refs_x` by `variant.ref_rows` references. Lane `r` keeps reference
/// `r`'s best eight positions in registers, ranked by SSD and then position, with the self-match
/// first. The offsets run row by row. For each offset every lane scores one 4x4 block of squared
/// differences under the cube's references, and each reference adds its four blocks from shared
/// memory. A reference whose blocks do not tile its patch, at a ragged frame edge, scores its
/// patch directly instead.
///
/// It writes each live reference's eight positions and its group word to `members`, in
/// [collab_search_int](crate::collab::kernels::integer::collab_search_int)'s format. The group
/// word is the centre group's size, with no grid flag.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub fn collab_centre_sliding<C: Int, N: Size>(
    codes: &Array<Vector<C, N>>,
    members: &mut Array<u32>,
    centre_slot: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] k_max: u32,
    #[comptime] spatial_radius: u32,
    #[comptime] frame_refs_x: u32,
    #[comptime] variant: IntegerVariant,
) {
    let frame_refs_y = comptime!(refs_along(height));
    let max_x = comptime!(width - PATCH_SIZE);
    let max_y = comptime!(height - PATCH_SIZE);
    let cube_refs_x = comptime!(variant.cube_refs_x);
    let cube_refs = comptime!(variant.cube_refs_x * variant.ref_rows);
    let block_columns = comptime!(variant.cube_refs_x + 1);
    let block_count = comptime!(variant.sliding_blocks());
    let span = comptime!(2 * spatial_radius + 1);
    let radius = comptime!(spatial_radius as i32);

    let thread = UNIT_POS_X;
    let first_x = CUBE_POS_X * cube_refs_x;
    let first_y = CUBE_POS_Y * comptime!(variant.ref_rows);

    // A lane past the cube's references follows the last one, and writes nothing.
    let owned = u32::min(thread, cube_refs - 1u32);
    let own_column = owned % cube_refs_x;
    let own_row = owned / cube_refs_x;
    let ref_x_index = first_x + own_column;
    let ref_y_index = first_y + own_row;
    let in_frame = ref_x_index < frame_refs_x && ref_y_index < frame_refs_y;
    let live = thread < cube_refs && in_frame;
    let ref_x_clamped = u32::min(ref_x_index, frame_refs_x - 1u32);
    let ref_y_clamped = u32::min(ref_y_index, frame_refs_y - 1u32);
    let ref_x = u32::min(ref_x_clamped * STEP, max_x);
    let ref_y = u32::min(ref_y_clamped * STEP, max_y);

    let left_corner = block_origin(first_x, own_column, frame_refs_x, max_x);
    let right_corner = block_origin(first_x, own_column + 1u32, frame_refs_x, max_x);
    let top_corner = block_origin(first_y, own_row, frame_refs_y, max_y);
    let bottom_corner = block_origin(first_y, own_row + 1u32, frame_refs_y, max_y);
    let tiles_x = right_corner == left_corner + SLIDING_BLOCK && left_corner == ref_x;
    let tiles_y = bottom_corner == top_corner + SLIDING_BLOCK && top_corner == ref_y;
    let tiles = tiles_x && tiles_y;

    // The lane's block of the cube, which stays fixed while the offsets move its candidate.
    let block_live = thread < block_count;
    let block_index = u32::min(thread, block_count - 1u32);
    let block_column = block_index % block_columns;
    let block_row = block_index / block_columns;
    let corner_x = block_origin(first_x, block_column, frame_refs_x, max_x);
    let corner_y = block_origin(first_y, block_row, frame_refs_y, max_y);
    let reference = load_block(codes, corner_x, corner_y, centre_slot, width, height, channels);

    let mut sums = SharedMemory::<i32>::new(comptime!(2 * block_count) as usize);
    let top_left = own_row * block_columns + own_column;
    let top_right = top_left + 1u32;
    let bottom_left = top_left + block_columns;
    let bottom_right = bottom_left + 1u32;

    let mut best_ssd = Array::<i32>::new(MAX_K as usize);
    let mut best_pos = Array::<u32>::new(MAX_K as usize);
    let last = comptime!(MAX_K - 1) as usize;

    #[unroll]
    for m in 0..MAX_K {
        best_ssd[m as usize] = UNFILLED_SSD;
        best_pos[m as usize] = 0u32;
    }

    // The block sums alternate between two halves of `sums`, so one barrier per offset suffices.
    let mut half = 0u32;

    for offset_y in 0..span {
        let dy = offset_y as i32 - radius;

        for offset_x in 0..span {
            let dx = offset_x as i32 - radius;
            let block_x = corner_x as i32 + dx;
            let block_y = corner_y as i32 + dy;
            let block_sum = block_ssd(
                codes,
                &reference,
                block_x,
                block_y,
                centre_slot,
                width,
                height,
                channels,
            );

            if block_live {
                sums[(half + block_index) as usize] = block_sum;
            }

            sync_cube();

            let upper = sums[(half + top_left) as usize] + sums[(half + top_right) as usize];
            let lower = sums[(half + bottom_left) as usize] + sums[(half + bottom_right) as usize];
            let candidate_x = ref_x as i32 + dx;
            let candidate_y = ref_y as i32 + dy;

            let mut ssd = upper + lower;
            if !tiles {
                ssd = direct_ssd(
                    codes,
                    ref_x,
                    ref_y,
                    candidate_x,
                    candidate_y,
                    centre_slot,
                    width,
                    height,
                    channels,
                );
            }

            let inside_x = candidate_x >= 0i32 && candidate_x <= max_x as i32;
            let inside_y = candidate_y >= 0i32 && candidate_y <= max_y as i32;
            let inside = inside_x && inside_y;
            let is_self = dx == 0i32 && dy == 0i32;
            let masked = select(inside, ssd, UNFILLED_SSD);
            let pinned = select(is_self, SELF_MATCH_SSD, masked);
            let packed_x = clamp_top_left(candidate_x, max_x);
            let packed_y = clamp_top_left(candidate_y, max_y);
            let packed = pack_pos_t(packed_x, packed_y, 0u32);

            let places = ranks_before_int(pinned, packed, best_ssd[last], best_pos[last]);
            if places {
                insert_ranked8(&mut best_ssd, &mut best_pos, pinned, packed);
            }

            half = block_count - half;
        }
    }

    let n_live = centre_window_positions(ref_x, ref_y, spatial_radius, max_x, max_y);
    let k_use = centre_group_size(n_live, k_max);

    if live {
        let ref_idx = ref_y_clamped * frame_refs_x + ref_x_clamped;
        let first_word = ref_idx * MEMBER_WORDS;

        #[unroll]
        for m in 0..MAX_K {
            members[(first_word + m) as usize] = best_pos[m as usize];
        }

        members[(first_word + MAX_K) as usize] = k_use;
    }
}

use cubecl::prelude::*;

use super::{IntShape, centre_window_positions, int_group_members};
use crate::collab::geometry::refs_along;
use crate::collab::kernels::fused::{MEMBER_WORDS, Walk};
use crate::collab::kernels::plane_ops::group_base;
use crate::collab::{MAX_K, PATCH_SIZE, STEP};

/// References one neighbour cube owns, one 8-lane group each.
pub(crate) const NEIGHBOUR_GROUPS: u32 = 8;

/// The integer collab neighbour search.
///
/// Each 8-lane group reads its reference's eight centre positions back from `members`, builds the
/// volume grid with [int_group_members](crate::collab::kernels::integer::int_group_members) and
/// overwrites the reference's words. A group reads and writes only its own reference's words, so
/// the pass runs in place. A group past the last reference reads nothing, builds from position
/// zero and writes nothing.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub fn collab_neighbour_int<C: Int, N: Size>(
    codes: &Array<Vector<C, N>>,
    mv_field: &Array<i32>,
    confidence: &Array<f32>,
    neighbour_slots: &Array<u32>,
    members: &mut Array<u32>,
    centre_slot: u32,
    c_min: f32,
    #[comptime] radius: u32,
    #[comptime] grid_frames: u32,
    #[comptime] refine: u32,
    #[comptime] mv_stride: u32,
    #[comptime] conf_stride: u32,
    #[comptime] blk_step: u32,
    #[comptime] blksize: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] k_max: u32,
    #[comptime] stored_ch: u32,
    #[comptime] spatial_radius: u32,
    #[comptime] frame_refs_x: u32,
    #[comptime] walk: Walk,
) {
    let group = UNIT_POS_X / 8u32;
    let sub = UNIT_POS_X % 8u32;
    let base = group_base();
    let frame_refs_y = comptime!(refs_along(height));
    let max_x = comptime!(width - PATCH_SIZE);
    let max_y = comptime!(height - PATCH_SIZE);
    let shape = comptime!(IntShape {
        spatial_radius,
        width,
        height,
        channels,
        stored_ch,
    });

    let ref_x_index = CUBE_POS_X * NEIGHBOUR_GROUPS + group;
    let ref_y_index = CUBE_POS_Y;
    let live = ref_x_index < frame_refs_x && ref_y_index < frame_refs_y;
    let ref_x_clamped = u32::min(ref_x_index, frame_refs_x - 1u32);
    let ref_y_clamped = u32::min(ref_y_index, frame_refs_y - 1u32);
    let ref_x = u32::min(ref_x_clamped * STEP, max_x);
    let ref_y = u32::min(ref_y_clamped * STEP, max_y);

    let ref_idx = ref_y_clamped * frame_refs_x + ref_x_clamped;
    let first_word = ref_idx * MEMBER_WORDS;

    // The branch holds no barrier or shuffle, so every lane still reaches the group's shuffles.
    let mut best_pos = 0u32;
    if live {
        best_pos = members[(first_word + sub) as usize];
    }

    let n_live = centre_window_positions(ref_x, ref_y, spatial_radius, max_x, max_y);

    let mut member_pos = Array::<u32>::new(MAX_K as usize);
    let group_word = int_group_members(
        codes,
        mv_field,
        confidence,
        neighbour_slots,
        best_pos,
        n_live,
        centre_slot,
        sub,
        base,
        c_min,
        &mut member_pos,
        walk,
        radius,
        grid_frames,
        refine,
        mv_stride,
        conf_stride,
        blk_step,
        blksize,
        blocks_x,
        blocks_y,
        k_max,
        shape,
    );

    // Every lane holds every member, so lane `m` writes member `m` and the row of words is
    // coalesced.
    if live {
        #[unroll]
        for m in 0..MAX_K {
            if sub == m {
                members[(first_word + m) as usize] = member_pos[m as usize];
            }
        }

        if sub == 0u32 {
            members[(first_word + MAX_K) as usize] = group_word;
        }
    }
}

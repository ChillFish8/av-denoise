use cubecl::prelude::*;

use super::shape::{SearchCursor, SearchShape};
use super::tile::load_centre_tile;
use super::variant::{Centre, CollabVariant, TRANSPOSE_REGION_FLOATS, tile_height, tile_width};
use super::{
    GROUP_SIZE_MASK,
    GROUP_USES_GRID,
    column_luma_of,
    filter_group,
    load_reference_column,
    search_group,
};
use crate::collab::kernels::group::clamp_top_left;
use crate::collab::kernels::plane_ops::group_base;
use crate::collab::{MAX_K, PATCH_SIZE, STEP};
use crate::nlmeans::kernels::helpers::channel_scale;

/// Words per reference in the member buffer, the eight members and the group word.
pub const MEMBER_WORDS: u32 = MAX_K + 1;

/// The search half of [collab_fused](crate::collab::kernels::fused::collab_fused).
///
/// Reference `r` writes words `r * MEMBER_WORDS..(r + 1) * MEMBER_WORDS` of `members`. These are
/// its eight members as `pack_pos_t` words, then the group word, the member count with
/// [GROUP_USES_GRID](crate::collab::kernels::fused::GROUP_USES_GRID) set when the group is a volume
/// grid. A dead group past the end of a row searches the last real reference and writes nothing.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub fn collab_search<S: Float, N: Size>(
    ring: &Array<Vector<f32, N>>,
    search_ring: &Array<Vector<S, N>>,
    mv_field: &Array<i32>,
    confidence: &Array<f32>,
    neighbour_slots: &Array<u32>,
    members: &mut Array<u32>,
    centre_slot: u32,
    c_min: f32,
    #[comptime] f16_search: bool,
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
    #[comptime] refs_x: u32,
    #[comptime] variant: CollabVariant,
) {
    let thread_id = UNIT_POS_X;
    let group = thread_id / 8u32;
    let sub = thread_id % 8u32;
    let base = group_base();
    let groups = comptime!(variant.groups);

    let max_x = comptime!(width - PATCH_SIZE);
    let max_y = comptime!(height - PATCH_SIZE);
    let shape = comptime!(SearchShape {
        f16_search,
        spatial_radius,
        width,
        height,
        channels,
        stored_ch,
    });

    let centre = comptime!(variant.centre);
    let lanes = comptime!(variant.lanes());
    let tile_len = comptime!(variant.tile_elements(spatial_radius, stored_ch));
    let mut tile = SharedMemory::<S>::new(tile_len as usize);

    let ref_x_index = CUBE_POS_X * groups + group;
    let live = ref_x_index < refs_x;
    let ref_x_clamped = ref_x_index.min(refs_x - 1u32);

    let ref_x = (ref_x_clamped * STEP).min(max_x);
    let ref_y = (CUBE_POS_Y * STEP).min(max_y);

    // The tile's corner is the first reference's search window corner, so every window of the
    // cube lies inside it.
    let first_ref_x = (CUBE_POS_X * groups * STEP).min(max_x);
    let tile_x0 = clamp_top_left(first_ref_x as i32 - spatial_radius as i32, max_x);
    let tile_y0 = clamp_top_left(ref_y as i32 - spatial_radius as i32, max_y);

    if comptime!(centre == Centre::Tiled) {
        let tile_w = comptime!(tile_width(groups, spatial_radius));
        let tile_h = comptime!(tile_height(spatial_radius));
        load_centre_tile(
            ring,
            search_ring,
            &mut tile,
            tile_x0,
            tile_y0,
            centre_slot,
            tile_w,
            tile_h,
            lanes,
            shape,
        );
        sync_cube();
    }

    let current = load_reference_column(ring, ref_x, ref_y, centre_slot, sub, width, height, channels);
    let scale = channel_scale(channels);

    let cursor = SearchCursor {
        rx: ref_x,
        ry: ref_y,
        tile_x0,
        tile_y0,
        centre_slot,
        sub,
        base,
        scale,
    };
    let mut member_pos = Array::<u32>::new(MAX_K as usize);
    let group_word = search_group(
        ring,
        search_ring,
        &tile,
        mv_field,
        confidence,
        neighbour_slots,
        &current,
        &cursor,
        c_min,
        &mut member_pos,
        variant,
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

    let ref_idx = CUBE_POS_Y * refs_x + ref_x_clamped;
    let first_word = ref_idx * MEMBER_WORDS;

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

/// The filter half of [collab_fused](crate::collab::kernels::fused::collab_fused), reading each
/// group from the `members` that [collab_search] wrote.
///
/// A cube of `8 * groups` threads owns `groups` reference patches. A dead group reads the last
/// real reference's group, the one it would have searched, and writes nothing.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub fn collab_filter<N: Size>(
    ring: &Array<Vector<f32, N>>,
    ring_scalar: &Array<f32>,
    members: &Array<u32>,
    neighbour_slots: &Array<u32>,
    sigma: &Array<f32>,
    noise_curve: &Array<f32>,
    strength_map: &Array<f32>,
    dct_profile: &Array<f32>,
    kaiser: &Array<f32>,
    accum: &mut Array<Atomic<i32>>,
    wsum: &mut Array<Atomic<i32>>,
    group_weight: &mut Array<f32>,
    centre_slot: u32,
    lambda_ht: f32,
    curve_valid: u32,
    map_mode: u32,
    weight_scale: f32,
    accum_scale: f32,
    pool_ratio: f32,
    #[comptime] grid_frames: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
    #[comptime] refs_x: u32,
    #[comptime] map_cols: u32,
    #[comptime] map_rows: u32,
    #[comptime] k_max: u32,
    #[comptime] pooled: bool,
    #[comptime] groups: u32,
) {
    let thread_id = UNIT_POS_X;
    let group = thread_id / 8u32;
    let sub = thread_id % 8u32;

    let max_x = comptime!(width - PATCH_SIZE);
    let max_y = comptime!(height - PATCH_SIZE);

    let transpose_floats = comptime!(groups * TRANSPOSE_REGION_FLOATS);
    let mut transpose_buf = SharedMemory::<f32>::new(transpose_floats as usize);

    let ref_x_index = CUBE_POS_X * groups + group;
    let live = ref_x_index < refs_x;
    let ref_x_clamped = ref_x_index.min(refs_x - 1u32);

    let ref_x = (ref_x_clamped * STEP).min(max_x);
    let ref_y = (CUBE_POS_Y * STEP).min(max_y);
    let ref_idx = CUBE_POS_Y * refs_x + ref_x_clamped;
    let first_word = ref_idx * MEMBER_WORDS;

    let mut member_pos = Array::<u32>::new(MAX_K as usize);
    #[unroll]
    for m in 0..MAX_K {
        member_pos[m as usize] = members[(first_word + m) as usize];
    }

    let group_word = members[(first_word + MAX_K) as usize];
    // A group never holds more than `k_max` members, but the bound tells the shader compiler so.
    // It then drops the same unused stack levels it drops in `collab_fused`, which keeps the
    // float code, and so the output, bit identical on RADV.
    let stored_k_use = group_word & GROUP_SIZE_MASK;
    let k_use = u32::min(stored_k_use, k_max);
    let use_grid = (group_word & GROUP_USES_GRID) != 0u32;

    let current = load_reference_column(ring, ref_x, ref_y, centre_slot, sub, width, height, channels);
    let column_luma = column_luma_of(&current, channels);

    filter_group(
        ring_scalar,
        neighbour_slots,
        sigma,
        noise_curve,
        strength_map,
        dct_profile,
        kaiser,
        accum,
        wsum,
        group_weight,
        &mut transpose_buf,
        &member_pos,
        k_use,
        use_grid,
        column_luma,
        ref_x,
        ref_y,
        ref_idx,
        live,
        centre_slot,
        sub,
        group,
        lambda_ht,
        curve_valid,
        map_mode,
        weight_scale,
        accum_scale,
        pool_ratio,
        grid_frames,
        width,
        height,
        channels,
        stored_ch,
        map_cols,
        map_rows,
        pooled,
    );
}

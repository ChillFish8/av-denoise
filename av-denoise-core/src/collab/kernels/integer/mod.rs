pub(crate) mod matrix;
pub(crate) mod scalar;
pub(crate) mod variant;

use cubecl::prelude::*;

use self::matrix::{MatrixRoute, matrix_centre_search};
use self::scalar::{int_centre_search, int_trajectory_search, load_code_column, load_int_centre_tile};
pub use self::variant::{INTEGER_VARIANT_COUNT, INTEGER_VARIANTS, IntCentre, IntNeighbour, IntegerVariant};
use super::fused::{GROUP_USES_GRID, MEMBER_WORDS, tile_height, tile_width};
use super::group::clamp_top_left;
use super::plane_ops::{UNFILLED_SSD, group_base};
use crate::collab::geometry::refs_along;
use crate::collab::{MAX_K, PATCH_SIZE, STEP};
use crate::tune::Int8Layout;

/// The comptime frame and patch shape the integer searches read.
///
/// `spatial_radius` is the centre search's reach around the reference. `stored_ch` is the lanes
/// each pixel stores, which can be more than the `channels` the SSD sums.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct IntShape {
    pub(crate) spatial_radius: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) channels: u32,
    pub(crate) stored_ch: u32,
}

/// One lane's position in its group's integer centre search.
///
/// `(rx, ry)` is the reference's top-left and `(tile_x0, tile_y0)` the centre tile's. `sub` is the
/// lane's column in its group and `base` the group's first lane in the plane.
#[derive(CubeType, Clone, Copy)]
pub(crate) struct IntCursor {
    pub(crate) rx: u32,
    pub(crate) ry: u32,
    pub(crate) tile_x0: u32,
    pub(crate) tile_y0: u32,
    pub(crate) centre_slot: u32,
    pub(crate) sub: u32,
    pub(crate) base: u32,
}

/// The collab search over the integer code ring, with exact i32 SSDs.
///
/// It writes `members` in [collab_search](crate::collab::kernels::fused::collab_search)'s format.
/// Reference `r` owns words `r * MEMBER_WORDS..(r + 1) * MEMBER_WORDS`, its eight `pack_pos_t`
/// members and then the group word. Candidates rank by SSD and then by position, so the members do
/// not depend on the order a walk visits them in. The self-match always holds slot 0.
///
/// A cube of `variant.lanes()` threads owns `variant.cube_refs_x` references along each of
/// `variant.ref_rows` rows, one 8-lane group each. `frame_refs_x` is the reference count along a
/// row. A group past the last reference of its row or column searches a clamped copy of the last
/// real one and writes nothing.
///
/// `codes` holds one integer code per stored lane, laid out as the f32 ring is. A tiled centre
/// first copies the union of the cube's search windows into a shared tile and scores from it. A
/// matrix centre scores that union on the matrix units, with `E` operands and `A` accumulators.
/// The f16 centre runs at `matrix_plane` and splits codes at `split_shift`, and the int8 centre
/// runs at `int8_plane` and splits codes as `int8` does. The other centres read none of these.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub fn collab_search_int<C: Int, N: Size, E: Numeric, A: Numeric>(
    codes: &Array<Vector<C, N>>,
    mv_field: &Array<i32>,
    confidence: &Array<f32>,
    neighbour_slots: &Array<u32>,
    members: &mut Array<u32>,
    centre_slot: u32,
    c_min: f32,
    #[comptime] split_shift: u32,
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
    #[comptime] matrix_plane: u32,
    #[comptime] int8_plane: u32,
    #[comptime] int8: Int8Layout,
    #[comptime] variant: IntegerVariant,
) {
    let route = comptime!(match variant.centre {
        IntCentre::MatrixI8 => MatrixRoute::int8(int8_plane, int8),
        _ => MatrixRoute::f16(matrix_plane, split_shift),
    });

    let thread_id = UNIT_POS_X;
    let group = thread_id / 8u32;
    let sub = thread_id % 8u32;
    let base = group_base();
    let cube_refs_x = comptime!(variant.cube_refs_x);
    let ref_rows = comptime!(variant.ref_rows);
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

    let group_column = group % cube_refs_x;
    let group_row = group / cube_refs_x;
    let ref_x_index = CUBE_POS_X * cube_refs_x + group_column;
    let ref_y_index = CUBE_POS_Y * ref_rows + group_row;
    let live = ref_x_index < frame_refs_x && ref_y_index < frame_refs_y;
    let ref_x_clamped = ref_x_index.min(frame_refs_x - 1u32);
    let ref_y_clamped = ref_y_index.min(frame_refs_y - 1u32);

    let ref_x = (ref_x_clamped * STEP).min(max_x);
    let ref_y = (ref_y_clamped * STEP).min(max_y);

    let centre = comptime!(variant.centre);
    let tile_len = comptime!(variant.tile_elements(spatial_radius, channels));
    let mut tile = SharedMemory::<i32>::new(tile_len as usize);

    // The tile's corner is the first reference's search window corner, so every window of the
    // cube lies inside it.
    let first_ref_x = (CUBE_POS_X * cube_refs_x * STEP).min(max_x);
    let first_ref_y = (CUBE_POS_Y * ref_rows * STEP).min(max_y);
    let tile_x0 = clamp_top_left(first_ref_x as i32 - spatial_radius as i32, max_x);
    let tile_y0 = clamp_top_left(first_ref_y as i32 - spatial_radius as i32, max_y);

    if comptime!(centre == IntCentre::Tiled) {
        let tile_w = comptime!(tile_width(cube_refs_x, spatial_radius));
        let tile_h = comptime!(tile_height(spatial_radius));
        let lanes = comptime!(variant.lanes());
        load_int_centre_tile(
            codes,
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

    let current = load_code_column(codes, ref_x, ref_y, centre_slot, sub, width, height, channels);

    let cursor = IntCursor {
        rx: ref_x,
        ry: ref_y,
        tile_x0,
        tile_y0,
        centre_slot,
        sub,
        base,
    };
    let mut member_pos = Array::<u32>::new(MAX_K as usize);
    let group_word = int_search_group::<C, N, E, A>(
        codes,
        &tile,
        mv_field,
        confidence,
        neighbour_slots,
        &current,
        &cursor,
        c_min,
        &mut member_pos,
        variant,
        route,
        frame_refs_x,
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

    let ref_idx = ref_y_clamped * frame_refs_x + ref_x_clamped;
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

/// Finds the group of the reference at the cursor and writes its members to `member_pos`.
///
/// It mirrors [search_group](crate::collab::kernels::fused::search_group) with exact i32 SSDs.
/// Every lane of the group ends up holding the same members. Returns the group word, the member
/// count in the low 8 bits with [GROUP_USES_GRID](crate::collab::kernels::fused::GROUP_USES_GRID)
/// set when the group is a volume grid.
///
/// A matrix centre runs [matrix_centre_search](crate::collab::kernels::integer::matrix::matrix_centre_search)
/// on `route`, so every lane of the cube must call it.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the search reads"
)]
pub(crate) fn int_search_group<C: Int, N: Size, E: Numeric, A: Numeric>(
    codes: &Array<Vector<C, N>>,
    tile: &SharedMemory<i32>,
    mv_field: &Array<i32>,
    confidence: &Array<f32>,
    neighbour_slots: &Array<u32>,
    current: &Array<i32>,
    cursor: &IntCursor,
    c_min: f32,
    member_pos: &mut Array<u32>,
    #[comptime] variant: IntegerVariant,
    #[comptime] route: MatrixRoute,
    #[comptime] frame_refs_x: u32,
    #[comptime] radius: u32,
    #[comptime] grid_frames: u32,
    #[comptime] refine: u32,
    #[comptime] mv_stride: u32,
    #[comptime] conf_stride: u32,
    #[comptime] blk_step: u32,
    #[comptime] blksize: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
    #[comptime] k_max: u32,
    #[comptime] shape: IntShape,
) -> u32 {
    let walk = comptime!(variant.walk);
    let width = comptime!(shape.width);
    let height = comptime!(shape.height);
    let channels = comptime!(shape.channels);
    let sub = cursor.sub;

    // cubecl treats consts as comptime, so the slot takes the constant through `runtime()`.
    let mut best_ssd = UNFILLED_SSD.runtime();
    let mut best_pos = 0u32;

    let n_live = if comptime!(variant.has_matrix_centre()) {
        matrix_centre_search::<C, N, E, A>(
            codes,
            cursor,
            &mut best_ssd,
            &mut best_pos,
            frame_refs_x,
            variant,
            route,
            shape,
        )
    } else {
        int_centre_search(
            codes,
            tile,
            current,
            cursor,
            &mut best_ssd,
            &mut best_pos,
            variant,
            shape,
        )
    };

    let mut k_use = 1u32;
    while k_use * 2u32 <= n_live && k_use * 2u32 <= k_max {
        k_use *= 2u32;
    }

    // The single-frame group, which is also the fallback. Lane `i` holds member `i`.
    #[unroll]
    for m in 0..MAX_K {
        member_pos[m as usize] = plane_shuffle(best_pos, cursor.base + m);
    }

    let mut use_grid = false;
    if comptime!(grid_frames > 1 && k_max == MAX_K) {
        let volumes = comptime!(MAX_K / grid_frames);
        let tail = comptime!(grid_frames - 1);

        let mut grid_pos = Array::<u32>::new(MAX_K as usize);
        let mut grid_ssd = Array::<i32>::new(MAX_K as usize);

        #[unroll]
        for volume in 0..volumes {
            let first = comptime!(volume * grid_frames);
            let anchor_packed = member_pos[volume as usize];
            let anchor_x = anchor_packed & 0x1FFFu32;
            let anchor_y = (anchor_packed >> 13u32) & 0x1FFFu32;

            grid_pos[first as usize] = anchor_packed;
            grid_ssd[first as usize] = 0i32;
            #[unroll]
            for j in 0..tail {
                grid_pos[comptime!(first + 1 + j) as usize] = 0u32;
                grid_ssd[comptime!(first + 1 + j) as usize] = UNFILLED_SSD;
            }

            let anchor = load_code_column(
                codes,
                anchor_x,
                anchor_y,
                cursor.centre_slot,
                sub,
                width,
                height,
                channels,
            );

            int_trajectory_search(
                codes,
                mv_field,
                confidence,
                neighbour_slots,
                &anchor,
                anchor_x,
                anchor_y,
                sub,
                c_min,
                &mut grid_ssd,
                &mut grid_pos,
                comptime!(first + 1),
                tail,
                walk,
                radius,
                refine,
                mv_stride,
                conf_stride,
                blk_step,
                blksize,
                blocks_x,
                blocks_y,
                shape,
            );
        }

        // The list is ascending, so a filled last slot means the whole volume is filled.
        use_grid = k_use == MAX_K;
        #[unroll]
        for volume in 0..volumes {
            let last = comptime!(volume * grid_frames + tail) as usize;
            use_grid = use_grid && grid_ssd[last] < UNFILLED_SSD;
        }

        #[unroll]
        for m in 0..MAX_K {
            member_pos[m as usize] = select(use_grid, grid_pos[m as usize], member_pos[m as usize]);
        }
        k_use = select(use_grid, MAX_K, k_use);
    }

    let grid_flag = select(use_grid, GROUP_USES_GRID, 0u32);
    k_use | grid_flag
}

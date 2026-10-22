pub(crate) mod batched;
pub(crate) mod centre;
pub(crate) mod distance;
pub(crate) mod grid;
pub(crate) mod pooled;
pub(crate) mod shape;
pub(crate) mod sliding;
mod split;
pub(crate) mod strength_map;
pub(crate) mod tile;
pub(crate) mod trajectory;
pub(crate) mod variant;

use cubecl::prelude::*;

use self::centre::centre_search;
use self::grid::{grid_fwd, grid_inv, grid_variance};
use self::pooled::pooled_threshold;
use self::shape::{SearchCursor, SearchShape};
pub use self::split::{MEMBER_WORDS, collab_filter, collab_search};
use self::strength_map::strength_map_scale;
use self::tile::load_centre_tile;
use self::trajectory::trajectory_search;
pub use self::variant::{
    Centre,
    CollabVariant,
    GROUP_LANES,
    NeighbourWalk,
    TRANSPOSE_REGION_FLOATS,
    Walk,
    tile_height,
    tile_width,
};
use super::aggregate::scatter_patch;
use super::group::{clamp_top_left, unpack_t};
use super::plane_ops::{UNFILLED_DISTANCE, group_base, plane_ssd_reduce8, transpose8};
use super::transforms::{
    RECIPROCAL_FLOOR,
    dct8_reg_fwd,
    dct8_reg_inv,
    haar_reg_fwd_level,
    haar_reg_inv_level,
    safe_reciprocal,
    variance_reg_level,
};
use crate::collab::{MAX_K, MAX_TEMPORAL_RADIUS, PATCH_AREA, PATCH_SIZE, STEP};
use crate::engine::kernels::read_line;
use crate::nlmeans::kernels::helpers::channel_scale;
use crate::noise::{NOISE_CURVE_BINS, STRENGTH_MAP_ALL, STRENGTH_MAP_LUMA};

/// The smallest factor the noise curve may scale the luma threshold by.
const NOISE_CURVE_SCALE_MIN: f32 = 0.33;

/// The largest factor the noise curve may scale the luma threshold by.
const NOISE_CURVE_SCALE_MAX: f32 = 3.0;

/// The bits of a group word that hold the member count.
pub const GROUP_SIZE_MASK: u32 = 0xFF;

/// The group word bit set when the group is a volume grid.
pub const GROUP_USES_GRID: u32 = 0x100;

// The widest neighbour index packed is `2 * radius`, and `pack_pos_t`'s 6-bit `t` field silently
// corrupts the word at 64 or more.
const _: () = assert!(
    2 * MAX_TEMPORAL_RADIUS < 64,
    "pack_pos_t's 6-bit t field must hold every neighbour index collab_fused packs"
);

// A lane holds one 8-value column of each of `MAX_K` members, so the group fits `PATCH_AREA`
// slots only while the group size and the patch side match. The stack transform's predicate
// ladder also names the levels 8, 4 and 2 outright.
const _: () = assert!(
    MAX_K == PATCH_SIZE && MAX_K == 8,
    "collab_fused's per-lane group array and its three-level stack transform are written for \
     MAX_K == PATCH_SIZE == 8"
);

/// Groups each reference patch with its most similar patches, hard-thresholds the group in the
/// transform domain and scatters every member back into its own frame.
///
/// A cube of `8 * variant.groups` threads owns `variant.groups` reference patches. Each 8-lane
/// group owns one, and lane `sub` owns column `sub` of every patch the group touches, so candidate
/// reads and scatter writes are coalesced and `plane_ssd_reduce8` completes each distance.
/// Candidates are read from global memory or, when tiled, from shared memory. The fixed-point
/// scatter adds integers, so the output does not depend on `variant.groups`.
///
/// Every lane must reach every barrier, since a barrier reached by only part of a workgroup is
/// undefined. The `transpose8` barriers sit in fully unrolled loops with no runtime condition around
/// them. A group past the end of a row, which every 1080p row has, works on a clamped copy of the
/// last real reference and is gated only where it writes.
///
/// The centre frame contributes the `spatial_radius` rectangle around the reference, clipped to
/// the frame, and the best eight positions are kept.
/// [pin_self_match](crate::collab::kernels::fused::centre::pin_self_match) holds the self-match in
/// slot 0. The first `MAX_K / grid_frames` positions become volume anchors. In each neighbour
/// frame an anchor searches one `refine` rectangle per covering motion block, around where that
/// block's vector moves it, and keeps its best match. Each rectangle is clipped once so every
/// candidate is a distinct position, and a position reached twice or already in the group is
/// scored once. A block whose confidence is below `c_min` is skipped, uniformly across the group.
///
/// A member is a `pack_pos_t` word, and member `s * grid_frames + t` is frame `t` of volume `s`.
/// The grid is used when the spatial search held at least `MAX_K` positions, `k_max` is `MAX_K`
/// and every volume filled. Otherwise the group falls back to the single-frame group, its
/// positions rounded down to a power of two and capped at `k_max`. A block skipped by `c_min` can
/// leave a volume short of frames, which makes the group fall back, so `c_min` can change the
/// output.
///
/// For each channel, every member runs through a 2D DCT as a column pass, a transpose and a row
/// pass. A grid group then runs a Haar along time and across volumes, and a fallback group a Haar
/// across its stack. The transforms unroll fully, which keeps the whole group in registers. A
/// coefficient survives when its magnitude reaches `lambda_ht` standard deviations of its
/// propagated noise, with every member carrying `sigma[c]^2`. The group DC of the spatial DC
/// always survives, because a group's mean brightness is signal. Both transforms then invert.
///
/// Channel 0's threshold is scaled by the noise curve at the reference's mean luma, interpolated
/// between bin centres and clamped to 0.33..=3, while the group weight keeps the plain sigma. The
/// strength map multiplier is the mean of the four 8x8 quarters the reference overlaps.
/// [STRENGTH_MAP_LUMA] scales channel 0's curve ratio before the clamp. [STRENGTH_MAP_ALL] scales
/// the chroma thresholds, and channel 0's too when there is no curve. [STRENGTH_MAP_OFF] leaves
/// every threshold alone. With `pooled` set, `pooled_threshold` keeps a coefficient on the mean
/// energy of itself and its four frequency neighbours, against `channel_lambda * pool_ratio`.
///
/// `group_weight` gets `1 / sum(v_j)` over the kept coefficients, the inverse-variance weight, so
/// a group that keeps more coefficients is trusted more. It comes from channel 0 only, because
/// luma dominates and one weight per group keeps aggregation simple. `weight_scale` maps it into
/// the fixed-point band before the scatter.
///
/// `ring` holds frames in ring-slot order. `ring_scalar` is the same buffer bound as single floats.
/// `centre_slot` is the slot the pass is centred on. `neighbour_slots` maps a packed neighbour
/// index to its slot. `accum` and `wsum` hold one region per slot, so a neighbour-frame member
/// scatters into its own frame's region. `accum_scale` is the fixed-point scale of that scatter.
/// `sigma` holds one value per stored channel.
/// `noise_curve` holds `NOISE_CURVE_BINS` ratios and is read only when `curve_valid` is not 0.
/// `strength_map` holds `map_cols * map_rows` row-major multipliers laid out by
/// [strength_map_dims](crate::collab::geometry::strength_map_dims). `kaiser` holds
/// [kaiser_window](crate::collab::kernels::aggregate::kaiser_window)'s taps. `dct_profile` holds
/// [dct_noise_profile](crate::collab::kernels::transforms::dct_noise_profile)'s values, and
/// coefficient `(u, v)`'s variance scales by `dct_profile[u] * dct_profile[v]`. `grid_frames` is
/// the frames per volume from [grid_frames](crate::collab::grid_frames), and 1 compiles the grid
/// out.
///
/// `variant` picks how the searches run, and every variant gives the same output.
/// [Walk::Uniform](crate::collab::kernels::fused::Walk::Uniform) walks both searches over fixed
/// comptime rectangles and masks the positions the clipped walk skips, so every group in a plane
/// takes the same turns. The CUDA backend lowers each group-scoped shuffle to a `__shfl_*_sync` over
/// the whole warp, so on Volta and later the clipped walk's different trip counts deadlock the warp.
/// [needs_warp_uniform_search](crate::collab::needs_warp_uniform_search) names the runtimes that
/// only run the uniform walk.
/// [Centre::Tiled](crate::collab::kernels::fused::Centre::Tiled) loads the union of the cube's
/// centre search windows into shared memory once and reads every centre candidate from it, in the
/// same order as the ring.
/// `batched` scores 8 centre candidates per round of shuffles and inserts them in walk order.
///
/// `search_ring` is an f16 copy of `ring`, with `S` as `half::f16`, that the candidate distances
/// read when `f16_search` is set. The reference columns always come from `ring`, and the member
/// stack reads each value as a single float from `ring_scalar`. With `f16_search` unset
/// `search_ring` is never read and can be the f32 `ring`, which keeps every f16 type out of the
/// kernel.
///
/// With `packed`, `wsum` holds one `i64` word per pixel with channel 0 and the weight, and `accum`
/// holds the other channels, as [scatter_patch] writes them. Otherwise `W` is `i32`.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub fn collab_fused<S: Float, N: Size, W: Int>(
    ring: &Array<Vector<f32, N>>,
    ring_scalar: &Array<f32>,
    search_ring: &Array<Vector<S, N>>,
    mv_field: &Array<i32>,
    confidence: &Array<f32>,
    neighbour_slots: &Array<u32>,
    sigma: &Array<f32>,
    noise_curve: &Array<f32>,
    strength_map: &Array<f32>,
    dct_profile: &Array<f32>,
    kaiser: &Array<f32>,
    accum: &mut Array<Atomic<i32>>,
    wsum: &mut Array<Atomic<W>>,
    group_weight: &mut Array<f32>,
    centre_slot: u32,
    c_min: f32,
    lambda_ht: f32,
    curve_valid: u32,
    map_mode: u32,
    weight_scale: f32,
    accum_scale: f32,
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
    #[comptime] map_cols: u32,
    #[comptime] map_rows: u32,
    pool_ratio: f32,
    #[comptime] pooled: bool,
    #[comptime] variant: CollabVariant,
    #[comptime] packed: bool,
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

    let transpose_floats = comptime!(groups * TRANSPOSE_REGION_FLOATS);
    let mut transpose_buf = SharedMemory::<f32>::new(transpose_floats as usize);

    let centre = comptime!(variant.centre);
    let lanes = comptime!(variant.lanes());
    let tile_len = comptime!(variant.tile_elements(spatial_radius, stored_ch));
    let mut tile = SharedMemory::<S>::new(tile_len as usize);

    // A dead group works on the last real reference of the row, so every read stays inside the
    // frame and every lane reaches every barrier. `live` stops it writing.
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

    // The channel scale multiplies the completed 64-pixel distance, not each squared difference.
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
    let k_use = group_word & GROUP_SIZE_MASK;
    let use_grid = (group_word & GROUP_USES_GRID) != 0u32;
    let column_luma = column_luma_of(&current, channels);

    filter_group::<W>(
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
        packed,
    );
}

/// Column `sub` of the patch at `(x, y)` in frame `slot`, every channel, row by row.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the load reads"
)]
pub(crate) fn load_reference_column<N: Size>(
    ring: &Array<Vector<f32, N>>,
    x: u32,
    y: u32,
    slot: u32,
    sub: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
) -> Array<f32> {
    let mut column = Array::<f32>::new(comptime!(PATCH_SIZE * channels) as usize);

    #[unroll]
    for r in 0..PATCH_SIZE {
        let pixel = read_line(ring, x + sub, y + r, slot, width, height);
        #[unroll]
        for c in 0..channels {
            column[(r * channels + c) as usize] = pixel[c as usize];
        }
    }

    column
}

/// The sum of channel 0 down a column loaded by [load_reference_column].
#[cube]
pub(crate) fn column_luma_of(column: &Array<f32>, #[comptime] channels: u32) -> f32 {
    let mut column_luma = 0.0f32;

    #[unroll]
    for r in 0..PATCH_SIZE {
        column_luma += column[(r * channels) as usize];
    }

    column_luma
}

/// Finds the group of the reference at the cursor and writes its members to `member_pos`.
///
/// Every lane of the group ends up holding the same members. Returns the group word, the member
/// count in the low 8 bits with [GROUP_USES_GRID] set when the group is a volume grid.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the search reads"
)]
pub(crate) fn search_group<S: Float, N: Size>(
    ring: &Array<Vector<f32, N>>,
    search_ring: &Array<Vector<S, N>>,
    tile: &SharedMemory<S>,
    mv_field: &Array<i32>,
    confidence: &Array<f32>,
    neighbour_slots: &Array<u32>,
    current: &Array<f32>,
    cursor: &SearchCursor,
    c_min: f32,
    member_pos: &mut Array<u32>,
    #[comptime] variant: CollabVariant,
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
    #[comptime] shape: SearchShape,
) -> u32 {
    let walk = comptime!(variant.walk);
    let neighbour_walk = comptime!(variant.neighbour);
    let width = comptime!(shape.width);
    let height = comptime!(shape.height);
    let channels = comptime!(shape.channels);
    let sub = cursor.sub;

    // An unplaced candidate carries `UNFILLED_DISTANCE` and the self-match carries `-1.0e38`.
    // cubecl treats consts as comptime, so the slot takes the constant through `runtime()`.
    let mut best_d = UNFILLED_DISTANCE.runtime();
    let mut best_pos = 0u32;

    let n_live = centre_search(
        ring,
        search_ring,
        tile,
        current,
        cursor,
        &mut best_d,
        &mut best_pos,
        variant,
        shape,
    );

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
        let mut grid_d = Array::<f32>::new(MAX_K as usize);

        #[unroll]
        for volume in 0..volumes {
            let first = comptime!(volume * grid_frames);
            let anchor_packed = member_pos[volume as usize];
            let anchor_x = anchor_packed & 0x1FFFu32;
            let anchor_y = (anchor_packed >> 13u32) & 0x1FFFu32;

            grid_pos[first as usize] = anchor_packed;
            grid_d[first as usize] = 0.0f32;
            #[unroll]
            for j in 0..tail {
                grid_pos[comptime!(first + 1 + j) as usize] = 0u32;
                grid_d[comptime!(first + 1 + j) as usize] = UNFILLED_DISTANCE;
            }

            let mut anchor = Array::<f32>::new(comptime!(PATCH_SIZE * channels) as usize);
            #[unroll]
            for r in 0..PATCH_SIZE {
                let pixel = read_line(
                    ring,
                    anchor_x + sub,
                    anchor_y + r,
                    cursor.centre_slot,
                    width,
                    height,
                );
                #[unroll]
                for c in 0..channels {
                    anchor[(r * channels + c) as usize] = pixel[c as usize];
                }
            }

            trajectory_search(
                ring,
                search_ring,
                mv_field,
                confidence,
                neighbour_slots,
                &anchor,
                anchor_x,
                anchor_y,
                sub,
                cursor.scale,
                c_min,
                &mut grid_d,
                &mut grid_pos,
                comptime!(first + 1),
                tail,
                walk,
                neighbour_walk,
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
            use_grid = use_grid && grid_d[last] < UNFILLED_DISTANCE;
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

/// Filters the group in `member_pos` and scatters every member back into its own frame.
///
/// `column_luma` is lane `sub`'s sum of channel 0 down the reference column, and the group sums
/// it into the patch luma the noise curve reads. `live` gates every write.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the filter binds"
)]
#[expect(
    clippy::collapsible_if,
    reason = "the outer condition of the group-DC exception is comptime, so nesting elides the \
              inner test in 63 of the 64 unrolled positions rather than emitting it and ANDing \
              a constant false into it"
)]
pub(crate) fn filter_group<W: Int>(
    ring_scalar: &Array<f32>,
    neighbour_slots: &Array<u32>,
    sigma: &Array<f32>,
    noise_curve: &Array<f32>,
    strength_map: &Array<f32>,
    dct_profile: &Array<f32>,
    kaiser: &Array<f32>,
    accum: &mut Array<Atomic<i32>>,
    wsum: &mut Array<Atomic<W>>,
    group_weight: &mut Array<f32>,
    transpose_buf: &mut SharedMemory<f32>,
    member_pos: &Array<u32>,
    k_use: u32,
    use_grid: bool,
    column_luma: f32,
    ref_x: u32,
    ref_y: u32,
    ref_idx: u32,
    live: bool,
    centre_slot: u32,
    sub: u32,
    group: u32,
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
    #[comptime] map_cols: u32,
    #[comptime] map_rows: u32,
    #[comptime] pooled: bool,
    #[comptime] packed: bool,
) {
    // The frame is picked with `select` rather than a branch, because a frame index that reaches
    // a ring read through a branch panics cubecl 0.10's GVN pass and the launch silently does
    // nothing.
    let mut member_slot = Array::<u32>::new(MAX_K as usize);
    #[unroll]
    for m in 0..MAX_K {
        let position = member_pos[m as usize];
        let neighbour_field = unpack_t(position);
        // Clamped so the read stays in range for a centre-frame member, whose value `select`
        // discards. This needs `neighbour_slots` to hold at least one entry, even at `radius = 0`.
        let neighbour = u32::max(neighbour_field, 1u32) - 1u32;
        member_slot[m as usize] = select(
            neighbour_field > 0u32,
            neighbour_slots[neighbour as usize],
            centre_slot,
        );
    }

    // Lane `sub` ends up owning vertical frequency `sub`, so its half of the separable profile is
    // read once.
    let prof_sub = dct_profile[sub as usize];

    // Every lane reaches the luma reduction, so the group stays converged.
    let patch_luma = plane_ssd_reduce8(column_luma) / comptime!(PATCH_AREA as f32);
    let bins = comptime!(NOISE_CURVE_BINS as f32);
    let unclamped_pos = patch_luma * bins - 0.5f32;
    let curve_pos = f32::clamp(unclamped_pos, 0.0f32, bins - 1.0f32);
    let lower_bin = u32::min(curve_pos as u32, comptime!(NOISE_CURVE_BINS as u32 - 2));
    let fraction = curve_pos - lower_bin as f32;
    let lower_ratio = noise_curve[lower_bin as usize];
    let upper_ratio = noise_curve[(lower_bin + 1u32) as usize];
    let ratio = lower_ratio + (upper_ratio - lower_ratio) * fraction;
    let map_scale = strength_map_scale(strength_map, ref_x, ref_y, map_cols, map_rows);
    let mapped_ratio = select(map_mode == STRENGTH_MAP_LUMA, ratio * map_scale, ratio);
    let curve_scale = f32::clamp(mapped_ratio, NOISE_CURVE_SCALE_MIN, NOISE_CURVE_SCALE_MAX);
    let other_lambda = select(map_mode == STRENGTH_MAP_ALL, lambda_ht * map_scale, lambda_ht);
    let luma_lambda = select(curve_valid != 0u32, lambda_ht * curve_scale, other_lambda);

    // Computed from channel 0 and reused by every later channel's scatter.
    let mut scaled_weight = 0.0f32;

    // The channel loop runs at runtime, which keeps one copy of the filter in the kernel. Every
    // loop inside it unrolls, and its comptime trip count keeps every lane on every barrier.
    for c in 0..channels {
        let sigma_c = sigma[c as usize];
        let base_sig2 = sigma_c * sigma_c;

        // Lane `sub` holds `stack[m * 8 + r]` for member `m`, row `r`, read from the member's own
        // frame.
        let mut stack = Array::<f32>::new(PATCH_AREA as usize);
        let mut member_variance = Array::<f32>::new(MAX_K as usize);
        #[unroll]
        for m in 0..MAX_K {
            let position = member_pos[m as usize];
            let member_x = position & 0x1FFFu32;
            let member_y = (position >> 13u32) & 0x1FFFu32;
            let src_slot = member_slot[m as usize];
            member_variance[m as usize] = base_sig2;

            // Each value reads as a single float, so the runtime channel indexes it directly.
            #[unroll]
            for r in 0..PATCH_SIZE {
                let line_index = (src_slot * height + member_y + r) * width + member_x + sub;
                let value_index = line_index * stored_ch + c;
                stack[(m * PATCH_SIZE + r) as usize] = ring_scalar[value_index as usize];
            }
        }

        // The spatial profile is constant along the stack axis and the ladder only averages, so
        // the profile multiplies in at the threshold instead.
        if comptime!(grid_frames > 1) {
            if use_grid {
                grid_variance(&mut member_variance, grid_frames);
            } else {
                stack_variance_ladder(&mut member_variance, k_use);
            }
        } else {
            stack_variance_ladder(&mut member_variance, k_use);
        }

        // Lane `sub` comes out holding coefficient `(u = i, v = sub)` at slot `i`.
        #[unroll]
        for m in 0..MAX_K {
            let mut line = Array::<f32>::new(PATCH_SIZE as usize);
            #[unroll]
            for i in 0..PATCH_SIZE {
                line[i as usize] = stack[(m * PATCH_SIZE + i) as usize];
            }
            dct8_reg_fwd(&mut line);
            transpose8(transpose_buf, &mut line, sub, group);
            dct8_reg_fwd(&mut line);
            #[unroll]
            for i in 0..PATCH_SIZE {
                stack[(m * PATCH_SIZE + i) as usize] = line[i as usize];
            }
        }

        // A lane owns every member at each of its positions, so nothing crosses lanes here.
        if comptime!(grid_frames > 1) {
            if use_grid {
                grid_fwd(&mut stack, grid_frames);
            } else {
                stack_haar_fwd(&mut stack, k_use);
            }
        } else {
            stack_haar_fwd(&mut stack, k_use);
        }

        let channel_lambda = select(c == 0u32, luma_lambda, other_lambda);
        let mut retained_v = 0.0f32;
        if comptime!(pooled) {
            let threshold = channel_lambda * pool_ratio;
            retained_v = pooled_threshold(
                &mut stack,
                &member_variance,
                dct_profile,
                prof_sub,
                sub,
                k_use,
                threshold,
                channel_lambda,
            );
        } else {
            #[unroll]
            for i in 0..PATCH_SIZE {
                let factor = dct_profile[i as usize] * prof_sub;
                #[unroll]
                for j in 0..MAX_K {
                    if j < k_use {
                        let coeff_variance = member_variance[j as usize] * factor;
                        let slot = (j * PATCH_SIZE + i) as usize;
                        let mut keep = f32::abs(stack[slot]) >= channel_lambda * f32::sqrt(coeff_variance);
                        if comptime!(j == 0u32 && i == 0u32) {
                            if sub == 0u32 {
                                keep = true;
                            }
                        }
                        if keep {
                            retained_v += coeff_variance;
                        } else {
                            stack[slot] = 0.0f32;
                        }
                    }
                }
            }
        }

        // The scatter needs the weight, so the reduction runs before the inverse transforms. It
        // runs for every channel so every lane reaches its shuffles, and only channel 0 keeps it.
        let sum = plane_ssd_reduce8(retained_v);
        let weight = safe_reciprocal(sum, RECIPROCAL_FLOOR);
        let first_channel = c == 0u32;
        if live && sub == 0u32 && first_channel {
            group_weight[ref_idx as usize] = weight;
        }

        scaled_weight = select(first_channel, weight * weight_scale, scaled_weight);

        // The inverse runs in the opposite order, which leaves the lane holding a column again so
        // the scatter is coalesced.
        if comptime!(grid_frames > 1) {
            if use_grid {
                grid_inv(&mut stack, grid_frames);
            } else {
                stack_haar_inv(&mut stack, k_use);
            }
        } else {
            stack_haar_inv(&mut stack, k_use);
        }

        #[unroll]
        for m in 0..MAX_K {
            let mut line = Array::<f32>::new(PATCH_SIZE as usize);
            #[unroll]
            for i in 0..PATCH_SIZE {
                line[i as usize] = stack[(m * PATCH_SIZE + i) as usize];
            }
            dct8_reg_inv(&mut line);
            transpose8(transpose_buf, &mut line, sub, group);
            dct8_reg_inv(&mut line);
            #[unroll]
            for i in 0..PATCH_SIZE {
                stack[(m * PATCH_SIZE + i) as usize] = line[i as usize];
            }
        }

        // Every member is written back into its own frame's region, so neighbour-frame members
        // feed the cross-frame ring.
        #[unroll]
        for m in 0..MAX_K {
            if live && m < k_use {
                let position = member_pos[m as usize];
                let member_x = position & 0x1FFFu32;
                let member_y = (position >> 13u32) & 0x1FFFu32;
                let dst_slot = member_slot[m as usize];
                #[unroll]
                for r in 0..PATCH_SIZE {
                    scatter_patch::<W>(
                        accum,
                        wsum,
                        kaiser,
                        stack[(m * PATCH_SIZE + r) as usize],
                        scaled_weight,
                        member_x,
                        member_y,
                        r * PATCH_SIZE + sub,
                        first_channel,
                        c,
                        width,
                        stored_ch,
                        dst_slot,
                        comptime!(width * height),
                        accum_scale,
                        packed,
                    );
                }
            }
        }
    }
}

/// The stack variance ladder for a single-frame group of `k_use` members.
#[cube]
fn stack_variance_ladder(v: &mut Array<f32>, k_use: u32) {
    if k_use >= 8u32 {
        variance_reg_level(v, 8u32);
    }
    if k_use >= 4u32 {
        variance_reg_level(v, 4u32);
    }
    if k_use >= 2u32 {
        variance_reg_level(v, 2u32);
    }
}

/// The stack Haar for a single-frame group of `k_use` members.
#[cube]
fn stack_haar_fwd(stack: &mut Array<f32>, k_use: u32) {
    if k_use >= 8u32 {
        haar_reg_fwd_level(stack, 8u32);
    }
    if k_use >= 4u32 {
        haar_reg_fwd_level(stack, 4u32);
    }
    if k_use >= 2u32 {
        haar_reg_fwd_level(stack, 2u32);
    }
}

/// The inverse of `stack_haar_fwd`.
#[cube]
fn stack_haar_inv(stack: &mut Array<f32>, k_use: u32) {
    if k_use >= 2u32 {
        haar_reg_inv_level(stack, 2u32);
    }
    if k_use >= 4u32 {
        haar_reg_inv_level(stack, 4u32);
    }
    if k_use >= 8u32 {
        haar_reg_inv_level(stack, 8u32);
    }
}

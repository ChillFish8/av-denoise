pub(crate) mod grid;
pub(crate) mod pooled;
pub(crate) mod search;
pub(crate) mod strength_map;

use cubecl::prelude::*;

use self::grid::{grid_fwd, grid_inv, grid_variance};
use self::pooled::pooled_threshold;
use self::search::{spatial_search, trajectory_search};
use self::strength_map::strength_map_scale;
pub use self::strength_map::{STRENGTH_MAP_ALL, STRENGTH_MAP_LUMA, STRENGTH_MAP_OFF};
use super::aggregate::scatter_patch;
use super::group::unpack_t;
use super::plane_ops::{group_base, plane_ssd_reduce8, transpose8};
use super::transforms::{
    RECIPROCAL_FLOOR,
    dct8_reg_fwd,
    dct8_reg_inv,
    fill_dct8_basis,
    haar_reg_fwd_level,
    haar_reg_inv_level,
    safe_reciprocal,
    variance_reg_level,
};
use crate::collab::{MAX_K, MAX_TEMPORAL_RADIUS, PATCH_AREA, PATCH_SIZE, STEP};
use crate::nlmeans::NOISE_CURVE_BINS;
use crate::nlmeans::kernels::helpers::{channel_scale, read_line};

/// The smallest factor the noise curve may scale the luma threshold by.
const NOISE_CURVE_SCALE_MIN: f32 = 0.33;

/// The largest factor the noise curve may scale the luma threshold by.
const NOISE_CURVE_SCALE_MAX: f32 = 3.0;

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
/// A cube of 64 threads owns eight reference patches. Each 8-lane group owns one, and lane `sub`
/// owns column `sub` of every patch the group touches, so candidate reads and scatter writes are
/// coalesced and `plane_ssd_reduce8` completes each distance. Candidates are read straight from
/// global memory, because neighbouring references search overlapping windows the cache already
/// serves.
///
/// Every lane must reach every barrier, since a barrier reached by only part of a workgroup is
/// undefined. The basis fill barrier is unconditional, and the `transpose8` barriers sit in fully
/// unrolled loops with no runtime condition around them. A group past the end of a row, which
/// every 1080p row has, works on a clamped copy of the last real reference and is gated only
/// where it writes.
///
/// The centre frame contributes the `spatial_radius` rectangle around the reference, clipped to
/// the frame, and the best eight positions are kept. The self-match scores a sentinel below every
/// real distance, which pins it into slot 0. The first `MAX_K / grid_frames` positions become
/// volume anchors. In each neighbour frame an anchor searches one `refine` rectangle per covering
/// motion block, around where that block's vector moves it, and keeps its best match. Each
/// rectangle is clipped once so every candidate is a distinct position, and a position reached
/// twice or already in the group is scored once. A block whose confidence is below `c_min` is
/// skipped, uniformly across the group.
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
/// `ring` holds frames in ring-slot order. `centre_slot` is the slot the pass is centred on.
/// `neighbour_slots` maps a packed neighbour index to its slot. `accum` and `wsum` hold one region
/// per slot, so a neighbour-frame member scatters into its own frame's region. `accum_scale` is
/// the fixed-point scale of that scatter. `sigma` holds one value per stored channel.
/// `noise_curve` holds `NOISE_CURVE_BINS` ratios and is read only when `curve_valid` is not 0.
/// `strength_map` holds `map_cols * map_rows` row-major multipliers laid out by
/// [strength_map_dims](crate::collab::geometry::strength_map_dims). `kaiser` holds
/// [kaiser_window](crate::collab::kernels::aggregate::kaiser_window)'s taps. `dct_profile` holds
/// [dct_noise_profile](crate::collab::kernels::transforms::dct_noise_profile)'s values, and
/// coefficient `(u, v)`'s variance scales by `dct_profile[u] * dct_profile[v]`. `grid_frames` is
/// the frames per volume from [grid_frames](crate::collab::grid_frames), and 1 compiles the grid
/// out.
///
/// `warp_uniform` walks both searches over fixed comptime rectangles and masks the positions the
/// clipped walk skips, so both settings score the same candidates in the same order. The clipped
/// rectangles and the `c_min` skip give neighbouring groups different trip counts. The CUDA
/// backend lowers each group-scoped shuffle to a `__shfl_*_sync` over the whole warp, so on Volta
/// and later those different trip counts deadlock the warp. A masked turn carries the
/// `3.0e38` an unfilled slot holds, so it never displaces a match. The wgpu backends reconverge on
/// their own, and [needs_warp_uniform_search](crate::collab::needs_warp_uniform_search) picks the
/// setting per runtime.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
#[expect(
    clippy::collapsible_if,
    reason = "the outer condition of the group-DC exception is comptime, so nesting elides the \
              inner test in 63 of the 64 unrolled positions rather than emitting it and ANDing \
              a constant false into it"
)]
pub fn collab_fused<N: Size>(
    ring: &Array<Vector<f32, N>>,
    mv_field: &Array<i32>,
    confidence: &Array<f32>,
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
    c_min: f32,
    lambda_ht: f32,
    curve_valid: u32,
    map_mode: u32,
    weight_scale: f32,
    accum_scale: f32,
    #[comptime] warp_uniform: bool,
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
) {
    let thread_id = UNIT_POS_X;
    let group = thread_id / 8u32;
    let sub = thread_id % 8u32;
    let base = group_base();

    let max_x = comptime!(width - PATCH_SIZE);
    let max_y = comptime!(height - PATCH_SIZE);

    // The basis is 256 B against the transpose buffer's 2,080 B, and registers rather than shared
    // memory bound this kernel's occupancy, so it stays shared.
    let mut basis = SharedMemory::<f32>::new(PATCH_AREA as usize);
    let mut transpose_buf = SharedMemory::<f32>::new(comptime!(8 * 65) as usize);
    fill_dct8_basis(&mut basis, thread_id);
    sync_cube();

    // A dead group works on the last real reference of the row, so every read stays inside the
    // frame and every lane reaches every barrier. `live` stops it writing.
    let ref_x_index = CUBE_POS_X * 8u32 + group;
    let live = ref_x_index < refs_x;
    let ref_x_clamped = ref_x_index.min(refs_x - 1u32);

    let ref_x = (ref_x_clamped * STEP).min(max_x);
    let ref_y = (CUBE_POS_Y * STEP).min(max_y);

    // Column `sub` of the reference patch, all channels, held in registers for the whole search.
    let mut current = Array::<f32>::new(comptime!(PATCH_SIZE * channels) as usize);
    #[unroll]
    for r in 0..PATCH_SIZE {
        let pixel = read_line(ring, ref_x + sub, ref_y + r, centre_slot, width, height);
        #[unroll]
        for c in 0..channels {
            current[(r * channels + c) as usize] = pixel[c as usize];
        }
    }

    // An unplaced candidate carries `3.0e38`, far above the largest real distance of 192 and below
    // `f32::MAX`, and the self-match carries `-1.0e38`. They are literals because cubecl treats
    // consts and `f32::INFINITY` as comptime-only, and the shift-insert needs runtime variables.
    let mut best_d = 3.0e38f32;
    let mut best_pos = 0u32;

    // The channel scale multiplies the completed 64-pixel distance, not each squared difference.
    let scale = channel_scale(channels);

    let n_live = spatial_search(
        ring,
        &current,
        ref_x,
        ref_y,
        centre_slot,
        sub,
        base,
        scale,
        &mut best_d,
        &mut best_pos,
        warp_uniform,
        spatial_radius,
        width,
        height,
        channels,
    );

    let ref_idx = CUBE_POS_Y * refs_x + ref_x_clamped;

    let mut k_use = 1u32;
    while k_use * 2u32 <= n_live && k_use * 2u32 <= k_max {
        k_use *= 2u32;
    }

    // The single-frame group, which is also the fallback. Lane `i` holds member `i`.
    let mut member_pos = Array::<u32>::new(MAX_K as usize);
    #[unroll]
    for m in 0..MAX_K {
        member_pos[m as usize] = plane_shuffle(best_pos, base + m);
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
                grid_d[comptime!(first + 1 + j) as usize] = 3.0e38f32;
            }

            let mut anchor = Array::<f32>::new(comptime!(PATCH_SIZE * channels) as usize);
            #[unroll]
            for r in 0..PATCH_SIZE {
                let pixel = read_line(ring, anchor_x + sub, anchor_y + r, centre_slot, width, height);
                #[unroll]
                for c in 0..channels {
                    anchor[(r * channels + c) as usize] = pixel[c as usize];
                }
            }

            trajectory_search(
                ring,
                mv_field,
                confidence,
                neighbour_slots,
                &anchor,
                anchor_x,
                anchor_y,
                sub,
                scale,
                c_min,
                &mut grid_d,
                &mut grid_pos,
                comptime!(first + 1),
                tail,
                warp_uniform,
                radius,
                refine,
                mv_stride,
                conf_stride,
                blk_step,
                blksize,
                blocks_x,
                blocks_y,
                width,
                height,
                channels,
            );
        }

        // The list is ascending, so a filled last slot means the whole volume is filled.
        use_grid = k_use == MAX_K;
        #[unroll]
        for volume in 0..volumes {
            let last = comptime!(volume * grid_frames + tail) as usize;
            use_grid = use_grid && grid_d[last] < 1.0e38f32;
        }

        #[unroll]
        for m in 0..MAX_K {
            member_pos[m as usize] = select(use_grid, grid_pos[m as usize], member_pos[m as usize]);
        }
        k_use = select(use_grid, MAX_K, k_use);
    }

    // The frame is picked with `select` rather than a branch, because a frame index that reaches
    // `read_line` through a branch panics cubecl 0.10's GVN pass and the launch silently does
    // nothing.
    let mut member_slot = Array::<u32>::new(MAX_K as usize);
    #[unroll]
    for m in 0..MAX_K {
        let packed = member_pos[m as usize];
        let neighbour_field = unpack_t(packed);
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
    let mut column_luma = 0.0f32;
    #[unroll]
    for r in 0..PATCH_SIZE {
        column_luma += current[(r * channels) as usize];
    }

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

    #[unroll]
    for c in 0..channels {
        let sigma_c = sigma[c as usize];
        let base_sig2 = sigma_c * sigma_c;

        // Lane `sub` holds `stack[m * 8 + r]` for member `m`, row `r`, read from the member's own
        // frame.
        let mut stack = Array::<f32>::new(PATCH_AREA as usize);
        let mut member_variance = Array::<f32>::new(MAX_K as usize);
        #[unroll]
        for m in 0..MAX_K {
            let packed = member_pos[m as usize];
            let member_x = packed & 0x1FFFu32;
            let member_y = (packed >> 13u32) & 0x1FFFu32;
            let src_slot = member_slot[m as usize];
            member_variance[m as usize] = base_sig2;
            #[unroll]
            for r in 0..PATCH_SIZE {
                let pixel = read_line(ring, member_x + sub, member_y + r, src_slot, width, height);
                stack[(m * PATCH_SIZE + r) as usize] = pixel[c as usize];
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
            dct8_reg_fwd(&basis, &mut line);
            transpose8(&mut transpose_buf, &mut line, sub, group);
            dct8_reg_fwd(&basis, &mut line);
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

        let channel_lambda = if comptime!(c == 0u32) {
            luma_lambda
        } else {
            other_lambda
        };
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

        // The scatter needs the weight, so the reduction runs before the inverse transforms.
        if comptime!(c == 0u32) {
            let sum = plane_ssd_reduce8(retained_v);
            let weight = safe_reciprocal(sum, RECIPROCAL_FLOOR);
            if live && sub == 0u32 {
                group_weight[ref_idx as usize] = weight;
            }
            scaled_weight = weight * weight_scale;
        }

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
            dct8_reg_inv(&basis, &mut line);
            transpose8(&mut transpose_buf, &mut line, sub, group);
            dct8_reg_inv(&basis, &mut line);
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
                let packed = member_pos[m as usize];
                let member_x = packed & 0x1FFFu32;
                let member_y = (packed >> 13u32) & 0x1FFFu32;
                let dst_slot = member_slot[m as usize];
                #[unroll]
                for r in 0..PATCH_SIZE {
                    scatter_patch(
                        accum,
                        wsum,
                        kaiser,
                        stack[(m * PATCH_SIZE + r) as usize],
                        scaled_weight,
                        member_x,
                        member_y,
                        r * PATCH_SIZE + sub,
                        comptime!(c == 0u32),
                        c,
                        width,
                        stored_ch,
                        dst_slot,
                        comptime!(width * height),
                        accum_scale,
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

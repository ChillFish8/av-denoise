pub(crate) mod grid;
pub(crate) mod search;

use cubecl::prelude::*;

use self::grid::{grid_fwd, grid_inv, grid_variance};
use self::search::{spatial_search, trajectory_search};
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
use crate::nlmeans::kernels::helpers::{channel_scale, read_line};

// The widest neighbour index this kernel ever packs is `2 * radius`,
// one past the last neighbour, and `radius` is capped at
// `MAX_TEMPORAL_RADIUS`. `pack_pos_t` gives `t` bits 26-31, so a value
// of 64 or more would silently overflow into nothing and corrupt the
// word. This ties the packer's field width to the radius ceiling that
// feeds it, so the bound is checked at compile time rather than
// assumed at the call site.
const _: () = assert!(
    2 * MAX_TEMPORAL_RADIUS < 64,
    "pack_pos_t's 6-bit t field must hold every neighbour index collab_fused packs"
);

// A lane holds one 8-value column of each of `MAX_K` members, so the
// whole group fits `PATCH_AREA` slots only while the group size and the
// patch side are the same number. The stack transform's predicate
// ladder below also names the three levels 8, 4 and 2 outright.
const _: () = assert!(
    MAX_K == PATCH_SIZE && MAX_K == 8,
    "collab_fused's per-lane group array and its three-level stack transform are written for \
     MAX_K == PATCH_SIZE == 8"
);

// A candidate that never placed carries the distance `3.0e38`, written
// as a literal at each use below. A real distance is a sum of at most
// `PATCH_AREA` squared differences between values in `[0, 1]`, scaled
// by at most 3, so it never exceeds 192. `3.0e38` sits far above that
// and just below `f32::MAX`, so it always compares greater than a live
// candidate. The self-match takes `-1.0e38` at the other end, which
// sorts it below every real distance and pins it into slot 0. Both are
// literals rather than consts or `f32::INFINITY` because cubecl treats
// all of those as compile-time-only, and the shift-insert needs genuine
// mutable runtime variables.

/// Groups each reference patch with the patches most similar to it,
/// filters the whole group jointly with a hard threshold in the
/// transform domain, and scatters every filtered member back into its
/// own frame.
///
/// # Work decomposition
///
/// One cube of 64 threads owns eight reference patches. Each 8-lane
/// group owns one of them, and lane `sub` of a group owns column `sub`
/// of every patch that group touches. That one mapping serves both
/// halves of the kernel. A candidate's 64 pixel differences are spread
/// eight ways during matching, and [`plane_ssd_reduce8`] folds the eight
/// column sums into the whole patch distance. A member's 64 filtered
/// pixels are spread the same eight ways during filtering, so both the
/// candidate reads and the scatter writes are coalesced.
///
/// The reference patch's own column stays in registers for the whole
/// matching phase. Candidate pixels are read straight from global
/// memory. Neighbouring reference patches search heavily overlapping
/// windows at a step of 4, so the cache already serves those reads well
/// and a shared-memory tile would only cost occupancy.
///
/// A row of references rarely divides into eights, so the last cube of
/// a row runs groups whose reference patch is past the end. A 1080p
/// frame has 479 references across, so this is a shipped path rather
/// than an edge case. Those groups stay live through the whole kernel,
/// working on a clamped copy of the last real reference, and are gated
/// only where they would write.
///
/// # Barriers
///
/// [`transpose8`] carries the only barrier inside the group-processing
/// loops. Every lane of the cube reaches it the same number of times,
/// because the transposes sit in fully unrolled loops with no run-time
/// condition around them. Nothing returns early, a dead group runs the
/// whole kernel, and the group size only ever gates which iterations do
/// arithmetic, never how many barriers a lane reaches. A workgroup
/// barrier reached by only part of the workgroup is undefined, so that
/// property is what the write gating and the clamped reference index
/// exist to preserve.
///
/// The basis fill carries one more barrier, before either transform
/// runs. It is unconditional and sits before `live` is computed, so
/// every lane reaches it whatever the reference index later clamps to.
///
/// # Search space
///
/// The centre frame contributes the `spatial_radius` rectangle around
/// the reference patch, clipped to the frame.
///
/// Each volume anchor then searches every neighbour frame. It contributes
/// one `refine` rectangle per motion block whose span contains the
/// anchor, each around the position that block's vector predicts the
/// anchor moved to, clipped the same way. A block grid at a step below
/// `blksize` gives several such blocks, so an anchor is searched wherever
/// any block covering it points. A position reached by more than one of
/// them is scored once, by the first rectangle that reaches it.
///
/// Clipping each rectangle once keeps every candidate within it a
/// distinct position. Clamping each offset in turn would land several
/// offsets on the same edge position and let one physical patch count as
/// two.
///
/// # Distance
///
/// A candidate's distance is the channel-scaled sum of squared pixel
/// differences over the whole patch.
///
/// # Confidence gate
///
/// Every candidate stays in the running whatever its distance, so a
/// group fills wherever the search space is large enough. A covering
/// block whose confidence sits below `c_min` never runs the pixel
/// comparison, while the frame's other covering blocks still search.
/// The confidence comes from a motion block every lane of the group
/// shares, so the skip is uniform across the group. A volume left short
/// of frames by this gate makes the whole group fall back to the
/// single-frame group centred on the reference frame, so `c_min` can
/// change the output.
///
/// # Selection
///
/// The spatial search keeps the eight best centre-frame positions, one
/// per lane, ascending, through
/// [shift_insert8_gated](crate::collab::kernels::plane_ops::shift_insert8_gated).
/// A tie never displaces an incumbent, and the self-match scores a
/// sentinel below every real distance, which pins it into slot 0.
///
/// The first `MAX_K / grid_frames` of those become volume anchors. Each
/// anchor keeps its best match in every neighbour frame, and the volume
/// keeps the `grid_frames - 1` best of those in ascending order. A
/// position an earlier volume already holds is skipped, so no patch
/// enters the group twice.
///
/// # Members
///
/// A member is a packed position. The neighbour it came from sits in the
/// bits above the coordinates, so the frame it was matched in is
/// recovered from the packed word when matching ends. Member
/// `s * grid_frames + t` is frame `t` of volume `s`, with the anchor at
/// `t = 0`.
///
/// # Group size
///
/// A group uses the grid when the spatial search held at least `MAX_K`
/// positions, `k_max` is `MAX_K`, and every volume filled all of its
/// frames. Otherwise it falls back to the single-frame group, the
/// spatial search's positions rounded down to a power of two and capped
/// at `k_max`. The decision is uniform across the group.
///
/// # What the filter does
///
/// For each active channel, every member's patch runs through a 2D DCT,
/// so each patch is described by 64 frequency coefficients instead of 64
/// pixel values. A grid group then runs a Haar along time within each
/// volume and a Haar across the volumes, at each spatial position. A
/// fallback group runs a Haar across its stack instead. Content the group
/// agrees on collects into the low levels. A coefficient survives a hard
/// threshold when its magnitude reaches `lambda_ht` standard deviations
/// of its own propagated noise, where every member carries the plain
/// `sigma[c]^2`. Both transforms then invert.
///
/// The spatial pass runs as a column DCT in registers, a transpose, and
/// a row DCT in registers, because a lane owns a column and the row pass
/// needs a row. The inverse runs the same three steps backwards, which
/// leaves the lane holding a column again in time for the scatter.
///
/// The one coefficient that is both the group average and the patch's
/// spatial DC always survives the threshold, whatever its magnitude. A
/// group's mean brightness is signal, not something a noise threshold
/// should be able to zero out.
///
/// # Group weight
///
/// `group_weight` is `1 / sum(v_j)` over the coefficients the threshold
/// kept, computed from channel 0 only (luma dominates, and one weight
/// per group keeps aggregation simple downstream). When every member has
/// the same noise variance and the group keeps `n` coefficients this is
/// `1 / (sigma^2 * n)`, the usual inverse-variance weight, so a group
/// whose content agreed enough to keep more of its coefficients is
/// trusted more. Each lane sums the variance it retained over its own
/// eight positions and [`plane_ssd_reduce8`] folds the group's eight
/// partials together, which is why no shared array is needed for it.
///
/// # Buffers
///
/// `ring` is the frame ring, laid out one frame after another in
/// physical ring-slot order. `centre_slot` is the slot the pass is
/// centred on and `neighbour_slots` maps a packed neighbour index onto
/// its physical slot.
///
/// `accum` and `wsum` hold one region per ring slot, the layout
/// [`scatter_patch`] addresses, so a member matched in a neighbour frame
/// scatters into that frame's own region rather than the centre's.
/// `accum_scale` is the fixed-point scale that scatter converts into.
///
/// `group_weight` holds one weight per reference, and `sigma` one value
/// per stored channel.
///
/// `kaiser` holds [`crate::collab::kernels::aggregate::kaiser_window`]'s 8 taps, which
/// taper each scattered patch toward its edges. Eight ones leave the aggregation uniform.
///
/// `dct_profile` holds
/// [`crate::collab::kernels::transforms::dct_noise_profile`]'s 8 values.
/// Every member's coefficient variance at DCT position `(u, v)` scales
/// by `dct_profile[u] * dct_profile[v]` before the threshold reads it.
/// At `rho = 0` every entry is `1.0` and the multiply is a no-op.
///
/// `grid_frames` is the frames per volume, from
/// [grid_frames](crate::collab::grid_frames). At 1 the grid compiles out
/// and every group is a single-frame one.
///
/// # Warp-uniform search
///
/// `warp_uniform` decides how the spatial and trajectory searches are
/// walked.
///
/// Both searches are group-scoped work: each 8-lane group owns one
/// reference patch, and every distance is completed by a shuffle across
/// just those eight lanes. Nothing in the algorithm needs the other
/// groups sharing a warp to keep step.
///
/// The CUDA backend nevertheless lowers each of those shuffles to a
/// `__shfl_*_sync` naming the whole 32-lane warp. On Volta and later
/// such a shuffle waits for every lane it names, so a group still
/// searching blocks on groups that have already left the loop, and those
/// never come back. The clipped rectangles and the `c_min` skip both
/// give neighbouring groups different trip counts, so the warp
/// deadlocks and the launch never retires a frame.
///
/// Setting `warp_uniform` walks fixed, comptime-sized rectangles
/// instead, in both searches, and masks every position the other walk
/// skips, whether clipped, gated, already scored or already claimed.
/// Every group in a warp then takes the same number of turns through the
/// same shuffles. A masked turn carries the same `3.0e38` an unfilled slot
/// holds, so it can never displace one.
///
/// The candidates that do score, and the order they are offered in, are
/// exactly the ones the unset path visits, so both settings produce the
/// same group. Leave it unset on the wgpu backends, whose subgroup
/// operations reconverge on their own and which would only pay for the
/// dead turns. [`crate::collab::needs_warp_uniform_search`] is what
/// picks it per runtime.
///
/// # Compilation cost
///
/// The transforms unroll fully, which keeps the whole group in registers.
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
    dct_profile: &Array<f32>,
    kaiser: &Array<f32>,
    accum: &mut Array<Atomic<i32>>,
    wsum: &mut Array<Atomic<i32>>,
    group_weight: &mut Array<f32>,
    centre_slot: u32,
    c_min: f32,
    lambda_ht: f32,
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
) {
    let tid = UNIT_POS_X;
    let grp = tid / 8u32;
    let sub = tid % 8u32;
    let base = group_base();

    let max_x = comptime!(width - PATCH_SIZE);
    let max_y = comptime!(height - PATCH_SIZE);

    // The spatial basis, filled once and read by every lane for the rest
    // of the kernel. It is 256 B against the transpose buffer's 2,080 B,
    // and every lane reads all 64 of its entries, so keeping it shared
    // costs nothing a per-lane copy would save. Shared memory is not
    // what bounds this kernel's occupancy in any case, registers are.
    let mut basis = SharedMemory::<f32>::new(PATCH_AREA as usize);
    let mut tbuf = SharedMemory::<f32>::new(comptime!(8 * 65) as usize);
    fill_dct8_basis(&mut basis, tid);
    sync_cube();

    // A dead group keeps working on the last real reference of the row
    // so every read stays inside the frame and every lane reaches every
    // barrier. `live` is what stops it writing.
    let ref_x_index = CUBE_POS_X * 8u32 + grp;
    let live = ref_x_index < refs_x;
    let ref_x_clamped = ref_x_index.min(refs_x - 1u32);

    let rx = (ref_x_clamped * STEP).min(max_x);
    let ry = (CUBE_POS_Y * STEP).min(max_y);

    // Column `sub` of the reference patch, all channels, in registers
    // for the whole search.
    let mut current = Array::<f32>::new(comptime!(PATCH_SIZE * channels) as usize);
    #[unroll]
    for r in 0..PATCH_SIZE {
        let px = read_line(ring, rx + sub, ry + r, centre_slot, width, height);
        #[unroll]
        for c in 0..channels {
            current[(r * channels + c) as usize] = px[c as usize];
        }
    }

    let mut best_d = 3.0e38f32;
    let mut best_pos = 0u32;

    // One scalar for the whole kernel, from the channel count. It
    // multiplies the completed 64-pixel distance, not each squared
    // difference.
    let scale = channel_scale(channels);

    // The number of positions the spatial rectangle holds, which fixes the fallback group size
    // below.
    let n_live = spatial_search(
        ring,
        &current,
        rx,
        ry,
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
                let px = read_line(ring, anchor_x + sub, anchor_y + r, centre_slot, width, height);
                #[unroll]
                for c in 0..channels {
                    anchor[(r * channels + c) as usize] = px[c as usize];
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

        // A grid needs a full spatial search for its anchors and every volume's last frame
        // filled. The list is ascending, so a filled last slot means the whole volume is.
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

    // The frame each member sits in, from its packed word, once before the channel loop.
    //
    // The frame is picked with [`select`] rather than a branch. A frame index that reaches
    // [`read_line`] through a branch trips a bug in cubecl 0.10's global value numbering, which
    // panics while compiling the shader and leaves the launch to do nothing at all.
    let mut member_slot = Array::<u32>::new(MAX_K as usize);
    #[unroll]
    for m in 0..MAX_K {
        let packed = member_pos[m as usize];
        let mt = unpack_t(packed);
        // Clamped so the read below stays in range for a centre-frame member, whose value
        // `select` then discards. The clamp lands on index 0, so it needs `neighbour_slots` to
        // hold at least one entry. That is what every caller actually supplies, including
        // `radius = 0` launches such as `Setup::spatial_only` and the standalone launch
        // documented at `nl4d::tests::pipeline`, which still pass a one-element
        // `neighbour_slots` even though there is no real neighbour to read.
        let neighbour = u32::max(mt, 1u32) - 1u32;
        member_slot[m as usize] = select(mt > 0u32, neighbour_slots[neighbour as usize], centre_slot);
    }

    // The correlation profile is separable and the same for every
    // member, so the lane's own half of it is read once. Lane `sub`
    // ends up owning vertical frequency `sub` at every horizontal
    // frequency, see the transform order below.
    let prof_sub = dct_profile[sub as usize];

    // The group's normalised weight, computed from channel 0 and reused
    // by every later channel's scatter.
    let mut gw = 0.0f32;

    #[unroll]
    for c in 0..channels {
        let sigma_c = sigma[c as usize];
        let base_sig2 = sigma_c * sigma_c;

        // Column `sub` of every member, read out of the member's own
        // frame. Lane `sub` holds `stack[m * 8 + r]` for member `m`, row
        // `r`.
        let mut stack = Array::<f32>::new(PATCH_AREA as usize);
        let mut v = Array::<f32>::new(MAX_K as usize);
        #[unroll]
        for m in 0..MAX_K {
            let packed = member_pos[m as usize];
            let mx = packed & 0x1FFFu32;
            let my = (packed >> 13u32) & 0x1FFFu32;
            let src_slot = member_slot[m as usize];
            v[m as usize] = base_sig2;
            #[unroll]
            for r in 0..PATCH_SIZE {
                let px = read_line(ring, mx + sub, my + r, src_slot, width, height);
                stack[(m * PATCH_SIZE + r) as usize] = px[c as usize];
            }
        }

        // The noise variance behind each member, propagated to a
        // per-stack-level variance. The spatial profile is a constant
        // factor across the stack axis and the ladder only averages, so
        // it multiplies in at the threshold instead of here.
        if comptime!(grid_frames > 1) {
            if use_grid {
                grid_variance(&mut v, grid_frames);
            } else {
                stack_variance_ladder(&mut v, k_use);
            }
        } else {
            stack_variance_ladder(&mut v, k_use);
        }

        // 2D DCT forward, independently for each member's patch. The
        // column pass runs over the rows the lane already holds, the
        // transpose hands the lane a row, and the row pass runs over
        // that. Lane `sub` comes out holding coefficient `(u = i, v =
        // sub)` at slot `i`.
        #[unroll]
        for m in 0..MAX_K {
            let mut line = Array::<f32>::new(PATCH_SIZE as usize);
            #[unroll]
            for i in 0..PATCH_SIZE {
                line[i as usize] = stack[(m * PATCH_SIZE + i) as usize];
            }
            dct8_reg_fwd(&basis, &mut line);
            transpose8(&mut tbuf, &mut line, sub, grp);
            dct8_reg_fwd(&basis, &mut line);
            #[unroll]
            for i in 0..PATCH_SIZE {
                stack[(m * PATCH_SIZE + i) as usize] = line[i as usize];
            }
        }

        // Haar transform along the stack axis, at each of the lane's
        // eight spatial positions. A lane owns every member at every
        // position it holds, so nothing crosses lanes here.
        if comptime!(grid_frames > 1) {
            if use_grid {
                grid_fwd(&mut stack, grid_frames);
            } else {
                stack_haar_fwd(&mut stack, k_use);
            }
        } else {
            stack_haar_fwd(&mut stack, k_use);
        }

        // Hard threshold, and the group-DC exception described above.
        // The lane's retained variance is summed here and folded across
        // the group below.
        let mut retained_v = 0.0f32;
        #[unroll]
        for i in 0..PATCH_SIZE {
            let factor = dct_profile[i as usize] * prof_sub;
            #[unroll]
            for j in 0..MAX_K {
                if j < k_use {
                    let vj = v[j as usize] * factor;
                    let slot = (j * PATCH_SIZE + i) as usize;
                    let mut keep = f32::abs(stack[slot]) >= lambda_ht * f32::sqrt(vj);
                    if comptime!(j == 0u32 && i == 0u32) {
                        if sub == 0u32 {
                            keep = true;
                        }
                    }
                    if keep {
                        retained_v += vj;
                    } else {
                        stack[slot] = 0.0f32;
                    }
                }
            }
        }

        // The group weight has to be known before the scatter below, and
        // only the first channel computes it, so the reduction runs here
        // rather than after the inverse transforms.
        if comptime!(c == 0u32) {
            let sum = plane_ssd_reduce8(retained_v);
            // `sum` adds non-negative variances, so it is never
            // negative. `safe_reciprocal` checks for a non-finite sum
            // explicitly rather than leaning on `f32::max` to discard
            // one, so the weight is finite here whatever a given GPU
            // does with NaN.
            let w = safe_reciprocal(sum, RECIPROCAL_FLOOR);
            if live && sub == 0u32 {
                group_weight[ref_idx as usize] = w;
            }
            // The accumulators count in fixed point, so the weight is
            // scaled into the band `weight_scale` was built to put it
            // in. Aggregation normalises by the weight sum, so scaling
            // every weight by the same constant leaves the result
            // exactly as it would have been.
            gw = w * weight_scale;
        }

        // Haar inverse, back from stack coefficients to per-member DCT
        // coefficients, then the spatial inverse in the opposite order
        // to the forward pass. The lane holds a column again by the end
        // of it, which is what makes the scatter below coalesced.
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
            transpose8(&mut tbuf, &mut line, sub, grp);
            dct8_reg_inv(&basis, &mut line);
            #[unroll]
            for i in 0..PATCH_SIZE {
                stack[(m * PATCH_SIZE + i) as usize] = line[i as usize];
            }
        }

        // Every member of the group is written back, not just the
        // reference patch, and each lands in its own frame's region of
        // the accumulators. A neighbour-frame member therefore feeds the
        // caller's cross-frame ring rather than being discarded once it
        // has served the group's shared statistics.
        #[unroll]
        for m in 0..MAX_K {
            if live && m < k_use {
                let packed = member_pos[m as usize];
                let mx = packed & 0x1FFFu32;
                let my = (packed >> 13u32) & 0x1FFFu32;
                let dst_slot = member_slot[m as usize];
                #[unroll]
                for r in 0..PATCH_SIZE {
                    scatter_patch(
                        accum,
                        wsum,
                        kaiser,
                        stack[(m * PATCH_SIZE + r) as usize],
                        gw,
                        mx,
                        my,
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

/// The inverse of [stack_haar_fwd].
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

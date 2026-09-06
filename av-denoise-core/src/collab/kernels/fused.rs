use cubecl::prelude::*;

use super::aggregate::scatter_patch;
use super::group::{clamp_top_left, pack_pos_t, unpack_t};
use super::plane_ops::{group_base, plane_ssd_reduce8, shift_insert8, shift_insert8_gated, transpose8};
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

/// The most extra variance a temporal member's mismatch may carry,
/// as a multiple of the channel's own variance.
///
/// A member's extra variance is its own match distance, which on a
/// badly matched patch has no relation to the channel sigma the group
/// weight is normalised against. Left uncapped it makes the retained
/// variance sum, and so the weight, unbounded below, and a weight small
/// enough to round away in the accumulators takes its pixel's only
/// information with it. See
/// [`crate::collab::kernels::aggregate::weight_scale`].
///
/// Capping restores the bound. A member here is already a 64 times
/// noisier observation than the channel it came from, which the
/// threshold treats as carrying almost nothing, so holding it at that
/// rather than letting it run further costs no filtering and buys a
/// weight that always survives the conversion to fixed point.
pub(crate) const MEMBER_SIGMA2_CAP: f32 = 64.0;

/// The lowest block index whose span contains the patch at `p` on one axis.
///
/// Block `b` spans `b * step..b * step + blksize`, so the patch
/// `p..p + PATCH_SIZE` needs `b * step + blksize >= p + PATCH_SIZE`.
/// The highest such block is `p / step`, which the caller clamps to the
/// grid and uses as the low end's ceiling.
///
/// This mirrors `covering_blocks` in the `mc_accuracy` bench's harness
/// module (`av-denoise-core/benches/harness/score.rs`), which the tests
/// below reproduce on the host to check the two stay in step.
#[cube]
fn covering_lo(p: u32, #[comptime] blksize: u32, #[comptime] step: u32) -> u32 {
    let past = u32::max(p + PATCH_SIZE, blksize) - blksize;
    past.div_ceil(step)
}

/// The host mirror of [`covering_lo`], for tests that cannot launch a
/// kernel.
#[cfg(test)]
fn covering_lo_host(p: u32, blksize: u32, step: u32) -> u32 {
    let past = u32::max(p + PATCH_SIZE, blksize) - blksize;
    past.div_ceil(step)
}

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
/// the reference patch, clipped to the frame. Each neighbour
/// contributes one `refine` rectangle per motion block whose span
/// contains the reference patch, each around the position that block's
/// vector predicts the patch moved to, clipped the same way. A block
/// grid at a step below `blksize` gives several such blocks, and taking
/// all of them means a patch is searched wherever any block covering it
/// points rather than only where its corner block points.
///
/// Rectangles from different blocks of one neighbour overlap when their
/// vectors are close. A position reached by more than one of them is
/// scored once, by the first rectangle that reaches it, and the later
/// rectangles skip it.
///
/// Clipping the rectangle once is what keeps every candidate within it a
/// distinct position. Clamping each offset in turn would land several
/// offsets on the same edge position, and admitting a position twice
/// would let one physical patch count as two and look like stronger
/// agreement than the group has. The overlap check across rectangles is
/// the same property held across the blocks of one neighbour.
///
/// # Distance
///
/// A candidate's distance is the channel-scaled sum of squared pixel
/// differences over the whole patch, minus `noise_floor`. `noise_floor`
/// is the distance two noisy copies of the same content show by chance,
/// so a genuine match is not penalised for the noise it carries. The
/// result is not clamped at zero for ranking, because subtracting a
/// constant from every candidate shifts them all equally. It is clamped
/// at zero where it becomes a member's mismatch variance below, so
/// `noise_floor` has to be the real expected distance of two noisy
/// copies, `channel_scale * 2 * PATCH_AREA * sum(sigma_c^2)`.
///
/// # No admission gate
///
/// Every candidate stays in the running whatever its distance, so a
/// group fills to `k_max` wherever the search space is that large.
/// `c_min` is a compute saving rather than an admission threshold. The
/// skip is per block. A covering block whose confidence sits below
/// `c_min` never runs the pixel comparison, and its whole rectangle is
/// skipped, while the neighbour's other covering blocks still search.
/// The confidence comes from a motion block that every lane of the
/// group shares, so the skip is uniform across the group.
///
/// # Selection
///
/// The eight best candidates live one per lane, ascending, and each
/// candidate is offered to [`shift_insert8_gated`] as it is scored. A
/// candidate that ties an incumbent does not displace it, so the first
/// candidate seen at a given distance keeps its slot. The self-match is
/// scored with a sentinel distance below every real one, which pins it
/// into slot 0 without a special case in the loop.
///
/// # Members
///
/// A member is a `(distance, position)` pair for the whole search, and
/// nothing else rides along. The position packs the neighbour it came
/// from into the bits above the coordinates, so the frame it was matched
/// in and its motion-block confidence are both recovered from the packed
/// word when matching ends. Carrying either through the insert would
/// cost a shuffle on every candidate.
///
/// The member set never leaves the kernel. Lane `i` holds member `i`, so
/// one broadcast per member is all the filter stage needs to give every
/// lane every position.
///
/// # Group size
///
/// The member count is the search space size rounded down to the
/// nearest power of two, capped at `k_max`. The stack transform is only
/// defined for power-of-two stack sizes, so a count of 5, 6, or 7 keeps
/// only 4 members. Every rectangle this kernel searches at shipped
/// settings is far larger than `k_max`, so the rounding only bites on
/// frames small enough for the clipped rectangle to hold fewer than
/// eight positions.
///
/// # What the filter does
///
/// For each active channel, every member's patch runs through a 2D DCT,
/// so each patch is described by 64 frequency coefficients instead of 64
/// pixel values. A Haar transform then runs across the stack axis, at
/// each spatial position independently, so content the group agrees on
/// collects into the low stack levels and content only one or two
/// members carry lands in the higher ones. A coefficient survives a hard
/// threshold when its magnitude reaches `lambda_ht` standard deviations
/// of its own propagated noise, with [`variance_reg_level`] propagating
/// the per-member variance to each stack level. A member matched in a
/// neighbour frame carries its own match distance as extra variance,
/// `mismatch_scale2 * max(distance, 0) / (3 * PATCH_AREA)`, which is
/// the per-channel, per-pixel mean square of its mismatch, so a poorer
/// match is a noisier observation. Both transforms then invert.
///
/// The spatial pass runs as a column DCT in registers, a transpose, and
/// a row DCT in registers, because a lane owns a column and the row pass
/// needs a row. The inverse runs the same three steps backwards, which
/// leaves the lane holding a column again in time for the scatter.
///
/// The one coefficient that is both the group average (Haar level 0) and
/// the patch's spatial DC (DCT position 0) always survives the
/// threshold, whatever its magnitude. A group's mean brightness is
/// signal, not something a noise threshold should be able to zero out.
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
/// `use_member_sigma` folds each temporal member's mismatch variance
/// into its own noise variance. False leaves every member on the plain
/// `sigma[c]^2`.
///
/// # Warp-uniform search
///
/// `warp_uniform` decides how the two candidate searches are walked.
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
/// instead and masks the positions falling outside the clipped one, so
/// every group in a warp takes the same number of turns through the
/// same shuffles. The masked turns score nothing: a candidate that is
/// not live carries the same `3.0e38` an unfilled slot holds, so it can
/// never displace one.
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
/// The group stays in registers because the transform loops unroll
/// fully rather than looping at run time. That unrolling is expensive
/// to compile, the unrolled IR is 6,513 instructions for the luma
/// variant and 11,903 for the chroma variant, and cubecl spends about
/// 9.5 s compiling the two of them at startup. That cost is the price
/// of the register-resident design, not a bug to fix by shrinking the
/// unroll.
/// The distance from the reference patch to the candidate whose
/// top-left pixel is `(x, y)` in frame `slot`.
///
/// Each lane holds one column of the reference patch and reads the
/// matching column of the candidate, so the eight per-lane partials
/// only become a whole-patch distance through
/// [`plane_ssd_reduce8`]. That reduction shuffles, so every lane of the
/// group has to reach it. Callers that end up discarding the result
/// still call this and drop the value afterwards rather than branching
/// around it.
#[cube]
fn candidate_distance<N: Size>(
    ring: &Array<Vector<f32, N>>,
    current: &Array<f32>,
    x: u32,
    y: u32,
    slot: u32,
    sub: u32,
    scale: f32,
    noise_floor: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
) -> f32 {
    let mut partial = 0.0f32;
    #[unroll]
    for r in 0..PATCH_SIZE {
        let px = read_line(ring, x + sub, y + r, slot, width, height);
        #[unroll]
        for c in 0..channels {
            let d = current[(r * channels + c) as usize] - px[c as usize];
            partial += d * d;
        }
    }
    plane_ssd_reduce8(partial) * scale - noise_floor
}

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
    noise_floor: f32,
    c_min: f32,
    mismatch_scale2: f32,
    lambda_ht: f32,
    weight_scale: f32,
    accum_scale: f32,
    #[comptime] use_member_sigma: bool,
    #[comptime] warp_uniform: bool,
    #[comptime] radius: u32,
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

    // The blocks a temporal candidate reads its motion vectors and
    // confidences from depend only on `rx` and `ry`, which are the same
    // for every candidate this group scores, so the range is worked out
    // once. The corner block, the one the patch's own top-left pixel
    // sits in, is `(bx_hi, by_hi)`, and a range whose low end equals its
    // high end searches that block alone.
    let bx_hi = (rx / blk_step).min(blocks_x - 1);
    let by_hi = (ry / blk_step).min(blocks_y - 1);
    let bx_lo = u32::min(covering_lo(rx, blksize, blk_step), bx_hi);
    let by_lo = u32::min(covering_lo(ry, blksize, blk_step), by_hi);

    // The number of positions actually scored, which fixes the group
    // size below. Rectangles in different frames cannot collide, and
    // within a frame a repeated position is counted once, so every
    // increment is a distinct position.
    let mut n_live = 0u32;

    // The spatial rectangle, clipped once.
    let s_left = clamp_top_left(rx as i32 - spatial_radius as i32, max_x);
    let s_right = clamp_top_left(rx as i32 + spatial_radius as i32, max_x);
    let s_top = clamp_top_left(ry as i32 - spatial_radius as i32, max_y);
    let s_bot = clamp_top_left(ry as i32 + spatial_radius as i32, max_y);
    n_live += (s_right - s_left + 1u32) * (s_bot - s_top + 1u32);

    // The reference patch scores the lowest distance there is, which on
    // textured content is enough to reach slot 0 on its own. On flat
    // content every candidate scores that same distance, and
    // `shift_insert8` leaves a tie with whichever candidate reached the
    // slot first. A sentinel below every real distance pins the
    // self-match whatever ties around it.
    if warp_uniform {
        // The clipped rectangle is never wider than the unclipped one,
        // so walking the unclipped span covers every position the other
        // path visits, in the same order, and the rest are masked. The
        // span is comptime, so every group in the warp takes the same
        // number of turns.
        let span = comptime!(2 * spatial_radius + 1);
        for dy in 0..span {
            for dx in 0..span {
                let wanted_y = s_top + dy;
                let wanted_x = s_left + dx;
                let live_pos = wanted_x <= s_right && wanted_y <= s_bot;
                // A masked turn still reads, so it is pinned to the last
                // live position rather than left to run off the frame.
                let cx = u32::min(wanted_x, s_right);
                let cy = u32::min(wanted_y, s_bot);

                let scored = candidate_distance(
                    ring,
                    &current,
                    cx,
                    cy,
                    centre_slot,
                    sub,
                    scale,
                    noise_floor,
                    width,
                    height,
                    channels,
                );
                // Only the branchless part of the insert is shared. The
                // gated form tests a group-local distance before it
                // shuffles, which is exactly the divergence this path
                // exists to avoid.
                let mut dist = select(live_pos, scored, 3.0e38f32);
                // A masked turn can land on the reference's own position
                // once it has been pinned, so `live_pos` has to gate the
                // sentinel too, or a dead turn would plant a second
                // self-match in the group.
                if live_pos && cx == rx && cy == ry {
                    dist = -1.0e38f32;
                }
                shift_insert8(&mut best_d, &mut best_pos, dist, pack_pos_t(cx, cy, 0u32), sub);
            }
        }
    } else {
        let mut cy = s_top;
        while cy <= s_bot {
            let mut cx = s_left;
            while cx <= s_right {
                let mut dist = candidate_distance(
                    ring,
                    &current,
                    cx,
                    cy,
                    centre_slot,
                    sub,
                    scale,
                    noise_floor,
                    width,
                    height,
                    channels,
                );
                if cx == rx && cy == ry {
                    dist = -1.0e38f32;
                }
                shift_insert8_gated(
                    &mut best_d,
                    &mut best_pos,
                    dist,
                    pack_pos_t(cx, cy, 0u32),
                    sub,
                    base,
                );
                cx += 1u32;
            }
            cy += 1u32;
        }
    }

    // One clipped rectangle per covering block per neighbour, around
    // that block's motion-predicted centre.
    let n_neighbours = comptime!(2 * radius);
    // The widest block range `covering_lo` can produce on one axis, so
    // the block loops unroll and every `seen_*` index is a constant.
    let covers = comptime!(blksize.div_ceil(blk_step));
    let max_rects = comptime!(covers * covers);
    let mut t = 0u32;
    while t < n_neighbours {
        let slot = neighbour_slots[t as usize];
        // `t + 1` is the neighbour field's value, one past the centre
        // frame's 0. The module-level assert above bounds it well inside
        // the six bits `pack_pos_t` gives it.
        let packed_t = t + 1u32;

        // The rectangles already searched for this neighbour, one slot
        // per covering block in visiting order. A slot starts empty,
        // `left` above `right`, which no position matches, so a slot
        // whose block the scan has not reached yet hides nothing and a
        // block the range or `c_min` skips leaves its slot empty.
        let mut seen_left = Array::<u32>::new(max_rects as usize);
        let mut seen_right = Array::<u32>::new(max_rects as usize);
        let mut seen_top = Array::<u32>::new(max_rects as usize);
        let mut seen_bot = Array::<u32>::new(max_rects as usize);
        #[unroll]
        for s in 0..max_rects {
            seen_left[s as usize] = 1u32;
            seen_right[s as usize] = 0u32;
            seen_top[s as usize] = 1u32;
            seen_bot[s as usize] = 0u32;
        }

        #[unroll]
        for iy in 0..covers {
            #[unroll]
            for ix in 0..covers {
                let wanted_bx = bx_lo + ix;
                let wanted_by = by_lo + iy;
                let block_live = wanted_bx <= bx_hi && wanted_by <= by_hi;

                if warp_uniform {
                    // Both the block range and the `c_min` test decide
                    // per group, so neither can gate a shuffle here.
                    // The block is pinned into range, read either way,
                    // and what it found is folded into `live_pos` below.
                    let cbx = u32::min(wanted_bx, bx_hi);
                    let cby = u32::min(wanted_by, by_hi);
                    let block = cby * blocks_x + cbx;
                    let conf = confidence[(t * conf_stride + block) as usize];
                    let block_scored = block_live && conf >= c_min;

                    let mv = (t * mv_stride + block * 2u32) as usize;
                    let px0 = rx as i32 + mv_field[mv];
                    let py0 = ry as i32 + mv_field[mv + 1];

                    let t_left = clamp_top_left(px0 - refine as i32, max_x);
                    let t_right = clamp_top_left(px0 + refine as i32, max_x);
                    let t_top = clamp_top_left(py0 - refine as i32, max_y);
                    let t_bot = clamp_top_left(py0 + refine as i32, max_y);

                    let span = comptime!(2 * refine + 1);
                    for dy in 0..span {
                        for dx in 0..span {
                            let wanted_y = t_top + dy;
                            let wanted_x = t_left + dx;
                            let in_rect = wanted_x <= t_right && wanted_y <= t_bot;
                            let nx = u32::min(wanted_x, t_right);
                            let ny = u32::min(wanted_y, t_bot);

                            let mut covered = false;
                            #[unroll]
                            for s in 0..max_rects {
                                if nx >= seen_left[s as usize]
                                    && nx <= seen_right[s as usize]
                                    && ny >= seen_top[s as usize]
                                    && ny <= seen_bot[s as usize]
                                {
                                    covered = true;
                                }
                            }

                            let live_pos = block_scored && in_rect && !covered;
                            if live_pos {
                                n_live += 1u32;
                            }

                            let scored = candidate_distance(
                                ring,
                                &current,
                                nx,
                                ny,
                                slot,
                                sub,
                                scale,
                                noise_floor,
                                width,
                                height,
                                channels,
                            );
                            shift_insert8(
                                &mut best_d,
                                &mut best_pos,
                                select(live_pos, scored, 3.0e38f32),
                                pack_pos_t(nx, ny, packed_t),
                                sub,
                            );
                        }
                    }

                    // A block the range or `c_min` skipped has to leave
                    // its slot empty, the way the other path leaves it
                    // untouched, or it would hide positions a later
                    // block still owes the search.
                    seen_left[(iy * covers + ix) as usize] = select(block_scored, t_left, 1u32);
                    seen_right[(iy * covers + ix) as usize] = select(block_scored, t_right, 0u32);
                    seen_top[(iy * covers + ix) as usize] = select(block_scored, t_top, 1u32);
                    seen_bot[(iy * covers + ix) as usize] = select(block_scored, t_bot, 0u32);
                } else {
                    let cbx = wanted_bx;
                    let cby = wanted_by;
                    if block_live {
                        let block = cby * blocks_x + cbx;
                        let conf = confidence[(t * conf_stride + block) as usize];
                        // Uniform across the group, because `block` is, so a
                        // skipped block costs no lane its share of the
                        // reduction. No barrier sits inside this branch
                        // either, so a group that skips a block a
                        // neighbouring group scores strands nothing.
                        if conf >= c_min {
                            let mv = (t * mv_stride + block * 2u32) as usize;
                            let px0 = rx as i32 + mv_field[mv];
                            let py0 = ry as i32 + mv_field[mv + 1];

                            let t_left = clamp_top_left(px0 - refine as i32, max_x);
                            let t_right = clamp_top_left(px0 + refine as i32, max_x);
                            let t_top = clamp_top_left(py0 - refine as i32, max_y);
                            let t_bot = clamp_top_left(py0 + refine as i32, max_y);

                            let mut ny = t_top;
                            while ny <= t_bot {
                                let mut nx = t_left;
                                while nx <= t_right {
                                    let mut covered = false;
                                    #[unroll]
                                    for s in 0..max_rects {
                                        if nx >= seen_left[s as usize]
                                            && nx <= seen_right[s as usize]
                                            && ny >= seen_top[s as usize]
                                            && ny <= seen_bot[s as usize]
                                        {
                                            covered = true;
                                        }
                                    }
                                    if !covered {
                                        n_live += 1u32;
                                        let dist = candidate_distance(
                                            ring,
                                            &current,
                                            nx,
                                            ny,
                                            slot,
                                            sub,
                                            scale,
                                            noise_floor,
                                            width,
                                            height,
                                            channels,
                                        );
                                        shift_insert8_gated(
                                            &mut best_d,
                                            &mut best_pos,
                                            dist,
                                            pack_pos_t(nx, ny, packed_t),
                                            sub,
                                            base,
                                        );
                                    }
                                    nx += 1u32;
                                }
                                ny += 1u32;
                            }

                            seen_left[(iy * covers + ix) as usize] = t_left;
                            seen_right[(iy * covers + ix) as usize] = t_right;
                            seen_top[(iy * covers + ix) as usize] = t_top;
                            seen_bot[(iy * covers + ix) as usize] = t_bot;
                        }
                    }
                }
            }
        }

        t += 1u32;
    }

    // Retire the group. Lane `i` holds member `i`, so one broadcast per
    // member hands every lane every position, once for the whole filter
    // rather than once per channel.
    let ref_idx = CUBE_POS_Y * refs_x + ref_x_clamped;

    let mut k_use = 1u32;
    while k_use * 2u32 <= n_live && k_use * 2u32 <= k_max {
        k_use *= 2u32;
    }

    // Where each member sits, which frame it sits in, and the extra
    // variance its motion block's confidence implies. All three come out
    // of the one packed word, once, before the channel loop.
    //
    // The frame is picked with [`select`] rather than a branch. A frame
    // index that reaches [`read_line`] through a branch trips a bug in
    // cubecl 0.10's global value numbering, which panics while compiling
    // the shader and leaves the launch to do nothing at all.
    let mut member_pos = Array::<u32>::new(MAX_K as usize);
    let mut member_slot = Array::<u32>::new(MAX_K as usize);
    let mut member_sig2 = Array::<f32>::new(MAX_K as usize);
    #[unroll]
    for m in 0..MAX_K {
        let packed = plane_shuffle(best_pos, base + m);
        let mt = unpack_t(packed);
        // Clamped so the read below stays in range for a centre-frame
        // member, whose value `select` then discards. The clamp lands on
        // index 0, so it needs `neighbour_slots` to hold at least one
        // entry. That is what every caller actually supplies, including
        // `radius = 0` launches such as `Setup::spatial_only` and the
        // standalone launch documented at `nl4d::tests::pipeline`,
        // which still pass a one-element `neighbour_slots` even though
        // there is no real neighbour to read.
        let n = u32::max(mt, 1u32) - 1u32;
        member_pos[m as usize] = packed;
        member_slot[m as usize] = select(mt > 0u32, neighbour_slots[n as usize], centre_slot);

        let mut sig2 = 0.0f32;
        if use_member_sigma {
            if warp_uniform {
                // `mt` is the member's own frame, so this test decides
                // per member and per group, and the broadcast under it
                // is warp-wide once CUDA has lowered it. Reading the
                // distance first and discarding it afterwards keeps
                // every lane on the same broadcast; `select` is what
                // makes a centre-frame member ignore what it read.
                let excess = f32::max(plane_shuffle(best_d, base + m), 0.0f32);
                let mismatch = mismatch_scale2 * excess / comptime!(3 * PATCH_AREA) as f32;
                sig2 = select(mt > 0u32, mismatch, 0.0f32);
            } else {
                // A centre-frame member is not motion-predicted, so there is
                // no mismatch to model and it keeps the plain `sigma^2`.
                if mt > 0u32 {
                    // The member's own distance, floor removed, in the search's
                    // three-channel-sum units. Per channel and per pixel that
                    // is the mean square of its mismatch.
                    let excess = f32::max(plane_shuffle(best_d, base + m), 0.0f32);
                    sig2 = mismatch_scale2 * excess / comptime!(3 * PATCH_AREA) as f32;
                }
            }
        }
        member_sig2[m as usize] = sig2;
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
            // Capped against this channel's own variance, so the
            // retained sum stays within a known factor of the smallest
            // one `weight_scale` normalises by. See `MEMBER_SIGMA2_CAP`.
            let extra = f32::min(member_sig2[m as usize], MEMBER_SIGMA2_CAP * base_sig2);
            v[m as usize] = base_sig2 + extra;
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
        if k_use >= 8u32 {
            variance_reg_level(&mut v, 8u32);
        }
        if k_use >= 4u32 {
            variance_reg_level(&mut v, 4u32);
        }
        if k_use >= 2u32 {
            variance_reg_level(&mut v, 2u32);
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
        if k_use >= 8u32 {
            haar_reg_fwd_level(&mut stack, 8u32);
        }
        if k_use >= 4u32 {
            haar_reg_fwd_level(&mut stack, 4u32);
        }
        if k_use >= 2u32 {
            haar_reg_fwd_level(&mut stack, 2u32);
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
            //
            // The band's lower bound is what `MEMBER_SIGMA2_CAP` exists
            // to restore, see its doc.
            gw = w * weight_scale;
        }

        // Haar inverse, back from stack coefficients to per-member DCT
        // coefficients, then the spatial inverse in the opposite order
        // to the forward pass. The lane holds a column again by the end
        // of it, which is what makes the scatter below coalesced.
        if k_use >= 2u32 {
            haar_reg_inv_level(&mut stack, 2u32);
        }
        if k_use >= 4u32 {
            haar_reg_inv_level(&mut stack, 4u32);
        }
        if k_use >= 8u32 {
            haar_reg_inv_level(&mut stack, 8u32);
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

#[cfg(test)]
mod tests {
    use super::covering_lo_host;

    /// The host mirror of `covering_blocks` in the `mc_accuracy` bench's
    /// harness (`benches/harness/score.rs`). Reproduced here, rather than
    /// imported, because that module lives outside the crate as bench-only
    /// code and cannot be a test dependency of the library.
    ///
    /// This pins the kernel's arithmetic against the harness's read of the
    /// same geometry rather than launching a real kernel, so it catches the
    /// two formulas drifting apart on paper but says nothing about whether
    /// [`super::covering_lo`] compiles or runs correctly on a GPU; the
    /// integration tests in `nl4d::tests` cover that by driving the whole
    /// pipeline.
    fn covering_blocks_host(p: u32, blksize: u32, step: u32, blocks: u32) -> (u32, u32) {
        let hi = (p / step).min(blocks - 1);
        let lo = if p + super::PATCH_SIZE <= blksize {
            0
        } else {
            (p + super::PATCH_SIZE - blksize).div_ceil(step)
        };
        (lo.min(hi), hi)
    }

    #[test]
    fn covering_lo_matches_the_harness_across_a_range_of_geometries() {
        for (blksize, overlap) in [(16u32, 8u32), (16, 12), (32, 24), (8, 4), (16, 0)] {
            let step = blksize - overlap;
            let blocks = 8u32;
            for p in (0..blocks * step).step_by(3) {
                let (expect_lo, hi) = covering_blocks_host(p, blksize, step, blocks);
                let got_lo = covering_lo_host(p, blksize, step).min(hi);
                assert_eq!(
                    got_lo, expect_lo,
                    "blksize={blksize} step={step} p={p}: covering_lo disagrees with the \
                     harness's covering_blocks"
                );
            }
        }
    }
}

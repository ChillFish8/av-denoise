use cubecl::prelude::*;

use crate::collab::kernels::aggregate::scatter_patch;
use crate::collab::kernels::transforms::{
    RECIPROCAL_FLOOR,
    dct8_line_fwd,
    dct8_line_inv,
    fill_dct8_basis,
    fill_haar8_basis,
    haar_fwd_stack,
    haar_inv_stack,
    safe_reciprocal,
};
use crate::collab::{MAX_K, PATCH_AREA, PATCH_SIZE};
use crate::nlmeans::kernels::helpers::read_line;

// Ties the `stride = 32u32` literal in the group-weight reduction below
// back to `PATCH_AREA`, the same way `collab_group_temporal` ties its own
// copy of this literal (see that kernel for why the literal can't be
// written as `PATCH_AREA / 2` directly).
const _: () = assert!(
    PATCH_AREA / 2 == 32,
    "update the `stride = 32u32` literal in collab_filter_ht to PATCH_AREA / 2"
);

// Ties the hardcoded `8usize`/`8u32` register-array sizes in
// `variance_ladder` and in the per-channel setup loop above it back to
// `MAX_K`, the same way `haar_fwd_stack`/`haar_inv_stack` tie theirs.
const _: () = assert!(
    MAX_K == 8,
    "update the hardcoded 8usize/8u32 slots in variance_ladder and its callers to MAX_K"
);

/// Propagates a per-member noise variance through the same butterfly
/// [`haar_fwd_stack`] applies to the signal.
///
/// `v` holds one variance per stack member on entry, at indices `0..8`.
/// Only `0..k_use` matter, the rest are ignored.
///
/// A Haar butterfly's two outputs are each a sum of two independent
/// inputs scaled by `1/sqrt(2)`, so both land on the same variance,
/// `(va + vb) / 2`. Running that averaging over the same levels, in the
/// same pairing order, as [`haar_fwd_stack`] runs over the signal turns
/// `v[j]` into the variance of stack coefficient `j`.
///
/// [`crate::collab::kernels::transforms::haar_variance_ladder`] runs the
/// identical computation on the host, for comparison in tests.
///
/// Every level snapshots the values it reads before writing any output,
/// because the output range overlaps the input range and an in-place
/// write would otherwise clobber a value a later pair still needs.
#[cube]
pub(crate) fn variance_ladder(v: &mut Array<f32>, k_use: u32) {
    let mut len = k_use;
    while len > 1u32 {
        let half = len / 2u32;

        let mut snapshot = Array::<f32>::new(8usize);
        #[unroll]
        for k in 0..8u32 {
            snapshot[k as usize] = v[k as usize];
        }

        let mut p = 0u32;
        while p < half {
            let va = snapshot[(2u32 * p) as usize];
            let vb = snapshot[(2u32 * p + 1u32) as usize];
            let avg = (va + vb) * 0.5f32;
            v[p as usize] = avg;
            v[(half + p) as usize] = avg;
            p += 1u32;
        }

        len = half;
    }
}

/// Filters one group of similar patches with a hard threshold in the
/// transform domain, and writes back the filtered reference patch.
///
/// One cube owns one reference patch's group, with the same `member_pos`
/// / `member_count` layout `collab_group_temporal` produces. Its
/// `CubeDim::new_2d(8, 8)` threads map one-to-one onto a patch's 64
/// pixels.
///
/// # What the filter does
///
/// For each active channel, every member's patch is loaded and run
/// through a 2D DCT, so each patch is described by 64 frequency
/// coefficients instead of 64 pixel values. A Haar transform then runs
/// across the stack axis, at each spatial position independently, so
/// content the group agrees on collects into the low stack levels and
/// content only one or two members carry lands in the higher ones. A
/// coefficient survives a hard threshold when its magnitude reaches
/// `lambda_ht` standard deviations of its own propagated noise, using
/// [`variance_ladder`] to know what that noise is at each stack level.
/// Both transforms then invert, and the result is member 0's patch, the
/// reference patch itself, filtered by whatever the rest of the group
/// agreed with it on.
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
/// `1 / (sigma^2 * n)`, the usual inverse-variance weight, a group whose
/// content agreed enough to keep more of its coefficients is trusted
/// more.
///
/// # Buffers
///
/// `input` is the frame ring
/// [`crate::collab::kernels::group_temporal::collab_group_temporal`] also
/// reads.
///
/// `member_pos`, `member_count`, `member_frame`, and `member_sig2` are
/// that kernel's outputs, all in its `ref_idx * k_max + m` layout except
/// `member_count`, which holds one count per reference.
///
/// `member_frame` is read only when `temporal` is true, and `member_sig2`
/// only when `use_member_sigma` is true. A 1-element dummy buffer is
/// valid for either when its flag is off.
///
/// `filtered` holds `refs * k_max * PATCH_AREA` lines and is written only
/// when `emit_filtered` is set. `group_weight` holds `refs` weights, and
/// `sigma` one value per stored channel.
///
/// `accum_scale` is the fixed-point scale the scatter at the end of this
/// kernel converts into. See
/// [`crate::collab::kernels::aggregate::scatter_patch`].
///
/// `dct_profile` holds
/// [`crate::collab::kernels::transforms::dct_noise_profile`]'s 8 values.
/// Every member's coefficient variance at DCT position `(u, v)` scales by
/// `dct_profile[u] * dct_profile[v]` before the threshold reads it, with
/// `u` and `v` read off the calling thread's own `local_x`/`local_y` (see
/// the body for why that is the coefficient this thread owns). At
/// `rho = 0` every entry is `1.0` and the multiply is a no-op.
///
/// `ht_wavelet` selects the transform basis. False, the default, fills
/// the DCT basis. True fills the orthonormal Haar-8 basis instead, for
/// comparing the two. The `dct8_line_*` helpers run either unchanged,
/// since both are orthonormal 8x8 matrices whose inverse is their
/// transpose.
///
/// # Temporal members
///
/// `temporal` selects where a member's pixels come from and where its
/// filtered pixels go.
///
/// False leaves `member_frame` unread. Every member is loaded from
/// `input` at `frame` and scatters back into that same frame's region of
/// `accum`/`wsum`.
///
/// True loads each member from `input` at its own `member_frame` entry,
/// so a member matched in a neighbour frame comes from that frame's own
/// pixels, and scatters its filtered pixels back into that frame's region
/// of the accumulators. A neighbour-frame member therefore contributes to
/// the caller's cross-frame accumulator ring rather than being discarded
/// once it has served the group's shared statistics.
///
/// Either way every member runs through the full forward transform,
/// threshold, and inverse transform, and lands in `filtered` when
/// `emit_filtered` is set.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn collab_filter_ht<N: Size>(
    input: &Array<Vector<f32, N>>,
    member_pos: &Array<u32>,
    member_frame: &Array<u32>,
    member_count: &Array<u32>,
    member_sig2: &Array<f32>,
    accum: &mut Array<Atomic<i32>>,
    wsum: &mut Array<Atomic<i32>>,
    filtered: &mut Array<Vector<f32, N>>,
    group_weight: &mut Array<f32>,
    frame: u32,
    sigma: &Array<f32>,
    dct_profile: &Array<f32>,
    lambda_ht: f32,
    weight_scale: f32,
    accum_scale: f32,
    #[comptime] use_member_sigma: bool,
    #[comptime] emit_filtered: bool,
    #[comptime] ht_wavelet: bool,
    #[comptime] temporal: bool,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] k_max: u32,
    #[comptime] stored_ch: u32,
    #[comptime] refs_x: u32,
) {
    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let tid = local_y * PATCH_SIZE + local_x;
    let ref_idx = CUBE_POS_Y * refs_x + CUBE_POS_X;
    let k_use = member_count[ref_idx as usize];

    // Every thread ends up owning DCT coefficient `(u = local_x, v =
    // local_y)` after the row/column DCT passes below run (the row pass
    // writes horizontal frequency `j` at position `row * 8 + j`, and the
    // column pass, reading that with `col` fixed to this thread's own
    // `tid % 8`, writes vertical frequency `j` at `j * 8 + col`, so
    // position `tid` ends up holding `(v = tid / 8, u = tid % 8) =
    // (local_y, local_x)`). The profile is purely spatial and identical
    // for every member of the stack, so it is read once here rather than
    // inside the per-channel loop below.
    let dct_profile_factor = dct_profile[local_x as usize] * dct_profile[local_y as usize];

    let mut basis = SharedMemory::<f32>::new(PATCH_AREA as usize);
    let mut stack = SharedMemory::<f32>::new(comptime!(k_max * PATCH_AREA) as usize);
    let mut wred = SharedMemory::<f32>::new(PATCH_AREA as usize);
    // The group's normalised weight, shared so every channel's scatter
    // reads the one the first channel computed.
    let mut gw = SharedMemory::<f32>::new(1usize);

    if ht_wavelet {
        fill_haar8_basis(&mut basis, tid);
    } else {
        fill_dct8_basis(&mut basis, tid);
    }
    sync_cube();

    let mut out_vec = Vector::<f32, N>::empty();

    #[unroll]
    for c in 0..channels {
        // Load channel `c` of every member into `stack`, one 64-slot row
        // per member, the layout `haar_fwd_stack` expects.
        let mut m = 0u32;
        while m < k_use {
            let packed = member_pos[(ref_idx * k_max + m) as usize];
            let member_x = packed & 0xFFFFu32;
            let member_y = packed >> 16u32;
            let mut src_frame = frame;
            if temporal {
                src_frame = member_frame[(ref_idx * k_max + m) as usize];
            }
            let pixel = read_line(
                input,
                member_x + local_x,
                member_y + local_y,
                src_frame,
                width,
                height,
            );
            stack[(m * PATCH_AREA + tid) as usize] = pixel[c as usize];
            m += 1u32;
        }
        sync_cube();

        // 2D DCT forward, independently for each member's patch. Row
        // pass then column pass, both in place over `stack`. A thread
        // owns its own eight slots within a pass, and the barrier
        // between the passes covers the hand-off to the threads that
        // read them next.
        let lines = k_use * PATCH_SIZE;
        if tid < lines {
            let member_k = tid / PATCH_SIZE;
            let row = tid % PATCH_SIZE;
            dct8_line_fwd(&basis, &mut stack, member_k * PATCH_AREA + row * PATCH_SIZE, 1u32);
        }
        sync_cube();
        if tid < lines {
            let member_k = tid / PATCH_SIZE;
            let col = tid % PATCH_SIZE;
            dct8_line_fwd(&basis, &mut stack, member_k * PATCH_AREA + col, PATCH_SIZE);
        }
        sync_cube();

        // The noise variance behind each member's DCT coefficients,
        // propagated to a per-stack-level variance.
        let sigma_c = sigma[c as usize];
        let base_sig2 = sigma_c * sigma_c;
        let mut v = Array::<f32>::new(8usize);
        #[unroll]
        for k in 0..8u32 {
            let mut sig2_k = base_sig2;
            if use_member_sigma && k < k_max {
                sig2_k += member_sig2[(ref_idx * k_max + k) as usize];
            }
            v[k as usize] = sig2_k * dct_profile_factor;
        }
        variance_ladder(&mut v, k_use);

        // Haar transform along the stack axis, one thread per spatial
        // position, no cross-thread reads or writes.
        haar_fwd_stack(&mut stack, tid, k_use);

        // Hard threshold, and the group-DC exception described above.
        let mut retained_v = 0.0f32;
        let mut j = 0u32;
        while j < k_use {
            let idx = (j * PATCH_AREA + tid) as usize;
            let coeff = stack[idx];
            let threshold = lambda_ht * f32::sqrt(v[j as usize]);
            let mut keep = f32::abs(coeff) >= threshold;
            if j == 0u32 && tid == 0u32 {
                keep = true;
            }
            if keep {
                retained_v += v[j as usize];
            } else {
                stack[idx] = 0.0f32;
            }
            j += 1u32;
        }

        // The group weight has to be known before the scatter below, and
        // only the first channel computes it, so the reduction runs here
        // rather than after the inverse transforms.
        if comptime!(c == 0u32) {
            wred[tid as usize] = retained_v;
            sync_cube();

            let mut stride = 32u32;
            while stride > 0u32 {
                if tid < stride {
                    wred[tid as usize] += wred[(tid + stride) as usize];
                }
                sync_cube();
                stride /= 2u32;
            }

            if tid == 0u32 {
                let sum = wred[0];
                // `sum` adds non-negative variances, so it is never
                // negative. `safe_reciprocal` checks for a non-finite sum
                // explicitly rather than leaning on `f32::max` to discard
                // one, so the weight is finite here whatever a given GPU
                // does with NaN.
                let w = safe_reciprocal(sum, RECIPROCAL_FLOOR);
                group_weight[ref_idx as usize] = w;
                // The accumulators count in fixed point, so the weight
                // is scaled into the band `weight_scale` was built to
                // put it in. Aggregation normalises by the weight sum,
                // so scaling every weight by the same constant leaves
                // the result exactly as it would have been.
                gw[0] = w * weight_scale;
            }
        }
        sync_cube();

        // Haar inverse, back from stack coefficients to per-member DCT
        // coefficients.
        haar_inv_stack(&mut stack, tid, k_use);
        sync_cube();

        // Every member of the group is written back, not just the
        // reference patch, so every member's DCT coefficients need
        // inverting. The row and column passes mirror the forward ones
        // exactly, one thread per (member, line), both in place over
        // `stack`. A thread owns its own eight slots within a pass, and
        // the barrier between the passes covers the hand-off to the
        // threads that read them next.
        let lines = k_use * PATCH_SIZE;
        if tid < lines {
            let member_k = tid / PATCH_SIZE;
            let row = tid % PATCH_SIZE;
            dct8_line_inv(&basis, &mut stack, member_k * PATCH_AREA + row * PATCH_SIZE, 1u32);
        }
        sync_cube();
        if tid < lines {
            let member_k = tid / PATCH_SIZE;
            let col = tid % PATCH_SIZE;
            dct8_line_inv(&basis, &mut stack, member_k * PATCH_AREA + col, PATCH_SIZE);
        }
        sync_cube();

        out_vec[c as usize] = stack[tid as usize];

        // Scatter every member back into its own frame's region of the
        // accumulators, following the same `temporal` split the load at
        // the top of this loop uses.
        let weight = gw[0];
        let mut mo = 0u32;
        while mo < k_use {
            let packed = member_pos[(ref_idx * k_max + mo) as usize];
            let mut member_frame_val = frame;
            if temporal {
                member_frame_val = member_frame[(ref_idx * k_max + mo) as usize];
            }
            scatter_patch(
                accum,
                wsum,
                stack[(mo * PATCH_AREA + tid) as usize],
                weight,
                packed & 0xFFFFu32,
                packed >> 16u32,
                tid,
                comptime!(c == 0u32),
                c,
                width,
                stored_ch,
                member_frame_val,
                comptime!(width * height),
                accum_scale,
            );
            mo += 1u32;
        }

        if emit_filtered {
            let mut me = 0u32;
            while me < k_use {
                let mut line = filtered[(ref_idx * k_max * PATCH_AREA + me * PATCH_AREA + tid) as usize];
                line[c as usize] = stack[(me * PATCH_AREA + tid) as usize];
                filtered[(ref_idx * k_max * PATCH_AREA + me * PATCH_AREA + tid) as usize] = line;
                me += 1u32;
            }
        }

        sync_cube();
    }
}

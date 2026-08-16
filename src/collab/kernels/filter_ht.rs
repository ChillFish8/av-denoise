use cubecl::prelude::*;

use crate::collab::kernels::transforms::{
    dct8_line_fwd,
    dct8_line_inv,
    fill_dct8_basis,
    haar_fwd_stack,
    haar_inv_stack,
    safe_reciprocal,
};
use crate::collab::{MAX_K, PATCH_AREA, PATCH_SIZE};
use crate::nlmeans::kernels::helpers::read_line;

// Ties the `stride = 32u32` literal in the group-weight reduction below
// back to `PATCH_AREA`, the same way `collab_group_spatial` ties its own
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
/// Only `0..k_use` matter, the rest are ignored. A Haar butterfly's two
/// outputs are each a sum of two independent inputs scaled by
/// `1/sqrt(2)`, so if `va` and `vb` are the variances of two inputs,
/// both outputs land on the same variance, `(va + vb) / 2`. Running that
/// averaging over the same levels, in the same pairing order, that
/// `haar_fwd_stack` runs over the signal itself turns `v[j]` into the
/// variance of stack coefficient `j`. The host-only
/// `haar_variance_ladder` in `transforms` runs the identical computation
/// off the GPU, for comparison in tests.
///
/// Every level snapshots the values it reads before writing any output.
/// The output range overlaps the input range, so an in-place write would
/// otherwise clobber a value a later pair in the same level still needs,
/// the same reasoning `haar_fwd_stack` follows.
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
/// / `member_count` layout `collab_group_spatial` produces. Its
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
/// `input` is the frame ring `collab_group_spatial` also reads, indexed
/// by `frame`. `member_pos` and `member_count` are `collab_group_spatial`'s
/// outputs, `refs * k_max` packed positions and `refs` counts.
/// `member_sig2` holds `refs * k_max` extra per-member variance, added to
/// `sigma[c]^2` when `use_member_sigma` is true. It is never read when that
/// flag is false, so a 1-element dummy buffer is valid there, the same
/// pattern `confidence_dummy` uses in `nlmeans`. `filtered` holds
/// `refs * PATCH_AREA` lines, member 0's filtered patch for every
/// reference. `group_weight` holds `refs` weights, and `sigma` holds one
/// value per stored channel.
///
/// `dct_profile` holds 8 values, [`crate::collab::kernels::transforms::dct_noise_profile`]'s
/// output. Every member's propagated coefficient variance at DCT
/// position `(u, v)` scales by `dct_profile[u] * dct_profile[v]` before
/// the threshold reads it, `u` and `v` read straight off the calling
/// thread's own `local_x`/`local_y` (see the body for why that's exactly
/// the coefficient position this thread ends up owning). At `rho = 0`
/// every entry is exactly `1.0`, so this multiply is a no-op and the
/// threshold behaves exactly as it did before this profile existed.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn collab_filter_ht<N: Size>(
    input: &Array<Vector<f32, N>>,
    member_pos: &Array<u32>,
    member_count: &Array<u32>,
    member_sig2: &Array<f32>,
    filtered: &mut Array<Vector<f32, N>>,
    group_weight: &mut Array<f32>,
    frame: u32,
    sigma: &Array<f32>,
    dct_profile: &Array<f32>,
    lambda_ht: f32,
    #[comptime] use_member_sigma: bool,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] k_max: u32,
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
    let mut scratch = SharedMemory::<f32>::new(comptime!(k_max * PATCH_AREA) as usize);
    let mut wred = SharedMemory::<f32>::new(PATCH_AREA as usize);

    fill_dct8_basis(&mut basis, tid);
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
            let pixel = read_line(
                input,
                member_x + local_x,
                member_y + local_y,
                frame,
                width,
                height,
            );
            stack[(m * PATCH_AREA + tid) as usize] = pixel[c as usize];
            m += 1u32;
        }
        sync_cube();

        // 2D DCT forward, independently for each member's patch. Row
        // pass then column pass, alternating `stack`/`scratch` as
        // source and destination, the same convention a single-patch
        // DCT follows.
        let lines = k_use * PATCH_SIZE;
        if tid < lines {
            let member_k = tid / PATCH_SIZE;
            let row = tid % PATCH_SIZE;
            dct8_line_fwd(
                &basis,
                &stack,
                &mut scratch,
                member_k * PATCH_AREA + row * PATCH_SIZE,
                1u32,
            );
        }
        sync_cube();
        if tid < lines {
            let member_k = tid / PATCH_SIZE;
            let col = tid % PATCH_SIZE;
            dct8_line_fwd(
                &basis,
                &scratch,
                &mut stack,
                member_k * PATCH_AREA + col,
                PATCH_SIZE,
            );
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

        // Haar inverse, back from stack coefficients to per-member DCT
        // coefficients.
        haar_inv_stack(&mut stack, tid, k_use);
        sync_cube();

        // Only member 0, the reference patch itself, is written out, so
        // only its DCT coefficients need inverting back to pixels.
        if tid < PATCH_SIZE {
            dct8_line_inv(&basis, &stack, &mut scratch, tid * PATCH_SIZE, 1u32);
        }
        sync_cube();
        if tid < PATCH_SIZE {
            dct8_line_inv(&basis, &scratch, &mut stack, tid, PATCH_SIZE);
        }
        sync_cube();

        out_vec[c as usize] = stack[tid as usize];

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
                // sum is a sum of non-negative propagated variances, so
                // it can never be negative for an ordinary finite
                // caller sigma. safe_reciprocal checks for a non-finite
                // sum explicitly rather than leaning on `f32::max` to
                // discard one, so group_weight is always finite here
                // regardless of GPU-specific NaN behaviour, capped at
                // 1e12 for an ordinary small sum and 0 for a sum that
                // turned out non-finite.
                //
                // Aggregation only ever multiplies this weight into a
                // convex combination of finite filtered patch values and
                // divides by the sum of the weights covering a pixel.
                // However large or unequal the weights get, that kind of
                // weighted mean cannot leave the range the patch values
                // it combines already span.
                group_weight[ref_idx as usize] = safe_reciprocal(sum, 1e-12f32);
            }
        }

        sync_cube();
    }

    filtered[(ref_idx * PATCH_AREA + tid) as usize] = out_vec;
}

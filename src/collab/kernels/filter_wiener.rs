use cubecl::prelude::*;

use crate::collab::kernels::aggregate::scatter_patch;
use crate::collab::kernels::filter_ht::variance_ladder;
use crate::collab::kernels::transforms::{
    RECIPROCAL_FLOOR,
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
// back to `PATCH_AREA`, the same way `collab_filter_ht` ties its own
// copy of this literal.
const _: () = assert!(
    PATCH_AREA / 2 == 32,
    "update the `stride = 32u32` literal in collab_filter_wiener to PATCH_AREA / 2"
);

// Ties the hardcoded `8usize`/`8u32` per-channel variance setup below
// back to `MAX_K`, the same way `collab_filter_ht` ties its own copy.
const _: () = assert!(
    MAX_K == 8,
    "update the hardcoded 8usize/8u32 slots in collab_filter_wiener's variance setup to MAX_K"
);

/// A small floor on the Wiener denominator, so a pilot coefficient of
/// exactly zero next to a zero propagated variance divides `0 / 0` into
/// `0` instead of `NaN`. Real coefficients never land this close to
/// zero, so it never perturbs a genuine shrinkage factor.
const WIENER_EPSILON: f32 = 1e-20;

/// The Wiener shrinkage factor for one coefficient, always in `[0, 1]`.
///
/// `p` is the pilot's coefficient value and `vj` is that coefficient's
/// propagated noise variance. `p * p / max(p * p + vj, WIENER_EPSILON)`
/// is mathematically bounded to `[0, 1]` whenever `vj` is a real,
/// non-negative variance, since the denominator can only be as small as
/// the numerator.
///
/// `vj` reaches this function in three qualitatively different states,
/// and each gets a different fallback.
///
/// A `vj` of `f32::INFINITY` is real information. It says the
/// coefficient is known to be pure noise, so it shrinks all the way to
/// 0, the same as an ordinary large finite variance would, just sooner.
///
/// A `vj` that is `NaN` carries no information at all, neither "large"
/// nor "small", simply undefined. Computing the ratio anyway would be
/// wrong twice over. `p * p + vj` is `NaN`, and `f32::max(NaN,
/// WIENER_EPSILON)` discards it in favor of the epsilon floor instead of
/// something large, turning the denominator tiny and the ratio into an
/// unbounded amplifier. Falling back to 0, the way the infinite case
/// does, would be just as wrong in the other direction. It would delete
/// a coefficient based on a noise level nobody actually measured. The
/// only response that neither destroys the signal nor risks amplifying
/// it is to leave the coefficient exactly as the noisy input carried it,
/// `w = 1`, full trust rather than full distrust, since trust of exactly
/// 1 can never amplify anything either.
///
/// A negative `vj` is not a real variance at all and is treated the same
/// as an infinite one, shrinking to 0, since there is no principled
/// value to fall back to for a quantity that should never be negative in
/// the first place.
///
/// A `p` that is itself not finite reaches the ordinary ratio branch
/// with a finite `vj`, so `f32::max(w_raw, 0.0)` catches that case
/// afterward. `.clamp()` is not used for this second guard on purpose,
/// since it returns `NaN` right back out for a `NaN` input, which is
/// exactly the case this function exists to catch.
#[cube]
pub(crate) fn wiener_shrinkage_factor(p: f32, vj: f32) -> f32 {
    let mut w = 0.0f32;
    if vj.is_nan() {
        w = 1.0f32;
    } else if !vj.is_inf() && vj >= 0.0f32 {
        let w_raw = p * p / f32::max(p * p + vj, WIENER_EPSILON);
        #[allow(clippy::manual_clamp)]
        let clamped = f32::min(f32::max(w_raw, 0.0f32), 1.0f32);
        w = clamped;
    }
    w
}

/// Filters one group of similar patches with Wiener shrinkage steered by
/// a pilot estimate, and writes back the filtered reference patch.
///
/// One cube owns one reference patch's group, with the same `member_pos`
/// / `member_count` layout `collab_group_spatial` produces, this time
/// grouped against the pilot rather than the noisy input. Its
/// `CubeDim::new_2d(8, 8)` threads map one-to-one onto a patch's 64
/// pixels.
///
/// # What the filter does
///
/// For each active channel, every member's pilot patch and its matching
/// noisy patch are loaded and each run through a 2D DCT, so every patch
/// is described by 64 frequency coefficients instead of 64 pixel values.
/// A Haar transform then runs across the stack axis of both stacks
/// independently, at each spatial position, the same decomposition
/// [`crate::collab::kernels::filter_ht::collab_filter_ht`] uses. Each
/// pilot coefficient sets a Wiener shrinkage factor `W = pilot^2 /
/// (pilot^2 + v_j)`, where `v_j` is that coefficient's propagated noise
/// variance from [`variance_ladder`]. The matching noisy coefficient is
/// scaled by `W` before both transforms invert, so a coefficient the
/// pilot found confidently present passes through close to unchanged,
/// and a coefficient the pilot found weak or absent shrinks toward zero
/// in proportion to how much noise it is expected to carry. The result
/// is member 0's patch, the reference patch itself, filtered by how well
/// the pilot's cleaner estimate explains it.
///
/// Wiener shrinkage degrades smoothly with each coefficient's
/// signal-to-noise ratio, unlike a hard threshold, so no separate
/// exception is needed to protect the group's mean brightness. A strong
/// pilot coefficient already keeps `W` close to 1 on its own.
///
/// # Group weight
///
/// `group_weight` is `1 / sum(W^2 * v_j)` over every coefficient the
/// stack holds, computed from channel 0 only, the same inverse-variance
/// convention `collab_filter_ht` uses. A group whose pilot coefficients
/// carried more signal keeps `W` closer to 1 across the stack and ends
/// up trusted more.
///
/// # Buffers
///
/// `noisy` and `pilot` are two separate whole-frame buffers, indexed by
/// `noisy_frame` and `pilot_frame` respectively, in the same layout
/// `collab_group_spatial`/`collab_filter_ht` read from. `member_pos` and
/// `member_count` come from grouping run against the pilot, so both
/// stacks are loaded from the same member positions. `member_sig2` holds
/// `refs * k_max` extra per-member variance, added to `sigma[c]^2` when
/// `use_member_sigma` is true, and is never read when that flag is
/// false, so a 1-element dummy buffer is valid there. `filtered` holds
/// `refs * PATCH_AREA` lines, member 0's filtered patch for every
/// reference. `group_weight` holds `refs` weights, and `sigma` holds one
/// value per stored channel.
///
/// `dct_profile` holds 8 values, [`crate::collab::kernels::transforms::dct_noise_profile`]'s
/// output. Every propagated coefficient variance at DCT position `(u,
/// v)` scales by `dct_profile[u] * dct_profile[v]` before it sets that
/// coefficient's Wiener shrinkage factor, the same mapping and the same
/// no-op-at-`rho = 0` guarantee [`crate::collab::kernels::filter_ht::collab_filter_ht`]
/// documents.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn collab_filter_wiener<N: Size>(
    noisy: &Array<Vector<f32, N>>,
    pilot: &Array<Vector<f32, N>>,
    member_pos: &Array<u32>,
    member_count: &Array<u32>,
    member_sig2: &Array<f32>,
    accum: &mut Array<Atomic<i32>>,
    wsum: &mut Array<Atomic<i32>>,
    filtered: &mut Array<Vector<f32, N>>,
    group_weight: &mut Array<f32>,
    noisy_frame: u32,
    pilot_frame: u32,
    sigma: &Array<f32>,
    dct_profile: &Array<f32>,
    weight_scale: f32,
    #[comptime] use_member_sigma: bool,
    #[comptime] emit_filtered: bool,
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

    // Same coefficient-position derivation `collab_filter_ht` documents:
    // this thread ends up owning DCT position `(u = local_x, v =
    // local_y)` after the row/column DCT passes below.
    let dct_profile_factor = dct_profile[local_x as usize] * dct_profile[local_y as usize];

    let mut basis = SharedMemory::<f32>::new(PATCH_AREA as usize);
    let mut pilot_stack = SharedMemory::<f32>::new(comptime!(k_max * PATCH_AREA) as usize);
    let mut noisy_stack = SharedMemory::<f32>::new(comptime!(k_max * PATCH_AREA) as usize);
    let mut scratch = SharedMemory::<f32>::new(comptime!(k_max * PATCH_AREA) as usize);
    let mut wred = SharedMemory::<f32>::new(PATCH_AREA as usize);
    // The group's normalised weight, shared so every channel's scatter
    // reads the one the first channel computed.
    let mut gw = SharedMemory::<f32>::new(1usize);

    fill_dct8_basis(&mut basis, tid);
    sync_cube();

    let mut out_vec = Vector::<f32, N>::empty();

    #[unroll]
    for c in 0..channels {
        // Load channel `c` of every member's pilot patch and its
        // matching noisy patch, one 64-slot row per member in each
        // stack.
        let mut m = 0u32;
        while m < k_use {
            let packed = member_pos[(ref_idx * k_max + m) as usize];
            let member_x = packed & 0xFFFFu32;
            let member_y = packed >> 16u32;
            let pilot_pixel = read_line(
                pilot,
                member_x + local_x,
                member_y + local_y,
                pilot_frame,
                width,
                height,
            );
            let noisy_pixel = read_line(
                noisy,
                member_x + local_x,
                member_y + local_y,
                noisy_frame,
                width,
                height,
            );
            pilot_stack[(m * PATCH_AREA + tid) as usize] = pilot_pixel[c as usize];
            noisy_stack[(m * PATCH_AREA + tid) as usize] = noisy_pixel[c as usize];
            m += 1u32;
        }
        sync_cube();

        // 2D DCT forward on the pilot stack, row pass then column pass,
        // through `scratch`, the same convention `collab_filter_ht`
        // follows.
        let lines = k_use * PATCH_SIZE;
        if tid < lines {
            let member_k = tid / PATCH_SIZE;
            let row = tid % PATCH_SIZE;
            dct8_line_fwd(
                &basis,
                &pilot_stack,
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
                &mut pilot_stack,
                member_k * PATCH_AREA + col,
                PATCH_SIZE,
            );
        }
        sync_cube();

        // 2D DCT forward on the noisy stack. `scratch` is free to reuse
        // here, the pilot pass above already finished reading it and the
        // `sync_cube()` after that pass makes that visible to every
        // thread before this pass writes it again.
        if tid < lines {
            let member_k = tid / PATCH_SIZE;
            let row = tid % PATCH_SIZE;
            dct8_line_fwd(
                &basis,
                &noisy_stack,
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
                &mut noisy_stack,
                member_k * PATCH_AREA + col,
                PATCH_SIZE,
            );
        }
        sync_cube();

        // The noise variance behind each member's DCT coefficients,
        // propagated to a per-stack-level variance, exactly as
        // `collab_filter_ht` computes it.
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

        // Haar transform along the stack axis, independently for both
        // stacks, one thread per spatial position.
        haar_fwd_stack(&mut pilot_stack, tid, k_use);
        haar_fwd_stack(&mut noisy_stack, tid, k_use);

        // Wiener shrinkage. Every coefficient the stack holds
        // contributes to the group weight, there is no discrete
        // retain/reject decision the way a hard threshold makes one.
        let mut var_sum = 0.0f32;
        let mut j = 0u32;
        while j < k_use {
            let idx = (j * PATCH_AREA + tid) as usize;
            let p = pilot_stack[idx];
            let vj = v[j as usize];
            let w = wiener_shrinkage_factor(p, vj);
            noisy_stack[idx] = w * noisy_stack[idx];
            // w * w * vj is meant to be this coefficient's contribution
            // to the group's own residual-noise estimate, but `vj`
            // itself is not finite for either of `wiener_shrinkage_factor`'s
            // non-finite fallbacks, and multiplying a non-finite value
            // by anything, including a `w` of exactly 0 or exactly 1,
            // is still non-finite. There is no real propagated-variance
            // number to contribute here in either case, so this
            // coefficient is left out of `wsum` entirely rather than
            // adding a poisoned term to it.
            if !vj.is_nan() && !vj.is_inf() {
                var_sum += w * w * vj;
            }
            j += 1u32;
        }

        // The group weight has to be known before the scatter below, and
        // only the first channel computes it, so the reduction runs here
        // rather than after the inverse transforms.
        if comptime!(c == 0u32) {
            wred[tid as usize] = var_sum;
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
                // Same reasoning as collab_filter_ht's group_weight.
                // safe_reciprocal checks for a non-finite sum explicitly
                // instead of leaning on `f32::max` to discard one, so
                // this is always finite regardless of GPU-specific NaN
                // behaviour.
                let w = safe_reciprocal(sum, RECIPROCAL_FLOOR);
                group_weight[ref_idx as usize] = w;
                gw[0] = w * weight_scale;
            }
        }
        sync_cube();

        // Haar inverse, back from stack coefficients to per-member DCT
        // coefficients, on the filtered noisy stack.
        haar_inv_stack(&mut noisy_stack, tid, k_use);
        sync_cube();

        // Every member of the group is written back, not just the
        // reference patch, so every member's DCT coefficients need
        // inverting. The row and column passes mirror the forward ones
        // exactly, one thread per (member, line).
        let lines = k_use * PATCH_SIZE;
        if tid < lines {
            let member_k = tid / PATCH_SIZE;
            let row = tid % PATCH_SIZE;
            dct8_line_inv(
                &basis,
                &noisy_stack,
                &mut scratch,
                member_k * PATCH_AREA + row * PATCH_SIZE,
                1u32,
            );
        }
        sync_cube();
        if tid < lines {
            let member_k = tid / PATCH_SIZE;
            let col = tid % PATCH_SIZE;
            dct8_line_inv(
                &basis,
                &scratch,
                &mut noisy_stack,
                member_k * PATCH_AREA + col,
                PATCH_SIZE,
            );
        }
        sync_cube();

        out_vec[c as usize] = noisy_stack[tid as usize];

        // Scatter every member back to where it came from. One thread
        // owns one pixel of each member's patch.
        let weight = gw[0];
        let mut mo = 0u32;
        while mo < k_use {
            let packed = member_pos[(ref_idx * k_max + mo) as usize];
            scatter_patch(
                accum,
                wsum,
                noisy_stack[(mo * PATCH_AREA + tid) as usize],
                weight,
                packed & 0xFFFFu32,
                packed >> 16u32,
                tid,
                comptime!(c == 0u32),
                c,
                width,
                stored_ch,
            );
            mo += 1u32;
        }

        if emit_filtered {
            let mut me = 0u32;
            while me < k_use {
                let slot = (ref_idx * k_max * PATCH_AREA + me * PATCH_AREA + tid) as usize;
                let mut line = filtered[slot];
                line[c as usize] = noisy_stack[(me * PATCH_AREA + tid) as usize];
                filtered[slot] = line;
                me += 1u32;
            }
        }

        sync_cube();
    }
}

use cubecl::prelude::*;

use crate::collab::kernels::filter_ht::variance_ladder;
use crate::collab::kernels::transforms::{
    dct8_line_fwd,
    dct8_line_inv,
    fill_dct8_basis,
    haar_fwd_stack,
    haar_inv_stack,
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
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn collab_filter_wiener<N: Size>(
    noisy: &Array<Vector<f32, N>>,
    pilot: &Array<Vector<f32, N>>,
    member_pos: &Array<u32>,
    member_count: &Array<u32>,
    member_sig2: &Array<f32>,
    filtered: &mut Array<Vector<f32, N>>,
    group_weight: &mut Array<f32>,
    noisy_frame: u32,
    pilot_frame: u32,
    sigma: &Array<f32>,
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

    let mut basis = SharedMemory::<f32>::new(PATCH_AREA as usize);
    let mut pilot_stack = SharedMemory::<f32>::new(comptime!(k_max * PATCH_AREA) as usize);
    let mut noisy_stack = SharedMemory::<f32>::new(comptime!(k_max * PATCH_AREA) as usize);
    let mut scratch = SharedMemory::<f32>::new(comptime!(k_max * PATCH_AREA) as usize);
    let mut wred = SharedMemory::<f32>::new(PATCH_AREA as usize);

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
            v[k as usize] = sig2_k;
        }
        variance_ladder(&mut v, k_use);

        // Haar transform along the stack axis, independently for both
        // stacks, one thread per spatial position.
        haar_fwd_stack(&mut pilot_stack, tid, k_use);
        haar_fwd_stack(&mut noisy_stack, tid, k_use);

        // Wiener shrinkage. Every coefficient the stack holds
        // contributes to the group weight, there is no discrete
        // retain/reject decision the way a hard threshold makes one.
        let mut wsum = 0.0f32;
        let mut j = 0u32;
        while j < k_use {
            let idx = (j * PATCH_AREA + tid) as usize;
            let p = pilot_stack[idx];
            let vj = v[j as usize];
            let w = p * p / f32::max(p * p + vj, WIENER_EPSILON);
            noisy_stack[idx] = w * noisy_stack[idx];
            wsum += w * w * vj;
            j += 1u32;
        }

        // Haar inverse, back from stack coefficients to per-member DCT
        // coefficients, on the filtered noisy stack.
        haar_inv_stack(&mut noisy_stack, tid, k_use);
        sync_cube();

        // Only member 0, the reference patch itself, is written out, so
        // only its DCT coefficients need inverting back to pixels.
        if tid < PATCH_SIZE {
            dct8_line_inv(&basis, &noisy_stack, &mut scratch, tid * PATCH_SIZE, 1u32);
        }
        sync_cube();
        if tid < PATCH_SIZE {
            dct8_line_inv(&basis, &scratch, &mut noisy_stack, tid, PATCH_SIZE);
        }
        sync_cube();

        out_vec[c as usize] = noisy_stack[tid as usize];

        if comptime!(c == 0u32) {
            wred[tid as usize] = wsum;
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
                group_weight[ref_idx as usize] = 1.0f32 / f32::max(sum, 1e-12f32);
            }
        }

        sync_cube();
    }

    filtered[(ref_idx * PATCH_AREA + tid) as usize] = out_vec;
}

use cubecl::prelude::*;

use crate::collab::{MAX_K, PATCH_AREA, PATCH_SIZE};

// Ties the hardcoded `8usize`/`8u32` per-thread register-array sizes in
// `haar_fwd_stack` and `haar_inv_stack` back to `MAX_K`. Both functions
// snapshot a thread's full column into a fixed-size local array every
// level rather than sizing it off `MAX_K` directly (see their own doc
// comments for why), so nothing else catches this constant drifting.
const _: () = assert!(
    MAX_K == 8,
    "update the hardcoded 8usize/8u32 slots in haar_fwd_stack and haar_inv_stack to MAX_K"
);

/// A reciprocal weight that never trusts a driver's `NaN`-`max` behaviour.
///
/// Returns `1 / max(denom, floor)` for an ordinary finite denominator,
/// the same value the unguarded expression this replaces would produce.
/// A `denom` that is `NaN` or infinite is caught explicitly first and
/// returns `0` instead.
///
/// `wiener_shrinkage_factor` treats an unknown (`NaN`) variance
/// differently from a known-infinite one, since it can fall back to
/// passing a coefficient's value through untouched, a safe middle
/// ground between full trust and full distrust. `safe_reciprocal` has
/// no such middle ground available. Its result only ever feeds a weight
/// or a normalising divisor, never a value that could stand in for real
/// content, so there is nothing to "pass through". Zero is the correct
/// fallback here regardless of whether the denominator is unknown or
/// known-unusable, because both cases mean this reciprocal cannot be
/// trusted to mean anything.
///
/// The explicit check matters because `f32::max(NaN, floor)` is not
/// guaranteed to discard the `NaN` on every GPU backend the way it does
/// on the CPU. SPIR-V's `FMax` instruction leaves a `NaN` operand
/// undefined, and only the separate `NMax` instruction promises to
/// discard it, so a caller that relied on `max` alone would have its
/// correctness depend on which instruction a given driver happened to
/// lower `f32::max` to.
#[cube]
pub(crate) fn safe_reciprocal(denom: f32, floor: f32) -> f32 {
    let mut inv = 0.0f32;
    if !denom.is_nan() && !denom.is_inf() {
        inv = 1.0f32 / f32::max(denom, floor);
    }
    inv
}

/// Fills an 8x8 orthonormal DCT-II basis into shared memory, one entry
/// per thread.
///
/// Entry `j*8+i` holds `c_j * cos(PI * (2i+1) * j / 16)`, with `c_0 =
/// 1/sqrt(8)` and `c_j = 0.5` for every other row. That scaling is what
/// makes the basis orthonormal, each row has unit norm and any two
/// distinct rows are orthogonal, so the same matrix run forwards is a
/// DCT and run transposed is its own inverse.
///
/// Only the first 64 threads of the calling cube write an entry, callers
/// with more threads than that call this once per thread anyway. The
/// caller must call `sync_cube()` before reading `basis`.
#[cube]
pub(crate) fn fill_dct8_basis(basis: &mut SharedMemory<f32>, thread_id: u32) {
    if thread_id < PATCH_AREA {
        let i = thread_id % PATCH_SIZE;
        let j = thread_id / PATCH_SIZE;
        let mut c = 0.5f32;
        if j == 0 {
            c = 1.0f32 / f32::sqrt(8.0f32);
        }
        let angle = std::f32::consts::PI * (2.0f32 * i as f32 + 1.0f32) * j as f32 / 16.0f32;
        basis[thread_id as usize] = c * f32::cos(angle);
    }
}

/// Runs one forward 8-point DCT over the 8 values at `src[base + i *
/// stride]`, writing the 8 coefficients to the same positions of `dst`.
///
/// One thread computes all 8 outputs of one line, so the caller decides
/// how many lines run at once by deciding how many threads call this.
/// Passing `base` and `stride` as row or column offsets turns the same
/// function into either a row transform or a column transform.
///
/// `dst` must not alias `src`, this reads every input before writing any
/// output, but it reads the whole line first only to keep the pattern
/// simple, not because aliasing would otherwise be safe.
#[cube]
pub(crate) fn dct8_line_fwd(
    basis: &SharedMemory<f32>,
    src: &SharedMemory<f32>,
    dst: &mut SharedMemory<f32>,
    base: u32,
    stride: u32,
) {
    #[unroll]
    for j in 0..PATCH_SIZE {
        let mut sum = 0.0f32;
        #[unroll]
        for i in 0..PATCH_SIZE {
            sum += basis[(j * PATCH_SIZE + i) as usize] * src[(base + i * stride) as usize];
        }
        dst[(base + j * stride) as usize] = sum;
    }
}

/// The inverse of `dct8_line_fwd`, the transpose of the same basis.
///
/// Because the basis is orthonormal its inverse is its own transpose, so
/// this reads `basis[j*8+i]` for output position `i` instead of output
/// position `j`, and sums over `j` instead of `i`. Everything else about
/// the calling convention matches `dct8_line_fwd`.
#[cube]
pub(crate) fn dct8_line_inv(
    basis: &SharedMemory<f32>,
    src: &SharedMemory<f32>,
    dst: &mut SharedMemory<f32>,
    base: u32,
    stride: u32,
) {
    #[unroll]
    for i in 0..PATCH_SIZE {
        let mut sum = 0.0f32;
        #[unroll]
        for j in 0..PATCH_SIZE {
            sum += basis[(j * PATCH_SIZE + i) as usize] * src[(base + j * stride) as usize];
        }
        dst[(base + i * stride) as usize] = sum;
    }
}

/// Runs the orthonormal Haar transform along the stack axis, in place,
/// for one spatial position `pos`.
///
/// The stack holds up to `MAX_K` patches, one per `k`, and element `k`
/// of the stack at this position lives at `stack[k * PATCH_AREA + pos]`.
/// `k_use` is the active stack size, a power of two no greater than
/// `MAX_K`.
///
/// Each butterfly is `(a, b) -> ((a+b)/sqrt(2), (a-b)/sqrt(2))`, which
/// has unit-norm rows and is its own inverse. The transform applies that
/// butterfly across the whole active stack, then recurses on just the
/// approximation half, halving the working length each time until one
/// value is left. That is a full multi-level decomposition, with the
/// coarsest approximation ending up at `k = 0` and finer detail bands
/// filling the rest in order. A `k_use` of 1 never enters the loop, so
/// the stack is left unchanged.
///
/// Every thread owns a different `pos`, so no other thread ever touches
/// this thread's slice of the stack and no `sync_cube()` is needed
/// inside. Each level snapshots every value it might read before writing
/// any output, because the output range overlaps the input range and an
/// in-place write would otherwise clobber a value a later pair still
/// needs.
#[cube]
pub(crate) fn haar_fwd_stack(stack: &mut SharedMemory<f32>, pos: u32, k_use: u32) {
    let inv_sqrt2 = 1.0f32 / f32::sqrt(2.0f32);
    let mut len = k_use;
    while len > 1 {
        let half = len / 2;

        let mut snapshot = Array::<f32>::new(8usize);
        #[unroll]
        for k in 0..8u32 {
            snapshot[k as usize] = stack[(k * PATCH_AREA + pos) as usize];
        }

        let mut p = 0u32;
        while p < half {
            let a = snapshot[(2u32 * p) as usize];
            let b = snapshot[(2u32 * p + 1) as usize];
            stack[(p * PATCH_AREA + pos) as usize] = (a + b) * inv_sqrt2;
            stack[((half + p) * PATCH_AREA + pos) as usize] = (a - b) * inv_sqrt2;
            p += 1;
        }

        len = half;
    }
}

/// The inverse of `haar_fwd_stack`.
///
/// The forward transform's butterfly is its own inverse, so undoing it
/// only takes running the same butterfly again, in the opposite level
/// order. The forward pass works from the full stack down to a single
/// value, so the inverse pass works back up from a pair to the full
/// stack, at each level combining the approximation half with the
/// detail half that level produced, and writing the result back
/// interleaved.
///
/// The calling convention matches `haar_fwd_stack`, including the
/// per-level snapshot that keeps in-place writes from clobbering values
/// a later pair still needs.
#[cube]
pub(crate) fn haar_inv_stack(stack: &mut SharedMemory<f32>, pos: u32, k_use: u32) {
    let inv_sqrt2 = 1.0f32 / f32::sqrt(2.0f32);
    let mut len = 2u32;
    while len <= k_use {
        let half = len / 2;

        let mut snapshot = Array::<f32>::new(8usize);
        #[unroll]
        for k in 0..8u32 {
            snapshot[k as usize] = stack[(k * PATCH_AREA + pos) as usize];
        }

        let mut p = 0u32;
        while p < half {
            let a = snapshot[p as usize];
            let b = snapshot[(half + p) as usize];
            stack[(2u32 * p * PATCH_AREA + pos) as usize] = (a + b) * inv_sqrt2;
            stack[((2u32 * p + 1) * PATCH_AREA + pos) as usize] = (a - b) * inv_sqrt2;
            p += 1;
        }

        len *= 2;
    }
}

/// The host-side mirror of the variance propagation `haar_fwd_stack`
/// applies to a per-coefficient noise variance instead of a signal.
///
/// A Haar butterfly's two outputs are each a sum of two independent
/// values scaled by `1/sqrt(2)`, so if `va` and `vb` are the variances
/// of `a` and `b`, both outputs land on the same variance, `(va +
/// vb) / 2`. Squaring `1/sqrt(2)` gives the `1/2`, and both outputs get
/// the same value because both are a sum of the same two inputs, just
/// with one sign flipped.
///
/// This runs the same multi-level recursion as `haar_fwd_stack`, over
/// plain host `f32`, so a filter kernel can propagate a per-patch sigma
/// down to a per-coefficient sigma without a GPU round trip.
pub fn haar_variance_ladder(sig2: &[f32], k_use: u32) -> Vec<f32> {
    let mut out = sig2.to_vec();
    let mut len = k_use;
    while len > 1 {
        let half = len / 2;
        let snapshot = out[..len as usize].to_vec();
        for p in 0..half {
            let va = snapshot[(2 * p) as usize];
            let vb = snapshot[(2 * p + 1) as usize];
            let avg = (va + vb) / 2.0;
            out[p as usize] = avg;
            out[(half + p) as usize] = avg;
        }
        len = half;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_variance_is_unchanged_by_the_ladder() {
        for k in [1u32, 2, 4, 8] {
            let sig2 = vec![0.3f32; k as usize];
            let out = haar_variance_ladder(&sig2, k);
            for (idx, &v) in out.iter().enumerate() {
                assert!((v - 0.3).abs() < 1e-6, "k={k} idx={idx}: got {v}");
            }
        }
    }

    #[test]
    fn two_element_ladder_averages_the_pair() {
        let out = haar_variance_ladder(&[1.0, 0.0], 2);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn k_use_of_one_is_the_identity() {
        let out = haar_variance_ladder(&[0.7], 1);
        assert_eq!(out, vec![0.7]);
    }

    #[test]
    fn eight_element_ladder_matches_hand_computed_levels() {
        // Level 1 (len=8, half=4) averages the pairs (0,1), (2,3),
        // (4,5), (6,7), and writes each pair's average into both its
        // approximation slot and its detail slot, giving
        // [2, 2, 3, 2, 2, 2, 3, 2].
        //
        // Level 2 (len=4, half=2) only touches positions 0..4, again
        // writing each pair's average into both output slots, giving
        // [2, 2.5, 2, 2.5] there and leaving 4..8 alone.
        //
        // Level 3 (len=2, half=1) only touches positions 0..2, folding
        // that last pair down to [2.25, 2.25].
        let sig2 = vec![1.0, 3.0, 2.0, 2.0, 5.0, 1.0, 4.0, 0.0];
        let out = haar_variance_ladder(&sig2, 8);

        let expected = [2.25f32, 2.25, 2.0, 2.5, 2.0, 2.0, 3.0, 2.0];
        for (idx, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6, "idx={idx}: got {got} want {want}");
        }
    }
}

use cubecl::prelude::*;

use crate::collab::{MAX_K, PATCH_AREA, PATCH_SIZE};

// The register-layout helpers below stride a group by `PATCH_SIZE`,
// because one lane holds one 8-value column of each of `MAX_K` members
// and the two counts happen to be the same number. A caller sizes the
// array it passes them as `PATCH_AREA`, which is only enough for the
// whole group while that holds.
const _: () = assert!(
    MAX_K == PATCH_SIZE,
    "haar_reg_fwd_level and haar_reg_inv_level stride a group by PATCH_SIZE, which only holds \
     the whole stack while MAX_K matches it"
);

/// The floor the filter passes to [`safe_reciprocal`] when it turns a
/// retained variance sum into a group weight.
///
/// A group weight therefore never exceeds `1 / RECIPROCAL_FLOOR`, which
/// is the bound [`crate::collab::kernels::aggregate::weight_scale`]
/// needs to fit the weights into a fixed-point accumulator.
pub const RECIPROCAL_FLOOR: f32 = 1e-12;

/// A reciprocal weight that never trusts a driver's `NaN`-`max`
/// behaviour.
///
/// Returns `1 / max(denom, floor)` for a finite denominator. A `denom`
/// that is `NaN` or infinite is caught first and returns `0`.
///
/// The result only ever feeds a weight or a normalising divisor, never a
/// value that could stand in for real content, so zero is the right
/// fallback for both cases. Either means the reciprocal cannot be trusted
/// to mean anything.
///
/// The explicit check matters because `f32::max(NaN, floor)` is not
/// guaranteed to discard the `NaN` on a GPU the way it does on the CPU.
/// SPIR-V's `FMax` leaves a `NaN` operand undefined and only `NMax`
/// promises to discard it, so leaning on `max` alone would make
/// correctness depend on which one a driver lowers `f32::max` to.
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

/// Runs one forward 8-point DCT over a line a lane already holds in
/// registers.
///
/// `line` holds the 8 values on entry and the 8 coefficients on return.
/// The basis stays in shared memory, because every lane reads all 64 of
/// its entries and a per-lane copy would cost 64 registers to save
/// nothing.
///
/// The whole line is snapshotted before any output is written, so
/// writing an output cannot disturb an input a later output still needs.
/// A lane owns every value it touches, so no barrier is needed here or
/// around the call.
#[cube]
pub(crate) fn dct8_reg_fwd(basis: &SharedMemory<f32>, line: &mut Array<f32>) {
    let mut src = Array::<f32>::new(8usize);
    #[unroll]
    for i in 0..PATCH_SIZE {
        src[i as usize] = line[i as usize];
    }
    #[unroll]
    for j in 0..PATCH_SIZE {
        let mut sum = 0.0f32;
        #[unroll]
        for i in 0..PATCH_SIZE {
            sum += basis[(j * PATCH_SIZE + i) as usize] * src[i as usize];
        }
        line[j as usize] = sum;
    }
}

/// The inverse of [`dct8_reg_fwd`], the transpose of the same basis.
///
/// Because the basis is orthonormal its inverse is its own transpose, so
/// this reads `basis[j*8+i]` for output position `i` instead of output
/// position `j`, and sums over `j` instead of `i`, matching
/// [`dct8_reg_fwd`]'s calling convention otherwise.
#[cube]
pub(crate) fn dct8_reg_inv(basis: &SharedMemory<f32>, line: &mut Array<f32>) {
    let mut src = Array::<f32>::new(8usize);
    #[unroll]
    for j in 0..PATCH_SIZE {
        src[j as usize] = line[j as usize];
    }
    #[unroll]
    for i in 0..PATCH_SIZE {
        let mut sum = 0.0f32;
        #[unroll]
        for j in 0..PATCH_SIZE {
            sum += basis[(j * PATCH_SIZE + i) as usize] * src[j as usize];
        }
        line[i as usize] = sum;
    }
}

/// One level of the forward stack Haar, over a group a lane holds
/// entirely in registers.
///
/// `stack` holds `MAX_K` members of 8 values each, member `k` at
/// `k * PATCH_SIZE + pos`, which is the slice of a group one lane owns
/// when each lane holds one column. This runs the butterfly at every one
/// of those 8 positions at once, so one call covers the whole level.
///
/// Each butterfly is `(a, b) -> ((a+b)/sqrt(2), (a-b)/sqrt(2))`, which
/// has unit-norm rows and is its own inverse. A full multi-level
/// decomposition calls this once per level, over a length that halves
/// each time, with the coarsest approximation ending up at `k = 0` and
/// finer detail bands filling the rest in order.
///
/// This takes the level's length as a `#[comptime]` argument, so every
/// index into `stack` is a compile-time constant. A register array
/// indexed by a runtime value is not a register array, it is scratch
/// memory, and the whole point of holding the group in registers is
/// lost the moment one dynamic index appears. The caller picks the
/// levels with a run of predicates on the group size rather than a
/// loop.
///
/// The level snapshots every value it reads before writing any output,
/// because the output range overlaps the input range.
#[cube]
pub(crate) fn haar_reg_fwd_level(stack: &mut Array<f32>, #[comptime] len: u32) {
    let half = comptime!(len / 2);
    #[unroll]
    for pos in 0..PATCH_SIZE {
        let mut snapshot = Array::<f32>::new(MAX_K as usize);
        #[unroll]
        for k in 0..len {
            snapshot[k as usize] = stack[(k * PATCH_SIZE + pos) as usize];
        }
        #[unroll]
        for p in 0..half {
            let a = snapshot[(2u32 * p) as usize];
            let b = snapshot[(2u32 * p + 1u32) as usize];
            stack[(p * PATCH_SIZE + pos) as usize] = (a + b) * std::f32::consts::FRAC_1_SQRT_2;
            stack[((half + p) * PATCH_SIZE + pos) as usize] =
                (a - b) * std::f32::consts::FRAC_1_SQRT_2;
        }
    }
}

/// One level of the inverse stack Haar, the mirror of
/// [`haar_reg_fwd_level`].
///
/// The butterfly is its own inverse, so this level combines the
/// approximation half with the detail half and writes the pair back
/// interleaved. A caller runs the levels in the opposite order to the
/// forward pass.
#[cube]
pub(crate) fn haar_reg_inv_level(stack: &mut Array<f32>, #[comptime] len: u32) {
    let half = comptime!(len / 2);
    #[unroll]
    for pos in 0..PATCH_SIZE {
        let mut snapshot = Array::<f32>::new(MAX_K as usize);
        #[unroll]
        for k in 0..len {
            snapshot[k as usize] = stack[(k * PATCH_SIZE + pos) as usize];
        }
        #[unroll]
        for p in 0..half {
            let a = snapshot[p as usize];
            let b = snapshot[(half + p) as usize];
            stack[(2u32 * p * PATCH_SIZE + pos) as usize] =
                (a + b) * std::f32::consts::FRAC_1_SQRT_2;
            stack[((2u32 * p + 1u32) * PATCH_SIZE + pos) as usize] =
                (a - b) * std::f32::consts::FRAC_1_SQRT_2;
        }
    }
}

/// One level of the variance propagation that shadows
/// [`haar_reg_fwd_level`].
///
/// A Haar butterfly's two outputs are each a sum of two independent
/// inputs scaled by `1/sqrt(2)`, so both land on the same variance,
/// `(va + vb) / 2`. Running this over the same levels, in the same
/// pairing order, as the signal turns `v[j]` into the variance of stack
/// coefficient `j`.
///
/// [`haar_variance_ladder`] is the host mirror this is checked against.
/// This takes the level length as a `#[comptime]` argument for the
/// reason [`haar_reg_fwd_level`] gives.
#[cube]
pub(crate) fn variance_reg_level(v: &mut Array<f32>, #[comptime] len: u32) {
    let half = comptime!(len / 2);
    let mut snapshot = Array::<f32>::new(MAX_K as usize);
    #[unroll]
    for k in 0..len {
        snapshot[k as usize] = v[k as usize];
    }
    #[unroll]
    for p in 0..half {
        let avg = (snapshot[(2u32 * p) as usize] + snapshot[(2u32 * p + 1u32) as usize]) * 0.5f32;
        v[p as usize] = avg;
        v[(half + p) as usize] = avg;
    }
}

/// The per-DCT-frequency multiplier a spatially correlated residual
/// leaves on top of an otherwise flat noise variance.
///
/// Non-local means leaves a residual whose covariance falls off with
/// distance as `rho^d`, rather than the flat covariance a white-noise
/// model assumes. Projecting that through the same orthonormal 8-point
/// DCT basis [`fill_dct8_basis`] builds spreads a flat variance unevenly
/// across frequencies. Low frequencies pick up more of the noise power,
/// high frequencies less.
///
/// `g(u) = sum_i sum_j B_u(i) * B_u(j) * rho^|i-j|`, where `B_u` is row
/// `u` of that basis. A patch's coefficient at `(u, v)` scales by
/// `g(u) * g(v)`, since the correlation model treats rows and columns
/// alike and the 2D DCT runs as two separable 1D passes.
///
/// `sum_u g(u) = 8` for every `rho`, because the basis is orthonormal, so
/// this redistributes variance across frequencies rather than changing
/// the total and needs no separate normalisation.
///
/// `rho <= 0` returns `[1.0; 8]` directly. The sum below is exactly 1.0
/// there in exact arithmetic, but eight squared cosine terms in floating
/// point land a few bits off, and a caller with correlation shaping off
/// needs its result to be bit for bit what no shaping at all would give.
pub fn dct_noise_profile(rho: f32) -> [f32; 8] {
    if rho <= 0.0 {
        return [1.0; 8];
    }

    let rho = rho as f64;
    let mut basis = [[0.0f64; 8]; 8];
    for (u, row) in basis.iter_mut().enumerate() {
        let c = if u == 0 { 1.0 / 8.0f64.sqrt() } else { 0.5 };
        for (i, entry) in row.iter_mut().enumerate() {
            let angle = std::f64::consts::PI * (2.0 * i as f64 + 1.0) * u as f64 / 16.0;
            *entry = c * angle.cos();
        }
    }

    let mut g = [0.0f32; 8];
    for (u, slot) in g.iter_mut().enumerate() {
        let mut sum = 0.0f64;
        for (i, &bi) in basis[u].iter().enumerate() {
            for (j, &bj) in basis[u].iter().enumerate() {
                sum += bi * bj * rho.powi((i as i32 - j as i32).abs());
            }
        }
        *slot = sum as f32;
    }
    g
}

/// The host-side mirror of the variance propagation the stack Haar
/// applies to a per-coefficient noise variance instead of a signal.
///
/// A Haar butterfly's two outputs are each a sum of two independent
/// values scaled by `1/sqrt(2)`, so if `va` and `vb` are the variances
/// of `a` and `b`, both outputs land on the same variance, `(va +
/// vb) / 2`. Squaring `1/sqrt(2)` gives the `1/2`, and both outputs get
/// the same value because both are a sum of the same two inputs, just
/// with one sign flipped.
///
/// This runs the same multi-level recursion as [`haar_reg_fwd_level`],
/// over plain host `f32`, so a filter kernel can propagate a per-patch
/// sigma down to a per-coefficient sigma without a GPU round trip.
///
/// Every caller is a test oracle, so this only builds under `cfg(test)`
/// with a GPU runtime feature enabled, matching the callers themselves.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) fn haar_variance_ladder(sig2: &[f32], k_use: u32) -> Vec<f32> {
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
    fn dct_noise_profile_rho_zero_is_uniform_identity() {
        let g = dct_noise_profile(0.0);
        assert_eq!(
            g, [1.0f32; 8],
            "rho=0 must give exactly 1.0 at every frequency, got {g:?}"
        );

        // A small negative rho is not a real correlation either, and
        // must fall onto the same exact identity rather than running
        // the summation with a negative base.
        let g_neg = dct_noise_profile(-0.1);
        assert_eq!(g_neg, [1.0f32; 8]);
    }

    #[test]
    fn dct_noise_profile_sums_to_eight_across_a_range_of_rho() {
        for rho in [0.05f32, 0.3, 0.5, 0.67, 0.8, 0.85, 0.86, 0.95, 0.99] {
            let g = dct_noise_profile(rho);
            let sum: f32 = g.iter().sum();
            assert!(
                (sum - 8.0).abs() < 1e-3,
                "rho={rho}: expected sum(g) == 8.0 (variance redistributed, not created or \
                 destroyed), got {sum}"
            );
        }
    }

    #[test]
    fn dct_noise_profile_is_monotonically_decreasing_for_positive_rho() {
        for rho in [0.05f32, 0.3, 0.5, 0.67, 0.8, 0.85, 0.86, 0.95, 0.99] {
            let g = dct_noise_profile(rho);
            for u in 0..7 {
                assert!(
                    g[u] > g[u + 1],
                    "rho={rho}: expected g to strictly decrease with frequency (low frequencies \
                     carry more of a positively correlated residual's noise power), got \
                     g[{u}]={} <= g[{}]={}",
                    g[u],
                    u + 1,
                    g[u + 1],
                );
            }
        }
    }

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

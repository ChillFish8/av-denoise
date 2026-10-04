use cubecl::prelude::*;

use crate::collab::{MAX_K, PATCH_AREA, PATCH_SIZE};

// A lane holds one 8-value column of each of `MAX_K` members, so the Haar helpers stride a group
// by `PATCH_SIZE` and a `PATCH_AREA` array only holds the whole group while the two match.
const _: () = assert!(
    MAX_K == PATCH_SIZE,
    "haar_reg_fwd_level and haar_reg_inv_level stride a group by PATCH_SIZE, which only holds \
     the whole stack while MAX_K matches it"
);

/// The floor the filter passes to `safe_reciprocal` when turning a retained variance sum into a
/// group weight.
///
/// It caps a group weight at `1 / RECIPROCAL_FLOOR`, the bound
/// [weight_scale](crate::collab::kernels::aggregate::weight_scale) needs to fit the weights into a
/// fixed-point accumulator.
pub const RECIPROCAL_FLOOR: f32 = 1e-12;

/// Returns `1 / max(denom, floor)`, or 0 when `denom` is NaN or infinite.
///
/// Zero is a safe fallback because the result only ever feeds a weight or a divisor. The explicit
/// check is needed because SPIR-V's `FMax` leaves a NaN operand undefined, so `f32::max` alone may
/// not discard it on a GPU.
#[cube]
pub(crate) fn safe_reciprocal(denom: f32, floor: f32) -> f32 {
    let mut reciprocal = 0.0f32;
    if !denom.is_nan() && !denom.is_inf() {
        reciprocal = 1.0f32 / f32::max(denom, floor);
    }
    reciprocal
}

/// Fills an 8x8 orthonormal DCT-II basis into shared memory, one entry per thread.
///
/// Entry `j * 8 + i` holds `c_j * cos(PI * (2i + 1) * j / 16)`, with `c_0 = 1/sqrt(8)` and
/// `c_j = 0.5` otherwise. The basis is orthonormal, so its transpose is its inverse. Only the first
/// 64 threads write, and the caller must `sync_cube()` before reading `basis`.
#[cube]
pub(crate) fn fill_dct8_basis(basis: &mut SharedMemory<f32>, thread_id: u32) {
    if thread_id < PATCH_AREA {
        let i = thread_id % PATCH_SIZE;
        let j = thread_id / PATCH_SIZE;
        let mut row_scale = 0.5f32;
        if j == 0 {
            row_scale = 1.0f32 / f32::sqrt(8.0f32);
        }
        let angle = std::f32::consts::PI * (2.0f32 * i as f32 + 1.0f32) * j as f32 / 16.0f32;
        basis[thread_id as usize] = row_scale * f32::cos(angle);
    }
}

/// Runs a forward 8-point DCT over a line one lane holds in registers.
///
/// `line` holds 8 values on entry and 8 coefficients on return. The basis stays in shared memory
/// because a per-lane copy would cost 64 registers. The lane owns every value it touches, so no
/// barrier is needed.
#[cube]
pub(crate) fn dct8_reg_fwd(basis: &SharedMemory<f32>, line: &mut Array<f32>) {
    let mut snapshot = Array::<f32>::new(8usize);
    #[unroll]
    for i in 0..PATCH_SIZE {
        snapshot[i as usize] = line[i as usize];
    }

    #[unroll]
    for j in 0..PATCH_SIZE {
        let mut sum = 0.0f32;
        #[unroll]
        for i in 0..PATCH_SIZE {
            sum += basis[(j * PATCH_SIZE + i) as usize] * snapshot[i as usize];
        }
        line[j as usize] = sum;
    }
}

/// The inverse of `dct8_reg_fwd`, using the transpose of the same basis.
#[cube]
pub(crate) fn dct8_reg_inv(basis: &SharedMemory<f32>, line: &mut Array<f32>) {
    let mut snapshot = Array::<f32>::new(8usize);
    #[unroll]
    for j in 0..PATCH_SIZE {
        snapshot[j as usize] = line[j as usize];
    }

    #[unroll]
    for i in 0..PATCH_SIZE {
        let mut sum = 0.0f32;
        #[unroll]
        for j in 0..PATCH_SIZE {
            sum += basis[(j * PATCH_SIZE + i) as usize] * snapshot[j as usize];
        }
        line[i as usize] = sum;
    }
}

/// One level of the forward stack Haar over a group one lane holds in registers.
///
/// `stack` holds `MAX_K` members of 8 values, member `k` at `k * PATCH_SIZE + pos`, and the
/// butterfly `(a, b) -> ((a + b) / sqrt(2), (a - b) / sqrt(2))` runs at all 8 positions. The
/// butterfly is orthonormal and its own inverse. A full
/// decomposition calls this once per level with a halving `len`, leaving the coarsest
/// approximation at `k = 0`. `len` is comptime so every index into `stack` is a constant, because
/// one runtime index turns the register array into scratch memory.
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
            let first = snapshot[(2u32 * p) as usize];
            let second = snapshot[(2u32 * p + 1u32) as usize];
            stack[(p * PATCH_SIZE + pos) as usize] = (first + second) * std::f32::consts::FRAC_1_SQRT_2;
            stack[((half + p) * PATCH_SIZE + pos) as usize] =
                (first - second) * std::f32::consts::FRAC_1_SQRT_2;
        }
    }
}

/// One level of the inverse stack Haar, the mirror of `haar_reg_fwd_level`.
///
/// Levels run in the opposite order to the forward pass.
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
            let low = snapshot[p as usize];
            let high = snapshot[(half + p) as usize];
            stack[(2u32 * p * PATCH_SIZE + pos) as usize] = (low + high) * std::f32::consts::FRAC_1_SQRT_2;
            stack[((2u32 * p + 1u32) * PATCH_SIZE + pos) as usize] =
                (low - high) * std::f32::consts::FRAC_1_SQRT_2;
        }
    }
}

/// One level of the variance propagation that shadows `haar_reg_fwd_level`.
///
/// Both outputs of a butterfly carry `(va + vb) / 2`, so running this over the same levels turns
/// `v[j]` into the variance of stack coefficient `j`.
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
        let mean = (snapshot[(2u32 * p) as usize] + snapshot[(2u32 * p + 1u32) as usize]) * 0.5f32;
        v[p as usize] = mean;
        v[(half + p) as usize] = mean;
    }
}

/// The per-DCT-frequency variance multiplier for a residual whose covariance falls off as `rho^d`.
///
/// `g(u) = sum_i sum_j B_u(i) * B_u(j) * rho^|i-j|`, where `B_u` is row `u` of the basis
/// `fill_dct8_basis` builds, and a coefficient at `(u, v)` scales by `g(u) * g(v)`. The basis is
/// orthonormal, so `sum_u g(u) = 8` and the total variance is unchanged. `rho <= 0` returns exactly
/// `[1.0; 8]`, because the float sum lands a few bits off and an unshaped caller needs a bit-exact
/// result.
pub fn dct_noise_profile(rho: f32) -> [f32; 8] {
    if rho <= 0.0 {
        return [1.0; 8];
    }

    let rho = rho as f64;
    let mut basis = [[0.0f64; 8]; 8];
    for (u, row) in basis.iter_mut().enumerate() {
        let row_scale = if u == 0 { 1.0 / 8.0f64.sqrt() } else { 0.5 };
        for (i, entry) in row.iter_mut().enumerate() {
            let angle = std::f64::consts::PI * (2.0 * i as f64 + 1.0) * u as f64 / 16.0;
            *entry = row_scale * angle.cos();
        }
    }

    let mut profile = [0.0f32; 8];
    for (u, slot) in profile.iter_mut().enumerate() {
        let mut sum = 0.0f64;
        for (i, &basis_i) in basis[u].iter().enumerate() {
            for (j, &basis_j) in basis[u].iter().enumerate() {
                let distance = (i as i32 - j as i32).abs();
                sum += basis_i * basis_j * rho.powi(distance);
            }
        }
        *slot = sum as f32;
    }
    profile
}

/// Host mirror of the variance propagation `variance_reg_level` applies over `k_use` members.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) fn haar_variance_ladder(sig2: &[f32], k_use: u32) -> Vec<f32> {
    let mut variances = sig2.to_vec();
    let mut len = k_use;
    while len > 1 {
        let half = len / 2;
        let snapshot = variances[..len as usize].to_vec();
        for p in 0..half {
            let first = snapshot[(2 * p) as usize];
            let second = snapshot[(2 * p + 1) as usize];
            let mean = (first + second) / 2.0;
            variances[p as usize] = mean;
            variances[(half + p) as usize] = mean;
        }
        len = half;
    }
    variances
}

#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
mod tests {
    use super::*;

    #[test]
    fn dct_noise_profile_rho_zero_is_uniform_identity() {
        let profile = dct_noise_profile(0.0);
        assert_eq!(
            profile, [1.0f32; 8],
            "rho=0 must give exactly 1.0 at every frequency, got {profile:?}"
        );

        // A negative rho is not a real correlation, so it takes the same exact identity.
        let negative_profile = dct_noise_profile(-0.1);
        assert_eq!(negative_profile, [1.0f32; 8]);
    }

    #[test]
    fn dct_noise_profile_sums_to_eight_across_a_range_of_rho() {
        for rho in [0.05f32, 0.3, 0.5, 0.67, 0.8, 0.85, 0.86, 0.95, 0.99] {
            let profile = dct_noise_profile(rho);
            let sum: f32 = profile.iter().sum();
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
            let profile = dct_noise_profile(rho);
            for u in 0..7 {
                assert!(
                    profile[u] > profile[u + 1],
                    "rho={rho}: expected g to strictly decrease with frequency (low frequencies \
                     carry more of a positively correlated residual's noise power), got \
                     g[{u}]={} <= g[{}]={}",
                    profile[u],
                    u + 1,
                    profile[u + 1],
                );
            }
        }
    }

    #[test]
    fn uniform_variance_is_unchanged_by_the_ladder() {
        for k in [1u32, 2, 4, 8] {
            let sig2 = vec![0.3f32; k as usize];
            let variances = haar_variance_ladder(&sig2, k);
            for (idx, &variance) in variances.iter().enumerate() {
                assert!((variance - 0.3).abs() < 1e-6, "k={k} idx={idx}: got {variance}");
            }
        }
    }

    #[test]
    fn two_element_ladder_averages_the_pair() {
        let variances = haar_variance_ladder(&[1.0, 0.0], 2);
        assert_eq!(variances.len(), 2);
        assert!((variances[0] - 0.5).abs() < 1e-6);
        assert!((variances[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn k_use_of_one_is_the_identity() {
        let variances = haar_variance_ladder(&[0.7], 1);
        assert_eq!(variances, vec![0.7]);
    }

    #[test]
    fn eight_element_ladder_matches_hand_computed_levels() {
        // Level 1 averages each pair into both its slots, giving [2, 2, 3, 2, 2, 2, 3, 2]. Level 2
        // turns positions 0..4 into [2, 2.5, 2, 2.5], and level 3 folds 0..2 into [2.25, 2.25].
        let sig2 = vec![1.0, 3.0, 2.0, 2.0, 5.0, 1.0, 4.0, 0.0];
        let variances = haar_variance_ladder(&sig2, 8);

        let expected = [2.25f32, 2.25, 2.0, 2.5, 2.0, 2.0, 3.0, 2.0];
        for (idx, (&got, &want)) in variances.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-6, "idx={idx}: got {got} want {want}");
        }
    }
}

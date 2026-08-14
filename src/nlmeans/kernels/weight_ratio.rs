use cubecl::prelude::*;

/// Reduces the per-pixel residual-noise ratio to one partial sum per
/// block.
///
/// A weighted average of samples that each carry independent noise of
/// variance `sigma^2`, combined with weights `w_i`, leaves behind a
/// residual variance of `sigma^2 * sum(w_i^2) / sum(w_i)^2`. The second
/// factor is the ratio this kernel computes, so it says how much of the
/// input noise variance survives the average at each pixel, independent
/// of what `sigma` actually is.
///
/// `ws` and `wsq` are one pixel's accumulated weight sum and
/// weight-squared sum. `m` is `self_weight * max_weight`, the centre
/// pixel's own contribution exactly as `nlm_finish` applies it, since the
/// centre pixel is itself one of the noisy samples being averaged. The
/// per-pixel ratio is `(wsq + m * m) / (ws + m)^2`.
///
/// A denominator too small to divide by, meaning nothing matched and the
/// centre carried no self weight either, contributes 1.0 instead. That
/// mirrors `nlm_finish` keeping the original pixel unchanged in the same
/// case, which removes no noise at all. The guard checks the squared
/// denominator directly, the quantity the division actually uses, since a
/// denominator can be comfortably nonzero on its own and still underflow
/// to exactly zero once squared in `f32`.
///
/// Each block covers a strided share of the pixels, the same grid-stride
/// pattern `gpu_zero_one` walks, and sums their ratios into shared
/// memory. Thread zero then folds that block's share into one partial,
/// written to `partials[CUBE_POS_X]`. The host sums the partials and
/// divides by the pixel count to get the mean, which is what
/// `NlmDenoiser::residual_ratio_sqrt` then takes the square root of.
#[cube(launch_unchecked)]
pub fn nlm_weight_ratio_partial(
    weight_sum: &Array<f32>,
    weight_sq_sum: &Array<f32>,
    max_weight: &Array<f32>,
    partials: &mut Array<f32>,
    self_weight: f32,
    pixels: u32,
    #[comptime] total_threads: u32,
    #[comptime] block: u32,
) {
    let mut scratch = SharedMemory::<f32>::new(block as usize);
    let tid = UNIT_POS_X;

    let mut sum = 0.0f32;
    let mut idx = ABSOLUTE_POS_X;
    while idx < pixels {
        let m = self_weight * max_weight[idx as usize];
        let denom = weight_sum[idx as usize] + m;
        let denom_sq = denom * denom;
        // The guard checks `denom_sq`, the quantity actually divided by,
        // rather than `denom` itself. A `denom` around `1e-24` passes a
        // guard written against `denom` at `1e-30`, but squaring it first
        // underflows to exactly `0.0` in `f32`, whose smallest positive
        // value is about `1.4e-45`. `weight_sq_sum` built from the same
        // tiny weights underflows the same way, turning the division into
        // `0.0 / 0.0`, which is `NaN`. Guarding the squared value directly
        // catches every denominator that would underflow once squared,
        // not only the ones that were already at or near zero before
        // squaring.
        //
        // The `.into()` on the fallback arm is what lets cubecl unify
        // the two branches, because both have to expand to the same
        // `NativeExpand<f32>`. Clippy cannot see that requirement.
        #[allow(clippy::useless_conversion)]
        let ratio = if denom_sq > 1e-30f32 {
            let num = weight_sq_sum[idx as usize] + m * m;
            num / denom_sq
        } else {
            1.0f32.into()
        };
        sum += ratio;
        idx += total_threads;
    }
    scratch[tid as usize] = sum;

    sync_cube();

    if tid == 0 {
        let mut total = 0.0f32;
        for t in 0..block {
            total += scratch[t as usize];
        }
        partials[CUBE_POS_X as usize] = total;
    }
}

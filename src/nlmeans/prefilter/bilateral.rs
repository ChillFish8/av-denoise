use cubecl::prelude::*;

use super::PrefilterCtx;
use crate::nlmeans::kernels::nlm_bilateral;
use crate::nlmeans::{BLOCK_X, BLOCK_Y};

/// Comptime radius derived from `sigma_s`. Truncating at `2·σ` covers
/// >95% of the Gaussian mass and bounds SMEM/register usage.
pub fn bilateral_radius(sigma_s: f32) -> u32 {
    ((2.0 * sigma_s).ceil() as u32).max(1)
}

/// Reciprocal normalisation factor for the bilateral Gaussian kernel's
/// spatial or range term (`1 / (2·σ²)`, shared by both since they use
/// the same Gaussian shape). Computed host-side and passed to the GPU
/// kernel as a plain `f32`, so this is the one place the value is ever
/// derived from `sigma`; the kernel itself only ever multiplies by it.
/// Squaring a very small but positive `sigma` can underflow to `0.0` in
/// `f32`, which makes this reciprocal infinite even though `sigma`
/// itself is finite and positive. Callers validating user input should
/// check this value rather than just the sign and finiteness of
/// `sigma`.
pub(crate) fn inv_two_sigma_sq(sigma: f32) -> f32 {
    1.0 / (2.0 * sigma * sigma)
}

pub(super) fn run_bilateral<R: Runtime>(
    client: &ComputeClient<R>,
    ctx: &PrefilterCtx<'_>,
    sigma_s: f32,
    sigma_r: f32,
) -> Result<(), anyhow::Error> {
    let radius = bilateral_radius(sigma_s);
    let total = (ctx.frame_count * ctx.height * ctx.width * ctx.stored_ch) as usize;
    let stored_ch = ctx.stored_ch as usize;

    let inv_two_sigma_s_sq = inv_two_sigma_sq(sigma_s);
    let inv_two_sigma_r_sq = inv_two_sigma_sq(sigma_r);

    unsafe {
        nlm_bilateral::launch_unchecked::<R>(
            client,
            CubeCount::new_2d(ctx.width.div_ceil(BLOCK_X), ctx.height.div_ceil(BLOCK_Y)),
            CubeDim::new_2d(BLOCK_X, BLOCK_Y),
            stored_ch,
            ArrayArg::from_raw_parts(ctx.input_buf.clone(), total),
            ArrayArg::from_raw_parts(ctx.reference_buf.clone(), total),
            ctx.frame,
            inv_two_sigma_s_sq,
            inv_two_sigma_r_sq,
            ctx.width,
            ctx.height,
            ctx.channels,
            radius,
            BLOCK_X,
            BLOCK_Y,
        );
    }

    Ok(())
}

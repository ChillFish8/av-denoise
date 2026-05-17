use cubecl::prelude::*;

use super::PrefilterCtx;
use crate::nlmeans::kernels::nlm_bilateral;
use crate::nlmeans::{BLOCK_X, BLOCK_Y};

/// Comptime radius derived from `sigma_s`. Truncating at `2·σ` covers
/// >95% of the Gaussian mass and bounds SMEM/register usage.
pub fn bilateral_radius(sigma_s: f32) -> u32 {
    ((2.0 * sigma_s).ceil() as u32).max(1)
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

    let inv_two_sigma_s_sq = 1.0 / (2.0 * sigma_s * sigma_s);
    let inv_two_sigma_r_sq = 1.0 / (2.0 * sigma_r * sigma_r);

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

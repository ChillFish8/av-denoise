use cubecl::prelude::*;

use super::PrefilterCtx;
use crate::nlmeans::kernels::nlm_bilateral;
use crate::nlmeans::{BLOCK_X, BLOCK_Y};

/// The kernel radius for `sigma_s`.
///
/// Two sigma covers over 95% of the Gaussian's mass while keeping shared memory and register use
/// bounded.
pub fn bilateral_radius(sigma_s: f32) -> u32 {
    ((2.0 * sigma_s).ceil() as u32).max(1)
}

/// The bilateral Gaussian's normalisation factor `1 / (2 * sigma^2)`.
///
/// The spatial and range terms share it, and the kernel only multiplies by it. A tiny positive
/// sigma can square to 0.0 in `f32` and make it infinite, so validation checks this value rather
/// than the sigma alone.
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

    let cubes_x = ctx.width.div_ceil(BLOCK_X);
    let cubes_y = ctx.height.div_ceil(BLOCK_Y);

    unsafe {
        nlm_bilateral::launch_unchecked::<R>(
            client,
            CubeCount::new_2d(cubes_x, cubes_y),
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

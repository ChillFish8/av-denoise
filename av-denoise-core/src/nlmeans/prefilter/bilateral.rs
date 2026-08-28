use cubecl::prelude::*;

use super::PrefilterCtx;
use crate::nlmeans::kernels::nlm_bilateral;
use crate::nlmeans::{BLOCK_X, BLOCK_Y};

/// The kernel radius derived from `sigma_s`.
///
/// Stopping at two sigma covers over 95% of the Gaussian's mass, and it
/// keeps shared memory and register use bounded.
pub fn bilateral_radius(sigma_s: f32) -> u32 {
    ((2.0 * sigma_s).ceil() as u32).max(1)
}

/// The normalisation factor `1 / (2 * sigma^2)` for the bilateral
/// Gaussian.
///
/// The spatial and range terms share this because they use the same
/// Gaussian shape.
///
/// It is computed on the host and passed to the kernel as a plain `f32`,
/// so this is the only place the value is derived from a sigma. The
/// kernel only ever multiplies by it.
///
/// Squaring a very small but positive sigma can underflow to 0.0 in
/// `f32`, which makes this factor infinite even though the sigma itself
/// was finite and positive. Code validating user input should check this
/// value rather than only the sign and finiteness of the sigma.
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

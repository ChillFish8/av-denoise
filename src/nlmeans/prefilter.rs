//! Prefilter plumbing. Each variant of [`PrefilterMode`] either
//! supplies the reference clip externally or is dispatched on the GPU
//! during `push_frame`. The reference clip is then consumed by the
//! `_ref` distance kernels so weight calculation sees a cleaner image
//! than the noisy input.

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::kernels::nlm_bilateral;
use super::{BLOCK_X, BLOCK_Y};

/// How the per-frame reference clip is produced.
///
/// `Bilateral` and any future GPU-internal variants run a kernel
/// during `push_frame`. `External` requires the caller to supply a
/// reference frame via [`super::NlmDenoiser::push_frame_with_reference`].
/// `None` disables the reference path entirely (zero-cost).
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum PrefilterMode {
    #[default]
    None,
    External,
    Bilateral {
        sigma_s: f32,
        sigma_r: f32,
    },
}

impl PrefilterMode {
    /// Whether the denoiser needs to allocate the reference ring buffer.
    pub(crate) fn needs_reference_buf(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Whether the variant computes its reference on the GPU during
    /// `push_frame` (as opposed to consuming a caller-supplied clip).
    pub(crate) fn is_gpu_internal(self) -> bool {
        matches!(self, Self::Bilateral { .. })
    }
}

/// Inputs for a single-slot prefilter dispatch. Lives only for the
/// duration of one `push_frame`, so borrows on the denoiser's buffers
/// are sound.
pub(crate) struct PrefilterCtx<'a> {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub stored_ch: u32,
    pub frame_count: u32,
    pub frame: u32,
    pub input_buf: &'a Handle,
    pub reference_buf: &'a Handle,
}

/// Dispatch the GPU prefilter for the most recently uploaded frame.
/// `None` and `External` are no-ops.
pub(crate) fn run_prefilter<R: Runtime>(
    mode: PrefilterMode,
    client: &ComputeClient<R>,
    ctx: &PrefilterCtx<'_>,
) -> Result<(), anyhow::Error> {
    match mode {
        PrefilterMode::None | PrefilterMode::External => Ok(()),
        PrefilterMode::Bilateral { sigma_s, sigma_r } => run_bilateral::<R>(client, ctx, sigma_s, sigma_r),
    }
}

/// Comptime radius derived from `sigma_s`. Truncating at `2·σ` covers
/// >95% of the Gaussian mass and bounds SMEM/register usage.
pub fn bilateral_radius(sigma_s: f32) -> u32 {
    ((2.0 * sigma_s).ceil() as u32).max(1)
}

fn run_bilateral<R: Runtime>(
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

    nlm_bilateral::launch::<R>(
        client,
        CubeCount::new_2d(ctx.width.div_ceil(BLOCK_X), ctx.height.div_ceil(BLOCK_Y)),
        CubeDim::new_2d(BLOCK_X, BLOCK_Y),
        unsafe { ArrayArg::from_raw_parts::<f32>(ctx.input_buf, total, stored_ch) },
        unsafe { ArrayArg::from_raw_parts::<f32>(ctx.reference_buf, total, stored_ch) },
        ScalarArg::new(ctx.frame),
        ScalarArg::new(inv_two_sigma_s_sq),
        ScalarArg::new(inv_two_sigma_r_sq),
        ctx.width,
        ctx.height,
        ctx.channels,
        ctx.stored_ch,
        radius,
        BLOCK_X,
        BLOCK_Y,
    )?;

    Ok(())
}

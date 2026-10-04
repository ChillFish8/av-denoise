use cubecl::prelude::*;
use cubecl::server::Handle;

use super::stats::{lower_quartile, sort_ascending};
use crate::nlmeans::align::StorageAlign;
use crate::nlmeans::kernels::{nlm_noise_partial, nlm_noise_reduce};
use crate::nlmeans::{BLOCK_1D, BLOCK_X, BLOCK_Y};

/// The inputs one Immerkær noise estimate needs.
pub(in crate::nlmeans) struct NoiseCtx<'a> {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub stored_ch: u32,
    pub frame_count: u32,
    pub frame: u32,
    pub slot: u32,
    pub input_buf: &'a Handle,
    pub partials_buf: &'a Handle,
    pub results_buf: &'a Handle,
}

/// The `f32` length of one frame's first-stage partials, four lanes per block.
pub(in crate::nlmeans) fn partials_len(width: u32, height: u32) -> usize {
    (width.div_ceil(BLOCK_X) * height.div_ceil(BLOCK_Y) * 4) as usize
}

/// The byte stride between partials ring slots, padded to the buffer-binding alignment.
///
/// wgpu rejects a bind-group offset that is not a multiple of its
/// `min_storage_buffer_offset_alignment`, and a small frame can leave [partials_len] short of one.
pub(in crate::nlmeans) fn noise_partials_slot_stride_bytes(
    width: u32,
    height: u32,
    align: StorageAlign,
) -> u64 {
    let partials_bytes = partials_len(width, height) as u64 * size_of::<f32>() as u64;
    align.pad_bytes(partials_bytes)
}

/// Runs both stages of the Immerkær noise estimate for one frame.
///
/// The mask is cheap and needs only one frame, but it reads correlated grain low because the grain
/// looks partly like content to it. The per-channel totals land in this frame's slot of the results
/// buffer, which holds four values per slot of the input ring's frame capacity.
pub(in crate::nlmeans) fn run_noise_estimate<R: Runtime>(
    client: &ComputeClient<R>,
    ctx: &NoiseCtx<'_>,
) -> Result<(), anyhow::Error> {
    let total_input = (ctx.frame_count * ctx.height * ctx.width * ctx.stored_ch) as usize;
    let partials_count = partials_len(ctx.width, ctx.height);
    let total_results = (ctx.frame_count * 4) as usize;
    let stored_ch = ctx.stored_ch as usize;
    let cubes_x = ctx.width.div_ceil(BLOCK_X);
    let cubes_y = ctx.height.div_ceil(BLOCK_Y);

    unsafe {
        nlm_noise_partial::launch_unchecked::<R>(
            client,
            CubeCount::new_2d(cubes_x, cubes_y),
            CubeDim::new_2d(BLOCK_X, BLOCK_Y),
            stored_ch,
            ArrayArg::from_raw_parts(ctx.input_buf.clone(), total_input),
            ArrayArg::from_raw_parts(ctx.partials_buf.clone(), partials_count),
            ctx.frame,
            ctx.width,
            ctx.height,
            ctx.channels,
            BLOCK_X,
            BLOCK_Y,
        );
    }

    let block_count = (partials_count / 4) as u32;

    unsafe {
        nlm_noise_reduce::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(BLOCK_1D),
            ArrayArg::from_raw_parts(ctx.partials_buf.clone(), partials_count),
            ArrayArg::from_raw_parts(ctx.results_buf.clone(), total_results),
            ctx.slot,
            block_count,
            BLOCK_1D,
        );
    }

    Ok(())
}

/// Turns the summed absolute mask responses into an Immerkær sigma.
///
/// The interior area leaves out the one-pixel border the mask cannot reach.
pub(in crate::nlmeans) fn sigma_from_abs_sum(abs_sum: f32, width: u32, height: u32) -> f32 {
    let interior = ((width - 2) as f32) * ((height - 2) as f32);
    (std::f32::consts::FRAC_PI_2).sqrt() * abs_sum / (6.0 * interior)
}

/// The per-channel lower quartile of the per-block Immerkær sigmas in one slot's partials.
///
/// Each block's sigma uses the [sigma_from_abs_sum] formula over its tile's overlap with the frame
/// interior. A block with no overlap is skipped so a spurious zero cannot dilute the quartile.
/// Where noise is uneven across a frame this reads lower than the frame-wide mean, which makes it
/// the cautious estimate. Channels past the active count stay 0.
pub(in crate::nlmeans) fn sigma_block_p25_from_partials(
    partials: &[f32],
    channels: u32,
    width: u32,
    height: u32,
) -> [f32; 3] {
    let cubes_x = width.div_ceil(BLOCK_X);
    let cubes_y = height.div_ceil(BLOCK_Y);
    let channels = channels as usize;

    let mut cube_sigmas: Vec<Vec<f32>> = vec![Vec::new(); channels];

    for cube_y in 0..cubes_y {
        let tile_y0 = cube_y * BLOCK_Y;
        let tile_y1 = ((cube_y + 1) * BLOCK_Y).min(height);
        let overlap_y0 = tile_y0.max(1);
        let overlap_y1 = tile_y1.min(height - 1);

        for cube_x in 0..cubes_x {
            let tile_x0 = cube_x * BLOCK_X;
            let tile_x1 = ((cube_x + 1) * BLOCK_X).min(width);
            let overlap_x0 = tile_x0.max(1);
            let overlap_x1 = tile_x1.min(width - 1);

            if overlap_x1 <= overlap_x0 || overlap_y1 <= overlap_y0 {
                continue;
            }

            let area = ((overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)) as f32;
            let cube_index = (cube_y * cubes_x + cube_x) as usize;
            let base = cube_index * 4;

            for (channel, sigmas) in cube_sigmas.iter_mut().enumerate() {
                let sum = partials[base + channel];
                let cube_sigma = std::f32::consts::FRAC_PI_2.sqrt() * sum / (6.0 * area);
                sigmas.push(cube_sigma);
            }
        }
    }

    let mut sigma_low = [0.0f32; 3];
    for (channel, sigmas) in cube_sigmas.iter_mut().enumerate() {
        if sigmas.is_empty() {
            continue;
        }

        sort_ascending(sigmas);
        sigma_low[channel] = lower_quartile(sigmas);
    }

    sigma_low
}

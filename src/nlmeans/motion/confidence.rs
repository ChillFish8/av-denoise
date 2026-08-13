use cubecl::prelude::*;
use cubecl::server::Handle;

use super::MotionCtx;
use super::analyse::confidence_byte_offset;
use super::pyramid::{level_dims, pyramid_slot_byte_offset};
use crate::nlmeans::kernels::motion::nlm_mc_block_match_fine;

/// Per-pixel mean-mismatch threshold in `[0, 1]` luma units (about
/// 5/255), the MDegrain-style reference point for "this block no
/// longer matches". Scaled by block area and `thsad_scale` to produce
/// the actual `thsad` a caller compares excess SAD against.
pub(crate) const THSAD_PIXEL: f32 = 0.02;

/// Expected block SAD between two independently noisy copies of
/// otherwise identical content, so it can be subtracted before a real
/// mismatch is judged. Each pixel's noisy-vs-noisy absolute difference
/// is the magnitude of a zero-mean Gaussian of scale `σ_y√2`, which has
/// mean `σ_y√2 · √(2/π) = 2σ_y/√π`. Summed over the block's
/// `blksize²` pixels.
pub(crate) fn sad_noise_floor(blksize: u32, sigma_y: f32) -> f32 {
    let block_area = (blksize * blksize) as f32;
    block_area * 2.0 * sigma_y / std::f32::consts::PI.sqrt()
}

/// Excess-SAD value at which confidence collapses to zero, scaled by
/// block area so it stays comparable across block sizes. `thsad_scale`
/// is the user-facing multiplier ([`crate::nlmeans::HqParams::thsad_scale`]).
pub(crate) fn thsad(blksize: u32, thsad_scale: f32) -> f32 {
    let block_area = (blksize * blksize) as f32;
    thsad_scale * block_area * THSAD_PIXEL
}

/// Confidence-only block match for one (centre, neighbour) pair, used
/// when motion compensation is off. Runs a single-candidate SAD (no
/// coarse seed, no search window) at level 0 of `luma_pyramid` and
/// writes the resulting confidence into `confidence` at the slot
/// reserved for `neighbour_idx`. The matching MV write is discarded
/// into `mv_scratch`, since without motion compensation nothing warps
/// by it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_confidence_for_neighbour<R: Runtime>(
    client: &ComputeClient<R>,
    ctx: &MotionCtx,
    width: u32,
    height: u32,
    frame_count: u32,
    centre_slot: u32,
    neighbour_slot: u32,
    neighbour_idx: u32,
    luma_pyramid: &Handle,
    mv_scratch: &Handle,
    confidence: &Handle,
    sad_noise_floor: f32,
    thsad: f32,
) -> Result<(), anyhow::Error> {
    let (fw, fh) = level_dims(width, height, 0);
    let centre = luma_pyramid.clone().offset_start(pyramid_slot_byte_offset(
        width,
        height,
        frame_count,
        0,
        centre_slot,
    ));
    let neighbour = luma_pyramid.clone().offset_start(pyramid_slot_byte_offset(
        width,
        height,
        frame_count,
        0,
        neighbour_slot,
    ));
    let level_len = (fw * fh) as usize;

    let conf_offset = confidence_byte_offset(ctx, neighbour_idx);
    let conf_slot = confidence.clone().offset_start(conf_offset);
    let conf_slot_len = (ctx.blocks_x as usize) * (ctx.blocks_y as usize);
    let mv_slot_len = (ctx.blocks_x as usize) * (ctx.blocks_y as usize) * 2;

    let grid = CubeCount::new_2d(ctx.blocks_x, ctx.blocks_y);
    let dim = CubeDim::new_2d(8, 8);

    unsafe {
        nlm_mc_block_match_fine::launch_unchecked::<R>(
            client,
            grid,
            dim,
            ArrayArg::from_raw_parts(centre, level_len),
            ArrayArg::from_raw_parts(neighbour, level_len),
            ArrayArg::from_raw_parts(mv_scratch.clone(), mv_slot_len),
            ArrayArg::from_raw_parts(conf_slot, conf_slot_len),
            true, // the no-MC confidence pass always wants its output
            sad_noise_floor,
            thsad,
            fw,
            fh,
            ctx.blksize,
            ctx.step,
            ctx.search_radius,
            0u32,
            ctx.blocks_x,
            ctx.blocks_y,
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sad_noise_floor_scales_with_block_area_and_sigma() {
        let sigma = 4.0 / 255.0;
        let got = sad_noise_floor(16, sigma);
        let expected = (16 * 16) as f32 * 2.0 * sigma / std::f32::consts::PI.sqrt();
        assert!((got - expected).abs() < 1e-6, "expected {expected}, got {got}");
    }

    #[test]
    fn sad_noise_floor_zero_for_zero_sigma() {
        assert_eq!(sad_noise_floor(16, 0.0), 0.0);
    }

    #[test]
    fn thsad_scales_with_block_area_and_scale() {
        let got = thsad(16, 2.0);
        let expected = 2.0 * (16 * 16) as f32 * THSAD_PIXEL;
        assert!((got - expected).abs() < 1e-6, "expected {expected}, got {got}");
    }

    #[test]
    fn thsad_default_scale_is_positive() {
        assert!(thsad(16, 1.0) > 0.0);
    }
}

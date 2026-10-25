use cubecl::prelude::*;
use cubecl::server::Handle;

use super::MotionCtx;
use super::analyse::{MatchTargets, launch_fine};

/// The average per-pixel error at which a block stops counting as a match.
///
/// It is in normalised luma units, roughly 5 of 255, which is the reference point MDegrain uses.
pub(crate) const THSAD_PIXEL: f32 = 0.02;

/// The block SAD two noisy copies of the same content are expected to score by chance.
///
/// Subtracting it first judges a block on its content rather than its noise. Each pixel's
/// noisy-against-noisy difference is the magnitude of a zero-mean Gaussian with scale
/// `sigma * sqrt(2)`, whose mean is `2 * sigma / sqrt(pi)`, summed over all `blksize^2` pixels.
pub(crate) fn sad_noise_floor(blksize: u32, sigma_y: f32) -> f32 {
    let block_area = (blksize * blksize) as f32;
    block_area * 2.0 * sigma_y / std::f32::consts::PI.sqrt()
}

/// How far past the noise floor a block can score before its confidence reaches zero.
///
/// It scales with block area so it stays comparable across block sizes. `thsad_scale` is
/// [HqParams::thsad_scale](crate::nlmeans::HqParams::thsad_scale).
pub(crate) fn thsad(blksize: u32, thsad_scale: f32) -> f32 {
    let block_area = (blksize * blksize) as f32;
    thsad_scale * block_area * THSAD_PIXEL
}

/// Scores how well each neighbour matches the centre frame without a motion search.
///
/// Each block is scored where it stands at pyramid level 0, into its neighbour's `confidence`
/// slice, against word `sad_field` of the derived noise values `noise`. The motion vectors the
/// kernel also writes go to `mv_field` and are discarded.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
pub(crate) fn run_confidence<R: Runtime>(
    client: &ComputeClient<R>,
    ctx: &MotionCtx,
    width: u32,
    height: u32,
    targets: &MatchTargets<'_>,
    noise: &Handle,
    sad_field: u32,
    thsad: f32,
) -> Result<(), anyhow::Error> {
    debug_assert!(
        targets.write_confidence,
        "the confidence pass always writes confidence"
    );

    launch_fine::<R>(
        client,
        ctx,
        width,
        height,
        targets,
        ctx.search_radius,
        0u32,
        noise,
        sad_field,
        thsad,
    );

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
        let floor = sad_noise_floor(16, 0.0);
        assert_eq!(floor, 0.0);
    }

    #[test]
    fn thsad_scales_with_block_area_and_scale() {
        let got = thsad(16, 2.0);
        let expected = 2.0 * (16 * 16) as f32 * THSAD_PIXEL;
        assert!((got - expected).abs() < 1e-6, "expected {expected}, got {got}");
    }

    #[test]
    fn thsad_default_scale_is_positive() {
        let threshold = thsad(16, 1.0);
        assert!(threshold > 0.0);
    }
}

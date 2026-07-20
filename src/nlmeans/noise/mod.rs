use cubecl::prelude::*;
use cubecl::server::Handle;

use super::kernels::{nlm_noise_partial, nlm_noise_reduce};
use super::{BLOCK_1D, BLOCK_X, BLOCK_Y};

/// Inputs for a single-slot Immerkær noise estimate dispatch. Lives
/// only for the duration of one estimate call, so borrows on the
/// denoiser's buffers are sound.
pub(super) struct NoiseCtx<'a> {
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

/// Number of `f32` elements the stage-1 partials buffer needs for a
/// `width × height` frame. One cube covers a `BLOCK_X × BLOCK_Y` tile
/// and contributes four lanes.
pub(super) fn partials_len(width: u32, height: u32) -> usize {
    (width.div_ceil(BLOCK_X) * height.div_ceil(BLOCK_Y) * 4) as usize
}

/// Dispatch both stages of the Immerkær noise estimate for `ctx.frame`,
/// writing the per-channel absolute-mask-response totals into
/// `ctx.results_buf` at `ctx.slot`. The results buffer is sized for
/// `ctx.frame_count` ring slots (four `f32`s each), mirroring the input
/// ring's frame capacity.
pub(super) fn run_noise_estimate<R: Runtime>(
    client: &ComputeClient<R>,
    ctx: &NoiseCtx<'_>,
) -> Result<(), anyhow::Error> {
    let total_input = (ctx.frame_count * ctx.height * ctx.width * ctx.stored_ch) as usize;
    let n_partials = partials_len(ctx.width, ctx.height);
    let total_results = (ctx.frame_count * 4) as usize;
    let stored_ch = ctx.stored_ch as usize;

    unsafe {
        nlm_noise_partial::launch_unchecked::<R>(
            client,
            CubeCount::new_2d(ctx.width.div_ceil(BLOCK_X), ctx.height.div_ceil(BLOCK_Y)),
            CubeDim::new_2d(BLOCK_X, BLOCK_Y),
            stored_ch,
            ArrayArg::from_raw_parts(ctx.input_buf.clone(), total_input),
            ArrayArg::from_raw_parts(ctx.partials_buf.clone(), n_partials),
            ctx.frame,
            ctx.width,
            ctx.height,
            ctx.channels,
            BLOCK_X,
            BLOCK_Y,
        );
    }

    let num_partials = (n_partials / 4) as u32;
    unsafe {
        nlm_noise_reduce::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(BLOCK_1D),
            ArrayArg::from_raw_parts(ctx.partials_buf.clone(), n_partials),
            ArrayArg::from_raw_parts(ctx.results_buf.clone(), total_results),
            ctx.slot,
            num_partials,
            BLOCK_1D,
        );
    }

    Ok(())
}

/// Immerkær estimate from the summed absolute mask responses. The
/// interior area excludes the one-pixel border the mask cannot reach.
pub(super) fn sigma_from_abs_sum(abs_sum: f32, width: u32, height: u32) -> f32 {
    let interior = ((width - 2) as f32) * ((height - 2) as f32);
    (std::f32::consts::FRAC_PI_2).sqrt() * abs_sum / (6.0 * interior)
}

/// EMA weight for the newest frame estimate.
const EMA_ALPHA: f32 = 0.2;
/// Lower bound on the smoothed sigma in `[0, 1]` units (0.1 in 8-bit
/// terms). Near-zero estimates would blow up the derived strength.
const SIGMA_FLOOR: f32 = 0.1 / 255.0;

/// Per-stream noise state. Smooths per-frame estimates with an EMA so
/// single busy frames cannot spike the strength, and floors the result
/// so near-clean content keeps a usable Welsch normalisation.
#[derive(Debug, Default)]
pub(super) struct NoiseEstimator {
    ema: Option<Vec<f32>>,
}

impl NoiseEstimator {
    /// Folds a new set of per-channel sigma samples into the running
    /// EMA and returns the smoothed result. The first call initialises
    /// the state directly from `sigmas` (no prior estimate to blend
    /// with). Every element is floored at [`SIGMA_FLOOR`] regardless of
    /// how it was produced.
    pub(super) fn update(&mut self, sigmas: &[f32]) -> &[f32] {
        match &mut self.ema {
            None => {
                self.ema = Some(sigmas.iter().map(|&s| s.max(SIGMA_FLOOR)).collect());
            },
            Some(ema) => {
                for (e, &s) in ema.iter_mut().zip(sigmas.iter()) {
                    *e = (EMA_ALPHA * s + (1.0 - EMA_ALPHA) * *e).max(SIGMA_FLOOR);
                }
            },
        }
        self.ema.as_deref().unwrap()
    }

    /// Clears the running estimate. The next [`Self::update`] call
    /// re-initialises from its sample instead of blending with stale
    /// state.
    pub(super) fn reset(&mut self) {
        self.ema = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_sample_initializes_state() {
        let mut est = NoiseEstimator::default();
        let out = est.update(&[0.05, 0.02]);
        assert_eq!(out, &[0.05, 0.02]);
    }

    #[test]
    fn update_converges_toward_changed_level() {
        let mut est = NoiseEstimator::default();
        est.update(&[0.02]);

        let mut last = 0.0;
        for _ in 0..200 {
            last = est.update(&[0.10])[0];
        }

        assert!(
            (last - 0.10).abs() < 1e-4,
            "expected convergence close to 0.10, got {last}"
        );
    }

    #[test]
    fn update_floors_near_zero_samples() {
        let mut est = NoiseEstimator::default();
        let out = est.update(&[0.0]);
        assert_eq!(out[0], SIGMA_FLOOR);

        let out = est.update(&[0.0]);
        assert_eq!(out[0], SIGMA_FLOOR);
    }

    #[test]
    fn reset_clears_state() {
        let mut est = NoiseEstimator::default();
        est.update(&[0.10]);
        est.reset();

        // After reset, the next update should re-initialise directly
        // from the sample rather than blending with the stale 0.10.
        let out = est.update(&[0.02]);
        assert_eq!(out, &[0.02]);
    }

    #[test]
    fn sigma_from_abs_sum_zero_for_zero_response() {
        // No mask response anywhere in the frame must estimate zero
        // noise regardless of frame size.
        assert_eq!(sigma_from_abs_sum(0.0, 64, 64), 0.0);
    }

    #[test]
    fn partials_len_matches_cube_grid() {
        assert_eq!(partials_len(32, 8), 4); // exactly one BLOCK_X x BLOCK_Y cube
        assert_eq!(partials_len(33, 9), 16); // spills into a 2x2 cube grid
        assert_eq!(partials_len(1920, 1080), 60 * 135 * 4);
    }
}

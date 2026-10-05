mod bilateral;

use cubecl::prelude::*;
use cubecl::server::Handle;

pub use self::bilateral::bilateral_radius;
pub(crate) use self::bilateral::inv_two_sigma_sq;

/// How each frame's reference image is produced.
///
/// Comparing patches on a noisy image compares the noise too, so a cleaner reference gives better
/// weights. The averaged pixels always come from the original input.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum PrefilterMode {
    /// No reference image, so patches are compared on the noisy input at no extra cost.
    #[default]
    None,
    /// A quick bilateral blur run on the GPU at push time.
    Bilateral { sigma_s: f32, sigma_r: f32 },
    /// A spatial NLM pilot pass at push time, kept as the reference image.
    NlmSpatial {
        /// How much of the main pass strength the pilot pass uses.
        strength_scale: f32,
    },
}

/// The default pilot strength, as a multiplier on the main pass strength.
///
/// A calibration sweep across noise levels puts the XPSNR plateau for `PrefilterMode::NlmSpatial` at
/// this value.
pub const DEFAULT_PILOT_STRENGTH_SCALE: f32 = 0.4;

impl PrefilterMode {
    /// Whether the denoiser needs to allocate the reference ring buffer.
    pub(crate) fn needs_reference_buf(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Whether this mode builds its reference on the GPU during a push.
    pub(crate) fn is_gpu_internal(self) -> bool {
        matches!(self, Self::Bilateral { .. } | Self::NlmSpatial { .. })
    }
}

/// Parses a `--prefilter`-style string into a [PrefilterMode].
///
/// `none` or an empty string gives [PrefilterMode::None]. `nlm` or `nlm:<strength_scale>` gives
/// [PrefilterMode::NlmSpatial], with bare `nlm` at [DEFAULT_PILOT_STRENGTH_SCALE].
/// `bilateral:<sigma_s>,<sigma_r>` gives [PrefilterMode::Bilateral].
pub fn parse_prefilter(value: &str) -> Result<PrefilterMode, anyhow::Error> {
    if value == "none" || value.is_empty() {
        return Ok(PrefilterMode::None);
    }

    if value == "nlm" {
        return Ok(PrefilterMode::NlmSpatial {
            strength_scale: DEFAULT_PILOT_STRENGTH_SCALE,
        });
    }

    if let Some(rest) = value.strip_prefix("nlm:") {
        let strength_scale: f32 = rest
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("--prefilter nlm expects a number: nlm:<strength_scale>"))?;

        return Ok(PrefilterMode::NlmSpatial { strength_scale });
    }

    if let Some(rest) = value.strip_prefix("bilateral:") {
        let parts: Vec<&str> = rest.split(',').collect();

        if parts.len() != 2 {
            anyhow::bail!("--prefilter bilateral expects two values: bilateral:<sigma_s>,<sigma_r>");
        }

        let sigma_s: f32 = parts[0].trim().parse()?;
        let sigma_r: f32 = parts[1].trim().parse()?;

        return Ok(PrefilterMode::Bilateral { sigma_s, sigma_r });
    }

    anyhow::bail!(
        "unknown prefilter '{value}', expected `none`, `nlm[:<strength_scale>]`, or `bilateral:<sigma_s>,<sigma_r>`"
    )
}

/// The inputs one prefilter dispatch needs.
///
/// It lives for one push, which makes the borrows on the denoiser's buffers sound.
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

/// Runs the GPU prefilter for the frame uploaded last.
pub(crate) fn run_prefilter<R: Runtime>(
    mode: PrefilterMode,
    client: &ComputeClient<R>,
    ctx: &PrefilterCtx<'_>,
) -> Result<(), anyhow::Error> {
    match mode {
        PrefilterMode::None => Ok(()),
        // The pilot needs the accumulators and `h2_inv_norm`, which this context lacks, so the
        // denoiser dispatches it directly.
        PrefilterMode::NlmSpatial { .. } => Ok(()),
        PrefilterMode::Bilateral { sigma_s, sigma_r } => {
            bilateral::run_bilateral::<R>(client, ctx, sigma_s, sigma_r)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_requires_no_reference_buffer() {
        assert!(!PrefilterMode::None.needs_reference_buf());
        assert!(!PrefilterMode::None.is_gpu_internal());
    }

    #[test]
    fn bilateral_is_gpu_internal() {
        let mode = PrefilterMode::Bilateral {
            sigma_s: 3.0,
            sigma_r: 0.02,
        };

        assert!(mode.needs_reference_buf());
        assert!(mode.is_gpu_internal());
    }

    #[test]
    fn nlm_spatial_is_gpu_internal() {
        let mode = PrefilterMode::NlmSpatial { strength_scale: 1.0 };

        assert!(mode.needs_reference_buf());
        assert!(mode.is_gpu_internal());
    }

    #[test]
    fn bilateral_radius_truncates_at_two_sigma() {
        let tiny = bilateral_radius(0.1);
        let unit = bilateral_radius(1.0);
        let wide = bilateral_radius(3.0);
        let fractional = bilateral_radius(3.5);

        assert_eq!(tiny, 1);
        assert_eq!(unit, 2);
        assert_eq!(wide, 6);
        assert_eq!(fractional, 7);
    }

    #[test]
    fn none_and_empty_prefilters_parse() {
        let none = parse_prefilter("none").unwrap();
        let empty = parse_prefilter("").unwrap();

        assert!(matches!(none, PrefilterMode::None));
        assert!(matches!(empty, PrefilterMode::None));
    }

    #[test]
    fn bilateral_with_values_parses() {
        let mode = parse_prefilter("bilateral:3.0,0.02").unwrap();
        assert!(matches!(
            mode,
            PrefilterMode::Bilateral {
                sigma_s,
                sigma_r,
            } if (sigma_s - 3.0).abs() < f32::EPSILON && (sigma_r - 0.02).abs() < f32::EPSILON
        ));
    }

    #[test]
    fn bare_nlm_uses_default_strength_scale() {
        let mode = parse_prefilter("nlm").unwrap();
        assert!(matches!(
            mode,
            PrefilterMode::NlmSpatial { strength_scale }
                if (strength_scale - DEFAULT_PILOT_STRENGTH_SCALE).abs() < f32::EPSILON
        ));
    }

    #[test]
    fn nlm_with_explicit_strength_scale_parses() {
        let mode = parse_prefilter("nlm:0.8").unwrap();
        assert!(matches!(
            mode,
            PrefilterMode::NlmSpatial { strength_scale } if (strength_scale - 0.8).abs() < f32::EPSILON
        ));
    }

    #[test]
    fn malformed_nlm_scale_is_rejected() {
        let error = parse_prefilter("nlm:x").expect_err("expected parse failure");
        assert!(error.to_string().contains("nlm"));
    }

    #[test]
    fn unknown_prefilter_is_rejected() {
        let result = parse_prefilter("garbage");
        assert!(result.is_err());
    }
}

mod bilateral;

pub use bilateral::bilateral_radius;
pub(crate) use bilateral::inv_two_sigma_sq;
use cubecl::prelude::*;
use cubecl::server::Handle;

/// How the reference image for each frame is produced.
///
/// NLM compares patches to decide how much two pixels look alike. Doing
/// that on a noisy image means comparing noise as well as content, so a
/// cleaner reference image can give better weights.
///
/// The pixels being averaged always come from the original input. Only
/// the weights change.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum PrefilterMode {
    /// No reference image, so patches are compared on the noisy input.
    /// This costs nothing extra.
    #[default]
    None,
    /// The caller supplies the reference frame through
    /// [`super::NlmDenoiser::push_frame_with_reference`].
    External,
    /// A quick bilateral blur run on the GPU at push time.
    Bilateral { sigma_s: f32, sigma_r: f32 },
    /// A spatial NLM pilot pass.
    ///
    /// Each frame is denoised with the windowed spatial kernel at push
    /// time and the result is kept as the reference image.
    NlmSpatial {
        /// How much of the main pass strength the pilot pass uses.
        strength_scale: f32,
    },
}

/// The measured default strength for the pilot pass, as a multiplier on
/// the main pass strength.
///
/// A calibration sweep across noise levels put the XPSNR plateau for
/// `PrefilterMode::NlmSpatial` at this value.
pub const DEFAULT_PILOT_STRENGTH_SCALE: f32 = 0.4;

impl PrefilterMode {
    /// Whether the denoiser needs to allocate the reference ring buffer.
    pub(crate) fn needs_reference_buf(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Whether this mode builds its reference on the GPU during
    /// `push_frame`, rather than taking one from the caller.
    pub(crate) fn is_gpu_internal(self) -> bool {
        matches!(self, Self::Bilateral { .. } | Self::NlmSpatial { .. })
    }
}

/// Parses a `--prefilter`-style string into a [`PrefilterMode`].
///
/// `"none"` or an empty string means [`PrefilterMode::None`].
///
/// `"nlm"` or `"nlm:<strength_scale>"` builds
/// [`PrefilterMode::NlmSpatial`]. `strength_scale` multiplies the main
/// pass strength for the pilot pass. Bare `"nlm"` uses
/// [`DEFAULT_PILOT_STRENGTH_SCALE`].
///
/// `"bilateral:<sigma_s>,<sigma_r>"` builds [`PrefilterMode::Bilateral`].
///
/// This never produces [`PrefilterMode::External`], since that mode has
/// no string form: it requires the caller to supply a reference frame
/// through [`super::NlmDenoiser::push_frame_with_reference`].
pub fn parse_prefilter(s: &str) -> Result<PrefilterMode, anyhow::Error> {
    if s == "none" || s.is_empty() {
        return Ok(PrefilterMode::None);
    }

    if s == "nlm" {
        return Ok(PrefilterMode::NlmSpatial {
            strength_scale: DEFAULT_PILOT_STRENGTH_SCALE,
        });
    }

    if let Some(rest) = s.strip_prefix("nlm:") {
        let strength_scale: f32 = rest
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("--prefilter nlm expects a number: nlm:<strength_scale>"))?;

        return Ok(PrefilterMode::NlmSpatial { strength_scale });
    }

    if let Some(rest) = s.strip_prefix("bilateral:") {
        let parts: Vec<&str> = rest.split(',').collect();

        if parts.len() != 2 {
            anyhow::bail!("--prefilter bilateral expects two values: bilateral:<sigma_s>,<sigma_r>");
        }

        let sigma_s: f32 = parts[0].trim().parse()?;
        let sigma_r: f32 = parts[1].trim().parse()?;

        return Ok(PrefilterMode::Bilateral { sigma_s, sigma_r });
    }

    anyhow::bail!(
        "unknown prefilter '{s}', expected `none`, `nlm[:<strength_scale>]`, or `bilateral:<sigma_s>,<sigma_r>`"
    )
}

/// The inputs one prefilter dispatch needs.
///
/// This lives only for the length of a single `push_frame`, which is
/// what makes the borrows on the denoiser's buffers sound.
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

/// Runs the GPU prefilter for the frame that was uploaded last.
///
/// `None` and `External` do nothing here.
pub(crate) fn run_prefilter<R: Runtime>(
    mode: PrefilterMode,
    client: &ComputeClient<R>,
    ctx: &PrefilterCtx<'_>,
) -> Result<(), anyhow::Error> {
    match mode {
        PrefilterMode::None | PrefilterMode::External => Ok(()),
        // The pilot needs the full accumulator context, meaning accum,
        // weight_sum, max_weight, and h2_inv_norm, which `PrefilterCtx`
        // does not carry. `NlmDenoiser::run_nlm_spatial_pilot`
        // dispatches it directly instead of coming through here.
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
    fn external_needs_buffer_but_not_gpu() {
        assert!(PrefilterMode::External.needs_reference_buf());
        assert!(!PrefilterMode::External.is_gpu_internal());
    }

    #[test]
    fn bilateral_is_gpu_internal() {
        let m = PrefilterMode::Bilateral {
            sigma_s: 3.0,
            sigma_r: 0.02,
        };

        assert!(m.needs_reference_buf());
        assert!(m.is_gpu_internal());
    }

    #[test]
    fn nlm_spatial_is_gpu_internal() {
        let m = PrefilterMode::NlmSpatial { strength_scale: 1.0 };

        assert!(m.needs_reference_buf());
        assert!(m.is_gpu_internal());
    }

    #[test]
    fn bilateral_radius_truncates_at_two_sigma() {
        assert_eq!(bilateral_radius(0.1), 1);
        assert_eq!(bilateral_radius(1.0), 2);
        assert_eq!(bilateral_radius(3.0), 6);
        assert_eq!(bilateral_radius(3.5), 7);
    }

    #[test]
    fn none_and_empty_prefilters_parse() {
        assert!(matches!(parse_prefilter("none").unwrap(), PrefilterMode::None));
        assert!(matches!(parse_prefilter("").unwrap(), PrefilterMode::None));
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
        let err = parse_prefilter("nlm:x").expect_err("expected parse failure");
        assert!(err.to_string().contains("nlm"));
    }

    #[test]
    fn unknown_prefilter_is_rejected() {
        assert!(parse_prefilter("garbage").is_err());
    }

    #[test]
    fn external_cannot_be_parsed_from_a_string() {
        for s in [
            "external",
            "none",
            "nlm",
            "nlm:0.5",
            "bilateral:3.0,0.02",
            "garbage",
        ] {
            if let Ok(mode) = parse_prefilter(s) {
                assert!(
                    !matches!(mode, PrefilterMode::External),
                    "parse_prefilter must never produce External, got it from '{s}'"
                );
            }
        }
    }
}

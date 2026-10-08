use super::psy::PsyParams;
use crate::collab::MAX_TEMPORAL_RADIUS;
use crate::nlmeans::{ChannelMode, HqParams, MotionCompensationMode, MotionEstimation, NlmParams};

/// The largest [Nl4dParams::kaiser_beta] worth accepting.
///
/// By 8 the end tap is under a fiftieth of the centre, so a patch's edge pixels contribute almost
/// nothing and the step-4 grid covers each pixel with a handful of centres rather than a blend.
/// Past that the window acts as a mask, and the smallest weights fall under what the fixed-point
/// accumulators resolve.
pub const MAX_KAISER_BETA: f32 = 8.0;

/// The most motion blocks that may cover a reference patch on one axis.
///
/// The fused kernel searches every covering block and unrolls its duplicate-rectangle arrays over
/// the square of this bound, so the bound caps the shader's register footprint. The default
/// geometry, `blksize = 16` at `overlap = 8`, uses 2. The step is `blksize - overlap`, so 4 admits
/// an overlap of up to three quarters of the block size.
pub const MAX_COVERING_BLOCKS: u32 = 4;

/// Tuning for the nl4d denoiser.
///
/// `nlm` configures the front end that builds the frame ring, motion field and confidence scores.
/// Its `temporal_radius` is overwritten from this struct's own at construction.
#[derive(Debug, Clone)]
pub struct Nl4dParams {
    /// Configuration for the front end.
    ///
    /// `hq` must be `Some` with `temporal_confidence` on, and motion compensation must be active,
    /// because the front end only builds a ring view when both are on.
    pub nlm: NlmParams,
    /// How many frames on each side of the centre frame the temporal search reaches, in `1..=8`.
    pub temporal_radius: u32,
    /// Half-width of the refine window around each neighbour frame's motion-predicted position, in
    /// `1..=4`.
    pub refine: u32,
    /// Half-width of the spatial candidate window in the centre frame, in `1..=16`.
    pub spatial_radius: u32,
    /// Hard-threshold multiplier on the propagated coefficient sigma.
    ///
    /// Higher removes more noise and more fine detail. Defaults to 4.158, though luma and chroma
    /// want separately tuned values. See [nl4d_default_lambda_ht](crate::nl4d_default_lambda_ht).
    pub lambda_ht: f32,
    /// The confidence floor below which a whole neighbour block is skipped, in `0.0..1.0`.
    ///
    /// A volume left short of frames by the skip makes its group filter from the centre frame
    /// alone.
    pub c_min: f32,
    /// The `beta` of the Kaiser window each filtered patch is tapered with as it is aggregated, in
    /// `0.0..=8.0`.
    ///
    /// A pixel is covered by many patches, each with its own threshold decision. Tapering each
    /// patch toward its edges blends those decisions instead of letting each reach its boundary at
    /// full strength. Defaults to 2.0, BM3D's own value, and `0.0` is uniform aggregation.
    pub kaiser_beta: f32,
    /// The penalty on a block's vector deviating from its neighbourhood median, in the field
    /// regularisation pass.
    ///
    /// The median is taken over the four adjacent blocks' vectors and zero. The penalty is this
    /// times the distance from the median in pixels, scaled so `1.0` weighs one pixel of deviation
    /// like a 5/255 per-pixel mismatch. Defaults to `1.0`, calibrated on the `mc_accuracy` bench to
    /// sit inside the plateau where larger values add little accuracy. `0.0` skips the pass.
    pub field_lambda: f32,
    /// Scales the luma threshold by how noisy each brightness level is in the current frame.
    ///
    /// On by default.
    pub noise_map: bool,
    /// Perceptual tuning by area, see [PsyParams](crate::PsyParams).
    ///
    /// Only takes effect with `noise_map` on. `None`, the default, filters every area alike.
    pub psy: Option<PsyParams>,
    /// Judges each transform coefficient together with its frequency neighbours instead of alone.
    ///
    /// Faint texture spreads over several neighbouring frequencies, so it survives where each
    /// coefficient alone would fall under the threshold. On by default.
    pub pooled_threshold: bool,
    /// Whether the denoiser measures the source's film grain for an AV1 grain table.
    ///
    /// Off by default. Only a denoiser that filters luma measures. It never changes the output.
    /// Measured chunks stay on the GPU until drained, so callers drain after each flush.
    pub grain_export: bool,
}

impl Default for Nl4dParams {
    fn default() -> Self {
        Self {
            nlm: NlmParams {
                temporal_radius: 2,
                channels: ChannelMode::Yuv,
                motion_compensation: MotionCompensationMode::Mvtools {
                    blksize: 16,
                    overlap: 8,
                    search_radius: 4,
                    pyramid_levels: 2,
                    estimation: MotionEstimation::Auto,
                },
                hq: Some(HqParams::default()),
                ..NlmParams::default()
            },
            temporal_radius: 2,
            refine: 2,
            spatial_radius: 9,
            lambda_ht: 4.158,
            c_min: 0.05,
            kaiser_beta: 2.0,
            field_lambda: 1.0,
            noise_map: true,
            psy: None,
            pooled_threshold: true,
            grain_export: false,
        }
    }
}

impl Nl4dParams {
    /// Rejects a configuration that would fail to launch or fail the front end's checks on submit.
    pub fn validate(&self) -> Result<(), String> {
        let Some(hq) = self.nlm.hq else {
            return Err(
                "nlm.hq must be Some, the front end's noise estimate and confidence weighting \
                 are what submit_machinery builds the ring view from"
                    .to_string(),
            );
        };

        if !self.nlm.motion_compensation.is_active() {
            return Err(
                "nlm.motion_compensation must be active, the temporal grouping kernel reads \
                 the motion field submit_machinery builds from it"
                    .to_string(),
            );
        }

        if !hq.temporal_confidence {
            return Err(
                "nlm.hq.temporal_confidence must be true, submit_machinery returns an error \
                 unless both motion compensation and the confidence buffer are active"
                    .to_string(),
            );
        }

        // An overlap at or past `blksize` is left to `nlm.validate()`, which names the real fault
        // instead of a covering-block count computed from a saturated step.
        if let MotionCompensationMode::Mvtools { blksize, overlap, .. } = self.nlm.motion_compensation
            && overlap < blksize
        {
            let step = blksize - overlap;
            let covers = blksize.div_ceil(step);
            if covers > MAX_COVERING_BLOCKS {
                return Err(format!(
                    "nlm.motion_compensation blksize={blksize} at overlap={overlap} gives a step \
                     of {step}, so {covers} blocks cover a patch on each axis, past the \
                     {MAX_COVERING_BLOCKS} the temporal grouping kernel unrolls its search over. \
                     Raise the step by lowering the overlap."
                ));
            }
        }

        if !(1..=MAX_TEMPORAL_RADIUS).contains(&self.temporal_radius) {
            return Err(format!(
                "temporal_radius={} must be in 1..={}",
                self.temporal_radius, MAX_TEMPORAL_RADIUS,
            ));
        }

        if !(1..=4).contains(&self.refine) {
            return Err(format!("refine={} must be in 1..=4", self.refine));
        }

        if !(1..=16).contains(&self.spatial_radius) {
            return Err(format!(
                "spatial_radius={} must be in 1..=16",
                self.spatial_radius
            ));
        }

        if !(self.lambda_ht.is_finite() && self.lambda_ht > 0.0) {
            return Err(format!(
                "lambda_ht must be finite and greater than 0, got {}",
                self.lambda_ht
            ));
        }

        if !(self.c_min.is_finite() && self.c_min >= 0.0 && self.c_min < 1.0) {
            return Err(format!("c_min must be finite and in [0, 1), got {}", self.c_min));
        }

        if !(self.kaiser_beta.is_finite() && (0.0..=MAX_KAISER_BETA).contains(&self.kaiser_beta)) {
            return Err(format!(
                "kaiser_beta must be finite and in 0..={MAX_KAISER_BETA}, got {}",
                self.kaiser_beta
            ));
        }

        if !(self.field_lambda.is_finite() && self.field_lambda >= 0.0) {
            return Err(format!(
                "field_lambda must be finite and at least 0, got {}",
                self.field_lambda
            ));
        }

        if let Some(psy) = &self.psy {
            psy.validate()?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_default() {
        let params = Nl4dParams::default();
        assert!(params.validate().is_ok());
    }

    #[test]
    fn the_noise_map_is_on_by_default() {
        let params = Nl4dParams::default();
        assert!(params.noise_map);
    }

    #[test]
    fn psy_is_off_by_default() {
        let params = Nl4dParams::default();
        assert_eq!(params.psy, None);
        assert!(params.pooled_threshold);
    }

    #[test]
    fn validate_checks_the_psy_params() {
        let psy = PsyParams {
            line_ring: 65,
            ..PsyParams::default()
        };
        let params = Nl4dParams {
            psy: Some(psy),
            ..Nl4dParams::default()
        };
        let error = params.validate().expect_err("line_ring out of range");
        assert!(error.contains("psy_line_ring"), "got {error}");
    }

    #[test]
    fn validate_accepts_block_geometries_up_to_the_covering_bound() {
        for (blksize, overlap, covers) in [(16u32, 8u32, 2u32), (16, 12, 4), (32, 24, 4), (8, 4, 2)] {
            let motion_compensation = MotionCompensationMode::Mvtools {
                blksize,
                overlap,
                search_radius: 4,
                pyramid_levels: 2,
                estimation: MotionEstimation::Auto,
            };
            let nlm = NlmParams {
                motion_compensation,
                ..Nl4dParams::default().nlm
            };
            let params = Nl4dParams {
                nlm,
                ..Nl4dParams::default()
            };
            assert!(
                params.validate().is_ok(),
                "blksize={blksize} overlap={overlap} covers {covers} blocks and should be accepted"
            );
        }
    }

    #[test]
    fn validate_rejects_a_block_geometry_past_the_covering_bound() {
        for (blksize, overlap) in [(16u32, 13u32), (16, 14), (32, 31), (32, 25)] {
            let motion_compensation = MotionCompensationMode::Mvtools {
                blksize,
                overlap,
                search_radius: 4,
                pyramid_levels: 2,
                estimation: MotionEstimation::Auto,
            };
            let nlm = NlmParams {
                motion_compensation,
                ..Nl4dParams::default().nlm
            };
            let params = Nl4dParams {
                nlm,
                ..Nl4dParams::default()
            };
            let error = params
                .validate()
                .expect_err("a step this small should be rejected");
            let blksize_label = format!("blksize={blksize}");
            let overlap_label = format!("overlap={overlap}");
            assert!(
                error.contains(&blksize_label) && error.contains(&overlap_label),
                "error should name the offending blksize and overlap, got {error}"
            );
        }
    }

    #[test]
    fn overlap_equal_to_blksize_reports_the_overlap_constraint_not_covering_blocks() {
        let motion_compensation = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 16,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Auto,
        };
        let nlm = NlmParams {
            motion_compensation,
            ..Nl4dParams::default().nlm
        };
        let params = Nl4dParams {
            nlm,
            ..Nl4dParams::default()
        };
        assert!(
            params.validate().is_ok(),
            "the covering-block check must not fire on a geometry nlm.validate() rejects on its \
             own terms"
        );

        let error = params
            .nlm
            .validate()
            .expect_err("overlap == blksize must be rejected")
            .to_string();
        assert!(
            error.contains("overlap") && error.contains("blksize"),
            "error should name the overlap constraint, got {error}"
        );
        assert!(
            !error.contains("cover a patch"),
            "error should not be the covering-block message, got {error}"
        );
    }

    #[test]
    fn validate_rejects_missing_hq() {
        let nlm = NlmParams {
            hq: None,
            ..Nl4dParams::default().nlm
        };
        let params = Nl4dParams {
            nlm,
            ..Nl4dParams::default()
        };
        let error = params.validate().expect_err("expected rejection");
        assert!(error.contains("nlm.hq"), "error should name nlm.hq, got {error}");
    }

    #[test]
    fn validate_rejects_inactive_motion_compensation() {
        let nlm = NlmParams {
            motion_compensation: MotionCompensationMode::None,
            ..Nl4dParams::default().nlm
        };
        let params = Nl4dParams {
            nlm,
            ..Nl4dParams::default()
        };
        let error = params.validate().expect_err("expected rejection");
        assert!(
            error.contains("motion_compensation"),
            "error should name nlm.motion_compensation, got {error}"
        );
    }

    #[test]
    fn validate_rejects_missing_temporal_confidence() {
        let hq = HqParams {
            temporal_confidence: false,
            ..HqParams::default()
        };
        let nlm = NlmParams {
            hq: Some(hq),
            ..Nl4dParams::default().nlm
        };
        let params = Nl4dParams {
            nlm,
            ..Nl4dParams::default()
        };
        let error = params.validate().expect_err("expected rejection");
        assert!(
            error.contains("temporal_confidence"),
            "error should name nlm.hq.temporal_confidence, got {error}"
        );
    }

    #[test]
    fn validate_rejects_temporal_radius_out_of_range() {
        for bad in [0u32, 9] {
            let params = Nl4dParams {
                temporal_radius: bad,
                ..Nl4dParams::default()
            };
            assert!(
                params.validate().is_err(),
                "temporal_radius={bad} should be rejected"
            );
        }
    }

    #[test]
    fn validate_rejects_refine_out_of_range() {
        for bad in [0u32, 5] {
            let params = Nl4dParams {
                refine: bad,
                ..Nl4dParams::default()
            };
            assert!(params.validate().is_err(), "refine={bad} should be rejected");
        }
    }

    #[test]
    fn validate_rejects_spatial_radius_out_of_range() {
        for bad in [0u32, 17] {
            let params = Nl4dParams {
                spatial_radius: bad,
                ..Nl4dParams::default()
            };
            assert!(
                params.validate().is_err(),
                "spatial_radius={bad} should be rejected"
            );
        }
    }

    #[test]
    fn validate_rejects_non_positive_lambda_ht() {
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            let params = Nl4dParams {
                lambda_ht: bad,
                ..Nl4dParams::default()
            };
            assert!(params.validate().is_err(), "lambda_ht={bad} should be rejected");
        }
    }

    #[test]
    fn validate_rejects_c_min_out_of_range() {
        for bad in [-0.1f32, 1.0, f32::NAN] {
            let params = Nl4dParams {
                c_min: bad,
                ..Nl4dParams::default()
            };
            assert!(params.validate().is_err(), "c_min={bad} should be rejected");
        }
    }

    #[test]
    fn validate_accepts_zero_and_positive_field_lambda() {
        for lambda in [0.0, 0.5, 4.0] {
            let params = Nl4dParams {
                field_lambda: lambda,
                ..Nl4dParams::default()
            };
            assert!(
                params.validate().is_ok(),
                "field_lambda={lambda} should be accepted"
            );
        }
    }

    #[test]
    fn validate_rejects_negative_or_non_finite_field_lambda() {
        for lambda in [-0.1, f32::NAN, f32::INFINITY] {
            let params = Nl4dParams {
                field_lambda: lambda,
                ..Nl4dParams::default()
            };
            let error = params
                .validate()
                .expect_err("field_lambda={lambda} should be rejected");
            assert!(
                error.contains("field_lambda"),
                "error should name field_lambda, got {error}"
            );
        }
    }
}

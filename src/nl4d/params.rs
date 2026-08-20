use crate::nlmeans::{ChannelMode, HqParams, MotionCompensationMode, MotionEstimation, NlmParams};

/// Tuning for [`super::Nl4dDenoiser`].
///
/// `nlm` supplies the front end that builds the frame ring, the motion
/// field, and the confidence scores the temporal grouping reads. Its own
/// `temporal_radius` is overwritten at construction time from this
/// struct's own `temporal_radius`, so it does not need to be set by the
/// caller.
#[derive(Debug, Clone)]
pub struct Nl4dParams {
    /// Machinery configuration for the front end. `hq` must be `Some`
    /// with `temporal_confidence` on, and `motion_compensation` must be
    /// active, because [`crate::nlmeans::NlmDenoiser::submit_machinery`]
    /// only builds a ring view when both are on, and this denoiser is
    /// built entirely on top of that call.
    pub nlm: NlmParams,
    /// How many frames on each side of the centre frame the temporal
    /// search reaches into. In `1..=8`.
    pub temporal_radius: u32,
    /// Half-width of the refine window searched around each neighbour
    /// frame's motion-predicted position. In `1..=4`.
    pub refine: u32,
    /// Half-width of the spatial candidate window searched in the
    /// centre frame. In `1..=16`.
    pub spatial_radius: u32,
    /// Hard-threshold multiplier on the propagated coefficient sigma.
    /// Calibrated to 5.3 by default (the Luma/Yuv value), see
    /// [`crate::Nl4dOptions`]'s struct-level docs and
    /// `nl4d_default_lambda_ht` for the ladder this came from. Callers
    /// building `Nl4dParams` directly, rather than through
    /// [`crate::Nl4dOptions`], get no per-plane resolution and should
    /// set this themselves for chroma.
    pub lambda_ht: f32,
    /// The confidence floor below which a whole neighbour block is
    /// skipped rather than scored, in `[0, 1)`. Only affects how much
    /// compute a submit spends, never which candidates are admitted once
    /// they are scored.
    pub c_min: f32,
    /// Whether a temporal member's mismatch variance reaches the
    /// hard-threshold shrinkage at all.
    ///
    /// Defaults to `true`, the design this denoiser exists to test.
    /// `false` runs the hard-threshold stage exactly as it ran before
    /// this mechanism existed, every member's variance the plain
    /// channel sigma with no per-member addition. This is the only way
    /// to turn the mechanism off, and it exists for an ablation that
    /// needs it off at otherwise identical settings.
    pub confidence_variance: bool,
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
            lambda_ht: 5.3,
            c_min: 0.05,
            confidence_variance: true,
        }
    }
}

impl Nl4dParams {
    /// Rejects a configuration that would fail to launch, or that would
    /// hit [`crate::nlmeans::NlmDenoiser::submit_machinery`]'s own
    /// preconditions only once a real submit ran.
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

        if !(1..=crate::collab::MAX_TEMPORAL_RADIUS).contains(&self.temporal_radius) {
            return Err(format!(
                "temporal_radius={} must be in 1..={}",
                self.temporal_radius,
                crate::collab::MAX_TEMPORAL_RADIUS,
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

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_default() {
        assert!(Nl4dParams::default().validate().is_ok());
    }

    #[test]
    fn validate_rejects_missing_hq() {
        let params = Nl4dParams {
            nlm: NlmParams {
                hq: None,
                ..Nl4dParams::default().nlm
            },
            ..Nl4dParams::default()
        };
        let err = params.validate().expect_err("expected rejection");
        assert!(err.contains("nlm.hq"), "error should name nlm.hq, got {err}");
    }

    #[test]
    fn validate_rejects_inactive_motion_compensation() {
        let params = Nl4dParams {
            nlm: NlmParams {
                motion_compensation: MotionCompensationMode::None,
                ..Nl4dParams::default().nlm
            },
            ..Nl4dParams::default()
        };
        let err = params.validate().expect_err("expected rejection");
        assert!(
            err.contains("motion_compensation"),
            "error should name nlm.motion_compensation, got {err}"
        );
    }

    /// The latent precondition `submit_machinery`/`flush_step_machinery`
    /// enforce at submit time. Both motion compensation and the
    /// confidence buffer have to be active, or those calls return an
    /// error. `validate` has to catch a configuration that would hit
    /// that error before construction ever gets that far.
    #[test]
    fn validate_rejects_missing_temporal_confidence() {
        let params = Nl4dParams {
            nlm: NlmParams {
                hq: Some(HqParams {
                    temporal_confidence: false,
                    ..HqParams::default()
                }),
                ..Nl4dParams::default().nlm
            },
            ..Nl4dParams::default()
        };
        let err = params.validate().expect_err("expected rejection");
        assert!(
            err.contains("temporal_confidence"),
            "error should name nlm.hq.temporal_confidence, got {err}"
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
}

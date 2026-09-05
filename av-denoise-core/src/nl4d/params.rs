use crate::nlmeans::{ChannelMode, HqParams, MotionCompensationMode, MotionEstimation, NlmParams};

/// The largest [`Nl4dParams::mismatch_scale`] worth accepting.
///
/// The mismatch variance is capped at
/// [`crate::collab::kernels::fused::MEMBER_SIGMA2_CAP`] times the channel
/// variance, and the worst-matched blocks reach that cap at a scale of
/// roughly `319 * sigma`. Even a source noisy enough to measure `sigma =
/// 0.05` saturates below 16, so nothing above this can move a pixel and
/// accepting it would only promise a range that is not there.
pub const MAX_MISMATCH_SCALE: f32 = 16.0;

/// The largest [`Nl4dParams::kaiser_beta`] worth accepting.
///
/// A Kaiser window's taps fall off faster the larger `beta` is. By 8 the
/// end tap is under a fiftieth of the centre, so a patch's edge pixels
/// contribute almost nothing and the step-4 grid is left covering each
/// pixel with a handful of centres rather than a blend. Past that the
/// window stops being a taper and starts being a mask, and the smallest
/// weights fall under what the fixed-point accumulators resolve.
pub const MAX_KAISER_BETA: f32 = 8.0;

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
    /// Higher shrinks more coefficients, so it removes more noise and
    /// more fine detail.
    ///
    /// Defaults to 5.3, the luma value. Chroma wants a different one, and
    /// callers building `Nl4dParams` directly get no per-plane
    /// resolution. See [`crate::nl4d_default_lambda_ht`].
    pub lambda_ht: f32,
    /// The confidence floor below which a whole neighbour block is
    /// skipped rather than scored, in `[0, 1)`. Only affects how much
    /// compute a submit spends, never which candidates are admitted once
    /// they are scored.
    pub c_min: f32,
    /// A multiplier on the mismatch variance a poorly matched temporal
    /// member carries into the hard threshold.
    ///
    /// The variance grows with the square of this, so `2.0` is a
    /// four-fold increase. `1.0`, the default, is the shipped
    /// calibration. `0.0` matches `confidence_variance: false`.
    ///
    /// The mechanism saturates. A member's extra variance is capped at
    /// [`crate::collab::kernels::fused::MEMBER_SIGMA2_CAP`] times the
    /// channel variance, which the worst-matched blocks reach somewhere
    /// between 3 and 13 depending on how noisy the source is, so raising
    /// this past that point stops changing anything.
    pub mismatch_scale: f32,
    /// The `beta` of the Kaiser window each filtered patch is tapered
    /// with as it is aggregated, in `0..=8`.
    ///
    /// A pixel is covered by many patches, each of which made its own
    /// threshold decision. Tapering a patch toward its edges blends
    /// those decisions rather than letting each reach its boundary at
    /// full strength. Larger tapers harder. BM3D uses 2.0.
    ///
    /// Defaults to 2.0, BM3D's own value. `0.0` is exactly uniform
    /// aggregation, which is what this did before the window existed.
    /// See [`crate::collab::kernels::aggregate::kaiser_window`].
    pub kaiser_beta: f32,
    /// Whether a temporal member's mismatch variance reaches the
    /// hard-threshold shrinkage.
    ///
    /// `true`, the default, treats a poorly matched member as a noisier
    /// observation, so the threshold trusts it less. `false` gives every
    /// member the plain channel sigma instead, which is what an ablation
    /// needs to isolate the effect of this mechanism.
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
            mismatch_scale: 1.0,
            kaiser_beta: 2.0,
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

        if !(self.mismatch_scale.is_finite() && (0.0..=MAX_MISMATCH_SCALE).contains(&self.mismatch_scale)) {
            return Err(format!(
                "mismatch_scale must be finite and in [0, {MAX_MISMATCH_SCALE}], got {}",
                self.mismatch_scale
            ));
        }

        if !(self.kaiser_beta.is_finite() && (0.0..=MAX_KAISER_BETA).contains(&self.kaiser_beta)) {
            return Err(format!(
                "kaiser_beta must be finite and in 0..={MAX_KAISER_BETA}, got {}",
                self.kaiser_beta
            ));
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
    fn validate_accepts_the_whole_mismatch_scale_range() {
        for scale in [0.0, 1.0, 8.0, MAX_MISMATCH_SCALE] {
            let params = Nl4dParams {
                mismatch_scale: scale,
                ..Nl4dParams::default()
            };
            assert!(
                params.validate().is_ok(),
                "mismatch_scale={scale} should be accepted"
            );
        }
    }

    /// Past the saturation point the dial cannot move a pixel, so a
    /// caller asking for more is asking for something that does not
    /// exist and should hear so rather than see no effect.
    #[test]
    fn validate_rejects_a_mismatch_scale_past_saturation_or_below_zero() {
        for scale in [-1.0, MAX_MISMATCH_SCALE + 0.1, f32::NAN, f32::INFINITY] {
            let params = Nl4dParams {
                mismatch_scale: scale,
                ..Nl4dParams::default()
            };
            let err = params
                .validate()
                .expect_err("mismatch_scale={scale} should be rejected");
            assert!(
                err.contains("mismatch_scale"),
                "error should name mismatch_scale, got {err}"
            );
        }
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

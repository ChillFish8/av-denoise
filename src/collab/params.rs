use crate::nlmeans::ChannelMode;

/// Tuning for the collaborative filter core.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollabParams {
    /// Which channels the filter runs on.
    pub channels: ChannelMode,
    /// Largest stack size a group can reach. Rounded down to a power of
    /// two per group at filter time. In `1..=MAX_K`.
    pub k_max: u32,
    /// Half-width of the spatial candidate window around a reference
    /// patch.
    pub spatial_radius: u32,
    /// Admission threshold multiplier on the per-patch noise floor. A
    /// candidate patch joins the group when its distance to the
    /// reference falls under this multiple of the floor.
    pub tau_match: f32,
    /// Hard-threshold multiplier on the propagated coefficient sigma,
    /// applied during collaborative filtering.
    pub lambda_ht: f32,
    /// The spatial correlation of the residual noise this filter is
    /// shrinking, in `[0, 1)`.
    ///
    /// A white-noise input assigns every DCT coefficient the same
    /// variance. Real input to this filter is rarely white, an
    /// upstream denoising pass (nl3d's own non-local means front end,
    /// for instance) typically leaves a residual whose covariance falls
    /// off with distance roughly as `rho^d`, concentrating noise power
    /// at low frequencies. `rho` lets the filter model that instead of
    /// assuming white noise, by scaling each DCT coefficient's variance
    /// through [`crate::collab::kernels::transforms::dct_noise_profile`]
    /// before shrinkage sees it.
    ///
    /// Defaults to `0.0`, which reproduces flat white-noise behaviour
    /// exactly. `nl3d` leaves this at the default. It once overrode it
    /// per frame from a measured table keyed by the front end's search
    /// radius (`nl3d::rho::rho_window`), but that shaping measured worse
    /// on real footage than the plain white-noise assumption, so nl3d no
    /// longer reaches for it; see `Nl3dDenoiser::new`'s doc comment for
    /// the numbers. The table and this field stay available for a
    /// caller who wants to opt into shaping directly.
    pub rho: f32,
    /// Runs only the hard-threshold stage. Stage 1's aggregate goes to
    /// the caller's output and the Wiener stage is skipped.
    ///
    /// Diagnostic switch for the brick-wash investigation. Defaults to
    /// false, which runs both stages.
    pub ht_only: bool,
    /// Replaces the sigma the grouping stage builds its admission
    /// floor from, in normalised `[0, 1]` units.
    ///
    /// Diagnostic override for the brick-wash investigation. The value
    /// is applied to every active channel, so it is only meaningful on
    /// single-channel runs. `None` keeps the per-frame sigmas the
    /// caller passes to `run_two_stage`.
    pub admission_sigma_override: Option<f32>,
    /// Replaces the sigma the shrinkage stages threshold and weight
    /// with, in normalised `[0, 1]` units.
    ///
    /// Diagnostic override for the brick-wash investigation. The value
    /// is applied to every active channel, so it is only meaningful on
    /// single-channel runs. `None` keeps the per-frame sigmas the
    /// caller passes to `run_two_stage`.
    pub shrinkage_sigma_override: Option<f32>,
    /// Selects the transform basis the hard-threshold stage's shrinkage
    /// runs in.
    ///
    /// Diagnostic switch for comparing the DCT basis this filter has
    /// always used against an orthonormal Haar-8 basis. Defaults to
    /// false, which keeps the DCT basis. See
    /// `crate::collab::kernels::transforms::fill_haar8_basis`.
    pub ht_wavelet: bool,
}

impl Default for CollabParams {
    fn default() -> Self {
        Self {
            channels: ChannelMode::Luma,
            k_max: 8,
            spatial_radius: 9,
            tau_match: 3.0,
            lambda_ht: 2.7,
            rho: 0.0,
            ht_only: false,
            admission_sigma_override: None,
            shrinkage_sigma_override: None,
            ht_wavelet: false,
        }
    }
}

impl CollabParams {
    /// Rejects parameter combinations that would fail to launch, by
    /// running the kernels past their shared-memory limits, or that
    /// would produce meaningless output.
    pub fn validate(&self) -> Result<(), anyhow::Error> {
        if self.k_max == 0 || self.k_max > super::MAX_K {
            anyhow::bail!(
                "k_max={} must be in 1..={}, because the group stack is sized off this bound",
                self.k_max,
                super::MAX_K,
            );
        }

        if !self.k_max.is_power_of_two() {
            anyhow::bail!(
                "k_max={} must be a power of two, because the collaborative transform halves \
                 the stack at each stage",
                self.k_max,
            );
        }

        if self.spatial_radius == 0 || self.spatial_radius > 16 {
            anyhow::bail!(
                "spatial_radius={} must be in 1..=16, because the candidate window's shared-\
                 memory tile grows with the square of the radius",
                self.spatial_radius,
            );
        }

        if !(self.tau_match.is_finite() && self.tau_match > 0.0) {
            anyhow::bail!(
                "tau_match must be finite and greater than 0, got {}. A tau_match of 0 admits \
                 no candidate but the reference patch itself",
                self.tau_match,
            );
        }

        if !(self.lambda_ht.is_finite() && self.lambda_ht > 0.0) {
            anyhow::bail!(
                "lambda_ht must be finite and greater than 0, got {}. A lambda_ht of 0 hard-\
                 thresholds every coefficient to zero",
                self.lambda_ht,
            );
        }

        if !(self.rho.is_finite() && self.rho >= 0.0 && self.rho < 1.0) {
            anyhow::bail!(
                "rho must be finite and in [0, 1), got {}. rho is a correlation, and 1.0 is a \
                 degenerate value this filter's noise model was never validated against",
                self.rho,
            );
        }

        for (name, sigma) in [
            ("admission_sigma_override", self.admission_sigma_override),
            ("shrinkage_sigma_override", self.shrinkage_sigma_override),
        ] {
            if let Some(s) = sigma {
                if !(s.is_finite() && s > 0.0) {
                    anyhow::bail!(
                        "{name} must be finite and greater than 0 when set, got {s}. A zero \
                         or negative sigma has no meaning as a noise level"
                    );
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_default() {
        assert!(CollabParams::default().validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_k_max() {
        let params = CollabParams {
            k_max: 0,
            ..CollabParams::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_k_max_above_the_maximum() {
        let params = CollabParams {
            k_max: super::super::MAX_K + 1,
            ..CollabParams::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_power_of_two_k_max() {
        let params = CollabParams {
            k_max: 3,
            ..CollabParams::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_accepts_every_power_of_two_up_to_max_k() {
        let mut k = 1;
        while k <= super::super::MAX_K {
            let params = CollabParams {
                k_max: k,
                ..CollabParams::default()
            };
            assert!(params.validate().is_ok(), "k_max={k} should be accepted");
            k *= 2;
        }
    }

    #[test]
    fn validate_rejects_zero_spatial_radius() {
        let params = CollabParams {
            spatial_radius: 0,
            ..CollabParams::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_rejects_spatial_radius_above_sixteen() {
        let params = CollabParams {
            spatial_radius: 17,
            ..CollabParams::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn validate_accepts_spatial_radius_at_the_boundary() {
        let params = CollabParams {
            spatial_radius: 16,
            ..CollabParams::default()
        };
        assert!(params.validate().is_ok());
    }

    #[test]
    fn validate_rejects_non_positive_tau_match() {
        let zero = CollabParams {
            tau_match: 0.0,
            ..CollabParams::default()
        };
        assert!(zero.validate().is_err());

        let negative = CollabParams {
            tau_match: -1.0,
            ..CollabParams::default()
        };
        assert!(negative.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_finite_tau_match() {
        let nan = CollabParams {
            tau_match: f32::NAN,
            ..CollabParams::default()
        };
        assert!(nan.validate().is_err());

        let inf = CollabParams {
            tau_match: f32::INFINITY,
            ..CollabParams::default()
        };
        assert!(inf.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_positive_lambda_ht() {
        let zero = CollabParams {
            lambda_ht: 0.0,
            ..CollabParams::default()
        };
        assert!(zero.validate().is_err());

        let negative = CollabParams {
            lambda_ht: -1.0,
            ..CollabParams::default()
        };
        assert!(negative.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_finite_lambda_ht() {
        let nan = CollabParams {
            lambda_ht: f32::NAN,
            ..CollabParams::default()
        };
        assert!(nan.validate().is_err());

        let inf = CollabParams {
            lambda_ht: f32::INFINITY,
            ..CollabParams::default()
        };
        assert!(inf.validate().is_err());
    }

    #[test]
    fn validate_accepts_rho_zero_and_rejects_out_of_range() {
        assert!(
            CollabParams {
                rho: 0.0,
                ..CollabParams::default()
            }
            .validate()
            .is_ok()
        );
        assert!(
            CollabParams {
                rho: 0.86,
                ..CollabParams::default()
            }
            .validate()
            .is_ok()
        );

        let negative = CollabParams {
            rho: -0.1,
            ..CollabParams::default()
        };
        assert!(negative.validate().is_err());

        let one = CollabParams {
            rho: 1.0,
            ..CollabParams::default()
        };
        assert!(one.validate().is_err());

        let nan = CollabParams {
            rho: f32::NAN,
            ..CollabParams::default()
        };
        assert!(nan.validate().is_err());
    }
}

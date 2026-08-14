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
}

impl Default for CollabParams {
    fn default() -> Self {
        Self {
            channels: ChannelMode::Luma,
            k_max: 8,
            spatial_radius: 9,
            tau_match: 3.0,
            lambda_ht: 2.7,
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
}

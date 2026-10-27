/// The motion search's tuning, for a denoiser that always tracks motion.
///
/// [MotionCompensationMode::Mvtools] carries the same five values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionSearch {
    /// The side length of each search block, in pixels at the finest pyramid level.
    pub blksize: u32,
    /// How many pixels neighbouring blocks overlap.
    ///
    /// It must be strictly below `blksize` so the step between blocks stays positive.
    pub overlap: u32,
    /// The search radius in pixels at the finest pyramid level.
    ///
    /// The coarse pass uses the same radius on a half-size image, so its reach is twice as far.
    pub search_radius: u32,
    /// How many levels the pyramid has, up to [MAX_PYRAMID_LEVELS].
    ///
    /// `1` searches at full resolution only, and `2` adds a half-size coarse pass that seeds it.
    pub pyramid_levels: u32,
    /// How motion toward each temporal neighbour is estimated.
    pub estimation: MotionEstimation,
}

impl Default for MotionSearch {
    fn default() -> Self {
        Self {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Auto,
        }
    }
}

impl From<MotionSearch> for MotionCompensationMode {
    fn from(search: MotionSearch) -> Self {
        Self::Mvtools {
            blksize: search.blksize,
            overlap: search.overlap,
            search_radius: search.search_radius,
            pyramid_levels: search.pyramid_levels,
            estimation: search.estimation,
        }
    }
}

/// How motion compensation is set up for a denoise pass.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum MotionCompensationMode {
    /// Motion compensation is off, and no extra buffers are allocated.
    #[default]
    None,
    /// An MVTools-style estimator tracks each block, and the neighbouring frames are shifted toward
    /// the centre frame at denoise time.
    Mvtools {
        /// The side length of each search block, in pixels at the finest pyramid level.
        blksize: u32,
        /// How many pixels neighbouring blocks overlap.
        ///
        /// It must be strictly below `blksize` so the step between blocks stays positive.
        overlap: u32,
        /// The search radius in pixels at the finest pyramid level.
        ///
        /// The coarse pass uses the same radius on a half-size image, so its reach is twice as far.
        search_radius: u32,
        /// How many levels the pyramid has, up to [MAX_PYRAMID_LEVELS].
        ///
        /// `1` searches at full resolution only, and `2` adds a half-size coarse pass that seeds it.
        pyramid_levels: u32,
        /// How motion toward each temporal neighbour is estimated.
        ///
        /// `Auto`, the default, picks a strategy from the temporal radius.
        estimation: MotionEstimation,
    },
}

/// How motion toward a temporal neighbour is estimated.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum MotionEstimation {
    /// Picks `Direct` or `Chained` from the temporal radius, as [MotionEstimation::resolve] does.
    #[default]
    Auto,
    /// Matches every neighbour against the centre frame directly at the configured search radius.
    ///
    /// Its cost grows with the temporal radius, because each neighbour repeats the coarse and fine
    /// search.
    Direct,
    /// Measures motion between adjacent frames once per push and joins it into a seed per
    /// neighbour.
    ///
    /// A small seeded search then cleans up whatever drift is left.
    Chained {
        /// The seeded refinement's search radius, in pixels at the finest pyramid level.
        ///
        /// It can be small because the joined seed already carries most of the movement.
        refine_radius: u32,
    },
}

/// The default refinement radius for [MotionEstimation::Chained].
pub const DEFAULT_REFINE_RADIUS: u32 = 2;

/// The temporal radius at which [MotionEstimation::Auto] switches from `Direct` to `Chained`.
///
/// Below it `Direct` tracks slightly better, because the motion still fits its search window. At or
/// above it `Chained` stays inside its window and runs faster, because its reach grows with the
/// radius.
pub const CHAINED_RADIUS_THRESHOLD: u32 = 3;

impl MotionEstimation {
    /// A `Chained` estimation with [DEFAULT_REFINE_RADIUS].
    pub fn chained_default() -> Self {
        Self::Chained {
            refine_radius: DEFAULT_REFINE_RADIUS,
        }
    }

    /// Resolves `Auto` at [CHAINED_RADIUS_THRESHOLD], passing `Direct` and `Chained` through.
    pub fn resolve(self, temporal_radius: u32) -> Self {
        match self {
            Self::Auto if temporal_radius >= CHAINED_RADIUS_THRESHOLD => Self::chained_default(),
            Self::Auto => Self::Direct,
            other => other,
        }
    }

    /// Rejects a refinement radius the seeded fine kernel cannot honour.
    pub(crate) fn validate(&self) -> Result<(), anyhow::Error> {
        let Self::Chained { refine_radius } = *self else {
            return Ok(());
        };

        if refine_radius == 0 || refine_radius > MAX_SEARCH_RADIUS {
            anyhow::bail!(
                "motion-estimation refine_radius={refine_radius} must be in 1..={MAX_SEARCH_RADIUS}"
            );
        }

        Ok(())
    }
}

/// The default block size, matching MVTools.
pub const DEFAULT_BLKSIZE: u32 = 16;

/// The default number of pixels neighbouring blocks overlap.
pub const DEFAULT_OVERLAP: u32 = 8;

/// The default search radius at the finest level.
///
/// With a two-level pyramid it reaches motion of roughly 12 pixels at full resolution.
pub const DEFAULT_SEARCH_RADIUS: u32 = 4;

/// The default number of pyramid levels.
///
/// One half-size coarse pass handles most heavy-motion anime while keeping kernel launches down.
pub const DEFAULT_PYRAMID_LEVELS: u32 = 2;

/// The hard ceiling on `pyramid_levels`.
///
/// Each extra level halves the resolution again and adds a kernel launch per neighbour. Three is
/// already more than 1080p content needs.
pub const MAX_PYRAMID_LEVELS: u32 = 3;

/// The hard ceiling on `search_radius`.
///
/// The analyse kernel scores a `(2 * radius + 1)^2` window per block, so the cost grows with the
/// square of the radius.
pub const MAX_SEARCH_RADIUS: u32 = 8;

/// The hard ceiling on `blksize`.
///
/// Above it the per-block shared-memory tile grows too large on RDNA-class GPUs.
pub const MAX_BLKSIZE: u32 = 32;

impl MotionCompensationMode {
    /// An `Mvtools` mode from the library defaults.
    ///
    /// It pins `estimation` to `Direct`, so it never switches to `Chained` at larger radii.
    pub fn mvtools_default() -> Self {
        Self::Mvtools {
            blksize: DEFAULT_BLKSIZE,
            overlap: DEFAULT_OVERLAP,
            search_radius: DEFAULT_SEARCH_RADIUS,
            pyramid_levels: DEFAULT_PYRAMID_LEVELS,
            estimation: MotionEstimation::Direct,
        }
    }

    pub(crate) fn is_active(self) -> bool {
        !matches!(self, Self::None)
    }

    /// The concrete estimation strategy this mode resolves to at `temporal_radius`.
    ///
    /// It is `None` when the mode is not `Mvtools`, and it is never `Auto`.
    pub(crate) fn resolved_estimation(&self, temporal_radius: u32) -> Option<MotionEstimation> {
        match *self {
            Self::Mvtools { estimation, .. } => {
                let resolved = estimation.resolve(temporal_radius);
                Some(resolved)
            },
            Self::None => None,
        }
    }

    /// Rejects parameter combinations the kernels cannot honour.
    pub fn validate(&self) -> Result<(), anyhow::Error> {
        let Self::Mvtools {
            blksize,
            overlap,
            search_radius,
            pyramid_levels,
            estimation,
        } = *self
        else {
            return Ok(());
        };

        if blksize < 4 {
            anyhow::bail!(
                "motion-compensation blksize={blksize} is too small, the minimum is 4 pixels per side"
            );
        }

        if blksize > MAX_BLKSIZE {
            anyhow::bail!(
                "motion-compensation blksize={blksize} exceeds the supported maximum of {MAX_BLKSIZE}"
            );
        }

        if blksize % 2 != 0 {
            anyhow::bail!(
                "motion-compensation blksize={blksize} must be even so the /2 coarse level is well-defined"
            );
        }

        if overlap >= blksize {
            anyhow::bail!(
                "motion-compensation overlap={overlap} must be strictly less than blksize, \
                 which is {blksize}, so the step between blocks stays positive"
            );
        }

        if search_radius == 0 || search_radius > MAX_SEARCH_RADIUS {
            anyhow::bail!(
                "motion-compensation search_radius={search_radius} must be in 1..={MAX_SEARCH_RADIUS}"
            );
        }

        if pyramid_levels == 0 || pyramid_levels > MAX_PYRAMID_LEVELS {
            anyhow::bail!(
                "motion-compensation pyramid_levels={pyramid_levels} must be in 1..={MAX_PYRAMID_LEVELS}"
            );
        }

        estimation.validate()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_is_inactive() {
        let mode = MotionCompensationMode::None;
        assert!(!mode.is_active());
        mode.validate().unwrap();
    }

    #[test]
    fn mvtools_default_is_active() {
        let mode = MotionCompensationMode::mvtools_default();
        assert!(mode.is_active());
        mode.validate().unwrap();
    }

    #[test]
    fn validate_rejects_tiny_blksize() {
        let mode = MotionCompensationMode::Mvtools {
            blksize: 2,
            overlap: 0,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        };
        assert!(mode.validate().is_err());
    }

    #[test]
    fn validate_rejects_odd_blksize() {
        let mode = MotionCompensationMode::Mvtools {
            blksize: 9,
            overlap: 0,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        };
        assert!(mode.validate().is_err());
    }

    #[test]
    fn validate_rejects_overlap_equal_to_blksize() {
        let mode = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 16,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        };
        // An overlap equal to blksize would leave a step of 0.
        assert!(mode.validate().is_err());
    }

    #[test]
    fn validate_accepts_half_overlap() {
        let mode = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        };
        mode.validate().unwrap();
    }

    #[test]
    fn validate_rejects_zero_search_radius() {
        let mode = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 4,
            search_radius: 0,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        };
        assert!(mode.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_pyramid_levels() {
        let mode = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 4,
            search_radius: 4,
            pyramid_levels: 0,
            estimation: MotionEstimation::Direct,
        };
        assert!(mode.validate().is_err());
    }

    #[test]
    fn chained_default_is_valid() {
        let estimation = MotionEstimation::chained_default();
        let mode = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            estimation,
        };
        mode.validate().unwrap();

        let expected_estimation = MotionEstimation::Chained {
            refine_radius: DEFAULT_REFINE_RADIUS,
        };
        let expected = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: expected_estimation,
        };
        assert_eq!(mode, expected);
    }

    #[test]
    fn validate_rejects_zero_refine_radius() {
        let estimation = MotionEstimation::Chained { refine_radius: 0 };
        let mode = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            estimation,
        };
        assert!(mode.validate().is_err());
    }

    #[test]
    fn validate_rejects_refine_radius_above_max() {
        let estimation = MotionEstimation::Chained {
            refine_radius: MAX_SEARCH_RADIUS + 1,
        };
        let mode = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            estimation,
        };
        assert!(mode.validate().is_err());
    }

    #[test]
    fn validate_accepts_refine_radius_at_max() {
        let estimation = MotionEstimation::Chained {
            refine_radius: MAX_SEARCH_RADIUS,
        };
        let mode = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            estimation,
        };
        mode.validate().unwrap();
    }

    #[test]
    fn motion_estimation_default_is_auto() {
        let estimation = MotionEstimation::default();
        assert_eq!(estimation, MotionEstimation::Auto);
    }

    #[test]
    fn resolve_auto_below_threshold_gives_direct() {
        assert_eq!(MotionEstimation::Auto.resolve(1), MotionEstimation::Direct);
        assert_eq!(MotionEstimation::Auto.resolve(2), MotionEstimation::Direct);
    }

    #[test]
    fn resolve_auto_at_and_above_threshold_gives_chained_default() {
        let chained = MotionEstimation::chained_default();
        let at_threshold = MotionEstimation::Auto.resolve(CHAINED_RADIUS_THRESHOLD);
        let above_threshold = MotionEstimation::Auto.resolve(8);

        assert_eq!(at_threshold, chained);
        assert_eq!(above_threshold, chained);
    }

    #[test]
    fn resolve_leaves_explicit_direct_unchanged_at_every_radius() {
        for radius in 1..=8u32 {
            assert_eq!(MotionEstimation::Direct.resolve(radius), MotionEstimation::Direct);
        }
    }

    #[test]
    fn resolve_leaves_explicit_chained_unchanged_at_every_radius() {
        let chained = MotionEstimation::Chained { refine_radius: 5 };
        for radius in 1..=8u32 {
            assert_eq!(chained.resolve(radius), chained);
        }
    }

    #[test]
    fn validate_accepts_auto() {
        let mode = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Auto,
        };
        mode.validate().unwrap();
    }

    #[test]
    fn resolved_estimation_is_none_when_mode_is_none() {
        assert_eq!(MotionCompensationMode::None.resolved_estimation(4), None);
    }

    #[test]
    fn resolved_estimation_resolves_auto_from_the_mode() {
        let mode = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Auto,
        };
        let chained = MotionEstimation::chained_default();
        assert_eq!(mode.resolved_estimation(1), Some(MotionEstimation::Direct));
        assert_eq!(mode.resolved_estimation(4), Some(chained));
    }
}

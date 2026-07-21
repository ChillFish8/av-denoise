mod analyse;
mod chain;
mod compensate;
mod confidence;
mod pyramid;

pub(crate) use analyse::{confidence_byte_offset, mv_field_byte_offset, run_analyse, run_seeded_refine};
pub(crate) use chain::{neighbour_idx_for_k, pair_byte_offset, run_pair_analyse, zero_pair_slot};
pub(crate) use compensate::run_compensate;
pub(crate) use confidence::{run_confidence_for_neighbour, sad_noise_floor, thsad};
use cubecl::prelude::*;
use cubecl::server::Handle;
pub(crate) use pyramid::{pyramid_pixels_per_frame, run_pyramid_build};

/// How motion compensation is configured for a denoise pass.
///
/// `None` disables motion compensation entirely (zero-cost; no extra
/// buffers are allocated). `Mvtools` enables an MVTools-inspired
/// per-block estimator and warps neighbours toward the centre at
/// denoise time.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum MotionCompensationMode {
    #[default]
    None,
    Mvtools {
        /// Side length of each motion-estimation block in pixels at
        /// the finest pyramid level.
        blksize: u32,
        /// Overlap between neighbouring blocks in pixels. Must be
        /// strictly less than `blksize` so the step (`blksize - overlap`)
        /// stays positive. Values > 0 reserve room for raised-cosine
        /// blending in the compensate step (v1 uses a winner-block rule).
        overlap: u32,
        /// Pixel search radius at the *finest* pyramid level. The
        /// coarse pass uses the same radius on the `/2` image so its
        /// effective reach is doubled.
        search_radius: u32,
        /// Number of pyramid levels. `1` disables the hierarchical
        /// coarse pass; `2` adds a `/2` coarse pass that seeds the
        /// fine pass. Bounded by [`MAX_PYRAMID_LEVELS`].
        pyramid_levels: u32,
        /// How temporal MVs are estimated. `Auto` (the default) picks
        /// the strategy from the temporal radius. Callers normally
        /// leave this at the default. Explicit `Direct`/`Chained` are
        /// mainly useful for pinning a variant in tests and benches.
        estimation: MotionEstimation,
    },
}

/// Strategy for estimating a temporal neighbour's motion vector.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum MotionEstimation {
    /// Resolve to `Direct` or `Chained` from the temporal radius at
    /// denoiser construction. See [`MotionEstimation::resolve`] for the
    /// exact rule and its empirical basis.
    #[default]
    Auto,
    /// Match every neighbour directly against the centre frame at the
    /// configured search radius. Cost scales with the temporal radius,
    /// since each neighbour repeats the full coarse+fine search.
    Direct,
    /// Estimate motion between adjacent frames only, once per pushed
    /// frame, then compose the per-step vectors into a seed for each
    /// neighbour and correct residual drift with a small seeded
    /// refinement search.
    Chained {
        /// Search radius for the seeded refinement pass, in pixels at
        /// the finest pyramid level. Small because the composed seed
        /// already carries most of the true displacement.
        refine_radius: u32,
    },
}

/// Default refinement radius for [`MotionEstimation::Chained`].
pub const DEFAULT_REFINE_RADIUS: u32 = 2;

/// Temporal radius at or above which [`MotionEstimation::Auto`]
/// resolves to `Chained` instead of `Direct`. Below this, `Direct`
/// tracks slightly better since the true motion still fits inside its
/// own search window. At or above it, `Chained` stays in-window and is
/// faster, since its reach scales with the radius instead of being
/// capped by a fixed search window.
pub const CHAINED_RADIUS_THRESHOLD: u32 = 3;

impl MotionEstimation {
    /// Convenience constructor for `Chained` with the library default
    /// refinement radius.
    pub fn chained_default() -> Self {
        Self::Chained {
            refine_radius: DEFAULT_REFINE_RADIUS,
        }
    }

    /// Resolve `Auto` against the temporal radius, returning a concrete
    /// `Direct` or `Chained` estimation. Never returns `Auto`. `Direct`
    /// and `Chained` pass through unchanged, regardless of
    /// `temporal_radius`. See [`CHAINED_RADIUS_THRESHOLD`] for the
    /// threshold this applies.
    pub fn resolve(self, temporal_radius: u32) -> Self {
        match self {
            Self::Auto if temporal_radius >= CHAINED_RADIUS_THRESHOLD => Self::chained_default(),
            Self::Auto => Self::Direct,
            other => other,
        }
    }

    /// Reject a refinement radius the seeded fine kernel can't honour.
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

/// Default block size used when callers don't override it. Matches the
/// MVTools default and lines up well with NLM's typical patch sizes.
pub const DEFAULT_BLKSIZE: u32 = 16;
/// Default block overlap (= `blksize / 2`).
pub const DEFAULT_OVERLAP: u32 = 8;
/// Default finest-level search radius. With a 2-level pyramid this
/// reaches motion up to roughly ±12 pixels at the finest scale.
pub const DEFAULT_SEARCH_RADIUS: u32 = 4;
/// Default number of pyramid levels. `2` gives a single `/2` coarse
/// pass, enough to handle most heavy-motion anime while keeping the
/// kernel count manageable.
pub const DEFAULT_PYRAMID_LEVELS: u32 = 2;

/// Hard ceiling on `pyramid_levels`. Each extra level halves the
/// resolution and adds an analyse-kernel launch per neighbour; 3 is
/// already overkill for 1080p content.
pub const MAX_PYRAMID_LEVELS: u32 = 3;
/// Hard ceiling on `search_radius`. The analyse kernel SAD-sweeps a
/// `(2·r + 1)²` window per block, so the cost is quadratic.
pub const MAX_SEARCH_RADIUS: u32 = 8;
/// Hard ceiling on `blksize`. Above this the per-block SMEM tile is
/// uncomfortably large on RDNA-class GPUs.
pub const MAX_BLKSIZE: u32 = 32;

impl MotionCompensationMode {
    /// Convenience constructor for `Mvtools` with library defaults.
    pub fn mvtools_default() -> Self {
        Self::Mvtools {
            blksize: DEFAULT_BLKSIZE,
            overlap: DEFAULT_OVERLAP,
            search_radius: DEFAULT_SEARCH_RADIUS,
            pyramid_levels: DEFAULT_PYRAMID_LEVELS,
            estimation: MotionEstimation::Direct,
        }
    }

    /// Whether motion compensation is active at all.
    pub(crate) fn is_active(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Resolved estimation strategy for this mode at `temporal_radius`.
    /// `None` when this mode isn't `Mvtools`. Never `Auto`, see
    /// [`MotionEstimation::resolve`]. The single source every
    /// estimation-dependent decision site (pair-ring allocation,
    /// push-time pair-analyse gating, the submit-path dispatch branch)
    /// goes through.
    pub(crate) fn resolved_estimation(&self, temporal_radius: u32) -> Option<MotionEstimation> {
        match *self {
            Self::Mvtools { estimation, .. } => Some(estimation.resolve(temporal_radius)),
            Self::None => None,
        }
    }

    /// Reject parameter combinations that the kernels can't honour.
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
            anyhow::bail!("motion-compensation blksize={blksize} is too small; minimum is 4 pixels per side");
        }
        if blksize > MAX_BLKSIZE {
            anyhow::bail!(
                "motion-compensation blksize={blksize} exceeds the supported maximum ({MAX_BLKSIZE})"
            );
        }
        if blksize % 2 != 0 {
            anyhow::bail!(
                "motion-compensation blksize={blksize} must be even so the /2 coarse level is well-defined"
            );
        }
        if overlap >= blksize {
            anyhow::bail!(
                "motion-compensation overlap={overlap} must be strictly less than blksize ({blksize}) so step > 0"
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

/// Per-denoiser MC state, owned by `NlmDenoiser` when MC is active.
///
/// Lives next to (not inside) the optional buffer handles so the hot
/// dispatch path can fish out comptime-relevant scalars without
/// pattern-matching the enum every call.
/// Per-denoiser MC state cached at construction time so the hot
/// dispatch path doesn't re-pattern-match the enum on every call.
/// Holds only the fields actually read by analyse / compensate
/// dispatchers; the full configuration lives on
/// [`MotionCompensationMode`].
#[derive(Debug, Clone)]
pub(crate) struct MotionCtx {
    pub blksize: u32,
    pub step: u32,
    pub search_radius: u32,
    pub pyramid_levels: u32,
    pub blocks_x: u32,
    pub blocks_y: u32,
}

impl MotionCtx {
    pub fn new(mode: MotionCompensationMode, width: u32, height: u32) -> Option<Self> {
        let MotionCompensationMode::Mvtools {
            blksize,
            overlap,
            search_radius,
            pyramid_levels,
            estimation: _,
        } = mode
        else {
            return None;
        };

        let step = blksize - overlap;
        let blocks_x = width.div_ceil(step).max(1);
        let blocks_y = height.div_ceil(step).max(1);

        Some(Self {
            blksize,
            step,
            search_radius,
            pyramid_levels,
            blocks_x,
            blocks_y,
        })
    }

    /// MV-field slot count per neighbour. One i16x2 per block.
    pub fn mv_slots_per_neighbour(&self) -> usize {
        (self.blocks_x * self.blocks_y) as usize
    }

    /// Padded per-neighbour confidence-buffer stride in bytes. One
    /// `f32` per block, rounded up to the GPU storage-buffer offset
    /// alignment (32 bytes). Small block counts (as few as one block
    /// for tiny frames) would otherwise produce a per-neighbour stride
    /// under 32 bytes, and `wgpu` rejects a bind-group offset that
    /// isn't a multiple of its `min_storage_buffer_offset_alignment`.
    /// The MV field avoids this by using two `i32` components per
    /// block (8 bytes), which happens to land on a 32-byte multiple
    /// for the block counts exercised so far. The confidence buffer's
    /// single `f32` per block doesn't have that margin, so it pads
    /// explicitly instead of relying on coincidence.
    pub(crate) fn confidence_bytes_per_neighbour(&self) -> u64 {
        let blocks = (self.blocks_x as u64) * (self.blocks_y as u64);
        (blocks * size_of::<f32>() as u64).next_multiple_of(32)
    }

    /// i32 elements per pair-ring direction sub-array, one `(dx, dy)`
    /// per block, matching the MV field's own per-neighbour layout.
    pub(crate) fn pair_direction_len(&self) -> u32 {
        self.blocks_x * self.blocks_y * 2
    }

    /// i32 elements per pair-ring slot, both directions back to back
    /// (see [`pair_ring_slot_count`] for the slot-count invariant).
    pub(crate) fn pair_slot_len(&self) -> u32 {
        2 * self.pair_direction_len()
    }

    /// Block geometry for the no-MC confidence pass. Uses the
    /// library's default block size and overlap, a single pyramid
    /// level (no coarse pass), and zero search radius (a static
    /// per-block SAD, no motion search). Used when confidence
    /// weighting is active but no `Mvtools` mode was configured to
    /// derive geometry from.
    pub(crate) fn confidence_only(width: u32, height: u32) -> Self {
        Self::new(
            MotionCompensationMode::Mvtools {
                blksize: DEFAULT_BLKSIZE,
                overlap: DEFAULT_OVERLAP,
                search_radius: 0,
                pyramid_levels: 1,
                estimation: MotionEstimation::Direct,
            },
            width,
            height,
        )
        .expect("Mvtools variant always yields Some")
    }
}

/// Pair-ring slot count for a temporal radius, `2 * radius`.
///
/// The pair ring stores one adjacent-frame motion field per gap
/// between consecutive frames in the temporal window. A window of
/// `2 * radius + 1` frames has exactly `2 * radius` such gaps, and a
/// gap's pair field is only ever read by composition while both its
/// frames remain in some window, a span of exactly `2 * radius`
/// consecutive frame pushes. Sizing the ring at `2 * radius` slots
/// means a slot's next reuse lands exactly when its previous contents
/// stop being needed, never before (see
/// `NlmDenoiser::pair_slot` for the derivation this relies on).
pub(crate) fn pair_ring_slot_count(temporal_radius: u32) -> u32 {
    2 * temporal_radius
}

/// Build the per-frame pyramid for the slot just uploaded by
/// `push_frame`. Cheap no-op if `pyramid_levels == 1`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_pyramid_for_slot<R: Runtime>(
    client: &ComputeClient<R>,
    mc: &MotionCtx,
    width: u32,
    height: u32,
    frame_count: u32,
    slot: u32,
    full_res: &Handle,
    pyramid: &Handle,
    stored_ch: u32,
) -> Result<(), anyhow::Error> {
    if mc.pyramid_levels <= 1 {
        return Ok(());
    }
    run_pyramid_build::<R>(
        client,
        mc,
        width,
        height,
        frame_count,
        slot,
        full_res,
        pyramid,
        stored_ch,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_is_inactive() {
        let m = MotionCompensationMode::None;
        assert!(!m.is_active());
        m.validate().unwrap();
    }

    #[test]
    fn mvtools_default_is_active() {
        let m = MotionCompensationMode::mvtools_default();
        assert!(m.is_active());
        m.validate().unwrap();
    }

    #[test]
    fn validate_rejects_tiny_blksize() {
        let m = MotionCompensationMode::Mvtools {
            blksize: 2,
            overlap: 0,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn validate_rejects_odd_blksize() {
        let m = MotionCompensationMode::Mvtools {
            blksize: 9,
            overlap: 0,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn validate_rejects_overlap_equal_to_blksize() {
        let m = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 16,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        };
        // overlap == blksize would give step=0.
        assert!(m.validate().is_err());
    }

    #[test]
    fn validate_accepts_half_overlap() {
        let m = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        };
        m.validate().unwrap();
    }

    #[test]
    fn validate_rejects_zero_search_radius() {
        let m = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 4,
            search_radius: 0,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_pyramid_levels() {
        let m = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 4,
            search_radius: 4,
            pyramid_levels: 0,
            estimation: MotionEstimation::Direct,
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn chained_default_is_valid() {
        let m = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::chained_default(),
        };
        m.validate().unwrap();
        assert_eq!(
            m,
            MotionCompensationMode::Mvtools {
                blksize: 16,
                overlap: 8,
                search_radius: 4,
                pyramid_levels: 2,
                estimation: MotionEstimation::Chained {
                    refine_radius: DEFAULT_REFINE_RADIUS
                },
            }
        );
    }

    #[test]
    fn validate_rejects_zero_refine_radius() {
        let m = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Chained { refine_radius: 0 },
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn validate_rejects_refine_radius_above_max() {
        let m = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Chained {
                refine_radius: MAX_SEARCH_RADIUS + 1,
            },
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn validate_accepts_refine_radius_at_max() {
        let m = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Chained {
                refine_radius: MAX_SEARCH_RADIUS,
            },
        };
        m.validate().unwrap();
    }

    #[test]
    fn motion_estimation_default_is_auto() {
        assert_eq!(MotionEstimation::default(), MotionEstimation::Auto);
    }

    #[test]
    fn resolve_auto_below_threshold_gives_direct() {
        assert_eq!(MotionEstimation::Auto.resolve(1), MotionEstimation::Direct);
        assert_eq!(MotionEstimation::Auto.resolve(2), MotionEstimation::Direct);
    }

    #[test]
    fn resolve_auto_at_and_above_threshold_gives_chained_default() {
        assert_eq!(
            MotionEstimation::Auto.resolve(CHAINED_RADIUS_THRESHOLD),
            MotionEstimation::chained_default()
        );
        assert_eq!(
            MotionEstimation::Auto.resolve(8),
            MotionEstimation::chained_default()
        );
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
        let m = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Auto,
        };
        m.validate().unwrap();
    }

    #[test]
    fn resolved_estimation_is_none_when_mode_is_none() {
        assert_eq!(MotionCompensationMode::None.resolved_estimation(4), None);
    }

    #[test]
    fn resolved_estimation_resolves_auto_from_the_mode() {
        let m = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Auto,
        };
        assert_eq!(m.resolved_estimation(1), Some(MotionEstimation::Direct));
        assert_eq!(
            m.resolved_estimation(4),
            Some(MotionEstimation::chained_default())
        );
    }

    #[test]
    fn pair_ring_slot_count_is_double_radius() {
        assert_eq!(pair_ring_slot_count(3), 6);
        assert_eq!(pair_ring_slot_count(1), 2);
    }

    #[test]
    fn motion_ctx_blocks_match_step() {
        let mode = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        };
        let ctx = MotionCtx::new(mode, 1920, 1080).unwrap();
        assert_eq!(ctx.step, 8);
        assert_eq!(ctx.blocks_x, 1920u32.div_ceil(8));
        assert_eq!(ctx.blocks_y, 1080u32.div_ceil(8));
    }
}

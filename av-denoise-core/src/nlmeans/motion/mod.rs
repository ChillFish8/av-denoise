//! Following motion between frames so temporal denoising stays sharp.
//!
//! Temporal denoising averages a pixel with the same position in nearby
//! frames. When the camera or the content moves, that position holds
//! different content in each frame, and averaging it blurs the moving
//! parts.
//!
//! This module works out where each block of pixels moved to, then
//! shifts the neighbouring frames back into line with the current one
//! before the denoising weights are computed.
//!
//! # How a frame is tracked
//!
//! `pyramid` builds a stack of progressively smaller copies of the luma
//! plane. A search on a small copy finds large movements cheaply, and
//! the answer then seeds a short search at full resolution.
//!
//! `analyse` runs that search. `chain` handles distant neighbours by
//! measuring motion between adjacent frames and joining the results,
//! which reaches further than any single search window.
//!
//! `confidence` scores how well each block actually matched, so a block
//! that was occluded or changed can be held back rather than blurred in.
//!
//! `compensate` applies the finished motion field to a frame.

mod analyse;
mod chain;
mod compensate;
mod confidence;
mod pyramid;

#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use analyse::mv_field_byte_offset;
pub(crate) use analyse::{confidence_byte_offset, run_analyse, run_seeded_refine};
pub(crate) use chain::neighbour_idx_for_k;
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) use chain::pair_byte_offset;
pub(crate) use chain::{run_pair_analyse, zero_pair_slot};
pub(crate) use compensate::run_compensate;
pub(crate) use confidence::{run_confidence_for_neighbour, sad_noise_floor, thsad};
use cubecl::prelude::*;
use cubecl::server::Handle;
pub(crate) use pyramid::{pyramid_pixels_per_frame, run_pyramid_build};

use crate::nlmeans::align::StorageAlign;

/// The motion search's tuning, for a denoiser that always tracks motion.
///
/// [`MotionCompensationMode::Mvtools`] carries the same five values for
/// a denoiser that can also turn motion compensation off.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionSearch {
    /// The side length of each motion-search block, in pixels at the
    /// finest pyramid level.
    pub blksize: u32,
    /// How many pixels neighbouring blocks overlap.
    ///
    /// This has to be strictly below `blksize`, so the step between
    /// blocks stays positive.
    pub overlap: u32,
    /// The search radius in pixels at the finest pyramid level.
    ///
    /// The coarse pass uses the same radius on a half-size image, so
    /// its real reach is twice as far.
    pub search_radius: u32,
    /// How many levels the pyramid has.
    ///
    /// `1` means a single full-resolution search. `2` adds a half-size
    /// coarse pass that seeds the fine one. The maximum is
    /// [`MAX_PYRAMID_LEVELS`].
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
    /// An estimator inspired by MVTools tracks each block, and the
    /// neighbouring frames are shifted toward the centre frame at
    /// denoise time.
    Mvtools {
        /// The side length of each motion-search block, in pixels at
        /// the finest pyramid level.
        blksize: u32,
        /// How many pixels neighbouring blocks overlap.
        ///
        /// This has to be strictly below `blksize`, so the step between
        /// blocks stays positive.
        ///
        /// Anything above 0 leaves room for the raised-cosine blend in
        /// the compensate step, which currently uses a
        /// winner-takes-all rule instead.
        overlap: u32,
        /// The search radius in pixels at the finest pyramid level.
        ///
        /// The coarse pass uses the same radius on a half-size image, so
        /// its real reach is twice as far.
        search_radius: u32,
        /// How many levels the pyramid has.
        ///
        /// `1` means a single full-resolution search. `2` adds a
        /// half-size coarse pass that seeds the fine one. The maximum is
        /// [`MAX_PYRAMID_LEVELS`].
        pyramid_levels: u32,
        /// How motion toward each temporal neighbour is estimated.
        ///
        /// `Auto`, the default, picks a strategy from the temporal
        /// radius and is what callers normally want. Naming `Direct` or
        /// `Chained` is mostly useful for pinning one strategy in tests
        /// and benches.
        estimation: MotionEstimation,
    },
}

/// How motion toward a temporal neighbour is estimated.
#[non_exhaustive]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum MotionEstimation {
    /// Picks `Direct` or `Chained` from the temporal radius when the
    /// denoiser is built.
    ///
    /// [`MotionEstimation::resolve`] describes the rule and where it
    /// came from.
    #[default]
    Auto,
    /// Matches every neighbour against the centre frame directly, at the
    /// configured search radius.
    ///
    /// The cost grows with the temporal radius, because each neighbour
    /// repeats the whole coarse and fine search.
    Direct,
    /// Measures motion only between adjacent frames, once per pushed
    /// frame.
    ///
    /// Those per-step vectors are then joined into a seed for each
    /// neighbour, and a small seeded search cleans up whatever drift is
    /// left.
    Chained {
        /// The search radius for the seeded refinement pass, in pixels
        /// at the finest pyramid level.
        ///
        /// It can be small, because the joined seed already carries most
        /// of the real movement.
        refine_radius: u32,
    },
}

/// The default refinement radius for [`MotionEstimation::Chained`].
pub const DEFAULT_REFINE_RADIUS: u32 = 2;

/// The temporal radius at which [`MotionEstimation::Auto`] switches from
/// `Direct` to `Chained`.
///
/// Below this, `Direct` tracks slightly better, because the real motion
/// still fits inside its own search window.
///
/// At or above it, `Chained` both stays inside its window and runs
/// faster, because its reach grows with the radius rather than being
/// capped by a fixed window.
pub const CHAINED_RADIUS_THRESHOLD: u32 = 3;

impl MotionEstimation {
    /// Builds a `Chained` estimation with the library's default
    /// refinement radius.
    pub fn chained_default() -> Self {
        Self::Chained {
            refine_radius: DEFAULT_REFINE_RADIUS,
        }
    }

    /// Resolves `Auto` against the temporal radius, always returning a
    /// concrete `Direct` or `Chained`.
    ///
    /// `Direct` and `Chained` pass through unchanged whatever the
    /// radius. [`CHAINED_RADIUS_THRESHOLD`] is where the switch happens.
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

/// The default block size, matching MVTools and lining up well with the
/// patch sizes NLM typically uses.
pub const DEFAULT_BLKSIZE: u32 = 16;

/// The default block overlap, which is half the default block size.
pub const DEFAULT_OVERLAP: u32 = 8;

/// The default search radius at the finest level.
///
/// With a two-level pyramid this reaches motion of roughly 12 pixels at
/// full resolution.
pub const DEFAULT_SEARCH_RADIUS: u32 = 4;

/// The default number of pyramid levels.
///
/// Two levels give a single half-size coarse pass, which handles most
/// heavy-motion anime while keeping the number of kernel launches down.
pub const DEFAULT_PYRAMID_LEVELS: u32 = 2;

/// The hard ceiling on `pyramid_levels`.
///
/// Each extra level halves the resolution again and adds a kernel launch
/// per neighbour. Three is already more than 1080p content needs.
pub const MAX_PYRAMID_LEVELS: u32 = 3;

/// The hard ceiling on `search_radius`.
///
/// The analyse kernel scores a `(2 * radius + 1)^2` window per block, so
/// the cost grows with the square of the radius.
pub const MAX_SEARCH_RADIUS: u32 = 8;

/// The hard ceiling on `blksize`.
///
/// Above this the per-block shared-memory tile grows uncomfortably large
/// on RDNA-class GPUs.
pub const MAX_BLKSIZE: u32 = 32;

impl MotionCompensationMode {
    /// Builds an `Mvtools` mode from the library defaults.
    ///
    /// This pins `estimation` to `Direct` rather than the field's own
    /// `Auto` default, so it never switches to `Chained` at larger
    /// temporal radii the way an `Auto` configuration would.
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

    /// The estimation strategy this mode resolves to at
    /// `temporal_radius`.
    ///
    /// Returns `None` when the mode is not `Mvtools`, and never returns
    /// `Auto`. See [`MotionEstimation::resolve`].
    ///
    /// Every decision that depends on the strategy goes through here,
    /// including pair-ring allocation, whether the push-time pair
    /// analyse runs, and which branch the submit path takes.
    pub(crate) fn resolved_estimation(&self, temporal_radius: u32) -> Option<MotionEstimation> {
        match *self {
            Self::Mvtools { estimation, .. } => Some(estimation.resolve(temporal_radius)),
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

/// The motion-compensation state a `NlmDenoiser` holds while motion
/// compensation is active.
///
/// It is worked out once at construction, so the hot dispatch path never
/// has to re-read the configuration enum.
///
/// Only the fields the analyse and compensate dispatchers use live here.
/// The full configuration stays on [`MotionCompensationMode`].
#[derive(Debug, Clone)]
pub(crate) struct MotionCtx {
    pub blksize: u32,
    pub step: u32,
    pub search_radius: u32,
    pub pyramid_levels: u32,
    pub blocks_x: u32,
    pub blocks_y: u32,
    /// The alignment every buffer this context slices per slot has to
    /// respect, meaning the motion field, the confidence buffer, the
    /// pair ring, and the pyramid.
    ///
    /// It is read from the runtime. See [`StorageAlign`].
    pub align: StorageAlign,
}

impl MotionCtx {
    pub fn new(mode: MotionCompensationMode, width: u32, height: u32, align: StorageAlign) -> Option<Self> {
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
            align,
        })
    }

    /// How many motion-field slots each neighbour needs, which is one
    /// per block.
    pub fn mv_slots_per_neighbour(&self) -> usize {
        (self.blocks_x * self.blocks_y) as usize
    }

    /// The padded per-neighbour motion-field stride in bytes.
    ///
    /// Each block stores two `i32` components, and the total is rounded
    /// up to the runtime's buffer-binding alignment. This is the same
    /// convention [`Self::confidence_bytes_per_neighbour`] uses.
    ///
    /// The padding matters because wgpu rejects a bind-group offset that
    /// is not a multiple of its `min_storage_buffer_offset_alignment`,
    /// and an odd block count leaves the unpadded stride short of that
    /// boundary.
    pub(crate) fn mv_field_bytes_per_neighbour(&self) -> u64 {
        let blocks = (self.blocks_x as u64) * (self.blocks_y as u64);
        self.align.pad_bytes(blocks * 2 * size_of::<i32>() as u64)
    }

    /// The padded per-neighbour confidence-buffer stride in bytes.
    ///
    /// Each block stores one `f32`, rounded up to the runtime's
    /// buffer-binding alignment the same way
    /// [`Self::mv_field_bytes_per_neighbour`] rounds the motion field.
    pub(crate) fn confidence_bytes_per_neighbour(&self) -> u64 {
        let blocks = (self.blocks_x as u64) * (self.blocks_y as u64);
        self.align.pad_bytes(blocks * size_of::<f32>() as u64)
    }

    /// How many `i32` elements one direction of a pair-ring slot holds,
    /// which is two per block.
    ///
    /// This is the unpadded count of the data itself. It is the
    /// zero-fill length in `zero_pair_slot` and the input
    /// [`Self::pair_direction_bytes`] pads.
    pub(crate) fn pair_direction_len(&self) -> u32 {
        self.blocks_x * self.blocks_y * 2
    }

    /// The padded per-direction pair-ring stride in bytes, rounded up to
    /// the runtime's buffer-binding alignment the same way
    /// [`Self::confidence_bytes_per_neighbour`] rounds its own stride.
    ///
    /// Both `pair_byte_offset`, which the host writes and zero-fills
    /// at, and the chain-compose kernel's internal read stride use this
    /// padded value, so a direction's data starts in the same place for
    /// every reader and writer.
    pub(crate) fn pair_direction_bytes(&self) -> u64 {
        self.align
            .pad_bytes(self.pair_direction_len() as u64 * size_of::<i32>() as u64)
    }

    /// The padded per-slot pair-ring stride in bytes, covering both
    /// directions back to back.
    pub(crate) fn pair_slot_bytes(&self) -> u64 {
        2 * self.pair_direction_bytes()
    }

    /// The padded per-direction pair-ring stride in `i32` elements.
    ///
    /// The chain-compose kernel reads the whole pair ring as one array
    /// and steps through it with this value, which matches
    /// [`Self::pair_direction_bytes`] exactly.
    pub(crate) fn pair_direction_stride(&self) -> u32 {
        (self.pair_direction_bytes() / size_of::<i32>() as u64) as u32
    }

    /// The padded per-slot pair-ring stride in `i32` elements, covering
    /// both directions back to back.
    pub(crate) fn pair_slot_stride(&self) -> u32 {
        2 * self.pair_direction_stride()
    }

    /// The block geometry for a confidence pass with no motion
    /// compensation.
    ///
    /// It uses the library's default block size and overlap, one pyramid
    /// level so there is no coarse pass, and a search radius of zero, so
    /// each block is scored where it stands with no motion search at
    /// all.
    ///
    /// This is what runs when confidence weighting is on but no
    /// `Mvtools` mode was configured to take geometry from.
    pub(crate) fn confidence_only(width: u32, height: u32, align: StorageAlign) -> Self {
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
            align,
        )
        .expect("Mvtools variant always yields Some")
    }
}

/// How many slots the pair ring needs for a given temporal radius.
///
/// The pair ring stores one adjacent-frame motion field per gap between
/// consecutive frames in the temporal window. A window of
/// `2 * radius + 1` frames has exactly `2 * radius` gaps.
///
/// A gap's field is only read while both of its frames are still inside
/// some window, which lasts exactly `2 * radius` frame pushes.
///
/// Sizing the ring to match means a slot is reused precisely when its
/// old contents stop being needed, and never sooner.
/// `NlmDenoiser::pair_slot` works this out in full.
pub(crate) fn pair_ring_slot_count(temporal_radius: u32) -> u32 {
    2 * temporal_radius
}

/// Builds the pyramid for the slot `push_frame` just uploaded.
///
/// Level 0 luma is always extracted, and the smaller levels follow when
/// `pyramid_levels` is above 1.
///
/// This is a thin wrapper around [`run_pyramid_build`], which already
/// handles both cases itself.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
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
        // An overlap equal to blksize would leave a step of 0.
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
        let ctx = MotionCtx::new(mode, 1920, 1080, StorageAlign::new(32)).unwrap();
        assert_eq!(ctx.step, 8);
        assert_eq!(ctx.blocks_x, 1920u32.div_ceil(8));
        assert_eq!(ctx.blocks_y, 1080u32.div_ceil(8));
    }
}

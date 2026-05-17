//! Motion compensation plumbing. When active, the denoiser estimates
//! per-block motion between the centre frame and each temporal
//! neighbour, then warps the neighbours into spatial alignment with the
//! centre before the NLM temporal weighting kicks in. The temporal
//! distance / accumulate kernels then read neighbour pixels from a
//! parallel "compensated" ring buffer instead of `input_buf` directly.
//!
//! Inspired by MVTools' `mv.Super` / `mv.Analyse` / `mv.Compensate`
//! pipeline, simplified to integer-pixel (`pel=1`) motion with a
//! two-level pyramid for hierarchical search.

mod analyse;
mod compensate;
mod pyramid;

pub(crate) use analyse::run_analyse;
pub(crate) use compensate::run_compensate;
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
    },
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
/// pass — enough to handle most heavy-motion anime while keeping the
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
        }
    }

    /// Whether motion compensation is active at all.
    pub(crate) fn is_active(self) -> bool {
        !matches!(self, Self::None)
    }

    /// Reject parameter combinations that the kernels can't honour.
    pub fn validate(&self) -> Result<(), anyhow::Error> {
        let Self::Mvtools {
            blksize,
            overlap,
            search_radius,
            pyramid_levels,
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
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn motion_ctx_blocks_match_step() {
        let mode = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
        };
        let ctx = MotionCtx::new(mode, 1920, 1080).unwrap();
        assert_eq!(ctx.step, 8);
        assert_eq!(ctx.blocks_x, 1920u32.div_ceil(8));
        assert_eq!(ctx.blocks_y, 1080u32.div_ceil(8));
    }
}

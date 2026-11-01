use super::search::{DEFAULT_BLKSIZE, DEFAULT_OVERLAP, MotionCompensationMode, MotionEstimation};
use crate::engine::align::StorageAlign;

/// How a fine block search widens for blocks that fail a small first window.
///
/// The first pass searches `first_radius`. Blocks that fail search again in `steps` widening stages,
/// the last at `ceiling`. A `first_radius` equal to `ceiling` searches once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SearchSchedule {
    pub first_radius: u32,
    pub ceiling: u32,
    pub steps: u32,
}

impl SearchSchedule {
    /// A schedule that searches `radius` once.
    pub fn fixed(radius: u32) -> Self {
        Self {
            first_radius: radius,
            ceiling: radius,
            steps: 1,
        }
    }

    pub fn escalates(&self) -> bool {
        self.first_radius < self.ceiling
    }

    /// The radius of each widening stage, ending at `ceiling`.
    ///
    /// The stages split the gap above `first_radius` evenly, rounding up. `steps` is clamped
    /// between 1 and the gap, so every stage is wider than the one before.
    pub fn stage_radii(&self) -> Vec<u32> {
        if !self.escalates() {
            return Vec::new();
        }

        let span = self.ceiling - self.first_radius;
        let steps = self.steps.clamp(1, span);

        (1..=steps)
            .map(|stage| self.first_radius + (span * stage).div_ceil(steps))
            .collect()
    }
}

/// The block geometry a [MotionCompensationMode] resolves to for one frame size.
#[derive(Debug, Clone)]
pub(crate) struct MotionCtx {
    pub blksize: u32,
    pub step: u32,
    pub search_radius: u32,
    /// The search radius of the coarse pass, on the smallest pyramid level.
    pub coarse_search_radius: u32,
    /// How many times wider a coarse block is than the content one fine block covers.
    pub coarse_block_factor: u32,
    pub pyramid_levels: u32,
    /// How the fine search widens, which searches `search_radius` once by default.
    pub schedule: SearchSchedule,
    pub blocks_x: u32,
    pub blocks_y: u32,
    /// The alignment each per-slot slice of the motion field, confidence, pair ring and pyramid
    /// must respect.
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
            coarse_search_radius: search_radius,
            coarse_block_factor: 1,
            pyramid_levels,
            schedule: SearchSchedule::fixed(search_radius),
            blocks_x,
            blocks_y,
            align,
        })
    }

    /// This geometry with the coarse pass searching `radius` with blocks `block_factor` times wider.
    pub fn with_coarse_search(self, radius: u32, block_factor: u32) -> Self {
        Self {
            coarse_search_radius: radius,
            coarse_block_factor: block_factor,
            ..self
        }
    }

    #[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
    pub fn mv_slots_per_neighbour(&self) -> usize {
        (self.blocks_x * self.blocks_y) as usize
    }

    /// The per-neighbour motion-field stride in bytes, two `i32` per block padded to the alignment.
    ///
    /// wgpu rejects a bind-group offset that is not a multiple of its
    /// `min_storage_buffer_offset_alignment`, and an odd block count leaves the unpadded stride
    /// short of it.
    pub(crate) fn mv_field_bytes_per_neighbour(&self) -> u64 {
        let blocks = (self.blocks_x as u64) * (self.blocks_y as u64);
        let unpadded = blocks * 2 * size_of::<i32>() as u64;
        self.align.pad_bytes(unpadded)
    }

    /// The per-neighbour confidence stride in bytes, one `f32` per block padded to the alignment.
    pub(crate) fn confidence_bytes_per_neighbour(&self) -> u64 {
        let blocks = (self.blocks_x as u64) * (self.blocks_y as u64);
        let unpadded = blocks * size_of::<f32>() as u64;
        self.align.pad_bytes(unpadded)
    }

    /// `i32` values between neighbours in a motion field.
    pub(crate) fn mv_stride(&self) -> u32 {
        let values = self.mv_field_bytes_per_neighbour() / size_of::<i32>() as u64;
        values as u32
    }

    /// `f32` values between neighbours in a confidence buffer.
    pub(crate) fn conf_stride(&self) -> u32 {
        let values = self.confidence_bytes_per_neighbour() / size_of::<f32>() as u64;
        values as u32
    }

    /// The unpadded `i32` count of one pair-ring direction, two per block.
    pub(crate) fn pair_direction_len(&self) -> u32 {
        self.blocks_x * self.blocks_y * 2
    }

    /// The per-direction pair-ring stride in bytes, padded to the alignment.
    ///
    /// The host offsets and the chain-compose kernel's read stride both use it, so every reader and
    /// writer finds a direction's data in the same place.
    pub(crate) fn pair_direction_bytes(&self) -> u64 {
        let unpadded = self.pair_direction_len() as u64 * size_of::<i32>() as u64;
        self.align.pad_bytes(unpadded)
    }

    /// The per-slot pair-ring stride in bytes, with both directions back to back.
    pub(crate) fn pair_slot_bytes(&self) -> u64 {
        2 * self.pair_direction_bytes()
    }

    /// [Self::pair_direction_bytes] in `i32` elements, which the chain-compose kernel steps by.
    pub(crate) fn pair_direction_stride(&self) -> u32 {
        (self.pair_direction_bytes() / size_of::<i32>() as u64) as u32
    }

    /// [Self::pair_slot_bytes] in `i32` elements.
    pub(crate) fn pair_slot_stride(&self) -> u32 {
        2 * self.pair_direction_stride()
    }

    /// The block geometry for a confidence pass without motion compensation.
    ///
    /// It uses the default block size and overlap with one pyramid level and a search radius of 0,
    /// so each block is scored where it stands.
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

/// How many slots the pair ring needs for a temporal radius.
///
/// A window of `2 * radius + 1` frames has `2 * radius` gaps, each holding one adjacent-frame field.
/// A gap's field is read only while both its frames sit in some window, which lasts exactly
/// `2 * radius` pushes, so a slot is reused exactly when its old field stops being needed.
pub(crate) fn pair_ring_slot_count(temporal_radius: u32) -> u32 {
    2 * temporal_radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_ring_slot_count_is_double_radius() {
        let radius_three = pair_ring_slot_count(3);
        let radius_one = pair_ring_slot_count(1);

        assert_eq!(radius_three, 6);
        assert_eq!(radius_one, 2);
    }

    #[test]
    fn fixed_schedule_has_no_stages() {
        let schedule = SearchSchedule::fixed(4);

        assert!(!schedule.escalates());
        assert!(schedule.stage_radii().is_empty());
    }

    #[test]
    fn stage_radii_split_the_gap_and_end_at_the_ceiling() {
        let schedule = SearchSchedule {
            first_radius: 2,
            ceiling: 8,
            steps: 4,
        };
        let radii = schedule.stage_radii();

        assert_eq!(radii, vec![4, 5, 7, 8]);
    }

    #[test]
    fn stage_radii_clamp_steps_to_the_gap() {
        let many_steps = SearchSchedule {
            first_radius: 2,
            ceiling: 4,
            steps: 9,
        };
        let no_steps = SearchSchedule {
            first_radius: 2,
            ceiling: 4,
            steps: 0,
        };

        assert_eq!(many_steps.stage_radii(), vec![3, 4]);
        assert_eq!(no_steps.stage_radii(), vec![4]);
    }

    #[test]
    fn new_contexts_search_their_radius_once() {
        let mode = MotionCompensationMode::Mvtools {
            blksize: 16,
            overlap: 8,
            search_radius: 6,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        };
        let align = StorageAlign::new(32);
        let ctx = MotionCtx::new(mode, 64, 64, align).unwrap();

        assert_eq!(ctx.schedule, SearchSchedule::fixed(6));
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
        let align = StorageAlign::new(32);
        let ctx = MotionCtx::new(mode, 1920, 1080, align).unwrap();
        assert_eq!(ctx.step, 8);
        assert_eq!(ctx.blocks_x, 1920u32.div_ceil(8));
        assert_eq!(ctx.blocks_y, 1080u32.div_ceil(8));
    }
}

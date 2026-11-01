use cubecl::prelude::*;

use crate::motion::global::AFFINE_WORDS;

/// The most search centres one block evaluates.
pub(super) const MAX_CANDIDATES: u32 = 2;

/// The share of the noise SAD floor a later candidate must beat the existing seed by.
///
/// Two true matches of the same content differ by noise alone, so their SADs spread by a few
/// percent of the floor. An eighth of the floor keeps those near-ties on the seed.
pub const MODEL_CANDIDATE_BIAS: f32 = 0.125;

/// A search centre in a level's pixels.
#[derive(CubeType, Clone, Copy)]
pub(crate) struct BlockVector {
    pub(crate) dx: i32,
    pub(crate) dy: i32,
}

/// The search centres one block evaluates, in order of preference.
#[derive(CubeType)]
pub(super) struct CandidateList {
    dx: Array<i32>,
    dy: Array<i32>,
}

#[cube]
impl CandidateList {
    pub(super) fn new() -> CandidateList {
        CandidateList {
            dx: Array::<i32>::new(MAX_CANDIDATES as usize),
            dy: Array::<i32>::new(MAX_CANDIDATES as usize),
        }
    }

    pub(super) fn set(&mut self, #[comptime] index: u32, dx: i32, dy: i32) {
        self.dx[index as usize] = dx;
        self.dy[index as usize] = dy;
    }

    pub(super) fn get(&self, #[comptime] index: u32) -> BlockVector {
        BlockVector {
            dx: self.dx[index as usize],
            dy: self.dy[index as usize],
        }
    }
}

/// The best offset one search window found, with its SAD.
#[derive(CubeType, Clone, Copy)]
pub(super) struct WindowBest {
    pub(super) sad: f32,
    /// The mean SAD over every offset of the window, or 0 when it was not measured.
    pub(super) mean: f32,
    pub(super) dx: i32,
    pub(super) dy: i32,
    /// The lowest SAD off the window's outer ring when the best offset lies on that ring, and 0
    /// otherwise or when it was not measured.
    pub(super) edge_bar: f32,
    /// The centre of the window the best offset came from.
    pub(super) centre_dx: i32,
    pub(super) centre_dy: i32,
}

/// How many search centres a block evaluates, its existing seed first.
pub(super) fn candidate_count(model_candidate: bool) -> u32 {
    if model_candidate { 2 } else { 1 }
}

/// The motion a neighbour's camera model predicts at a block's centre, rounded to whole pixels.
///
/// `predictions` holds [AFFINE_WORDS] words per neighbour, mapping a full-resolution point in the
/// centre frame to the neighbour. The block is measured on a level `level_scale` times smaller, and
/// so is the result. A result is capped to the level's size, so a wild model cannot overflow.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the prediction reads"
)]
pub(crate) fn model_displacement(
    predictions: &[f32],
    neighbour_index: u32,
    block_origin_x: i32,
    block_origin_y: i32,
    #[comptime] blksize: u32,
    #[comptime] level_scale: u32,
    #[comptime] level_width: u32,
    #[comptime] level_height: u32,
) -> BlockVector {
    let base = (neighbour_index * AFFINE_WORDS) as usize;
    let m00 = predictions[base];
    let m01 = predictions[base + 1];
    let m10 = predictions[base + 2];
    let m11 = predictions[base + 3];
    let tx = predictions[base + 4];
    let ty = predictions[base + 5];

    let scale = comptime!(level_scale as f32);
    let half_block = comptime!((blksize / 2) as i32);
    let centre_x = (block_origin_x + half_block) as f32 * scale;
    let centre_y = (block_origin_y + half_block) as f32 * scale;
    let mapped_x = m00 * centre_x + m01 * centre_y + tx;
    let mapped_y = m10 * centre_x + m11 * centre_y + ty;
    let level_dx = (mapped_x - centre_x) / scale;
    let level_dy = (mapped_y - centre_y) / scale;

    let limit_x = comptime!(level_width as f32);
    let limit_y = comptime!(level_height as f32);
    let capped_dx = f32::clamp(level_dx, -limit_x, limit_x);
    let capped_dy = f32::clamp(level_dy, -limit_y, limit_y);

    // `floor(x + 0.5)` rounds halves the same way on every backend.
    let rounded_dx = f32::floor(capped_dx + 0.5f32);
    let rounded_dy = f32::floor(capped_dy + 0.5f32);

    BlockVector {
        dx: rounded_dx as i32,
        dy: rounded_dy as i32,
    }
}

/// Whether neighbour `neighbour_index`'s camera model moves anything.
///
/// A neighbour without a model composes to an exact identity, which adds no search centre.
#[cube]
pub(crate) fn predicts_motion(predictions: &[f32], neighbour_index: u32) -> bool {
    let base = (neighbour_index * AFFINE_WORDS) as usize;
    let scales = predictions[base] != 1.0f32 || predictions[base + 3] != 1.0f32;
    let shears = predictions[base + 1] != 0.0f32 || predictions[base + 2] != 0.0f32;
    let shifts = predictions[base + 4] != 0.0f32 || predictions[base + 5] != 0.0f32;

    scales || shears || shifts
}

/// The search centre the camera model adds, or `seed` when it adds no window.
///
/// An identity model adds no window. Nor does a model whose point lies within `search_radius` of
/// `seed` on both axes, since the seed's window already holds that point.
#[cube]
pub(super) fn model_centre(
    model: BlockVector,
    seed: BlockVector,
    moves: bool,
    #[comptime] search_radius: u32,
) -> BlockVector {
    let reach = comptime!(search_radius as i32);
    let gap_x = i32::abs(model.dx - seed.dx);
    let gap_y = i32::abs(model.dy - seed.dy);
    let outside = gap_x > reach || gap_y > reach;
    let adds_window = moves && outside;

    BlockVector {
        dx: select(adds_window, model.dx, seed.dx),
        dy: select(adds_window, model.dy, seed.dy),
    }
}

/// Whether a later candidate's SAD beats the incumbent's.
///
/// It must be lower by more than `bias`, so ties and near-ties keep the earlier candidate.
#[cube]
pub(super) fn beats_incumbent(challenger_sad: f32, incumbent_sad: f32, bias: f32) -> bool {
    let challenger_cost = challenger_sad + bias;
    challenger_cost < incumbent_sad
}

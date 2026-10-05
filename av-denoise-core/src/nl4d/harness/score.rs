use super::synth::Clip;
use crate::collab::PATCH_SIZE;
use crate::collab::geometry::{ref_pos, refs_along};
use crate::nl4d::MotionSnapshot;

/// The inclusive range of blocks whose span contains the patch starting at `patch_start`.
///
/// Each block spans `b * step..b * step + blksize`, and the range is clamped to the grid. When
/// `step == blksize` and the patch straddles a tile boundary, no block contains it, so the corner
/// block is returned as the best available.
pub fn covering_blocks(patch_start: u32, blksize: u32, step: u32, blocks: u32) -> (u32, u32) {
    let last_block = (patch_start / step).min(blocks - 1);
    let first_block = if patch_start + PATCH_SIZE <= blksize {
        0
    } else {
        (patch_start + PATCH_SIZE - blksize).div_ceil(step)
    };

    (first_block.min(last_block), last_block)
}

/// How a patch's ground truth classifies it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchKind {
    Plain,
    Boundary,
    Occluded,
}

/// Aggregated results for one patch kind.
#[derive(Debug, Clone, Default)]
pub struct KindScore {
    pub patches: usize,
    pub in_window_corner: usize,
    pub in_window_covering: usize,
    pub epe: Vec<f32>,
    pub confidence: Vec<f32>,
}

impl KindScore {
    pub fn in_window_rate_corner(&self) -> f64 {
        if self.patches == 0 {
            0.0
        } else {
            self.in_window_corner as f64 / self.patches as f64
        }
    }

    pub fn in_window_rate_covering(&self) -> f64 {
        if self.patches == 0 {
            0.0
        } else {
            self.in_window_covering as f64 / self.patches as f64
        }
    }

    pub fn epe_mean(&self) -> f64 {
        if self.epe.is_empty() {
            0.0
        } else {
            self.epe.iter().map(|&error| error as f64).sum::<f64>() / self.epe.len() as f64
        }
    }

    pub fn epe_p95(&self) -> f64 {
        percentile(&self.epe, 0.95)
    }

    pub fn confidence_median(&self) -> f64 {
        percentile(&self.confidence, 0.5)
    }
}

fn percentile(values: &[f32], quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in scores"));
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;

    sorted[index] as f64
}

/// The full score of one field against one clip.
#[derive(Debug, Clone, Default)]
pub struct Score {
    pub plain: KindScore,
    pub boundary: KindScore,
    pub occluded: KindScore,
}

impl Score {
    fn kind_mut(&mut self, kind: PatchKind) -> &mut KindScore {
        match kind {
            PatchKind::Plain => &mut self.plain,
            PatchKind::Boundary => &mut self.boundary,
            PatchKind::Occluded => &mut self.occluded,
        }
    }
}

/// The largest-axis distance between a truth and an integer vector.
fn endpoint_error(truth: [f32; 2], vector: [i32; 2]) -> f32 {
    let error_x = (truth[0] - vector[0] as f32).abs();
    let error_y = (truth[1] - vector[1] as f32).abs();

    error_x.max(error_y)
}

/// Scores `snapshot` against `clip` over nl4d's reference grid and every neighbour.
///
/// A patch is in window when its truth lies within `refine` pixels of the vector on both axes.
/// The corner reading uses the block the patch's corner sits on. The covering reading takes the
/// best of every block that covers the patch.
pub fn score(clip: &Clip, snapshot: &MotionSnapshot, refine: u32) -> Score {
    let (width, height) = (clip.width, clip.height);
    let mut result = Score::default();

    assert_eq!(
        snapshot.vectors.len(),
        snapshot.confidence.len(),
        "vectors and confidence must carry the same neighbour count and convention"
    );
    assert_eq!(
        snapshot.vectors.len(),
        clip.truth.len(),
        "the snapshot's neighbour count must match the clip's truth, which both index by \
         `neighbour_idx_for_k`"
    );

    for (t, truth) in clip.truth.iter().enumerate() {
        let occluded = &clip.occluded[t];
        for ref_y in 0..refs_along(height) {
            for ref_x in 0..refs_along(width) {
                let patch_x = ref_pos(ref_x, width);
                let patch_y = ref_pos(ref_y, height);

                let mut sum = [0.0f32; 2];
                let mut any_occluded = false;
                for y in patch_y..patch_y + PATCH_SIZE {
                    for x in patch_x..patch_x + PATCH_SIZE {
                        let idx = (y * width + x) as usize;
                        sum[0] += truth[idx][0];
                        sum[1] += truth[idx][1];
                        any_occluded |= occluded[idx];
                    }
                }

                let area = (PATCH_SIZE * PATCH_SIZE) as f32;
                let mean = [sum[0] / area, sum[1] / area];
                let mut spread = 0.0f32;
                for y in patch_y..patch_y + PATCH_SIZE {
                    for x in patch_x..patch_x + PATCH_SIZE {
                        let displacement = truth[(y * width + x) as usize];
                        let spread_x = (displacement[0] - mean[0]).abs();
                        let spread_y = (displacement[1] - mean[1]).abs();
                        spread = spread.max(spread_x).max(spread_y);
                    }
                }

                let kind = if any_occluded {
                    PatchKind::Occluded
                } else if spread > 0.5 {
                    PatchKind::Boundary
                } else {
                    PatchKind::Plain
                };

                let (first_block_x, last_block_x) =
                    covering_blocks(patch_x, snapshot.blksize, snapshot.step, snapshot.blocks_x);
                let (first_block_y, last_block_y) =
                    covering_blocks(patch_y, snapshot.blksize, snapshot.step, snapshot.blocks_y);
                let corner = (last_block_y * snapshot.blocks_x + last_block_x) as usize;
                let corner_vector = snapshot.vectors[t][corner];
                let corner_error = endpoint_error(mean, corner_vector);

                let mut best_error = corner_error;
                for by in first_block_y..=last_block_y {
                    for bx in first_block_x..=last_block_x {
                        let vector = snapshot.vectors[t][(by * snapshot.blocks_x + bx) as usize];
                        let block_error = endpoint_error(mean, vector);
                        best_error = best_error.min(block_error);
                    }
                }

                let kind_score = result.kind_mut(kind);
                kind_score.patches += 1;
                if corner_error <= refine as f32 {
                    kind_score.in_window_corner += 1;
                }

                if best_error <= refine as f32 {
                    kind_score.in_window_covering += 1;
                }

                kind_score.epe.push(corner_error);
                kind_score.confidence.push(snapshot.confidence[t][corner]);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl4d::MotionSnapshot;

    /// A 32x32 clip at radius 1 whose truth toward k = +1 is a uniform `[3, 1]`, with nothing
    /// occluded.
    fn uniform_clip() -> Clip {
        let (width, height) = (32u32, 32u32);
        let pixels = (width * height) as usize;

        Clip {
            width,
            height,
            radius: 1,
            frames: vec![vec![0.5; pixels]; 3],
            truth: vec![vec![[-3.0, -1.0]; pixels], vec![[3.0, 1.0]; pixels]],
            occluded: vec![vec![false; pixels]; 2],
        }
    }

    /// One vector for every block of every neighbour.
    fn uniform_snapshot(vector_x: i32, vector_y: i32) -> MotionSnapshot {
        let (blocks_x, blocks_y) = (4u32, 4u32);
        let blocks = (blocks_x * blocks_y) as usize;

        MotionSnapshot {
            blocks_x,
            blocks_y,
            step: 8,
            blksize: 16,
            offsets: vec![-1, 1],
            vectors: vec![
                vec![[-vector_x, -vector_y]; blocks],
                vec![[vector_x, vector_y]; blocks],
            ],
            confidence: vec![vec![0.9; blocks]; 2],
        }
    }

    #[test]
    #[should_panic(expected = "neighbour count must match")]
    fn score_asserts_the_snapshots_neighbour_count_matches_the_clips_truth() {
        let mut snapshot = uniform_snapshot(3, 1);
        snapshot.vectors.truncate(1);
        snapshot.confidence.truncate(1);

        let clip = uniform_clip();
        score(&clip, &snapshot, 2);
    }

    #[test]
    #[should_panic(expected = "same neighbour count and convention")]
    fn score_asserts_vectors_and_confidence_carry_the_same_neighbour_count() {
        let mut snapshot = uniform_snapshot(3, 1);
        snapshot.confidence.pop();

        let clip = uniform_clip();
        score(&clip, &snapshot, 2);
    }

    #[test]
    fn covering_blocks_for_the_default_geometry() {
        // At blksize 16 and step 8, the patch at 0 is covered by block 0 only, the patch at 8 by
        // blocks 0 and 1, and the patch at 16 by blocks 1 and 2.
        let at_zero = covering_blocks(0, 16, 8, 8);
        let at_eight = covering_blocks(8, 16, 8, 8);
        let at_sixteen = covering_blocks(16, 16, 8, 8);
        // step == blksize gives exactly one block.
        let step_equals_blksize = covering_blocks(24, 8, 8, 8);
        // The upper end clamps to the grid.
        let clamped = covering_blocks(56, 16, 8, 7);

        assert_eq!(at_zero, (0, 0));
        assert_eq!(at_eight, (0, 1));
        assert_eq!(at_sixteen, (1, 2));
        assert_eq!(step_equals_blksize, (3, 3));
        assert_eq!(clamped, (6, 6));
    }

    #[test]
    fn a_straddling_patch_at_step_equal_blksize_falls_back_to_the_corner_block() {
        // The patch spans 10..18, which neither block 0..16 nor block 16..32 fully contains.
        let covering = covering_blocks(10, 16, 16, 8);
        assert_eq!(covering, (0, 0));
    }

    #[test]
    fn an_exact_field_scores_every_patch_in_window_with_zero_error() {
        let clip = uniform_clip();
        let snapshot = uniform_snapshot(3, 1);
        let scores = score(&clip, &snapshot, 2);
        assert!(scores.plain.patches > 0);
        assert_eq!(scores.boundary.patches, 0);
        assert_eq!(scores.occluded.patches, 0);
        assert_eq!(scores.plain.in_window_rate_corner(), 1.0);
        assert_eq!(scores.plain.in_window_rate_covering(), 1.0);
        assert_eq!(scores.plain.epe_mean(), 0.0);
        assert!((scores.plain.confidence_median() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn an_error_past_the_refine_window_scores_out_of_window() {
        // Off by 3 on x, so refine 2 puts it out of window with an endpoint error of 3.
        let clip = uniform_clip();
        let snapshot = uniform_snapshot(6, 1);
        let scores = score(&clip, &snapshot, 2);
        assert_eq!(scores.plain.in_window_rate_corner(), 0.0);
        assert!((scores.plain.epe_mean() - 3.0).abs() < 1e-6);
        assert!((scores.plain.epe_p95() - 3.0).abs() < 1e-6);

        let scores = score(&clip, &snapshot, 3);
        assert_eq!(scores.plain.in_window_rate_corner(), 1.0);
    }

    #[test]
    fn the_covering_reading_takes_the_best_covering_block() {
        // Every other block is wrong, so patches whose corner block is wrong still count in the
        // covering reading through another block.
        let mut snapshot = uniform_snapshot(3, 1);
        for block_y in 0..4u32 {
            for block_x in 0..4u32 {
                if (block_x + block_y) % 2 == 0 {
                    snapshot.vectors[1][(block_y * 4 + block_x) as usize] = [30, 30];
                }
            }
        }

        let clip = uniform_clip();
        let scores = score(&clip, &snapshot, 2);
        let covering_rate = scores.plain.in_window_rate_covering();
        let corner_rate = scores.plain.in_window_rate_corner();
        assert!(covering_rate > corner_rate);
    }

    #[test]
    fn boundary_and_occluded_patches_are_classified_by_the_truth() {
        let mut clip = uniform_clip();
        let width = clip.width as usize;

        // A vertical motion boundary at x = 16 toward k = +1.
        for y in 0..32usize {
            for x in 16..32usize {
                clip.truth[1][y * width + x] = [0.0, 0.0];
            }
        }

        // Pixel column 20 occluded toward k = +1.
        for y in 0..32usize {
            clip.occluded[1][y * width + 20] = true;
        }

        let snapshot = uniform_snapshot(3, 1);
        let scores = score(&clip, &snapshot, 2);
        assert!(
            scores.boundary.patches > 0,
            "patches straddling x = 16 are boundary patches"
        );
        assert!(
            scores.occluded.patches > 0,
            "patches touching column 20 are occluded"
        );
        assert!(scores.plain.patches > 0);
    }
}

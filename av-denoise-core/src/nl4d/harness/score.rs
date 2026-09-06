use super::synth::Clip;
use crate::collab::PATCH_SIZE;
use crate::collab::geometry::{ref_pos, refs_along};
use crate::nl4d::MotionSnapshot;

/// The inclusive range of blocks whose `b * step..b * step + blksize`
/// span contains the patch `p..p + PATCH_SIZE`, clamped to the grid.
///
/// When `step == blksize` and the patch straddles a tile boundary, no
/// block fully contains it. The corner block is returned as the best
/// available in that case, because a consumer still needs something to
/// search.
pub fn covering_blocks(p: u32, blksize: u32, step: u32, blocks: u32) -> (u32, u32) {
    let hi = (p / step).min(blocks - 1);
    let lo = if p + PATCH_SIZE <= blksize {
        0
    } else {
        (p + PATCH_SIZE - blksize).div_ceil(step)
    };
    (lo.min(hi), hi)
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
            self.epe.iter().map(|&e| e as f64).sum::<f64>() / self.epe.len() as f64
        }
    }

    pub fn epe_p95(&self) -> f64 {
        percentile(&self.epe, 0.95)
    }

    pub fn confidence_median(&self) -> f64 {
        percentile(&self.confidence, 0.5)
    }
}

fn percentile(values: &[f32], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in scores"));
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx] as f64
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
fn endpoint_error(truth: [f32; 2], v: [i32; 2]) -> f32 {
    (truth[0] - v[0] as f32).abs().max((truth[1] - v[1] as f32).abs())
}

/// Scores `snap` against `clip` over nl4d's reference grid and every
/// neighbour.
///
/// A patch is in window when its truth lies within `refine` pixels of
/// the vector on both axes. The corner reading uses the block whose
/// corner the patch sits on, the block nl4d reads today. The covering
/// reading takes the best of every block that covers the patch.
pub fn score(clip: &Clip, snap: &MotionSnapshot, refine: u32) -> Score {
    let (w, h) = (clip.width, clip.height);
    let mut out = Score::default();

    assert_eq!(
        snap.vectors.len(),
        snap.confidence.len(),
        "vectors and confidence must carry the same neighbour count and convention"
    );
    assert_eq!(
        snap.vectors.len(),
        clip.truth.len(),
        "the snapshot's neighbour count must match the clip's truth, which both index by \
         `neighbour_idx_for_k`"
    );

    for (t, truth) in clip.truth.iter().enumerate() {
        let occluded = &clip.occluded[t];
        for ry in 0..refs_along(h) {
            for rx in 0..refs_along(w) {
                let px = ref_pos(rx, w);
                let py = ref_pos(ry, h);

                let mut sum = [0.0f32; 2];
                let mut any_occluded = false;
                for y in py..py + PATCH_SIZE {
                    for x in px..px + PATCH_SIZE {
                        let idx = (y * w + x) as usize;
                        sum[0] += truth[idx][0];
                        sum[1] += truth[idx][1];
                        any_occluded |= occluded[idx];
                    }
                }
                let area = (PATCH_SIZE * PATCH_SIZE) as f32;
                let mean = [sum[0] / area, sum[1] / area];
                let mut spread = 0.0f32;
                for y in py..py + PATCH_SIZE {
                    for x in px..px + PATCH_SIZE {
                        let d = truth[(y * w + x) as usize];
                        spread = spread.max((d[0] - mean[0]).abs()).max((d[1] - mean[1]).abs());
                    }
                }
                let kind = if any_occluded {
                    PatchKind::Occluded
                } else if spread > 0.5 {
                    PatchKind::Boundary
                } else {
                    PatchKind::Plain
                };

                let (bx_lo, bx_hi) = covering_blocks(px, snap.blksize, snap.step, snap.blocks_x);
                let (by_lo, by_hi) = covering_blocks(py, snap.blksize, snap.step, snap.blocks_y);
                let corner = (by_hi * snap.blocks_x + bx_hi) as usize;
                let corner_v = snap.vectors[t][corner];
                let corner_err = endpoint_error(mean, corner_v);

                let mut best_err = corner_err;
                for by in by_lo..=by_hi {
                    for bx in bx_lo..=bx_hi {
                        let v = snap.vectors[t][(by * snap.blocks_x + bx) as usize];
                        best_err = best_err.min(endpoint_error(mean, v));
                    }
                }

                let k = out.kind_mut(kind);
                k.patches += 1;
                if corner_err <= refine as f32 {
                    k.in_window_corner += 1;
                }
                if best_err <= refine as f32 {
                    k.in_window_covering += 1;
                }
                k.epe.push(corner_err);
                k.confidence.push(snap.confidence[t][corner]);
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl4d::MotionSnapshot;

    /// A 32x32 clip at radius 1 whose truth toward k = +1 is a uniform
    /// `[3, 1]`, with nothing occluded.
    fn uniform_clip() -> Clip {
        let (w, h) = (32u32, 32u32);
        let n = (w * h) as usize;
        Clip {
            width: w,
            height: h,
            radius: 1,
            frames: vec![vec![0.5; n]; 3],
            truth: vec![vec![[-3.0, -1.0]; n], vec![[3.0, 1.0]; n]],
            occluded: vec![vec![false; n]; 2],
        }
    }

    /// One vector for every block of every neighbour.
    fn uniform_snapshot(vx: i32, vy: i32) -> MotionSnapshot {
        let (blocks_x, blocks_y) = (4u32, 4u32);
        let blocks = (blocks_x * blocks_y) as usize;
        MotionSnapshot {
            blocks_x,
            blocks_y,
            step: 8,
            blksize: 16,
            offsets: vec![-1, 1],
            vectors: vec![vec![[-vx, -vy]; blocks], vec![[vx, vy]; blocks]],
            confidence: vec![vec![0.9; blocks]; 2],
        }
    }

    #[test]
    #[should_panic(expected = "neighbour count must match")]
    fn score_asserts_the_snapshots_neighbour_count_matches_the_clips_truth() {
        // The clip carries truth for both neighbours (k = -1, +1) but
        // the snapshot only carries one, so the two disagree on how
        // many neighbours `neighbour_idx_for_k` indexes.
        let mut snap = uniform_snapshot(3, 1);
        snap.vectors.truncate(1);
        snap.confidence.truncate(1);
        score(&uniform_clip(), &snap, 2);
    }

    #[test]
    #[should_panic(expected = "same neighbour count and convention")]
    fn score_asserts_vectors_and_confidence_carry_the_same_neighbour_count() {
        let mut snap = uniform_snapshot(3, 1);
        snap.confidence.pop();
        score(&uniform_clip(), &snap, 2);
    }

    #[test]
    fn covering_blocks_for_the_default_geometry() {
        // blksize 16, step 8: patch at 0 is covered by block 0 only,
        // patch at 8 by blocks 0 and 1, patch at 16 by blocks 1 and 2.
        assert_eq!(covering_blocks(0, 16, 8, 8), (0, 0));
        assert_eq!(covering_blocks(8, 16, 8, 8), (0, 1));
        assert_eq!(covering_blocks(16, 16, 8, 8), (1, 2));
        // step == blksize gives exactly one block.
        assert_eq!(covering_blocks(24, 8, 8, 8), (3, 3));
        // The upper end clamps to the grid.
        assert_eq!(covering_blocks(56, 16, 8, 7), (6, 6));
    }

    #[test]
    fn a_straddling_patch_at_step_equal_blksize_falls_back_to_the_corner_block() {
        // blksize 16, step 16: block 0 spans 0..16, block 1 spans
        // 16..32. The patch at p = 10 spans 10..18, which no single
        // block fully contains. No range is empty here, so the corner
        // block (the one the patch's start pixel sits in) is returned
        // as the best available search target, matching what the corner
        // reading already reads today.
        assert_eq!(covering_blocks(10, 16, 16, 8), (0, 0));
    }

    #[test]
    fn an_exact_field_scores_every_patch_in_window_with_zero_error() {
        let s = score(&uniform_clip(), &uniform_snapshot(3, 1), 2);
        assert!(s.plain.patches > 0);
        assert_eq!(s.boundary.patches, 0);
        assert_eq!(s.occluded.patches, 0);
        assert_eq!(s.plain.in_window_rate_corner(), 1.0);
        assert_eq!(s.plain.in_window_rate_covering(), 1.0);
        assert_eq!(s.plain.epe_mean(), 0.0);
        assert!((s.plain.confidence_median() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn an_error_past_the_refine_window_scores_out_of_window() {
        // Off by 3 on x, refine 2: out of window, endpoint error 3.
        let s = score(&uniform_clip(), &uniform_snapshot(6, 1), 2);
        assert_eq!(s.plain.in_window_rate_corner(), 0.0);
        assert!((s.plain.epe_mean() - 3.0).abs() < 1e-6);
        assert!((s.plain.epe_p95() - 3.0).abs() < 1e-6);
        // Refine 3 admits it.
        let s = score(&uniform_clip(), &uniform_snapshot(6, 1), 3);
        assert_eq!(s.plain.in_window_rate_corner(), 1.0);
    }

    #[test]
    fn the_covering_reading_takes_the_best_covering_block() {
        // Corner blocks wrong, every other block right. Patches whose
        // corner block is wrong but which another block covers still
        // count in the covering reading.
        let mut snap = uniform_snapshot(3, 1);
        for by in 0..4u32 {
            for bx in 0..4u32 {
                if (bx + by) % 2 == 0 {
                    snap.vectors[1][(by * 4 + bx) as usize] = [30, 30];
                }
            }
        }
        let s = score(&uniform_clip(), &snap, 2);
        assert!(s.plain.in_window_rate_covering() > s.plain.in_window_rate_corner());
    }

    #[test]
    fn boundary_and_occluded_patches_are_classified_by_the_truth() {
        let mut clip = uniform_clip();
        let w = clip.width as usize;
        // A vertical motion boundary at x = 16 toward k = +1.
        for y in 0..32usize {
            for x in 16..32usize {
                clip.truth[1][y * w + x] = [0.0, 0.0];
            }
        }
        // Pixel column 20 occluded toward k = +1.
        for y in 0..32usize {
            clip.occluded[1][y * w + 20] = true;
        }
        let s = score(&clip, &uniform_snapshot(3, 1), 2);
        assert!(
            s.boundary.patches > 0,
            "patches straddling x = 16 are boundary patches"
        );
        assert!(s.occluded.patches > 0, "patches touching column 20 are occluded");
        assert!(s.plain.patches > 0);
    }
}

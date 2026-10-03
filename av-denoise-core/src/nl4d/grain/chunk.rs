use super::consts::{HIST_LEN, LAG_COUNT, STRENGTH_GROUPS};

/// The grain statistics of up to [CHUNK_FRAMES](crate::nl4d::grain::consts::CHUNK_FRAMES)
/// consecutive completed frames of one scene.
#[derive(Debug, Clone, PartialEq)]
pub struct GrainChunk {
    /// How many completed frames the chunk covers.
    pub frames: u32,
    /// Accepted source blocks, 16 luma bins by 64 std buckets.
    pub source_hist: Vec<u32>,
    /// Accepted kept-grain blocks, in the same layout.
    pub kept_hist: Vec<u32>,
    /// The 46 source autocovariance sums of each strength group, in normalised units squared.
    pub autocov: Vec<f64>,
    /// How many source pixels each strength group's sums cover.
    pub pixels: Vec<f64>,
}

impl GrainChunk {
    pub fn empty() -> Self {
        Self {
            frames: 0,
            source_hist: vec![0; HIST_LEN],
            kept_hist: vec![0; HIST_LEN],
            autocov: vec![0.0; STRENGTH_GROUPS * LAG_COUNT],
            pixels: vec![0.0; STRENGTH_GROUPS],
        }
    }

    pub fn merge(&mut self, other: &GrainChunk) {
        self.frames += other.frames;
        add_counts(&mut self.source_hist, &other.source_hist);
        add_counts(&mut self.kept_hist, &other.kept_hist);
        add_sums(&mut self.autocov, &other.autocov);
        add_sums(&mut self.pixels, &other.pixels);
    }

    pub fn source_blocks(&self) -> u64 {
        self.source_hist.iter().map(|&count| count as u64).sum()
    }

    /// The 46 lag sums of one strength group.
    pub(crate) fn group_autocov(&self, group: usize) -> &[f64] {
        &self.autocov[group * LAG_COUNT..(group + 1) * LAG_COUNT]
    }
}

fn add_counts(target: &mut [u32], extra: &[u32]) {
    for (count, more) in target.iter_mut().zip(extra.iter()) {
        *count += more;
    }
}

fn add_sums(target: &mut [f64], extra: &[f64]) {
    for (sum, more) in target.iter_mut().zip(extra.iter()) {
        *sum += more;
    }
}

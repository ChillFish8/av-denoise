use super::consts::{HIST_LEN, LAG_COUNT};

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
    /// The 46 source autocovariance sums, in normalised units squared.
    pub autocov: Vec<f64>,
    /// How many source pixels the sums cover.
    pub pixels: f64,
}

impl GrainChunk {
    pub fn empty() -> Self {
        Self {
            frames: 0,
            source_hist: vec![0; HIST_LEN],
            kept_hist: vec![0; HIST_LEN],
            autocov: vec![0.0; LAG_COUNT],
            pixels: 0.0,
        }
    }

    pub fn merge(&mut self, other: &GrainChunk) {
        self.frames += other.frames;
        add_counts(&mut self.source_hist, &other.source_hist);
        add_counts(&mut self.kept_hist, &other.kept_hist);
        for (sum, extra) in self.autocov.iter_mut().zip(other.autocov.iter()) {
            *sum += extra;
        }

        self.pixels += other.pixels;
    }

    pub fn source_blocks(&self) -> u64 {
        self.source_hist.iter().map(|&count| count as u64).sum()
    }
}

fn add_counts(target: &mut [u32], extra: &[u32]) {
    for (count, more) in target.iter_mut().zip(extra.iter()) {
        *count += more;
    }
}

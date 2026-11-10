use super::consts::{AUTOCOV_LEN, CHROMA_PLANES, FRAME_TABLE_LEN, HIST_LEN, LAG_COUNT, STRENGTH_GROUPS};

/// The grain statistics of up to 24 consecutive completed frames of one scene.
#[derive(Debug, Clone, PartialEq)]
pub struct GrainChunk {
    /// How many completed frames the chunk covers.
    pub frames: u32,
    /// Each frame's table statistics in frame order, 2800 values per luma frame.
    ///
    /// A chroma frame holds the Cb table and then the Cr table, each in the luma layout. Counts
    /// are whole numbers and lag sums are the GPU's own f32 values. Empty when the table is not
    /// measured.
    pub frame_tables: Vec<f32>,
    /// Each frame's persistence sums in frame order, 24 values per luma frame and 6 per chroma
    /// frame.
    pub frame_records: Vec<f64>,
    /// Each frame's noise reading, 30 values per frame, empty when the reading is off.
    pub frame_noise: Vec<f32>,
    /// Each frame's colour fit, 1974 values per plane per frame, empty when the colour model is
    /// off.
    ///
    /// Each plane holds its log grain table, its log gain table and its fit header in turn.
    pub frame_colour: Vec<f32>,
}

/// Summed table statistics of one or more frames, the inputs the table fit reads.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TableSums {
    /// Accepted source block counts, 16 luma bins by 64 std buckets.
    pub source_hist: Vec<u32>,
    /// Accepted kept-grain block counts, in the same layout.
    pub kept_hist: Vec<u32>,
    /// The 46 source lag sums of each strength group.
    pub autocov: Vec<f64>,
    /// How many source pixels each strength group's sums cover.
    pub pixels: Vec<f64>,
}

/// The grain chunks drained from each plane group, in frame order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GrainChunks {
    /// Chunks measured on luma.
    pub luma: Vec<GrainChunk>,
    /// Chunks measured on the two chroma planes.
    pub chroma: Vec<GrainChunk>,
}

impl GrainChunk {
    pub fn empty() -> Self {
        Self {
            frames: 0,
            frame_tables: Vec::new(),
            frame_records: Vec::new(),
            frame_noise: Vec::new(),
            frame_colour: Vec::new(),
        }
    }

    /// The chunk's table statistics, the sum of its frames.
    pub(crate) fn summed_table(&self) -> TableSums {
        let mut sums = TableSums::zeroed();
        let (frame_tables, _) = self.frame_tables.as_chunks::<FRAME_TABLE_LEN>();
        for frame_table in frame_tables {
            sums.add_frame(frame_table);
        }

        sums
    }

    /// The chunk's table statistics of chroma plane `plane`, 0 for Cb and 1 for Cr, summed over
    /// its frames.
    #[cfg_attr(
        not(all(test, any(feature = "vulkan", feature = "metal"))),
        expect(dead_code, reason = "only the chroma grain tests read it")
    )]
    pub(crate) fn summed_chroma_table(&self, plane: usize) -> TableSums {
        let mut sums = TableSums::zeroed();
        let (frame_tables, _) = self.frame_tables.as_chunks::<FRAME_TABLE_LEN>();
        let plane_tables = frame_tables.iter().skip(plane).step_by(CHROMA_PLANES);
        for plane_table in plane_tables {
            sums.add_frame(plane_table);
        }

        sums
    }
}

impl TableSums {
    fn zeroed() -> Self {
        Self {
            source_hist: vec![0; HIST_LEN],
            kept_hist: vec![0; HIST_LEN],
            autocov: vec![0.0; STRENGTH_GROUPS * LAG_COUNT],
            pixels: vec![0.0; STRENGTH_GROUPS],
        }
    }

    pub(crate) fn add(&mut self, other: &TableSums) {
        add_counts(&mut self.source_hist, &other.source_hist);
        add_counts(&mut self.kept_hist, &other.kept_hist);
        add_sums(&mut self.autocov, &other.autocov);
        add_sums(&mut self.pixels, &other.pixels);
    }

    /// Adds one frame's table statistics, laid out as in [GrainChunk::frame_tables].
    fn add_frame(&mut self, frame_table: &[f32; FRAME_TABLE_LEN]) {
        let (source_counts, rest) = frame_table.split_at(HIST_LEN);
        let (kept_counts, grouped) = rest.split_at(HIST_LEN);
        add_whole_counts(&mut self.source_hist, source_counts);
        add_whole_counts(&mut self.kept_hist, kept_counts);

        let (records, _) = grouped.as_chunks::<AUTOCOV_LEN>();
        let (targets, _) = self.autocov.as_chunks_mut::<LAG_COUNT>();
        let groups = targets.iter_mut().zip(self.pixels.iter_mut());
        for ((target, pixels), record) in groups.zip(records) {
            add_frame_sums(target, &record[..LAG_COUNT]);
            *pixels += record[LAG_COUNT] as f64;
        }
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

/// Adds counts held as whole `f32` values.
fn add_whole_counts(target: &mut [u32], extra: &[f32]) {
    for (count, &more) in target.iter_mut().zip(extra.iter()) {
        *count += more as u32;
    }
}

fn add_sums(target: &mut [f64], extra: &[f64]) {
    for (sum, more) in target.iter_mut().zip(extra.iter()) {
        *sum += more;
    }
}

/// Adds one frame's f32 sums into f64 totals.
fn add_frame_sums(target: &mut [f64], extra: &[f32]) {
    for (sum, &more) in target.iter_mut().zip(extra.iter()) {
        *sum += more as f64;
    }
}

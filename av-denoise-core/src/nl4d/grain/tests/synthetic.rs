use crate::nl4d::grain::chunk::GrainChunk;
use crate::nl4d::grain::consts::{
    AR_COEFFS,
    AR_OFFSETS,
    AUTOCOV_LEN,
    BUCKETS_PER_GROUP,
    CELL,
    CHUNK_FRAMES,
    FRAME_TABLE_LEN,
    HIST_LEN,
    LAG_COUNT,
    LAGS,
    LUMA_BINS,
    MAX_LAG_X,
    MAX_LAG_Y,
    STD_BUCKETS,
    STRENGTH_GROUPS,
};
use crate::nl4d::grain::fit::{bucket_edges, bucket_of};
use crate::nl4d::grain::table::FitOptions;

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut mixed = *state;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    mixed ^ (mixed >> 31)
}

fn uniform(state: &mut u64) -> f64 {
    let bits = splitmix(state) >> 11;
    (bits as f64 + 0.5) / (1u64 << 53) as f64
}

/// Unit-variance Gaussian samples, Box-Muller over a fixed generator.
pub(super) fn gaussian_field(width: usize, height: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut field = Vec::with_capacity(width * height);
    while field.len() < width * height {
        let first = uniform(&mut state);
        let second = uniform(&mut state);
        let radius = (-2.0 * first.ln()).sqrt();
        let angle = std::f64::consts::TAU * second;
        field.push(radius * angle.cos());
        field.push(radius * angle.sin());
    }

    field.truncate(width * height);

    field
}

/// A causal lag-3 AR field with these weights, driven by unit Gaussian noise.
///
/// The first 50 rows and columns warm the recursion up and are dropped.
pub(super) fn ar_field(coeffs: &[f64; AR_COEFFS], width: usize, height: usize, seed: u64) -> Vec<f64> {
    let warm = 50;
    let full_width = width + warm + 3;
    let full_height = height + warm;
    let noise = gaussian_field(full_width, full_height, seed);
    let mut grain = vec![0.0f64; full_width * full_height];
    for y in 3..full_height {
        for x in 3..full_width - 3 {
            let mut total = noise[y * full_width + x];
            for (&(dy, dx), &coeff) in AR_OFFSETS.iter().zip(coeffs.iter()) {
                let neighbour_y = (y as i32 + dy) as usize;
                let neighbour_x = (x as i32 + dx) as usize;
                total += coeff * grain[neighbour_y * full_width + neighbour_x];
            }

            grain[y * full_width + x] = total;
        }
    }

    let mut field = Vec::with_capacity(width * height);
    for y in warm..warm + height {
        let start = y * full_width + warm;
        field.extend_from_slice(&grain[start..start + width]);
    }

    field
}

/// The 46 lag sums over every pixel whose neighbours stay inside the field, then the pixel count.
pub(super) fn autocov_of(field: &[f64], width: usize, height: usize) -> Vec<f64> {
    let mut sums = vec![0.0f64; LAG_COUNT + 1];
    let lag_x = MAX_LAG_X as usize;
    let lag_y = MAX_LAG_Y as usize;
    for y in 0..height - lag_y {
        for x in lag_x..width - lag_x {
            let centre = field[y * width + x];
            for (lane, &(dy, dx)) in LAGS.iter().enumerate() {
                let neighbour_y = (y as i32 + dy) as usize;
                let neighbour_x = (x as i32 + dx) as usize;
                sums[lane] += centre * field[neighbour_y * width + neighbour_x];
            }

            sums[LAG_COUNT] += 1.0;
        }
    }

    sums
}

/// The record the measure kernel builds from `field`, every 8x8 cell's mean taken off the cell
/// and its lag neighbours.
pub(super) fn cell_mean_removed_record(field: &[f64], width: usize, height: usize) -> Vec<f64> {
    let cell = CELL as usize;
    let mut sums = vec![0.0f64; LAG_COUNT + 1];

    for cell_top in (0..height - 2 * cell).step_by(cell) {
        for cell_left in (cell..width - 2 * cell).step_by(cell) {
            let mut total = 0.0;
            for y in cell_top..cell_top + cell {
                let row = &field[y * width + cell_left..y * width + cell_left + cell];
                total += row.iter().sum::<f64>();
            }

            let mean = total / (cell * cell) as f64;
            for y in cell_top..cell_top + cell {
                for x in cell_left..cell_left + cell {
                    let centre = field[y * width + x] - mean;
                    for (lane, &(dy, dx)) in LAGS.iter().enumerate() {
                        let neighbour_y = (y as i32 + dy) as usize;
                        let neighbour_x = (x as i32 + dx) as usize;
                        let neighbour = field[neighbour_y * width + neighbour_x] - mean;
                        sums[lane] += centre * neighbour;
                    }
                }
            }

            sums[LAG_COUNT] += (cell * cell) as f64;
        }
    }

    sums
}

/// Lag-3 AR weights with only a 0.4 left weight.
pub(super) fn left_weights() -> [f64; AR_COEFFS] {
    let mut weights = [0.0f64; AR_COEFFS];
    let left = AR_OFFSETS
        .iter()
        .position(|&offset| offset == (0, -1))
        .expect("left offset");
    weights[left] = 0.4;

    weights
}

/// The autocovariance record of a lag-3 AR field with a 0.4 left weight.
pub(super) fn grain_record() -> Vec<f64> {
    let weights = left_weights();
    let field = ar_field(&weights, 300, 300, 5);

    autocov_of(&field, 300, 300)
}

/// A chunk of `CHUNK_FRAMES` frames with zeroed table statistics.
///
/// The helpers below write every statistic into the chunk's first frame.
pub(super) fn table_chunk() -> GrainChunk {
    let mut chunk = GrainChunk::empty();
    chunk.frames = CHUNK_FRAMES;
    chunk.frame_tables = vec![0.0; CHUNK_FRAMES as usize * FRAME_TABLE_LEN];

    chunk
}

/// Sets the first frame's source count at `slot` of the histogram.
pub(super) fn set_source_count(chunk: &mut GrainChunk, slot: usize, count: u32) {
    chunk.frame_tables[slot] = count as f32;
}

/// Sets the first frame's one-frame count at `slot` of the histogram.
pub(super) fn set_frame_count(chunk: &mut GrainChunk, slot: usize, count: u32) {
    chunk.frame_tables[HIST_LEN + slot] = count as f32;
}

/// The first frame's 47-value record of one strength group.
fn group_record_mut(chunk: &mut GrainChunk, group: usize) -> &mut [f32] {
    let start = 2 * HIST_LEN + group * AUTOCOV_LEN;

    &mut chunk.frame_tables[start..start + AUTOCOV_LEN]
}

/// Adds a 47-value record into one strength group of `chunk`.
pub(super) fn add_group_record(chunk: &mut GrainChunk, group: usize, record: &[f64]) {
    let sums = group_record_mut(chunk, group);
    for (sum, &extra) in sums.iter_mut().zip(record) {
        *sum += extra as f32;
    }
}

/// Zeroes the pixel count of every strength group.
pub(super) fn clear_pixels(chunk: &mut GrainChunk) {
    for group in 0..STRENGTH_GROUPS {
        let record = group_record_mut(chunk, group);
        record[LAG_COUNT] = 0.0;
    }
}

/// A chunk whose source blocks all sit in `bucket` in luma bins 4 to 11, with no AR record.
pub(super) fn chunk_in_bucket(bucket: usize, blocks_per_bin: u32) -> GrainChunk {
    let mut chunk = table_chunk();

    for bin in 4..12 {
        set_source_count(&mut chunk, bin * STD_BUCKETS + bucket, blocks_per_bin);
    }

    chunk
}

/// A chunk whose source blocks all sit at `std_codes` in luma bins 4 to 11, with an AR record.
///
/// The one-frame blocks sit at `frame_codes` in the same bins, or with the source blocks when it is
/// `None`. The record goes into the strength group of the source blocks' bucket.
pub(super) fn chunk_at(std_codes: f32, blocks_per_bin: u32, frame_codes: Option<f32>) -> GrainChunk {
    let edges = bucket_edges();
    let bucket = bucket_of(std_codes / 255.0, &edges);
    let mut chunk = chunk_in_bucket(bucket, blocks_per_bin);

    let frame_std = frame_codes.unwrap_or(std_codes);
    let frame_bucket = bucket_of(frame_std / 255.0, &edges);
    for bin in 4..12 {
        set_frame_count(&mut chunk, bin * STD_BUCKETS + frame_bucket, blocks_per_bin);
    }

    let record = grain_record();
    add_group_record(&mut chunk, bucket / BUCKETS_PER_GROUP, &record);

    chunk
}

/// Zeroes the first frame's one-frame counts in every luma bin outside `kept`.
pub(super) fn keep_frame_bins(chunk: &mut GrainChunk, kept: std::ops::Range<usize>) {
    for bin in 0..LUMA_BINS {
        if kept.contains(&bin) {
            continue;
        }

        let start = HIST_LEN + bin * STD_BUCKETS;
        chunk.frame_tables[start..start + STD_BUCKETS].fill(0.0);
    }
}

/// A chroma chunk with the frame tables of `cb` and `cr` in every frame.
pub(super) fn chroma_chunk(cb: &GrainChunk, cr: &GrainChunk) -> GrainChunk {
    let mut chunk = GrainChunk::empty();
    chunk.frames = cb.frames;

    let (cb_tables, _) = cb.frame_tables.as_chunks::<FRAME_TABLE_LEN>();
    let (cr_tables, _) = cr.frame_tables.as_chunks::<FRAME_TABLE_LEN>();
    for (cb_table, cr_table) in cb_tables.iter().zip(cr_tables) {
        chunk.frame_tables.extend_from_slice(cb_table);
        chunk.frame_tables.extend_from_slice(cr_table);
    }

    chunk
}

pub(super) fn chroma_options() -> FitOptions {
    let options = FitOptions::default();

    options.with_chroma(true)
}

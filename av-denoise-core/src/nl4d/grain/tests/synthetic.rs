use crate::nl4d::grain::chunk::GrainChunk;
use crate::nl4d::grain::consts::{
    AR_COEFFS,
    AR_OFFSETS,
    CHUNK_FRAMES,
    LAG_COUNT,
    LAGS,
    MAX_LAG_X,
    MAX_LAG_Y,
    STD_BUCKETS,
};
use crate::nl4d::grain::fit::{bucket_edges, bucket_of};

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

/// A chunk whose source blocks all sit at `std_codes` in luma bins 4 to 11, with an AR record.
pub(super) fn chunk_at(std_codes: f32, blocks_per_bin: u32, kept_codes: Option<f32>) -> GrainChunk {
    let edges = bucket_edges();
    let mut chunk = GrainChunk::empty();
    chunk.frames = CHUNK_FRAMES;

    let bucket = bucket_of(std_codes / 255.0, &edges);
    for bin in 4..12 {
        chunk.source_hist[bin * STD_BUCKETS + bucket] = blocks_per_bin;
        if let Some(kept) = kept_codes {
            let kept_bucket = bucket_of(kept / 255.0, &edges);
            chunk.kept_hist[bin * STD_BUCKETS + kept_bucket] = blocks_per_bin;
        }
    }

    let mut weights = [0.0f64; AR_COEFFS];
    let left = AR_OFFSETS
        .iter()
        .position(|&offset| offset == (0, -1))
        .expect("left offset");
    weights[left] = 0.4;

    let field = ar_field(&weights, 300, 300, 5);
    let autocov = autocov_of(&field, 300, 300);
    chunk.autocov = autocov[..autocov.len() - 1].to_vec();
    chunk.pixels = autocov[autocov.len() - 1];
    chunk
}

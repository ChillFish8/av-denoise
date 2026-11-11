use super::consts::{
    AR_COEFFS,
    AR_OFFSETS,
    CELL,
    LAG_COUNT,
    LAGS,
    MAX_LAG_X,
    MAX_LAG_Y,
    STD_BUCKETS,
    STD_MAX,
    STD_MIN,
};

/// Rounds of the fixed-point solve in [undo_mean_removal].
const MEAN_REMOVAL_ROUNDS: usize = 20;
/// The largest offset between two pixels of one cell, on either axis.
const CELL_SPAN: i32 = CELL as i32 - 1;
/// Frequencies on each axis of the grid [floor_fraction] reads.
const SPECTRUM_STEPS: usize = 64;

/// The 65 std bucket edges, log-spaced between `STD_MIN` and `STD_MAX`.
pub(crate) fn bucket_edges() -> [f32; STD_BUCKETS + 1] {
    let ratio = (STD_MAX as f64) / (STD_MIN as f64);
    let mut edges = [0.0f32; STD_BUCKETS + 1];
    for (index, edge) in edges.iter_mut().enumerate() {
        let fraction = index as f64 / STD_BUCKETS as f64;
        *edge = (STD_MIN as f64 * ratio.powf(fraction)) as f32;
    }

    edges
}

/// The bucket a std falls in, with anything at or above the top edge in the last bucket.
#[cfg(test)]
pub(crate) fn bucket_of(std: f32, edges: &[f32]) -> usize {
    let mut bucket = 0;
    for (index, &edge) in edges.iter().enumerate().take(STD_BUCKETS).skip(1) {
        if std >= edge {
            bucket = index;
        }
    }

    bucket
}

/// The median std of a 64-bucket histogram, interpolated geometrically inside its bucket.
pub(crate) fn hist_median(counts: &[u32], edges: &[f32]) -> Option<f64> {
    let total: u64 = counts.iter().map(|&count| count as u64).sum();
    if total == 0 {
        return None;
    }

    let target = 0.5 * total as f64;
    let mut below = 0.0f64;
    for (bucket, &count) in counts.iter().enumerate() {
        let count = count as f64;
        if count > 0.0 && below + count >= target {
            let fraction = (target - below) / count;
            let low = edges[bucket] as f64;
            let high = edges[bucket + 1] as f64;
            return Some(low * (high / low).powf(fraction));
        }

        below += count;
    }

    None
}

/// The record lane of a lag folded onto the measured half-plane, `None` past the measured lags.
pub(crate) fn measured_lane(dy: i32, dx: i32) -> Option<usize> {
    let flip = dy < 0 || (dy == 0 && dx < 0);
    let (dy, dx) = if flip { (-dy, -dx) } else { (dy, dx) };
    if dy > MAX_LAG_Y || dx.abs() > MAX_LAG_X {
        return None;
    }

    let row_width = 2 * MAX_LAG_X + 1;
    let lane = if dy == 0 {
        dx
    } else {
        (MAX_LAG_X + 1) + (dy - 1) * row_width + dx + MAX_LAG_X
    };
    Some(lane as usize)
}

fn lag_index(dy: i32, dx: i32) -> usize {
    measured_lane(dy, dx).expect("every AR lag difference is measured")
}

/// The correlation at lag `dy, dx` averaged over every pixel pair of one cell.
///
/// This is how much a cell's mean removal takes off that lag, in units of the zero lag.
fn cell_mean_term(rho: &[f64], dy: i32, dx: i32) -> f64 {
    let cell_pairs = (CELL * CELL * CELL * CELL) as f64;
    let mut total = 0.0;

    for offset_y in -CELL_SPAN..=CELL_SPAN {
        for offset_x in -CELL_SPAN..=CELL_SPAN {
            let Some(lane) = measured_lane(dy + offset_y, dx + offset_x) else {
                continue;
            };

            let overlap_y = CELL as i32 - offset_y.abs();
            let overlap_x = CELL as i32 - offset_x.abs();
            let pairs = (overlap_y * overlap_x) as f64;
            total += pairs / cell_pairs * rho[lane];
        }
    }

    total
}

/// Undoes the bias that removing each cell's mean puts in a measured record.
///
/// The mean-removed products average `R(l) - a(l)`. Normalised by the zero lag, the true
/// correlation solves `rho(l) = m(l) * (1 - a(0)) + a(l)` for the measured `m`, found by
/// fixed-point iteration from `rho = m`. Lags past the measured set count as zero. The result
/// holds `rho` scaled by the measured zero-lag sum, then the unchanged pixel count.
pub(crate) fn undo_mean_removal(record: &[f64]) -> Vec<f64> {
    let zero_lag = record[0];
    let mut corrected = record.to_vec();
    if zero_lag <= 0.0 {
        return corrected;
    }

    let measured: Vec<f64> = record[..LAG_COUNT].iter().map(|&sum| sum / zero_lag).collect();
    let mut rho = measured.clone();
    rho[0] = 1.0;

    for _ in 0..MEAN_REMOVAL_ROUNDS {
        let zero_term = cell_mean_term(&rho, 0, 0);
        let mut next = Vec::with_capacity(LAG_COUNT);
        for (&(dy, dx), &measured_lag) in LAGS.iter().zip(&measured) {
            let lag_term = cell_mean_term(&rho, dy, dx);
            next.push(measured_lag * (1.0 - zero_term) + lag_term);
        }

        rho = next;
    }

    for (sum, &value) in corrected.iter_mut().zip(&rho) {
        *sum = value * zero_lag;
    }

    corrected
}

/// Solves AV1's 24 lag-3 weights from the summed autocovariance.
///
/// `autocov` holds the 46 lag sums and then the pixel count. Returns `None` for an empty or
/// singular system.
pub(crate) fn yule_walker(autocov: &[f64]) -> Option<[f64; AR_COEFFS]> {
    let pixels = autocov[LAG_COUNT];
    if pixels <= 0.0 {
        return None;
    }

    let mut matrix = vec![[0.0f64; AR_COEFFS + 1]; AR_COEFFS];
    for (row, &(row_y, row_x)) in AR_OFFSETS.iter().enumerate() {
        for (col, &(col_y, col_x)) in AR_OFFSETS.iter().enumerate() {
            let lag = lag_index(row_y - col_y, row_x - col_x);
            matrix[row][col] = autocov[lag] / pixels;
        }

        let target = lag_index(row_y, row_x);
        matrix[row][AR_COEFFS] = autocov[target] / pixels;
    }

    solve(matrix)
}

/// Gaussian elimination with partial pivoting on an augmented 24x25 system.
fn solve(mut matrix: Vec<[f64; AR_COEFFS + 1]>) -> Option<[f64; AR_COEFFS]> {
    for col in 0..AR_COEFFS {
        let pivot = (col..AR_COEFFS).max_by(|&first, &second| {
            let first_size = matrix[first][col].abs();
            let second_size = matrix[second][col].abs();
            first_size.total_cmp(&second_size)
        })?;
        if matrix[pivot][col].abs() < 1e-18 {
            return None;
        }

        matrix.swap(col, pivot);

        for row in col + 1..AR_COEFFS {
            let pivot_row = matrix[col];
            let factor = matrix[row][col] / pivot_row[col];
            let targets = matrix[row][col..].iter_mut();
            for (target, &source) in targets.zip(&pivot_row[col..]) {
                *target -= factor * source;
            }
        }
    }

    let mut solution = [0.0f64; AR_COEFFS];
    for row in (0..AR_COEFFS).rev() {
        let mut value = matrix[row][AR_COEFFS];
        for entry in row + 1..AR_COEFFS {
            value -= matrix[row][entry] * solution[entry];
        }

        solution[row] = value / matrix[row][row];
    }

    let finite = solution.iter().all(|value| value.is_finite());
    finite.then_some(solution)
}

/// The share of a texture's variance a flat, white floor accounts for, between 0 and 1.
///
/// White noise adds the same power at every frequency, so the floor is the lowest point of the
/// weights' [ar_spectrum]. The share is that floor over the spectrum's mean. Coarse grain has almost
/// no power at the finest frequencies, so its lowest point and share are close to 0.
pub(crate) fn floor_fraction(coeffs: &[f64; AR_COEFFS]) -> f64 {
    let spectrum = ar_spectrum(coeffs);
    let lowest = spectrum.iter().copied().fold(f64::INFINITY, f64::min);
    let total: f64 = spectrum.iter().sum();

    let mean = total / spectrum.len() as f64;
    let fraction = lowest / mean;
    if !fraction.is_finite() {
        return 0.0;
    }

    fraction.clamp(0.0, 1.0)
}

/// The power of unit noise through the AR filter on a grid of [SPECTRUM_STEPS] by
/// [SPECTRUM_STEPS] frequencies, row by row.
pub(crate) fn ar_spectrum(coeffs: &[f64; AR_COEFFS]) -> Vec<f64> {
    let step = std::f64::consts::TAU / SPECTRUM_STEPS as f64;
    let mut spectrum = Vec::with_capacity(SPECTRUM_STEPS * SPECTRUM_STEPS);

    for row in 0..SPECTRUM_STEPS {
        for col in 0..SPECTRUM_STEPS {
            let frequency_y = row as f64 * step;
            let frequency_x = col as f64 * step;
            let power = ar_power(coeffs, frequency_y, frequency_x);
            spectrum.push(power);
        }
    }

    spectrum
}

/// The power of unit noise through the AR filter at one spatial frequency.
fn ar_power(coeffs: &[f64; AR_COEFFS], frequency_y: f64, frequency_x: f64) -> f64 {
    let mut real = 1.0;
    let mut imaginary = 0.0;
    for (&(dy, dx), &coeff) in AR_OFFSETS.iter().zip(coeffs) {
        let phase = frequency_y * dy as f64 + frequency_x * dx as f64;
        real -= coeff * phase.cos();
        imaginary -= coeff * phase.sin();
    }

    let gain = real * real + imaginary * imaginary;

    1.0 / gain
}

/// Rounds the weights to AV1 integers with the finest shift that keeps each in a signed byte.
///
/// AV1 allows an AR shift between 6 and 9. Weights that overflow at a shift of 6 are clamped.
pub(crate) fn quantise_ar(coeffs: &[f64; AR_COEFFS]) -> ([i32; AR_COEFFS], u32) {
    let shift = finest_ar_shift(coeffs);
    let quantised = quantise_at(coeffs, shift);

    (quantised, shift)
}

/// The finest AR shift between 6 and 9 at which every rounded weight fits a signed byte.
///
/// Weights that overflow at every shift give 6.
pub(crate) fn finest_ar_shift(coeffs: &[f64; AR_COEFFS]) -> u32 {
    for shift in [9u32, 8, 7] {
        let scale = (1u32 << shift) as f64;
        let fits = coeffs.iter().all(|&coeff| {
            let rounded = (coeff * scale).round() as i32;
            (-128..=127).contains(&rounded)
        });
        if fits {
            return shift;
        }
    }

    6
}

/// Rounds the weights to AV1 integers at `shift`, clamped to a signed byte.
pub(crate) fn quantise_at(coeffs: &[f64; AR_COEFFS], shift: u32) -> [i32; AR_COEFFS] {
    let scale = (1u32 << shift) as f64;

    coeffs.map(|coeff| {
        let rounded = (coeff * scale).round() as i32;
        rounded.clamp(-128, 127)
    })
}

/// Turns per-bin target sigmas into AV1 scaling points and the scaling shift.
///
/// `points` holds `(luma, sigma)` with luma in 8-bit codes and sigma in normalised units.
/// `sigma_template` is the template std in 8-bit units. AV1 allows a scaling shift between 8 and
/// 11 and a byte per scaling value, so the finest shift that fits the largest value wins and
/// values past 255 are capped.
pub(crate) fn scaling_points(points: &[(u8, f64)], sigma_template: f64) -> (Vec<(u8, u8)>, u32) {
    let peak = scaling_peak(points, sigma_template);
    let shift = finest_scaling_shift(peak);
    let scaled = scaling_points_at(points, sigma_template, shift);

    (scaled, shift)
}

/// The scaling value one sigma needs before the scaling shift.
fn raw_scaling(sigma: f64, sigma_template: f64) -> f64 {
    sigma * 255.0 / sigma_template
}

/// The largest scaling value the points need before the scaling shift, at least 0.
pub(crate) fn scaling_peak(points: &[(u8, f64)], sigma_template: f64) -> f64 {
    points
        .iter()
        .map(|&(_, sigma)| raw_scaling(sigma, sigma_template))
        .fold(0.0f64, f64::max)
}

/// The finest scaling shift between 8 and 11 at which `peak` stays within a byte.
///
/// A peak past a byte at every shift gives 8.
pub(crate) fn finest_scaling_shift(peak: f64) -> u32 {
    for shift in [11u32, 10, 9] {
        if peak * (1u32 << shift) as f64 <= 255.0 {
            return shift;
        }
    }

    8
}

/// The scaling points at `shift`, with values past 255 capped.
pub(crate) fn scaling_points_at(points: &[(u8, f64)], sigma_template: f64, shift: u32) -> Vec<(u8, u8)> {
    let factor = (1u32 << shift) as f64;

    points
        .iter()
        .map(|&(luma, sigma)| {
            let value = raw_scaling(sigma, sigma_template);
            let scaled = (value * factor).round().min(255.0) as u8;
            (luma, scaled)
        })
        .collect()
}

/// Correlation-correction points as `(rho, factor)`, sorted by `rho`.
///
/// Calibrated on synthetic correlated-grain sweeps against the clean bench reference. Each factor
/// is how far the quality peak sits above the true sigma at that correlation, relative to the
/// white-noise optimum at the same sigma. At the heaviest correlation both quality metrics prefer
/// the raised value, and past the last point the table holds flat.
const CORRELATION_FACTOR_TABLE: [(f32, f32); 4] = [(0.0, 1.0), (0.3, 1.05), (0.5, 1.25), (0.65, 1.45)];

/// The factor that scales a measured temporal sigma up to allow for grain correlation.
pub(in crate::nlmeans) fn correlation_factor(rho: f32) -> f32 {
    interpolate_table(&CORRELATION_FACTOR_TABLE, rho)
}

/// Linearly interpolates a table of points sorted by `x`, clamping `x` to the table's range.
pub(super) fn interpolate_table(table: &[(f32, f32)], x: f32) -> f32 {
    let x = x.clamp(table[0].0, table[table.len() - 1].0);

    for segment in table.windows(2) {
        let (start_x, start_y) = segment[0];
        let (end_x, end_y) = segment[1];
        if x <= end_x {
            if end_x == start_x {
                return end_y;
            }

            let fraction = (x - start_x) / (end_x - start_x);
            return start_y + fraction * (end_y - start_y);
        }
    }

    table[table.len() - 1].1
}

/// The share of the noise floor that applies at a candidate offset under grain correlation.
///
/// A nearby candidate shares part of its grain with the centre patch, so only part of the
/// white-noise floor is independent noise, and that part grows with distance. The centre returns 0
/// because its true distance is zero. With no measured correlation every other offset returns 1,
/// which reproduces the flat white-noise floor.
pub(in crate::nlmeans) fn spatial_offset_factor(dx: i32, dy: i32, rho: f32) -> f32 {
    if dx == 0 && dy == 0 {
        return 0.0;
    }

    if rho <= 0.0 {
        return 1.0;
    }

    let distance = ((dx * dx + dy * dy) as f32).sqrt();
    let log_rho = rho.ln();
    1.0 - (distance * log_rho).exp()
}

pub(crate) fn spatial_offset_lut_len(search_radius: u32) -> usize {
    let side = (2 * search_radius + 1) as usize;
    side * side
}

/// Builds the row-major noise-floor table for a search window.
///
/// Each entry is the flat noise offset scaled by [spatial_offset_factor] at that candidate. It is
/// cheap enough to rebuild on every submit, at most 289 entries at the largest search radius.
pub(crate) fn build_spatial_offset_lut(search_radius: u32, rho: f32, noise_offset: f32) -> Vec<f32> {
    let radius = search_radius as i32;
    let side = (2 * search_radius + 1) as usize;
    let mut lut = vec![0.0f32; side * side];

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let index = ((dy + radius) as usize) * side + (dx + radius) as usize;
            lut[index] = noise_offset * spatial_offset_factor(dx, dy, rho);
        }
    }

    lut
}

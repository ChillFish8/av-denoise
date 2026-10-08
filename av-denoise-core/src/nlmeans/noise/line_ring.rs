use super::curve::NoiseCurve;

/// How many times the noise curve's sigma a line quarter's RMS gradient must reach.
const LINE_SIGMA_FACTOR: f32 = 3.0;

/// The luma range a line quarter must span, in 8-bit code values.
const LINE_CONTRAST_FLOOR: f32 = 32.0;

/// How many 2x2 gradient windows a full 8x8 quarter's structure tensor sums over.
const FULL_QUARTER_WINDOWS: f32 = 49.0;

/// What the line detector reads from one full 8x8 quarter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LineInput {
    /// The quarter's mean luma, between 0 and 1.
    pub(crate) luma: f32,
    /// The quarter's raw luma range, between 0 and 1.
    pub(crate) luma_range: f32,
    /// The trace of the quarter's structure tensor.
    pub(crate) tensor_trace: f32,
}

/// Marks each quarter that holds a strong line, row-major.
///
/// A quarter is a line when its RMS gradient reaches [LINE_SIGMA_FACTOR] times the noise `curve`'s
/// sigma at its luma, and its luma range reaches [LINE_CONTRAST_FLOOR] codes. A `None` input is a
/// partial quarter at the frame edge and is never a line.
pub(crate) fn line_quarters(inputs: &[Option<LineInput>], curve: &NoiseCurve) -> Vec<bool> {
    inputs
        .iter()
        .map(|input| match input {
            Some(input) => is_line(input, curve),
            None => false,
        })
        .collect()
}

fn is_line(input: &LineInput, curve: &NoiseCurve) -> bool {
    let mean_square_gradient = (input.tensor_trace / FULL_QUARTER_WINDOWS).max(0.0);
    let rms_gradient = mean_square_gradient.sqrt();
    let sigma = curve.sigma_at(input.luma);
    let strong = rms_gradient >= LINE_SIGMA_FACTOR * sigma;
    let contrasty = input.luma_range * 255.0 >= LINE_CONTRAST_FLOOR;
    strong && contrasty
}

/// Grows every marked quarter into a square reaching `radius` quarters on each side.
///
/// The square is clamped at the grid's edges. `marked` is row-major over `cols` by `rows`.
pub(crate) fn dilate(marked: &[bool], cols: usize, rows: usize, radius: usize) -> Vec<bool> {
    assert_eq!(marked.len(), cols * rows);

    let mut grown = vec![false; cols * rows];
    for row in 0..rows {
        for col in 0..cols {
            if !marked[row * cols + col] {
                continue;
            }

            let row_start = row.saturating_sub(radius);
            let row_end = (row + radius).min(rows - 1);
            let col_start = col.saturating_sub(radius);
            let col_end = (col + radius).min(cols - 1);
            for grown_row in row_start..=row_end {
                let start = grown_row * cols + col_start;
                let end = grown_row * cols + col_end;
                grown[start..=end].fill(true);
            }
        }
    }

    grown
}

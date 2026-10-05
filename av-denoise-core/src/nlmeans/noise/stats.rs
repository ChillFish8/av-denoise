/// Sorts `values` into ascending order in place.
///
/// [median] and [lower_quartile] both need sorted input, so a caller wanting both sorts once.
pub(super) fn sort_ascending(values: &mut [f32]) {
    values.sort_by(|left, right| left.partial_cmp(right).expect("noise stats are never NaN"));
}

/// The median of a sorted, non-empty slice.
///
/// An even count averages the two middle elements.
pub(super) fn median(values: &[f32]) -> f32 {
    let count = values.len();
    if count % 2 == 1 {
        values[count / 2]
    } else {
        (values[count / 2 - 1] + values[count / 2]) / 2.0
    }
}

/// The lower quartile of a sorted, non-empty slice.
///
/// It interpolates between the two neighbouring elements when the quarter point falls between them.
pub(super) fn lower_quartile(values: &[f32]) -> f32 {
    let count = values.len();
    if count == 1 {
        return values[0];
    }

    let position = 0.25 * (count - 1) as f32;
    let lower_index = position.floor() as usize;
    let upper_index = position.ceil() as usize;
    if lower_index == upper_index {
        values[lower_index]
    } else {
        let fraction = position - lower_index as f32;
        values[lower_index] + fraction * (values[upper_index] - values[lower_index])
    }
}

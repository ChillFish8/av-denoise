use cubecl::prelude::*;

use crate::collab::{MAX_K, PATCH_SIZE};

/// The floor applied to a variance or profile before inverting it, so zero noise stays finite.
const VARIANCE_FLOOR: f32 = 1.0e-20;

/// Hard-thresholds a group on the pooled energy of each coefficient and its frequency neighbours.
///
/// A coefficient's energy is its square over its own noise variance. Its pooled energy is the
/// mean over itself and its four neighbours at the same stack index, leaving out any past the
/// patch edge. It is kept when that mean reaches `threshold` squared. The spatial DC is never a
/// neighbour and is judged on its own against `dc_lambda`. The group DC is always kept.
///
/// Lane `sub` holds vertical frequency `sub` at every horizontal frequency, so the vertical
/// neighbours come from shuffles. Each row walks left to right holding only the energies of the
/// previous, current and next coefficient, which keeps register use low. Every lane of the 8-lane
/// group has to call this. Returns the noise variance this lane kept.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a per-lane value the threshold loop already holds"
)]
pub(crate) fn pooled_threshold(
    stack: &mut Array<f32>,
    v: &Array<f32>,
    dct_profile: &Array<f32>,
    prof_sub: f32,
    sub: u32,
    k_use: u32,
    threshold: f32,
    dc_lambda: f32,
) -> f32 {
    let dc_lane = sub == 0u32;
    let has_up = sub > 0u32;
    let has_down = sub < 7u32;

    // Lane 1's upper neighbour in column 0, and lane 0's left neighbour in column 1, are
    // both the spatial DC, so neither is counted.
    let up_count = select(has_up, 1.0f32, 0.0f32);
    let down_count = select(has_down, 1.0f32, 0.0f32);
    let vertical_count = 1.0f32 + up_count + down_count;
    let first_column_count = vertical_count + select(sub == 1u32, 0.0f32, 1.0f32);
    let second_column_count = vertical_count + select(dc_lane, 1.0f32, 2.0f32);

    let bar = threshold * threshold;
    let first_column_bar = bar * first_column_count;
    let second_column_bar = bar * second_column_count;
    let middle_column_bar = bar * (vertical_count + 2.0f32);
    let last_column_bar = bar * (vertical_count + 1.0f32);

    let mut retained = 0.0f32;
    #[unroll]
    for j in 0..MAX_K {
        let lane_variance = v[j as usize] * prof_sub;
        let lane_inverse = floored_reciprocal(lane_variance);
        let first_coeff = stack[(j * PATCH_SIZE) as usize];
        let first_energy = coefficient_energy(first_coeff, lane_inverse, dct_profile[0usize]);

        // The spatial DC's energy is zero while it stands in as a neighbour.
        let mut current = select(dc_lane, 0.0f32, first_energy);
        let mut previous = 0.0f32;
        let mut kept_profile = 0.0f32;

        #[unroll]
        for i in 0..PATCH_SIZE {
            let slot = (j * PATCH_SIZE + i) as usize;
            let profile = dct_profile[i as usize];
            let coeff = stack[slot];

            let mut next = 0.0f32;
            if comptime!(i < 7u32) {
                let next_coeff = stack[slot + 1];
                let next_profile = dct_profile[(i + 1u32) as usize];
                next = coefficient_energy(next_coeff, lane_inverse, next_profile);
            }

            // Neighbours past the top or bottom row read as zero, and the column bar leaves
            // them out of the count.
            let shuffled_up = plane_shuffle_up(current, 1u32);
            let shuffled_down = plane_shuffle_down(current, 1u32);
            let up = select(has_up, shuffled_up, 0.0f32);
            let down = select(has_down, shuffled_down, 0.0f32);

            let mut sum = current + down + up;
            if comptime!(i > 0u32) {
                sum += previous;
            }
            if comptime!(i < 7u32) {
                sum += next;
            }

            let column_bar = if comptime!(i == 0u32) {
                first_column_bar
            } else if comptime!(i == 1u32) {
                second_column_bar
            } else if comptime!(i == 7u32) {
                last_column_bar
            } else {
                middle_column_bar
            };

            // Multiplied out rather than divided, so an exact tie at the bar is not lost
            // to the GPU's inexact division.
            let mut keep = sum >= column_bar;

            if comptime!(i == 0u32) {
                let dc_variance = lane_variance * profile;
                let own_test = f32::abs(coeff) >= dc_lambda * f32::sqrt(dc_variance);
                keep = select(dc_lane, own_test, keep);

                if comptime!(j == 0u32) {
                    keep = keep || dc_lane;
                }
            }

            let live = j < k_use;
            let dropped = live && !keep;
            kept_profile += select(keep, profile, 0.0f32);
            stack[slot] = select(dropped, 0.0f32, coeff);

            previous = current;
            current = next;
        }

        if j < k_use {
            retained += kept_profile * lane_variance;
        }
    }

    retained
}

/// The reciprocal of a noise variance, floored so zero noise stays finite.
#[cube]
fn floored_reciprocal(variance: f32) -> f32 {
    let floored = f32::max(variance, VARIANCE_FLOOR);
    1.0f32 / floored
}

/// A coefficient's square over its noise variance, from the lane's variance reciprocal and the
/// column's profile.
#[cube]
fn coefficient_energy(coeff: f32, lane_inverse: f32, profile: f32) -> f32 {
    let profile_inverse = floored_reciprocal(profile);
    coeff * coeff * lane_inverse * profile_inverse
}

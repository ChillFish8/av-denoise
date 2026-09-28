use cubecl::prelude::*;

use crate::collab::{MAX_K, PATCH_AREA, PATCH_SIZE};

/// The smallest noise variance a coefficient's energy divides by, so zero noise stays finite.
const VARIANCE_FLOOR: f32 = 1.0e-20;

/// Hard-thresholds a group on the pooled energy of each coefficient and its frequency neighbours.
///
/// A coefficient's energy is its square over its own noise variance. Its pooled energy is the
/// mean over itself and its four neighbours at the same stack index, leaving out any past the
/// patch edge. It is kept when that mean reaches `threshold` squared. The spatial DC is never a
/// neighbour and is judged on its own against `dc_lambda`. The group DC is always kept.
///
/// Lane `sub` holds vertical frequency `sub` at every horizontal frequency, so the vertical
/// neighbours come from shuffles. Every lane of the 8-lane group has to call this. Returns the
/// noise variance this lane kept.
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
    base: u32,
    k_use: u32,
    threshold: f32,
    dc_lambda: f32,
) -> f32 {
    let dc_lane = sub == 0u32;

    let mut energy = Array::<f32>::new(PATCH_AREA as usize);
    #[unroll]
    for j in 0..MAX_K {
        #[unroll]
        for i in 0..PATCH_SIZE {
            let slot = (j * PATCH_SIZE + i) as usize;
            let variance = f32::max(v[j as usize] * dct_profile[i as usize] * prof_sub, VARIANCE_FLOOR);
            let coeff = stack[slot];
            let value = coeff * coeff / variance;
            if comptime!(i == 0u32) {
                energy[slot] = select(dc_lane, 0.0f32, value);
            } else {
                energy[slot] = value;
            }
        }
    }

    let up_lane = u32::max(sub, 1u32) - 1u32;
    let down_lane = u32::min(sub + 1u32, 7u32);
    let up_weight = select(sub > 0u32, 1.0f32, 0.0f32);
    let down_weight = select(sub < 7u32, 1.0f32, 0.0f32);

    // Lane 1's upper neighbour in column 0, and lane 0's left neighbour in column 1, are
    // both the spatial DC.
    let up_weight_first_column = select(sub == 1u32, 0.0f32, up_weight);
    let left_weight_second_column = select(dc_lane, 0.0f32, 1.0f32);
    let bar = threshold * threshold;

    let mut retained = 0.0f32;
    #[unroll]
    for j in 0..MAX_K {
        #[unroll]
        for i in 0..PATCH_SIZE {
            let slot = (j * PATCH_SIZE + i) as usize;
            let variance = v[j as usize] * dct_profile[i as usize] * prof_sub;
            let coeff = stack[slot];
            let own = coeff * coeff / f32::max(variance, VARIANCE_FLOOR);

            let up = plane_shuffle(energy[slot], base + up_lane);
            let down = plane_shuffle(energy[slot], base + down_lane);

            let mut sum = own + down * down_weight;
            let mut count = 1.0f32 + down_weight;
            if comptime!(i == 0u32) {
                sum += up * up_weight_first_column;
                count += up_weight_first_column;
            } else {
                sum += up * up_weight;
                count += up_weight;
            }

            if comptime!(i == 1u32) {
                sum += energy[slot - 1];
                count += left_weight_second_column;
            }
            if comptime!(i > 1u32) {
                sum += energy[slot - 1];
                count += 1.0f32;
            }
            if comptime!(i < 7u32) {
                sum += energy[slot + 1];
                count += 1.0f32;
            }

            let own_test = f32::abs(coeff) >= dc_lambda * f32::sqrt(variance);
            // Multiplied out rather than divided, so an exact tie at the bar is not lost
            // to the GPU's inexact division.
            let mut keep = sum >= bar * count;
            if comptime!(i == 0u32) {
                keep = select(dc_lane, own_test, keep);
            }
            if comptime!(j == 0u32 && i == 0u32) {
                keep = keep || dc_lane;
            }

            if j < k_use {
                if keep {
                    retained += variance;
                } else {
                    stack[slot] = 0.0f32;
                }
            }
        }
    }

    retained
}

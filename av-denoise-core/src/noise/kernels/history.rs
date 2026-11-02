use cubecl::prelude::*;

use crate::noise::reading::READING_WORDS;
use crate::noise::spatial::RESULTS_PER_SLOT;

/// Copies ring slot `slot`'s reading and Immerkær abs sums into history slot `history_slot`, in one
/// launch.
///
/// Every thread of the grid takes its grid-stride share of the words.
#[cube(launch_unchecked)]
pub fn noise_record(
    readings: &[f32],
    results: &[f32],
    records: &mut [f32],
    history_results: &mut [f32],
    slot: u32,
    history_slot: u32,
) {
    let stride = CUBE_COUNT_X * CUBE_DIM_X;
    let mut index = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let results_per_slot = comptime!(RESULTS_PER_SLOT as u32);

    while index < READING_WORDS {
        let source = slot * READING_WORDS + index;
        let target = history_slot * READING_WORDS + index;
        records[target as usize] = readings[source as usize];

        if index < results_per_slot {
            let source = slot * results_per_slot + index;
            let target = history_slot * results_per_slot + index;
            history_results[target as usize] = results[source as usize];
        }

        index += stride;
    }
}

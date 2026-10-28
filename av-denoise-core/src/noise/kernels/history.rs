use cubecl::prelude::*;

use crate::noise::reading::READING_WORDS;
use crate::noise::spatial::RESULTS_PER_SLOT;

const RESULT_WORDS: u32 = RESULTS_PER_SLOT as u32;

/// Copies ring slot `slot`'s reading, Immerkær abs sums and, with a map, its map into history slot
/// `history_slot`, in one launch.
///
/// Every thread of the grid takes its grid-stride share of the words. `map_len` 0 leaves both map
/// bindings untouched.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding, a slot or a comptime shape"
)]
pub fn noise_record(
    readings: &[f32],
    results: &[f32],
    slot_maps: &[f32],
    records: &mut [f32],
    history_results: &mut [f32],
    history_maps: &mut [f32],
    slot: u32,
    history_slot: u32,
    #[comptime] map_len: u32,
) {
    let stride = CUBE_COUNT_X * CUBE_DIM_X;
    let mut index = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let words = comptime!(READING_WORDS.max(map_len));

    while index < words {
        if index < READING_WORDS {
            let source = slot * READING_WORDS + index;
            let target = history_slot * READING_WORDS + index;
            records[target as usize] = readings[source as usize];
        }

        if index < RESULT_WORDS {
            let source = slot * RESULT_WORDS + index;
            let target = history_slot * RESULT_WORDS + index;
            history_results[target as usize] = results[source as usize];
        }

        if index < map_len {
            let source = slot * map_len + index;
            let target = history_slot * map_len + index;
            history_maps[target as usize] = slot_maps[source as usize];
        }

        index += stride;
    }
}

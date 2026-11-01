use cubecl::prelude::*;

use super::{
    META_COUNT,
    META_MEDIAN_ENTRY,
    META_MEDIAN_X,
    META_MEDIAN_Y,
    META_OWN_X,
    META_OWN_Y,
    NEIGHBOURHOOD,
    PassBlock,
    STAGED_FIELD_WORDS,
    STAGED_WORDS,
    on_grid,
};
use crate::motion::kernels::{BlockVector, model_displacement, predicts_motion};

/// Loads the 3x3 neighbourhoods of `mv_in` and, with `has_seeds`, `seeds` around the block into
/// `staged`, one word per thread, with both fields starting `mv_base` values in.
///
/// The field's block at offset `(dx, dy)` lands at words `2 * (3 * (dy + 1) + dx + 1)` onward, and
/// the seeds follow [STAGED_FIELD_WORDS] words later. Off-grid blocks hold 0. Every thread must
/// call it.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the staging reads"
)]
pub(crate) fn stage_neighbourhood(
    mv_in: &[i32],
    seeds: &[i32],
    mv_base: u32,
    staged: &mut Shared<[i32]>,
    block: &PassBlock,
    #[comptime] has_seeds: bool,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
) {
    let words = comptime!(
        if has_seeds {
            STAGED_WORDS
        } else {
            STAGED_FIELD_WORDS
        }
    );
    let mut word = block.thread_id;
    while word < words {
        let reads_seeds = word >= STAGED_FIELD_WORDS;
        let field_word = word % STAGED_FIELD_WORDS;
        let position = field_word / 2u32;
        let component = field_word % 2u32;
        let col = block.block_col as i32 + (position % 3u32) as i32 - 1i32;
        let row = block.block_row as i32 + (position / 3u32) as i32 - 1i32;

        let mut value = 0i32;
        if on_grid(col, row, blocks_x, blocks_y) {
            let index = (mv_base + (row as u32 * blocks_x + col as u32) * 2u32 + component) as usize;
            if reads_seeds {
                value = seeds[index];
            } else {
                value = mv_in[index];
            }
        }

        staged[word as usize] = value;
        word += block.threads;
    }
}

/// The staged vector of the block at offset `(dx, dy)`, each between -1 and 1, in the field that
/// starts `offset` words into `staged`.
#[cube]
fn staged_vector(staged: &Shared<[i32]>, offset: u32, dx: i32, dy: i32) -> BlockVector {
    let position = (3i32 * (dy + 1i32) + dx + 1i32) as u32;
    let index = (offset + position * 2u32) as usize;

    BlockVector {
        dx: staged[index],
        dy: staged[index + 1],
    }
}

/// Appends `vector` to the first `count` entries of `list` unless one already holds it.
///
/// Returns the list's new length.
#[cube]
pub(crate) fn push_unique(list: &mut Shared<[i32]>, count: u32, vector: BlockVector) -> u32 {
    let mut seen = false;
    let mut entry = 0u32;
    while entry < count {
        let same_x = list[(entry * 2u32) as usize] == vector.dx;
        let same_y = list[(entry * 2u32 + 1u32) as usize] == vector.dy;
        if same_x && same_y {
            seen = true;
        }

        entry += 1u32;
    }

    let mut next = count;
    if !seen {
        list[(count * 2u32) as usize] = vector.dx;
        list[(count * 2u32 + 1u32) as usize] = vector.dy;
        next = count + 1u32;
    }

    next
}

/// The 3x3 position of neighbour `slot` of the eight around a block, skipping the centre.
#[cube]
fn neighbour_position(slot: u32) -> u32 {
    let past_centre = slot >= 4u32;
    slot + select(past_centre, 1u32, 0u32)
}

/// Writes the component-wise lower median of the block's staged 3x3 neighbours to `meta`, or the
/// block's own vector without any.
///
/// One thread per component and neighbour counts the neighbours below and up to its value, and
/// the one whose rank holds the lower median writes it. Equal values write the same word. Every
/// thread must call it, and a barrier must follow before the median is read.
#[cube]
pub(crate) fn neighbourhood_median(
    staged: &Shared<[i32]>,
    meta: &mut Shared<[i32]>,
    block: &PassBlock,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
) {
    let mut neighbours = 0u32;
    for slot in 0..NEIGHBOURHOOD {
        let position = neighbour_position(slot);
        let col = block.block_col as i32 + (position % 3u32) as i32 - 1i32;
        let row = block.block_row as i32 + (position / 3u32) as i32 - 1i32;
        if on_grid(col, row, blocks_x, blocks_y) {
            neighbours += 1u32;
        }
    }

    if neighbours == 0u32 {
        if block.thread_id == 0u32 {
            meta[META_MEDIAN_X] = staged[8];
            meta[META_MEDIAN_Y] = staged[9];
        }
    } else {
        let median_rank = (neighbours - 1u32) / 2u32;
        let items = comptime!(2 * NEIGHBOURHOOD);
        let mut item = block.thread_id;
        while item < items {
            let component = item / NEIGHBOURHOOD;
            let position = neighbour_position(item % NEIGHBOURHOOD);
            let col = block.block_col as i32 + (position % 3u32) as i32 - 1i32;
            let row = block.block_row as i32 + (position / 3u32) as i32 - 1i32;

            if on_grid(col, row, blocks_x, blocks_y) {
                let value = staged[(position * 2u32 + component) as usize];
                let mut below = 0u32;
                let mut up_to = 0u32;
                for slot in 0..NEIGHBOURHOOD {
                    let other_position = neighbour_position(slot);
                    let other_col = block.block_col as i32 + (other_position % 3u32) as i32 - 1i32;
                    let other_row = block.block_row as i32 + (other_position / 3u32) as i32 - 1i32;
                    if on_grid(other_col, other_row, blocks_x, blocks_y) {
                        let other = staged[(other_position * 2u32 + component) as usize];
                        below += select(other < value, 1u32, 0u32);
                        up_to += select(other <= value, 1u32, 0u32);
                    }
                }

                if below <= median_rank && median_rank < up_to {
                    meta[META_MEDIAN_X + component as usize] = value;
                }
            }

            item += block.threads;
        }
    }
}

/// Writes the block's distinct candidate vectors to `list`, with the median
/// [neighbourhood_median] left in `meta` among them.
///
/// The candidates come in order of preference. They are its own vector, its neighbourhood median,
/// its left, right, upper and lower neighbours' vectors, and zero. With `model_candidate`, the
/// camera model's prediction follows, when the model moves anything. With `has_seeds`, the block's
/// seed and the seeds of its four diagonal blocks come last. Off-grid blocks add nothing. A vector
/// already in the list is skipped, so each is scored once. The field and seeds come from `staged`,
/// as [stage_neighbourhood] lays them out. Run by one thread.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the gather reads"
)]
pub(crate) fn gather_candidates(
    staged: &Shared<[i32]>,
    predictions: &[f32],
    list: &mut Shared<[i32]>,
    meta: &mut Shared<[i32]>,
    block: &PassBlock,
    #[comptime] has_seeds: bool,
    #[comptime] model_candidate: bool,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
) {
    let block_col = block.block_col;
    let block_row = block.block_row;
    let own = staged_vector(staged, 0u32, 0i32, 0i32);
    let own_x = own.dx;
    let own_y = own.dy;
    let median_x = meta[META_MEDIAN_X];
    let median_y = meta[META_MEDIAN_Y];

    let mut count = 0u32;
    let own_vector = BlockVector { dx: own_x, dy: own_y };
    count = push_unique(list, count, own_vector);
    let median_vector = BlockVector {
        dx: median_x,
        dy: median_y,
    };
    count = push_unique(list, count, median_vector);

    // Left, right, up, down.
    let mut side = 0u32;
    while side < 4u32 {
        let mut offset_x = 0i32;
        let mut offset_y = 0i32;
        if side == 0u32 {
            offset_x = -1i32;
        } else if side == 1u32 {
            offset_x = 1i32;
        } else if side == 2u32 {
            offset_y = -1i32;
        } else {
            offset_y = 1i32;
        }

        let neighbour_col = block_col as i32 + offset_x;
        let neighbour_row = block_row as i32 + offset_y;
        if on_grid(neighbour_col, neighbour_row, blocks_x, blocks_y) {
            let vector = staged_vector(staged, 0u32, offset_x, offset_y);
            count = push_unique(list, count, vector);
        }

        side += 1u32;
    }

    let zero = BlockVector { dx: 0i32, dy: 0i32 };
    count = push_unique(list, count, zero);

    if model_candidate {
        let neighbour_index = block.neighbour_index;
        if predicts_motion(predictions, neighbour_index) {
            let model = model_displacement(
                predictions,
                neighbour_index,
                block.origin_x,
                block.origin_y,
                blksize,
                1u32,
                width,
                height,
            );
            count = push_unique(list, count, model);
        }
    }

    if has_seeds {
        let seed = staged_vector(staged, STAGED_FIELD_WORDS, 0i32, 0i32);
        count = push_unique(list, count, seed);

        let mut corner = 0u32;
        while corner < 4u32 {
            let offset_col = (corner % 2u32) as i32 * 2i32 - 1i32;
            let offset_row = (corner / 2u32) as i32 * 2i32 - 1i32;
            let corner_col = block_col as i32 + offset_col;
            let corner_row = block_row as i32 + offset_row;
            if on_grid(corner_col, corner_row, blocks_x, blocks_y) {
                let corner_seed = staged_vector(staged, STAGED_FIELD_WORDS, offset_col, offset_row);
                count = push_unique(list, count, corner_seed);
            }

            corner += 1u32;
        }
    }

    // The median is always in the list, either where it was pushed or where it repeated an entry.
    let mut median_entry = 0u32;
    let mut entry = count;
    while entry > 0u32 {
        entry -= 1u32;
        let same_x = list[(entry * 2u32) as usize] == median_x;
        let same_y = list[(entry * 2u32 + 1u32) as usize] == median_y;
        if same_x && same_y {
            median_entry = entry;
        }
    }

    meta[META_COUNT] = count as i32;
    meta[META_MEDIAN_ENTRY] = median_entry as i32;
    meta[META_OWN_X] = own_x;
    meta[META_OWN_Y] = own_y;
}

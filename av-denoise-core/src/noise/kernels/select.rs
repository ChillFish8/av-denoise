use cubecl::prelude::*;

use crate::noise::select::{
    MAX_QUERIES,
    SELECT_BINS,
    SELECT_BINS_PER_THREAD,
    SELECT_CUBE_THREADS,
    SELECT_DIGIT_BITS,
    SELECT_GROUP_THREADS,
    SELECT_QUERY_WORDS,
    SELECT_SHIFTS,
    SELECT_SLOT_WORDS,
    SELECT_SLOTS,
    SELECT_TAG_WORDS,
    SLOT_EMPTY,
    SLOT_STOPPED,
    SLOT_WRITTEN,
    UNTAGGED,
};

const SHIFT_0: u32 = SELECT_SHIFTS[0];
const SHIFT_1: u32 = SELECT_SHIFTS[1];
const SHIFT_2: u32 = SELECT_SHIFTS[2];
const MASK_1: u32 = (1 << SELECT_DIGIT_BITS[1]) - 1;
const MASK_2: u32 = (1 << SELECT_DIGIT_BITS[2]) - 1;

/// The order key of `value`, an unsigned integer that sorts as the float does.
///
/// A positive float gets its sign bit set, which lifts it above every negative one. A negative float
/// has every bit flipped, which reverses its order.
#[cube]
pub(crate) fn order_key(value: f32) -> u32 {
    let bits = u32::reinterpret(value);
    let negative = bits >= 0x8000_0000u32;

    select(negative, bits ^ u32::MAX, bits ^ 0x8000_0000u32)
}

/// The float whose order key is `key`.
#[cube]
pub(crate) fn key_value(key: u32) -> f32 {
    let positive = key >= 0x8000_0000u32;
    let bits = select(positive, key ^ 0x8000_0000u32, key ^ u32::MAX);

    f32::reinterpret(bits)
}

/// The 0-based rank `rule` picks from `count` elements, mirroring
/// [RankRule::rank](crate::noise::select::RankRule::rank).
#[cube]
fn rule_rank(rule: u32, count: u32) -> u32 {
    let last = u32::max(count, 1u32) - 1;

    let mut rank = 0u32;
    rank = select(rule == 1u32, last, rank);
    rank = select(rule == 2u32, last / 2, rank);
    rank = select(rule == 3u32, count / 2, rank);
    rank = select(rule == 4u32, last / 4, rank);
    select(rule == 5u32, (count + 2) / 4, rank)
}

/// Resolves one digit of every wanted query, leaving `[prefix, remaining rank, count, written]` per
/// query in `out`.
///
/// `row_pass` 0 counts the tag's pass-0 row and takes the rank from the query's rule. A tag below its
/// minimum count is left stopped. A later `row_pass` starts from the slot published after the
/// previous digit and walks that query's own row. An empty or stopped slot is copied without a walk.
///
/// `rows` holds the batch's histogram rows from row `first_row` of its layout on.
///
/// The cube splits into groups of 32 threads, and each group walks one query's row per round. Every
/// thread of the cube must call this with the same arguments.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding or a comptime shape"
)]
pub(crate) fn resolve_queries(
    rows: &Array<i32>,
    first_row: u32,
    state: &Array<u32>,
    queries: &Array<u32>,
    tag_ranges: &Array<u32>,
    wanted: &SharedMemory<u32>,
    wanted_count: u32,
    row_pass: u32,
    out: &mut SharedMemory<u32>,
    #[comptime] tag_count: u32,
    #[comptime] query_count: u32,
    #[comptime] cube_threads: u32,
) {
    comptime! {
        assert!(
            cube_threads.is_multiple_of(SELECT_GROUP_THREADS),
            "a resolving cube splits into whole groups"
        );
    }

    let groups = comptime!(cube_threads / SELECT_GROUP_THREADS);
    let mut sums = SharedMemory::<u32>::new(cube_threads as usize);

    let thread = UNIT_POS_X;
    let group = thread / SELECT_GROUP_THREADS;
    let lane = thread % SELECT_GROUP_THREADS;
    let group_base = group * SELECT_GROUP_THREADS;
    let rounds = wanted_count.div_ceil(groups);

    // A counting walk uses no slot value. Its reads point at slot 1, which no counting launch writes.
    let counting = row_pass == 0;
    let later_slot = u32::max(row_pass, 1u32) - 1;
    let previous_slot = select(counting, 1u32, later_slot);
    let later_rows = tag_count + later_slot * query_count;
    let later_shift = select(row_pass == 1u32, SHIFT_1, SHIFT_2);
    let shift = select(counting, SHIFT_0, later_shift);

    for round in 0..rounds {
        let position = round * groups + group;
        let active = position < wanted_count;
        let pick = select(active, position, 0u32);
        let query = wanted[pick as usize];
        let tag = queries[(query * SELECT_QUERY_WORDS) as usize];
        let rule = queries[(query * SELECT_QUERY_WORDS + 1) as usize];
        let min_count = tag_ranges[(tag * SELECT_TAG_WORDS + 2) as usize];

        let slot_base = (query * SELECT_SLOTS + previous_slot) * SELECT_SLOT_WORDS;
        let slot_prefix = state[slot_base as usize];
        let slot_remaining = state[(slot_base + 1) as usize];
        let slot_count = state[(slot_base + 2) as usize];
        let slot_written = state[(slot_base + 3) as usize];

        let prefix = select(counting, 0u32, slot_prefix);
        let walk = counting || slot_written == SLOT_WRITTEN;
        let row = select(counting, tag, later_rows + query) - first_row;
        let bin_start = row * SELECT_BINS + lane * SELECT_BINS_PER_THREAD;

        let mut lane_sum = 0u32;
        if walk {
            for bin in 0..SELECT_BINS_PER_THREAD {
                let value = rows[(bin_start + bin) as usize];
                lane_sum += u32::cast_from(value);
            }
        }

        sums[thread as usize] = lane_sum;

        sync_cube();

        let mut below = 0u32;
        let mut total = 0u32;
        for other in 0..SELECT_GROUP_THREADS {
            let value = sums[(group_base + other) as usize];
            below += select(other < lane, value, 0u32);
            total += value;
        }

        let rank = rule_rank(rule, total);
        let counted_flag = select(total < min_count, SLOT_STOPPED, SLOT_WRITTEN);
        let first_flag = select(total == 0u32, SLOT_EMPTY, counted_flag);

        let count = select(counting, total, slot_count);
        let remaining = select(counting, rank, slot_remaining);
        let written = select(counting, first_flag, slot_written);

        let found = written == SLOT_WRITTEN && remaining < total;
        let in_lane = remaining >= below && remaining < below + lane_sum;
        let out_base = query * SELECT_SLOT_WORDS;

        if active && found && in_lane {
            let mut skipped = below;
            let mut digit = 0u32;
            let mut rest = 0u32;
            let mut done = false;

            for bin in 0..SELECT_BINS_PER_THREAD {
                let value = u32::cast_from(rows[(bin_start + bin) as usize]);
                let hit = !done && remaining < skipped + value;

                let bin_digit = lane * SELECT_BINS_PER_THREAD + bin;
                let passed = u32::min(remaining, skipped);
                let bin_rest = remaining - passed;

                digit = select(hit, bin_digit, digit);
                rest = select(hit, bin_rest, rest);
                done = done || hit;
                skipped += value;
            }

            out[out_base as usize] = prefix | (digit << shift);
            out[(out_base + 1) as usize] = rest;
            out[(out_base + 2) as usize] = count;
            out[(out_base + 3) as usize] = SLOT_WRITTEN;
        }

        if active && !found && lane == 0 {
            let flag = select(written == SLOT_WRITTEN, SLOT_EMPTY, written);
            let kept = select(flag == SLOT_STOPPED, count, 0u32);

            out[out_base as usize] = 0u32;
            out[(out_base + 1) as usize] = 0u32;
            out[(out_base + 2) as usize] = kept;
            out[(out_base + 3) as usize] = flag;
        }

        sync_cube();
    }
}

/// The value and count of `query` after [resolve_queries] has resolved its last digit.
///
/// An empty set gives 0.0 and 0. A tag below its minimum count gives 0.0 and its count.
#[cube]
pub(crate) fn select_result(out: &SharedMemory<u32>, query: u32) -> (f32, u32) {
    let base = query * SELECT_SLOT_WORDS;
    let key = out[base as usize];
    let count = out[(base + 2) as usize];
    let written = out[(base + 3) as usize];
    let resolved = written == SLOT_WRITTEN;
    let value = key_value(key);

    (select(resolved, value, 0.0f32), count)
}

/// Zeroes a batch's histogram and empties its state.
///
/// Each calling thread clears every `stride`-th word from `first` on, so the callers' `first`
/// values must cover `0..stride` once each.
#[cube]
pub(crate) fn clear_batch(
    histogram: &mut Array<i32>,
    state: &mut Array<u32>,
    first: u32,
    stride: u32,
    #[comptime] tag_count: u32,
    #[comptime] query_count: u32,
) {
    let histogram_words = comptime!((tag_count + 2 * query_count) * SELECT_BINS);
    let state_words = comptime!(query_count * SELECT_SLOTS * SELECT_SLOT_WORDS);

    let mut word = first;
    while word < histogram_words {
        histogram[word as usize] = 0i32;
        word += stride;
    }

    let mut word = first;
    while word < state_words {
        state[word as usize] = SLOT_EMPTY;
        word += stride;
    }
}

/// One radix pass of a select batch over 256 elements per cube.
///
/// Pass 0 adds each tagged element to its tag's row. Passes 1 and 2 first resolve, from the previous
/// pass's rows, every query whose tag occurs in the cube and publish the result to the state. Each
/// element then adds its next digit to the row of each query of its tag whose prefix it matches.
/// Threads past `len` take part in every barrier and resolution and skip only their own element.
///
/// `rows_in` holds the previous pass's rows, read with plain loads, and `rows_out` this pass's rows.
/// Pass 0 never reads `rows_in`.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding or a comptime shape"
)]
pub fn noise_select_histogram(
    keys: &Array<u32>,
    tags: &Array<u32>,
    queries: &Array<u32>,
    tag_ranges: &Array<u32>,
    rows_in: &Array<i32>,
    rows_out: &mut Array<Atomic<i32>>,
    state: &mut Array<u32>,
    pass: u32,
    #[comptime] len: u32,
    #[comptime] tag_count: u32,
    #[comptime] query_count: u32,
) {
    comptime! {
        assert!(tag_count <= SELECT_CUBE_THREADS, "every tag needs a flag thread");
        assert!(query_count <= MAX_QUERIES, "too many queries for one batch");
    }

    let thread = UNIT_POS_X;
    let index = CUBE_POS_X * SELECT_CUBE_THREADS + thread;

    let in_range = index < len;
    let element = select(in_range, index, 0u32);
    let tag = select(in_range, tags[element as usize], UNTAGGED);
    let key = keys[element as usize];
    let tagged = tag < tag_count;

    if pass == 0 {
        if tagged {
            let bin = key >> SHIFT_0;
            let slot = tag * SELECT_BINS + bin;
            Atomic::fetch_add(&rows_out[slot as usize], 1i32);
        }
    } else {
        let mut present = SharedMemory::<u32>::new(tag_count as usize);
        let mut wanted = SharedMemory::<u32>::new(comptime!(query_count + 1) as usize);
        let mut out = SharedMemory::<u32>::new(comptime!(query_count * SELECT_SLOT_WORDS) as usize);

        if thread < tag_count {
            present[thread as usize] = 0u32;
        }

        sync_cube();

        if tagged {
            present[tag as usize] = 1u32;
        }

        sync_cube();

        if thread == 0 {
            let mut found = 0u32;

            for query in 0..query_count {
                let query_tag = queries[(query * SELECT_QUERY_WORDS) as usize];
                if present[query_tag as usize] == 1u32 {
                    wanted[found as usize] = query;
                    found += 1;
                }
            }

            wanted[query_count as usize] = found;
        }

        sync_cube();

        let wanted_count = wanted[query_count as usize];
        let row_pass = pass - 1;
        let first_row = select(row_pass == 0u32, 0u32, tag_count);

        resolve_queries(
            rows_in,
            first_row,
            state,
            queries,
            tag_ranges,
            &wanted,
            wanted_count,
            row_pass,
            &mut out,
            tag_count,
            query_count,
            SELECT_CUBE_THREADS,
        );

        if thread < wanted_count * SELECT_SLOT_WORDS {
            let query = wanted[(thread / SELECT_SLOT_WORDS) as usize];
            let part = thread % SELECT_SLOT_WORDS;
            let slot_base = (query * SELECT_SLOTS + row_pass) * SELECT_SLOT_WORDS;
            let source = query * SELECT_SLOT_WORDS + part;
            state[(slot_base + part) as usize] = out[source as usize];
        }

        if tagged {
            let first = tag_ranges[(tag * SELECT_TAG_WORDS) as usize];
            let tag_queries = tag_ranges[(tag * SELECT_TAG_WORDS + 1) as usize];

            let second = pass == 2;
            let prefix_shift = select(second, SHIFT_1, SHIFT_0);
            let shift = select(second, SHIFT_2, SHIFT_1);
            let mask = select(second, MASK_2, MASK_1);

            let digit = (key >> shift) & mask;
            let key_prefix = key >> prefix_shift;

            for query in first..first + tag_queries {
                let base = query * SELECT_SLOT_WORDS;
                let prefix = out[base as usize] >> prefix_shift;
                let written = out[(base + 3) as usize];

                if written == SLOT_WRITTEN && prefix == key_prefix {
                    let slot = query * SELECT_BINS + digit;
                    Atomic::fetch_add(&rows_out[slot as usize], 1i32);
                }
            }
        }
    }
}

/// Writes every query's value and count from a queued batch, then clears the batch.
///
/// One cube of `cube_threads` threads resolves every query's last digit.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding or a comptime shape"
)]
pub fn noise_select_resolve(
    histogram: &mut Array<i32>,
    state: &mut Array<u32>,
    queries: &Array<u32>,
    tag_ranges: &Array<u32>,
    values: &mut Array<f32>,
    counts: &mut Array<u32>,
    #[comptime] tag_count: u32,
    #[comptime] query_count: u32,
    #[comptime] cube_threads: u32,
) {
    comptime! {
        assert!(query_count <= cube_threads, "every query needs a result thread");
    }

    let thread = UNIT_POS_X;
    let mut wanted = SharedMemory::<u32>::new(query_count as usize);
    let mut out = SharedMemory::<u32>::new(comptime!(query_count * SELECT_SLOT_WORDS) as usize);

    if thread < query_count {
        wanted[thread as usize] = thread;
    }

    sync_cube();

    resolve_queries(
        histogram,
        0u32,
        state,
        queries,
        tag_ranges,
        &wanted,
        query_count,
        2u32,
        &mut out,
        tag_count,
        query_count,
        cube_threads,
    );

    if thread < query_count {
        let (value, count) = select_result(&out, thread);
        values[thread as usize] = value;
        counts[thread as usize] = count;
    }

    clear_batch(histogram, state, thread, cube_threads, tag_count, query_count);
}

use std::collections::BTreeSet;

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::MAX_GRID_1D;
use crate::noise::kernels::{key_value, noise_select_resolve, order_key};
use crate::noise::select::{
    MAX_QUERIES,
    RankRule,
    SELECT_BINS,
    SELECT_CUBE_THREADS,
    SELECT_QUERY_WORDS,
    SELECT_SHIFTS,
    SELECT_SLOT_WORDS,
    SELECT_SLOTS,
    SELECT_TAG_WORDS,
    SLOT_STOPPED,
    SLOT_WRITTEN,
    SelectBatch,
    SelectQuery,
    UNTAGGED,
    key_value_host,
    order_key_host,
    resolve_cube_threads,
};
use crate::noise::stats::{lower_quartile, median};
use crate::tune::tests::fixtures::upload;
use crate::tune::tests::runtime::test_client;

const ALL_RULES: [RankRule; 6] = [
    RankRule::First,
    RankRule::Last,
    RankRule::MedianLow,
    RankRule::MedianHigh,
    RankRule::QuartileLow,
    RankRule::QuartileHigh,
];

const PERCENTILE_RULES: [RankRule; 4] = [
    RankRule::MedianLow,
    RankRule::MedianHigh,
    RankRule::QuartileLow,
    RankRule::QuartileHigh,
];

const MEDIAN_RULES: [RankRule; 2] = [RankRule::MedianLow, RankRule::MedianHigh];

#[cube(launch_unchecked)]
fn order_key_round_trip(values: &[f32], keys: &mut [u32], back: &mut [f32]) {
    let index = UNIT_POS_X as usize;
    let key = order_key(values[index]);

    keys[index] = key;
    back[index] = key_value(key);
}

/// Every rule in `rules` for every tag below `tag_count`, grouped by tag.
fn queries_for(tag_count: u32, rules: &[RankRule]) -> Vec<SelectQuery> {
    let mut queries = Vec::new();

    for tag in 0..tag_count {
        for &rule in rules {
            queries.push(SelectQuery { tag, rule });
        }
    }

    queries
}

/// A value between 0 and 1 from a hash of `seed` and `index`.
fn hashed_unit(seed: u32, index: u32) -> f32 {
    let mut hash = index.wrapping_mul(0x9E37_79B9) ^ seed.wrapping_mul(0x85EB_CA6B);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x2C1B_3C6D);
    hash ^= hash >> 12;
    hash = hash.wrapping_mul(0x297A_2D39);
    hash ^= hash >> 15;

    (hash >> 8) as f32 / (1u32 << 24) as f32
}

fn batch_from_values(
    client: &Client,
    values: &[f32],
    tags: &[u32],
    min_counts: &[u32],
    queries: &[SelectQuery],
) -> SelectBatch {
    let keys: Vec<u32> = values.iter().copied().map(order_key_host).collect();

    SelectBatch::from_elements(client, &keys, tags, min_counts, queries)
}

fn read_u32(client: &Client, handle: &Handle) -> Vec<u32> {
    let bytes = client.read_one(handle.clone()).expect("readback failed");
    u32::from_bytes(&bytes).to_vec()
}

fn read_i32(client: &Client, handle: &Handle) -> Vec<i32> {
    let bytes = client.read_one(handle.clone()).expect("readback failed");
    i32::from_bytes(&bytes).to_vec()
}

fn read_f32(client: &Client, handle: &Handle) -> Vec<f32> {
    let bytes = client.read_one(handle.clone()).expect("readback failed");
    f32::from_bytes(&bytes).to_vec()
}

/// Resolves every query of a queued batch with one cube of `cube_threads` threads.
fn resolve(client: &Client, batch: &SelectBatch, cube_threads: u32) -> (Vec<f32>, Vec<u32>) {
    let query_count = batch.query_count();
    let values = upload(client, &vec![-1.0; query_count as usize]);
    let counts = upload(client, &vec![-1.0; query_count as usize]);
    let histogram_len = batch.histogram_len() as usize;
    let state_len = batch.state_len() as usize;
    let query_words = (query_count * SELECT_QUERY_WORDS) as usize;
    let tag_words = (batch.tag_count() * SELECT_TAG_WORDS) as usize;

    unsafe {
        noise_select_resolve::launch_unchecked(
            client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(cube_threads),
            BufferArg::from_raw_parts(batch.histogram().clone(), histogram_len),
            BufferArg::from_raw_parts(batch.state().clone(), state_len),
            BufferArg::from_raw_parts(batch.query_table().clone(), query_words),
            BufferArg::from_raw_parts(batch.tag_table().clone(), tag_words),
            BufferArg::from_raw_parts(values.clone(), query_count as usize),
            BufferArg::from_raw_parts(counts.clone(), query_count as usize),
            batch.tag_count(),
            query_count,
            cube_threads,
        );
    }

    let values = read_f32(client, &values);
    let counts = read_u32(client, &counts);

    (values, counts)
}

/// The sorted values tagged `tag`.
fn sorted_set(values: &[f32], tags: &[u32], tag: u32) -> Vec<f32> {
    let mut set: Vec<f32> = values
        .iter()
        .zip(tags)
        .filter(|(_, element_tag)| **element_tag == tag)
        .map(|(value, _)| *value)
        .collect();
    set.sort_by(f32::total_cmp);

    set
}

/// The value and count the host sort gives `query`, or 0.0 and 0 for an empty set.
fn host_select(values: &[f32], tags: &[u32], query: SelectQuery) -> (f32, u32) {
    let set = sorted_set(values, tags, query.tag);
    let count = set.len() as u32;
    if count == 0 {
        return (0.0, 0);
    }

    let rank = query.rule.rank(count);
    (set[rank as usize], count)
}

/// Tag 1 is left empty, so `residue` 1 is untagged.
fn select_tag(residue: u32) -> u32 {
    if residue == 1 { UNTAGGED } else { residue }
}

/// Queues and resolves a batch, and checks every query against the host sort bit for bit.
fn assert_matches_host(
    client: &Client,
    values: &[f32],
    tags: &[u32],
    queries: &[SelectQuery],
    cube_threads: u32,
) {
    let tag_count = queries.iter().map(|query| query.tag).max().unwrap_or(0) + 1;
    let min_counts = vec![0u32; tag_count as usize];
    let batch = batch_from_values(client, values, tags, &min_counts, queries);
    batch.queue(client);

    let (gpu_values, gpu_counts) = resolve(client, &batch, cube_threads);

    for (index, &query) in queries.iter().enumerate() {
        let (host_value, host_count) = host_select(values, tags, query);
        let gpu_value = gpu_values[index];
        let gpu_count = gpu_counts[index];
        let gpu_bits = gpu_value.to_bits();
        let host_bits = host_value.to_bits();

        assert_eq!(gpu_count, host_count, "count of {query:?}");
        assert_eq!(
            gpu_bits, host_bits,
            "{query:?}: gpu {gpu_value} host {host_value}"
        );
    }
}

#[test]
fn rank_rules_pick_the_host_median_and_quartile_neighbours() {
    for count in 1..=9u32 {
        let sorted: Vec<f32> = (0..count).map(|index| (index * index) as f32 + 0.5).collect();
        let median_low = RankRule::MedianLow.rank(count) as usize;
        let median_high = RankRule::MedianHigh.rank(count) as usize;
        let median_pair = (sorted[median_low] + sorted[median_high]) * 0.5;
        let host_median = median(&sorted);
        let pair_bits = median_pair.to_bits();
        let host_median_bits = host_median.to_bits();
        assert_eq!(pair_bits, host_median_bits, "median of {count}");

        let position = 0.25 * (count - 1) as f32;
        let floor = position.floor() as u32;
        let ceiling = position.ceil() as u32;
        let quartile_low = RankRule::QuartileLow.rank(count);
        let quartile_high = RankRule::QuartileHigh.rank(count);
        assert_eq!(quartile_low, floor, "quartile floor of {count}");
        assert_eq!(quartile_high, ceiling, "quartile ceiling of {count}");

        let low = sorted[quartile_low as usize];
        let high = sorted[quartile_high as usize];
        let fraction = ((count - 1) % 4) as f32 * 0.25;
        let quartile = low + fraction * (high - low);
        let host_quartile = lower_quartile(&sorted);
        let quartile_bits = quartile.to_bits();
        let host_quartile_bits = host_quartile.to_bits();
        assert_eq!(quartile_bits, host_quartile_bits, "quartile of {count}");

        let first = RankRule::First.rank(count);
        let last = RankRule::Last.rank(count);
        let last_index = count - 1;
        assert_eq!(first, 0);
        assert_eq!(last, last_index);
    }
}

#[test]
fn order_keys_sort_like_the_floats() {
    let client = test_client();
    let values = [-1.0f32, -0.0, 0.0, 1.0e-40, 0.3 / 255.0, 0.25, 1.0, 3.0e38];
    let count = values.len();
    let input = upload(&client, &values);
    let keys = upload(&client, &vec![0.0; count]);
    let back = upload(&client, &vec![0.0; count]);

    unsafe {
        order_key_round_trip::launch_unchecked(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(count as u32),
            BufferArg::from_raw_parts(input.clone(), count),
            BufferArg::from_raw_parts(keys.clone(), count),
            BufferArg::from_raw_parts(back.clone(), count),
        );
    }

    let keys = read_u32(&client, &keys);
    let back = read_f32(&client, &back);

    let ascending = keys.windows(2).all(|pair| pair[0] < pair[1]);
    assert!(ascending, "keys do not ascend with the floats: {keys:x?}");

    for (index, &value) in values.iter().enumerate() {
        let host_key = order_key_host(value);
        let round_trip = back[index];
        let host_round_trip = key_value_host(host_key);
        let gpu_key = keys[index];
        let value_bits = value.to_bits();
        let round_trip_bits = round_trip.to_bits();
        let host_round_trip_bits = host_round_trip.to_bits();
        assert_eq!(gpu_key, host_key, "key of {value}");
        assert_eq!(round_trip_bits, value_bits, "round trip of {value}");
        assert_eq!(host_round_trip_bits, value_bits, "host round trip of {value}");
    }
}

#[test]
fn every_rank_rule_matches_the_host_sort() {
    let client = test_client();
    let len = 10_000u32;
    let values: Vec<f32> = (0..len).map(|index| hashed_unit(3, index)).collect();
    let tags: Vec<u32> = (0..len)
        .map(|index| match index % 6 {
            5 => UNTAGGED,
            tag => tag,
        })
        .collect();
    let queries = queries_for(5, &ALL_RULES);

    assert_matches_host(&client, &values, &tags, &queries, SELECT_CUBE_THREADS);

    let threads = resolve_cube_threads(&client);
    assert_matches_host(&client, &values, &tags, &queries, threads);
}

#[test]
fn an_all_tied_set_selects_the_tied_value() {
    let client = test_client();
    let tied_values = [0.25f32, 0.5, 0.125];
    let mut values = vec![0.25f32; 5000];
    let mut tags = vec![0u32; 5000];

    for index in 0..700usize {
        let tied = tied_values[index % tied_values.len()];
        values.push(tied);
        tags.push(1);
    }

    let queries = queries_for(2, &ALL_RULES);
    let threads = resolve_cube_threads(&client);

    assert_matches_host(&client, &values, &tags, &queries, threads);
}

#[test]
fn sets_of_two_to_nine_answer_every_rule_like_the_host() {
    let client = test_client();
    let mut values = Vec::new();
    let mut tags = Vec::new();

    for tag in 0..8u32 {
        let count = tag + 2;

        for index in 0..count {
            let value = hashed_unit(47 + tag, index);
            values.push(value);
            tags.push(tag);
        }
    }

    let all_queries = queries_for(8, &ALL_RULES);
    let threads = resolve_cube_threads(&client);

    for queries in all_queries.chunks(MAX_QUERIES as usize) {
        assert_matches_host(&client, &values, &tags, queries, threads);
    }
}

#[test]
fn a_new_batch_starts_with_every_element_untagged() {
    let client = test_client();
    let len = 777u32;
    let queries = queries_for(2, &ALL_RULES);
    let batch = SelectBatch::new(&client, len, &[0, 0], &queries);
    let batch_len = batch.len();
    assert_eq!(batch_len, len);

    let keys = read_u32(&client, batch.keys());
    let tags = read_u32(&client, batch.tags());
    let key_count = keys.len() as u32;
    let all_untagged = tags.iter().all(|&tag| tag == UNTAGGED);
    assert_eq!(key_count, len);
    assert!(all_untagged, "a new batch holds a tagged element");

    batch.queue(&client);

    let threads = resolve_cube_threads(&client);
    let (gpu_values, gpu_counts) = resolve(&client, &batch, threads);
    let all_empty = gpu_counts.iter().all(|&count| count == 0);
    let all_zero = gpu_values.iter().all(|&value| value.to_bits() == 0);
    assert!(all_empty, "an untagged batch counted elements: {gpu_counts:?}");
    assert!(all_zero, "an untagged batch selected values: {gpu_values:?}");
}

#[test]
fn an_empty_query_reports_zero_count() {
    let client = test_client();
    let len = 1000u32;
    let values: Vec<f32> = (0..len).map(|index| hashed_unit(5, index)).collect();
    let tags: Vec<u32> = (0..len).map(|index| select_tag(index % 3)).collect();
    let queries = queries_for(3, &ALL_RULES);
    let min_counts = [0u32; 3];
    let batch = batch_from_values(&client, &values, &tags, &min_counts, &queries);
    batch.queue(&client);

    let threads = resolve_cube_threads(&client);
    let (gpu_values, gpu_counts) = resolve(&client, &batch, threads);

    for (index, query) in queries.iter().enumerate() {
        let (host_value, host_count) = host_select(&values, &tags, *query);
        let gpu_count = gpu_counts[index];
        let gpu_bits = gpu_values[index].to_bits();
        let host_bits = host_value.to_bits();

        assert_eq!(gpu_count, host_count, "count of {query:?}");
        assert_eq!(gpu_bits, host_bits, "value of {query:?}");

        if query.tag == 1 {
            let zero_bits = 0.0f32.to_bits();
            assert_eq!(gpu_count, 0, "tag 1 has no elements");
            assert_eq!(gpu_bits, zero_bits, "an empty set gives 0.0");
        }
    }
}

#[test]
fn a_single_element_set_answers_every_rule_with_it() {
    let client = test_client();
    let len = 600u32;
    let values: Vec<f32> = (0..len).map(|index| hashed_unit(9, index)).collect();
    let mut tags = vec![0u32; len as usize];
    tags[417] = 1;

    let queries = queries_for(2, &ALL_RULES);
    let threads = resolve_cube_threads(&client);
    assert_matches_host(&client, &values, &tags, &queries, threads);

    let min_counts = [0u32; 2];
    let batch = batch_from_values(&client, &values, &tags, &min_counts, &queries);
    batch.queue(&client);

    let (gpu_values, gpu_counts) = resolve(&client, &batch, threads);
    let single = values[417];

    for (index, query) in queries.iter().enumerate() {
        if query.tag != 1 {
            continue;
        }

        let gpu_count = gpu_counts[index];
        let gpu_bits = gpu_values[index].to_bits();
        let single_bits = single.to_bits();
        assert_eq!(gpu_count, 1, "count of {query:?}");
        assert_eq!(gpu_bits, single_bits, "value of {query:?}");
    }
}

#[test]
fn untagged_elements_never_count() {
    let client = test_client();
    let len = 4096u32;
    let mut values = Vec::with_capacity(len as usize);
    let mut tags = Vec::with_capacity(len as usize);

    for index in 0..len {
        let untagged = index % 2 == 1;
        let value = if untagged { 1.0e30 } else { hashed_unit(11, index) };
        let tag = if untagged { UNTAGGED } else { (index / 2) % 2 };
        values.push(value);
        tags.push(tag);
    }

    let queries = queries_for(2, &ALL_RULES);
    let min_counts = [0u32; 2];
    let batch = batch_from_values(&client, &values, &tags, &min_counts, &queries);
    batch.queue(&client);

    let threads = resolve_cube_threads(&client);
    let (gpu_values, gpu_counts) = resolve(&client, &batch, threads);

    for (index, query) in queries.iter().enumerate() {
        let (host_value, _) = host_select(&values, &tags, *query);
        let gpu_value = gpu_values[index];
        let gpu_count = gpu_counts[index];
        let gpu_bits = gpu_value.to_bits();
        let host_bits = host_value.to_bits();
        let tagged_count = len / 4;

        assert_eq!(gpu_count, tagged_count, "count of {query:?}");
        assert_eq!(gpu_bits, host_bits, "value of {query:?}");
        assert!(gpu_value < 1.0, "{query:?} picked an untagged value {gpu_value}");
    }
}

#[test]
fn sigma_like_values_sharing_one_exponent_select_exactly() {
    let client = test_client();
    let len = 65_280u32;
    let one_bin_low = 1.0f32 / 64.0;
    let one_bin_span = 1.0f32 / 256.0;
    let mut values = Vec::with_capacity(len as usize);

    for index in 0..len {
        let unit = hashed_unit(13, index);
        let value = match index % 3 {
            2 => one_bin_low + one_bin_span * unit,
            _ => (2.0 + 10.0 * unit) / 255.0,
        };
        values.push(value);
    }

    let tags: Vec<u32> = (0..len).map(|index| index % 3).collect();

    // Between 2 and 12 out of 255 spans 11 pass-0 bins, four in each of the two lower exponents and
    // three in the top one.
    let all_digits = pass_0_digits(&values, &tags, None);
    let one_bin_digits = pass_0_digits(&values, &tags, Some(2));
    assert_eq!(all_digits, 11, "sigma-like values span {all_digits} pass-0 bins");
    assert_eq!(one_bin_digits, 1, "tag 2 spans {one_bin_digits} pass-0 bins");

    let queries = queries_for(3, &ALL_RULES);
    let threads = resolve_cube_threads(&client);
    assert_matches_host(&client, &values, &tags, &queries, threads);
}

/// How many pass-0 bins the values tagged `tag`, or every value, fall in.
fn pass_0_digits(values: &[f32], tags: &[u32], tag: Option<u32>) -> usize {
    let digits: BTreeSet<u32> = values
        .iter()
        .zip(tags)
        .filter(|(_, element_tag)| tag.is_none_or(|wanted| **element_tag == wanted))
        .map(|(&value, _)| order_key_host(value) >> SELECT_SHIFTS[0])
        .collect();

    digits.len()
}

#[test]
fn the_4k_sample_and_curve_batches_select_within_the_grid_limit() {
    let client = test_client();
    let threads = resolve_cube_threads(&client);

    let sample_len = 8 * 32_400u32;
    let sample_values: Vec<f32> = (0..sample_len)
        .map(|index| 2.0 / 255.0 + 10.0 / 255.0 * hashed_unit(17, index))
        .collect();
    let sample_tags: Vec<u32> = (0..sample_len)
        .map(|index| match index % 5 {
            4 => UNTAGGED,
            tag => tag,
        })
        .collect();
    let sample_queries = queries_for(4, &PERCENTILE_RULES);
    let sample_query_count = sample_queries.len();
    let sample_cubes = sample_len.div_ceil(SELECT_CUBE_THREADS);
    assert_eq!(sample_query_count, 16);
    assert!(sample_cubes <= MAX_GRID_1D, "{sample_cubes} cubes");
    assert_matches_host(&client, &sample_values, &sample_tags, &sample_queries, threads);

    let curve_len = 4 * 32_400u32;
    let curve_values: Vec<f32> = (0..curve_len)
        .map(|index| 2.0 / 255.0 + 10.0 / 255.0 * hashed_unit(19, index))
        .collect();
    let curve_tags: Vec<u32> = (0..curve_len).map(|index| (index / 97) % 16).collect();
    let curve_queries = queries_for(16, &MEDIAN_RULES);
    let curve_query_count = curve_queries.len() as u32;
    let curve_cubes = curve_len.div_ceil(SELECT_CUBE_THREADS);
    assert_eq!(curve_query_count, MAX_QUERIES);
    assert!(curve_cubes <= MAX_GRID_1D, "{curve_cubes} cubes");
    assert_matches_host(&client, &curve_values, &curve_tags, &curve_queries, threads);
}

#[test]
fn a_tag_below_its_min_count_is_never_resolved() {
    let client = test_client();
    let mut values = Vec::new();
    let mut tags = Vec::new();

    for index in 0..31u32 {
        values.push(hashed_unit(23, index));
        tags.push(0);
    }

    for index in 0..32u32 {
        values.push(hashed_unit(29, index));
        tags.push(1);
    }

    for index in 0..300u32 {
        values.push(hashed_unit(31, index));
        tags.push(UNTAGGED);
    }

    let queries = queries_for(2, &MEDIAN_RULES);
    let min_counts = [32u32, 32];
    let batch = batch_from_values(&client, &values, &tags, &min_counts, &queries);
    batch.queue(&client);

    let histogram = read_i32(&client, batch.histogram());
    let state = read_u32(&client, batch.state());
    let tag_count = batch.tag_count() as usize;
    let query_count = batch.query_count() as usize;
    let bins = SELECT_BINS as usize;

    for (index, query) in queries.iter().enumerate() {
        if query.tag != 0 {
            continue;
        }

        for pass in 1..3usize {
            let row = tag_count + (pass - 1) * query_count + index;
            let start = row * bins;
            let row_bins = &histogram[start..start + bins];
            let nonzero = row_bins.iter().filter(|&&bin| bin != 0).count();
            assert_eq!(
                nonzero, 0,
                "pass {pass} row of {query:?} has {nonzero} nonzero bins"
            );
        }

        for slot in 0..SELECT_SLOTS as usize {
            let base = (index * SELECT_SLOTS as usize + slot) * SELECT_SLOT_WORDS as usize;
            let count = state[base + 2];
            let written = state[base + 3];
            assert_eq!(count, 31, "slot {slot} count of {query:?}");
            assert_eq!(written, SLOT_STOPPED, "slot {slot} flag of {query:?}");
        }
    }

    let threads = resolve_cube_threads(&client);
    let (gpu_values, gpu_counts) = resolve(&client, &batch, threads);

    for (index, query) in queries.iter().enumerate() {
        let gpu_count = gpu_counts[index];
        let gpu_bits = gpu_values[index].to_bits();

        if query.tag == 0 {
            let zero_bits = 0.0f32.to_bits();
            assert_eq!(gpu_count, 31, "count of {query:?}");
            assert_eq!(gpu_bits, zero_bits, "value of {query:?}");
            continue;
        }

        let (host_value, host_count) = host_select(&values, &tags, *query);
        let host_bits = host_value.to_bits();
        assert_eq!(gpu_count, host_count, "count of {query:?}");
        assert_eq!(gpu_bits, host_bits, "value of {query:?}");
    }
}

/// The `[prefix, remaining rank, count, written]` words a full walk gives after digit `digit`.
fn host_slot(keys: &[u32], rank: u32, digit: usize) -> [u32; 4] {
    let mut sorted = keys.to_vec();
    sorted.sort_unstable();

    let target = sorted[rank as usize];
    let shift = SELECT_SHIFTS[digit];
    let target_prefix = target >> shift;
    let below = sorted
        .iter()
        .filter(|&&key| (key >> shift) < target_prefix)
        .count() as u32;

    [
        target_prefix << shift,
        rank - below,
        keys.len() as u32,
        SLOT_WRITTEN,
    ]
}

#[test]
fn the_published_state_matches_a_full_walk() {
    let client = test_client();
    let len = 20_000u32;
    let values: Vec<f32> = (0..len)
        .map(|index| 2.0 / 255.0 + 10.0 / 255.0 * hashed_unit(37, index))
        .collect();
    let tags: Vec<u32> = (0..len)
        .map(|index| match index % 5 {
            4 => UNTAGGED,
            tag => tag,
        })
        .collect();
    let mut queries = queries_for(4, &ALL_RULES);
    queries.push(SelectQuery {
        tag: 4,
        rule: RankRule::First,
    });
    let min_counts = [0u32; 5];
    let batch = batch_from_values(&client, &values, &tags, &min_counts, &queries);
    batch.queue(&client);

    let state = read_u32(&client, batch.state());
    let keys: Vec<u32> = values.iter().copied().map(order_key_host).collect();

    for (index, query) in queries.iter().enumerate() {
        let tag_keys: Vec<u32> = keys
            .iter()
            .zip(&tags)
            .filter(|(_, tag)| **tag == query.tag)
            .map(|(key, _)| *key)
            .collect();

        for slot in 0..SELECT_SLOTS as usize {
            let base = (index * SELECT_SLOTS as usize + slot) * SELECT_SLOT_WORDS as usize;
            let words = &state[base..base + SELECT_SLOT_WORDS as usize];

            if tag_keys.is_empty() {
                assert_eq!(words, &[0, 0, 0, 0], "slot {slot} of the empty {query:?}");
                continue;
            }

            let rank = query.rule.rank(tag_keys.len() as u32);
            let expected = host_slot(&tag_keys, rank, slot);
            assert_eq!(words, &expected, "slot {slot} of {query:?}");
        }
    }
}

#[test]
fn many_queries_in_one_cube_resolve_like_one_at_a_time() {
    let client = test_client();
    let len = 256u32;
    let values: Vec<f32> = (0..len)
        .map(|index| 2.0 / 255.0 + 10.0 / 255.0 * hashed_unit(41, index))
        .collect();
    let tags: Vec<u32> = (0..len).map(|index| index % 16).collect();
    let queries = queries_for(16, &MEDIAN_RULES);
    let query_count = queries.len() as u32;
    assert_eq!(query_count, MAX_QUERIES);

    assert_matches_host(&client, &values, &tags, &queries, 1024);
}

#[test]
fn a_batch_answers_identically_twice() {
    let client = test_client();
    let len = 30_000u32;
    let values: Vec<f32> = (0..len).map(|index| hashed_unit(43, index)).collect();
    let tags: Vec<u32> = (0..len).map(|index| index % 5).collect();
    let queries = queries_for(5, &ALL_RULES);
    let min_counts = [0u32; 5];
    let batch = batch_from_values(&client, &values, &tags, &min_counts, &queries);
    let threads = resolve_cube_threads(&client);

    batch.queue(&client);
    let (first_values, first_counts) = resolve(&client, &batch, threads);

    let histogram = read_i32(&client, batch.histogram());
    let state = read_u32(&client, batch.state());
    let histogram_zeroed = histogram.iter().all(|&bin| bin == 0);
    let state_empty = state.iter().all(|&word| word == 0);
    assert!(histogram_zeroed, "the resolve left histogram counts behind");
    assert!(state_empty, "the resolve left published slots behind");

    batch.queue(&client);
    let (second_values, second_counts) = resolve(&client, &batch, threads);

    let first_bits: Vec<u32> = first_values.iter().map(|value| value.to_bits()).collect();
    let second_bits: Vec<u32> = second_values.iter().map(|value| value.to_bits()).collect();
    assert_eq!(first_bits, second_bits);
    assert_eq!(first_counts, second_counts);
}

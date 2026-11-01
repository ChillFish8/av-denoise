use cubecl::prelude::*;

use super::records::{
    BlockSpec,
    QuarterSpec,
    frame_records,
    hashed_partials,
    hashed_unit,
    static_block,
    ulp_gap,
};
use crate::engine::align::StorageAlign;
use crate::noise::NOISE_CHANNELS;
use crate::noise::curve::{NOISE_CURVE_BINS, NoiseCurve};
use crate::noise::reading::{
    BLOCK_SET_QUERIES,
    BLOCK_SET_TAGS,
    IMMERKAER_CHANNEL_QUERIES,
    IMMERKAER_QUERY_EXCLUDED_BASE,
    IMMERKAER_QUERY_REFERENCE_BASE,
    READING_CURVE_VALID,
    READING_SAMPLE_VALID,
    READING_WORDS,
    ReadingRecord,
    SAMPLE_CHANNEL_QUERIES,
    SAMPLE_QUERY_ACCEPTED,
    SAMPLE_QUERY_QUARTER,
    SAMPLE_QUERY_RHO,
    TAG_ACCEPTED,
    TAG_EXCLUDED_BASE,
    TAG_QUARTER,
    TAG_REFERENCE_BASE,
    TAG_RHO,
    immerkaer_queries,
    immerkaer_query_count,
    immerkaer_query_texture,
    sample_queries,
    sample_query_count,
    tag_moved_texture,
};
use crate::noise::select::{RankRule, SelectQuery};
use crate::noise::spatial::{RESULTS_PER_SLOT, sigma_block_p25_from_partials};
use crate::noise::state::{
    BLOCK_ACCEPTED,
    NoiseShape,
    NoiseState,
    ReadingSettings,
    SAMPLE_RUN_ACCEPTED,
    SlotStats,
    channel_flag,
};
use crate::noise::stats::{lower_quartile, median};
use crate::noise::temporal::{
    RHO_SIGMA_GATE,
    STATIC_GATE,
    TemporalNoiseSample,
    accepted_channel_indices,
    accepted_static_blocks,
    min_static_blocks,
    temporal_noise_reading,
    temporal_stats_blocks,
};
use crate::tune::gpu::sync;
use crate::tune::tests::runtime::test_client;

const MAX_ULPS: u32 = 2;
const RESULTS_ROW: [f32; RESULTS_PER_SLOT] = [0.0; RESULTS_PER_SLOT];

/// The geometry of one plane a test reads.
#[derive(Debug, Clone, Copy)]
struct Plane {
    width: u32,
    height: u32,
    channels: u32,
}

const LUMA_1080P: Plane = Plane {
    width: 1920,
    height: 1080,
    channels: 1,
};

const CHROMA_1080P: Plane = Plane {
    width: 960,
    height: 540,
    channels: 2,
};

impl Plane {
    fn luma(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            channels: 1,
        }
    }

    fn chroma(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            channels: 2,
        }
    }

    fn blocks(self) -> usize {
        let (blocks_x, blocks_y) = temporal_stats_blocks(self.width, self.height);
        (blocks_x * blocks_y) as usize
    }

    fn records(self, blocks: &[BlockSpec]) -> Vec<f32> {
        frame_records(self.width, self.height, self.channels, blocks)
    }
}

/// What the host reference reads from one slot's stats.
struct HostReading {
    immerkaer_low: [f32; NOISE_CHANNELS as usize],
    sample: Option<TemporalNoiseSample>,
    accepted: Vec<usize>,
}

fn host_reading(plane: Plane, partials: &[f32], records: &[f32]) -> HostReading {
    let immerkaer_low = sigma_block_p25_from_partials(partials, plane.channels, plane.width, plane.height);
    let reading = temporal_noise_reading(records, plane.channels, plane.width, plane.height, false);
    let accepted_blocks = accepted_static_blocks(records, plane.channels, plane.width, plane.height);
    let accepted = accepted_blocks
        .unwrap_or_default()
        .iter()
        .map(|block| block.index)
        .collect();

    HostReading {
        immerkaer_low,
        sample: reading.sample,
        accepted,
    }
}

fn shape_for(client: &Client, plane: Plane, temporal: bool) -> NoiseShape {
    NoiseShape {
        width: plane.width,
        height: plane.height,
        channels: plane.channels,
        frames: 3,
        ring_radius: 1,
        temporal,
        align: StorageAlign::from_client(client),
    }
}

fn state_for(
    client: &Client,
    plane: Plane,
    temporal: bool,
    settings: ReadingSettings,
    partials: &[f32],
    records: &[f32],
) -> NoiseState {
    let shape = shape_for(client, plane, temporal);

    NoiseState::from_stats_for_test(client, shape, settings, partials, RESULTS_ROW, records)
}

/// The reading of `slot` and, with temporal stats, the indices of the blocks the last reading
/// accepted.
fn read_back(state: &NoiseState, slot: u32) -> (ReadingRecord, Vec<usize>) {
    let reading = state.read_reading(slot);
    let flags = state.read_block_flags();
    let accepted = accepted_indices(&flags);

    (reading, accepted)
}

/// The GPU reading of slot 0 and the indices of the blocks it accepted.
fn gpu_reading(
    client: &Client,
    plane: Plane,
    temporal: bool,
    partials: &[f32],
    records: &[f32],
) -> (ReadingRecord, Vec<usize>) {
    let settings = ReadingSettings::default();
    let state = state_for(client, plane, temporal, settings, partials, records);
    state.queue_reading(0).expect("the reading queues");

    read_back(&state, 0)
}

fn accepted_indices(flags: &[u32]) -> Vec<usize> {
    flags
        .iter()
        .enumerate()
        .filter(|(_, flag)| **flag & BLOCK_ACCEPTED != 0)
        .map(|(index, _)| index)
        .collect()
}

fn assert_close(gpu: f32, host: f32, max_ulps: u32, what: &str) {
    let gap = ulp_gap(gpu, host);
    assert!(gap <= max_ulps, "{what}: gpu {gpu} host {host}, {gap} ulp apart");
}

fn assert_immerkaer_matches(gpu: &ReadingRecord, host: &HostReading) {
    for channel in 0..NOISE_CHANNELS as usize {
        let what = format!("immerkaer_low[{channel}]");
        assert_close(
            gpu.immerkaer_low[channel],
            host.immerkaer_low[channel],
            MAX_ULPS,
            &what,
        );
    }
}

fn assert_sample_matches(gpu: Option<TemporalNoiseSample>, host: Option<TemporalNoiseSample>, max_ulps: u32) {
    let (gpu, host) = match (gpu, host) {
        (Some(gpu), Some(host)) => (gpu, host),
        (None, None) => return,
        (gpu, host) => panic!("the gpu sample {gpu:?} and the host sample {host:?} disagree"),
    };

    for channel in 0..NOISE_CHANNELS as usize {
        let sigma_what = format!("sigma[{channel}]");
        let low_what = format!("sigma_low[{channel}]");
        assert_close(gpu.sigma[channel], host.sigma[channel], max_ulps, &sigma_what);
        assert_close(
            gpu.sigma_low[channel],
            host.sigma_low[channel],
            max_ulps,
            &low_what,
        );
    }

    assert_close(gpu.rho, host.rho, max_ulps, "rho");
    assert_close(
        gpu.static_fraction,
        host.static_fraction,
        max_ulps,
        "static_fraction",
    );
}

fn assert_reading_matches(gpu: &ReadingRecord, gpu_accepted: &[usize], host: &HostReading) {
    let host_accepted = host.accepted.len() as u32;

    assert_immerkaer_matches(gpu, host);
    assert_sample_matches(gpu.sample, host.sample, MAX_ULPS);
    assert_eq!(gpu.accepted, host_accepted, "accepted count");
    let host_blocks = host.accepted.as_slice();
    let has_curve = gpu.curve.is_some();
    assert_eq!(gpu_accepted, host_blocks, "accepted blocks");
    assert!(!has_curve);
}

fn assert_sampled(gpu: &ReadingRecord, host: &HostReading, expected: bool) {
    let gpu_sampled = gpu.sample.is_some();
    let host_sampled = host.sample.is_some();

    assert_eq!(gpu_sampled, expected, "gpu sample");
    assert_eq!(host_sampled, expected, "host sample");
}

/// Runs one plane on the GPU and on the host and checks they agree.
fn check_plane(plane: Plane, partials: &[f32], records: &[f32]) -> (ReadingRecord, HostReading) {
    let client = test_client();
    let (gpu, gpu_accepted) = gpu_reading(&client, plane, true, partials, records);
    let host = host_reading(plane, partials, records);

    assert_reading_matches(&gpu, &gpu_accepted, &host);

    (gpu, host)
}

/// A block whose mean residual is far past the static gate.
fn moving_block(sigma: f32) -> BlockSpec {
    BlockSpec::uniform(4.0 * STATIC_GATE, sigma, 0.2)
}

fn static_blocks(plane: Plane, seed: u32) -> Vec<BlockSpec> {
    let count = plane.blocks() as u32;
    (0..count).map(|index| static_block(seed, index)).collect()
}

/// Asserts `queries` holds `tag` with each of `rules` in turn, starting at `first`.
fn assert_query_run(queries: &[SelectQuery], first: u32, tag: u32, rules: &[RankRule]) {
    for (offset, &rule) in rules.iter().enumerate() {
        let index = first as usize + offset;
        let expected = SelectQuery { tag, rule };

        assert_eq!(queries[index], expected, "query {index}");
    }
}

#[test]
fn the_query_indices_name_their_queries() {
    let quartile_pair = [RankRule::QuartileLow, RankRule::QuartileHigh];
    let median_pair = [RankRule::MedianLow, RankRule::MedianHigh];
    let sample_channel = [
        RankRule::MedianLow,
        RankRule::MedianHigh,
        RankRule::QuartileLow,
        RankRule::QuartileHigh,
    ];
    let reference = [
        RankRule::QuartileLow,
        RankRule::QuartileHigh,
        RankRule::First,
        RankRule::Last,
    ];

    for channels in 1..=NOISE_CHANNELS {
        let spatial = immerkaer_queries(false, channels);
        let immerkaer = immerkaer_queries(true, channels);
        let sample = sample_queries(channels);

        assert_eq!(spatial.len(), IMMERKAER_QUERY_REFERENCE_BASE as usize);
        assert_eq!(immerkaer.len(), immerkaer_query_count(channels) as usize);
        assert_eq!(sample.len(), sample_query_count(channels) as usize);

        for channel in 0..NOISE_CHANNELS {
            let immerkaer_first = IMMERKAER_CHANNEL_QUERIES * channel;
            let sample_first = SAMPLE_CHANNEL_QUERIES * channel;

            assert_query_run(&spatial, immerkaer_first, channel, &quartile_pair);
            assert_query_run(&immerkaer, immerkaer_first, channel, &quartile_pair);
            assert_query_run(&sample, sample_first, channel, &sample_channel);
        }

        for channel in 0..channels {
            let reference_query = IMMERKAER_QUERY_REFERENCE_BASE + BLOCK_SET_QUERIES * channel;
            let excluded_query = IMMERKAER_QUERY_EXCLUDED_BASE + BLOCK_SET_QUERIES * channel;
            let reference_tag = TAG_REFERENCE_BASE + BLOCK_SET_TAGS * channel;
            let excluded_tag = TAG_EXCLUDED_BASE + BLOCK_SET_TAGS * channel;
            let accepted_query = SAMPLE_QUERY_ACCEPTED + channel;
            let accepted_tag = TAG_ACCEPTED + channel;

            assert_query_run(&immerkaer, reference_query, reference_tag, &reference);
            assert_query_run(&immerkaer, excluded_query, excluded_tag, &[RankRule::First]);
            assert_query_run(&sample, accepted_query, accepted_tag, &[RankRule::First]);
        }

        let texture_query = immerkaer_query_texture(channels);
        let texture_tag = tag_moved_texture(channels);
        assert_query_run(&immerkaer, texture_query, texture_tag, &[RankRule::TextureShare]);
        assert_query_run(&sample, SAMPLE_QUERY_RHO, TAG_RHO, &median_pair);
        assert_query_run(&sample, SAMPLE_QUERY_QUARTER, TAG_QUARTER, &median_pair);
    }
}

#[test]
fn a_reading_survives_its_words() {
    let sample = TemporalNoiseSample {
        sigma: [0.01, 0.02],
        sigma_low: [0.005, 0.015],
        rho: 0.3,
        static_fraction: 0.75,
    };
    let mut ratios = [1.0f32; NOISE_CURVE_BINS];
    ratios[3] = 1.25;
    let curve = NoiseCurve {
        ratios,
        sigma_quarter_median: 0.012,
    };
    let record = ReadingRecord {
        immerkaer_low: [0.004, 0.003],
        sample: Some(sample),
        accepted: 1234,
        curve: Some(curve),
        populated_bins: 5,
    };

    let words = record.to_words();
    let back = ReadingRecord::from_words(&words);
    assert_eq!(back, record);

    let empty = ReadingRecord {
        sample: None,
        curve: None,
        ..record
    };
    let empty_words = empty.to_words();
    let sample_valid = empty_words[READING_SAMPLE_VALID as usize];
    let curve_valid = empty_words[READING_CURVE_VALID as usize];
    let empty_back = ReadingRecord::from_words(&empty_words);
    assert_eq!(sample_valid, 0.0);
    assert_eq!(curve_valid, 0.0);
    assert_eq!(empty_back, empty);
}

#[test]
fn the_immerkaer_quartile_matches_the_host_within_two_ulp() {
    let client = test_client();

    for (seed, plane) in [(3, LUMA_1080P), (5, CHROMA_1080P)] {
        let partials = hashed_partials(plane.width, plane.height, seed);
        let blocks = static_blocks(plane, seed);
        let records = plane.records(&blocks);
        let (gpu, _) = gpu_reading(&client, plane, true, &partials, &records);
        let host = host_reading(plane, &partials, &records);

        assert_immerkaer_matches(&gpu, &host);
    }
}

#[test]
fn a_static_frame_reading_matches_the_host_within_two_ulp() {
    for (seed, plane) in [(11, Plane::luma(256, 192)), (13, Plane::chroma(256, 192))] {
        let partials = hashed_partials(plane.width, plane.height, seed);
        let blocks = static_blocks(plane, seed);
        let records = plane.records(&blocks);
        let (gpu, host) = check_plane(plane, &partials, &records);

        let block_count = plane.blocks();
        let host_count = host.accepted.len();
        let sampled = gpu.sample.is_some();
        assert!(sampled);
        assert_eq!(host_count, block_count);
    }
}

#[test]
fn power_of_two_shapes_give_the_host_fraction_and_counts_exactly() {
    let plane = Plane::luma(64, 64);
    let partials = hashed_partials(plane.width, plane.height, 17);
    let mut blocks = static_blocks(plane, 17);
    for index in [1, 6, 9, 14, 15] {
        blocks[index] = moving_block(3.0 / 255.0);
    }

    let records = plane.records(&blocks);
    let (gpu, host) = check_plane(plane, &partials, &records);

    let gpu_sample = gpu.sample.expect("the gpu has a sample");
    let host_sample = host.sample.expect("the host has a sample");
    let gpu_fraction = gpu_sample.static_fraction.to_bits();
    let host_fraction = host_sample.static_fraction.to_bits();
    assert_eq!(gpu_fraction, host_fraction);
    assert_eq!(gpu.accepted, 11);
}

#[test]
fn a_zeroed_slot_has_no_sample() {
    let plane = Plane::luma(256, 192);
    let partials = hashed_partials(plane.width, plane.height, 19);
    let quiet = vec![BlockSpec::default(); plane.blocks()];
    let records = plane.records(&quiet);
    let (gpu, host) = check_plane(plane, &partials, &records);

    assert_sampled(&gpu, &host, false);
}

#[test]
fn letterboxed_frames_take_the_same_none_path_as_the_host() {
    let plane = Plane::luma(256, 192);
    let (blocks_x, blocks_y) = temporal_stats_blocks(plane.width, plane.height);
    let partials = hashed_partials(plane.width, plane.height, 23);

    let mut barred = static_blocks(plane, 23);
    for block_x in 0..blocks_x {
        let top = block_x as usize;
        let bottom = ((blocks_y - 1) * blocks_x + block_x) as usize;
        barred[top] = BlockSpec::default();
        barred[bottom] = BlockSpec::default();
    }

    let records = plane.records(&barred);
    let (gpu, host) = check_plane(plane, &partials, &records);
    assert_sampled(&gpu, &host, true);

    let mut textured = vec![BlockSpec::default(); plane.blocks()];
    for index in [17, 40, 77, 130] {
        textured[index] = BlockSpec::uniform(0.0, 6.0 / 255.0, 0.3);
    }

    let records = plane.records(&textured);
    let (gpu, host) = check_plane(plane, &partials, &records);
    assert_sampled(&gpu, &host, false);
}

#[test]
fn a_textured_remainder_with_no_spread_has_no_sample() {
    let plane = Plane::luma(128, 128);
    let partials = hashed_partials(plane.width, plane.height, 29);

    let mut blocks = vec![BlockSpec::uniform(0.0, 5.0 / 255.0, 0.3); plane.blocks()];
    let records = plane.records(&blocks);
    let (gpu, host) = check_plane(plane, &partials, &records);
    assert_sampled(&gpu, &host, true);

    blocks[21] = BlockSpec::uniform(0.0, 0.5 * RHO_SIGMA_GATE, 0.3);
    let records = plane.records(&blocks);
    let (gpu, host) = check_plane(plane, &partials, &records);
    assert_sampled(&gpu, &host, false);
}

/// A frame whose blocks all move except `accepted` static blocks spread over it.
fn fraction_frame(plane: Plane, accepted: usize, seed: u32) -> Vec<f32> {
    let block_count = plane.blocks();
    let mut blocks = vec![moving_block(3.0 / 255.0); block_count];

    for position in 0..accepted {
        let index = position * block_count / accepted;
        blocks[index] = static_block(seed, position as u32);
    }

    plane.records(&blocks)
}

#[test]
fn the_static_fraction_gate_matches_the_host_at_its_boundary() {
    let cases = [
        (Plane::luma(160, 160), 5),
        (Plane::luma(320, 320), 20),
        (Plane::luma(640, 360), 46),
    ];

    for (plane, boundary) in cases {
        let block_count = plane.blocks() as u32;
        let min_accepted = min_static_blocks(block_count);
        assert_eq!(min_accepted, boundary, "{block_count} blocks");

        let partials = hashed_partials(plane.width, plane.height, 31);

        for (accepted, expected) in [(boundary - 1, false), (boundary, true)] {
            let records = fraction_frame(plane, accepted as usize, 31);
            let (gpu, host) = check_plane(plane, &partials, &records);

            assert_sampled(&gpu, &host, expected);
            assert_eq!(gpu.accepted, accepted);
        }
    }
}

#[test]
fn an_outlier_above_five_times_the_quartile_is_rejected() {
    let plane = Plane::luma(128, 128);
    let partials = hashed_partials(plane.width, plane.height, 37);
    let mut blocks = static_blocks(plane, 37);
    let rejected = 12;
    let survivor = 50;

    // Both outliers sort above every background block, so placeholders at the top give the
    // quartile the full set has.
    blocks[rejected] = BlockSpec::uniform(0.0, 1.0, 0.3);
    blocks[survivor] = BlockSpec::uniform(0.0, 1.0, 0.3);
    let mut reference: Vec<f32> = blocks.iter().map(|block| block.sigma[0]).collect();
    reference.sort_by(f32::total_cmp);
    let quartile = lower_quartile(&reference);
    blocks[rejected] = BlockSpec::uniform(0.0, 6.0 * quartile, 0.3);
    blocks[survivor] = BlockSpec::uniform(0.0, 4.0 * quartile, 0.3);

    let records = plane.records(&blocks);
    let (_, host) = check_plane(plane, &partials, &records);

    let rejected_kept = host.accepted.contains(&rejected);
    let survivor_kept = host.accepted.contains(&survivor);
    assert!(!rejected_kept);
    assert!(survivor_kept);
}

#[test]
fn ragged_frames_match_the_host_reading() {
    for (seed, plane) in [(41, Plane::luma(70, 54)), (43, Plane::chroma(35, 27))] {
        let partials = hashed_partials(plane.width, plane.height, seed);
        let blocks = static_blocks(plane, seed);
        let records = plane.records(&blocks);
        let (gpu, _) = check_plane(plane, &partials, &records);

        let sampled = gpu.sample.is_some();
        assert!(sampled);
    }
}

#[test]
fn stale_tags_never_join_a_set() {
    let client = test_client();
    let luma = Plane::luma(256, 192);
    let partials = hashed_partials(luma.width, luma.height, 53);
    let blocks = static_blocks(luma, 53);
    let records = luma.records(&blocks);
    let settings = ReadingSettings::default();
    let mut state = state_for(&client, luma, true, settings, &partials, &records);

    state.poison_for_test();
    state.queue_reading(0).expect("the reading queues");

    let (gpu, gpu_accepted) = read_back(&state, 0);
    let host = host_reading(luma, &partials, &records);
    assert_reading_matches(&gpu, &gpu_accepted, &host);

    let sample = gpu.sample.expect("the luma state has a sample");
    let immerkaer_second = gpu.immerkaer_low[1];
    let sigma_second = sample.sigma[1];
    let sigma_low_second = sample.sigma_low[1];
    assert_eq!(immerkaer_second, 0.0);
    assert_eq!(sigma_second, 0.0);
    assert_eq!(sigma_low_second, 0.0);
}

#[test]
fn a_second_slot_reads_only_its_own_stats() {
    let client = test_client();
    let plane = Plane::chroma(256, 192);
    let first_partials = hashed_partials(plane.width, plane.height, 71);
    let first_blocks = static_blocks(plane, 71);
    let first_records = plane.records(&first_blocks);

    let second_partials = hashed_partials(plane.width, plane.height, 73);
    let mut second_blocks = static_blocks(plane, 73);
    for index in (0..second_blocks.len()).step_by(3) {
        second_blocks[index] = moving_block(5.0 / 255.0);
    }

    let second_records = plane.records(&second_blocks);

    let slots = [
        SlotStats {
            partials: &first_partials,
            results_row: RESULTS_ROW,
            records: &first_records,
        },
        SlotStats {
            partials: &second_partials,
            results_row: RESULTS_ROW,
            records: &second_records,
        },
    ];
    let shape = shape_for(&client, plane, true);
    let settings = ReadingSettings::default();
    let state = NoiseState::from_slots_for_test(&client, shape, settings, &slots);

    state.queue_reading(0).expect("the first reading queues");
    state.queue_reading(1).expect("the second reading queues");

    let first = state.read_reading(0);
    let (second, second_accepted) = read_back(&state, 1);
    let first_host = host_reading(plane, &first_partials, &first_records);
    let second_host = host_reading(plane, &second_partials, &second_records);
    let first_count = first_host.accepted.len() as u32;

    assert_immerkaer_matches(&first, &first_host);
    assert_sample_matches(first.sample, first_host.sample, MAX_ULPS);
    assert_eq!(first.accepted, first_count);
    assert_reading_matches(&second, &second_accepted, &second_host);
}

#[test]
fn quarter_lanes_never_join_the_channel_or_correlation_sets() {
    let client = test_client();
    let plane = Plane::luma(256, 192);
    let partials = hashed_partials(plane.width, plane.height, 79);
    let quarter = QuarterSpec {
        mean: 0.0,
        sigma: 9.0 / 255.0,
        luma: 0.5,
        flatness: 0.0,
        luma_min: 0.4,
        luma_max: 0.6,
    };
    let mut blocks = static_blocks(plane, 79);
    for block in &mut blocks {
        block.quarters = [quarter; 4];
    }

    let records = plane.records(&blocks);
    let settings = ReadingSettings { luma_fields: true };
    let state = state_for(&client, plane, true, settings, &partials, &records);
    state.queue_reading(0).expect("the reading queues");

    let (gpu, gpu_accepted) = read_back(&state, 0);
    let host = host_reading(plane, &partials, &records);

    assert_reading_matches(&gpu, &gpu_accepted, &host);
    assert_sampled(&gpu, &host, true);
}

#[test]
fn a_radius_zero_state_reads_immerkaer_only() {
    let client = test_client();
    let plane = Plane::chroma(256, 192);
    let partials = hashed_partials(plane.width, plane.height, 59);
    let (gpu, _) = gpu_reading(&client, plane, false, &partials, &[]);
    let host_low = sigma_block_p25_from_partials(&partials, plane.channels, plane.width, plane.height);
    let host = HostReading {
        immerkaer_low: host_low,
        sample: None,
        accepted: Vec::new(),
    };

    assert_immerkaer_matches(&gpu, &host);
    let sampled = gpu.sample.is_some();
    assert!(!sampled);
    assert_eq!(gpu.accepted, 0);

    sync(&client).expect("the device syncs");
}

#[test]
fn the_reading_is_identical_across_runs() {
    let client = test_client();
    let plane = Plane::chroma(256, 192);
    let partials = hashed_partials(plane.width, plane.height, 61);
    let blocks = static_blocks(plane, 61);
    let records = plane.records(&blocks);

    let (first, _) = gpu_reading(&client, plane, true, &partials, &records);
    let (second, _) = gpu_reading(&client, plane, true, &partials, &records);
    let first_words = first.to_words();
    let second_words = second.to_words();
    let first_bits: Vec<u32> = first_words.iter().map(|word| word.to_bits()).collect();
    let second_bits: Vec<u32> = second_words.iter().map(|word| word.to_bits()).collect();

    assert_eq!(first_bits, second_bits);
}

/// Static blocks where two in three moved, and every fifth carries a high texture at a raised sigma.
fn moved_blocks(plane: Plane, seed: u32) -> Vec<BlockSpec> {
    let mut blocks = static_blocks(plane, seed);

    for (index, block) in blocks.iter_mut().enumerate() {
        let unit = hashed_unit(seed + 7, index as u32);
        let textured = index % 5 == 0;
        block.moved = index % 3 != 0;
        block.texture = if textured { 1000.0 + unit } else { unit };

        if textured {
            block.sigma = block.sigma.map(|sigma| 1.8 * sigma);
        }
    }

    blocks
}

#[test]
fn gpu_sample_matches_host_reference_with_moved_blocks() {
    for (seed, plane) in [(83, Plane::luma(256, 192)), (89, Plane::chroma(256, 192))] {
        let partials = hashed_partials(plane.width, plane.height, seed);
        let blocks = moved_blocks(plane, seed);
        let records = plane.records(&blocks);
        let (gpu, host) = check_plane(plane, &partials, &records);

        let host_record = ReadingRecord {
            immerkaer_low: host.immerkaer_low,
            sample: host.sample,
            accepted: host.accepted.len() as u32,
            curve: None,
            populated_bins: 0,
        };
        let gpu_words = gpu.to_words();
        let host_words = host_record.to_words();
        for (word, (gpu_word, host_word)) in gpu_words.iter().zip(host_words).enumerate() {
            let what = format!("reading word {word}");
            assert_close(*gpu_word, host_word, MAX_ULPS, &what);
        }

        let unmoved: Vec<BlockSpec> = blocks
            .iter()
            .map(|block| BlockSpec {
                moved: false,
                ..*block
            })
            .collect();
        let unmoved_records = plane.records(&unmoved);
        let unmoved_host = host_reading(plane, &partials, &unmoved_records);
        let sample = host.sample.expect("the moved frame has a sample");
        let unmoved_sample = unmoved_host.sample.expect("the unmoved frame has a sample");
        let block_count = plane.blocks() as u32;

        assert!(
            sample.sigma[0] < unmoved_sample.sigma[0],
            "the filter drops the textured blocks"
        );
        assert_eq!(sample.static_fraction, unmoved_sample.static_fraction);
        assert_eq!(gpu.accepted, block_count);
    }
}

#[test]
fn a_reading_with_every_sigma_left_out_has_no_sample() {
    let plane = Plane::luma(320, 16);
    let partials = hashed_partials(plane.width, plane.height, 97);
    let mut blocks = Vec::new();

    for index in 0..plane.blocks() {
        let low_texture = index < 8;
        let sigma = if low_texture { 20.0 / 255.0 } else { 1.0 / 255.0 };
        let texture = if low_texture { 1.0 } else { 100.0 };
        let base = BlockSpec::uniform(0.0, sigma, 0.1);
        let block = BlockSpec {
            texture,
            moved: true,
            ..base
        };
        blocks.push(block);
    }

    let records = plane.records(&blocks);
    let (gpu, host) = check_plane(plane, &partials, &records);

    assert_sampled(&gpu, &host, false);
    assert_eq!(gpu.accepted, 12);
}

/// A luma frame with moved texture, outliers, quiet blocks, moving blocks and curve quarters.
fn recorded_luma_blocks() -> Vec<BlockSpec> {
    let plane = Plane::luma(256, 192);
    let mut blocks = moved_blocks(plane, 101);

    for (index, block) in blocks.iter_mut().enumerate() {
        let luma = 0.2 + 0.2 * (index % 4) as f32;
        let quarter = QuarterSpec {
            mean: 0.0,
            sigma: block.sigma[0],
            luma,
            flatness: 0.0,
            luma_min: luma - 0.01,
            luma_max: luma + 0.01,
        };
        block.quarters = [quarter; 4];
    }

    for index in [7, 33, 120] {
        blocks[index].sigma = [40.0 / 255.0; NOISE_CHANNELS as usize];
    }

    for index in [2, 58] {
        blocks[index].sigma = [0.5 * RHO_SIGMA_GATE; NOISE_CHANNELS as usize];
    }

    for index in [11, 90, 150, 151] {
        blocks[index] = moving_block(3.0 / 255.0);
    }

    blocks
}

/// The reading words of [recorded_luma_blocks], recorded before each channel had its own checks.
const RECORDED_LUMA_WORDS: [u32; READING_WORDS as usize] = [
    0x3adaa02c, 0x00000000, 0x3f800000, 0x3c3c3903, 0x00000000, 0x3c1cd525, 0x00000000, 0x3e8cd626,
    0x3f73ffff, 0x43370000, 0x3f800000, 0x3c47f521, 0x40800000, 0x3f6b3254, 0x3f6b3254, 0x3f6b3254,
    0x3f6b3254, 0x3f72067f, 0x3f78daab, 0x3f7faed7, 0x3f7b0aa1, 0x3f76666a, 0x3f71c234, 0x3f808b01,
    0x3f8834e8, 0x3f8fdecf, 0x3f8fdecf, 0x3f8fdecf, 0x3f8fdecf,
];

#[test]
fn a_single_channel_reading_matches_the_channel_0_rules_word_for_word() {
    let client = test_client();
    let plane = Plane::luma(256, 192);
    let partials = hashed_partials(plane.width, plane.height, 101);
    let blocks = recorded_luma_blocks();
    let records = plane.records(&blocks);
    let settings = ReadingSettings { luma_fields: true };
    let state = state_for(&client, plane, true, settings, &partials, &records);
    state.queue_reading(0).expect("the reading queues");

    let (gpu, _) = read_back(&state, 0);
    let words = gpu.to_words();
    let bits: Vec<u32> = words.iter().map(|word| word.to_bits()).collect();

    assert_eq!(bits, RECORDED_LUMA_WORDS);
}

/// The median and lower quartile of channel `channel`'s sigma over every block not in `left_out`.
fn expected_channel_sigma(blocks: &[BlockSpec], channel: usize, left_out: &[usize]) -> (f32, f32) {
    let mut sigmas: Vec<f32> = blocks
        .iter()
        .enumerate()
        .filter(|(index, _)| !left_out.contains(index))
        .map(|(_, block)| block.sigma[channel])
        .collect();
    sigmas.sort_by(f32::total_cmp);

    let sigma = median(&sigmas);
    let sigma_low = lower_quartile(&sigmas);

    (sigma, sigma_low)
}

fn assert_within(actual: f32, expected: f32, what: &str) {
    let gap = (actual - expected).abs();
    assert!(gap <= 1e-6, "{what}: {actual} expected {expected}");
}

/// Asserts channel `channel` of `sample` reads the sigma of every block not in `left_out`.
fn assert_channel_excludes(
    sample: TemporalNoiseSample,
    blocks: &[BlockSpec],
    channel: usize,
    left_out: &[usize],
) {
    let (sigma, sigma_low) = expected_channel_sigma(blocks, channel, left_out);
    let sigma_what = format!("sigma[{channel}]");
    let low_what = format!("sigma_low[{channel}]");

    assert_within(sample.sigma[channel], sigma, &sigma_what);
    assert_within(sample.sigma_low[channel], sigma_low, &low_what);
}

/// Static grain where every fourth block carries texture in channel 1 alone.
///
/// Half the textured blocks pan past the static gate and half sit far above the outlier ceiling.
fn channel_1_textured_blocks(plane: Plane, seed: u32) -> (Vec<BlockSpec>, Vec<usize>) {
    let mut blocks = static_blocks(plane, seed);
    let mut textured = Vec::new();

    for (index, block) in blocks.iter_mut().enumerate() {
        match index % 8 {
            0 => {
                block.mean[1] = 4.0 * STATIC_GATE;
                block.sigma[1] = 20.0 / 255.0;
            },
            4 => block.sigma[1] = 30.0 / 255.0,
            _ => continue,
        }

        textured.push(index);
    }

    (blocks, textured)
}

/// Asserts each channel's accepted flags and accepted lanes name the blocks the host accepts, and
/// returns how many blocks each channel accepted.
fn assert_channel_accepted(plane: Plane, partials: &[f32], records: &[f32]) -> Vec<usize> {
    let client = test_client();
    let settings = ReadingSettings::default();
    let state = state_for(&client, plane, true, settings, partials, records);
    state.queue_reading(0).expect("the reading queues");

    let flags = state.read_block_flags();
    let sample_tags = state.read_sample_tags();
    let block_count = plane.blocks();
    let mut counts = Vec::new();

    for channel in 0..plane.channels {
        let host = accepted_channel_indices(records, plane.channels, plane.width, plane.height, channel);
        let host_indices = host.expect("the channel's ceiling is trusted");
        let flag = channel_flag(BLOCK_ACCEPTED, channel);
        let flagged: Vec<usize> = (0..block_count)
            .filter(|&index| flags[index] & flag != 0)
            .collect();

        let run_start = (SAMPLE_RUN_ACCEPTED + channel) as usize * block_count;
        let accepted_tag = TAG_ACCEPTED + channel;
        let tagged: Vec<usize> = (0..block_count)
            .filter(|&index| sample_tags[run_start + index] == accepted_tag)
            .collect();

        assert_eq!(flagged, host_indices, "channel {channel} accepted flags");
        assert_eq!(tagged, host_indices, "channel {channel} accepted lanes");
        counts.push(host_indices.len());
    }

    counts
}

#[test]
fn a_channel_1_moving_texture_block_is_excluded_from_channel_1_only() {
    let client = test_client();
    let plane = Plane::chroma(256, 192);
    let luma = Plane::luma(plane.width, plane.height);
    let partials = hashed_partials(plane.width, plane.height, 103);
    let (blocks, textured) = channel_1_textured_blocks(plane, 103);
    let records = plane.records(&blocks);
    let (gpu, host) = check_plane(plane, &partials, &records);

    let sample = gpu.sample.expect("the chroma frame has a sample");
    let host_sample = host.sample.expect("the host has a sample");
    assert_channel_excludes(sample, &blocks, 1, &textured);
    assert_channel_excludes(host_sample, &blocks, 1, &textured);
    assert_channel_excludes(sample, &blocks, 0, &[]);

    let luma_records = luma.records(&blocks);
    let (luma_gpu, _) = gpu_reading(&client, luma, true, &partials, &luma_records);
    let luma_sample = luma_gpu.sample.expect("the channel 0 frame has a sample");

    assert_eq!(sample.sigma[0].to_bits(), luma_sample.sigma[0].to_bits());
    assert_eq!(sample.sigma_low[0].to_bits(), luma_sample.sigma_low[0].to_bits());
    assert_eq!(sample.rho.to_bits(), luma_sample.rho.to_bits());
    assert_eq!(
        sample.static_fraction.to_bits(),
        luma_sample.static_fraction.to_bits()
    );
    assert_eq!(gpu.accepted, luma_gpu.accepted);

    let block_count = plane.blocks();
    let counts = assert_channel_accepted(plane, &partials, &records);
    assert_eq!(counts, [block_count, block_count - textured.len()]);
}

#[test]
fn each_channel_rejects_its_own_outliers() {
    let plane = Plane::chroma(256, 192);
    let partials = hashed_partials(plane.width, plane.height, 107);
    let mut blocks = static_blocks(plane, 107);
    let channel_0_outliers = [5, 40, 77, 130];
    let channel_1_outliers = [9, 60, 101, 170];

    for index in channel_0_outliers {
        blocks[index].sigma[0] = 30.0 / 255.0;
    }

    for index in channel_1_outliers {
        blocks[index].sigma[1] = 30.0 / 255.0;
    }

    let records = plane.records(&blocks);
    let (gpu, host) = check_plane(plane, &partials, &records);

    let sample = gpu.sample.expect("the chroma frame has a sample");
    let host_sample = host.sample.expect("the host has a sample");
    assert_channel_excludes(sample, &blocks, 0, &channel_0_outliers);
    assert_channel_excludes(sample, &blocks, 1, &channel_1_outliers);
    assert_channel_excludes(host_sample, &blocks, 0, &channel_0_outliers);
    assert_channel_excludes(host_sample, &blocks, 1, &channel_1_outliers);

    let block_count = plane.blocks();
    let counts = assert_channel_accepted(plane, &partials, &records);
    assert_eq!(counts, [block_count - 4, block_count - 4]);
}

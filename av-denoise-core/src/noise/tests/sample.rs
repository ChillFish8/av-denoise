use cubecl::prelude::*;

use super::records::{BlockSpec, QuarterSpec, frame_records, hashed_partials, static_block, ulp_gap};
use crate::engine::align::StorageAlign;
use crate::noise::curve::{NOISE_CURVE_BINS, NoiseCurve};
use crate::noise::reading::{READING_CURVE_VALID, READING_SAMPLE_VALID, ReadingRecord};
use crate::noise::spatial::sigma_block_p25_from_partials;
use crate::noise::state::{BLOCK_ACCEPTED, NoiseShape, NoiseState, ReadingSettings, SlotStats};
use crate::noise::stats::lower_quartile;
use crate::noise::strength_map::QuarterSettings;
use crate::noise::temporal::{
    RHO_SIGMA_GATE,
    STATIC_GATE,
    TemporalNoiseSample,
    accepted_static_blocks,
    min_static_blocks,
    temporal_noise_reading,
    temporal_stats_blocks,
};
use crate::tune::gpu::sync;
use crate::tune::tests::runtime::{TestRuntime, test_client};

const MAX_ULPS: u32 = 2;
const RESULTS_ROW: [f32; 4] = [0.0; 4];

/// The geometry of one plane a test reads.
#[derive(Debug, Clone, Copy)]
struct Plane {
    width: u32,
    height: u32,
    channels: u32,
    stored_ch: u32,
}

const LUMA_1080P: Plane = Plane {
    width: 1920,
    height: 1080,
    channels: 1,
    stored_ch: 1,
};

const CHROMA_1080P: Plane = Plane {
    width: 960,
    height: 540,
    channels: 2,
    stored_ch: 2,
};

impl Plane {
    fn luma(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            channels: 1,
            stored_ch: 1,
        }
    }

    fn chroma(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            channels: 2,
            stored_ch: 2,
        }
    }

    fn blocks(self) -> usize {
        let (blocks_x, blocks_y) = temporal_stats_blocks(self.width, self.height);
        (blocks_x * blocks_y) as usize
    }

    fn records(self, blocks: &[BlockSpec]) -> Vec<f32> {
        frame_records(self.width, self.height, self.stored_ch, blocks)
    }
}

/// What the host reference reads from one slot's stats.
struct HostReading {
    immerkaer_low: [f32; 3],
    sample: Option<TemporalNoiseSample>,
    accepted: Vec<usize>,
}

fn host_reading(plane: Plane, partials: &[f32], records: &[f32]) -> HostReading {
    let immerkaer_low = sigma_block_p25_from_partials(partials, plane.channels, plane.width, plane.height);
    let reading = temporal_noise_reading(
        records,
        plane.channels,
        plane.stored_ch,
        plane.width,
        plane.height,
        false,
        QuarterSettings::default(),
    );
    let accepted_blocks = accepted_static_blocks(
        records,
        plane.channels,
        plane.stored_ch,
        plane.width,
        plane.height,
    );
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

fn shape_for(client: &ComputeClient<TestRuntime>, plane: Plane, temporal: bool) -> NoiseShape {
    NoiseShape {
        width: plane.width,
        height: plane.height,
        channels: plane.channels,
        stored_ch: plane.stored_ch,
        frames: 3,
        temporal,
        align: StorageAlign::from_client(client),
    }
}

fn state_for(
    client: &ComputeClient<TestRuntime>,
    plane: Plane,
    temporal: bool,
    settings: ReadingSettings,
    partials: &[f32],
    records: &[f32],
) -> NoiseState<TestRuntime> {
    let shape = shape_for(client, plane, temporal);

    NoiseState::from_stats_for_test(client, shape, settings, partials, RESULTS_ROW, records)
}

/// The reading of `slot` and, with temporal stats, the indices of the blocks the last reading
/// accepted.
fn read_back(state: &NoiseState<TestRuntime>, slot: u32) -> (ReadingRecord, Vec<usize>) {
    let reading = state.read_reading(slot);
    let flags = state.read_block_flags();
    let accepted = accepted_indices(&flags);

    (reading, accepted)
}

/// The GPU reading of slot 0 and the indices of the blocks it accepted.
fn gpu_reading(
    client: &ComputeClient<TestRuntime>,
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
    for channel in 0..3 {
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

    for channel in 0..3 {
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

#[test]
fn a_reading_survives_its_words() {
    let sample = TemporalNoiseSample {
        sigma: [0.01, 0.02, 0.0],
        sigma_low: [0.005, 0.015, 0.0],
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
        immerkaer_low: [0.004, 0.003, 0.0],
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
    let sigma_low_third = sample.sigma_low[2];
    assert_eq!(immerkaer_second, 0.0);
    assert_eq!(sigma_second, 0.0);
    assert_eq!(sigma_low_third, 0.0);
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
    let settings = ReadingSettings {
        luma_fields: true,
        ..ReadingSettings::default()
    };
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

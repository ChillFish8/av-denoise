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
use crate::noise::curve::{CLIP_HIGH, CLIP_LOW, FLAT_FACTOR, NOISE_CURVE_BINS, NoiseCurve};
use crate::noise::reading::ReadingRecord;
use crate::noise::select::UNTAGGED;
use crate::noise::state::{NoiseShape, NoiseState, ReadingSettings, SlotStats};
use crate::noise::strength_map::QuarterSettings;
use crate::noise::temporal::{
    STATIC_GATE,
    TEMPORAL_NOISE_BLOCK,
    TEMPORAL_QUARTER_SIZE,
    TEMPORAL_QUARTERS,
    TemporalNoiseSample,
    UNMEASURED_FLATNESS,
    temporal_noise_reading,
    temporal_stats_blocks,
};
use crate::tune::tests::runtime::{TestRuntime, test_client};

const MAX_ULPS: u32 = 2;
const RESULTS_ROW: [f32; 4] = [0.0; 4];
/// A flatness far past any flat gate.
const TEXTURED: f32 = 1.0;
/// Launches of a luma reading with the curve.
const CURVE_LAUNCHES: u32 = 13;
/// Launches of a luma reading without the curve.
const SAMPLE_LAUNCHES: u32 = 9;

/// A luma frame of static blocks whose quarters stay textured until a test places them in bins.
struct CurveFrame {
    width: u32,
    height: u32,
    blocks: Vec<BlockSpec>,
    /// The next quarter slot `place` fills, counted over blocks in row-major order.
    next: usize,
}

impl CurveFrame {
    fn new(width: u32, height: u32, seed: u32) -> Self {
        let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
        let count = blocks_x * blocks_y;
        let textured = QuarterSpec {
            flatness: TEXTURED,
            ..binned_quarter(8, 0.5)
        };

        let mut blocks = Vec::with_capacity(count as usize);
        for index in 0..count {
            let mut block = static_block(seed, index);
            block.quarters = [textured; 4];
            blocks.push(block);
        }

        Self {
            width,
            height,
            blocks,
            next: 0,
        }
    }

    /// Fills the next `count` quarter slots with `quarter(position)`, `position` counting from 0.
    fn place(&mut self, count: usize, quarter: impl Fn(u32) -> QuarterSpec) {
        for position in 0..count {
            let slot = self.next + position;
            let block = slot / TEMPORAL_QUARTERS as usize;
            let index = slot % TEMPORAL_QUARTERS as usize;
            self.blocks[block].quarters[index] = quarter(position as u32);
        }

        self.next += count;
    }

    /// Fills the next `count` quarter slots with flat quarters in `bin`.
    fn place_bin(&mut self, bin: u32, count: usize) {
        self.place(count, |position| {
            let unit = hashed_unit(bin + 100, position);
            binned_quarter(bin, unit)
        });
    }

    fn records(&self) -> Vec<f32> {
        frame_records(self.width, self.height, 1, &self.blocks)
    }
}

/// A flat, static, unclipped quarter at the centre of `bin`, with a sigma that grows with the bin.
fn binned_quarter(bin: u32, unit: f32) -> QuarterSpec {
    let luma = (bin as f32 + 0.5) / NOISE_CURVE_BINS as f32;
    let sigma = (3.0 + 0.5 * bin as f32 + 2.0 * unit) / 255.0;

    QuarterSpec {
        mean: 0.0,
        sigma,
        luma,
        flatness: 0.0,
        luma_min: luma - 0.01,
        luma_max: luma + 0.01,
    }
}

fn curve_settings(luma_fields: bool) -> ReadingSettings {
    ReadingSettings {
        luma_fields,
        ..ReadingSettings::default()
    }
}

fn luma_shape(client: &ComputeClient<TestRuntime>, width: u32, height: u32) -> NoiseShape {
    NoiseShape {
        width,
        height,
        channels: 1,
        stored_ch: 1,
        frames: 3,
        temporal: true,
        align: StorageAlign::from_client(client),
    }
}

/// A luma state whose slot 0 holds `records`.
fn curve_state(
    client: &ComputeClient<TestRuntime>,
    frame: &CurveFrame,
    records: &[f32],
    luma_fields: bool,
) -> NoiseState<TestRuntime> {
    let shape = luma_shape(client, frame.width, frame.height);
    let settings = curve_settings(luma_fields);
    let partials = hashed_partials(frame.width, frame.height, 7);

    NoiseState::from_stats_for_test(client, shape, settings, &partials, RESULTS_ROW, records)
}

/// The host's sample and curve for `records`.
fn host_curve(frame: &CurveFrame, records: &[f32]) -> (Option<TemporalNoiseSample>, Option<NoiseCurve>) {
    let reading = temporal_noise_reading(
        records,
        1,
        1,
        frame.width,
        frame.height,
        true,
        QuarterSettings::default(),
    );

    (reading.sample, reading.curve)
}

/// The GPU reading of `frame` with the curve and how many launches it took.
fn gpu_curve(
    client: &ComputeClient<TestRuntime>,
    frame: &CurveFrame,
    records: &[f32],
) -> (ReadingRecord, u32) {
    let state = curve_state(client, frame, records, true);
    let launches = state.queue_reading(0).expect("the reading queues");
    let reading = state.read_reading(0);

    (reading, launches)
}

fn assert_close(gpu: f32, host: f32, what: &str) {
    let gap = ulp_gap(gpu, host);
    assert!(gap <= MAX_ULPS, "{what}: gpu {gpu} host {host}, {gap} ulp apart");
}

fn assert_curve_matches(gpu: Option<NoiseCurve>, host: Option<NoiseCurve>) {
    let (gpu, host) = match (gpu, host) {
        (Some(gpu), Some(host)) => (gpu, host),
        (None, None) => return,
        (gpu, host) => panic!("the gpu curve {gpu:?} and the host curve {host:?} disagree"),
    };

    assert_close(
        gpu.sigma_quarter_median,
        host.sigma_quarter_median,
        "sigma_quarter_median",
    );

    for bin in 0..NOISE_CURVE_BINS {
        let what = format!("ratio[{bin}]");
        assert_close(gpu.ratios[bin], host.ratios[bin], &what);
    }
}

/// Runs `frame` on the GPU and on the host, checks the curves agree and returns the GPU reading.
fn check_frame(frame: &CurveFrame) -> (ReadingRecord, Option<NoiseCurve>) {
    let client = test_client();
    let records = frame.records();
    let (gpu, launches) = gpu_curve(&client, frame, &records);
    let (host_sample, host) = host_curve(frame, &records);
    let gpu_sampled = gpu.sample.is_some();
    let host_sampled = host_sample.is_some();

    assert_eq!(launches, CURVE_LAUNCHES, "launches");
    assert_eq!(gpu_sampled, host_sampled, "sample");
    assert_curve_matches(gpu.curve, host);

    (gpu, host)
}

#[test]
fn the_curve_matches_the_host_within_two_ulp_away_from_gates() {
    for seed in [3, 5] {
        let mut frame = CurveFrame::new(256, 192, seed);
        for bin in [1, 3, 7, 9, 13] {
            frame.place_bin(bin, 120);
        }

        let (gpu, host) = check_frame(&frame);

        let curve = host.expect("the host has a curve");
        let first_ratio = curve.ratios[0];
        let last_ratio = curve.ratios[NOISE_CURVE_BINS - 1];
        assert_eq!(gpu.populated_bins, 5);
        assert!(last_ratio > first_ratio);
    }
}

#[test]
fn a_bin_with_31_quarters_stays_empty_and_32_populates_it() {
    let mut ratios = Vec::new();

    for (count, populated) in [(31, 3), (32, 4)] {
        let mut frame = CurveFrame::new(256, 192, 11);
        for bin in [2, 6, 10] {
            frame.place_bin(bin, 40);
        }

        frame.place(count, |position| {
            let unit = hashed_unit(4, position);
            let quarter = binned_quarter(4, unit);

            QuarterSpec {
                sigma: 3.0 * quarter.sigma,
                ..quarter
            }
        });

        let (gpu, host) = check_frame(&frame);
        let curve = host.expect("the host has a curve");
        assert_eq!(gpu.populated_bins, populated);
        ratios.push(curve.ratios[4]);
    }

    let (empty, populated) = (ratios[0], ratios[1]);
    assert!(populated > empty);
}

#[test]
fn two_populated_bins_give_no_curve() {
    let mut frame = CurveFrame::new(256, 192, 13);
    frame.place_bin(3, 60);
    frame.place_bin(11, 60);

    let (gpu, host) = check_frame(&frame);
    let sampled = gpu.sample.is_some();
    let has_curve = host.is_some();
    assert!(sampled);
    assert!(!has_curve);
    assert_eq!(gpu.populated_bins, 2);

    frame.place_bin(7, 60);
    let (gpu, host) = check_frame(&frame);
    let has_curve = host.is_some();
    assert!(has_curve);
    assert_eq!(gpu.populated_bins, 3);
}

#[test]
fn the_flat_gate_uses_the_sample_median() {
    let client = test_client();
    let mut frame = CurveFrame::new(256, 192, 17);
    let blank = frame.records();
    let (sample, _) = host_curve(&frame, &blank);
    let sigma_median = sample.expect("the host has a sample").sigma[0];
    let flat_limit = FLAT_FACTOR * sigma_median * sigma_median;

    for bin in [2, 6, 10] {
        frame.place_bin(bin, 40);
    }

    // Quarters far noisier than the blocks, so a gate built from their median or from the blocks'
    // lower quartile lets a different set through.
    let first_rejected = frame.next;
    for (bin, scale) in [(4, 1.25), (13, 0.8)] {
        frame.place(40, |position| {
            let unit = hashed_unit(bin, position);
            let quarter = binned_quarter(bin, unit);

            QuarterSpec {
                sigma: 3.0 * quarter.sigma,
                flatness: scale * flat_limit,
                ..quarter
            }
        });
    }

    let records = frame.records();
    let state = curve_state(&client, &frame, &records, true);
    state.queue_reading(0).expect("the reading queues");
    let gpu = state.read_reading(0);
    let tags = state.read_curve_tags();
    let (_, host) = host_curve(&frame, &records);

    assert_curve_matches(gpu.curve, host);
    assert_eq!(gpu.populated_bins, 4);

    let rejected = &tags[first_rejected..first_rejected + 40];
    let passed = &tags[first_rejected + 40..first_rejected + 80];
    let all_rejected = rejected.iter().all(|&tag| tag == UNTAGGED);
    let all_passed = passed.iter().all(|&tag| tag == 13);
    assert!(all_rejected);
    assert!(all_passed);
}

#[test]
fn clipped_quarters_never_reach_a_bin() {
    let client = test_client();
    let mut frame = CurveFrame::new(256, 192, 19);
    for bin in [2, 6, 10] {
        frame.place_bin(bin, 40);
    }

    let first_clipped = frame.next;
    frame.place(40, |position| {
        let unit = hashed_unit(14, position);
        let quarter = binned_quarter(14, unit);

        QuarterSpec {
            luma_max: 0.99,
            ..quarter
        }
    });

    frame.place(40, |position| {
        let unit = hashed_unit(1, position);
        let quarter = binned_quarter(1, unit);

        QuarterSpec {
            luma_min: 0.01,
            ..quarter
        }
    });

    // Pixels exactly at either clip limit are not clipped.
    frame.place(40, |position| {
        let unit = hashed_unit(13, position);
        let quarter = binned_quarter(13, unit);

        QuarterSpec {
            luma_min: CLIP_LOW,
            luma_max: CLIP_HIGH,
            ..quarter
        }
    });

    let records = frame.records();
    let state = curve_state(&client, &frame, &records, true);
    state.queue_reading(0).expect("the reading queues");
    let gpu = state.read_reading(0);
    let tags = state.read_curve_tags();
    let (_, host) = host_curve(&frame, &records);

    assert_curve_matches(gpu.curve, host);
    assert_eq!(gpu.populated_bins, 4);

    let clipped = &tags[first_clipped..first_clipped + 80];
    let at_limits = &tags[first_clipped + 80..first_clipped + 120];
    let all_clipped = clipped.iter().all(|&tag| tag == UNTAGGED);
    let all_kept = at_limits.iter().all(|&tag| tag == 13);
    assert!(all_clipped);
    assert!(all_kept);
}

/// The pixels quarter `quarter` of block `block` holds in a `width` by `height` frame.
fn quarter_pixels(width: u32, height: u32, block: u32, quarter: u32) -> u32 {
    let (blocks_x, _) = temporal_stats_blocks(width, height);
    let block_x = block % blocks_x;
    let block_y = block / blocks_x;
    let block_width = TEMPORAL_NOISE_BLOCK.min(width - block_x * TEMPORAL_NOISE_BLOCK);
    let block_height = TEMPORAL_NOISE_BLOCK.min(height - block_y * TEMPORAL_NOISE_BLOCK);
    let offset_x = (quarter % 2) * TEMPORAL_QUARTER_SIZE;
    let offset_y = (quarter / 2) * TEMPORAL_QUARTER_SIZE;
    let quarter_width = block_width.saturating_sub(offset_x).min(TEMPORAL_QUARTER_SIZE);
    let quarter_height = block_height.saturating_sub(offset_y).min(TEMPORAL_QUARTER_SIZE);

    quarter_width * quarter_height
}

#[test]
fn ragged_quarters_never_reach_a_bin() {
    let client = test_client();
    let mut frame = CurveFrame::new(70, 54, 23);
    let full_pixels = TEMPORAL_QUARTER_SIZE * TEMPORAL_QUARTER_SIZE;
    let slots = frame.blocks.len() * TEMPORAL_QUARTERS as usize;
    let (width, height) = (frame.width, frame.height);

    // A quarter smaller than 8x8 reads the unmeasured flatness, as the stats kernel writes it.
    frame.place(slots, |slot| {
        let block = slot / TEMPORAL_QUARTERS;
        let quarter = slot % TEMPORAL_QUARTERS;
        let pixels = quarter_pixels(width, height, block, quarter);
        let ragged = pixels > 0 && pixels < full_pixels;
        let unit = hashed_unit(8, slot);
        let binned = binned_quarter(8, unit);
        let flatness = if ragged { UNMEASURED_FLATNESS } else { 0.0 };

        QuarterSpec { flatness, ..binned }
    });

    let records = frame.records();
    let state = curve_state(&client, &frame, &records, true);
    state.queue_reading(0).expect("the reading queues");
    let gpu = state.read_reading(0);
    let tags = state.read_curve_tags();
    let (_, host) = host_curve(&frame, &records);

    assert_curve_matches(gpu.curve, host);
    assert_eq!(gpu.populated_bins, 1);

    let mut full_quarters = 0;
    for (slot, &tag) in tags.iter().enumerate() {
        let block = slot as u32 / TEMPORAL_QUARTERS;
        let quarter = slot as u32 % TEMPORAL_QUARTERS;
        let pixels = quarter_pixels(width, height, block, quarter);
        let full = pixels == full_pixels;
        let expected = if full { 8 } else { UNTAGGED };

        assert_eq!(
            tag, expected,
            "quarter {quarter} of block {block}, {pixels} pixels"
        );
        full_quarters += usize::from(full);
    }

    assert_eq!(full_quarters, 48);
}

#[test]
fn no_curve_without_a_sample() {
    for (accepted, sampled) in [(40, false), (46, true)] {
        let mut frame = CurveFrame::new(640, 360, 29);
        let block_count = frame.blocks.len();
        let moving_quarter = binned_quarter(12, 0.5);

        // Moving blocks carry quarters that would fill bin 12 if they were ever binned.
        for position in accepted..block_count {
            let mut block = BlockSpec::uniform(4.0 * STATIC_GATE, 3.0 / 255.0, 0.2);
            block.quarters = [moving_quarter; 4];
            frame.blocks[position] = block;
        }

        for bin in [2, 6, 10] {
            frame.place_bin(bin, 50);
        }

        let (gpu, host) = check_frame(&frame);
        let gpu_sampled = gpu.sample.is_some();
        let has_curve = host.is_some();
        assert_eq!(gpu_sampled, sampled);
        assert_eq!(has_curve, sampled);
        assert_eq!(gpu.populated_bins, 3);
    }
}

#[test]
fn luma_fields_off_never_builds_a_curve() {
    let client = test_client();
    let mut frame = CurveFrame::new(256, 192, 31);
    for bin in [2, 6, 10] {
        frame.place_bin(bin, 40);
    }

    let records = frame.records();
    let (_, host) = host_curve(&frame, &records);
    let host_has_curve = host.is_some();
    assert!(host_has_curve);

    let state = curve_state(&client, &frame, &records, false);
    let has_batch = state.has_curve_batch();
    let launches = state.queue_reading(0).expect("the reading queues");
    let gpu = state.read_reading(0);
    let gpu_has_curve = gpu.curve.is_some();
    assert!(!has_batch);
    assert_eq!(launches, SAMPLE_LAUNCHES);
    assert!(!gpu_has_curve);
    assert_eq!(gpu.populated_bins, 0);

    let mut state = curve_state(&client, &frame, &records, true);
    state.set_settings(curve_settings(false));
    let has_batch = state.has_curve_batch();
    let launches = state.queue_reading(0).expect("the reading queues");
    assert!(!has_batch);
    assert_eq!(launches, SAMPLE_LAUNCHES);

    state.set_settings(curve_settings(true));
    let has_batch = state.has_curve_batch();
    let launches = state.queue_reading(1).expect("the reading queues");
    assert!(has_batch);
    assert_eq!(launches, CURVE_LAUNCHES);

    let launches = state.queue_reading(0).expect("the reading queues");
    let gpu = state.read_reading(0);
    assert_eq!(launches, CURVE_LAUNCHES);
    assert_curve_matches(gpu.curve, host);
}

#[test]
fn a_second_slot_builds_its_own_curve() {
    let client = test_client();
    let mut first = CurveFrame::new(256, 192, 37);
    for bin in [1, 5, 9] {
        first.place_bin(bin, 60);
    }

    let mut second = CurveFrame::new(256, 192, 41);
    for bin in [3, 7, 11, 15] {
        second.place_bin(bin, 60);
    }

    let first_records = first.records();
    let second_records = second.records();
    let first_partials = hashed_partials(256, 192, 43);
    let second_partials = hashed_partials(256, 192, 47);
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
    let shape = luma_shape(&client, 256, 192);
    let settings = curve_settings(true);
    let state = NoiseState::from_slots_for_test(&client, shape, settings, &slots);

    state.queue_reading(0).expect("the first reading queues");
    state.queue_reading(1).expect("the second reading queues");

    let first_gpu = state.read_reading(0);
    let second_gpu = state.read_reading(1);
    let (_, first_host) = host_curve(&first, &first_records);
    let (_, second_host) = host_curve(&second, &second_records);

    assert_curve_matches(first_gpu.curve, first_host);
    assert_curve_matches(second_gpu.curve, second_host);
    assert_eq!(first_gpu.populated_bins, 3);
    assert_eq!(second_gpu.populated_bins, 4);
}

#[test]
fn stale_curve_tags_never_join_a_bin() {
    let client = test_client();
    let mut frame = CurveFrame::new(256, 192, 53);
    for bin in [2, 6, 10] {
        frame.place_bin(bin, 40);
    }

    let records = frame.records();
    let mut state = curve_state(&client, &frame, &records, true);
    state.poison_for_test();
    state.queue_reading(0).expect("the reading queues");

    let gpu = state.read_reading(0);
    let (_, host) = host_curve(&frame, &records);
    assert_curve_matches(gpu.curve, host);
    assert_eq!(gpu.populated_bins, 3);
}

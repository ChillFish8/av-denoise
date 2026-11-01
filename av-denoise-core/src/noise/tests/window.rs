use cubecl::prelude::*;
use cubecl::server::Handle;

use super::fold_reference::{
    COEFFICIENTS,
    CONSUMER,
    HostRecord,
    derived_pairs,
    fold_window,
    largest_relative,
    relative_gap,
    state_pairs,
};
use super::records::{hashed_unit, valid_channels};
use crate::engine::align::StorageAlign;
use crate::noise::NOISE_CHANNELS;
use crate::noise::correlation::build_spatial_offset_lut;
use crate::noise::curve::{NOISE_CURVE_BINS, NoiseCurve};
use crate::noise::derived::DerivedValues;
use crate::noise::estimator::SIGMA_FLOOR;
use crate::noise::fold::{FoldSettings, NoiseSettings};
use crate::noise::history::{NOISE_WINDOW_RADIUS, ReadingHistory, WindowBounds, history_capacity};
use crate::noise::reading::{READING_WORDS, ReadingRecord};
use crate::noise::snapshot::NoiseSnapshot;
use crate::noise::spatial::{RESULTS_PER_SLOT, immerkaer_sigma_scale};
use crate::noise::state::{NoiseShape, NoiseState, ReadingSettings};
use crate::noise::temporal::TemporalNoiseSample;
use crate::tune::tests::runtime::test_client;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const RING_FRAMES: u32 = 3;
const TEMPORAL_RADIUS: u32 = 1;
const TABLE_RADIUS: u32 = 2;
const MAX_RELATIVE: f32 = 1.0e-5;
const RECORD_COUNT: u32 = 40;

fn shape(client: &Client, channels: u32) -> NoiseShape {
    NoiseShape {
        width: WIDTH,
        height: HEIGHT,
        channels,
        frames: RING_FRAMES,
        ring_radius: RING_FRAMES / 2,
        temporal: true,
        align: StorageAlign::from_client(client),
    }
}

fn settings(sigma_scale: f32) -> NoiseSettings {
    let reading = ReadingSettings { luma_fields: true };
    let fold = FoldSettings {
        sigma_scale,
        consumer: CONSUMER,
        coefficients: COEFFICIENTS,
        offset_tables: Some(TABLE_RADIUS),
    };

    NoiseSettings { reading, fold }
}

/// The abs sum whose Immerkær sigma is about `sigma`.
fn abs_sum_for(sigma: f32) -> f32 {
    let interior = ((WIDTH - 2) * (HEIGHT - 2)) as f32;
    sigma * 6.0 * interior / immerkaer_sigma_scale()
}

/// A record with hashed values on the active channels and, when asked, a sample and a curve.
fn hashed_record(seed: u32, channels: u32, sampled: bool, curved: bool) -> HostRecord {
    let mut immerkaer = [0.0f32; NOISE_CHANNELS as usize];
    let mut immerkaer_low = [0.0f32; NOISE_CHANNELS as usize];
    let mut sigma = [0.0f32; NOISE_CHANNELS as usize];
    let mut sigma_low = [0.0f32; NOISE_CHANNELS as usize];
    for channel in 0..channels as usize {
        let index = channel as u32;
        immerkaer[channel] = (1.0 + 4.0 * hashed_unit(seed, 10 + index)) / 255.0;
        immerkaer_low[channel] = (1.0 + 2.0 * hashed_unit(seed, index)) / 255.0;
        sigma[channel] = (2.0 + 2.0 * hashed_unit(seed, 20 + index)) / 255.0;
        sigma_low[channel] = sigma[channel] * (0.7 + 0.2 * hashed_unit(seed, 30 + index));
    }

    let sample = sampled.then_some(TemporalNoiseSample {
        valid: valid_channels(channels),
        sigma,
        sigma_low,
        rho: 0.75 * hashed_unit(seed, 40),
        static_fraction: 0.5,
    });

    let mut ratios = [0.0f32; NOISE_CURVE_BINS];
    for (bin, ratio) in ratios.iter_mut().enumerate() {
        *ratio = 0.8 + 0.4 * hashed_unit(seed, 50 + bin as u32);
    }

    let curve = (sampled && curved).then_some(NoiseCurve {
        ratios,
        sigma_quarter_median: (2.0 + hashed_unit(seed, 70)) / 255.0,
    });

    let reading = ReadingRecord {
        immerkaer_low,
        sample,
        accepted: 40,
        curve,
        populated_bins: if curve.is_some() { 6 } else { 0 },
    };

    HostRecord { reading, immerkaer }
}

/// Records whose every third frame has no sample and every fourth sampled frame no curve.
fn hashed_records(count: u32, channels: u32, seed: u32) -> Vec<HostRecord> {
    (0..count)
        .map(|frame| hashed_record(seed + frame, channels, frame % 3 != 1, frame % 4 != 2))
        .collect()
}

/// A record with every value and the curve ratios scaled by `level`, and a sample of correlation
/// 0.2.
fn level_record(level: f32) -> HostRecord {
    let sample = TemporalNoiseSample {
        valid: valid_channels(1),
        sigma: [level, 0.0],
        sigma_low: [level * 0.8, 0.0],
        rho: 0.2,
        static_fraction: 0.5,
    };
    let codes = level * 255.0;
    let curve = NoiseCurve {
        ratios: [0.5 * codes; NOISE_CURVE_BINS],
        sigma_quarter_median: level,
    };
    let reading = ReadingRecord {
        immerkaer_low: [level * 0.5, 0.0],
        sample: Some(sample),
        accepted: 40,
        curve: Some(curve),
        populated_bins: 6,
    };

    HostRecord {
        reading,
        immerkaer: [level * 0.75, 0.0],
    }
}

/// A history holding frames `0..=newest` of `records`, each in the slot its frame number names.
fn upload_history(client: &Client, records: &[HostRecord], newest: u32) -> ReadingHistory {
    let capacity = history_capacity(TEMPORAL_RADIUS) as usize;
    let reading_words = READING_WORDS as usize;
    let mut words = vec![0.0f32; capacity * reading_words];
    let mut results = vec![0.0f32; capacity * RESULTS_PER_SLOT];

    let recorded = &records[..=newest as usize];
    for (frame, record) in recorded.iter().enumerate() {
        let slot = frame % capacity;
        let reading = record.reading.to_words();
        let words_start = slot * reading_words;
        words[words_start..words_start + reading_words].copy_from_slice(&reading);

        let results_start = slot * RESULTS_PER_SLOT;
        for channel in 0..RESULTS_PER_SLOT {
            results[results_start + channel] = abs_sum_for(record.immerkaer[channel]);
        }
    }

    ReadingHistory::from_words(client, TEMPORAL_RADIUS, &words, &results)
}

/// Everything one window fold writes.
#[derive(Debug, Clone, PartialEq)]
struct WindowRun {
    snapshot: NoiseSnapshot,
    curve: Vec<f32>,
    spatial: Vec<f32>,
    pilot: Vec<f32>,
}

impl WindowRun {
    fn bits(&self) -> Vec<u32> {
        let derived = self.snapshot.derived.to_words();
        let words = derived
            .iter()
            .chain(&self.curve)
            .chain(&self.spatial)
            .chain(&self.pilot);

        words.map(|word| word.to_bits()).collect()
    }
}

fn read_words(client: &Client, handle: &Handle) -> Vec<f32> {
    let bytes = client.read_one(handle.clone()).expect("the buffer reads back");
    f32::from_bytes(&bytes).to_vec()
}

/// Folds the window `bounds` of a history holding frames `0..=newest` of `records`.
fn run_window(
    channels: u32,
    settings: NoiseSettings,
    records: &[HostRecord],
    bounds: WindowBounds,
    newest: u32,
) -> WindowRun {
    let client = test_client();
    let shape = shape(&client, channels);
    let state = NoiseState::new(&client, shape, settings, DerivedValues::default());
    let history = upload_history(&client, records, newest);

    state
        .fold_history(&history, bounds)
        .expect("the window fold queues");

    let snapshot = state.snapshot().expect("the snapshot reads back");
    let spatial_handle = state.spatial_offsets().expect("the state keeps tables");
    let pilot_handle = state.pilot_offsets().expect("the state keeps tables");

    WindowRun {
        snapshot,
        curve: state.read_curve(),
        spatial: read_words(&client, spatial_handle),
        pilot: read_words(&client, pilot_handle),
    }
}

/// Checks `run` against the host model of the same window.
fn assert_matches_host(
    run: &WindowRun,
    records: &[HostRecord],
    bounds: WindowBounds,
    settings: NoiseSettings,
    channels: u32,
) {
    let host = fold_window(records, bounds, settings.fold.sigma_scale);
    let expected = host.snapshot(settings, channels);

    let mut pairs = state_pairs(&run.snapshot, &expected, channels);
    let derived = derived_pairs(&run.snapshot.derived, &expected.derived, channels);
    pairs.extend(derived);
    let relative = largest_relative(&pairs);
    assert!(relative <= MAX_RELATIVE, "the window is {relative:e} apart");

    let expected_curve = match expected.curve {
        Some((ratios, _)) if expected.derived.curve_valid => ratios.to_vec(),
        _ => vec![0.0; NOISE_CURVE_BINS],
    };
    assert_eq!(run.curve, expected_curve, "the curve buffer");

    let derived_values = run.snapshot.derived;
    let spatial = build_spatial_offset_lut(TABLE_RADIUS, derived_values.rho, derived_values.noise_offset);
    let pilot = build_spatial_offset_lut(TABLE_RADIUS, 0.0, derived_values.input_noise_offset);
    let table_pairs = run
        .spatial
        .iter()
        .zip(&spatial)
        .chain(run.pilot.iter().zip(&pilot));
    for (&gpu, &host) in table_pairs {
        let gap = relative_gap(gpu, host);
        assert!(gap <= MAX_RELATIVE, "a table entry is {gap:e} apart");
    }
}

#[test]
fn window_fold_matches_the_host_model() {
    let cases = [(1, settings(1.1)), (2, settings(1.0)), (1, settings(0.9))];
    let newest = RECORD_COUNT - 1;
    let centres = [0, RECORD_COUNT / 2, newest];

    for (channels, case_settings) in cases {
        let records = hashed_records(RECORD_COUNT, channels, 17);

        for centre in centres {
            let bounds = WindowBounds::around(centre, newest);
            let run = run_window(channels, case_settings, &records, bounds, bounds.last());

            assert_matches_host(&run, &records, bounds, case_settings, channels);
        }
    }
}

#[test]
fn one_spiked_record_does_not_move_the_window() {
    let window = 2 * NOISE_WINDOW_RADIUS + 1;
    let newest = window - 1;
    let calm: Vec<HostRecord> = (0..window).map(|_| level_record(2.0 / 255.0)).collect();
    let mut spiked = calm.clone();
    spiked[5] = level_record(6.0 / 255.0);

    let bounds = WindowBounds::around(NOISE_WINDOW_RADIUS, newest);
    let case_settings = settings(1.0);
    let calm_run = run_window(1, case_settings, &calm, bounds, newest);
    let spiked_run = run_window(1, case_settings, &spiked, bounds, newest);

    assert_eq!(bounds.len(), window);
    assert_eq!(spiked_run.bits(), calm_run.bits());
}

#[test]
fn a_window_without_samples_falls_back_to_immerkaer() {
    let records: Vec<HostRecord> = (0..9)
        .map(|frame| hashed_record(frame + 3, 1, false, false))
        .collect();
    let bounds = WindowBounds::around(4, 8);
    let case_settings = settings(1.0);
    let run = run_window(1, case_settings, &records, bounds, 8);

    let mut unboosted: Vec<f32> = records
        .iter()
        .map(|record| record.reading.immerkaer_low[0])
        .collect();
    unboosted.sort_by(f32::total_cmp);
    let expected = unboosted[4].max(SIGMA_FLOOR);
    let derived = run.snapshot.derived;
    let words = derived.to_words();
    let zeros = vec![0.0f32; NOISE_CURVE_BINS];

    assert_eq!(derived.sigma[0], expected);
    assert_eq!(run.snapshot.temporal_only, None);
    assert_eq!(run.snapshot.rho, None);
    assert_eq!(run.snapshot.curve, None);
    assert_eq!(derived.rho, 0.0);
    assert!(words.iter().all(|word| word.is_finite()));
    assert_eq!(run.curve, zeros);
}

#[test]
fn a_clipped_window_uses_only_frames_inside_the_stream() {
    let newest = history_capacity(TEMPORAL_RADIUS) - 1;
    let inside = hashed_records(newest + 1, 1, 5);
    let mut outside = inside.clone();
    for record in outside.iter_mut().skip(9) {
        *record = level_record(1000.0 / 255.0);
    }

    let bounds = WindowBounds::around(4, 8);
    let case_settings = settings(1.0);
    let inside_run = run_window(1, case_settings, &inside, bounds, newest);
    let outside_run = run_window(1, case_settings, &outside, bounds, newest);

    assert_eq!((bounds.first(), bounds.last()), (0, 8));
    assert_eq!(bounds.len(), 9);
    assert_eq!(outside_run.bits(), inside_run.bits());
    assert_matches_host(&outside_run, &outside, bounds, case_settings, 1);
}

#[test]
fn curve_ratios_take_the_elementwise_median() {
    let mut records: Vec<HostRecord> = (0..3)
        .map(|seed| hashed_record(seed + 60, 1, true, true))
        .collect();
    for (index, record) in records.iter_mut().enumerate() {
        let curve = record.reading.curve.as_mut().expect("the record has a curve");
        for (bin, ratio) in curve.ratios.iter_mut().enumerate() {
            let rank = (bin + index) % 3;
            *ratio = 0.9 + 0.1 * rank as f32 + 0.001 * index as f32;
        }
    }

    let bounds = WindowBounds::around(1, 2);
    let case_settings = settings(1.0);
    let run = run_window(1, case_settings, &records, bounds, 2);

    let middle = |value_of: &dyn Fn(&HostRecord) -> f32| {
        let mut values: Vec<f32> = records.iter().map(value_of).collect();
        values.sort_by(f32::total_cmp);
        values[1]
    };
    let expected_curve: Vec<f32> = (0..NOISE_CURVE_BINS)
        .map(|bin| middle(&|record| record.reading.curve.unwrap().ratios[bin]))
        .collect();

    for record in &records {
        let ratios = record.reading.curve.unwrap().ratios.to_vec();
        assert_ne!(
            ratios, expected_curve,
            "one record already holds the median curve"
        );
    }

    assert_eq!(run.curve, expected_curve);
    assert_matches_host(&run, &records, bounds, case_settings, 1);
}

#[test]
fn the_window_fold_is_byte_identical_across_runs() {
    let records = hashed_records(RECORD_COUNT, 1, 23);
    let newest = RECORD_COUNT - 1;
    let bounds = WindowBounds::around(RECORD_COUNT / 2, newest);
    let case_settings = settings(1.0);

    let first = run_window(1, case_settings, &records, bounds, bounds.last());
    let second = run_window(1, case_settings, &records, bounds, bounds.last());

    assert_eq!(first.bits(), second.bits());
}

/// Clears channel `channel`'s sample from `record`, as a reading writes a starved channel.
///
/// A starved channel 0 also clears the correlation, the static fraction and the curve.
fn starve_channel(record: &mut HostRecord, channel: usize) {
    let reading = &mut record.reading;
    let Some(sample) = reading.sample.as_mut() else {
        return;
    };

    sample.valid[channel] = false;
    sample.sigma[channel] = 0.0;
    sample.sigma_low[channel] = 0.0;

    if channel == 0 {
        sample.rho = 0.0;
        sample.static_fraction = 0.0;
        reading.curve = None;
        reading.populated_bins = 0;
    }

    if !sample.valid.contains(&true) {
        reading.sample = None;
    }
}

/// The floored median of `values`, the mean of the two middle values for an even count.
fn floored_median(mut values: Vec<f32>) -> f32 {
    values.sort_by(f32::total_cmp);

    let count = values.len();
    let lower = values[(count - 1) / 2];
    let upper = values[count / 2];
    let median = (lower + upper) * 0.5;

    median.max(SIGMA_FLOOR)
}

/// Channel `channel`'s sigma over the frames of `records` with a sample of it.
fn sampled_sigmas(records: &[HostRecord], channel: usize) -> Vec<f32> {
    records
        .iter()
        .filter_map(|record| record.reading.sample)
        .filter(|sample| sample.valid[channel])
        .map(|sample| sample.sigma[channel])
        .collect()
}

/// Two-channel records where some frames starve channel 1 and some starve channel 0.
fn starved_records(count: u32, seed: u32) -> Vec<HostRecord> {
    let mut records = hashed_records(count, 2, seed);

    for (frame, record) in records.iter_mut().enumerate() {
        if frame % 5 == 3 {
            starve_channel(record, 1);
        }

        if frame % 7 == 0 {
            starve_channel(record, 0);
        }
    }

    records
}

#[test]
fn per_channel_samples_match_the_host_model() {
    let newest = RECORD_COUNT - 1;
    let records = starved_records(RECORD_COUNT, 37);
    let case_settings = settings(1.05);

    for centre in [0, 7, RECORD_COUNT / 2, newest] {
        let bounds = WindowBounds::around(centre, newest);
        let run = run_window(2, case_settings, &records, bounds, bounds.last());

        assert_matches_host(&run, &records, bounds, case_settings, 2);
    }
}

#[test]
fn a_starved_channel_keeps_its_windowed_value_from_neighbouring_frames() {
    let mut records: Vec<HostRecord> = (0..5)
        .map(|frame| hashed_record(frame + 80, 2, true, true))
        .collect();
    starve_channel(&mut records[2], 1);

    let bounds = WindowBounds::around(2, 4);
    let case_settings = settings(1.0);
    let run = run_window(2, case_settings, &records, bounds, 4);

    let channel_0 = floored_median(sampled_sigmas(&records, 0));
    let channel_1 = floored_median(sampled_sigmas(&records, 1));
    let temporal = run.snapshot.temporal_only.expect("the window has samples");
    let derived = run.snapshot.derived;

    assert_eq!(bounds.len(), 5);
    assert_eq!(sampled_sigmas(&records, 1).len(), 4);
    assert_eq!(temporal, [channel_0, channel_1]);
    assert_eq!(derived.sigma, [channel_0, channel_1]);
    assert_matches_host(&run, &records, bounds, case_settings, 2);
}

#[test]
fn a_channel_starved_for_the_whole_window_falls_back_alone() {
    let fed: Vec<HostRecord> = (0..9)
        .map(|frame| hashed_record(frame + 90, 2, true, true))
        .collect();
    let mut starved = fed.clone();
    for record in &mut starved {
        starve_channel(record, 1);
    }

    let bounds = WindowBounds::around(4, 8);
    let case_settings = settings(1.0);
    let fed_run = run_window(2, case_settings, &fed, bounds, 8);
    let starved_run = run_window(2, case_settings, &starved, bounds, 8);

    let immerkaer_low: Vec<f32> = starved
        .iter()
        .map(|record| record.reading.immerkaer_low[1])
        .collect();
    let fallback = floored_median(immerkaer_low);
    let snapshot = &starved_run.snapshot;
    let fed_snapshot = &fed_run.snapshot;
    let temporal = snapshot.temporal_only.expect("channel 0 has samples");
    let fed_temporal = fed_snapshot.temporal_only.expect("channel 0 has samples");
    let low_unboosted = snapshot.low_unboosted.expect("the low chain is written");
    let derived = snapshot.derived;

    assert!(fallback.is_finite() && fallback > 0.0);
    assert_eq!(low_unboosted[1], fallback);
    assert_eq!(temporal[1], fallback);
    assert_eq!(derived.sigma[1], fallback);
    assert_eq!(temporal[0], fed_temporal[0]);
    assert_eq!(derived.sigma[0], fed_snapshot.derived.sigma[0]);
    assert_eq!(snapshot.rho, fed_snapshot.rho);
    assert_eq!(snapshot.curve, fed_snapshot.curve);
    assert_matches_host(&starved_run, &starved, bounds, case_settings, 2);
}

/// The windowed chains, rho and curve of `snapshot`, in the order the stream state holds them.
fn snapshot_words(snapshot: &NoiseSnapshot) -> Vec<f32> {
    let mut words = Vec::new();
    let chains = [
        snapshot.median,
        snapshot.low,
        snapshot.low_unboosted,
        snapshot.temporal_only,
    ];
    for chain in chains {
        let flag = if chain.is_some() { 1.0 } else { 0.0 };
        words.push(flag);
        words.extend(chain.unwrap_or_default());
    }

    let rho_flag = if snapshot.rho.is_some() { 1.0 } else { 0.0 };
    words.push(rho_flag);
    words.push(snapshot.rho.unwrap_or_default());

    let curve_flag = if snapshot.curve.is_some() { 1.0 } else { 0.0 };
    let (ratios, quarter_median) = snapshot.curve.unwrap_or(([0.0; NOISE_CURVE_BINS], 0.0));
    words.push(curve_flag);
    words.push(quarter_median);
    words.extend(ratios);

    words
}

/// The state and output words of two single-channel windows, recorded before each channel had its
/// own sample.
///
/// The first window mixes frames with and without samples and curves. The second has no sample.
const RECORDED_WINDOW_BITS: [u32; 218] = [
    0x3f800000, 0x3c522755, 0x00000000, 0x3f800000, 0x3c29fecc, 0x00000000, 0x3f800000, 0x3c1fdb37,
    0x00000000, 0x3f800000, 0x3c307c08, 0x00000000, 0x3f800000, 0x3ea24aa2, 0x3f800000, 0x3c24aefd,
    0x3f7d91e6, 0x3f7ffb06, 0x3f86f1d6, 0x3f840418, 0x3f7a07a2, 0x3f80df75, 0x3f820aee, 0x3f7bbef2,
    0x3f8d0cc8, 0x3f7f041a, 0x3f71248f, 0x3f8628b4, 0x3f8c7216, 0x3f811e65, 0x3f6b46ba, 0x3f718c2b,
    0x3c307c08, 0x00000000, 0x394d9e0d, 0x3f800000, 0x3fc504e0, 0x4013c3a8, 0x3e59d696, 0x3e0b6a9d,
    0x3bb061c1, 0x3bb061c1, 0x3ea24aa2, 0x3f7d91e6, 0x3f7ffb06, 0x3f86f1d6, 0x3f840418, 0x3f7a07a2,
    0x3f80df75, 0x3f820aee, 0x3f7bbef2, 0x3f8d0cc8, 0x3f7f041a, 0x3f71248f, 0x3f8628b4, 0x3f8c7216,
    0x3f811e65, 0x3f6b46ba, 0x3f718c2b, 0x3ba98a5e, 0x3ba2deba, 0x3b9ea8ff, 0x3ba2deba, 0x3ba98a5e,
    0x3ba2deba, 0x3b8da4f7, 0x3b70f23a, 0x3b8da4f7, 0x3ba2deba, 0x3b9ea8ff, 0x3b70f23a, 0x00000000,
    0x3b70f23a, 0x3b9ea8ff, 0x3ba2deba, 0x3b8da4f7, 0x3b70f23a, 0x3b8da4f7, 0x3ba2deba, 0x3ba98a5e,
    0x3ba2deba, 0x3b9ea8ff, 0x3ba2deba, 0x3ba98a5e, 0x3bb061c1, 0x3bb061c1, 0x3bb061c1, 0x3bb061c1,
    0x3bb061c1, 0x3bb061c1, 0x3bb061c1, 0x3bb061c1, 0x3bb061c1, 0x3bb061c1, 0x3bb061c1, 0x3bb061c1,
    0x00000000, 0x3bb061c1, 0x3bb061c1, 0x3bb061c1, 0x3bb061c1, 0x3bb061c1, 0x3bb061c1, 0x3bb061c1,
    0x3bb061c1, 0x3bb061c1, 0x3bb061c1, 0x3bb061c1, 0x3bb061c1, 0x3f800000, 0x3c262ed6, 0x00000000,
    0x3f800000, 0x3bfc15e4, 0x00000000, 0x3f800000, 0x3bfc15e4, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x3bfc15e4, 0x00000000, 0x38d1c14d,
    0x00000000, 0x3f9bcbe9, 0x3fe9b1dd, 0x3eae2ec3, 0x3e5ef42c, 0x3b41ee2d, 0x3b41ee2d, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d,
    0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x00000000, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d,
    0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d,
    0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d,
    0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x00000000, 0x3b41ee2d, 0x3b41ee2d,
    0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d, 0x3b41ee2d,
    0x3b41ee2d, 0x3b41ee2d,
];

#[test]
fn a_single_channel_window_fold_matches_today() {
    let newest = RECORD_COUNT - 1;
    let mixed = hashed_records(RECORD_COUNT, 1, 31);
    let unsampled: Vec<HostRecord> = (0..RECORD_COUNT)
        .map(|frame| hashed_record(frame + 41, 1, false, false))
        .collect();
    let case_settings = settings(1.1);
    let bounds = WindowBounds::around(RECORD_COUNT / 2, newest);
    let mut bits = Vec::new();

    for records in [&mixed, &unsampled] {
        let run = run_window(1, case_settings, records, bounds, bounds.last());
        let state = snapshot_words(&run.snapshot);
        let state_bits = state.iter().map(|word| word.to_bits());

        bits.extend(state_bits);
        bits.extend(run.bits());
    }

    assert_eq!(bits, RECORDED_WINDOW_BITS);
}

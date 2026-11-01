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
use super::records::hashed_unit;
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

use cubecl::prelude::*;

use super::fold_reference::{
    COEFFICIENTS,
    CONSUMER,
    HostRecord,
    MAX_ULPS,
    derived_pairs,
    fold_window,
    largest_relative,
    state_pairs,
};
use super::records::{hashed_unit, ulp_gap};
use crate::engine::align::StorageAlign;
use crate::noise::NOISE_CHANNELS;
use crate::noise::correlation::{CORRELATION_FACTOR_TABLE, correlation_factor};
use crate::noise::curve::{NOISE_CURVE_BINS, NoiseCurve};
use crate::noise::derived::{DerivedCoefficients, DerivedValues, NoiseConsumer};
use crate::noise::estimator::SIGMA_FLOOR;
use crate::noise::fold::{FoldSettings, NoiseSettings};
use crate::noise::history::WindowBounds;
use crate::noise::reading::ReadingRecord;
use crate::noise::snapshot::NoiseSnapshot;
use crate::noise::spatial::{RESULTS_PER_SLOT, immerkaer_sigma_scale, sigma_from_abs_sum};
use crate::noise::state::{NoiseShape, NoiseState, ReadingSettings};
use crate::noise::temporal::TemporalNoiseSample;
use crate::tune::gpu::sync;
use crate::tune::tests::runtime::test_client;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const FRAMES: u32 = 5;
const RADIUS: u32 = 2;
const MAX_RELATIVE: f32 = 1.0e-5;
const STREAM_FRAMES: u32 = 30;

fn shape(client: &Client, channels: u32) -> NoiseShape {
    NoiseShape {
        width: WIDTH,
        height: HEIGHT,
        channels,
        frames: FRAMES,
        ring_radius: FRAMES / 2,
        temporal: true,
        align: StorageAlign::from_client(client),
    }
}

fn settings() -> NoiseSettings {
    let reading = ReadingSettings { luma_fields: true };
    let fold = FoldSettings {
        sigma_scale: 1.0,
        consumer: CONSUMER,
        coefficients: COEFFICIENTS,
        offset_tables: None,
    };

    NoiseSettings { reading, fold }
}

/// The abs sum whose Immerkær sigma is about `sigma`.
fn abs_sum_for(sigma: f32) -> f32 {
    let interior = ((WIDTH - 2) * (HEIGHT - 2)) as f32;
    sigma * 6.0 * interior / immerkaer_sigma_scale()
}

/// A sample with sigmas between 2 and 4 codes, lower quartiles at about 0.8 of them, and a
/// correlation between 0 and 0.75.
fn hashed_sample(seed: u32, channels: u32) -> TemporalNoiseSample {
    let mut sigma = [0.0f32; NOISE_CHANNELS as usize];
    let mut sigma_low = [0.0f32; NOISE_CHANNELS as usize];
    for channel in 0..channels {
        let unit = hashed_unit(seed, 20 + channel);
        let low_unit = hashed_unit(seed, 30 + channel);
        sigma[channel as usize] = (2.0 + 2.0 * unit) / 255.0;
        sigma_low[channel as usize] = sigma[channel as usize] * (0.7 + 0.2 * low_unit);
    }

    let rho_unit = hashed_unit(seed, 40);

    TemporalNoiseSample {
        sigma,
        sigma_low,
        rho: 0.75 * rho_unit,
        static_fraction: 0.5,
    }
}

fn hashed_curve(seed: u32) -> NoiseCurve {
    let mut ratios = [0.0f32; NOISE_CURVE_BINS];
    for (bin, ratio) in ratios.iter_mut().enumerate() {
        let unit = hashed_unit(seed, 50 + bin as u32);
        *ratio = 0.8 + 0.4 * unit;
    }

    let median_unit = hashed_unit(seed, 70);

    NoiseCurve {
        ratios,
        sigma_quarter_median: (2.0 + median_unit) / 255.0,
    }
}

/// A record with hashed Immerkær sigmas of between 1 and 5 codes, a hashed quartile and, when
/// asked, a sample and a curve.
fn hashed_record(seed: u32, channels: u32, sampled: bool, curved: bool) -> HostRecord {
    let mut immerkaer = [0.0f32; NOISE_CHANNELS as usize];
    let mut immerkaer_low = [0.0f32; NOISE_CHANNELS as usize];
    for channel in 0..channels {
        let sigma_unit = hashed_unit(seed, 10 + channel);
        let low_unit = hashed_unit(seed, channel);
        immerkaer[channel as usize] = (1.0 + 4.0 * sigma_unit) / 255.0;
        immerkaer_low[channel as usize] = (1.0 + 2.0 * low_unit) / 255.0;
    }

    let sample = sampled.then(|| hashed_sample(seed, channels));
    let curve = (sampled && curved).then(|| hashed_curve(seed));
    let reading = ReadingRecord {
        immerkaer_low,
        sample,
        accepted: 40,
        curve,
        populated_bins: if curve.is_some() { 6 } else { 0 },
    };

    HostRecord { reading, immerkaer }
}

/// A no-sample record with no Immerkær response.
fn empty_record() -> HostRecord {
    let reading = ReadingRecord {
        immerkaer_low: [0.0; NOISE_CHANNELS as usize],
        sample: None,
        accepted: 0,
        curve: None,
        populated_bins: 0,
    };

    HostRecord {
        reading,
        immerkaer: [0.0; NOISE_CHANNELS as usize],
    }
}

/// The Immerkær abs sums of `record`.
fn record_sums(record: &HostRecord) -> [f32; RESULTS_PER_SLOT] {
    let mut sums = [0.0f32; RESULTS_PER_SLOT];
    for (sum, &sigma) in sums.iter_mut().zip(&record.immerkaer) {
        *sum = abs_sum_for(sigma);
    }

    sums
}

/// The stream record of `frame`, every third without a sample and every fourth sampled one
/// without a curve.
fn stream_record(frame: u32, channels: u32, seed: u32) -> HostRecord {
    let sampled = frame % 3 != 1;
    let curved = frame % 4 != 2;

    hashed_record(seed + frame, channels, sampled, curved)
}

/// Everything one window fold writes.
#[derive(Debug, Clone, PartialEq)]
struct FoldRun {
    snapshot: NoiseSnapshot,
    curve: Vec<f32>,
}

impl FoldRun {
    fn bits(&self) -> Vec<u32> {
        let derived = self.snapshot.derived.to_words();
        let words = derived.iter().chain(&self.curve);

        words.map(|word| word.to_bits()).collect()
    }
}

/// A GPU noise state fed one frame per push through a ring of `FRAMES` slots, with the host
/// records of every frame since the last reset beside it.
struct Stream {
    client: Client,
    channels: u32,
    settings: NoiseSettings,
    state: NoiseState,
    readings: Vec<ReadingRecord>,
    sums: Vec<[f32; RESULTS_PER_SLOT]>,
    records: Vec<HostRecord>,
}

impl Stream {
    fn new(channels: u32, settings: NoiseSettings) -> Self {
        let client = test_client();
        let shape = shape(&client, channels);
        let frames = FRAMES as usize;
        let empty = empty_record();
        let readings = vec![empty.reading.clone(); frames];
        let sums = vec![[0.0f32; RESULTS_PER_SLOT]; frames];
        let state = NoiseState::with_readings_for_test(&client, shape, settings, &readings, &sums);

        Self {
            client,
            channels,
            settings,
            state,
            readings,
            sums,
            records: Vec::new(),
        }
    }

    /// Writes `record` into the next stream frame's ring slot and records it.
    fn push(&mut self, record: HostRecord) {
        let frame = self.records.len() as u32;
        let slot = frame % FRAMES;
        let index = slot as usize;
        self.readings[index] = record.reading.clone();
        self.sums[index] = record_sums(&record);

        self.state.set_readings_for_test(&self.readings, &self.sums);
        self.state.record_reading(slot, frame);
        self.records.push(record);
    }

    /// Folds the window centred on `centre` over every recorded frame and checks it against the
    /// host model.
    fn fold(&mut self, centre: u32) -> FoldRun {
        let newest = self.state.stream_frame_count() - 1;
        let bounds = WindowBounds::around(centre, newest);
        self.state
            .queue_window_fold(bounds)
            .expect("the window fold queues");

        let run = FoldRun {
            snapshot: self.state.snapshot().expect("the snapshot reads back"),
            curve: self.state.read_curve(),
        };
        self.check(&run, bounds);

        run
    }

    fn check(&self, run: &FoldRun, bounds: WindowBounds) {
        let channels = self.channels;
        let settings = self.settings;
        let host = fold_window(&self.records, bounds, settings.fold.sigma_scale);
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
    }

    fn reset(&mut self) {
        self.state.reset_stream();
        self.records.clear();
    }

    fn finish(&self) {
        sync(&self.client).expect("the device syncs");
    }
}

/// Streams `STREAM_FRAMES` frames, folding the first frame alone and then the ring centre after
/// every push from a full ring.
fn run_stream(channels: u32, settings: NoiseSettings, seed: u32) -> Vec<FoldRun> {
    let mut stream = Stream::new(channels, settings);
    let mut runs = Vec::new();

    for frame in 0..STREAM_FRAMES {
        let record = stream_record(frame, channels, seed);
        stream.push(record);

        let recorded = stream.state.stream_frame_count();
        assert_eq!(recorded, frame + 1, "one recorded frame per push");

        if frame == 0 {
            let seeded = stream.fold(0);
            runs.push(seeded);
        }

        if frame >= 2 * RADIUS {
            let centred = stream.fold(frame - RADIUS);
            runs.push(centred);
        }
    }

    stream.finish();
    runs
}

#[test]
fn the_fold_matches_the_host_window_over_a_stream() {
    let cases = [(1, "luma"), (2, "chroma")];

    for (channels, what) in cases {
        let runs = run_stream(channels, settings(), 17);
        let curved = runs.iter().filter(|run| run.snapshot.curve.is_some()).count();
        let sampled = runs
            .iter()
            .filter(|run| run.snapshot.temporal_only.is_some())
            .count();

        assert!(curved > 0, "{what}: no fold holds a curve");
        assert!(sampled > 0, "{what}: no fold holds a sample");
    }
}

#[test]
fn a_reset_clears_the_history() {
    const CHANNELS: u32 = 2;
    let stream_settings = settings();
    let mut used = Stream::new(CHANNELS, stream_settings);
    let mut fresh = Stream::new(CHANNELS, stream_settings);

    for frame in 0..12 {
        let record = hashed_record(300 + frame, CHANNELS, true, true);
        used.push(record);
    }

    used.fold(8);
    used.reset();

    let recorded = used.state.stream_frame_count();
    let early = WindowBounds::around(0, 0);
    let refused = used.state.queue_window_fold(early);
    assert_eq!(recorded, 0);
    assert!(refused.is_err(), "a fold before any frame is recorded");

    for frame in 0..3 {
        let record = hashed_record(400 + frame, CHANNELS, frame != 1, true);
        used.push(record.clone());
        fresh.push(record);
    }

    let after_reset = used.fold(0);
    let from_fresh = fresh.fold(0);
    used.finish();

    assert_eq!(after_reset.bits(), from_fresh.bits());
    assert_eq!(after_reset.snapshot, from_fresh.snapshot);
}

#[test]
fn a_window_never_reaches_past_the_newest_recorded_frame() {
    let mut stream = Stream::new(1, settings());
    let record = hashed_record(5, 1, true, false);
    stream.push(record);

    let bounds = WindowBounds::around(1, 1);
    let refused = stream.state.queue_window_fold(bounds);
    stream.finish();

    assert!(refused.is_err(), "frame 1 is not recorded yet");
}

#[test]
fn a_perturbed_frame_leaves_the_derived_values_unmoved() {
    const PERTURBED_FRAME: u32 = 10;
    let channels = 1;
    let stream_settings = settings();
    let mut clean = Stream::new(channels, stream_settings);
    let mut perturbed = Stream::new(channels, stream_settings);
    let mut compared = 0;

    for frame in 0..STREAM_FRAMES {
        let record = hashed_record(200, channels, true, true);
        let mut changed = record.clone();
        if frame == PERTURBED_FRAME {
            let sample = changed.reading.sample.as_mut().expect("a sample");
            sample.sigma[0] *= 3.0;
            sample.sigma_low[0] *= 3.0;
            changed.immerkaer[0] *= 3.0;
        }

        clean.push(record);
        perturbed.push(changed);

        if frame < 2 * RADIUS {
            continue;
        }

        let centre = frame - RADIUS;
        let clean_run = clean.fold(centre);
        let perturbed_run = perturbed.fold(centre);

        assert_eq!(perturbed_run.bits(), clean_run.bits(), "fold at {centre}");
        assert_eq!(perturbed_run.snapshot, clean_run.snapshot, "fold at {centre}");
        compared += 1;
    }

    clean.finish();
    assert!(compared > 0);
}

#[test]
fn the_correlation_factor_matches_the_host_table() {
    let (first_rho, _) = CORRELATION_FACTOR_TABLE[0];
    let (last_rho, _) = CORRELATION_FACTOR_TABLE[CORRELATION_FACTOR_TABLE.len() - 1];
    let mut rhos = vec![first_rho - 0.2, last_rho + 0.2, 1.0];
    for segment in CORRELATION_FACTOR_TABLE.windows(2) {
        let (start, _) = segment[0];
        let (end, _) = segment[1];
        for step in 0..=8 {
            let fraction = step as f32 / 8.0;
            rhos.push(start + fraction * (end - start));
        }
    }

    // A unit sigma over no Immerkær response makes the median the factor itself.
    let mut stream_settings = settings();
    stream_settings.fold.coefficients = DerivedCoefficients::default();
    let mut stream = Stream::new(1, stream_settings);
    let mut gaps = Vec::new();

    for rho in rhos {
        let sample = TemporalNoiseSample {
            sigma: [1.0, 0.0],
            sigma_low: [1.0, 0.0],
            rho,
            static_fraction: 0.5,
        };
        let mut record = empty_record();
        record.reading.sample = Some(sample);

        stream.reset();
        stream.push(record);

        let run = stream.fold(0);
        let median = run.snapshot.median.expect("the median chain is written");
        let host = correlation_factor(rho);
        let gap = ulp_gap(median[0], host);
        gaps.push(gap);

        assert!(
            gap <= MAX_ULPS,
            "rho {rho}: factor {} against {host}, {gap} ulp",
            median[0]
        );
    }

    stream.finish();
}

#[test]
fn sigma_scale_scales_each_reading() {
    let mut scaled = settings();
    scaled.fold.sigma_scale = 1.5;
    let mut stream = Stream::new(1, scaled);
    let record = hashed_record(31, 1, true, false);
    let sample = record.reading.sample.expect("the record has a sample");
    stream.push(record);

    let run = stream.fold(0);
    stream.finish();

    let temporal = run
        .snapshot
        .temporal_only
        .expect("the sample feeds the temporal chain");
    let expected = sample.sigma[0] * 1.5;
    assert_eq!(temporal[0], expected);
}

/// The GPU's derived values for a one-frame window of a sampled record under `settings`.
fn derived_for(settings: NoiseSettings, curved: bool) -> (NoiseSnapshot, DerivedValues) {
    let channels = 2;
    let mut stream = Stream::new(channels, settings);
    let record = hashed_record(61, channels, true, curved);
    stream.push(record);

    let run = stream.fold(0);
    stream.finish();

    let snapshot = run.snapshot;
    let fold_settings = settings.fold;
    let expected =
        fold_settings
            .coefficients
            .derive_host(&snapshot, channels as usize, fold_settings.consumer);

    (snapshot, expected)
}

#[test]
fn the_derived_values_follow_the_coefficients() {
    let fixed = DerivedCoefficients {
        h2_per_inverse_variance: None,
        h2_fixed: 0.375,
        ..COEFFICIENTS
    };
    let floor_off = DerivedCoefficients {
        offset_per_variance: 0.0,
        ..COEFFICIENTS
    };
    let pilot = DerivedCoefficients {
        main_offset: false,
        ..COEFFICIENTS
    };
    let no_motion = DerivedCoefficients {
        sad_per_sigma: 0.0,
        confidence_sad_per_sigma: 0.0,
        ..COEFFICIENTS
    };

    for coefficients in [COEFFICIENTS, fixed, floor_off, pilot, no_motion] {
        let mut case = settings();
        case.fold.coefficients = coefficients;
        let (snapshot, expected) = derived_for(case, true);
        let derived = snapshot.derived;
        let median = snapshot.median.expect("the median chain is written");

        let pilot_h2 = derived.h2_inv_norm * coefficients.pilot_h2_factor;
        let main_offset = if coefficients.main_offset {
            derived.input_noise_offset
        } else {
            0.0
        };
        let sad_floor = coefficients.sad_per_sigma * median[0];
        let confidence_floor = coefficients.confidence_sad_per_sigma * median[0];

        assert!(derived.h2_inv_norm > 0.0);
        assert_eq!(derived.pilot_h2_inv_norm, pilot_h2);
        assert_eq!(derived.noise_offset, main_offset);
        assert_eq!(derived.sad_noise_floor, sad_floor);
        assert_eq!(derived.confidence_sad_noise_floor, confidence_floor);
        assert_eq!(expected.curve_valid, derived.curve_valid);

        if coefficients.h2_per_inverse_variance.is_none() {
            assert_eq!(derived.h2_inv_norm, coefficients.h2_fixed);
        }
    }

    // No profile gain, and a gain small enough to put the norm under the floor, both clamp.
    let unread = NoiseConsumer {
        weight_g_max: 0.0,
        ..CONSUMER
    };
    let tiny_gain = NoiseConsumer {
        weight_g_max: 1.0e-5,
        ..CONSUMER
    };
    // The default consumer clamps at the same floor the host's weight scale does.
    let default_consumer = NoiseConsumer::default();
    for consumer in [unread, tiny_gain, default_consumer] {
        let mut case = settings();
        case.fold.consumer = consumer;
        let (snapshot, _) = derived_for(case, true);

        assert_eq!(snapshot.derived.weight_scale, consumer.weight_floor);
    }

    let unapplied = NoiseConsumer {
        curve_applies: false,
        ..CONSUMER
    };
    let curve_cases = [(CONSUMER, true), (CONSUMER, false), (unapplied, true)];
    for (consumer, curved) in curve_cases {
        let mut case = settings();
        case.fold.consumer = consumer;
        let (snapshot, _) = derived_for(case, curved);
        let curve_valid = curved && consumer.curve_applies;
        let applies = consumer.curve_applies;

        assert_eq!(
            snapshot.derived.curve_valid, curve_valid,
            "applies {applies}, curved {curved}"
        );
    }
}

#[test]
fn a_fold_on_a_letterboxed_stream_keeps_every_derived_value_finite() {
    let mut stream = Stream::new(2, settings());
    let mut runs = Vec::new();

    for frame in 0..FRAMES {
        stream.push(empty_record());

        let centre = frame.saturating_sub(RADIUS);
        let run = stream.fold(centre);
        runs.push(run);
    }

    stream.finish();

    for run in runs {
        let words = run.snapshot.derived.to_words();
        let finite = words.iter().all(|word| word.is_finite());
        let median = run.snapshot.median.expect("the median chain is written");

        assert!(finite, "derived words {words:?}");
        assert_eq!(median, [SIGMA_FLOOR, SIGMA_FLOOR]);
        assert!(run.snapshot.derived.weight_scale > 0.0);
    }
}

#[test]
fn the_fold_is_byte_identical_across_runs() {
    let stream_settings = settings();
    let first = run_stream(1, stream_settings, 5);
    let second = run_stream(1, stream_settings, 5);

    for (index, (first_run, second_run)) in first.iter().zip(&second).enumerate() {
        assert_eq!(first_run.bits(), second_run.bits(), "fold {index}");
    }
}

#[test]
fn set_settings_rebuilds_the_fold_settings() {
    let mut stream = Stream::new(1, settings());
    let record = hashed_record(81, 1, true, true);
    let sample = record.reading.sample.expect("the record has a sample");
    stream.push(record.clone());
    let before = stream.fold(0);

    let mut after = settings();
    after.fold.sigma_scale = 2.0;
    stream.state.set_settings(after);
    stream.settings = after;
    stream.reset();
    stream.push(record);
    let scaled = stream.fold(0);
    stream.finish();

    let unscaled_chain = before
        .snapshot
        .temporal_only
        .expect("the sample feeds the temporal chain");
    let scaled_chain = scaled
        .snapshot
        .temporal_only
        .expect("the sample feeds the temporal chain");
    let doubled = sample.sigma[0] * 2.0;
    assert_eq!(unscaled_chain[0], sample.sigma[0]);
    assert_eq!(scaled_chain[0], doubled);
}

#[test]
fn recording_copies_a_slot_into_its_frame_slot() {
    let mut stream = Stream::new(2, settings());
    let record = hashed_record(77, 2, true, true);
    stream.push(record.clone());
    stream.state.record_reading(0, 21);

    let (reading, results) = stream.state.read_history_frame(21);
    let recorded = stream.state.stream_frame_count();
    stream.finish();

    let expected_sums = record_sums(&record);
    assert_eq!(reading, record.reading);
    assert_eq!(results, expected_sums);
    assert_eq!(recorded, 22);

    let immerkaer = sigma_from_abs_sum(results[0], WIDTH, HEIGHT);
    assert!(immerkaer > 0.0);
}

use cubecl::prelude::*;

use super::fold_reference::{
    COEFFICIENTS,
    CONSUMER,
    MAX_ULPS,
    assert_ulps,
    derived_pairs,
    host_from_snapshot,
    host_reading,
    host_snapshot,
    largest_relative,
    largest_ulps,
    relative_gap,
    state_pairs,
};
use super::records::{hashed_unit, ulp_gap};
use crate::collab::geometry::strength_map_dims;
use crate::engine::align::StorageAlign;
use crate::noise::correlation::{CORRELATION_FACTOR_TABLE, correlation_factor};
use crate::noise::curve::{NOISE_CURVE_BINS, NoiseCurve};
use crate::noise::derived::{DerivedCoefficients, DerivedValues, NoiseConsumer};
use crate::noise::estimator::SIGMA_FLOOR;
use crate::noise::fold::{Fold, FoldSettings, NoiseSettings};
use crate::noise::host::HostNoise;
use crate::noise::reading::ReadingRecord;
use crate::noise::snapshot::NoiseSnapshot;
use crate::noise::spatial::{immerkaer_sigma_scale, sigma_from_abs_sum};
use crate::noise::state::{NoiseShape, NoiseState, ReadingSettings, StrengthMapKind};
use crate::noise::strength_map::StrengthMapParams;
use crate::noise::temporal::TemporalNoiseSample;
use crate::noise::{NOISE_BLOCK_1D, STRENGTH_MAP_ALL, STRENGTH_MAP_LUMA, STRENGTH_MAP_OFF};
use crate::tune::gpu::sync;
use crate::tune::tests::runtime::{TestRuntime, test_client};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const FRAMES: u32 = 5;
const CENTRE: u32 = 2;
const MAX_RELATIVE: f32 = 1.0e-5;
const STREAM_FOLDS: u32 = 30;
const LUMA_PARAMS: StrengthMapParams = StrengthMapParams {
    flat_boost: 1.5,
    shadow_soften: 0.65,
};

fn shape(client: &ComputeClient<TestRuntime>, channels: u32) -> NoiseShape {
    NoiseShape {
        width: WIDTH,
        height: HEIGHT,
        channels,
        stored_ch: channels,
        frames: FRAMES,
        temporal: true,
        align: StorageAlign::from_client(client),
    }
}

fn quarters() -> usize {
    let (cols, rows) = strength_map_dims(WIDTH, HEIGHT);
    (cols * rows) as usize
}

fn settings(map: StrengthMapKind, borrow_ahead: bool) -> NoiseSettings {
    let reading = ReadingSettings {
        luma_fields: true,
        map,
        ..ReadingSettings::default()
    };
    let fold = FoldSettings {
        sigma_scale: 1.0,
        borrow_ahead,
        consumer: CONSUMER,
        coefficients: COEFFICIENTS,
        offset_tables: None,
    };

    NoiseSettings { reading, fold }
}

fn luma_map() -> StrengthMapKind {
    StrengthMapKind::Luma(LUMA_PARAMS)
}

/// The abs sum whose Immerkær sigma is about `sigma`.
fn abs_sum_for(sigma: f32) -> f32 {
    let interior = ((WIDTH - 2) * (HEIGHT - 2)) as f32;
    sigma * 6.0 * interior / immerkaer_sigma_scale()
}

/// Immerkær sums of between 1 and 5 codes on each active channel.
fn hashed_sums(seed: u32, channels: u32) -> [f32; 4] {
    let mut sums = [0.0f32; 4];
    for channel in 0..channels {
        let unit = hashed_unit(seed, 10 + channel);
        let sigma = (1.0 + 4.0 * unit) / 255.0;
        sums[channel as usize] = abs_sum_for(sigma);
    }

    sums
}

/// A sample with sigmas between 2 and 4 codes, lower quartiles at about 0.8 of them, and a
/// correlation between 0 and 0.75.
fn hashed_sample(seed: u32, channels: u32) -> TemporalNoiseSample {
    let mut sigma = [0.0f32; 3];
    let mut sigma_low = [0.0f32; 3];
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

/// A reading with a hashed Immerkær quartile and, when asked, a sample and a curve.
fn hashed_reading(seed: u32, channels: u32, sampled: bool, curved: bool) -> ReadingRecord {
    let mut immerkaer_low = [0.0f32; 3];
    for channel in 0..channels {
        let unit = hashed_unit(seed, channel);
        immerkaer_low[channel as usize] = (1.0 + 2.0 * unit) / 255.0;
    }

    let sample = sampled.then(|| hashed_sample(seed, channels));
    let curve = (sampled && curved).then(|| hashed_curve(seed));

    ReadingRecord {
        immerkaer_low,
        sample,
        accepted: 40,
        curve,
        populated_bins: if curve.is_some() { 6 } else { 0 },
    }
}

/// A slot map whose every multiplier names the seed and the quarter.
fn hashed_map(seed: u32) -> Vec<f32> {
    hashed_map_of(seed, quarters())
}

/// A slot map of `quarters` multipliers that name the seed and the quarter.
fn hashed_map_of(seed: u32, quarters: usize) -> Vec<f32> {
    (0..quarters)
        .map(|quarter| 1.0 + 0.01 * seed as f32 + 1.0e-4 * quarter as f32)
        .collect()
}

/// A no-sample reading with no Immerkær response.
fn empty_reading() -> ReadingRecord {
    ReadingRecord {
        immerkaer_low: [0.0; 3],
        sample: None,
        accepted: 0,
        curve: None,
        populated_bins: 0,
    }
}

/// What every slot of a ring holds.
#[derive(Clone)]
struct Slots {
    readings: Vec<ReadingRecord>,
    sums: Vec<[f32; 4]>,
    maps: Vec<Vec<f32>>,
}

impl Slots {
    fn empty() -> Self {
        let frames = FRAMES as usize;

        Self {
            readings: vec![empty_reading(); frames],
            sums: vec![[0.0; 4]; frames],
            maps: vec![vec![1.0; quarters()]; frames],
        }
    }

    /// Slot `slot` with a hashed reading, sums and map from `seed`.
    fn set_hashed(&mut self, slot: u32, seed: u32, channels: u32, sampled: bool, curved: bool) {
        let index = slot as usize;
        self.readings[index] = hashed_reading(seed, channels, sampled, curved);
        self.sums[index] = hashed_sums(seed, channels);
        self.maps[index] = hashed_map(seed);
    }

    /// The maps a state with `settings` takes, none without a strength map.
    fn maps_for(&self, settings: NoiseSettings) -> &[Vec<f32>] {
        if settings.reading.map == StrengthMapKind::Off {
            return &[];
        }

        &self.maps
    }
}

/// The logical position whose sample a fold of `centre` takes, by the host's borrow rule.
fn host_lender(readings: &[ReadingRecord], ring_start: u32, centre: u32, borrow: bool) -> u32 {
    let centre_slot = ((ring_start + centre) % FRAMES) as usize;
    let has_sample = readings[centre_slot].sample.is_some();
    if has_sample || !borrow {
        return centre;
    }

    for logical in centre + 1..FRAMES {
        let slot = ((ring_start + logical) % FRAMES) as usize;
        if readings[slot].sample.is_some() {
            return logical;
        }
    }

    centre
}

/// Folds `fold` of `slots` into `host` as the front does, returning the slot whose map the fold
/// copies.
fn host_fold(
    host: &mut HostNoise,
    slots: &Slots,
    fold: Fold,
    settings: NoiseSettings,
    channels: u32,
) -> Option<u32> {
    let (ring_start, centre, seed) = match fold {
        Fold::Seed { slot } => (slot, 0, true),
        Fold::Centre { ring_start, centre } => (ring_start, centre, false),
    };

    let centre_slot = (ring_start + centre) % FRAMES;
    let centre_reading = &slots.readings[centre_slot as usize];
    let sums = slots.sums[centre_slot as usize];
    let mut immerkaer = [0.0f32; 3];
    for channel in 0..channels as usize {
        immerkaer[channel] = sigma_from_abs_sum(sums[channel], WIDTH, HEIGHT);
    }

    let sigma_scale = settings.fold.sigma_scale;
    let active = channels as usize;
    let immerkaer_low = centre_reading.immerkaer_low;

    if seed {
        host.fold(immerkaer, None, immerkaer_low, active, sigma_scale);
        return None;
    }

    let borrow = settings.fold.borrow_ahead;
    let lender = host_lender(&slots.readings, ring_start, centre, borrow);
    let lender_slot = (ring_start + lender) % FRAMES;
    let lender_reading = host_reading(&slots.readings[lender_slot as usize]);

    host.adopt(&lender_reading);
    host.fold(
        immerkaer,
        lender_reading.sample,
        immerkaer_low,
        active,
        sigma_scale,
    );

    let copies = lender_reading.sample.is_some() && lender_reading.classes.is_some();
    copies.then_some(lender_slot)
}

/// The largest gaps a stream showed.
#[derive(Debug, Default, Clone, Copy)]
struct StreamGaps {
    /// Each fold against the host fold of the GPU's previous state.
    step_ulps: u32,
    /// The derived values against `derive_host` of the GPU's own state.
    derived_ulps: u32,
    /// The state and derived values against a host stream run alongside.
    stream_ulps: u32,
    stream_relative: f32,
}

/// A GPU noise state and the host reference beside it, over one ring.
struct Stream {
    client: ComputeClient<TestRuntime>,
    channels: u32,
    settings: NoiseSettings,
    state: NoiseState<TestRuntime>,
    slots: Slots,
    host: HostNoise,
    /// The map the host expects the GPU to hold, ones until a lender with classes.
    host_map: Vec<f32>,
    previous: NoiseSnapshot,
    gaps: StreamGaps,
}

impl Stream {
    fn new(channels: u32, settings: NoiseSettings, slots: Slots) -> Self {
        let client = test_client();
        let shape = shape(&client, channels);
        let maps = slots.maps_for(settings);
        let state =
            NoiseState::with_readings_for_test(&client, shape, settings, &slots.readings, &slots.sums, maps);
        let previous = state.snapshot().expect("the snapshot reads back");

        Self {
            client,
            channels,
            settings,
            state,
            slots,
            host: HostNoise::default(),
            host_map: vec![1.0; quarters()],
            previous,
            gaps: StreamGaps::default(),
        }
    }

    /// Replaces the ring's slots on both sides.
    fn write(&mut self, slots: Slots) {
        let maps = slots.maps_for(self.settings);
        self.state
            .set_readings_for_test(&slots.readings, &slots.sums, maps);
        self.slots = slots;
    }

    /// Folds on both sides and checks the GPU against the host, returning the GPU snapshot.
    ///
    /// Each fold is within 2 ulp of the host fold of the GPU's previous state, and the derived
    /// values within 2 ulp of `derive_host` of the GPU's state. The stream is within 1e-5 relative
    /// of the host stream. The current map and the curve buffer are checked bit for bit.
    fn fold(&mut self, fold: Fold) -> NoiseSnapshot {
        let channels = self.channels;
        let settings = self.settings;
        self.state.queue_fold(fold).expect("the fold queues");
        let gpu = self.state.snapshot().expect("the snapshot reads back");

        let mut stepped = host_from_snapshot(&self.previous, channels);
        host_fold(&mut stepped, &self.slots, fold, settings, channels);
        let step = host_snapshot(&stepped, settings, channels);
        let step_pairs = state_pairs(&gpu, &step, channels);
        assert_ulps(&step_pairs, "one fold");

        let fold_settings = settings.fold;
        let own_derived = fold_settings.coefficients.derive_host(
            &gpu,
            channels as usize,
            fold_settings.consumer,
            settings.reading.map,
        );
        let own_pairs = derived_pairs(&gpu.derived, &own_derived, channels);
        assert_ulps(&own_pairs, "the derived values");

        let copied = host_fold(&mut self.host, &self.slots, fold, settings, channels);
        if let (Some(slot), true) = (copied, settings.reading.map != StrengthMapKind::Off) {
            self.host_map = self.slots.maps[slot as usize].clone();
        }

        let expected = host_snapshot(&self.host, settings, channels);
        let mut stream_pairs = state_pairs(&gpu, &expected, channels);
        let stream_derived = derived_pairs(&gpu.derived, &expected.derived, channels);
        stream_pairs.extend(stream_derived);
        let stream_relative = largest_relative(&stream_pairs);
        assert!(
            stream_relative <= MAX_RELATIVE,
            "the stream is {stream_relative:e} apart"
        );

        self.check_buffers(&gpu);

        let step_ulps = largest_ulps(&step_pairs);
        let derived_ulps = largest_ulps(&own_pairs);
        let stream_ulps = largest_ulps(&stream_pairs);
        self.gaps.step_ulps = self.gaps.step_ulps.max(step_ulps);
        self.gaps.derived_ulps = self.gaps.derived_ulps.max(derived_ulps);
        self.gaps.stream_ulps = self.gaps.stream_ulps.max(stream_ulps);
        self.gaps.stream_relative = self.gaps.stream_relative.max(stream_relative);
        self.previous = gpu.clone();

        gpu
    }

    /// Checks the curve buffer against the snapshot and, with a map, the current map against the
    /// host's.
    fn check_buffers(&self, gpu: &NoiseSnapshot) {
        let curve = self.state.read_curve();
        let expected_curve = match gpu.curve {
            Some((ratios, _)) if gpu.derived.curve_valid => ratios.to_vec(),
            _ => vec![0.0; NOISE_CURVE_BINS],
        };
        assert_eq!(curve, expected_curve, "the curve buffer");

        if self.settings.reading.map == StrengthMapKind::Off {
            return;
        }

        let map = self.state.read_current_map();
        assert_eq!(map, self.host_map, "the current map");
    }

    fn finish(&self) -> StreamGaps {
        sync(&self.client).expect("the device syncs");
        self.gaps
    }
}

/// The slots after `fold` writes its new reading, every third without a sample and every fourth
/// sampled one without a curve.
fn stream_slots(slots: &mut Slots, fold: u32, channels: u32, seed: u32) {
    let slot = fold % FRAMES;
    let sampled = fold % 3 != 1;
    let curved = fold % 4 != 2;
    slots.set_hashed(slot, seed + fold, channels, sampled, curved);
}

/// Runs a seed then a fold per new slot, writing one slot ahead of each fold as a stream does.
fn run_stream(channels: u32, settings: NoiseSettings, seed: u32) -> (StreamGaps, Vec<NoiseSnapshot>) {
    let mut slots = Slots::empty();
    for fold in 0..FRAMES {
        stream_slots(&mut slots, fold, channels, seed);
    }

    let mut stream = Stream::new(channels, settings, slots.clone());
    let mut snapshots = Vec::new();
    let seeded = stream.fold(Fold::Seed { slot: 0 });
    snapshots.push(seeded);

    for fold in 0..STREAM_FOLDS {
        let ring_start = fold % FRAMES;
        let centre = Fold::Centre {
            ring_start,
            centre: CENTRE,
        };
        let snapshot = stream.fold(centre);
        snapshots.push(snapshot);

        // The oldest slot takes the next frame.
        stream_slots(&mut slots, fold + FRAMES, channels, seed);
        stream.write(slots.clone());
    }

    let gaps = stream.finish();
    (gaps, snapshots)
}

#[test]
fn the_fold_matches_the_host_fold_over_a_stream() {
    let cases = [
        (1, settings(luma_map(), true), "luma, luma map, borrowing"),
        (
            2,
            settings(StrengthMapKind::Chroma { flat_boost: 1.5 }, true),
            "chroma, chroma map",
        ),
        (3, settings(StrengthMapKind::Off, false), "three channels, no map"),
    ];

    for (channels, case_settings, what) in cases {
        let (gaps, snapshots) = run_stream(channels, case_settings, 17);
        let curved = snapshots
            .iter()
            .filter(|snapshot| snapshot.curve.is_some())
            .count();
        let missing = snapshots
            .iter()
            .filter(|snapshot| snapshot.curve.is_none())
            .count();
        println!("{what}: {gaps:?}");

        assert!(curved > 0, "{what}: no fold holds a curve");
        assert!(missing > 0, "{what}: every fold holds a curve");
    }
}

/// A one-slot ring at `slot` of `slots`, with a hashed sampled and curved reading there.
fn sampled_slots(slot: u32, channels: u32, seed: u32) -> Slots {
    let mut slots = Slots::empty();
    slots.set_hashed(slot, seed, channels, true, true);
    slots
}

#[test]
fn the_seed_folds_immerkaer_alone_and_keeps_the_curve() {
    let slots = sampled_slots(1, 1, 3);
    let mut stream = Stream::new(1, settings(luma_map(), false), slots);
    let snapshot = stream.fold(Fold::Seed { slot: 1 });
    let map = stream.state.read_current_map();
    stream.finish();

    let ones = vec![1.0f32; quarters()];
    assert!(snapshot.median.is_some());
    assert_eq!(snapshot.temporal_only, None);
    assert_eq!(snapshot.rho, None);
    assert_eq!(snapshot.curve, None);
    assert!(!snapshot.classes_present);
    assert_eq!(map, ones);
}

/// Logical 0 and 1 have no sample, 2 and 3 do, from a ring that starts in slot 3.
fn borrowing_slots(channels: u32) -> Slots {
    let ring_start = 3;
    let mut slots = Slots::empty();
    for logical in 0..FRAMES {
        let slot = (ring_start + logical) % FRAMES;
        let sampled = logical == 2 || logical == 3;
        slots.set_hashed(slot, 100 + logical, channels, sampled, true);
    }

    slots
}

#[test]
fn the_fold_borrows_the_first_sample_ahead() {
    let slots = borrowing_slots(1);
    let lender_slot = (3 + 2) % FRAMES;
    let lender = slots.readings[lender_slot as usize].clone();
    let lender_map = slots.maps[lender_slot as usize].clone();
    let mut stream = Stream::new(1, settings(luma_map(), true), slots);

    let first = Fold::Centre {
        ring_start: 3,
        centre: 0,
    };
    let snapshot = stream.fold(first);
    let map = stream.state.read_current_map();
    stream.finish();

    let sample = lender.sample.expect("logical 2 has a sample");
    let curve = lender.curve.expect("logical 2 has a curve");
    let temporal = snapshot
        .temporal_only
        .expect("the borrowed sample seeds the temporal chain");
    let expected_temporal = sample.sigma[0].max(SIGMA_FLOOR);
    let held = snapshot.curve.expect("the borrowed curve is held");

    assert_eq!(temporal[0], expected_temporal);
    assert_eq!(held, (curve.ratios, curve.sigma_quarter_median));
    assert_eq!(snapshot.rho, Some(sample.rho));
    assert_eq!(map, lender_map);
}

#[test]
fn the_fold_falls_back_when_nothing_ahead_has_a_sample() {
    // Logical 1 has no sample and only logical 0, behind it, has one.
    let mut slots = Slots::empty();
    slots.set_hashed(0, 7, 2, true, true);
    slots.set_hashed(1, 8, 2, false, false);
    let mut stream = Stream::new(2, settings(luma_map(), true), slots);

    let centre = Fold::Centre {
        ring_start: 0,
        centre: 1,
    };
    let snapshot = stream.fold(centre);
    stream.finish();

    let unboosted = snapshot.low_unboosted.expect("the low chain updates");
    assert_eq!(snapshot.temporal_only, None);
    assert_eq!(snapshot.curve, None);
    assert_eq!(snapshot.derived.sigma, unboosted);
}

#[test]
fn the_fold_never_borrows_without_shifted_edges() {
    let slots = borrowing_slots(1);
    let mut stream = Stream::new(1, settings(luma_map(), false), slots);

    let first = Fold::Centre {
        ring_start: 3,
        centre: 0,
    };
    let snapshot = stream.fold(first);
    let map = stream.state.read_current_map();
    stream.finish();

    let ones = vec![1.0f32; quarters()];
    assert_eq!(snapshot.temporal_only, None);
    assert_eq!(snapshot.rho, None);
    assert_eq!(snapshot.curve, None);
    assert_eq!(map, ones);
}

/// A stream that adopts slot 0's curve and map, then folds slot 1.
fn adopt_then_fold(second_sampled: bool, second_curved: bool) -> (NoiseSnapshot, Vec<f32>, Vec<f32>) {
    let mut slots = Slots::empty();
    slots.set_hashed(0, 11, 1, true, true);
    slots.set_hashed(1, 12, 1, second_sampled, second_curved);
    let first_map = slots.maps[0].clone();
    let mut stream = Stream::new(1, settings(luma_map(), false), slots);

    let first = Fold::Centre {
        ring_start: 0,
        centre: 0,
    };
    stream.fold(first);

    let second = Fold::Centre {
        ring_start: 1,
        centre: 0,
    };
    let snapshot = stream.fold(second);
    let map = stream.state.read_current_map();
    stream.finish();

    (snapshot, map, first_map)
}

#[test]
fn a_lender_without_a_curve_clears_the_curve_and_leaves_the_map() {
    let (snapshot, map, first_map) = adopt_then_fold(true, false);

    assert_eq!(snapshot.curve, None);
    assert!(!snapshot.classes_present);
    assert_eq!(snapshot.derived.map_mode, STRENGTH_MAP_OFF);
    assert_eq!(map, first_map);
}

#[test]
fn a_fold_without_a_sample_keeps_the_previous_curve_and_map() {
    let (snapshot, map, first_map) = adopt_then_fold(false, false);

    assert!(snapshot.curve.is_some());
    assert!(snapshot.classes_present);
    assert_eq!(snapshot.derived.map_mode, STRENGTH_MAP_LUMA);
    assert_eq!(map, first_map);
}

#[test]
fn the_first_temporal_sample_seeds_rho_directly() {
    let mut slots = Slots::empty();
    slots.set_hashed(0, 21, 1, true, false);
    slots.set_hashed(1, 22, 1, true, false);
    let first_rho = slots.readings[0].sample.map(|sample| sample.rho);
    let mut stream = Stream::new(1, settings(StrengthMapKind::Off, false), slots);

    let seeded = stream.fold(Fold::Seed { slot: 0 });
    let first = Fold::Centre {
        ring_start: 0,
        centre: 0,
    };
    let after_first = stream.fold(first);
    let second = Fold::Centre {
        ring_start: 1,
        centre: 0,
    };
    let after_second = stream.fold(second);
    stream.finish();

    assert_eq!(seeded.rho, None);
    assert_eq!(after_first.rho, first_rho);
    assert_ne!(after_second.rho, first_rho);
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

    // A unit sigma over no Immerkær response makes the first median the factor itself.
    let mut slots = Slots::empty();
    let mut stream_settings = settings(StrengthMapKind::Off, false);
    stream_settings.fold.coefficients = DerivedCoefficients::default();
    let mut gaps = Vec::new();

    let client = test_client();
    let shape = shape(&client, 1);
    let mut state = NoiseState::with_readings_for_test(
        &client,
        shape,
        stream_settings,
        &slots.readings,
        &slots.sums,
        &[],
    );

    for rho in rhos {
        let sample = TemporalNoiseSample {
            sigma: [1.0, 0.0, 0.0],
            sigma_low: [1.0, 0.0, 0.0],
            rho,
            static_fraction: 0.5,
        };
        slots.readings[0].sample = Some(sample);
        state.set_readings_for_test(&slots.readings, &slots.sums, &[]);
        state.queue_reset();

        let fold = Fold::Centre {
            ring_start: 0,
            centre: 0,
        };
        state.queue_fold(fold).expect("the fold queues");
        let snapshot = state.snapshot().expect("the snapshot reads back");
        let median = snapshot.median.expect("the median chain updates");
        let host = correlation_factor(rho);
        let gap = ulp_gap(median[0], host);
        gaps.push(gap);

        assert!(
            gap <= MAX_ULPS,
            "rho {rho}: factor {} against {host}, {gap} ulp",
            median[0]
        );
    }

    sync(&client).expect("the device syncs");
    let largest = gaps.iter().copied().max().unwrap_or(0);
    println!("correlation factor largest gap {largest} ulp");
}

#[test]
fn sigma_scale_scales_before_smoothing() {
    let mut slots = Slots::empty();
    slots.set_hashed(0, 31, 1, true, false);
    slots.set_hashed(1, 32, 1, true, false);
    let first_reading = slots.readings[0].clone();
    let mut scaled = settings(StrengthMapKind::Off, false);
    scaled.fold.sigma_scale = 1.5;
    let mut stream = Stream::new(1, scaled, slots);

    let first = Fold::Centre {
        ring_start: 0,
        centre: 0,
    };
    let after_first = stream.fold(first);
    let second = Fold::Centre {
        ring_start: 1,
        centre: 0,
    };
    stream.fold(second);
    stream.finish();

    let sample = first_reading.sample.expect("slot 0 has a sample");
    let temporal = after_first
        .temporal_only
        .expect("the sample seeds the temporal chain");
    let expected = sample.sigma[0] * 1.5;
    assert_eq!(temporal[0], expected);
}

#[test]
fn a_reset_starts_the_next_stream_from_the_seed() {
    let mut slots = Slots::empty();
    for slot in 0..FRAMES {
        slots.set_hashed(slot, 40 + slot, 2, true, true);
    }

    let stream_settings = settings(StrengthMapKind::Chroma { flat_boost: 1.5 }, false);
    let client = test_client();
    let shape = shape(&client, 2);
    let maps = slots.maps_for(stream_settings);
    let used = NoiseState::with_readings_for_test(
        &client,
        shape,
        stream_settings,
        &slots.readings,
        &slots.sums,
        maps,
    );
    let fresh = NoiseState::with_readings_for_test(
        &client,
        shape,
        stream_settings,
        &slots.readings,
        &slots.sums,
        maps,
    );

    for ring_start in 0..FRAMES {
        let fold = Fold::Centre {
            ring_start,
            centre: CENTRE,
        };
        used.queue_fold(fold).expect("the fold queues");
    }

    used.queue_reset();
    used.queue_fold(Fold::Seed { slot: 3 }).expect("the seed queues");
    fresh.queue_fold(Fold::Seed { slot: 3 }).expect("the seed queues");

    let after_reset = used.snapshot().expect("the snapshot reads back");
    let after_seed = fresh.snapshot().expect("the snapshot reads back");
    sync(&client).expect("the device syncs");

    assert_eq!(after_reset.rho, None);
    assert_eq!(after_reset.curve, None);
    assert!(!after_reset.classes_present);
    assert_eq!(after_reset, after_seed);
}

/// The chains' values, active channels only, in a fixed order.
fn chain_values(snapshot: &NoiseSnapshot, channels: usize) -> Vec<f32> {
    let chains = [
        snapshot.median,
        snapshot.low,
        snapshot.low_unboosted,
        snapshot.temporal_only,
    ];

    chains
        .iter()
        .flat_map(|chain| chain.unwrap_or_default()[..channels].to_vec())
        .collect()
}

#[test]
fn a_perturbed_frame_recovers_within_15_folds() {
    const PERTURBED_FOLD: u32 = 5;
    const DELTA: f32 = 1.0e-3;
    const RECOVERED_FOLD: u32 = 20;
    let channels = 1;
    let stream_settings = settings(StrengthMapKind::Off, false);
    let client = test_client();
    let shape = shape(&client, channels);

    let mut slots = Slots::empty();
    let mut clean = NoiseState::with_readings_for_test(
        &client,
        shape,
        stream_settings,
        &slots.readings,
        &slots.sums,
        &[],
    );
    let mut perturbed = NoiseState::with_readings_for_test(
        &client,
        shape,
        stream_settings,
        &slots.readings,
        &slots.sums,
        &[],
    );

    for fold in 0..STREAM_FOLDS {
        // Every fold reads the same sample with no Immerkær response, so each chain holds the
        // sample's value and a relative change in it moves the chain by the same relative step.
        let slot = fold % FRAMES;
        slots.set_hashed(slot, 200, channels, true, false);
        slots.sums[slot as usize] = [0.0; 4];
        slots.readings[slot as usize].immerkaer_low = [0.0; 3];
        clean.set_readings_for_test(&slots.readings, &slots.sums, &[]);

        let mut changed = slots.clone();
        if fold == PERTURBED_FOLD {
            let sample = changed.readings[slot as usize].sample.as_mut().expect("a sample");
            sample.sigma[0] *= 1.0 + DELTA;
            sample.sigma_low[0] *= 1.0 + DELTA;
        }
        perturbed.set_readings_for_test(&changed.readings, &changed.sums, &[]);

        let centre = Fold::Centre {
            ring_start: slot,
            centre: 0,
        };
        clean.queue_fold(centre).expect("the fold queues");
        perturbed.queue_fold(centre).expect("the fold queues");

        let clean_snapshot = clean.snapshot().expect("the snapshot reads back");
        let perturbed_snapshot = perturbed.snapshot().expect("the snapshot reads back");
        let clean_values = chain_values(&clean_snapshot, channels as usize);
        let perturbed_values = chain_values(&perturbed_snapshot, channels as usize);
        let gap = clean_values
            .iter()
            .zip(&perturbed_values)
            .map(|(&clean_value, &perturbed_value)| relative_gap(perturbed_value, clean_value))
            .fold(0.0f32, f32::max);

        if fold >= PERTURBED_FOLD {
            let folds_since = (fold - PERTURBED_FOLD) as i32;
            let bound = 0.2 * DELTA * 0.8f32.powi(folds_since) + MAX_RELATIVE;
            assert!(gap <= bound, "fold {fold}: {gap:e} apart, bound {bound:e}");
        }

        if fold == PERTURBED_FOLD || fold == PERTURBED_FOLD + 1 {
            assert!(gap > MAX_RELATIVE, "fold {fold}: the flip shows only {gap:e}");
        }

        let recovering = (PERTURBED_FOLD..RECOVERED_FOLD).contains(&fold);
        if !recovering {
            assert!(gap <= MAX_RELATIVE, "fold {fold}: {gap:e} apart");
        }
    }

    sync(&client).expect("the device syncs");
}

/// The GPU's derived values for one fold of a sampled, curved slot 0 under `settings`.
fn derived_for(settings: NoiseSettings, curved: bool) -> (NoiseSnapshot, DerivedValues) {
    let channels = 2;
    let mut slots = Slots::empty();
    slots.set_hashed(0, 61, channels, true, curved);
    let mut stream = Stream::new(channels, settings, slots);

    let fold = Fold::Centre {
        ring_start: 0,
        centre: 0,
    };
    let snapshot = stream.fold(fold);
    stream.finish();

    let fold_settings = settings.fold;
    let expected = fold_settings.coefficients.derive_host(
        &snapshot,
        channels as usize,
        fold_settings.consumer,
        settings.reading.map,
    );

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
        let mut case = settings(StrengthMapKind::Off, false);
        case.fold.coefficients = coefficients;
        let (snapshot, expected) = derived_for(case, true);
        let derived = snapshot.derived;
        let median = snapshot.median.expect("the median chain updates");

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
        let mut case = settings(StrengthMapKind::Off, false);
        case.fold.consumer = consumer;
        let (snapshot, _) = derived_for(case, true);

        assert_eq!(snapshot.derived.weight_scale, consumer.weight_floor);
    }

    let unapplied = NoiseConsumer {
        curve_applies: false,
        ..CONSUMER
    };
    let chroma = StrengthMapKind::Chroma { flat_boost: 1.5 };
    let modes = [
        (StrengthMapKind::Off, CONSUMER, true, STRENGTH_MAP_OFF),
        (luma_map(), CONSUMER, true, STRENGTH_MAP_LUMA),
        (luma_map(), CONSUMER, false, STRENGTH_MAP_OFF),
        (luma_map(), unapplied, true, STRENGTH_MAP_OFF),
        (chroma, CONSUMER, true, STRENGTH_MAP_ALL),
        (chroma, unapplied, true, STRENGTH_MAP_ALL),
        (chroma, CONSUMER, false, STRENGTH_MAP_OFF),
    ];
    for (map, consumer, curved, mode) in modes {
        let mut case = settings(map, false);
        case.fold.consumer = consumer;
        let (snapshot, _) = derived_for(case, curved);
        let curve_valid = curved && consumer.curve_applies;

        assert_eq!(snapshot.derived.map_mode, mode, "{map:?}, curved {curved}");
        assert_eq!(
            snapshot.derived.curve_valid, curve_valid,
            "{map:?}, curved {curved}"
        );
    }
}

#[test]
fn a_fold_on_a_letterboxed_stream_keeps_every_derived_value_finite() {
    let slots = Slots::empty();
    let mut stream = Stream::new(2, settings(luma_map(), true), slots);
    let seeded = stream.fold(Fold::Seed { slot: 0 });
    let mut snapshots = vec![seeded];

    for ring_start in 0..FRAMES {
        let fold = Fold::Centre {
            ring_start,
            centre: CENTRE,
        };
        let snapshot = stream.fold(fold);
        snapshots.push(snapshot);
    }

    stream.finish();

    for snapshot in snapshots {
        let words = snapshot.derived.to_words();
        let finite = words.iter().all(|word| word.is_finite());
        let median = snapshot.median.expect("the median chain updates");

        assert!(finite, "derived words {words:?}");
        assert_eq!(median, [SIGMA_FLOOR, SIGMA_FLOOR, 0.0]);
        assert!(snapshot.derived.weight_scale > 0.0);
    }
}

#[test]
fn the_fold_is_byte_identical_across_runs() {
    let stream_settings = settings(luma_map(), true);
    let (_, first) = run_stream(1, stream_settings, 5);
    let (_, second) = run_stream(1, stream_settings, 5);

    for (index, (first_snapshot, second_snapshot)) in first.iter().zip(&second).enumerate() {
        let first_bits = snapshot_bits(first_snapshot);
        let second_bits = snapshot_bits(second_snapshot);
        assert_eq!(first_bits, second_bits, "fold {index}");
    }
}

fn snapshot_bits(snapshot: &NoiseSnapshot) -> Vec<u32> {
    let mut values = chain_values(snapshot, 3);
    let rho = snapshot.rho.unwrap_or(-1.0);
    let derived = snapshot.derived.to_words();
    values.push(rho);
    values.extend(derived);

    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn set_settings_rebuilds_the_fold_settings() {
    let client = test_client();
    let shape = shape(&client, 1);
    let mut slots = Slots::empty();
    slots.set_hashed(0, 81, 1, true, true);
    let sample = slots.readings[0].sample.expect("slot 0 has a sample");
    let before = settings(StrengthMapKind::Off, false);
    let mut state =
        NoiseState::with_readings_for_test(&client, shape, before, &slots.readings, &slots.sums, &[]);
    let fold = Fold::Centre {
        ring_start: 0,
        centre: 0,
    };

    state.queue_fold(fold).expect("the fold queues");
    let unmapped = state.snapshot().expect("the snapshot reads back");

    let mut after = settings(luma_map(), false);
    after.fold.sigma_scale = 2.0;
    state.set_settings(after);
    state.queue_reset();
    state.queue_fold(fold).expect("the fold queues");
    let mapped = state.snapshot().expect("the snapshot reads back");
    sync(&client).expect("the device syncs");

    let temporal = mapped.temporal_only.expect("the sample seeds the temporal chain");
    let scaled = sample.sigma[0] * 2.0;
    assert_eq!(unmapped.derived.map_mode, STRENGTH_MAP_OFF);
    assert_eq!(mapped.derived.map_mode, STRENGTH_MAP_LUMA);
    assert_eq!(temporal[0], scaled);
}

#[test]
fn a_4k_luma_map_copies_across_the_grid() {
    const WIDTH_4K: u32 = 3840;
    const HEIGHT_4K: u32 = 2160;
    let client = test_client();
    let shape = NoiseShape {
        width: WIDTH_4K,
        height: HEIGHT_4K,
        channels: 1,
        stored_ch: 1,
        frames: FRAMES,
        temporal: true,
        align: StorageAlign::from_client(&client),
    };
    let (cols, rows) = strength_map_dims(WIDTH_4K, HEIGHT_4K);
    let quarters = (cols * rows) as usize;

    // Slot 0 lends a curve and its map, slot 1 has a sample without a curve, slot 2 no sample.
    let mut readings = vec![empty_reading(); FRAMES as usize];
    readings[0] = hashed_reading(91, 1, true, true);
    readings[1] = hashed_reading(92, 1, true, false);
    readings[2] = hashed_reading(93, 1, false, false);
    let sums = vec![[0.0f32; 4]; FRAMES as usize];
    let maps: Vec<Vec<f32>> = (0..FRAMES).map(|slot| hashed_map_of(slot, quarters)).collect();

    let map_settings = settings(luma_map(), false);
    let state = NoiseState::with_readings_for_test(&client, shape, map_settings, &readings, &sums, &maps);
    let map_cubes = (quarters as u32).div_ceil(NOISE_BLOCK_1D);
    assert!(map_cubes > 1, "the map spans {map_cubes} cubes");

    let mut copied_maps = Vec::new();
    for ring_start in 0..3 {
        let fold = Fold::Centre {
            ring_start,
            centre: 0,
        };
        state.queue_fold(fold).expect("the fold queues");
        let map = state.read_current_map();
        copied_maps.push(map);
    }

    let expected: Vec<u32> = maps[0].iter().map(|value| value.to_bits()).collect();
    for (index, map) in copied_maps.iter().enumerate() {
        let bits: Vec<u32> = map.iter().map(|value| value.to_bits()).collect();
        assert_eq!(bits, expected, "the map after fold {index}");
    }
}

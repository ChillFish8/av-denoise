use cubecl::prelude::*;
use cubecl::server::Handle;

use super::fold_reference::{COEFFICIENTS, CONSUMER, relative_gap};
use crate::collab::geometry::strength_map_dims;
use crate::engine::align::StorageAlign;
use crate::noise::NOISE_BLOCK_1D;
use crate::noise::correlation::{build_spatial_offset_lut, spatial_offset_index};
use crate::noise::curve::{NOISE_CURVE_BINS, NoiseCurve};
use crate::noise::derived::{DerivedCoefficients, DerivedValues};
use crate::noise::estimator::EMA_ALPHA;
use crate::noise::fold::{Fold, FoldSettings, NoiseSettings};
use crate::noise::reading::ReadingRecord;
use crate::noise::state::{NoiseShape, NoiseState, ReadingSettings, StrengthMapKind};
use crate::noise::strength_map::StrengthMapParams;
use crate::noise::temporal::TemporalNoiseSample;
use crate::tune::gpu::sync;
use crate::tune::tests::runtime::test_client;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const FRAMES: u32 = 3;
const MAX_RELATIVE: f32 = 1.0e-5;
const RADII: [u32; 2] = [2, 8];
const RHOS: [f32; 3] = [0.0, 0.3, 0.65];
const LUMA_PARAMS: StrengthMapParams = StrengthMapParams {
    flat_boost: 1.5,
    shadow_soften: 0.65,
};
/// The fold the tests run, which takes slot 0's sample.
const FIRST_FOLD: Fold = Fold::Centre {
    ring_start: 0,
    centre: 0,
};
/// The fold after it, which takes slot 1's sample.
const SECOND_FOLD: Fold = Fold::Centre {
    ring_start: 1,
    centre: 0,
};
/// The derived values a test state starts from, whose host-built tables are nonzero off the
/// centre.
const STARTING_VALUES: DerivedValues = DerivedValues {
    sigma: [0.0; 3],
    weight_scale: 0.0,
    curve_valid: false,
    map_mode: 0,
    sad_noise_floor: 0.0,
    confidence_sad_noise_floor: 0.0,
    h2_inv_norm: 0.0,
    pilot_h2_inv_norm: 0.0,
    noise_offset: 0.5,
    input_noise_offset: 0.75,
    rho: 0.0,
};

fn shape(client: &Client, width: u32, height: u32) -> NoiseShape {
    NoiseShape {
        width,
        height,
        channels: 1,
        stored_ch: 1,
        frames: FRAMES,
        temporal: true,
        align: StorageAlign::from_client(client),
    }
}

fn settings(radius: u32, coefficients: DerivedCoefficients, map: StrengthMapKind) -> NoiseSettings {
    let reading = ReadingSettings {
        luma_fields: true,
        map,
        ..ReadingSettings::default()
    };
    let fold = FoldSettings {
        sigma_scale: 1.0,
        borrow_ahead: false,
        consumer: CONSUMER,
        coefficients,
        offset_tables: Some(radius),
    };

    NoiseSettings { reading, fold }
}

/// A reading with a sample of correlation `rho` and, when asked, a flat curve.
fn sampled_reading(rho: f32, curved: bool) -> ReadingRecord {
    let sample = TemporalNoiseSample {
        sigma: [3.0 / 255.0, 0.0, 0.0],
        sigma_low: [2.5 / 255.0, 0.0, 0.0],
        rho,
        static_fraction: 0.5,
    };
    let curve = curved.then_some(NoiseCurve {
        ratios: [1.0; NOISE_CURVE_BINS],
        sigma_quarter_median: 2.0 / 255.0,
    });

    ReadingRecord {
        immerkaer_low: [1.0 / 255.0, 0.0, 0.0],
        sample: Some(sample),
        accepted: 40,
        curve,
        populated_bins: if curved { 6 } else { 0 },
    }
}

/// A state started from [STARTING_VALUES] whose slots hold samples of the correlations `rhos`,
/// before any fold.
fn ring_state(
    client: &Client,
    shape: NoiseShape,
    settings: NoiseSettings,
    rhos: [f32; FRAMES as usize],
    slot_maps: &[Vec<f32>],
) -> NoiseState {
    let curved = !slot_maps.is_empty();
    let readings: Vec<ReadingRecord> = rhos.iter().map(|&rho| sampled_reading(rho, curved)).collect();
    let sums = vec![[0.0f32; 4]; FRAMES as usize];
    let mut state = NoiseState::new(client, shape, settings, STARTING_VALUES);
    state.set_readings_for_test(&readings, &sums, slot_maps);

    state
}

/// A state whose every slot holds a sample of correlation `rho`, after one fold of slot 0.
fn folded_state(
    client: &Client,
    shape: NoiseShape,
    settings: NoiseSettings,
    rho: f32,
    slot_maps: &[Vec<f32>],
) -> NoiseState {
    let rhos = [rho; FRAMES as usize];
    let state = ring_state(client, shape, settings, rhos, slot_maps);

    state.queue_fold(FIRST_FOLD).expect("the fold queues");

    state
}

/// Whether every entry off the centre of the table is nonzero.
fn nonzero_off_centre(table: &[f32], radius: u32) -> bool {
    let centre = spatial_offset_index(radius, 0, 0) as usize;
    let mut entries = table.iter().enumerate();

    entries.all(|(index, &entry)| index == centre || entry != 0.0)
}

/// Checks that both tables of `state` start nonzero off the centre, so a later zero table can
/// only come from a fold.
fn check_nonzero_start(client: &Client, state: &NoiseState, radius: u32) {
    let spatial = read_table(client, state.spatial_offsets());
    let pilot = read_table(client, state.pilot_offsets());
    let spatial_nonzero = nonzero_off_centre(&spatial, radius);
    let pilot_nonzero = nonzero_off_centre(&pilot, radius);

    assert!(
        spatial_nonzero,
        "the spatial table starts nonzero at radius {radius}"
    );
    assert!(pilot_nonzero, "the pilot table starts nonzero at radius {radius}");
}

fn read_table(client: &Client, table: Option<&Handle>) -> Vec<f32> {
    let handle = table.expect("the state holds offset tables").clone();
    let bytes = client.read_one(handle).expect("the table reads back");
    let entries = f32::from_bytes(&bytes);

    entries.to_vec()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Checks the spatial table against the host table at the derived rho and noise offset,
/// returning the largest relative gap.
fn check_spatial_table(client: &Client, state: &NoiseState, radius: u32, derived: DerivedValues) -> f32 {
    let table = read_table(client, state.spatial_offsets());
    let expected = build_spatial_offset_lut(radius, derived.rho, derived.noise_offset);
    let table_len = table.len();
    let expected_len = expected.len();
    assert_eq!(table_len, expected_len, "the table's entries at radius {radius}");

    let centre = spatial_offset_index(radius, 0, 0) as usize;
    let centre_bits = table[centre].to_bits();
    assert_eq!(centre_bits, 0, "the centre at radius {radius}");

    let mut largest = 0.0f32;
    let pairs = table.iter().zip(&expected).enumerate();
    for (index, (&entry, &host)) in pairs {
        let gap = relative_gap(entry, host);
        assert!(
            gap <= MAX_RELATIVE,
            "entry {index} at radius {radius} is {gap:e} apart, gpu {entry:e} host {host:e}"
        );
        largest = largest.max(gap);
    }

    largest
}

#[test]
fn the_spatial_table_matches_the_host_table() {
    let client = test_client();
    let shape = shape(&client, WIDTH, HEIGHT);
    let mut largest = 0.0f32;

    for radius in RADII {
        for rho in RHOS {
            let table_settings = settings(radius, COEFFICIENTS, StrengthMapKind::Off);
            let second_rho = 0.5 * rho + 0.2;
            let rhos = [rho, second_rho, second_rho];
            let state = ring_state(&client, shape, table_settings, rhos, &[]);

            state.queue_fold(FIRST_FOLD).expect("the fold queues");
            let first = state.snapshot().expect("the snapshot reads back");
            assert_eq!(first.derived.rho, rho, "the first sample seeds rho");

            let first_gap = check_spatial_table(&client, &state, radius, first.derived);

            state.queue_fold(SECOND_FOLD).expect("the fold queues");
            let second = state.snapshot().expect("the snapshot reads back");
            let derived = second.derived;
            let smoothed = EMA_ALPHA * second_rho + (1.0 - EMA_ALPHA) * rho;
            let rho_gap = relative_gap(derived.rho, smoothed);
            assert_ne!(derived.rho, second_rho, "the second fold smooths rho");
            assert!(rho_gap <= MAX_RELATIVE, "the smoothed rho is {rho_gap:e} apart");
            assert!(derived.noise_offset > 0.0, "the noise offset is in use");

            let second_gap = check_spatial_table(&client, &state, radius, derived);
            largest = largest.max(first_gap).max(second_gap);
        }
    }

    println!("largest spatial table gap {largest:e}");
    sync(&client).expect("the device syncs");
}

#[test]
fn the_pilot_table_is_flat_with_a_zero_centre() {
    let client = test_client();
    let shape = shape(&client, WIDTH, HEIGHT);
    let pilot_coefficients = DerivedCoefficients {
        main_offset: false,
        ..COEFFICIENTS
    };

    for radius in RADII {
        let table_settings = settings(radius, pilot_coefficients, StrengthMapKind::Off);
        let rhos = [0.65; FRAMES as usize];
        let state = ring_state(&client, shape, table_settings, rhos, &[]);
        check_nonzero_start(&client, &state, radius);

        state.queue_fold(FIRST_FOLD).expect("the fold queues");
        let snapshot = state.snapshot().expect("the snapshot reads back");
        let derived = snapshot.derived;
        assert!(derived.input_noise_offset > 0.0, "the input offset is in use");

        let pilot = read_table(&client, state.pilot_offsets());
        let expected = build_spatial_offset_lut(radius, 0.0, derived.input_noise_offset);
        let pilot_bits = bits(&pilot);
        let expected_bits = bits(&expected);
        assert_eq!(pilot_bits, expected_bits, "the pilot table at radius {radius}");

        let spatial = read_table(&client, state.spatial_offsets());
        let spatial_bits = bits(&spatial);
        let zero_bits = vec![0u32; spatial.len()];
        assert_eq!(spatial_bits, zero_bits, "the spatial table at radius {radius}");
    }

    sync(&client).expect("the device syncs");
}

#[test]
fn a_zero_offset_gives_a_zero_table() {
    let client = test_client();
    let shape = shape(&client, WIDTH, HEIGHT);
    let unfloored = DerivedCoefficients {
        offset_per_variance: 0.0,
        ..COEFFICIENTS
    };

    for radius in RADII {
        let table_settings = settings(radius, unfloored, StrengthMapKind::Off);
        let rhos = [0.3; FRAMES as usize];
        let state = ring_state(&client, shape, table_settings, rhos, &[]);
        check_nonzero_start(&client, &state, radius);

        state.queue_fold(FIRST_FOLD).expect("the fold queues");
        let spatial = read_table(&client, state.spatial_offsets());
        let pilot = read_table(&client, state.pilot_offsets());
        let spatial_bits = bits(&spatial);
        let pilot_bits = bits(&pilot);
        let zero_bits = vec![0u32; spatial.len()];

        assert_eq!(spatial_bits, zero_bits, "the spatial table at radius {radius}");
        assert_eq!(pilot_bits, zero_bits, "the pilot table at radius {radius}");
    }

    sync(&client).expect("the device syncs");
}

/// A slot map of `quarters` multipliers that name the slot and the quarter.
fn slot_map(slot: u32, quarters: u32) -> Vec<f32> {
    (0..quarters)
        .map(|quarter| 1.0 + 0.01 * slot as f32 + 1.0e-4 * quarter as f32)
        .collect()
}

#[test]
fn a_fold_with_a_map_writes_the_tables_and_copies_the_map() {
    const WIDTH_4K: u32 = 3840;
    const HEIGHT_4K: u32 = 2160;
    const RADIUS: u32 = 8;
    let client = test_client();
    let shape = shape(&client, WIDTH_4K, HEIGHT_4K);
    let (cols, rows) = strength_map_dims(WIDTH_4K, HEIGHT_4K);
    let quarters = cols * rows;
    let map_cubes = quarters.div_ceil(NOISE_BLOCK_1D);
    assert!(map_cubes > 1, "the map spans {map_cubes} cubes");

    let slot_maps: Vec<Vec<f32>> = (0..FRAMES).map(|slot| slot_map(slot, quarters)).collect();
    let map = StrengthMapKind::Luma(LUMA_PARAMS);
    let table_settings = settings(RADIUS, COEFFICIENTS, map);
    let state = folded_state(&client, shape, table_settings, 0.3, &slot_maps);
    let snapshot = state.snapshot().expect("the snapshot reads back");

    let current_map = state.read_current_map();
    let map_bits = bits(&current_map);
    let lender_map_bits = bits(&slot_maps[0]);
    assert_eq!(map_bits, lender_map_bits, "the current map");

    let gap = check_spatial_table(&client, &state, RADIUS, snapshot.derived);
    println!("largest spatial table gap with a map {gap:e}");

    let pilot = read_table(&client, state.pilot_offsets());
    let expected_pilot = build_spatial_offset_lut(RADIUS, 0.0, snapshot.derived.input_noise_offset);
    let pilot_bits = bits(&pilot);
    let expected_pilot_bits = bits(&expected_pilot);
    assert_eq!(pilot_bits, expected_pilot_bits, "the pilot table");

    sync(&client).expect("the device syncs");
}

fn fixed_values(noise_offset: f32, input_noise_offset: f32) -> DerivedValues {
    DerivedValues {
        noise_offset,
        input_noise_offset,
        rho: 0.4,
        ..DerivedValues::default()
    }
}

/// Checks both tables of `state` bit for bit against the host tables of `derived` at a rho of 0.
fn check_host_tables(client: &Client, state: &NoiseState, radius: u32, derived: DerivedValues) {
    let spatial = read_table(client, state.spatial_offsets());
    let pilot = read_table(client, state.pilot_offsets());
    let expected_spatial = build_spatial_offset_lut(radius, 0.0, derived.noise_offset);
    let expected_pilot = build_spatial_offset_lut(radius, 0.0, derived.input_noise_offset);

    let spatial_bits = bits(&spatial);
    let pilot_bits = bits(&pilot);
    let expected_spatial_bits = bits(&expected_spatial);
    let expected_pilot_bits = bits(&expected_pilot);

    assert_eq!(spatial_bits, expected_spatial_bits, "the spatial table");
    assert_eq!(pilot_bits, expected_pilot_bits, "the pilot table");
}

#[test]
fn a_fixed_state_builds_the_host_tables() {
    let client = test_client();
    let shape = shape(&client, WIDTH, HEIGHT);

    for radius in RADII {
        let derived = fixed_values(0.004, 0.006);
        let state = NoiseState::fixed(&client, shape, derived, Some(radius));

        check_host_tables(&client, &state, radius, derived);
    }

    let untabled = NoiseState::fixed(&client, shape, DerivedValues::default(), None);
    let spatial_absent = untabled.spatial_offsets().is_none();
    let pilot_absent = untabled.pilot_offsets().is_none();
    assert!(spatial_absent, "a state without tables has no spatial table");
    assert!(pilot_absent, "a state without tables has no pilot table");

    sync(&client).expect("the device syncs");
}

#[test]
fn new_fixed_values_and_a_new_radius_rebuild_the_host_tables() {
    let client = test_client();
    let shape = shape(&client, WIDTH, HEIGHT);
    let first = fixed_values(0.004, 0.006);
    let mut state = NoiseState::fixed(&client, shape, first, Some(2));

    let replaced = fixed_values(0.0, 0.009);
    state.set_fixed(replaced);
    check_host_tables(&client, &state, 2, replaced);

    let mut wider = state.settings();
    wider.fold.offset_tables = Some(8);
    state.set_settings(wider);
    check_host_tables(&client, &state, 8, replaced);

    let mut untabled = state.settings();
    untabled.fold.offset_tables = None;
    state.set_settings(untabled);
    let spatial_absent = state.spatial_offsets().is_none();
    let pilot_absent = state.pilot_offsets().is_none();
    assert!(spatial_absent, "dropping the radius drops the spatial table");
    assert!(pilot_absent, "dropping the radius drops the pilot table");

    sync(&client).expect("the device syncs");
}

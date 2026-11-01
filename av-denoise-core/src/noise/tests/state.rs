use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::align::StorageAlign;
use crate::noise::derived::DerivedValues;
use crate::noise::fold::{FoldSettings, NoiseSettings};
use crate::noise::history::WindowBounds;
use crate::noise::spatial::{NoiseCtx, noise_partials_slot_stride_bytes, run_noise_estimate};
use crate::noise::state::{NoiseShape, NoiseState, ReadingSettings};
use crate::noise::temporal::{TemporalStatsCtx, run_temporal_noise_stats};
use crate::tune::gpu::sync;
use crate::tune::tests::fixtures::{hashed_ring, upload};
use crate::tune::tests::runtime::test_client;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const FRAMES: u32 = 3;

fn luma_shape(client: &Client, temporal: bool) -> NoiseShape {
    NoiseShape {
        width: WIDTH,
        height: HEIGHT,
        channels: 1,
        frames: FRAMES,
        ring_radius: FRAMES / 2,
        temporal,
        align: StorageAlign::from_client(client),
    }
}

fn luma_settings(luma_fields: bool) -> ReadingSettings {
    ReadingSettings { luma_fields }
}

fn luma_state(client: &Client, luma_fields: bool) -> NoiseState {
    let shape = luma_shape(client, true);
    let settings = luma_settings(luma_fields);

    let noise_settings = NoiseSettings {
        reading: settings,
        ..NoiseSettings::default()
    };

    NoiseState::new(client, shape, noise_settings, DerivedValues::default())
}

fn luma_ring(client: &Client) -> Handle {
    let frames = hashed_ring(WIDTH, HEIGHT, 1, FRAMES, 7);
    upload(client, &frames)
}

/// Reads every handle back as raw `f32` bits.
fn read_bits(client: &Client, handles: &[Handle]) -> Vec<Vec<u32>> {
    let handles = handles.to_vec();
    let bytes = client.read(handles).expect("the handles read back");

    bytes
        .iter()
        .map(|buffer| u32::from_bytes(buffer).to_vec())
        .collect()
}

/// Runs the Immerkær estimate and the temporal stats for `slot` against `previous` on `state`'s buffers.
fn launch_directly(
    client: &Client,
    state: &NoiseState,
    ring: &Handle,
    slot: u32,
    previous: u32,
    luma_fields: bool,
) {
    let shape = state.shape();
    let stride = noise_partials_slot_stride_bytes(shape.width, shape.height, shape.align);
    let partials_start = u64::from(slot) * stride;
    let partials_slot = state.partials().clone().offset_start(partials_start);
    let stats_buf = state.stats().expect("temporal stats are measured");

    let noise_ctx = NoiseCtx {
        width: shape.width,
        height: shape.height,
        channels: shape.channels,
        frame_count: shape.frames,
        frame: slot,
        slot,
        input_buf: ring,
        partials_buf: &partials_slot,
        results_buf: state.results(),
    };
    run_noise_estimate(client, &noise_ctx).expect("the estimate queues");

    let stats_ctx = TemporalStatsCtx {
        width: shape.width,
        height: shape.height,
        channels: shape.channels,
        frame_count: shape.frames,
        slot_new: slot,
        slot_prev: previous,
        input_buf: ring,
        stats_buf,
        align: shape.align,
        motion: None,
    };
    run_temporal_noise_stats(client, &stats_ctx, luma_fields).expect("the stats queue");
}

fn assert_queued_stats_match_the_direct_launches(luma_fields: bool) {
    let client = test_client();
    let ring = luma_ring(&client);
    let queued = luma_state(&client, luma_fields);
    let direct = luma_state(&client, luma_fields);

    queued.queue_stats(&ring, 1, Some(0)).expect("the stats queue");
    launch_directly(&client, &direct, &ring, 1, 0, luma_fields);

    let queued_handles = queued.reading_handles(1);
    let direct_handles = direct.reading_handles(1);
    let queued_bits = read_bits(&client, &queued_handles);
    let direct_bits = read_bits(&client, &direct_handles);

    assert_eq!(queued_bits.len(), 3);
    assert_eq!(queued_bits, direct_bits);
}

#[test]
fn queued_stats_match_the_direct_launches() {
    assert_queued_stats_match_the_direct_launches(true);
}

#[test]
fn queued_stats_without_luma_fields_match_the_direct_launches() {
    assert_queued_stats_match_the_direct_launches(false);
}

#[test]
fn a_slot_with_no_previous_frame_reads_a_zeroed_record() {
    let client = test_client();
    let ring = luma_ring(&client);
    let state = luma_state(&client, true);
    let handles = state.reading_handles(2);

    state.queue_stats(&ring, 2, Some(1)).expect("the stats queue");
    let measured = read_bits(&client, &handles);
    let measured_any = measured[2].iter().any(|&value| value != 0);
    assert!(measured_any);

    state.queue_stats(&ring, 2, None).expect("the stats queue");
    let zeroed = read_bits(&client, &handles);

    let partials_any = zeroed[1].iter().any(|&value| value != 0);
    let records_zero = zeroed[2].iter().all(|&value| value == 0);
    assert!(partials_any);
    assert!(records_zero);
}

#[test]
fn a_state_without_temporal_stats_hands_out_no_records() {
    let client = test_client();
    let ring = luma_ring(&client);
    let shape = luma_shape(&client, false);
    let settings = NoiseSettings {
        reading: luma_settings(true),
        ..NoiseSettings::default()
    };
    let state = NoiseState::new(&client, shape, settings, DerivedValues::default());

    state.queue_stats(&ring, 1, Some(0)).expect("the estimate queues");

    let handles = state.reading_handles(1);
    assert_eq!(handles.len(), 2);
    assert!(state.stats().is_none());

    sync(&client).expect("the device syncs");
}

#[test]
fn set_settings_replaces_every_setting() {
    let client = test_client();
    let mut state = luma_state(&client, true);

    let scaled_reading = ReadingSettings { luma_fields: false };
    let fold = FoldSettings {
        sigma_scale: 1.25,
        ..FoldSettings::default()
    };
    let scaled = NoiseSettings {
        reading: scaled_reading,
        fold,
    };
    state.set_settings(scaled);
    assert_eq!(state.settings(), scaled);

    let luma_reading = ReadingSettings { luma_fields: true };
    let luma = NoiseSettings {
        reading: luma_reading,
        ..NoiseSettings::default()
    };
    state.set_settings(luma);
    assert_eq!(state.settings(), luma);

    sync(&client).expect("the device syncs");
}

#[test]
fn a_fixed_state_queues_nothing_and_snapshots_its_values() {
    let client = test_client();
    let ring = luma_ring(&client);
    let shape = luma_shape(&client, true);
    let fixed = DerivedValues {
        sigma: [0.02, 0.0],
        sad_noise_floor: 5.75,
        h2_inv_norm: 12.5,
        ..DerivedValues::default()
    };
    let state = NoiseState::fixed(&client, shape, fixed, None);

    state.queue_stats(&ring, 1, Some(0)).expect("the stats queue");
    let launches = state.queue_reading(1).expect("the reading queues");
    state.record_reading(1, 0);
    let bounds = WindowBounds::around(0, 0);
    state.queue_window_fold(bounds).expect("the fold queues");
    state.reset_stream();

    assert!(!state.is_measured());
    assert_eq!(launches, 0);
    assert_eq!(state.stream_frame_count(), 0);

    let snapshot = state.snapshot().expect("the snapshot reads back");
    assert_eq!(snapshot.derived, fixed);
    assert_eq!(snapshot.median, None);
    assert_eq!(snapshot.curve, None);

    let curve = state.read_curve();
    assert!(curve.iter().all(|&ratio| ratio == 0.0));

    sync(&client).expect("the device syncs");
}

#[test]
fn a_fixed_state_snapshots_the_values_it_was_last_given() {
    let client = test_client();
    let shape = luma_shape(&client, false);
    let unset = DerivedValues::default();
    let mut state = NoiseState::fixed(&client, shape, unset, None);
    let replaced = DerivedValues {
        confidence_sad_noise_floor: 3.5,
        ..DerivedValues::default()
    };

    state.set_fixed(replaced);

    let snapshot = state.snapshot().expect("the snapshot reads back");
    assert_eq!(snapshot.derived, replaced);

    sync(&client).expect("the device syncs");
}

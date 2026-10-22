use std::sync::Arc;

use cubecl::prelude::*;

use crate::bench_api::HostIo;
use crate::nlmeans::denoiser::readings::SlotNoiseReading;
use crate::nlmeans::{ChannelMode, HqParams, MotionCompensationMode, NlmDenoiser, NlmParams, PrefilterMode};
use crate::tune::gpu::sync;
use crate::tune::tests::fixtures::hashed_unit;
use crate::tune::tests::runtime::{TestRuntime, test_client};

const SIZE: u32 = 64;
const RADIUS: u32 = 2;
const TOTAL_FRAMES: u32 = 2 * RADIUS + 1;
/// The width of the uniform grain added to every frame.
const GRAIN_SPAN: f32 = 0.08;

fn hq_denoiser(client: &ComputeClient<TestRuntime>, shifted_edges: bool) -> NlmDenoiser<TestRuntime> {
    let params = NlmParams {
        temporal_radius: RADIUS,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::mvtools_default(),
        hq: Some(HqParams::default()),
    };
    let mut denoiser = NlmDenoiser::new(client, params, SIZE, SIZE);
    denoiser.set_shifted_edges(shifted_edges);
    denoiser.set_luma_noise_fields(true);
    denoiser
}

fn shifted_hq_denoiser(client: &ComputeClient<TestRuntime>) -> NlmDenoiser<TestRuntime> {
    hq_denoiser(client, true)
}

/// A still textured scene with fresh grain for every `seed`.
fn grain_frame(seed: u32) -> Vec<f32> {
    (0..SIZE * SIZE)
        .map(|index| {
            let x_fraction = (index % SIZE) as f32 / SIZE as f32;
            let y_fraction = (index / SIZE) as f32 / SIZE as f32;
            let wave_x = (x_fraction * 6.0 * std::f32::consts::PI).sin();
            let wave_y = (y_fraction * 4.0 * std::f32::consts::PI).cos();
            let texture = 0.5 + 0.15 * wave_x * wave_y;
            let grain = (hashed_unit(index, seed) - 0.5) * GRAIN_SPAN;
            texture + grain
        })
        .collect()
}

fn push_grain(denoiser: &mut NlmDenoiser<TestRuntime>, seeds: std::ops::Range<u32>) {
    for seed in seeds {
        let frame = grain_frame(seed);
        denoiser.push_frame(&frame);
    }
}

/// Reads one slot with a blocking read.
fn blocking_reading(
    denoiser: &NlmDenoiser<TestRuntime>,
    client: &ComputeClient<TestRuntime>,
    slot: u32,
) -> SlotNoiseReading {
    let handles = denoiser.noise_reading_handles(slot).expect("auto noise is on");
    let bytes = client.read(handles);
    let slices: Vec<&[u8]> = bytes.iter().map(|buffer| &buffer[..]).collect();
    let shape = denoiser.reading_shape().expect("auto noise is on");

    SlotNoiseReading::from_bytes(&slices, &shape)
}

fn same_reading(stored: &Arc<SlotNoiseReading>, expected: &SlotNoiseReading) -> bool {
    let stored_sums = stored.immerkaer_sums.map(f32::to_bits);
    let expected_sums = expected.immerkaer_sums.map(f32::to_bits);
    let stored_low = stored.immerkaer_low.map(f32::to_bits);
    let expected_low = expected.immerkaer_low.map(f32::to_bits);
    let sample_match = stored.temporal.sample == expected.temporal.sample;
    let curve_match = stored.temporal.curve == expected.temporal.curve;
    let classes_match = stored.temporal.classes == expected.temporal.classes;

    stored_sums == expected_sums && stored_low == expected_low && sample_match && curve_match && classes_match
}

/// Checks every logical ring position's stored reading against a blocking read of its slot.
///
/// Returns how many of the readings carry a temporal sample.
fn assert_store_matches_blocking_reads(
    denoiser: &mut NlmDenoiser<TestRuntime>,
    client: &ComputeClient<TestRuntime>,
) -> usize {
    let mut with_sample = 0;

    for logical in 0..TOTAL_FRAMES {
        let slot = denoiser.ring_slot(logical);
        let expected = blocking_reading(denoiser, client, slot);
        let stored = denoiser.noise_reading_at(logical).expect("the reading lands");

        assert!(same_reading(&stored, &expected), "logical {logical}");

        if stored.temporal.sample.is_some() {
            with_sample += 1;
        }
    }

    with_sample
}

#[test]
fn the_store_hands_out_what_a_blocking_read_returns() {
    let client = test_client();
    let mut denoiser = shifted_hq_denoiser(&client);
    push_grain(&mut denoiser, 0..TOTAL_FRAMES);

    let with_sample = assert_store_matches_blocking_reads(&mut denoiser, &client);

    assert!(with_sample > 0, "no reading carried a temporal sample");
    assert_eq!(denoiser.noise_reading_fallbacks(), 0);
}

#[test]
fn a_short_stream_hands_out_what_a_blocking_read_returns_for_its_copies() {
    let client = test_client();
    let mut denoiser = shifted_hq_denoiser(&client);
    push_grain(&mut denoiser, 0..2);
    denoiser.fill_ring_with_last_frame().expect("the ring fills");

    assert_store_matches_blocking_reads(&mut denoiser, &client);
    assert_eq!(denoiser.noise_reading_fallbacks(), 0);
}

#[test]
fn a_steady_stream_never_reads_a_slot_twice() {
    let client = test_client();
    let mut denoiser = shifted_hq_denoiser(&client);

    for seed in 0..16 {
        let frame = grain_frame(seed);
        denoiser.push_frame(&frame);

        denoiser.denoise().expect("denoise");
    }

    assert_eq!(denoiser.noise_reading_fallbacks(), 0);
}

#[test]
fn a_slot_with_no_reading_falls_back_to_a_counted_read() {
    let client = test_client();
    let mut denoiser = shifted_hq_denoiser(&client);
    push_grain(&mut denoiser, 0..TOTAL_FRAMES);

    let slot = denoiser.ring_slot(RADIUS);
    let expected = blocking_reading(&denoiser, &client, slot);
    denoiser.noise_readings.clear();

    let fallback = denoiser
        .noise_reading_at(RADIUS)
        .expect("the fallback read lands");
    let again = denoiser
        .noise_reading_at(RADIUS)
        .expect("the stored reading is handed out");

    assert!(same_reading(&fallback, &expected));
    assert!(same_reading(&again, &expected));
    assert_eq!(denoiser.noise_reading_fallbacks(), 1);
}

#[test]
fn a_reset_with_readings_in_flight_reuses_every_slot_correctly() {
    let client = test_client();
    let mut reused = shifted_hq_denoiser(&client);
    push_grain(&mut reused, 100..103);
    reused.reset_stream_state();

    for slot in 0..3u32 {
        let stale = reused
            .noise_readings
            .get(slot, slot as usize)
            .expect("no read error");
        assert!(stale.is_none(), "slot {slot} kept the old stream's reading");
    }

    push_grain(&mut reused, 0..TOTAL_FRAMES);
    reused.denoise().expect("denoise");

    let mut fresh = shifted_hq_denoiser(&client);
    push_grain(&mut fresh, 0..TOTAL_FRAMES);
    fresh.denoise().expect("denoise");

    assert_eq!(reused.current_sigmas(), fresh.current_sigmas());
    assert_eq!(
        reused.current_sigmas_temporal_only(),
        fresh.current_sigmas_temporal_only()
    );
    assert_eq!(reused.current_noise_curve(), fresh.current_noise_curve());
}

#[test]
fn dropping_a_denoiser_with_readings_in_flight_leaves_the_device_usable() {
    let client = test_client();
    let mut abandoned = shifted_hq_denoiser(&client);
    push_grain(&mut abandoned, 0..3);
    drop(abandoned);

    let mut after = shifted_hq_denoiser(&client);
    push_grain(&mut after, 0..TOTAL_FRAMES);
    let denoised = after.denoise().expect("the device still runs");

    assert!(denoised.is_some());
    sync(&client).expect("the device syncs");
}

#[test]
fn a_push_never_polls_the_readings_it_started() {
    let client = test_client();
    let mut denoiser = hq_denoiser(&client, false);
    push_grain(&mut denoiser, 0..1);
    sync(&client).expect("the device syncs");

    denoiser.process_landed_readings().expect("no read error");

    // The first push primes the leading edge, so slots 1..=RADIUS hold its duplicates.
    for slot in 1..=RADIUS {
        assert!(
            !denoiser.noise_readings.is_landed(slot),
            "slot {slot} was polled by the push that started it"
        );
    }

    let timing = denoiser.timing();
    assert_eq!(timing.readback_waits, 1);
}

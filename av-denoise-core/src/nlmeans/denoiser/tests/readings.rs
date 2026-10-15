use std::time::{Duration, Instant};

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::nlmeans::BLOCK_1D;
use crate::nlmeans::align::StorageAlign;
use crate::nlmeans::denoiser::readings::{InFlightRead, NoiseReadings, ReadingShape, SlotNoiseReading};
use crate::nlmeans::kernels::gpu_copy;
use crate::nlmeans::noise::{
    QuarterSettings,
    RESULTS_PER_SLOT,
    noise_partials_slot_stride_bytes,
    noise_partials_slot_view,
    noise_results_slot_view,
    partials_len,
    sigma_block_p25_from_partials,
    temporal_stats_buf_bytes,
    temporal_stats_slot_view,
    zero_temporal_stats_slot,
};
use crate::tune::gpu::sync;
use crate::tune::tests::runtime::{TestRuntime, test_client};

const SIZE: u32 = 64;

fn luma_shape() -> ReadingShape {
    ReadingShape {
        width: SIZE,
        height: SIZE,
        channels: 1,
        stored_ch: 1,
        with_curve: false,
        settings: QuarterSettings::default(),
    }
}

fn ramp(count: usize, scale: f32) -> Vec<f32> {
    (0..count).map(|index| (index % 17) as f32 * scale).collect()
}

/// Copies every value of `source` over `target` with the ring copy kernel.
fn overwrite(client: &ComputeClient<TestRuntime>, source: &Handle, target: &Handle, length: u32) {
    let grid = length.div_ceil(BLOCK_1D);
    let total_threads = grid * BLOCK_1D;
    let cube_count = CubeCount::new_1d(grid);
    let cube_dim = CubeDim::new_1d(BLOCK_1D);
    let source_handle = source.clone();
    let target_handle = target.clone();

    unsafe {
        let source_arg = ArrayArg::from_raw_parts(source_handle, length as usize);
        let target_arg = ArrayArg::from_raw_parts(target_handle, length as usize);

        gpu_copy::launch_unchecked::<TestRuntime>(
            client,
            cube_count,
            cube_dim,
            source_arg,
            target_arg,
            0,
            0,
            length,
            total_threads,
        );
    }
}

#[test]
fn a_reading_turns_its_buffers_into_values() {
    let row = [0.1f32, 0.2, 0.3, 0.4];
    let partials_count = partials_len(SIZE, SIZE);
    let partials = ramp(partials_count, 0.5);
    let row_bytes = f32::as_bytes(&row);
    let partials_bytes = f32::as_bytes(&partials);
    let buffers = [row_bytes, partials_bytes];
    let shape = luma_shape();

    let reading = SlotNoiseReading::from_bytes(&buffers, &shape);

    let expected_low = sigma_block_p25_from_partials(&partials, 1, SIZE, SIZE);
    assert_eq!(reading.immerkaer_sums, row);
    assert_eq!(reading.immerkaer_low, expected_low);
    assert!(reading.temporal.sample.is_none());
}

#[test]
fn a_results_slot_view_reads_only_its_row() {
    let client = test_client();
    let frame_count = 3u32;
    let results = ramp(frame_count as usize * RESULTS_PER_SLOT, 1.0);
    let results_bytes = f32::as_bytes(&results);
    let results_buf = client.create_from_slice(results_bytes);

    let view = noise_results_slot_view(&results_buf, frame_count, 1);
    let bytes = client.read_one(view).expect("the view reads");
    let values = f32::from_bytes(&bytes);

    let expected = &results[RESULTS_PER_SLOT..2 * RESULTS_PER_SLOT];
    assert_eq!(values, expected);
}

#[test]
fn a_partials_slot_view_reads_only_its_slot() {
    let client = test_client();
    let align = StorageAlign::from_client(&client);
    let frame_count = 3u32;
    let slot = 2u32;
    let stride_bytes = noise_partials_slot_stride_bytes(SIZE, SIZE, align);
    let stride_floats = stride_bytes as usize / size_of::<f32>();
    let partials = ramp(frame_count as usize * stride_floats, 0.5);
    let partials_bytes = f32::as_bytes(&partials);
    let partials_buf = client.create_from_slice(partials_bytes);

    let view = noise_partials_slot_view(&partials_buf, SIZE, SIZE, frame_count, slot, align);
    let bytes = client.read_one(view).expect("the view reads");
    let values = f32::from_bytes(&bytes);

    let start = slot as usize * stride_floats;
    let end = start + partials_len(SIZE, SIZE);
    assert_eq!(values, &partials[start..end]);
}

#[test]
fn a_started_read_lands_between_the_kernels_around_it() {
    let client = test_client();
    let align = StorageAlign::from_client(&client);
    let frame_count = 3u32;
    let stats_bytes = temporal_stats_buf_bytes(SIZE, SIZE, 1, frame_count, align);
    let stats_count = stats_bytes / size_of::<f32>();
    let ones = vec![1.0f32; stats_count];
    let ones_bytes = f32::as_bytes(&ones);
    let stats = client.create_from_slice(ones_bytes);
    let sevens = vec![7.0f32; stats_count];
    let sevens_bytes = f32::as_bytes(&sevens);
    let source = client.create_from_slice(sevens_bytes);

    zero_temporal_stats_slot::<TestRuntime>(&client, &stats, SIZE, SIZE, 1, 1, align);
    let slot = temporal_stats_slot_view(&stats, SIZE, SIZE, 1, frame_count, 1, align);
    let mut read = InFlightRead::start(&client, vec![slot]);
    overwrite(&client, &source, &stats, stats_count as u32);

    let landed = read.wait().expect("the read lands");
    let values = f32::from_bytes(&landed[0]);
    assert!(!values.is_empty());
    assert!(
        values.iter().all(|&value| value == 0.0),
        "the read ran before the zero kernel or after the overwrite"
    );
}

#[test]
fn a_polled_read_hands_back_the_bytes_once() {
    let client = test_client();
    let values = ramp(4096, 0.25);
    let values_bytes = f32::as_bytes(&values);
    let buffer = client.create_from_slice(values_bytes);
    let mut read = InFlightRead::start(&client, vec![buffer]);

    let deadline = Instant::now() + Duration::from_secs(30);
    let landed = loop {
        if let Some(result) = read.poll_now() {
            break result.expect("the read lands");
        }

        assert!(Instant::now() < deadline, "the read never landed");
        std::thread::yield_now();
    };

    let landed_values = f32::from_bytes(&landed[0]);
    assert_eq!(landed_values, values.as_slice());
    assert!(read.poll_now().is_none());
}

#[test]
fn polling_after_a_wait_returns_nothing() {
    let client = test_client();
    let values = ramp(256, 1.0);
    let values_bytes = f32::as_bytes(&values);
    let buffer = client.create_from_slice(values_bytes);
    let mut read = InFlightRead::start(&client, vec![buffer]);

    read.wait().expect("the read lands");

    assert!(read.poll_now().is_none());
}

#[test]
fn dropping_a_started_read_leaves_the_device_usable() {
    let client = test_client();
    let values = ramp(4096, 0.25);
    let values_bytes = f32::as_bytes(&values);
    let buffer = client.create_from_slice(values_bytes);

    for _ in 0..8 {
        let handles = vec![buffer.clone()];
        let read = InFlightRead::start(&client, handles);
        drop(read);
    }

    let later = client.create_from_slice(values_bytes);
    let bytes = client.read_one(later).expect("the device still reads");
    let read_back = f32::from_bytes(&bytes);
    assert_eq!(read_back, values.as_slice());

    sync(&client).expect("the device syncs");
}

/// Uploads a results row and a partials buffer that read back as one slot's reading.
fn slot_handles(client: &ComputeClient<TestRuntime>, row: [f32; 4], partials_scale: f32) -> Vec<Handle> {
    let partials_count = partials_len(SIZE, SIZE);
    let partials = ramp(partials_count, partials_scale);
    let row_bytes = f32::as_bytes(&row);
    let partials_bytes = f32::as_bytes(&partials);
    let row_handle = client.create_from_slice(row_bytes);
    let partials_handle = client.create_from_slice(partials_bytes);

    vec![row_handle, partials_handle]
}

#[test]
fn a_reading_taken_before_it_lands_waits_and_returns_its_values() {
    let client = test_client();
    let mut readings = NoiseReadings::new(3);
    let row = [0.5f32, 0.25, 0.125, 0.0];
    let handles = slot_handles(&client, row, 0.5);
    let shape = luma_shape();

    readings.start(&client, 1, 7, handles, shape);
    let reading = readings
        .get(1, 7)
        .expect("the read lands")
        .expect("the push left a reading");

    assert_eq!(reading.immerkaer_sums, row);
    assert_eq!(readings.stalls().waits(), 1);
}

#[test]
fn a_reading_for_another_push_is_not_handed_out() {
    let client = test_client();
    let mut readings = NoiseReadings::new(3);
    let handles = slot_handles(&client, [0.5; 4], 0.5);
    let shape = luma_shape();

    readings.start(&client, 1, 3, handles, shape);
    let stale = readings.get(1, 6).expect("no read error");

    assert!(stale.is_none());
}

#[test]
fn starting_a_slot_again_drops_the_previous_frames_reading() {
    let client = test_client();
    let mut readings = NoiseReadings::new(2);
    let first = slot_handles(&client, [1.0; 4], 0.5);
    let second = slot_handles(&client, [2.0; 4], 0.75);
    let shape = luma_shape();

    readings.start(&client, 0, 1, first, shape);
    readings.start(&client, 0, 3, second, shape);

    let old = readings.get(0, 1).expect("no read error");
    let new = readings
        .get(0, 3)
        .expect("no read error")
        .expect("the newer push left a reading");
    assert!(old.is_none());
    assert_eq!(new.immerkaer_sums, [2.0; 4]);
}

#[test]
fn processing_skips_the_newest_push_and_lands_the_rest() {
    let client = test_client();
    let mut readings = NoiseReadings::new(3);
    let older = slot_handles(&client, [1.0; 4], 0.5);
    let newest = slot_handles(&client, [2.0; 4], 0.5);
    let shape = luma_shape();

    readings.start(&client, 0, 4, older, shape);
    readings.start(&client, 1, 5, newest, shape);
    sync(&client).expect("the device drains");

    readings.process_landed(5).expect("no read error");

    assert!(readings.is_landed(0));
    assert!(!readings.is_landed(1));

    let landed = readings
        .get(0, 4)
        .expect("no read error")
        .expect("the older push left a reading");
    assert_eq!(landed.immerkaer_sums, [1.0; 4]);
}

#[test]
fn a_failed_reading_reports_its_error_on_every_get() {
    let mut readings = NoiseReadings::new(2);

    readings.fail_slot(1, 4, "device lost");

    let first = readings.get(1, 4).expect_err("the read failed");
    let second = readings.get(1, 4).expect_err("the read still failed");
    assert_eq!(first.to_string(), "noise readback failed: device lost");
    assert_eq!(second.to_string(), first.to_string());
}

#[test]
fn clearing_with_reads_in_flight_leaves_every_slot_reusable() {
    let client = test_client();
    let mut readings = NoiseReadings::new(3);
    let shape = luma_shape();

    for slot in 0..3u32 {
        let handles = slot_handles(&client, [slot as f32; 4], 0.5);
        readings.start(&client, slot, slot as usize, handles, shape);
    }

    readings.clear();

    let fresh = slot_handles(&client, [9.0; 4], 0.25);
    readings.start(&client, 2, 0, fresh, shape);
    let cleared = readings.get(0, 0).expect("no read error");
    let reused = readings
        .get(2, 0)
        .expect("no read error")
        .expect("the new stream left a reading");
    assert!(cleared.is_none());
    assert_eq!(reused.immerkaer_sums, [9.0; 4]);
}

#![cfg(any(feature = "vulkan", feature = "metal"))]

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::synthetic::gaussian_field;
use crate::nl4d::grain::chunk::GrainChunk;
use crate::nl4d::grain::consts::{
    CELL,
    CHUNK_FRAMES,
    CLIP_HIGH,
    CLIP_LOW,
    FLAT_RANGE,
    FRAME_RECORD_LEN,
    HISTORY_ENTRY_LEN,
    HISTORY_FRAMES,
    LAG_RECORD_LEN,
    LUMA_HIGH,
    LUMA_LOW,
    PATTERNS,
    PERSIST_BLOCK_CELLS,
    PERSIST_THREADS,
    PERSISTENCE_LAGS,
    PROJECTIONS,
    REDUCE_THREADS,
};
use crate::nl4d::grain::export::pattern_words;
use crate::nl4d::grain::report::{FrameGrain, frame_rows};
use crate::nl4d::grain::segment::SceneGrain;
use crate::nl4d::grain::{GrainExport, GrainGeometry, GrainParts, persistence_partials_len};
use crate::nl4d::kernels::{grain_persist, grain_persist_reduce, grain_project};

const SIZE: u32 = 128;
const PIXELS: usize = (SIZE * SIZE) as usize;
const CELLS_X: u32 = SIZE / CELL;
const CELLS: usize = (CELLS_X * CELLS_X) as usize;
const CELL_PIXELS: usize = (CELL * CELL) as usize;
const FLAT: f32 = 0.5;
const GRAIN_CODES: f64 = 3.0;
const GRAIN: f32 = (GRAIN_CODES / 255.0) as f32;
/// The step a denoised edge adds to the right half of a cell.
const EDGE_STEP: f32 = 10.0 / 255.0;
/// Frames in a scene, long enough that every lag of [MID_ROW] has both sides.
const SCENE_FRAMES: u32 = 20;
const MID_ROW: usize = 9;
/// How far the shifted grain moves each frame, in pixels.
const GRAIN_SHIFT: u32 = 3;
/// Frames a refreshed grain field is held for.
const HOLD_FRAMES: u32 = 3;
const TOLERANCE: f64 = 1e-4;
/// The size of a typical projection, the floor under the projection tolerance.
const PROJECTION_SCALE: f64 = 0.1;
const MOTION_STEP: u32 = 8;
const REPORT_ONLY: GrainParts = GrainParts {
    table: false,
    persistence: true,
};
const SQUARE: Plane = Plane {
    width: SIZE,
    height: SIZE,
};
/// A frame of 2304 cells, so the persist kernel covers several blocks and a partial last one.
const WIDE: Plane = Plane {
    width: 512,
    height: 288,
};
const WEAK_GRAIN_CODES: f64 = 1.0;
/// How far a ramp rises across one cell's width, in codes.
const RAMP_CODES: f32 = 4.0;
/// A denoised luma under the luma gate and a source luma under it but above the clip gate.
const DARK_LUMA: f32 = 10.0 / 255.0;
const DARK_SOURCE: f32 = 12.0 / 255.0;
/// A denoised luma over the luma gate.
const BRIGHT_LUMA: f32 = 245.0 / 255.0;

fn make_client() -> Client {
    crate::tests::compilation_cache::install_test_compilation_cache();
    crate::testing::test_client()
}

fn pattern_sign(projection: usize, x: u32, y: u32) -> f64 {
    let bit = (PATTERNS[projection] >> (y * CELL + x)) & 1;
    if bit == 1 { 1.0 } else { -1.0 }
}

/// Runs the project kernel's maths on the host, one history entry per cell.
fn mirror_project(width: u32, height: u32, source: &[f32], denoised: &[f32]) -> Vec<f32> {
    let cells_x = width / CELL;
    let cells_y = height / CELL;
    let mut history = Vec::with_capacity((cells_x * cells_y) as usize * HISTORY_ENTRY_LEN);

    for cell_y in 0..cells_y {
        for cell_x in 0..cells_x {
            let mut projections = [0.0f64; PROJECTIONS];
            let mut clean_sum = 0.0f64;
            let mut clean_low = f32::INFINITY;
            let mut clean_high = f32::NEG_INFINITY;
            let mut source_low = f32::INFINITY;
            let mut source_high = f32::NEG_INFINITY;

            for local_y in 0..CELL {
                for local_x in 0..CELL {
                    let x = cell_x * CELL + local_x;
                    let y = cell_y * CELL + local_y;
                    let index = (y * width + x) as usize;
                    let luma = source[index];
                    let clean = denoised[index];

                    for (projection, sum) in projections.iter_mut().enumerate() {
                        let sign = pattern_sign(projection, local_x, local_y);
                        *sum += sign * luma as f64;
                    }

                    clean_sum += clean as f64;
                    clean_low = clean_low.min(clean);
                    clean_high = clean_high.max(clean);
                    source_low = source_low.min(luma);
                    source_high = source_high.max(luma);
                }
            }

            let mean = (clean_sum / CELL_PIXELS as f64) as f32;
            let flat = clean_high - clean_low < FLAT_RANGE
                && mean > LUMA_LOW
                && mean < LUMA_HIGH
                && source_low >= CLIP_LOW
                && source_high <= CLIP_HIGH;

            for sum in projections {
                history.push(sum as f32);
            }

            let flag = if flat { 1.0 } else { 0.0 };
            history.push(flag);
        }
    }

    history
}

/// Runs the persist kernel's maths on the host for one pair of history slots.
fn mirror_persist(now: &[f32], earlier: &[f32]) -> [f64; LAG_RECORD_LEN] {
    let auto_scale = 1.0 / (2 * PROJECTIONS * CELL_PIXELS) as f64;
    let cross_scale = 1.0 / (PROJECTIONS * CELL_PIXELS) as f64;
    let mut sums = [0.0f64; LAG_RECORD_LEN];

    let (now_entries, _) = now.as_chunks::<HISTORY_ENTRY_LEN>();
    let (earlier_entries, _) = earlier.as_chunks::<HISTORY_ENTRY_LEN>();
    for (now_entry, earlier_entry) in now_entries.iter().zip(earlier_entries) {
        let both_flat = now_entry[PROJECTIONS] > 0.5 && earlier_entry[PROJECTIONS] > 0.5;
        if !both_flat {
            continue;
        }

        let mut squares = 0.0f64;
        let mut products = 0.0f64;
        for (&now_value, &earlier_value) in now_entry.iter().zip(earlier_entry).take(PROJECTIONS) {
            let now_value = now_value as f64;
            let earlier_value = earlier_value as f64;
            squares += now_value * now_value + earlier_value * earlier_value;
            products += now_value * earlier_value;
        }

        sums[0] += 1.0;
        sums[1] += squares * auto_scale;
        sums[2] += products * cross_scale;
    }

    sums
}

#[derive(Clone, Copy)]
struct Plane {
    width: u32,
    height: u32,
}

impl Plane {
    fn pixels(&self) -> usize {
        (self.width * self.height) as usize
    }

    fn cells_x(&self) -> u32 {
        self.width / CELL
    }

    fn cells(&self) -> usize {
        (self.cells_x() * (self.height / CELL)) as usize
    }

    fn history_len(&self) -> usize {
        HISTORY_FRAMES as usize * self.cells() * HISTORY_ENTRY_LEN
    }
}

fn unit_grain(seed: u64) -> Vec<f32> {
    unit_grain_on(SQUARE, seed)
}

/// Unit Gaussian samples over one frame.
fn unit_grain_on(plane: Plane, seed: u64) -> Vec<f32> {
    let field = gaussian_field(plane.width as usize, plane.height as usize, seed);
    field.iter().map(|&sample| sample as f32).collect()
}

/// A flat frame the size of the first grain field, with each field added at its std.
fn frame_of(layers: &[(&[f32], f32)]) -> Vec<f32> {
    let pixels = layers[0].0.len();
    let mut frame = vec![FLAT; pixels];
    for &(field, sigma) in layers {
        for (value, &sample) in frame.iter_mut().zip(field) {
            *value += sigma * sample;
        }
    }

    frame
}

fn fresh_frames(count: u32) -> Vec<Vec<f32>> {
    (0..count)
        .map(|index| {
            let grain = unit_grain(1000 + index as u64);
            frame_of(&[(&grain, GRAIN)])
        })
        .collect()
}

fn static_frames(count: u32) -> Vec<Vec<f32>> {
    let grain = unit_grain(7);
    let frame = frame_of(&[(&grain, GRAIN)]);

    vec![frame; count as usize]
}

fn held_frames(count: u32) -> Vec<Vec<f32>> {
    (0..count)
        .map(|index| {
            let grain = unit_grain(2000 + (index / HOLD_FRAMES) as u64);
            frame_of(&[(&grain, GRAIN)])
        })
        .collect()
}

fn half_fixed_frames(count: u32) -> Vec<Vec<f32>> {
    half_fixed_frames_on(SQUARE, count)
}

/// A fixed field and a fresh field of equal variance, `GRAIN` std in all.
fn half_fixed_frames_on(plane: Plane, count: u32) -> Vec<Vec<f32>> {
    let half_sigma = GRAIN * std::f32::consts::FRAC_1_SQRT_2;
    let fixed = unit_grain_on(plane, 11);

    (0..count)
        .map(|index| {
            let fresh = unit_grain_on(plane, 3000 + index as u64);
            frame_of(&[(&fixed, half_sigma), (&fresh, half_sigma)])
        })
        .collect()
}

/// One grain field moving right by [GRAIN_SHIFT] pixels each frame.
fn shifted_frames(count: u32) -> Vec<Vec<f32>> {
    let field_width = SIZE + GRAIN_SHIFT * count;
    let field = gaussian_field(field_width as usize, SIZE as usize, 13);

    (0..count)
        .map(|index| {
            let offset = index * GRAIN_SHIFT;
            let mut frame = Vec::with_capacity(PIXELS);
            for y in 0..SIZE {
                let start = (y * field_width + offset) as usize;
                let row = &field[start..start + SIZE as usize];
                let values = row.iter().map(|&sample| FLAT + GRAIN * sample as f32);
                frame.extend(values);
            }

            frame
        })
        .collect()
}

/// A horizontal ramp around [FLAT] rising [RAMP_CODES] across each cell's width.
fn ramp() -> Vec<f32> {
    let slope = RAMP_CODES / ((CELL - 1) as f32 * 255.0);
    let centre = (SIZE / 2) as f32;
    let mut values = Vec::with_capacity(PIXELS);
    for _ in 0..SIZE {
        for x in 0..SIZE {
            let value = FLAT + (x as f32 - centre) * slope;
            values.push(value);
        }
    }

    values
}

/// Fresh weak grain each frame over the same ramp.
fn ramp_frames(count: u32) -> Vec<Vec<f32>> {
    let base = ramp();
    let sigma = (WEAK_GRAIN_CODES / 255.0) as f32;

    (0..count)
        .map(|index| {
            let grain = unit_grain(4000 + index as u64);
            let values = base.iter().zip(&grain);
            values.map(|(&value, &sample)| value + sigma * sample).collect()
        })
        .collect()
}

fn flat_output() -> Vec<f32> {
    vec![FLAT; PIXELS]
}

/// Fills every pixel of each listed cell with `value`.
fn with_cells_filled(frame: &[f32], cells: &[(u32, u32)], value: f32) -> Vec<f32> {
    let mut changed = frame.to_vec();
    for &(cell_x, cell_y) in cells {
        for y in cell_y * CELL..(cell_y + 1) * CELL {
            for x in cell_x * CELL..(cell_x + 1) * CELL {
                changed[(y * SIZE + x) as usize] = value;
            }
        }
    }

    changed
}

fn edged_output(cells: &[(u32, u32)]) -> Vec<f32> {
    edged_output_on(SQUARE, cells)
}

/// A flat output with a step across the right half of each listed cell.
fn edged_output_on(plane: Plane, cells: &[(u32, u32)]) -> Vec<f32> {
    let mut output = vec![FLAT; plane.pixels()];
    for &(cell_x, cell_y) in cells {
        for y in cell_y * CELL..(cell_y + 1) * CELL {
            for x in cell_x * CELL + CELL / 2..(cell_x + 1) * CELL {
                output[(y * plane.width + x) as usize] += EDGE_STEP;
            }
        }
    }

    output
}

/// Sets the first pixel of each listed cell to `value`.
fn with_pixels_set(frame: &[f32], cells: &[(u32, u32)], value: f32) -> Vec<f32> {
    let mut changed = frame.to_vec();
    for &(cell_x, cell_y) in cells {
        let index = (cell_y * CELL * SIZE + cell_x * CELL) as usize;
        changed[index] = value;
    }

    changed
}

fn quantised(frames: &[Vec<f32>], max_code: f32) -> Vec<Vec<f32>> {
    frames
        .iter()
        .map(|frame| {
            frame
                .iter()
                .map(|&value| (value * max_code).round() / max_code)
                .collect()
        })
        .collect()
}

fn geometry() -> GrainGeometry {
    GrainGeometry {
        width: SIZE,
        height: SIZE,
        channels: 1,
        blocks_x: SIZE / MOTION_STEP,
        blocks_y: SIZE / MOTION_STEP,
        step: MOTION_STEP,
        ring_frames: 1,
    }
}

fn upload(client: &Client, values: &[f32]) -> Handle {
    let bytes = f32::as_bytes(values);
    client.create_from_slice(bytes)
}

fn measure_frame(client: &Client, export: &mut GrainExport, source: &[f32], denoised: &[f32]) {
    let input = upload(client, source);
    let output = upload(client, denoised);
    let outputs = [output.clone(), output];

    export.measure(client, &input, &outputs, 0, None, 0);
}

/// Measures one stream, each frame against its own denoised output.
fn measure_stream(client: &Client, export: &mut GrainExport, frames: &[Vec<f32>], outputs: &[Vec<f32>]) {
    for (source, denoised) in frames.iter().zip(outputs) {
        measure_frame(client, export, source, denoised);
    }
}

/// Measures one stream against flat outputs and drains its chunks.
fn chunks_of(frames: &[Vec<f32>]) -> Vec<GrainChunk> {
    let flat = flat_output();
    let outputs = vec![flat; frames.len()];

    chunks_with_outputs(frames, &outputs)
}

fn chunks_with_outputs(frames: &[Vec<f32>], outputs: &[Vec<f32>]) -> Vec<GrainChunk> {
    let client = make_client();
    let geometry = geometry();
    let mut export = GrainExport::new(&client, geometry, REPORT_ONLY);

    measure_stream(&client, &mut export, frames, outputs);

    export.drain(&client).expect("drain")
}

fn rows_of(frames: &[Vec<f32>]) -> Vec<FrameGrain> {
    let chunks = chunks_of(frames);

    rows_from(chunks)
}

fn rows_from(chunks: Vec<GrainChunk>) -> Vec<FrameGrain> {
    let scene = SceneGrain {
        first_frame: 0,
        chunks,
    };

    frame_rows(&[scene])
}

fn all_records(chunks: &[GrainChunk]) -> Vec<f64> {
    chunks
        .iter()
        .flat_map(|chunk| chunk.frame_records.iter().copied())
        .collect()
}

/// The `[cells, auto, cross]` sums of one frame's record at a lag from 1.
fn lag_sums(records: &[f64], frame: usize, lag: usize) -> [f64; LAG_RECORD_LEN] {
    let start = frame * FRAME_RECORD_LEN + (lag - 1) * LAG_RECORD_LEN;
    let sums = &records[start..start + LAG_RECORD_LEN];

    [sums[0], sums[1], sums[2]]
}

/// Each frame's lag 1 cell count.
fn lag_one_cells(records: &[f64]) -> Vec<f64> {
    let (frame_records, _) = records.as_chunks::<FRAME_RECORD_LEN>();

    frame_records.iter().map(|record| record[0]).collect()
}

fn persistence_of(row: &FrameGrain) -> [f64; PERSISTENCE_LAGS] {
    row.persistence
        .map(|reading| reading.expect("every lag should have a reading"))
}

/// The mean persistence of each lag over every row that reads it.
fn mean_persistence(rows: &[FrameGrain]) -> [f64; PERSISTENCE_LAGS] {
    let mut means = [0.0; PERSISTENCE_LAGS];
    for (lag_index, mean) in means.iter_mut().enumerate() {
        let readings: Vec<f64> = rows.iter().filter_map(|row| row.persistence[lag_index]).collect();
        assert!(!readings.is_empty());

        let total: f64 = readings.iter().sum();
        *mean = total / readings.len() as f64;
    }

    means
}

fn assert_close(actual: f64, expected: f64, scale: f64) {
    let limit = TOLERANCE * expected.abs().max(scale);
    assert!(
        (actual - expected).abs() <= limit,
        "{actual} is not within {limit} of {expected}"
    );
}

fn history_handle(client: &Client, plane: Plane) -> Handle {
    let zeros = vec![0.0f32; plane.history_len()];

    upload(client, &zeros)
}

fn launch_project(
    client: &Client,
    history: &Handle,
    plane: Plane,
    source: &[f32],
    denoised: &[f32],
    slot: u32,
) {
    let input = upload(client, source);
    let output = upload(client, denoised);
    let words = pattern_words();
    let patterns_bytes = u32::as_bytes(&words);
    let patterns = client.create_from_slice(patterns_bytes);
    let pixels = plane.pixels();

    unsafe {
        grain_project::launch_unchecked(
            client,
            CubeCount::new_2d(plane.cells_x(), plane.height / CELL),
            CubeDim::new_2d(CELL, CELL),
            1usize,
            BufferArg::from_raw_parts(input, pixels),
            BufferArg::from_raw_parts(output, pixels),
            BufferArg::from_raw_parts(patterns, words.len()),
            BufferArg::from_raw_parts(history.clone(), plane.history_len()),
            0u32,
            slot,
            plane.width,
            plane.height,
            1u32,
        );
    }
}

fn read_floats(client: &Client, handle: Handle) -> Vec<f32> {
    let bytes = client.read_one(handle).expect("readback");
    f32::from_bytes(&bytes).to_vec()
}

fn slot_values(history: &[f32], slot: u32) -> &[f32] {
    let slot_len = CELLS * HISTORY_ENTRY_LEN;
    let start = slot as usize * slot_len;

    &history[start..start + slot_len]
}

#[test]
fn every_pattern_is_balanced_ramp_free_and_distinct() {
    let full = u64::MAX;
    for (index, &pattern) in PATTERNS.iter().enumerate() {
        let mut x_moment = 0.0;
        let mut y_moment = 0.0;
        for y in 0..CELL {
            for x in 0..CELL {
                let sign = pattern_sign(index, x, y);
                x_moment += sign * x as f64;
                y_moment += sign * y as f64;
            }
        }

        assert_eq!(pattern.count_ones(), 32, "pattern {index}");
        assert_eq!(x_moment, 0.0, "pattern {index}");
        assert_eq!(y_moment, 0.0, "pattern {index}");

        for &other in &PATTERNS[index + 1..] {
            assert_ne!(pattern, other);
            assert_ne!(pattern, other ^ full);
        }
    }
}

#[test]
fn projections_and_flags_match_the_mirror() {
    let client = make_client();
    let grain = unit_grain(21);
    let frame = frame_of(&[(&grain, GRAIN)]);
    let high_cells = [(2, 3), (9, 9)];
    let low_cells = [(4, 4)];
    let edged_cells = [(0, 0), (5, 7), (15, 15)];
    let clipped_frame = with_pixels_set(&frame, &high_cells, 1.0);
    let source = with_pixels_set(&clipped_frame, &low_cells, 0.0);
    let denoised = edged_output(&edged_cells);
    let left_out = high_cells.len() + low_cells.len() + edged_cells.len();
    let slot = 4;
    let history = history_handle(&client, SQUARE);

    launch_project(&client, &history, SQUARE, &source, &denoised, slot);

    let gpu_history = read_floats(&client, history);
    let gpu_slot = slot_values(&gpu_history, slot);
    let expected = mirror_project(SIZE, SIZE, &source, &denoised);
    let (expected_entries, _) = expected.as_chunks::<HISTORY_ENTRY_LEN>();
    let flagged = expected_entries
        .iter()
        .filter(|entry| entry[PROJECTIONS] == 1.0)
        .count();

    assert_eq!(flagged, CELLS - left_out);
    for (&actual, &wanted) in gpu_slot.iter().zip(&expected) {
        assert_close(actual as f64, wanted as f64, PROJECTION_SCALE);
    }
}

#[test]
fn the_luma_gate_reads_the_denoised_output() {
    let client = make_client();
    let dark_cells = [(1, 1), (8, 3)];
    let bright_cells = [(5, 12)];
    let dark_source_cell = (10, 10);
    let grain = unit_grain(23);
    let frame = frame_of(&[(&grain, GRAIN)]);
    let source = with_cells_filled(&frame, &[dark_source_cell], DARK_SOURCE);
    let flat = flat_output();
    let dark_output = with_cells_filled(&flat, &dark_cells, DARK_LUMA);
    let denoised = with_cells_filled(&dark_output, &bright_cells, BRIGHT_LUMA);
    let slot = 0;
    let history = history_handle(&client, SQUARE);

    launch_project(&client, &history, SQUARE, &source, &denoised, slot);

    let gpu_history = read_floats(&client, history);
    let gpu_slot = slot_values(&gpu_history, slot);
    let (gpu_entries, _) = gpu_slot.as_chunks::<HISTORY_ENTRY_LEN>();
    let expected = mirror_project(SIZE, SIZE, &source, &denoised);
    let (expected_entries, _) = expected.as_chunks::<HISTORY_ENTRY_LEN>();
    let flag_at =
        |(cell_x, cell_y): (u32, u32)| gpu_entries[(cell_y * CELLS_X + cell_x) as usize][PROJECTIONS];
    let flagged = gpu_entries
        .iter()
        .filter(|entry| entry[PROJECTIONS] == 1.0)
        .count();
    let left_out = dark_cells.len() + bright_cells.len();

    assert_eq!(flagged, CELLS - left_out);
    assert!(dark_cells.iter().all(|&cell| flag_at(cell) == 0.0));
    assert!(bright_cells.iter().all(|&cell| flag_at(cell) == 0.0));
    assert_eq!(flag_at(dark_source_cell), 1.0);
    for (gpu_entry, expected_entry) in gpu_entries.iter().zip(expected_entries) {
        assert_eq!(gpu_entry[PROJECTIONS], expected_entry[PROJECTIONS]);
    }
}

#[test]
fn sums_match_the_mirror_at_every_lag() {
    let client = make_client();
    let frames = half_fixed_frames_on(WIDE, HISTORY_FRAMES);
    let history = history_handle(&client, WIDE);
    let first_slot = 5;
    let mut expected_slots = Vec::with_capacity(frames.len());

    for (index, frame) in frames.iter().enumerate() {
        let edged_cells = [(index as u32, 2), (3, index as u32)];
        let denoised = edged_output_on(WIDE, &edged_cells);
        let slot = (first_slot + index as u32) % HISTORY_FRAMES;
        launch_project(&client, &history, WIDE, frame, &denoised, slot);

        let expected = mirror_project(WIDE.width, WIDE.height, frame, &denoised);
        expected_slots.push(expected);
    }

    // The sums land in the third of three frame records.
    let record_frame = 2;
    let now_slot = (first_slot + HISTORY_FRAMES - 1) % HISTORY_FRAMES;
    let record_offset = (record_frame * FRAME_RECORD_LEN) as u32;
    let record_len = (record_frame + 1) * FRAME_RECORD_LEN;
    let zeros = vec![0.0f32; record_len];
    let record = upload(&client, &zeros);
    let cells = WIDE.cells() as u32;
    let blocks = cells.div_ceil(PERSIST_BLOCK_CELLS);
    let partials_len = persistence_partials_len(WIDE.width, WIDE.height) as usize;
    let partials = client.empty(partials_len * size_of::<f32>());
    let lags = PERSISTENCE_LAGS as u32;

    unsafe {
        grain_persist::launch_unchecked(
            &client,
            CubeCount::new_2d(blocks, lags),
            CubeDim::new_1d(PERSIST_THREADS),
            BufferArg::from_raw_parts(history, WIDE.history_len()),
            BufferArg::from_raw_parts(partials.clone(), partials_len),
            now_slot,
            cells,
            blocks,
        );

        grain_persist_reduce::launch_unchecked(
            &client,
            CubeCount::new_1d(lags),
            CubeDim::new_1d(REDUCE_THREADS),
            BufferArg::from_raw_parts(partials, partials_len),
            BufferArg::from_raw_parts(record.clone(), record_len),
            record_offset,
            blocks,
        );
    }

    let gpu_record = read_floats(&client, record);
    let records: Vec<f64> = gpu_record.iter().map(|&value| value as f64).collect();
    let newest = expected_slots.last().expect("nine frames");

    for lag in 1..=PERSISTENCE_LAGS {
        let earlier = &expected_slots[expected_slots.len() - 1 - lag];
        let wanted = mirror_persist(newest, earlier);
        let [cells, auto_sum, cross_sum] = lag_sums(&records, record_frame, lag);

        assert!(wanted[0] > 0.0);
        assert_eq!(cells, wanted[0], "lag {lag}");
        assert_close(auto_sum, wanted[1], wanted[1]);
        assert_close(cross_sum, wanted[2], wanted[1]);
    }

    let earlier_records = &records[..record_offset as usize];
    let untouched = earlier_records.iter().all(|&value| value == 0.0);
    assert!(untouched);
}

#[test]
fn fresh_grain_on_every_frame_reads_as_moving() {
    let frames = fresh_frames(SCENE_FRAMES);

    let rows = rows_of(&frames);

    let row = &rows[MID_ROW];
    let persistence = persistence_of(row);
    let strength = row.strength.expect("a strength reading");
    assert!(persistence.iter().all(|&share| share < 0.05), "{persistence:?}");
    assert!((strength / GRAIN_CODES - 1.0).abs() < 0.05, "strength {strength}");
}

#[test]
fn the_same_grain_on_every_frame_reads_as_static() {
    let frames = static_frames(SCENE_FRAMES);

    let rows = rows_of(&frames);

    let persistence = persistence_of(&rows[MID_ROW]);
    assert!(persistence.iter().all(|&share| share > 0.95), "{persistence:?}");
}

#[test]
fn grain_refreshed_every_three_frames_reads_as_slow() {
    let frames = held_frames(SCENE_FRAMES);

    let rows = rows_of(&frames);

    let means = mean_persistence(&rows);
    assert!((0.57..0.77).contains(&means[0]), "{means:?}");
    assert!((0.23..0.43).contains(&means[1]), "{means:?}");
    assert!(means[2..].iter().all(|&share| share < 0.05), "{means:?}");
}

#[test]
fn half_fixed_half_fresh_variance_reads_near_one_half() {
    let frames = half_fixed_frames(SCENE_FRAMES);

    let rows = rows_of(&frames);

    let persistence = persistence_of(&rows[MID_ROW]);
    let near_half = persistence.iter().all(|&share| (0.4..=0.6).contains(&share));
    assert!(near_half, "{persistence:?}");
}

#[test]
fn the_same_grain_shifted_by_three_pixels_reads_as_moving() {
    let frames = shifted_frames(SCENE_FRAMES);

    let rows = rows_of(&frames);

    let persistence = persistence_of(&rows[MID_ROW]);
    assert!(persistence[0] < 0.1, "{persistence:?}");
}

#[test]
fn a_static_ramp_under_grain_does_not_read_as_persistent() {
    let frames = ramp_frames(SCENE_FRAMES);
    let ramp_output = ramp();
    let outputs = vec![ramp_output; frames.len()];

    let chunks = chunks_with_outputs(&frames, &outputs);

    let rows = rows_from(chunks);
    let row = &rows[MID_ROW];
    let persistence = persistence_of(row);
    let strength = row.strength.expect("a strength reading");
    assert_eq!(row.cells as usize, 2 * CELLS);
    assert!(persistence[0] < 0.05, "{persistence:?}");
    assert!(
        (strength / WEAK_GRAIN_CODES - 1.0).abs() < 0.1,
        "strength {strength}"
    );
}

#[test]
fn a_cell_with_an_edge_in_either_denoised_frame_is_left_out() {
    let frames = static_frames(8);
    let first_edges = [(1, 1), (6, 2), (10, 12)];
    let second_edges = [(3, 3), (14, 0)];
    let flat = flat_output();
    let mut outputs = vec![flat; frames.len()];
    outputs[4] = edged_output(&first_edges);
    outputs[5] = edged_output(&second_edges);

    let chunks = chunks_with_outputs(&frames, &outputs);

    let records = all_records(&chunks);
    let lag_one_cells = lag_one_cells(&records);
    let all_cells = CELLS as f64;
    let first_count = first_edges.len() as f64;
    let second_count = second_edges.len() as f64;
    assert_eq!(lag_one_cells[3], all_cells);
    assert_eq!(lag_one_cells[4], all_cells - first_count);
    assert_eq!(lag_one_cells[5], all_cells - first_count - second_count);
    assert_eq!(lag_one_cells[6], all_cells - second_count);
    assert_eq!(lag_one_cells[7], all_cells);
}

#[test]
fn clipped_source_pixels_leave_the_cell_out() {
    let mut frames = static_frames(6);
    let high_cells = [(2, 2), (7, 11)];
    let low_cells = [(12, 4)];
    let high_frame = with_pixels_set(&frames[3], &high_cells, 1.0);
    frames[3] = with_pixels_set(&high_frame, &low_cells, 0.0);

    let chunks = chunks_of(&frames);

    let records = all_records(&chunks);
    let lag_one_cells = lag_one_cells(&records);
    let all_cells = CELLS as f64;
    let clipped = (high_cells.len() + low_cells.len()) as f64;
    assert_eq!(lag_one_cells[2], all_cells);
    assert_eq!(lag_one_cells[3], all_cells - clipped);
    assert_eq!(lag_one_cells[4], all_cells - clipped);
    assert_eq!(lag_one_cells[5], all_cells);
}

#[test]
fn eight_and_ten_bit_inputs_give_the_same_sums() {
    let frames = half_fixed_frames(10);
    let eight_bit = quantised(&frames, 255.0);
    let ten_bit = quantised(&frames, 1023.0);

    let eight_chunks = chunks_of(&eight_bit);
    let ten_chunks = chunks_of(&ten_bit);

    let eight_records = all_records(&eight_chunks);
    let ten_records = all_records(&ten_chunks);
    let last_frame = frames.len() - 1;
    for lag in 1..=PERSISTENCE_LAGS {
        let [eight_cells, eight_auto, eight_cross] = lag_sums(&eight_records, last_frame, lag);
        let [ten_cells, ten_auto, ten_cross] = lag_sums(&ten_records, last_frame, lag);

        assert_eq!(eight_cells, ten_cells);
        assert!((eight_auto / ten_auto - 1.0).abs() < 0.02, "lag {lag}");
        assert!((eight_cross - ten_cross).abs() < 0.02 * ten_auto, "lag {lag}");
    }
}

#[test]
fn the_first_frame_of_a_stream_records_nothing() {
    let frames = static_frames(3);

    let chunks = chunks_of(&frames);

    let first_record = &chunks[0].frame_records[..FRAME_RECORD_LEN];
    assert!(first_record.iter().all(|&value| value == 0.0));
}

#[test]
fn each_lag_starts_once_the_stream_is_long_enough() {
    let frames = static_frames(PERSISTENCE_LAGS as u32 + 2);

    let chunks = chunks_of(&frames);

    let records = all_records(&chunks);
    for lag in 1..=PERSISTENCE_LAGS {
        for frame in 0..lag {
            let sums = lag_sums(&records, frame, lag);
            assert_eq!(sums, [0.0; LAG_RECORD_LEN], "lag {lag}");
        }

        let [cells, auto_sum, cross_sum] = lag_sums(&records, lag, lag);
        assert_eq!(cells, CELLS as f64, "lag {lag}");
        assert!(auto_sum > 0.0);
        assert!((cross_sum / auto_sum - 1.0).abs() < 1e-3);
    }
}

#[test]
fn a_new_stream_never_compares_with_the_old_one() {
    let client = make_client();
    let geometry = geometry();
    let mut export = GrainExport::new(&client, geometry, REPORT_ONLY);
    let first_stream = static_frames(5);
    let second_stream = static_frames(3);
    let flat = flat_output();
    let outputs = vec![flat; first_stream.len()];

    measure_stream(&client, &mut export, &first_stream, &outputs);
    export.reset_stream();
    measure_stream(&client, &mut export, &second_stream, &outputs);

    let chunks = export.drain(&client).expect("drain");
    assert_eq!(chunks.len(), 2);

    let second_records = &chunks[1].frame_records;
    let first_record = &second_records[..FRAME_RECORD_LEN];
    let [lag_two_cells, _, _] = lag_sums(second_records, 2, 2);
    let lag_three = lag_sums(second_records, 2, 3);
    assert!(first_record.iter().all(|&value| value == 0.0));
    assert_eq!(lag_two_cells, CELLS as f64);
    assert_eq!(lag_three, [0.0; LAG_RECORD_LEN]);
}

#[test]
fn records_land_at_each_frames_offset_across_a_chunk_boundary() {
    let frames = static_frames(CHUNK_FRAMES + 2);

    let chunks = chunks_of(&frames);

    let lengths: Vec<usize> = chunks.iter().map(|chunk| chunk.frame_records.len()).collect();
    let chunk_records_len = CHUNK_FRAMES as usize * FRAME_RECORD_LEN;
    assert_eq!(lengths, vec![chunk_records_len, 2 * FRAME_RECORD_LEN]);

    let records = all_records(&chunks);
    for frame in 0..frames.len() {
        for lag in 1..=PERSISTENCE_LAGS {
            let [cells, _, _] = lag_sums(&records, frame, lag);
            let wanted = if lag <= frame { CELLS as f64 } else { 0.0 };
            assert_eq!(cells, wanted, "frame {frame} lag {lag}");
        }
    }
}

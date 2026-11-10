#![cfg(any(feature = "vulkan", feature = "metal"))]

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::mirror::{StillStream, mirror_still_chunks};
use super::synthetic::{ar_field, gaussian_field, left_weights};
use crate::bench_api::HostIo;
use crate::nl4d::denoiser::buffer_sizes;
use crate::nl4d::grain::chunk::{GrainChunk, TableSums};
use crate::nl4d::grain::consts::{
    AUTOCOV_LEN,
    CHROMA_RECORD_LEN,
    CHUNK_FRAMES,
    FRAME_RECORD_LEN,
    FRAME_TABLE_LEN,
    HIST_LEN,
    LAG_COUNT,
    LUMA_BINS,
    STD_BUCKETS,
    STRENGTH_GROUPS,
};
use crate::nl4d::grain::fit::{bucket_edges, hist_median};
use crate::nl4d::grain::segment::SceneGrain;
use crate::nl4d::grain::{FitOptions, GrainExport, GrainGeometry, GrainParts, MeasuredPlanes, build_table};
use crate::nl4d::snapshot::LastFields;
use crate::nl4d::{Nl4dDenoiser, Nl4dParams};
use crate::nlmeans::{
    ChannelMode,
    HqParams,
    MotionCompensationMode,
    MotionEstimation,
    NlmParams,
    PrefilterMode,
};

const WIDTH: u32 = 96;
const HEIGHT: u32 = 64;
const PAN_WIDTH: u32 = 192;
const PAN_HEIGHT: u32 = 128;
/// How far the panning content moves each frame, in whole pixels.
const PAN_STEP: (i32, i32) = (3, 2);
/// Side length of each flat tile in the panning content.
///
/// Small enough that a vector off by a frame's motion pairs most flat cells across a tile edge.
const PAN_TILE: i32 = 12;
const SIGMA: f32 = 3.0 / 255.0;
const FLICKER: f32 = 2.0 / 255.0;
/// The lane of lag `(0, 3)` in a record.
const LAG_RIGHT_3: usize = 3;
const SAVED_VECTORS: &str = "grain saved vectors";
const PERSISTENCE_HISTORY: &str = "grain persistence history";
const PERSISTENCE_PARTIALS: &str = "grain persistence partials";
/// The motion block size of a stream fed straight to a [GrainExport].
const DIRECT_STEP: u32 = 8;
/// Ring slots a stream fed straight to a [GrainExport] cycles through.
const DIRECT_RING_FRAMES: u32 = 3;
const LUMA_PASS: GrainParts = GrainParts {
    planes: MeasuredPlanes::Luma,
    noise: false,
};
const BANDED_SIZE: u32 = 128;
/// The height of each luma band, four cell rows.
const BAND_HEIGHT: u32 = 32;
const BAND_LEVELS: [f32; 4] = [0.3, 0.45, 0.6, 0.75];
/// The scale of the AR source grain.
const TEXTURE_GAIN: f64 = 2.0 / 255.0;
/// The std of the grain left in each denoised output.
const KEPT_GAIN: f32 = 0.4 / 255.0;
const TABLE_FRAMES: u32 = 50;
const FRAME_RATE: (u64, u64) = (24, 1);
/// Frames compared with the host mirror, one full chunk and a partial one.
const MIRRORED_FRAMES: u32 = 30;
/// How far a summed lag sum may sit from the mirror, relative to its group's zero lag.
const SUM_TOLERANCE: f64 = 1e-5;
const FLAT_LEVEL: f32 = 0.5;
const LONE_FRAMES: u32 = 10;
/// The only frame whose pair differs.
const LONE_FRAME: u32 = 3;

fn make_client() -> Client {
    crate::tests::compilation_cache::install_test_compilation_cache();
    crate::testing::test_client()
}

fn params(grain_data: bool) -> Nl4dParams {
    let motion_compensation = MotionCompensationMode::Mvtools {
        blksize: 16,
        overlap: 8,
        search_radius: 4,
        max_search_escalation: 4,
        pyramid_levels: 2,
        estimation: MotionEstimation::Auto,
    };
    let hq = HqParams::with_sigma(6.0 / 255.0);
    let nlm = NlmParams {
        temporal_radius: 2,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation,
        hq: Some(hq),
    };

    Nl4dParams {
        nlm,
        temporal_radius: 2,
        grain_data,
        ..Nl4dParams::default()
    }
}

/// Flat 0.5 frames with fresh grain at `sigma` each frame.
fn grain_frames(sigma: f32, count: u32) -> Vec<Vec<f32>> {
    (0..count)
        .map(|seed| {
            let field = gaussian_field(WIDTH as usize, HEIGHT as usize, seed as u64 + 100);
            field
                .iter()
                .map(|&sample| (0.5 + sigma * sample as f32).clamp(0.0, 1.0))
                .collect()
        })
        .collect()
}

/// Flat [PAN_TILE] tiles of 0.4 and 0.6 moving by [PAN_STEP] each frame, with fresh grain at `sigma`.
fn panning_frames(sigma: f32, count: u32) -> Vec<Vec<f32>> {
    (0..count)
        .map(|frame_index| {
            let offset_x = PAN_STEP.0 * frame_index as i32;
            let offset_y = PAN_STEP.1 * frame_index as i32;
            let field = gaussian_field(PAN_WIDTH as usize, PAN_HEIGHT as usize, frame_index as u64 + 200);
            let mut frame = Vec::with_capacity(field.len());

            for y in 0..PAN_HEIGHT as i32 {
                for x in 0..PAN_WIDTH as i32 {
                    let tile_x = (x - offset_x).div_euclid(PAN_TILE);
                    let tile_y = (y - offset_y).div_euclid(PAN_TILE);
                    let odd_tile = (tile_x + tile_y).rem_euclid(2) == 1;
                    let base = if odd_tile { 0.6 } else { 0.4 };
                    let sample = field[(y * PAN_WIDTH as i32 + x) as usize] as f32;
                    frame.push(base + sigma * sample);
                }
            }

            frame
        })
        .collect()
}

/// Denoises and flushes one stream, appending every output frame to `outputs`.
fn denoise_stream(denoiser: &mut Nl4dDenoiser, frames: &[Vec<f32>], outputs: &mut Vec<Vec<f32>>) {
    for frame in frames {
        denoiser.push_frame(frame);

        if let Some(output) = denoiser.denoise().expect("denoise") {
            outputs.push(output);
        }
    }

    denoiser
        .flush(|frame| {
            let output = frame.to_vec();
            outputs.push(output);
        })
        .expect("flush");
}

fn run(params: Nl4dParams, frames: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<GrainChunk>, bool) {
    run_sized(params, frames, WIDTH, HEIGHT)
}

fn run_sized(
    params: Nl4dParams,
    frames: &[Vec<f32>],
    width: u32,
    height: u32,
) -> (Vec<Vec<f32>>, Vec<GrainChunk>, bool) {
    let client = make_client();
    let mut denoiser = Nl4dDenoiser::new(&client, params, width, height).expect("construction");
    let has_export = denoiser.has_grain_data();
    let mut outputs = Vec::new();

    denoise_stream(&mut denoiser, frames, &mut outputs);

    let chunks = denoiser.drain_grain_chunks().expect("drain");

    (outputs, chunks, has_export)
}

/// The table statistics of every chunk, summed.
fn merged_table(chunks: &[GrainChunk]) -> TableSums {
    let empty = GrainChunk::empty();
    let mut merged = empty.summed_table();
    for chunk in chunks {
        let table = chunk.summed_table();
        merged.add(&table);
    }

    merged
}

/// The median grain std of a histogram, over every luma bin.
fn median_of(hist: &[u32]) -> f64 {
    let mut all_bins = vec![0u32; STD_BUCKETS];
    for bin in 0..LUMA_BINS {
        for bucket in 0..STD_BUCKETS {
            all_bins[bucket] += hist[bin * STD_BUCKETS + bucket];
        }
    }

    let edges = bucket_edges();
    hist_median(&all_bins, &edges).expect("some accepted blocks")
}

/// `R(0, 3) / R(0, 0)` with both lags summed over every strength group.
fn lag_3_ratio(table: &TableSums) -> f64 {
    let mut zero_lag = 0.0;
    let mut lag_3 = 0.0;
    for group in 0..STRENGTH_GROUPS {
        let sums = table.group_autocov(group);
        zero_lag += sums[0];
        lag_3 += sums[LAG_RIGHT_3];
    }

    assert!(zero_lag > 0.0);

    lag_3 / zero_lag
}

fn frames_counted(chunks: &[GrainChunk]) -> u32 {
    chunks.iter().map(|chunk| chunk.frames).sum()
}

#[test]
fn grain_data_off_allocates_and_launches_nothing() {
    let frames = grain_frames(SIGMA, 9);
    let grain_off = params(false);
    let names = buffer_names(&grain_off);
    let (_, chunks, has_export) = run(grain_off, &frames);

    assert!(!has_export);
    assert!(chunks.is_empty());
    assert!(!names.iter().any(|name| name.starts_with("grain")));
}

#[test]
fn export_does_not_change_the_output() {
    let frames = grain_frames(SIGMA, 9);
    let export_off = params(false);
    let export_on = params(true);
    let (off, _, _) = run(export_off, &frames);
    let (on, _, _) = run(export_on, &frames);

    assert_eq!(off, on);
}

#[test]
fn export_measures_the_source_grain() {
    let frames = grain_frames(SIGMA, 9);
    let export_on = params(true);
    let (_, chunks, has_export) = run(export_on, &frames);
    let table = merged_table(&chunks);
    let median = median_of(&table.source_hist);
    let counted = frames_counted(&chunks);
    let any_pixels = table.pixels.iter().any(|&pixels| pixels > 0.0);

    assert!(has_export);
    assert_eq!(counted, 9);
    assert!(
        (median / SIGMA as f64 - 1.0).abs() < 0.1,
        "median {median} vs sigma {SIGMA}"
    );
    assert!(any_pixels);
}

#[test]
fn flicker_keeps_the_strength_and_a_short_texture() {
    let mut frames = grain_frames(SIGMA, 9);
    for frame in frames.iter_mut().skip(1).step_by(2) {
        for sample in frame.iter_mut() {
            *sample += FLICKER;
        }
    }

    let export_on = params(true);
    let (_, chunks, _) = run(export_on, &frames);
    let table = merged_table(&chunks);
    let median = median_of(&table.source_hist);
    let ratio = lag_3_ratio(&table);

    assert!(
        (median / SIGMA as f64 - 1.0).abs() < 0.1,
        "median {median} vs sigma {SIGMA}"
    );
    assert!(ratio < 0.1, "ratio {ratio}");
}

#[test]
fn panning_content_measures_the_source_grain() {
    let frames = panning_frames(SIGMA, 9);
    let export_on = params(true);
    let (outputs, chunks, _) = run_sized(export_on, &frames, PAN_WIDTH, PAN_HEIGHT);
    let table = merged_table(&chunks);
    let accepted: u32 = table.source_hist.iter().sum();
    let median = median_of(&table.source_hist);

    assert_eq!(outputs.len(), 9);
    assert!(accepted > 0);
    assert!(
        (median / SIGMA as f64 - 1.0).abs() < 0.1,
        "median {median} vs sigma {SIGMA}"
    );
}

#[test]
fn kept_grain_is_weaker_than_source_grain() {
    let frames = grain_frames(SIGMA, 9);
    let export_on = params(true);
    let (_, chunks, _) = run(export_on, &frames);
    let table = merged_table(&chunks);
    let kept_total: u32 = table.kept_hist.iter().sum();
    let source_total: u32 = table.source_hist.iter().sum();

    let kept_median = median_of(&table.kept_hist);
    let source_median = median_of(&table.source_hist);

    assert!(kept_total > 0);
    assert!(source_total > 0);
    assert_eq!(table.kept_hist.len(), HIST_LEN);
    assert!(
        kept_median < source_median,
        "kept {kept_median} vs source {source_median}"
    );
}

#[test]
fn short_scene_measures_only_real_pairs() {
    let client = make_client();
    let export_on = params(true);
    let mut denoiser = Nl4dDenoiser::new(&client, export_on, WIDTH, HEIGHT).expect("construction");
    let frames = grain_frames(SIGMA, 3);
    let mut outputs = Vec::new();

    denoise_stream(&mut denoiser, &frames, &mut outputs);

    let chunks = denoiser.drain_grain_chunks().expect("drain");
    let counted = frames_counted(&chunks);

    assert_eq!(outputs.len(), 3);
    assert_eq!(counted, 3);
    assert_eq!(denoiser.grain_measured_with_entry(), 2);
}

/// [grain_frames] with each sample copied into both chroma channels.
fn chroma_grain_frames(count: u32) -> Vec<Vec<f32>> {
    let plane_frames = grain_frames(SIGMA, count);

    plane_frames
        .into_iter()
        .map(|frame| frame.iter().flat_map(|&sample| [sample, sample]).collect())
        .collect()
}

#[test]
fn a_chroma_denoiser_reports_chroma_persistence_without_the_table() {
    let mut chroma = params(true);
    chroma.nlm.channels = ChannelMode::Chroma;
    let names = buffer_names(&chroma);
    let frames = chroma_grain_frames(9);

    let (_, chunks, has_export) = run(chroma, &frames);

    assert!(has_export);
    assert!(!chunks.is_empty());
    assert!(chunks.iter().all(has_no_table_readings));
    for chunk in &chunks {
        let expected_len = chunk.frames as usize * CHROMA_RECORD_LEN;
        assert_eq!(chunk.frame_records.len(), expected_len);
    }

    assert!(!names.contains(&SAVED_VECTORS));
    assert!(names.contains(&PERSISTENCE_HISTORY));
    assert!(names.contains(&PERSISTENCE_PARTIALS));
}

#[test]
fn chunks_close_when_full_and_at_each_stream_end() {
    let client = make_client();
    let export_on = params(true);
    let mut denoiser = Nl4dDenoiser::new(&client, export_on, WIDTH, HEIGHT).expect("construction");
    let long_stream = grain_frames(SIGMA, 30);
    let short_stream = grain_frames(SIGMA, 5);
    let mut outputs = Vec::new();

    denoise_stream(&mut denoiser, &long_stream, &mut outputs);
    denoise_stream(&mut denoiser, &short_stream, &mut outputs);

    let chunks = denoiser.drain_grain_chunks().expect("drain");
    let frames: Vec<u32> = chunks.iter().map(|chunk| chunk.frames).collect();
    let drained_again = denoiser.drain_grain_chunks().expect("drain");

    assert_eq!(frames, vec![CHUNK_FRAMES, 30 - CHUNK_FRAMES, 5]);
    assert!(drained_again.is_empty());
    for chunk in &chunks {
        let expected_len = chunk.frames as usize * FRAME_TABLE_LEN;
        assert_eq!(chunk.frame_tables.len(), expected_len);
    }
}

/// The names of the buffers a denoiser with `params` counts.
fn buffer_names(params: &Nl4dParams) -> Vec<&'static str> {
    let client = make_client();
    let sizes = buffer_sizes(&client, params, WIDTH, HEIGHT, false);

    sizes.into_iter().map(|(name, _)| name).collect()
}

fn has_table_readings(chunk: &GrainChunk) -> bool {
    let table = chunk.summed_table();
    let any_source = table.source_hist.iter().any(|&count| count > 0);
    let any_autocov = table.autocov.iter().any(|&sum| sum != 0.0);
    any_source && any_autocov
}

fn has_no_table_readings(chunk: &GrainChunk) -> bool {
    chunk.frame_tables.is_empty()
}

/// Whether every frame has a record, and frame 1 compared some cells with frame 0.
fn has_persistence_readings(chunk: &GrainChunk) -> bool {
    let full_length = chunk.frames as usize * FRAME_RECORD_LEN;
    if chunk.frames < 2 || chunk.frame_records.len() != full_length {
        return false;
    }

    let second_frame_cells = chunk.frame_records[FRAME_RECORD_LEN];
    second_frame_cells > 0.0
}

#[test]
fn grain_data_measures_the_table_and_persistence_on_luma() {
    let frames = grain_frames(SIGMA, 9);
    let grain_on = params(true);
    let names = buffer_names(&grain_on);

    let (_, chunks, _) = run(grain_on, &frames);

    let counted = frames_counted(&chunks);
    assert_eq!(counted, 9);
    assert!(chunks.iter().all(has_table_readings));
    assert!(chunks.iter().all(has_persistence_readings));
    assert!(names.contains(&SAVED_VECTORS));
    assert!(names.contains(&PERSISTENCE_HISTORY));
    assert!(names.contains(&PERSISTENCE_PARTIALS));
}

fn upload(client: &Client, values: &[f32]) -> Handle {
    let bytes = f32::as_bytes(values);
    client.create_from_slice(bytes)
}

/// Zero vectors at full confidence for every block, as one neighbour.
fn still_fields(client: &Client, blocks: u32) -> LastFields {
    let vectors = vec![0i32; 2 * blocks as usize];
    let confidence = vec![1.0f32; blocks as usize];
    let vector_bytes = i32::as_bytes(&vectors);
    let confidence_bytes = f32::as_bytes(&confidence);
    let mv_field = client.create_from_slice(vector_bytes);
    let confidence = client.create_from_slice(confidence_bytes);

    LastFields {
        mv_field,
        confidence,
        mv_stride: 2 * blocks,
        conf_stride: blocks,
        neighbours: 1,
    }
}

fn ring_slot(index: usize) -> u32 {
    (index % DIRECT_RING_FRAMES as usize) as u32
}

fn write_slot(ring: &mut [f32], slot: u32, frame: &[f32]) {
    let start = slot as usize * frame.len();
    ring[start..start + frame.len()].copy_from_slice(frame);
}

/// Feeds `stream` straight to a luma [GrainExport], as the denoiser does, and drains it.
fn export_stream(stream: &StillStream) -> Vec<GrainChunk> {
    let client = make_client();
    let geometry = GrainGeometry {
        width: stream.width,
        height: stream.height,
        channels: 1,
        blocks_x: stream.blocks_x(),
        blocks_y: stream.blocks_y(),
        step: stream.step,
        ring_frames: DIRECT_RING_FRAMES,
    };
    let blocks = stream.blocks_x() * stream.blocks_y();
    let fields = still_fields(&client, blocks);
    let frame_len = (stream.width * stream.height) as usize;
    let frame_count = stream.sources.len();
    let mut ring = vec![0.0f32; frame_len * DIRECT_RING_FRAMES as usize];
    let mut export = GrainExport::new(&client, geometry, LUMA_PASS);

    for index in 0..frame_count {
        let slot = ring_slot(index);
        let has_next = index + 1 < frame_count;
        let next_slot = has_next.then(|| ring_slot(index + 1));
        let next_neighbour = has_next.then_some(0u32);
        write_slot(&mut ring, slot, &stream.sources[index]);
        if let Some(next) = next_slot {
            write_slot(&mut ring, next, &stream.sources[index + 1]);
        }

        let input = upload(&client, &ring);
        let output_slot = index % 2;
        let previous = index.saturating_sub(1);
        let current_output = upload(&client, &stream.outputs[index]);
        let previous_output = upload(&client, &stream.outputs[previous]);
        let outputs = if output_slot == 0 {
            [current_output, previous_output]
        } else {
            [previous_output, current_output]
        };

        export.save_vectors(&client, &fields, slot, next_neighbour);
        export.measure(&client, &input, &outputs, slot, next_slot, output_slot);
    }

    export.drain(&client).expect("drain")
}

fn band_level(y: u32) -> f32 {
    BAND_LEVELS[(y / BAND_HEIGHT) as usize]
}

/// Four bands of luma under fresh AR grain each frame, with weak grain left in each output.
fn banded_stream(frame_count: u32) -> StillStream {
    let size = BANDED_SIZE as usize;
    let weights = left_weights();
    let mut sources = Vec::with_capacity(frame_count as usize);
    let mut outputs = Vec::with_capacity(frame_count as usize);

    for frame_index in 0..frame_count {
        let texture = ar_field(&weights, size, size, frame_index as u64 + 300);
        let kept = gaussian_field(size, size, frame_index as u64 + 400);
        let mut source = Vec::with_capacity(size * size);
        let mut output = Vec::with_capacity(size * size);

        for y in 0..BANDED_SIZE {
            let level = band_level(y);
            for x in 0..size {
                let index = y as usize * size + x;
                let source_grain = (TEXTURE_GAIN * texture[index]) as f32;
                let kept_grain = KEPT_GAIN * kept[index] as f32;
                source.push(level + source_grain);
                output.push(level + kept_grain);
            }
        }

        sources.push(source);
        outputs.push(output);
    }

    StillStream {
        width: BANDED_SIZE,
        height: BANDED_SIZE,
        step: DIRECT_STEP,
        sources,
        outputs,
    }
}

/// The table [banded_stream] gives over [TABLE_FRAMES] frames.
const BANDED_TABLE: &str = "filmgrn1
E 0 20625000 1 7391 1
\tp 3 8 0 11 0 1 128 192 256 128 192 256
\tsY 5  72 127 120 127 152 127 184 125 200 127
\tsCb 2 0 0 255 0
\tsCr 2 0 0 255 0
\tcY 0 0 -1 0 0 0 -1 1 -1 0 0 1 0 0 0 1 0 1 0 -1 0 -1 1 102
\tcCb 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128
\tcCr 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128
";

/// Asserts `actual` matches `expected` within [SUM_TOLERANCE] of `scale`.
fn assert_sum_close(actual: f64, expected: f64, scale: f64, label: &str) {
    let difference = (actual - expected).abs();
    let tolerance = SUM_TOLERANCE * scale.max(1.0);

    assert!(difference <= tolerance, "{label}: {actual} vs {expected}");
}

#[test]
fn frame_tables_sum_to_the_chunk_measurement() {
    let stream = banded_stream(MIRRORED_FRAMES);
    let edges = bucket_edges();
    let chunks = export_stream(&stream);
    let mirrored = mirror_still_chunks(&stream, &edges);

    assert_eq!(chunks.len(), mirrored.len());
    for (chunk, mirror) in chunks.iter().zip(&mirrored) {
        let summed = chunk.summed_table();
        let (source_hist, kept_hist) = mirror.hist.split_at(HIST_LEN);
        let (records, _) = mirror.autocov.as_chunks::<AUTOCOV_LEN>();
        let frame_count = chunk.frames as usize;

        assert_eq!(chunk.frame_tables.len(), frame_count * FRAME_TABLE_LEN);
        assert_eq!(summed.source_hist, source_hist);
        assert_eq!(summed.kept_hist, kept_hist);
        assert!(summed.source_blocks() > 0);

        for (group, record) in records.iter().enumerate() {
            let zero_lag = record[0];
            let sums = summed.group_autocov(group);
            for (lane, &sum) in sums.iter().enumerate() {
                assert_sum_close(sum, record[lane], zero_lag, "lag sum");
            }

            let pixels = summed.pixels[group];
            assert_sum_close(pixels, record[LAG_COUNT], record[LAG_COUNT], "pixels");
        }
    }
}

/// The same grain field on frames 0 to 3 and flat frames after, so only frame 3's pair has grain.
fn grain_leaving_after_frame_3() -> StillStream {
    let size = BANDED_SIZE as usize;
    let field = gaussian_field(size, size, 500);
    let grained: Vec<f32> = field
        .iter()
        .map(|&sample| FLAT_LEVEL + SIGMA * sample as f32)
        .collect();
    let flat = vec![FLAT_LEVEL; size * size];
    let sources: Vec<Vec<f32>> = (0..LONE_FRAMES)
        .map(|index| {
            if index <= LONE_FRAME {
                grained.clone()
            } else {
                flat.clone()
            }
        })
        .collect();
    let outputs = vec![flat; LONE_FRAMES as usize];

    StillStream {
        width: BANDED_SIZE,
        height: BANDED_SIZE,
        step: DIRECT_STEP,
        sources,
        outputs,
    }
}

#[test]
fn each_frame_holds_only_its_own_statistics() {
    let stream = grain_leaving_after_frame_3();
    let chunks = export_stream(&stream);

    assert_eq!(chunks.len(), 1);

    let chunk = &chunks[0];
    let (frame_tables, _) = chunk.frame_tables.as_chunks::<FRAME_TABLE_LEN>();

    assert_eq!(frame_tables.len(), LONE_FRAMES as usize);
    for (index, frame_table) in frame_tables.iter().enumerate() {
        let has_values = frame_table.iter().any(|&value| value != 0.0);
        assert_eq!(has_values, index == LONE_FRAME as usize, "frame {index}");
    }

    let summed = chunk.summed_table();
    assert!(summed.source_blocks() > 0);
}

#[test]
fn a_table_from_frame_tables_matches_the_chunk_table() {
    let stream = banded_stream(TABLE_FRAMES);
    let chunks = export_stream(&stream);
    let scene = SceneGrain {
        first_frame: 0,
        chunks,
        chroma_chunks: Vec::new(),
    };

    let options = FitOptions::default();
    let table = build_table(&[scene], FRAME_RATE, &options);

    assert_eq!(table, BANDED_TABLE);
}

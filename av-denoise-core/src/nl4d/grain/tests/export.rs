#![cfg(any(feature = "vulkan", feature = "metal"))]

use cubecl::prelude::*;

use super::synthetic::gaussian_field;
use crate::bench_api::HostIo;
use crate::nl4d::denoiser::buffer_sizes;
use crate::nl4d::grain::chunk::GrainChunk;
use crate::nl4d::grain::consts::{
    CHUNK_FRAMES,
    FRAME_RECORD_LEN,
    HIST_LEN,
    LUMA_BINS,
    STD_BUCKETS,
    STRENGTH_GROUPS,
};
use crate::nl4d::grain::fit::{bucket_edges, hist_median};
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

fn make_client() -> Client {
    crate::tests::compilation_cache::install_test_compilation_cache();
    crate::testing::test_client()
}

fn params(grain_export: bool) -> Nl4dParams {
    params_with(grain_export, false)
}

fn params_with(grain_export: bool, grain_report: bool) -> Nl4dParams {
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
        grain_export,
        grain_report,
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
    let has_export = denoiser.has_grain_export();
    let mut outputs = Vec::new();

    denoise_stream(&mut denoiser, frames, &mut outputs);

    let chunks = denoiser.drain_grain_chunks().expect("drain");

    (outputs, chunks, has_export)
}

fn merged(chunks: &[GrainChunk]) -> GrainChunk {
    let mut merged = GrainChunk::empty();
    for chunk in chunks {
        merged.merge(chunk);
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
fn lag_3_ratio(chunk: &GrainChunk) -> f64 {
    let mut zero_lag = 0.0;
    let mut lag_3 = 0.0;
    for group in 0..STRENGTH_GROUPS {
        let sums = chunk.group_autocov(group);
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
fn export_off_allocates_nothing_and_drains_nothing() {
    let frames = grain_frames(SIGMA, 9);
    let export_off = params(false);
    let (_, chunks, has_export) = run(export_off, &frames);

    assert!(!has_export);
    assert!(chunks.is_empty());
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
    let merged = merged(&chunks);
    let median = median_of(&merged.source_hist);
    let counted = frames_counted(&chunks);
    let any_pixels = chunks
        .iter()
        .any(|chunk| chunk.pixels.iter().any(|&pixels| pixels > 0.0));

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
    let merged = merged(&chunks);
    let median = median_of(&merged.source_hist);
    let ratio = lag_3_ratio(&merged);

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
    let merged = merged(&chunks);
    let accepted: u32 = merged.source_hist.iter().sum();
    let median = median_of(&merged.source_hist);

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
    let merged = merged(&chunks);
    let kept_total: u32 = merged.kept_hist.iter().sum();
    let source_total: u32 = merged.source_hist.iter().sum();

    let kept_median = median_of(&merged.kept_hist);
    let source_median = median_of(&merged.source_hist);

    assert!(kept_total > 0);
    assert!(source_total > 0);
    assert_eq!(merged.kept_hist.len(), HIST_LEN);
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

#[test]
fn chroma_denoisers_never_export() {
    let mut chroma = params(true);
    chroma.nlm.channels = ChannelMode::Chroma;
    let plane_frames = grain_frames(SIGMA, 9);
    let frames: Vec<Vec<f32>> = plane_frames
        .into_iter()
        .map(|frame| frame.iter().flat_map(|&sample| [sample, sample]).collect())
        .collect();
    let (_, chunks, has_export) = run(chroma, &frames);

    assert!(!has_export);
    assert!(chunks.is_empty());
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
}

/// The names of the buffers a denoiser with `params` counts.
fn buffer_names(params: &Nl4dParams) -> Vec<&'static str> {
    let client = make_client();
    let sizes = buffer_sizes(&client, params, WIDTH, HEIGHT, false);

    sizes.into_iter().map(|(name, _)| name).collect()
}

fn has_table_readings(chunk: &GrainChunk) -> bool {
    let any_source = chunk.source_hist.iter().any(|&count| count > 0);
    let any_autocov = chunk.autocov.iter().any(|&sum| sum != 0.0);
    any_source && any_autocov
}

fn has_no_table_readings(chunk: &GrainChunk) -> bool {
    let empty = GrainChunk::empty();
    chunk.source_hist == empty.source_hist
        && chunk.kept_hist == empty.kept_hist
        && chunk.autocov == empty.autocov
        && chunk.pixels == empty.pixels
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
fn a_table_only_export_measures_no_persistence() {
    let frames = grain_frames(SIGMA, 9);
    let table_only = params_with(true, false);
    let names = buffer_names(&table_only);

    let (_, chunks, _) = run(table_only, &frames);

    let counted = frames_counted(&chunks);
    assert_eq!(counted, 9);
    assert!(chunks.iter().all(|chunk| chunk.frame_records.is_empty()));
    assert!(chunks.iter().all(has_table_readings));
    assert!(names.contains(&SAVED_VECTORS));
    assert!(!names.contains(&PERSISTENCE_HISTORY));
    assert!(!names.contains(&PERSISTENCE_PARTIALS));
}

#[test]
fn a_report_only_export_skips_the_table_measurement() {
    let frames = grain_frames(SIGMA, 9);
    let report_only = params_with(false, true);
    let names = buffer_names(&report_only);

    let (_, chunks, has_export) = run(report_only, &frames);

    let counted = frames_counted(&chunks);
    assert!(has_export);
    assert_eq!(counted, 9);
    assert!(chunks.iter().all(has_no_table_readings));
    assert!(chunks.iter().all(has_persistence_readings));
    assert!(!names.contains(&SAVED_VECTORS));
    assert!(names.contains(&PERSISTENCE_HISTORY));
    assert!(names.contains(&PERSISTENCE_PARTIALS));
}

#[test]
fn both_switches_measure_both_parts() {
    let frames = grain_frames(SIGMA, 9);
    let both = params_with(true, true);
    let names = buffer_names(&both);

    let (_, chunks, _) = run(both, &frames);

    let counted = frames_counted(&chunks);
    assert_eq!(counted, 9);
    assert!(chunks.iter().all(has_table_readings));
    assert!(chunks.iter().all(has_persistence_readings));
    assert!(names.contains(&SAVED_VECTORS));
    assert!(names.contains(&PERSISTENCE_HISTORY));
    assert!(names.contains(&PERSISTENCE_PARTIALS));
}

#[test]
fn the_table_does_not_change_with_the_report_on() {
    let frames = grain_frames(SIGMA, 9);
    let table_only = params_with(true, false);
    let both = params_with(true, true);

    let (_, table_chunks, _) = run(table_only, &frames);
    let (_, both_chunks, _) = run(both, &frames);

    assert_eq!(table_chunks.len(), both_chunks.len());
    for (table_chunk, both_chunk) in table_chunks.iter().zip(&both_chunks) {
        assert_eq!(table_chunk.frames, both_chunk.frames);
        assert_eq!(table_chunk.source_hist, both_chunk.source_hist);
        assert_eq!(table_chunk.kept_hist, both_chunk.kept_hist);
        assert_eq!(table_chunk.autocov, both_chunk.autocov);
        assert_eq!(table_chunk.pixels, both_chunk.pixels);
    }
}

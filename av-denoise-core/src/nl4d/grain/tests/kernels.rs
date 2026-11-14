#![cfg(any(feature = "vulkan", feature = "metal"))]

use cubecl::prelude::*;

use super::mirror::{MirrorFrame, mirror_measure};
use super::synthetic::gaussian_field;
use crate::nl4d::grain::consts::{
    AUTOCOV_LEN,
    GROUPED_AUTOCOV_LEN,
    HIST_LEN,
    PARTIAL_LEN,
    REDUCE_THREADS,
    STD_BUCKETS,
    STRENGTH_GROUPS,
};
use crate::nl4d::grain::fit::{bucket_edges, bucket_of, hist_median};
use crate::nl4d::kernels::{grain_measure, grain_reduce_partials, grain_save_vectors};
use crate::noise::TEMPORAL_NOISE_BLOCK;
use crate::noise::colour::COLOUR_MEAN_FIELDS;

const THREADS: u32 = 16;
const PLAIN: Shape = Shape {
    width: 96,
    height: 64,
    step: 8,
};
/// A frame with partial cells at the right and bottom, and 16-pixel motion blocks.
const RAGGED: Shape = Shape {
    width: 100,
    height: 70,
    step: 16,
};
/// A ragged frame of nine whole cell rows and a partial one.
const RAGGED_CHROMA: Shape = Shape {
    width: 100,
    height: 8 * 9 + 4,
    step: 16,
};
/// The flat level of the channel a two-channel scene does not measure.
const OTHER_LEVEL: f32 = 0.8;
/// A flat level below the luma band.
const DARK_LEVEL: f32 = 10.0 / 255.0;
const SIGMA: f32 = 2.0 / 255.0;
/// The colour block luma of the left half of a two-level means grid, on the 8-bit scale.
const LEFT_LUMA: f32 = 40.0;
/// The colour block luma of the right half of a two-level means grid, on the 8-bit scale.
const RIGHT_LUMA: f32 = 200.0;
/// The colour block luma of a ragged grid's last block row, alone in brightness bin 14.
const BOTTOM_LUMA: f32 = 232.0;
/// Counts in one frame's histograms.
const FRAME_HIST_LEN: usize = 2 * HIST_LEN;
/// Frame slots a measure or reduce writes into, the kernel's frame landing in the last one.
const MEASURED_SLOTS: usize = 2;
const FLICKER: f32 = 2.0 / 255.0;
/// The lane of lag `(0, 3)` in a record.
const LAG_RIGHT_3: usize = 3;

fn make_client() -> Client {
    crate::tests::compilation_cache::install_test_compilation_cache();
    crate::testing::test_client()
}

#[test]
fn save_vectors_copies_one_neighbour_into_its_entry() {
    let client = make_client();
    let blocks = 37u32;
    let neighbours = 4u32;
    let entries = 3u32;
    let mv_host: Vec<i32> = (0..neighbours * blocks * 2).map(|i| i as i32 - 50).collect();
    let conf_host: Vec<f32> = (0..neighbours * blocks).map(|i| i as f32 * 0.01).collect();
    let saved_mv_len = (entries * blocks * 2) as usize;
    let saved_conf_len = (entries * blocks) as usize;
    let saved_mv_zeros = vec![0i32; saved_mv_len];
    let saved_conf_zeros = vec![0.0f32; saved_conf_len];

    let mv_bytes = i32::as_bytes(&mv_host);
    let conf_bytes = f32::as_bytes(&conf_host);
    let saved_mv_zero_bytes = i32::as_bytes(&saved_mv_zeros);
    let saved_conf_zero_bytes = f32::as_bytes(&saved_conf_zeros);
    let mv = client.create_from_slice(mv_bytes);
    let conf = client.create_from_slice(conf_bytes);
    let saved_mv = client.create_from_slice(saved_mv_zero_bytes);
    let saved_conf = client.create_from_slice(saved_conf_zero_bytes);

    let neighbour = 2u32;
    let entry = 1u32;
    let grid = CubeCount::new_1d(1);
    let dim = CubeDim::new_1d(THREADS);

    unsafe {
        grain_save_vectors::launch_unchecked(
            &client,
            grid,
            dim,
            BufferArg::from_raw_parts(mv, mv_host.len()),
            BufferArg::from_raw_parts(conf, conf_host.len()),
            BufferArg::from_raw_parts(saved_mv.clone(), saved_mv_len),
            BufferArg::from_raw_parts(saved_conf.clone(), saved_conf_len),
            neighbour * blocks * 2,
            neighbour * blocks,
            entry,
            blocks,
            THREADS,
        );
    }

    let saved_mv_bytes = client.read_one(saved_mv).expect("readback");
    let saved_conf_bytes = client.read_one(saved_conf).expect("readback");
    let saved_mv = i32::from_bytes(&saved_mv_bytes);
    let saved_conf = f32::from_bytes(&saved_conf_bytes);

    let mv_start = (neighbour * blocks * 2) as usize;
    let mv_end = mv_start + (blocks * 2) as usize;
    let saved_start = (entry * blocks * 2) as usize;
    let saved_end = saved_start + (blocks * 2) as usize;
    assert_eq!(&saved_mv[saved_start..saved_end], &mv_host[mv_start..mv_end]);
    assert!(saved_mv[..saved_start].iter().all(|&value| value == 0));
    assert!(saved_mv[saved_end..].iter().all(|&value| value == 0));

    let conf_start = (neighbour * blocks) as usize;
    let saved_conf_start = (entry * blocks) as usize;
    let saved_conf_end = saved_conf_start + blocks as usize;
    let expected = &conf_host[conf_start..conf_start + blocks as usize];
    assert_eq!(&saved_conf[saved_conf_start..saved_conf_end], expected);
}

#[derive(Clone, Copy)]
struct Shape {
    width: u32,
    height: u32,
    step: u32,
}

impl Shape {
    fn blocks_x(&self) -> u32 {
        self.width.div_ceil(self.step)
    }

    fn blocks_y(&self) -> u32 {
        self.height.div_ceil(self.step)
    }

    fn blocks(&self) -> usize {
        (self.blocks_x() * self.blocks_y()) as usize
    }

    fn pixels(&self) -> usize {
        (self.width * self.height) as usize
    }

    fn cells_x(&self) -> u32 {
        self.width / 8
    }

    fn cells(&self) -> usize {
        (self.cells_x() * (self.height / 8)) as usize
    }

    fn block_at(&self, block_x: u32, block_y: u32) -> usize {
        (block_y * self.blocks_x() + block_x) as usize
    }
}

struct Scene {
    shape: Shape,
    /// Interleaved samples per pixel.
    channels: u32,
    /// The channel the measurement reads.
    channel: u32,
    luma_band: bool,
    source_t: Vec<f32>,
    source_next: Vec<f32>,
    out_t: Vec<f32>,
    source_mv: Vec<(i32, i32)>,
    source_conf: Vec<f32>,
    /// Each colour block's luma, Cb and Cr means, `None` when not held.
    luma_means: Option<Vec<f32>>,
    /// The colour blocks in one row, 0 when one-frame cells bin by their own mean.
    means_blocks_x: u32,
}

/// Moves `values` by `shift` whole pixels, repeating the frame edge.
fn shifted(values: &[f32], shape: Shape, shift: (i32, i32)) -> Vec<f32> {
    let width = shape.width as i32;
    let height = shape.height as i32;
    let mut moved = vec![0.0f32; values.len()];

    for y in 0..height {
        for x in 0..width {
            let from_x = (x - shift.0).clamp(0, width - 1);
            let from_y = (y - shift.1).clamp(0, height - 1);
            let index = (y * width + x) as usize;
            moved[index] = values[(from_y * width + from_x) as usize];
        }
    }

    moved
}

fn add_noise(values: &[f32], sigma: f32, seed: u64, shape: Shape) -> Vec<f32> {
    let noise = gaussian_field(shape.width as usize, shape.height as usize, seed);
    values
        .iter()
        .zip(noise.iter())
        .map(|(&value, &sample)| value + sigma * sample as f32)
        .collect()
}

/// Fresh grain each frame over a flat 0.5 base, or 16x16 tiles of 0.4 and 0.6 when `tiled`.
///
/// `shift` moves the next frame's content by whole pixels, and the saved vectors follow it. The
/// output is the clean base.
fn scene_with(shape: Shape, shift: (i32, i32), tiled: bool, textured_cells: &[(u32, u32)]) -> Scene {
    let width = shape.width;
    let mut base = vec![0.5f32; shape.pixels()];

    if tiled {
        for y in 0..shape.height {
            for x in 0..width {
                let odd = (x / 16 + y / 16) % 2 == 1;
                base[(y * width + x) as usize] = if odd { 0.6 } else { 0.4 };
            }
        }
    }

    for &(cell_x, cell_y) in textured_cells {
        for y in cell_y * 8..cell_y * 8 + 8 {
            for x in cell_x * 8..cell_x * 8 + 8 {
                base[(y * width + x) as usize] = if (x + y) % 2 == 0 { 0.3 } else { 0.7 };
            }
        }
    }

    let source_t = add_noise(&base, SIGMA, 1, shape);
    let moved_base = shifted(&base, shape, shift);
    let source_next = add_noise(&moved_base, SIGMA, 2, shape);
    let blocks = shape.blocks();

    Scene {
        shape,
        channels: 1,
        channel: 0,
        luma_band: true,
        source_t,
        source_next,
        out_t: base,
        source_mv: vec![shift; blocks],
        source_conf: vec![1.0; blocks],
        luma_means: None,
        means_blocks_x: 0,
    }
}

fn flat_scene(shift: (i32, i32), textured_cells: &[(u32, u32)]) -> Scene {
    scene_with(PLAIN, shift, false, textured_cells)
}

fn moving_scene(shape: Shape) -> Scene {
    scene_with(shape, (1, -1), false, &[])
}

/// The same grain at `t` and `t + 1` over a flat base, with still vectors.
fn static_scene() -> Scene {
    let mut scene = flat_scene((0, 0), &[]);
    scene.source_next = scene.source_t.clone();

    scene
}

/// Bins the scene's one-frame cells by a colour grid, each block's luma given by `luma_of`.
///
/// The grid covers the scene in 16x16 blocks by `div_ceil`. Without `held`, the scene has the
/// grid's shape but no means.
fn with_luma_means(mut scene: Scene, held: bool, luma_of: impl Fn(u32, u32) -> f32) -> Scene {
    let blocks_x = scene.shape.width.div_ceil(TEMPORAL_NOISE_BLOCK);
    let blocks_y = scene.shape.height.div_ceil(TEMPORAL_NOISE_BLOCK);
    let mut means = Vec::with_capacity((blocks_x * blocks_y * COLOUR_MEAN_FIELDS) as usize);
    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            let luma = luma_of(block_x, block_y);
            means.extend_from_slice(&[luma, 128.0, 128.0]);
        }
    }

    scene.means_blocks_x = blocks_x;
    scene.luma_means = held.then_some(means);

    scene
}

/// A flat two-channel scene measured on channel 1, binned by [LEFT_LUMA] and [RIGHT_LUMA] blocks.
fn two_level_chroma_scene(held: bool) -> Scene {
    let flat = flat_scene((0, 0), &[]);
    let chroma = on_channel(flat, 1, false);
    let half = chroma.shape.width.div_ceil(TEMPORAL_NOISE_BLOCK) / 2;

    with_luma_means(
        chroma,
        held,
        |block_x, _| {
            if block_x < half { LEFT_LUMA } else { RIGHT_LUMA }
        },
    )
}

/// Places each of `samples` on `channel` of a two-channel pixel, the other channel at [OTHER_LEVEL].
fn interleaved(samples: &[f32], channel: u32) -> Vec<f32> {
    let mut pixels = Vec::with_capacity(2 * samples.len());
    for &sample in samples {
        let mut pixel = [OTHER_LEVEL; 2];
        pixel[channel as usize] = sample;
        pixels.extend_from_slice(&pixel);
    }

    pixels
}

/// Moves a one-channel `scene` onto `channel` of a two-channel scene and measures that channel.
///
/// The other channel is flat [OTHER_LEVEL] without grain in every frame.
fn on_channel(scene: Scene, channel: u32, luma_band: bool) -> Scene {
    let source_t = interleaved(&scene.source_t, channel);
    let source_next = interleaved(&scene.source_next, channel);
    let out_t = interleaved(&scene.out_t, channel);

    Scene {
        channels: 2,
        channel,
        luma_band,
        source_t,
        source_next,
        out_t,
        ..scene
    }
}

/// Moves every sample of a one-channel `scene` so its flat base sits at `level`.
fn lowered(mut scene: Scene, level: f32) -> Scene {
    let offset = level - 0.5;
    let samples = scene
        .source_t
        .iter_mut()
        .chain(scene.source_next.iter_mut())
        .chain(scene.out_t.iter_mut());
    for sample in samples {
        *sample += offset;
    }

    scene
}

/// The median of a histogram's counts over every luma bin.
fn merged_median(hist: &[u32]) -> f64 {
    let mut merged = vec![0u32; STD_BUCKETS];
    for (index, &count) in hist.iter().enumerate() {
        merged[index % STD_BUCKETS] += count;
    }

    let edges = bucket_edges();
    hist_median(&merged, &edges).expect("accepted blocks")
}

fn source_median(hist: &[u32]) -> f64 {
    merged_median(&hist[..HIST_LEN])
}

fn source_total(hist: &[u32]) -> u32 {
    hist[..HIST_LEN].iter().sum()
}

fn frame_total(hist: &[u32]) -> u32 {
    hist[HIST_LEN..].iter().sum()
}

/// The one-frame count of each luma bin.
fn frame_bin_totals(hist: &[u32]) -> Vec<u32> {
    let (bins, _) = hist[HIST_LEN..].as_chunks::<STD_BUCKETS>();

    bins.iter().map(|bin| bin.iter().sum()).collect()
}

fn run_measure(scene: &Scene, has_source: bool) -> (Vec<u32>, Vec<f64>) {
    let client = make_client();
    let shape = scene.shape;
    let samples = shape.pixels() * scene.channels as usize;
    let blocks = shape.blocks();
    let cells = shape.cells();

    let mut ring = scene.source_t.clone();
    ring.extend_from_slice(&scene.source_next);

    let mut saved_mv = Vec::with_capacity(2 * blocks);
    for &(dx, dy) in &scene.source_mv {
        saved_mv.push(dx);
        saved_mv.push(dy);
    }

    let placeholder = vec![0.0f32];
    let held_means = scene.luma_means.as_ref();
    let luma_means = held_means.unwrap_or(&placeholder);
    let has_luma_means = held_means.is_some();
    let edges = bucket_edges();
    let hist_zeros = vec![0i32; MEASURED_SLOTS * FRAME_HIST_LEN];

    let ring_bytes = f32::as_bytes(&ring);
    let out_t_bytes = f32::as_bytes(&scene.out_t);
    let saved_mv_bytes = i32::as_bytes(&saved_mv);
    let saved_conf_bytes = f32::as_bytes(&scene.source_conf);
    let edges_bytes = f32::as_bytes(&edges);
    let means_bytes = f32::as_bytes(luma_means);
    let hist_zero_bytes = i32::as_bytes(&hist_zeros);
    let input = client.create_from_slice(ring_bytes);
    let out_t = client.create_from_slice(out_t_bytes);
    let saved_mv = client.create_from_slice(saved_mv_bytes);
    let saved_conf = client.create_from_slice(saved_conf_bytes);
    let edges_buf = client.create_from_slice(edges_bytes);
    let means = client.create_from_slice(means_bytes);
    let hist = client.create_from_slice(hist_zero_bytes);
    let partials = client.empty(cells * PARTIAL_LEN * size_of::<f32>());

    let grid = CubeCount::new_2d(shape.width / 8, shape.height / 8);
    let dim = CubeDim::new_2d(8, 8);
    let blocks_x = shape.blocks_x();
    let blocks_y = shape.blocks_y();

    unsafe {
        grain_measure::launch_unchecked(
            &client,
            grid,
            dim,
            scene.channels as usize,
            BufferArg::from_raw_parts(input, 2 * samples),
            BufferArg::from_raw_parts(out_t, samples),
            BufferArg::from_raw_parts(saved_mv, 2 * blocks),
            BufferArg::from_raw_parts(saved_conf, blocks),
            BufferArg::from_raw_parts(edges_buf, edges.len()),
            BufferArg::from_raw_parts(means, luma_means.len()),
            BufferArg::from_raw_parts(hist.clone(), MEASURED_SLOTS * FRAME_HIST_LEN),
            BufferArg::from_raw_parts(partials.clone(), cells * PARTIAL_LEN),
            0u32,
            1u32,
            0u32,
            has_source as u32,
            has_luma_means as u32,
            FRAME_HIST_LEN as u32,
            shape.width,
            shape.height,
            scene.channels,
            blocks_x,
            blocks_y,
            shape.step,
            scene.channel,
            scene.luma_band,
            scene.means_blocks_x,
        );
    }

    let hist_bytes = client.read_one(hist).expect("readback");
    let partial_bytes = client.read_one(partials).expect("readback");
    let hist_counts = i32::from_bytes(&hist_bytes);
    let (earlier_slot, measured_slot) = hist_counts.split_at(FRAME_HIST_LEN);
    let hist: Vec<u32> = measured_slot[..FRAME_HIST_LEN]
        .iter()
        .map(|&count| count as u32)
        .collect();
    let partials = f32::from_bytes(&partial_bytes);

    assert!(earlier_slot.iter().all(|&count| count == 0));

    let mut autocov = vec![0.0f64; GROUPED_AUTOCOV_LEN];
    let (cell_partials, _) = partials.as_chunks::<PARTIAL_LEN>();
    for partial in cell_partials {
        let group = partial[AUTOCOV_LEN] as usize;
        for lane in 0..AUTOCOV_LEN {
            autocov[group * AUTOCOV_LEN + lane] += partial[lane] as f64;
        }
    }

    (hist, autocov)
}

fn mirror_of(scene: &Scene, has_source: bool) -> (Vec<u32>, Vec<f64>) {
    let edges = bucket_edges();
    let shape = scene.shape;
    let frame = MirrorFrame {
        width: shape.width,
        height: shape.height,
        channels: scene.channels,
        channel: scene.channel,
        luma_band: scene.luma_band,
        source_t: &scene.source_t,
        source_next: &scene.source_next,
        out_t: &scene.out_t,
        source_mv: &scene.source_mv,
        source_conf: &scene.source_conf,
        luma_means: scene.luma_means.as_deref(),
        means_blocks_x: scene.means_blocks_x,
        blocks_x: shape.blocks_x(),
        blocks_y: shape.blocks_y(),
        step: shape.step,
        has_source,
    };
    let record = mirror_measure(&frame, &edges);
    (record.hist, record.autocov)
}

fn assert_autocov_close(gpu: &[f64], host: &[f64]) {
    for lane in 0..GROUPED_AUTOCOV_LEN {
        let tolerance = 1e-4 * host[lane].abs().max(1e-6);
        let difference = (gpu[lane] - host[lane]).abs();
        assert!(
            difference <= tolerance,
            "lane {lane}: {} vs {}",
            gpu[lane],
            host[lane]
        );
    }
}

fn assert_matches_mirror(scene: &Scene) -> Vec<u32> {
    let (gpu_hist, gpu_autocov) = run_measure(scene, true);
    let (host_hist, host_autocov) = mirror_of(scene, true);

    let pixels = pixels_of(&gpu_autocov);

    assert_eq!(gpu_hist, host_hist);
    assert_autocov_close(&gpu_autocov, &host_autocov);
    assert!(pixels > 0.0);

    gpu_hist
}

/// The pixel count summed over every strength group.
fn pixels_of(autocov: &[f64]) -> f64 {
    let (records, _) = autocov.as_chunks::<AUTOCOV_LEN>();
    records.iter().map(|record| record[AUTOCOV_LEN - 1]).sum()
}

/// `R(0, 3) / R(0, 0)` with both lags summed over every strength group.
fn lag_3_ratio(autocov: &[f64]) -> f64 {
    let mut zero_lag = 0.0;
    let mut lag_3 = 0.0;
    let (records, _) = autocov.as_chunks::<AUTOCOV_LEN>();
    for record in records {
        zero_lag += record[0];
        lag_3 += record[LAG_RIGHT_3];
    }

    assert!(zero_lag > 0.0);

    lag_3 / zero_lag
}

/// Brightens every pixel of source frame `t + 1` by [FLICKER].
fn with_flicker(mut scene: Scene) -> Scene {
    for sample in &mut scene.source_next {
        *sample += FLICKER;
    }

    scene
}

/// Sets every pixel of source frame `t` in the given row of cells to `value`.
fn fill_cell_row(scene: &mut Scene, cell_y: u32, value: f32) {
    let width = scene.shape.width as usize;
    let start = cell_y as usize * 8 * width;

    for sample in &mut scene.source_t[start..start + 8 * width] {
        *sample = value;
    }
}

#[test]
fn measure_matches_the_mirror_on_flat_grain() {
    let scene = flat_scene((0, 0), &[]);
    let hist = assert_matches_mirror(&scene);
    let source = source_total(&hist);

    assert!(source > 0);
}

#[test]
fn frame_hist_matches_the_mirror_on_flat_grain() {
    let scene = moving_scene(PLAIN);
    let hist = assert_matches_mirror(&scene);
    let source = source_total(&hist);
    let frame = frame_total(&hist);

    assert!(source > 0);
    assert!(frame > 0);
}

#[test]
fn measure_matches_the_mirror_on_a_ragged_frame() {
    let scene = moving_scene(RAGGED);
    let hist = assert_matches_mirror(&scene);
    let source = source_total(&hist);
    let frame = frame_total(&hist);

    assert!(source > 0);
    assert!(frame > 0);
}

#[test]
fn static_grain_fills_the_frame_hist_in_one_frame() {
    let scene = static_scene();
    let edges = bucket_edges();
    let (gpu_hist, _) = run_measure(&scene, true);
    let (host_hist, _) = mirror_of(&scene, true);
    let source = source_total(&gpu_hist);
    let frame_median = merged_median(&gpu_hist[HIST_LEN..]);
    let median_bucket = bucket_of(frame_median as f32, &edges);
    let sigma_bucket = bucket_of(SIGMA, &edges);

    assert_eq!(gpu_hist, host_hist);
    assert!(source <= 1, "{source} source cells");
    assert!(
        median_bucket.abs_diff(sigma_bucket) <= 1,
        "median {frame_median} vs sigma {SIGMA}"
    );
}

#[test]
fn a_one_frame_cell_needs_no_motion_confidence() {
    let mut scene = flat_scene((0, 0), &[]);
    scene.source_conf.fill(0.0);
    let (gpu_hist, _) = run_measure(&scene, true);
    let (host_hist, _) = mirror_of(&scene, true);
    let source = source_total(&gpu_hist);
    let frame = frame_total(&gpu_hist);

    assert_eq!(gpu_hist, host_hist);
    assert_eq!(source, 0);
    assert_eq!(frame as usize, PLAIN.cells());
}

/// A flat scene whose source frame `t` takes `pattern(x, y)` at every pixel.
fn patterned_scene(pattern: impl Fn(u32, u32) -> f32) -> Scene {
    let mut scene = flat_scene((0, 0), &[]);
    let width = PLAIN.width;

    for y in 0..PLAIN.height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            scene.source_t[index] = pattern(x, y);
        }
    }

    scene
}

/// Moves a patterned scene onto channel 1, binned by a colour grid of [LEFT_LUMA] blocks.
fn chroma_of(scene: Scene) -> Scene {
    let chroma = on_channel(scene, 1, false);
    with_luma_means(chroma, true, |_, _| LEFT_LUMA)
}

/// The geometric middle of std bucket `bucket`, clear of both its edges.
fn bucket_middle(bucket: usize) -> f32 {
    let edges = bucket_edges();
    let product = edges[bucket] * edges[bucket + 1];
    product.sqrt()
}

/// Checks every one-frame cell of the frame landed in std bucket `bucket`.
fn assert_frame_cells_in_bucket(hist: &[u32], bucket: usize) {
    let (bins, _) = hist[HIST_LEN..].as_chunks::<STD_BUCKETS>();
    let in_bucket: u32 = bins.iter().map(|bin| bin[bucket]).sum();
    let frame = frame_total(hist);

    assert_eq!(frame as usize, PLAIN.cells());
    assert_eq!(in_bucket, frame, "{bins:?}");
}

#[test]
fn one_frame_hist_counts_a_constant_cell_in_bucket_0() {
    let luma = patterned_scene(|_, _| 0.5);
    let constant = patterned_scene(|_, _| 0.5);
    let chroma = chroma_of(constant);

    let luma_hist = assert_matches_mirror(&luma);
    let chroma_hist = assert_matches_mirror(&chroma);

    assert_frame_cells_in_bucket(&luma_hist, 0);
    assert_frame_cells_in_bucket(&chroma_hist, 0);
}

#[test]
fn one_frame_hist_bins_the_high_pass_of_a_checkerboard() {
    let bucket = 40;
    let amplitude = bucket_middle(bucket) / std::f32::consts::SQRT_2;
    let checkerboard = |x: u32, y: u32| {
        let sign = if (x + y).is_multiple_of(2) { 1.0 } else { -1.0 };
        0.5 + sign * amplitude
    };
    let luma = patterned_scene(checkerboard);
    let pattern = patterned_scene(checkerboard);
    let chroma = chroma_of(pattern);

    let luma_hist = assert_matches_mirror(&luma);
    let chroma_hist = assert_matches_mirror(&chroma);

    assert_frame_cells_in_bucket(&luma_hist, bucket);
    assert_frame_cells_in_bucket(&chroma_hist, bucket);
}

#[test]
fn one_frame_hist_bins_a_ramp_by_its_slope() {
    let bucket = 40;
    let slope = 2.0 * bucket_middle(bucket);
    let ramp = |x: u32, _| 0.5 + slope * ((x % 8) as f32 - 3.5);
    let scene = patterned_scene(ramp);
    let hist = assert_matches_mirror(&scene);

    assert_frame_cells_in_bucket(&hist, bucket);
}

#[test]
fn chroma_frame_hist_bins_by_the_luma_means() {
    let scene = two_level_chroma_scene(true);
    let (gpu_hist, _) = run_measure(&scene, true);
    let (host_hist, _) = mirror_of(&scene, true);
    let bins = frame_bin_totals(&gpu_hist);
    let elsewhere: u32 = bins
        .iter()
        .enumerate()
        .filter(|&(bin, _)| bin != 2 && bin != 12)
        .map(|(_, &count)| count)
        .sum();

    assert_eq!(gpu_hist, host_hist);
    assert!(bins[2] > 0);
    assert!(bins[12] > 0);
    assert_eq!(elsewhere, 0);
}

#[test]
fn chroma_frame_hist_is_empty_without_luma_means() {
    let held = two_level_chroma_scene(true);
    let missing = two_level_chroma_scene(false);
    let (held_hist, held_autocov) = run_measure(&held, true);
    let (missing_hist, missing_autocov) = run_measure(&missing, true);
    let (host_hist, _) = mirror_of(&missing, true);
    let held_frame = frame_total(&held_hist);
    let missing_frame = frame_total(&missing_hist);

    assert_eq!(missing_hist, host_hist);
    assert_eq!(missing_frame, 0);
    assert!(held_frame > 0);
    assert_eq!(missing_hist[..HIST_LEN], held_hist[..HIST_LEN]);
    assert_autocov_close(&missing_autocov, &held_autocov);
}

#[test]
fn chroma_frame_hist_matches_the_mirror_on_a_ragged_frame() {
    let moving = moving_scene(RAGGED_CHROMA);
    let chroma = on_channel(moving, 1, false);
    let bottom_row = RAGGED_CHROMA.height.div_ceil(TEMPORAL_NOISE_BLOCK) - 1;
    let scene = with_luma_means(chroma, true, |block_x, block_y| {
        if block_y == bottom_row {
            return BOTTOM_LUMA;
        }

        24.0 + 20.0 * block_x as f32
    });
    let hist = assert_matches_mirror(&scene);
    let bins = frame_bin_totals(&hist);
    let source = source_total(&hist);

    assert_eq!(scene.means_blocks_x, 7);
    assert_eq!(bottom_row, 4);
    assert!(source > 0);
    assert!(bins[14] > 0, "{bins:?}");
}

#[test]
fn chroma_measure_matches_the_mirror_on_flat_grain() {
    let flat = flat_scene((0, 0), &[]);
    let scene = on_channel(flat, 1, false);
    let hist = assert_matches_mirror(&scene);
    let source = source_total(&hist);

    assert!(source > 0);
}

#[test]
fn chroma_measure_ignores_the_luma_band() {
    let flat = flat_scene((0, 0), &[]);
    let dark = lowered(flat, DARK_LEVEL);
    let unbanded = on_channel(dark, 1, false);
    let flat_again = flat_scene((0, 0), &[]);
    let dark_again = lowered(flat_again, DARK_LEVEL);
    let banded = on_channel(dark_again, 1, true);

    let unbanded_hist = assert_matches_mirror(&unbanded);
    let (banded_hist, banded_autocov) = run_measure(&banded, true);
    let (host_hist, _) = mirror_of(&banded, true);
    let unbanded_source = source_total(&unbanded_hist);
    let banded_source = source_total(&banded_hist);

    assert!(unbanded_source > 0);
    assert_eq!(banded_hist, host_hist);
    assert_eq!(banded_source, 0);
    assert!(banded_autocov.iter().all(|&sum| sum == 0.0));
}

#[test]
fn luma_measure_reads_channel_zero_of_two() {
    let single = moving_scene(PLAIN);
    let moving = moving_scene(PLAIN);
    let paired = on_channel(moving, 0, true);

    let (single_hist, single_autocov) = run_measure(&single, true);
    let paired_hist = assert_matches_mirror(&paired);
    let (_, paired_autocov) = run_measure(&paired, true);
    let source = source_total(&paired_hist);
    let frame = frame_total(&paired_hist);

    assert_eq!(paired_hist, single_hist);
    assert_autocov_close(&paired_autocov, &single_autocov);
    assert!(source > 0);
    assert!(frame > 0);
}

#[test]
fn flicker_leaves_the_record_unchanged() {
    let steady = flat_scene((0, 0), &[]);
    let unflickered = flat_scene((0, 0), &[]);
    let flickered = with_flicker(unflickered);
    let (steady_hist, steady_autocov) = run_measure(&steady, true);
    let (flicker_hist, flicker_autocov) = run_measure(&flickered, true);
    let (host_hist, host_autocov) = mirror_of(&flickered, true);
    let steady_ratio = lag_3_ratio(&steady_autocov);
    let flicker_ratio = lag_3_ratio(&flicker_autocov);
    let steady_source = source_total(&steady_hist);

    assert_eq!(flicker_hist, host_hist);
    assert_autocov_close(&flicker_autocov, &host_autocov);
    assert!(steady_source > 0);
    assert_eq!(flicker_hist, steady_hist);
    assert!(
        (flicker_ratio - steady_ratio).abs() < 0.02,
        "{flicker_ratio} vs {steady_ratio}"
    );
    assert!(flicker_ratio < 0.1, "{flicker_ratio}");
}

#[test]
fn followed_motion_measures_the_same_grain_as_a_still_pair() {
    let still = scene_with(PLAIN, (0, 0), true, &[]);
    let moved = scene_with(PLAIN, (3, -2), true, &[]);
    let mut ignored = scene_with(PLAIN, (3, -2), true, &[]);
    ignored.source_mv.fill((0, 0));

    let (still_hist, _) = run_measure(&still, true);
    let (moved_hist, _) = run_measure(&moved, true);
    let (ignored_hist, _) = run_measure(&ignored, true);
    let still_median = source_median(&still_hist);
    let moved_median = source_median(&moved_hist);
    let ignored_median = source_median(&ignored_hist);

    assert!(
        (moved_median / still_median - 1.0).abs() < 0.05,
        "{moved_median} vs {still_median}"
    );
    assert!(
        ignored_median > 1.2 * still_median,
        "{ignored_median} vs {still_median}"
    );
}

#[test]
fn a_textured_cell_is_rejected() {
    let plain = flat_scene((0, 0), &[]);
    let textured = flat_scene((0, 0), &[(3, 3)]);
    let (plain_hist, _) = run_measure(&plain, true);
    let (textured_hist, _) = run_measure(&textured, true);
    let plain_source = source_total(&plain_hist);
    let textured_source = source_total(&textured_hist);

    assert_eq!(textured_source + 1, plain_source);
}

#[test]
fn a_cell_without_grain_is_rejected() {
    let plain = flat_scene((0, 0), &[]);
    let mut empty = flat_scene((0, 0), &[]);
    let width = PLAIN.width;

    for y in 24..32 {
        for x in 24..32 {
            let index = (y * width + x) as usize;
            empty.source_next[index] = empty.source_t[index];
        }
    }

    let (plain_hist, _) = run_measure(&plain, true);
    let (empty_hist, _) = run_measure(&empty, true);
    let plain_source = source_total(&plain_hist);
    let empty_source = source_total(&empty_hist);

    assert_eq!(empty_source + 1, plain_source);
}

#[test]
fn a_low_confidence_cell_is_rejected() {
    let mut scene = flat_scene((0, 0), &[]);
    let (before, _) = run_measure(&scene, true);
    let block = PLAIN.block_at(3, 3);
    scene.source_conf[block] = 0.5;
    let (after, _) = run_measure(&scene, true);
    let source_before = source_total(&before);
    let source_after = source_total(&after);

    assert_eq!(source_after + 1, source_before);
}

#[test]
fn missing_pairs_give_zero_counts() {
    let scene = flat_scene((0, 0), &[]);
    let (hist, autocov) = run_measure(&scene, false);

    assert!(hist.iter().all(|&count| count == 0));
    assert!(autocov.iter().all(|&sum| sum == 0.0));
}

#[test]
fn clipped_blocks_are_rejected() {
    let plain = flat_scene((0, 0), &[]);
    let mut clipped = flat_scene((0, 0), &[]);
    fill_cell_row(&mut clipped, 0, 0.0);
    fill_cell_row(&mut clipped, 2, 1.0);

    let (plain_hist, _) = run_measure(&plain, true);
    let (gpu_hist, _) = run_measure(&clipped, true);
    let (host_hist, _) = mirror_of(&clipped, true);
    let interior_per_row = PLAIN.cells_x() - 2;
    let plain_source = source_total(&plain_hist);
    let clipped_source = source_total(&gpu_hist);

    assert_eq!(gpu_hist, host_hist);
    assert_eq!(clipped_source + 2 * interior_per_row, plain_source);
}

#[test]
fn ten_bit_and_eight_bit_give_the_same_record() {
    let mut eight = flat_scene((0, 0), &[]);
    let mut ten = flat_scene((0, 0), &[]);
    let quantise_eight = |value: f32| (value * 255.0).round() / 255.0;
    let quantise_ten = |value: f32| (value * 1023.0).round() / 1023.0;

    for value in eight.source_t.iter_mut().chain(eight.source_next.iter_mut()) {
        *value = quantise_eight(*value);
    }

    for value in ten.source_t.iter_mut().chain(ten.source_next.iter_mut()) {
        *value = quantise_ten(*value);
    }

    let (eight_hist, eight_autocov) = run_measure(&eight, true);
    let (ten_hist, ten_autocov) = run_measure(&ten, true);
    let eight_source = source_total(&eight_hist);
    let ten_source = source_total(&ten_hist);

    assert!(eight_source > 0);
    assert_eq!(eight_source, ten_source);
    assert_within_one_bucket(&eight_hist[..HIST_LEN], &ten_hist[..HIST_LEN]);
    assert_groups_close(&eight_autocov, &ten_autocov, 0.05);
}

/// Checks every luma bin holds the same blocks, each moved by at most one std bucket.
fn assert_within_one_bucket(first: &[u32], second: &[u32]) {
    let (first_bins, _) = first.as_chunks::<STD_BUCKETS>();
    let (second_bins, _) = second.as_chunks::<STD_BUCKETS>();

    for (bin, (first_row, second_row)) in first_bins.iter().zip(second_bins).enumerate() {
        let first_cumulative = cumulative(first_row);
        let second_cumulative = cumulative(second_row);
        let first_total = first_cumulative[STD_BUCKETS - 1];
        let second_total = second_cumulative[STD_BUCKETS - 1];
        assert_eq!(first_total, second_total, "bin {bin}");

        for (bucket, &first_count) in first_cumulative.iter().enumerate() {
            let below = bucket.saturating_sub(1);
            let above = (bucket + 1).min(STD_BUCKETS - 1);
            let second_low = second_cumulative[below];
            let second_high = second_cumulative[above];
            assert!(
                (second_low..=second_high).contains(&first_count),
                "bin {bin} bucket {bucket}: {first_count} outside {second_low}..={second_high}"
            );
        }
    }
}

fn cumulative(row: &[u32]) -> Vec<u32> {
    row.iter()
        .scan(0u32, |total, &count| {
            *total += count;
            Some(*total)
        })
        .collect()
}

/// Checks the records hold the same pixels, with each group's per-pixel lags within `tolerance`.
///
/// Quantisation can move a block into a neighbouring std bucket, and so into the next group, so
/// each group is compared per pixel and the pixel total over every group.
fn assert_groups_close(first: &[f64], second: &[f64], tolerance: f64) {
    let first_totals = summed_record(first);
    let second_totals = summed_record(second);
    let (first_records, _) = first.as_chunks::<AUTOCOV_LEN>();
    let (second_records, _) = second.as_chunks::<AUTOCOV_LEN>();
    let pixels_lane = AUTOCOV_LEN - 1;

    assert!(first_totals[0] > 0.0);
    assert_eq!(first_totals[pixels_lane], second_totals[pixels_lane]);

    let groups = first_records.iter().zip(second_records).enumerate();
    for (group, (first_record, second_record)) in groups {
        let first_pixels = first_record[pixels_lane];
        let second_pixels = second_record[pixels_lane];
        if first_pixels == 0.0 || second_pixels == 0.0 {
            assert_eq!(first_pixels, second_pixels, "group {group}");
            continue;
        }

        let variance = first_record[0] / first_pixels;
        for lane in 0..pixels_lane {
            let first_lag = first_record[lane] / first_pixels;
            let second_lag = second_record[lane] / second_pixels;
            let difference = (first_lag - second_lag).abs();
            assert!(
                difference <= tolerance * variance,
                "group {group} lane {lane}: {first_lag} vs {second_lag}"
            );
        }
    }
}

/// Every strength group's record added into one.
fn summed_record(autocov: &[f64]) -> Vec<f64> {
    let mut summed = vec![0.0f64; AUTOCOV_LEN];
    let (records, _) = autocov.as_chunks::<AUTOCOV_LEN>();
    for record in records {
        for (total, &value) in summed.iter_mut().zip(record) {
            *total += value;
        }
    }

    summed
}

#[test]
fn reduce_adds_every_cell_into_its_group() {
    let client = make_client();
    let cells = 1000usize;
    let groups_used = 13;
    let mut partials_host = Vec::with_capacity(cells * PARTIAL_LEN);
    for cell in 0..cells {
        for lane in 0..AUTOCOV_LEN {
            let value = ((cell + lane) % 17) as f32 * 0.25;
            partials_host.push(value);
        }

        let group = (cell * 7) % groups_used + (STRENGTH_GROUPS - groups_used);
        partials_host.push(group as f32);
    }

    let chunk_len = MEASURED_SLOTS * GROUPED_AUTOCOV_LEN;
    let chunk_start: Vec<f32> = (0..chunk_len).map(|lane| lane as f32).collect();
    let partials_bytes = f32::as_bytes(&partials_host);
    let start_bytes = f32::as_bytes(&chunk_start);
    let partials = client.create_from_slice(partials_bytes);
    let chunk = client.create_from_slice(start_bytes);

    let grid = CubeCount::new_1d(AUTOCOV_LEN as u32);
    let dim = CubeDim::new_1d(REDUCE_THREADS);

    unsafe {
        grain_reduce_partials::launch_unchecked(
            &client,
            grid,
            dim,
            BufferArg::from_raw_parts(partials, partials_host.len()),
            BufferArg::from_raw_parts(chunk.clone(), chunk_len),
            GROUPED_AUTOCOV_LEN as u32,
            cells as u32,
        );
    }

    let (earlier_start, start) = chunk_start.split_at(GROUPED_AUTOCOV_LEN);
    let mut expected: Vec<f64> = start.iter().map(|&value| value as f64).collect();
    let (cell_partials, _) = partials_host.as_chunks::<PARTIAL_LEN>();
    for partial in cell_partials {
        let group = partial[AUTOCOV_LEN] as usize;
        for lane in 0..AUTOCOV_LEN {
            expected[group * AUTOCOV_LEN + lane] += partial[lane] as f64;
        }
    }

    let bytes = client.read_one(chunk).expect("readback");
    let chunk_values = f32::from_bytes(&bytes);
    let (earlier_slot, reduced) = chunk_values.split_at(GROUPED_AUTOCOV_LEN);

    assert_eq!(earlier_slot, earlier_start);
    for index in 0..GROUPED_AUTOCOV_LEN {
        let difference = (reduced[index] as f64 - expected[index]).abs();
        assert!(
            difference < 1e-2,
            "index {index}: {} vs {}",
            reduced[index],
            expected[index]
        );
    }

    let unused = (STRENGTH_GROUPS - groups_used) * AUTOCOV_LEN;
    let grew = reduced[unused..]
        .iter()
        .zip(&start[unused..])
        .all(|(&sum, &first)| sum > first);
    assert_eq!(&reduced[..unused], &start[..unused]);
    assert!(grew);
}

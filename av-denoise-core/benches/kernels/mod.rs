pub mod accumulate;
pub mod bilateral;
pub mod cast_f16;
pub mod collab_aggregate;
pub mod collab_fused;
pub mod copy;
pub mod dist_2d_weight;
pub mod dist_2d_weight_ref;
pub mod distance;
pub mod distance_pair;
pub mod distance_pair_ref;
pub mod distance_ref;
pub mod egress;
pub mod finish;
pub mod fused_window;
pub mod grain;
pub mod horizontal_sum;
pub mod horizontal_sum_pair;
pub mod ingest;
pub mod mc_block_match_coarse;
pub mod mc_block_match_fine;
pub mod mc_chain_compose;
pub mod mc_confidence;
pub mod mc_downscale;
pub mod mc_warp;
pub mod mv_regularise;
pub mod nl4d_geometry;
pub mod noise_partial;
pub mod temporal_noise_stats;
pub mod vertical_weight;
pub mod vweight_pair_accumulate;
pub mod zero;

use av_denoise_core::bench_api::NlmParams;
pub use av_denoise_core::bench_api::{BLOCK_X, BLOCK_Y};
use cubecl::benchmark::{Benchmark, BenchmarkComputations, TimingMethod};
use cubecl::prelude::*;
use cubecl::server::Handle;

pub const WIDTH: u32 = 1920;
pub const HEIGHT: u32 = 1080;
pub const PATCH_RADIUS: u32 = 4;
pub const SEARCH_RADIUS: u32 = 2;
pub const Q_X: i32 = 1;
pub const Q_Y: i32 = 0;
pub const BILATERAL_SIGMA_S: f32 = 3.0;
pub const BILATERAL_SIGMA_R: f32 = 0.02;
pub const BLOCK_1D: u32 = 256;
pub const COPY_GRID_1D: u32 = 1024;

/// The logical channel count and row label of each channel mode.
pub const CHANNELS: &[(u32, &str)] = &[(1, "luma"), (2, "chroma"), (3, "yuv")];

pub fn stored_channels(channels: u32) -> u32 {
    match channels {
        1 => 1,
        2 => 2,
        _ => 4,
    }
}

pub fn make_synthetic_frame(width: u32, height: u32, channels: u32) -> Vec<f32> {
    let mut data = Vec::with_capacity((width * height * channels) as usize);
    for y in 0..height {
        for x in 0..width {
            let base = 0.5 + 0.2 * (x as f32 * 0.05).sin() * (y as f32 * 0.03).cos();
            for channel in 0..channels {
                let seed = (y * width + x) * channels + channel;
                let hash = seed
                    .wrapping_mul(2654435761)
                    .wrapping_add(seed.wrapping_mul(340573321));
                let noise = (hash as f32 / u32::MAX as f32 - 0.5) * 0.1;
                data.push((base + noise).clamp(0.0, 1.0));
            }
        }
    }

    data
}

/// A synthetic frame padded to the power-of-two lane count `NlmDenoiser` stores.
pub fn make_padded_frame(width: u32, height: u32, channels: u32) -> Vec<f32> {
    let stored_ch = stored_channels(channels);
    if stored_ch == channels {
        return make_synthetic_frame(width, height, channels);
    }

    let synthetic = make_synthetic_frame(width, height, channels);
    let mut data = vec![0.0f32; (width * height * stored_ch) as usize];
    for i in 0..(width * height) as usize {
        for channel in 0..channels as usize {
            data[i * stored_ch as usize + channel] = synthetic[i * channels as usize + channel];
        }
    }

    data
}

/// The Welsch coefficient for the bench parameters.
///
/// The channel mode stays at its default because the coefficient only depends on
/// `patch_radius` and `strength`.
pub fn h2_inv_norm() -> f32 {
    let params = NlmParams {
        patch_radius: PATCH_RADIUS,
        ..NlmParams::default()
    };

    params.h2_inv_norm()
}

pub fn cube_count_2d() -> CubeCount {
    let cubes_x = WIDTH.div_ceil(BLOCK_X);
    let cubes_y = HEIGHT.div_ceil(BLOCK_Y);

    CubeCount::new_2d(cubes_x, cubes_y)
}

pub fn cube_dim_2d() -> CubeDim {
    CubeDim::new_2d(BLOCK_X, BLOCK_Y)
}

pub fn block_sync<R: Runtime>(client: &ComputeClient<R>) {
    let sync = client.sync();
    cubecl::future::block_on(sync).unwrap();
}

pub fn shapes_with_channels(channels: u32) -> Vec<Vec<usize>> {
    vec![vec![WIDTH as usize, HEIGHT as usize, channels as usize]]
}

/// Buffers for a kernel that reads one frame and writes one output.
#[derive(Clone)]
pub struct InputOutput {
    pub input: Handle,
    pub output: Handle,
    pub frame_len: usize,
}

const NAME_WIDTH: usize = 44;

pub fn print_header() {
    println!(
        "  {:<NAME_WIDTH$}  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "kernel", "samp", "mean", "median", "min", "max", "fps",
    );

    let rule = "-".repeat(NAME_WIDTH + 6 + 12 * 5);
    println!("  {}", rule);
}

pub fn run<B: Benchmark>(bench: B) {
    let name = bench.name();
    if let Ok(filter) = std::env::var("BENCH_FILTER")
        && !filter.split('|').any(|pattern| name.contains(pattern))
    {
        return;
    }

    match bench.run(TimingMethod::Device) {
        Ok(durations) => {
            let computations = BenchmarkComputations::new(&durations);
            let mean_s = computations.mean.as_secs_f64();
            let fps = if mean_s > 0.0 { 1.0 / mean_s } else { 0.0 };
            let mean = fmt_us(computations.mean);
            let median = fmt_us(computations.median);
            let min = fmt_us(computations.min);
            let max = fmt_us(computations.max);

            println!(
                "  {:<NAME_WIDTH$}  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10.2}",
                name,
                durations.durations.len(),
                mean,
                median,
                min,
                max,
                fps,
            );
        },
        Err(err) => println!("  {name:<NAME_WIDTH$}  error: {err}"),
    }
}

fn fmt_us(duration: core::time::Duration) -> String {
    let micros = duration.as_secs_f64() * 1_000_000.0;
    if micros >= 1000.0 {
        format!("{:.3} ms", micros / 1000.0)
    } else {
        format!("{micros:.2} µs")
    }
}

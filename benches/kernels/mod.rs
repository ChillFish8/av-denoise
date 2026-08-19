use av_denoise::nlmeans::NlmParams;
pub use av_denoise::nlmeans::{BLOCK_X, BLOCK_Y};
use cubecl::benchmark::{Benchmark, BenchmarkComputations, TimingMethod};
use cubecl::prelude::*;
use cubecl::server::Handle;

pub mod accumulate;
pub mod bilateral;
pub mod collab_aggregate;
pub mod collab_ht;
pub mod collab_weight_ratio;
pub mod copy;
pub mod dist_2d_weight;
pub mod dist_2d_weight_ref;
pub mod distance;
pub mod distance_pair;
pub mod distance_pair_ref;
pub mod distance_ref;
pub mod finish;
pub mod fused_window;
pub mod horizontal_sum;
pub mod horizontal_sum_pair;
pub mod mc_block_match_coarse;
pub mod mc_block_match_fine;
pub mod mc_chain_compose;
pub mod mc_confidence;
pub mod mc_downscale;
pub mod mc_warp;
pub mod nl4d_group_temporal;
pub mod noise_partial;
pub mod temporal_noise_stats;
pub mod vertical_weight;
pub mod vweight_pair_accumulate;
pub mod zero;

pub const W: u32 = 1920;
pub const H: u32 = 1080;
pub const PATCH_RADIUS: u32 = 4;
pub const SEARCH_RADIUS: u32 = 2;
pub const Q_X: i32 = 1;
pub const Q_Y: i32 = 0;
pub const BILATERAL_SIGMA_S: f32 = 3.0;
pub const BILATERAL_SIGMA_R: f32 = 0.02;
pub const BLOCK_1D: u32 = 256;
pub const COPY_GRID_1D: u32 = 1024;

/// (logical channels, label).
pub const CHANNELS: &[(u32, &str)] = &[(1, "luma"), (2, "chroma"), (3, "yuv")];

pub fn stored_channels(ch: u32) -> u32 {
    match ch {
        1 => 1,
        2 => 2,
        _ => 4,
    }
}

pub fn make_synthetic_frame(w: u32, h: u32, ch: u32) -> Vec<f32> {
    let mut data = Vec::with_capacity((w * h * ch) as usize);
    for y in 0..h {
        for x in 0..w {
            let base = 0.5 + 0.2 * (x as f32 * 0.05).sin() * (y as f32 * 0.03).cos();
            for c in 0..ch {
                let seed = (y * w + x) * ch + c;
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

/// Pad to next-pow2 lane count (matches `NlmDenoiser` internal storage).
pub fn make_padded_frame(w: u32, h: u32, ch: u32) -> Vec<f32> {
    let stored = stored_channels(ch);
    if stored == ch {
        return make_synthetic_frame(w, h, ch);
    }
    let src = make_synthetic_frame(w, h, ch);
    let mut data = vec![0.0f32; (w * h * stored) as usize];
    for i in 0..(w * h) as usize {
        for c in 0..ch as usize {
            data[i * stored as usize + c] = src[i * ch as usize + c];
        }
    }
    data
}

/// Welsch coefficient for the bench-default parameter set. Channel mode
/// is irrelevant here; `NlmParams::h2_inv_norm` only reads `patch_radius`
/// and `strength`.
pub fn h2_inv_norm() -> f32 {
    NlmParams {
        patch_radius: PATCH_RADIUS,
        ..NlmParams::default()
    }
    .h2_inv_norm()
}

pub fn cube_count_2d() -> CubeCount {
    CubeCount::new_2d(W.div_ceil(BLOCK_X), H.div_ceil(BLOCK_Y))
}

pub fn cube_dim_2d() -> CubeDim {
    CubeDim::new_2d(BLOCK_X, BLOCK_Y)
}

pub fn block_sync<R: Runtime>(client: &ComputeClient<R>) {
    cubecl::future::block_on(client.sync()).unwrap();
}

pub fn shapes_with_ch(ch: u32) -> Vec<Vec<usize>> {
    vec![vec![W as usize, H as usize, ch as usize]]
}

/// Shared input shape for kernels that take one framebuffer and write
/// one output of the same logical size (`dist_2d_weight`, its `_ref`
/// twin, and `bilateral`).
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
    println!("  {}", "-".repeat(NAME_WIDTH + 6 + 12 * 5));
}

pub fn run<B: Benchmark>(bench: B) {
    let name = bench.name();
    match bench.run(TimingMethod::Device) {
        Ok(durations) => {
            let c = BenchmarkComputations::new(&durations);
            let mean_s = c.mean.as_secs_f64();
            let fps = if mean_s > 0.0 { 1.0 / mean_s } else { 0.0 };
            println!(
                "  {:<NAME_WIDTH$}  {:>5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10.2}",
                name,
                durations.durations.len(),
                fmt_us(c.mean),
                fmt_us(c.median),
                fmt_us(c.min),
                fmt_us(c.max),
                fps,
            );
        },
        Err(err) => println!("  {name:<NAME_WIDTH$}  error: {err}"),
    }
}

fn fmt_us(d: core::time::Duration) -> String {
    let us = d.as_secs_f64() * 1_000_000.0;
    if us >= 1000.0 {
        format!("{:.3} ms", us / 1000.0)
    } else {
        format!("{us:.2} µs")
    }
}

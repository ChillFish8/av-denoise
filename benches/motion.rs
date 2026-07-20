use std::hint::black_box;
use std::time::{Duration, Instant};

use av_denoise::nlmeans::{
    ChannelMode,
    MotionCompensationMode,
    NlmDenoiser,
    NlmParams,
    Pending,
    PrefilterMode,
};
use cubecl::prelude::*;

#[allow(dead_code)]
#[path = "kernels/mod.rs"]
mod kernels;

use kernels::mc_block_match_coarse::BlockMatchCoarseBench;
use kernels::mc_block_match_fine::BlockMatchFineBench;
use kernels::mc_downscale::DownscaleBench;
use kernels::mc_warp::WarpBench;
use kernels::{CHANNELS, print_header, run};

const W: u32 = 1920;
const H: u32 = 1080;

const WARMUP_PIPELINE: usize = 2;
const ITERS_PIPELINE: usize = 200;

fn make_synthetic_frame(w: u32, h: u32, ch: u32) -> Vec<f32> {
    kernels::make_synthetic_frame(w, h, ch)
}

struct BenchResult {
    name: String,
    backend: String,
    iterations: usize,
    fps: f64,
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

impl BenchResult {
    fn print(&self) {
        println!(
            "[{:<7}] {:<58} {:>4} iters  {:>9.2} fps  {:>7.2} ms/frame  \
             (min: {:>6.2}, max: {:>6.2})",
            self.backend, self.name, self.iterations, self.fps, self.mean_ms, self.min_ms, self.max_ms,
        );
    }
}

fn run_pipeline_bench<R: Runtime>(
    name: &str,
    backend: &str,
    client: &ComputeClient<R>,
    warmup: usize,
    iterations: usize,
    mut f: impl FnMut(),
) -> BenchResult {
    for _ in 0..warmup {
        f();
        futures::executor::block_on(client.sync()).unwrap();
    }

    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        f();
        futures::executor::block_on(client.sync()).unwrap();
        times.push(start.elapsed());
    }

    let total: Duration = times.iter().sum();
    let min = times.iter().min().unwrap();
    let max = times.iter().max().unwrap();
    let mean = total / iterations as u32;
    let fps = iterations as f64 / total.as_secs_f64();

    BenchResult {
        name: name.to_string(),
        backend: backend.to_string(),
        iterations,
        fps,
        mean_ms: mean.as_secs_f64() * 1000.0,
        min_ms: min.as_secs_f64() * 1000.0,
        max_ms: max.as_secs_f64() * 1000.0,
    }
}

const TEMPORAL_RADIUS: u32 = 1;

fn temporal_params(channels: ChannelMode, mc: MotionCompensationMode) -> NlmParams {
    NlmParams {
        temporal_radius: TEMPORAL_RADIUS,
        search_radius: 2,
        patch_radius: 4,
        strength: 1.2,
        self_weight: 1.0,
        channels,
        prefilter: PrefilterMode::None,
        motion_compensation: mc,
        hq: None,
    }
}

fn mc_default() -> MotionCompensationMode {
    MotionCompensationMode::Mvtools {
        blksize: 16,
        overlap: 8,
        search_radius: 4,
        pyramid_levels: 2,
    }
}

/// Eager temporal pipeline (push → denoise → wait inline). Submits
/// one frame and blocks on its readback before the next push, so the
/// per-frame number is the full critical-path cost.
fn bench_eager<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    channels: ChannelMode,
    ch_name: &str,
    mc: MotionCompensationMode,
    tag: &str,
) -> BenchResult {
    let ch = channels.count();
    let params = temporal_params(channels, mc);
    let frame = make_synthetic_frame(W, H, ch);
    let total_frames = 1 + 2 * params.temporal_radius as usize;
    let name = format!("denoise_temporal{tag}_1080p_{ch_name}");

    let mut denoiser = NlmDenoiser::<R>::new(client, params, W, H);
    for _ in 0..total_frames - 1 {
        denoiser.push_frame(&frame);
    }
    futures::executor::block_on(client.sync()).unwrap();

    run_pipeline_bench(&name, backend, client, WARMUP_PIPELINE, ITERS_PIPELINE, || {
        denoiser.push_frame(&frame);
        let result = denoiser.denoise().unwrap().unwrap();
        black_box(&result);
    })
}

/// Pipelined variant: kernels for frame N+1 are submitted before the
/// readback for frame N completes, so GPU and host overlap. Mirrors
/// the equivalent helper in `benches/nlmeans.rs`.
fn bench_pipelined<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    channels: ChannelMode,
    ch_name: &str,
    mc: MotionCompensationMode,
    tag: &str,
) -> BenchResult {
    let ch = channels.count();
    let params = temporal_params(channels, mc);
    let frame = make_synthetic_frame(W, H, ch);
    let total_frames = 1 + 2 * params.temporal_radius as usize;
    let name = format!("denoise_temporal_pipelined{tag}_1080p_{ch_name}");

    let mut denoiser = NlmDenoiser::<R>::new(client, params, W, H);
    for _ in 0..total_frames - 1 {
        denoiser.push_frame(&frame);
    }
    futures::executor::block_on(client.sync()).unwrap();

    denoiser.push_frame(&frame);
    let mut in_flight: Option<Pending<R>> = Some(denoiser.denoise_submit().unwrap().unwrap());

    let result = run_pipeline_bench(&name, backend, client, WARMUP_PIPELINE, ITERS_PIPELINE, || {
        denoiser.push_frame(&frame);
        let next = denoiser.denoise_submit().unwrap().unwrap();
        let output = in_flight.take().unwrap().wait().unwrap();
        black_box(&output);
        in_flight = Some(next);
    });

    if let Some(pending) = in_flight.take() {
        let _ = pending.wait().unwrap();
    }
    result
}

fn run_kernels<R: Runtime>(backend: &str, client: &ComputeClient<R>) {
    println!();
    println!("--- {backend}: MC kernels ---");
    print_header();

    run(DownscaleBench {
        client: client.clone(),
    });
    run(BlockMatchCoarseBench {
        client: client.clone(),
    });
    run(BlockMatchFineBench {
        client: client.clone(),
    });
    for &(ch, ch_name) in CHANNELS {
        run(WarpBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }
}

/// For each channel mode, print the four temporal-pipeline rows
/// (with vs without MC, eager vs pipelined) adjacent so the cost
/// delta is visible at a glance.
fn run_pipelines<R: Runtime>(backend: &str, client: &ComputeClient<R>) {
    println!();
    println!("--- {backend}: temporal pipeline (with vs without MC) ---");

    let variants: &[(MotionCompensationMode, &str)] =
        &[(MotionCompensationMode::None, "_no_mc"), (mc_default(), "_mc")];
    let channels = [
        ("luma", ChannelMode::Luma),
        ("chroma", ChannelMode::Chroma),
        ("yuv", ChannelMode::Yuv),
    ];

    for &(ch_name, mode) in &channels {
        for &(mc, tag) in variants {
            bench_eager::<R>(client, backend, mode, ch_name, mc, tag).print();
            bench_pipelined::<R>(client, backend, mode, ch_name, mc, tag).print();
        }
    }
    println!();
}

fn run_all<R: Runtime>(backend: &str, device: &R::Device) {
    let client = R::client(device);
    run_kernels::<R>(backend, &client);
    run_pipelines::<R>(backend, &client);
}

#[derive(clap::Parser, Debug)]
#[command(about = "Motion-compensation benches: per-kernel + end-to-end pipeline", long_about = None)]
struct Cli {
    /// GPU device to bind to. Format: `default`, `discrete[:N]`,
    /// `integrated[:N]`, `virtual[:N]`, or `cpu`.
    #[arg(long, default_value = "default")]
    device: av_denoise::Device,

    /// Swallowed: cargo passes this when invoking the bench binary.
    #[arg(long, hide = true)]
    bench: bool,
}

fn main() {
    use clap::Parser;
    let cli = Cli::parse();

    println!("Motion-Compensation Benchmarks - 1920x1080");
    println!("  pipeline: warmup={WARMUP_PIPELINE}, timed={ITERS_PIPELINE}");

    #[cfg(feature = "vulkan")]
    {
        let device = cli.device.to_wgpu().expect("wgpu device conversion failed");
        println!("  device:   {device:?}");
        run_all::<cubecl::wgpu::WgpuRuntime>("vulkan", &device);
    }

    #[cfg(not(feature = "vulkan"))]
    {
        let _ = cli;
        eprintln!("No GPU backend enabled. Run with --features vulkan");
        std::process::exit(1);
    }
}

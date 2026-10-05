#[expect(
    dead_code,
    reason = "the shared kernel module is included by several bench binaries, each of which uses \
              only part of it"
)]
#[path = "kernels/mod.rs"]
mod kernels;

use std::hint::black_box;
use std::time::{Duration, Instant};

use av_denoise_core::bench_api::{Device, HostIo, NlmDenoiser, NlmParams, start_read, wait_read};
use av_denoise_core::{ChannelMode, MotionCompensationMode, MotionEstimation, PrefilterMode};
use clap::Parser;
use cubecl::prelude::*;
use kernels::mc_block_match_coarse::BlockMatchCoarseBench;
use kernels::mc_block_match_fine::BlockMatchFineBench;
use kernels::mc_confidence::McConfidenceBench;
use kernels::mc_downscale::DownscaleBench;
use kernels::mc_warp::WarpBench;
use kernels::mv_regularise::MvRegulariseBench;
use kernels::{CHANNELS, make_synthetic_frame, print_header, run};

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

const WARMUP_PIPELINE: usize = 2;
const ITERS_PIPELINE: usize = 200;

const TEMPORAL_RADIUS: u32 = 1;

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
    mut step: impl FnMut(),
) -> BenchResult {
    for _ in 0..warmup {
        step();
        let sync = client.sync();
        futures::executor::block_on(sync).unwrap();
    }

    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        step();
        let sync = client.sync();
        futures::executor::block_on(sync).unwrap();
        let elapsed = start.elapsed();
        times.push(elapsed);
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

fn temporal_params(
    radius: u32,
    channels: ChannelMode,
    motion_compensation: MotionCompensationMode,
) -> NlmParams {
    NlmParams {
        temporal_radius: radius,
        search_radius: 2,
        patch_radius: 4,
        strength: 1.2,
        self_weight: 1.0,
        channels,
        prefilter: PrefilterMode::None,
        motion_compensation,
        hq: None,
    }
}

fn mc_default() -> MotionCompensationMode {
    MotionCompensationMode::Mvtools {
        blksize: 16,
        overlap: 8,
        search_radius: 4,
        pyramid_levels: 2,
        estimation: MotionEstimation::Direct,
    }
}

/// [mc_default]'s block geometry with `Chained` estimation at the library's default refinement radius.
fn mc_chained_default() -> MotionCompensationMode {
    MotionCompensationMode::Mvtools {
        blksize: 16,
        overlap: 8,
        search_radius: 4,
        pyramid_levels: 2,
        estimation: MotionEstimation::chained_default(),
    }
}

/// The eager temporal pipeline cost.
///
/// Each frame is pushed, denoised and waited on inline before the next push, so the per-frame
/// number is the full critical-path cost.
fn bench_eager<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    radius: u32,
    channels: ChannelMode,
    channel_name: &str,
    motion_compensation: MotionCompensationMode,
    tag: &str,
) -> BenchResult {
    let channel_count = channels.count();
    let params = temporal_params(radius, channels, motion_compensation);
    let frame = make_synthetic_frame(WIDTH, HEIGHT, channel_count);
    let total_frames = 1 + 2 * params.temporal_radius as usize;
    let name = format!("denoise_temporal{tag}_1080p_{channel_name}");

    let mut denoiser = NlmDenoiser::<R>::new(client, params, WIDTH, HEIGHT);
    for _ in 0..total_frames - 1 {
        denoiser.push_frame(&frame);
    }

    let sync = client.sync();
    futures::executor::block_on(sync).unwrap();

    run_pipeline_bench(&name, backend, client, WARMUP_PIPELINE, ITERS_PIPELINE, || {
        denoiser.push_frame(&frame);
        let result = denoiser.denoise().unwrap().unwrap();
        black_box(&result);
    })
}

/// The pipelined temporal pipeline cost.
///
/// Frame N+1's kernels are submitted before frame N's readback completes, so GPU and host work
/// overlap.
fn bench_pipelined<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    radius: u32,
    channels: ChannelMode,
    channel_name: &str,
    motion_compensation: MotionCompensationMode,
    tag: &str,
) -> BenchResult {
    let channel_count = channels.count();
    let params = temporal_params(radius, channels, motion_compensation);
    let frame = make_synthetic_frame(WIDTH, HEIGHT, channel_count);
    let total_frames = 1 + 2 * params.temporal_radius as usize;
    let name = format!("denoise_temporal_pipelined{tag}_1080p_{channel_name}");

    let mut denoiser = NlmDenoiser::<R>::new(client, params, WIDTH, HEIGHT);
    for _ in 0..total_frames - 1 {
        denoiser.push_frame(&frame);
    }

    let sync = client.sync();
    futures::executor::block_on(sync).unwrap();

    denoiser.push_frame(&frame);
    let first = denoiser.denoise_submit_gpu().unwrap().unwrap();
    let first_read = start_read(client, first.handle);
    let mut in_flight = Some(first_read);

    let result = run_pipeline_bench(&name, backend, client, WARMUP_PIPELINE, ITERS_PIPELINE, || {
        denoiser.push_frame(&frame);
        let next = denoiser.denoise_submit_gpu().unwrap().unwrap();
        let next_read = start_read(client, next.handle);

        let previous = in_flight.take().unwrap();
        let output = wait_read(previous);
        black_box(&output);
        in_flight = Some(next_read);
    });

    if let Some(read) = in_flight.take() {
        let _ = wait_read(read);
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
    run(McConfidenceBench {
        client: client.clone(),
    });
    run(MvRegulariseBench {
        client: client.clone(),
    });

    for &(channels, channel_name) in CHANNELS {
        run(WarpBench {
            client: client.clone(),
            channels,
            channel_name,
        });
    }
}

/// Prints each channel mode's four temporal-pipeline rows side by side.
///
/// The rows cover with and without motion compensation, eager and pipelined, so the cost delta is
/// visible at a glance.
fn run_pipelines<R: Runtime>(backend: &str, client: &ComputeClient<R>) {
    println!();
    println!("--- {backend}: temporal pipeline (with vs without MC) ---");

    let motion_compensation = mc_default();
    let variants: &[(MotionCompensationMode, &str)] = &[
        (MotionCompensationMode::None, "_no_mc"),
        (motion_compensation, "_mc"),
    ];
    let channel_modes = [
        ("luma", ChannelMode::Luma),
        ("chroma", ChannelMode::Chroma),
        ("yuv", ChannelMode::Yuv),
    ];

    for &(channel_name, mode) in &channel_modes {
        for &(variant, tag) in variants {
            let eager = bench_eager::<R>(client, backend, TEMPORAL_RADIUS, mode, channel_name, variant, tag);
            eager.print();

            let pipelined =
                bench_pipelined::<R>(client, backend, TEMPORAL_RADIUS, mode, channel_name, variant, tag);
            pipelined.print();
        }
    }

    println!();
}

/// Direct and chained motion estimation throughput at radii 2 and 4, luma only.
///
/// The cost delta lives entirely in the motion compensation path rather than per-channel weighting,
/// so extra channels would add bench time without adding signal. Eager and pipelined rows for each
/// radius and strategy print side by side.
fn run_mc_estimation_comparison<R: Runtime>(backend: &str, client: &ComputeClient<R>) {
    println!();
    println!("--- {backend}: MC estimation comparison (direct vs chained, r2/r4) ---");

    for &radius in &[2u32, 4u32] {
        let direct = mc_default();
        let chained = mc_chained_default();
        for (variant, label) in [(direct, "direct"), (chained, "chained")] {
            let tag = format!("_mc_r{radius}_{label}");

            let eager = bench_eager::<R>(client, backend, radius, ChannelMode::Luma, "luma", variant, &tag);
            eager.print();

            let pipelined =
                bench_pipelined::<R>(client, backend, radius, ChannelMode::Luma, "luma", variant, &tag);
            pipelined.print();
        }
    }

    println!();
}

fn run_all<R: Runtime>(backend: &str, device: &R::Device) {
    let client = R::client(device);
    run_kernels::<R>(backend, &client);
    run_pipelines::<R>(backend, &client);
    run_mc_estimation_comparison::<R>(backend, &client);
}

#[derive(clap::Parser, Debug)]
#[command(about = "Motion-compensation benches: per-kernel + end-to-end pipeline", long_about = None)]
struct Cli {
    /// GPU device to bind to, one of `default`, `discrete[:N]`, `integrated[:N]`, `virtual[:N]` or `cpu`.
    #[arg(long, default_value = "default")]
    device: Device,

    /// Swallowed, since cargo passes this when invoking the bench binary.
    #[arg(long, hide = true)]
    bench: bool,
}

fn main() {
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

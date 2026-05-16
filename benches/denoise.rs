//! End-to-end benchmark of the public `Denoiser` library API.
//!
//! Builds a [`Denoiser`] for each accelerator that can initialise,
//! then drives a steady-state push/recv loop at 1920×1080 over a few
//! channel-mode × denoising-mode configurations and reports throughput.

use std::time::{Duration, Instant};

use av_denoise::accelerate::Accelerator;
use av_denoise::{ChannelMode, Denoiser, DenoiserOptions, DenoisingMode, Device, PrefilterMode};

const W: u32 = 1920;
const H: u32 = 1080;

const WARMUP: usize = 5;
const ITERS: usize = 100;

const BILATERAL_SIGMA_S: f32 = 3.0;
const BILATERAL_SIGMA_R: f32 = 0.02;

#[derive(clap::Parser, Debug)]
#[command(about = "End-to-end Denoiser benchmark", long_about = None)]
struct Cli {
    /// GPU device to bind to. Format: `default`, `discrete[:N]`,
    /// `integrated[:N]`, `virtual[:N]`, or `cpu`.
    #[arg(long, default_value = "default")]
    device: Device,

    /// Accelerator priority list (comma-delimited). Defaults to all
    /// compiled-in accelerators.
    #[arg(long, value_delimiter = ',', default_values_t = av_denoise::accelerate::get_default_accelerators())]
    accelerators: Vec<Accelerator>,

    /// Swallowed: cargo passes this when invoking the bench binary.
    #[arg(long, hide = true)]
    bench: bool,
}

fn make_synthetic_frame(w: u32, h: u32, ch: u32) -> Vec<f32> {
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

struct BenchResult {
    name: String,
    accelerator: Accelerator,
    iterations: usize,
    fps: f64,
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

impl BenchResult {
    fn print(&self) {
        println!(
            "[{:<8?}] {:<48} {:>4} iters  {:>9.2} fps  {:>7.2} ms/frame  \
             (min: {:>6.2}, max: {:>6.2})",
            self.accelerator, self.name, self.iterations, self.fps, self.mean_ms, self.min_ms, self.max_ms,
        );
    }
}

fn options(channel_mode: ChannelMode, mode: DenoisingMode, prefilter: PrefilterMode) -> DenoiserOptions {
    DenoiserOptions::builder()
        .channel_mode(channel_mode)
        .mode(mode)
        .prefilter(prefilter)
        .build()
}

fn bench_push_recv(
    name: &str,
    accelerators: &[Accelerator],
    device: &Device,
    channel_mode: ChannelMode,
    mode: DenoisingMode,
    prefilter: PrefilterMode,
) -> Result<BenchResult, anyhow::Error> {
    let ch = channel_mode.count();
    let frame = make_synthetic_frame(W, H, ch);

    let mut denoiser = Denoiser::new(accelerators, device, W, H, options(channel_mode, mode, prefilter))?;
    let accelerator = denoiser.selected_accelerator();

    // Fill the temporal window so subsequent push/recv steady-state lines up.
    let temporal_radius = match mode {
        DenoisingMode::Spacial => 0,
        DenoisingMode::Temporal { radius } => radius,
    };
    let window = 2 * temporal_radius + 1;
    for _ in 0..window.saturating_sub(1) {
        denoiser.push_frame(&frame)?;
    }

    for _ in 0..WARMUP {
        denoiser.push_frame(&frame)?;
        let _ = denoiser.recv_frame()?;
    }

    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        denoiser.push_frame(&frame)?;
        let _out = denoiser.recv_frame()?;
        times.push(start.elapsed());
    }

    let total: Duration = times.iter().sum();
    let min = times.iter().min().copied().unwrap_or_default();
    let max = times.iter().max().copied().unwrap_or_default();
    let mean = total / ITERS as u32;
    let fps = ITERS as f64 / total.as_secs_f64();

    Ok(BenchResult {
        name: name.to_string(),
        accelerator,
        iterations: ITERS,
        fps,
        mean_ms: mean.as_secs_f64() * 1000.0,
        min_ms: min.as_secs_f64() * 1000.0,
        max_ms: max.as_secs_f64() * 1000.0,
    })
}

fn main() {
    use clap::Parser;
    let cli = Cli::parse();

    println!("Denoiser E2E Benchmarks - {W}×{H}");
    println!("  warmup={WARMUP}, timed={ITERS}");
    println!("  device:        {:?}", cli.device);
    println!("  accelerators:  {:?}", cli.accelerators);
    println!();

    let bilateral = PrefilterMode::Bilateral {
        sigma_s: BILATERAL_SIGMA_S,
        sigma_r: BILATERAL_SIGMA_R,
    };

    let configs: &[(&str, ChannelMode, DenoisingMode, PrefilterMode)] = &[
        (
            "spatial_luma",
            ChannelMode::Luma,
            DenoisingMode::Spacial,
            PrefilterMode::None,
        ),
        (
            "spatial_chroma",
            ChannelMode::Chroma,
            DenoisingMode::Spacial,
            PrefilterMode::None,
        ),
        (
            "spatial_yuv",
            ChannelMode::Yuv,
            DenoisingMode::Spacial,
            PrefilterMode::None,
        ),
        (
            "temporal_r1_yuv",
            ChannelMode::Yuv,
            DenoisingMode::Temporal { radius: 1 },
            PrefilterMode::None,
        ),
        (
            "temporal_r2_yuv",
            ChannelMode::Yuv,
            DenoisingMode::Temporal { radius: 2 },
            PrefilterMode::None,
        ),
        (
            "spatial_luma+bilateral",
            ChannelMode::Luma,
            DenoisingMode::Spacial,
            bilateral,
        ),
        (
            "spatial_chroma+bilateral",
            ChannelMode::Chroma,
            DenoisingMode::Spacial,
            bilateral,
        ),
        (
            "spatial_yuv+bilateral",
            ChannelMode::Yuv,
            DenoisingMode::Spacial,
            bilateral,
        ),
        (
            "temporal_r1_yuv+bilateral",
            ChannelMode::Yuv,
            DenoisingMode::Temporal { radius: 1 },
            bilateral,
        ),
        (
            "temporal_r2_yuv+bilateral",
            ChannelMode::Yuv,
            DenoisingMode::Temporal { radius: 2 },
            bilateral,
        ),
    ];

    for (name, ch, mode, prefilter) in configs {
        match bench_push_recv(name, &cli.accelerators, &cli.device, *ch, *mode, *prefilter) {
            Ok(result) => result.print(),
            Err(err) => eprintln!("[{name}] failed: {err:?}"),
        }
    }
}

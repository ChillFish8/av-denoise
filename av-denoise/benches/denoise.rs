use std::time::{Duration, Instant};

use av_denoise::accelerate::Accelerator;
use av_denoise::{
    Algorithm,
    ChannelMode,
    DenoiserError,
    DenoiserOptions,
    DenoisingMode,
    Device,
    HostDenoiser,
    MotionCompensationMode,
    NlmeansOptions,
    PrefilterMode,
};
use clap::Parser;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

const WARMUP: usize = 5;
const ITERS: usize = 100;

const BILATERAL_SIGMA_S: f32 = 3.0;
const BILATERAL_SIGMA_R: f32 = 0.02;

#[derive(clap::Parser, Debug)]
#[command(about = "End-to-end Denoiser benchmark", long_about = None)]
struct Cli {
    /// GPU device to bind to, one of `default`, `discrete[:N]`, `integrated[:N]`, `virtual[:N]` or `cpu`.
    #[arg(long, default_value = "default")]
    device: Device,

    /// Accelerator priority list, comma-delimited. Defaults to all compiled-in accelerators.
    #[arg(long, value_delimiter = ',', default_values_t = av_denoise::accelerate::get_default_accelerators())]
    accelerators: Vec<Accelerator>,

    /// Swallowed, since cargo passes this when invoking the bench binary.
    #[arg(long, hide = true)]
    bench: bool,
}

/// One 8-bit plane per channel, a smooth pattern plus hashed noise.
fn make_synthetic_planes(width: u32, height: u32, channels: u32) -> Vec<Vec<u8>> {
    let mut planes = vec![Vec::with_capacity((width * height) as usize); channels as usize];

    for y in 0..height {
        for x in 0..width {
            let base = 0.5 + 0.2 * (x as f32 * 0.05).sin() * (y as f32 * 0.03).cos();

            for (channel, plane) in planes.iter_mut().enumerate() {
                let seed = (y * width + x) * channels + channel as u32;
                let hash = seed
                    .wrapping_mul(2654435761)
                    .wrapping_add(seed.wrapping_mul(340573321));
                let noise = (hash as f32 / u32::MAX as f32 - 0.5) * 0.1;
                let value = (base + noise).clamp(0.0, 1.0);
                plane.push((value * 255.0 + 0.5) as u8);
            }
        }
    }

    planes
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

fn options(channel_mode: ChannelMode, mode: DenoisingMode, algorithm: Algorithm) -> DenoiserOptions {
    DenoiserOptions::builder()
        .channel_mode(channel_mode)
        .mode(mode)
        .algorithm(algorithm)
        .build()
}

/// The fast NLM path with a prefilter and a motion-compensation mode.
fn nlm(prefilter: PrefilterMode, motion_compensation: MotionCompensationMode) -> Algorithm {
    let nlmeans_options = NlmeansOptions {
        prefilter,
        motion_compensation,
        ..NlmeansOptions::default()
    };
    Algorithm::Nlmeans(nlmeans_options)
}

fn bench_push_recv(
    name: &str,
    accelerators: &[Accelerator],
    device: &Device,
    channel_mode: ChannelMode,
    mode: DenoisingMode,
    algorithm: Algorithm,
) -> Result<BenchResult, anyhow::Error> {
    let channels = channel_mode.count();
    let planes = make_synthetic_planes(WIDTH, HEIGHT, channels);
    let frame: Vec<&[u8]> = planes.iter().map(Vec::as_slice).collect();

    let denoiser_options = options(channel_mode, mode, algorithm);
    let mut denoiser = HostDenoiser::create(accelerators, device, WIDTH, HEIGHT, denoiser_options)?;
    let accelerator = denoiser.selected_accelerator();

    // Fill the temporal window so the steady-state push/recv lines up. NLMeans mirrors the first
    // frame into the leading ring slots and emits early, so pushing `window - 1` frames can hit
    // `QueueFull` at radius 2 or more. A full queue drains one frame before the push is retried.
    let temporal_radius = match mode {
        DenoisingMode::Spacial => 0,
        DenoisingMode::Temporal { radius } => radius,
    };
    let window = 2 * temporal_radius + 1;
    for _ in 0..window.saturating_sub(1) {
        if let Err(DenoiserError::QueueFull) = denoiser.push(&frame) {
            let _ = denoiser.recv()?;
            denoiser.push(&frame)?;
        }
    }

    while denoiser.recv()?.is_some() {}

    for _ in 0..WARMUP {
        denoiser.push(&frame)?;
        let _ = denoiser.recv()?;
    }

    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        denoiser.push(&frame)?;
        let _received = denoiser.recv()?;
        times.push(start.elapsed());
    }

    // Drain the temporal tail so every pushed frame is accounted for. An unpolled `Pending` is free
    // to drop, so this is bookkeeping rather than a safety requirement.
    denoiser.flush(|_| {})?;

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
    // SAFETY: single-threaded at entry, no race possible.
    unsafe { av_denoise::raise_codegen_stack_limit() };

    let cli = Cli::parse();

    println!("Denoiser E2E Benchmarks - {WIDTH}×{HEIGHT}");
    println!("  warmup={WARMUP}, timed={ITERS}");
    println!("  device:        {:?}", cli.device);
    println!("  accelerators:  {:?}", cli.accelerators);
    println!();

    let bilateral = PrefilterMode::Bilateral {
        sigma_s: BILATERAL_SIGMA_S,
        sigma_r: BILATERAL_SIGMA_R,
    };
    let motion_compensation = MotionCompensationMode::mvtools_default();

    let plain = nlm(PrefilterMode::None, MotionCompensationMode::None);
    let plain_mc = nlm(PrefilterMode::None, motion_compensation);
    let bilateral_only = nlm(bilateral, MotionCompensationMode::None);
    let bilateral_mc = nlm(bilateral, motion_compensation);

    // Each temporal config is followed by its motion-compensation variant, so the cost of
    // `--motion-compensation` shows on adjacent rows.
    let configs: &[(&str, ChannelMode, DenoisingMode, Algorithm)] = &[
        ("spatial_luma", ChannelMode::Luma, DenoisingMode::Spacial, plain),
        (
            "spatial_chroma",
            ChannelMode::Chroma,
            DenoisingMode::Spacial,
            plain,
        ),
        ("spatial_yuv", ChannelMode::Yuv, DenoisingMode::Spacial, plain),
        (
            "temporal_r1_yuv",
            ChannelMode::Yuv,
            DenoisingMode::Temporal { radius: 1 },
            plain,
        ),
        (
            "temporal_r1_yuv+mc",
            ChannelMode::Yuv,
            DenoisingMode::Temporal { radius: 1 },
            plain_mc,
        ),
        (
            "temporal_r2_yuv",
            ChannelMode::Yuv,
            DenoisingMode::Temporal { radius: 2 },
            plain,
        ),
        (
            "temporal_r2_yuv+mc",
            ChannelMode::Yuv,
            DenoisingMode::Temporal { radius: 2 },
            plain_mc,
        ),
        (
            "spatial_luma+bilateral",
            ChannelMode::Luma,
            DenoisingMode::Spacial,
            bilateral_only,
        ),
        (
            "spatial_chroma+bilateral",
            ChannelMode::Chroma,
            DenoisingMode::Spacial,
            bilateral_only,
        ),
        (
            "spatial_yuv+bilateral",
            ChannelMode::Yuv,
            DenoisingMode::Spacial,
            bilateral_only,
        ),
        (
            "temporal_r1_yuv+bilateral",
            ChannelMode::Yuv,
            DenoisingMode::Temporal { radius: 1 },
            bilateral_only,
        ),
        (
            "temporal_r1_yuv+bilateral+mc",
            ChannelMode::Yuv,
            DenoisingMode::Temporal { radius: 1 },
            bilateral_mc,
        ),
        (
            "temporal_r2_yuv+bilateral",
            ChannelMode::Yuv,
            DenoisingMode::Temporal { radius: 2 },
            bilateral_only,
        ),
        (
            "temporal_r2_yuv+bilateral+mc",
            ChannelMode::Yuv,
            DenoisingMode::Temporal { radius: 2 },
            bilateral_mc,
        ),
    ];

    for (name, channel_mode, mode, algorithm) in configs {
        let outcome = bench_push_recv(
            name,
            &cli.accelerators,
            &cli.device,
            *channel_mode,
            *mode,
            *algorithm,
        );
        match outcome {
            Ok(result) => result.print(),
            Err(error) => eprintln!("[{name}] failed: {error:?}"),
        }
    }
}

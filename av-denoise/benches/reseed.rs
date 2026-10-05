use std::time::{Duration, Instant};

use av_denoise::accelerate::Accelerator;
use av_denoise::{
    Algorithm,
    ChannelIntent,
    ChannelMode,
    DenoiserOptions,
    DenoisingMode,
    Depth,
    Device,
    FrameLayout,
    HostDenoiser,
    PlanarDenoiser,
    PlaneOptions,
    Planes,
    Subsampling,
    push_needs_retry,
};
use clap::Parser;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const RADIUS: u32 = 2;

const WARMUP: usize = 5;
const ITERS: usize = 100;

#[derive(clap::Parser, Debug)]
#[command(about = "Cost of a reseed relative to a sequential frame", long_about = None)]
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

fn layout() -> FrameLayout {
    FrameLayout {
        width: WIDTH,
        height: HEIGHT,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Eight,
    }
}

fn plane_options(accelerators: &[Accelerator], device: &Device) -> PlaneOptions {
    PlaneOptions {
        accelerators: accelerators.to_vec(),
        device: device.clone(),
        intent: ChannelIntent::LumaChroma,
        mode: DenoisingMode::Temporal { radius: RADIUS },
        algorithm: Algorithm::default(),
        luma_strength: None,
        chroma_strength: None,
        luma_lambda_ht: None,
        chroma_lambda_ht: None,
    }
}

/// A small xorshift generator, so the synthetic clip is the same on every run.
fn pseudo_random(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

/// One plane's wire bytes for frame `frame_index`.
///
/// A spatial ramp plus a per-frame offset and a deterministic dither give a temporal filter real
/// signal and real noise to work with.
fn ramp_plane(pixels: usize, width: u32, frame_index: usize, plane_seed: u64) -> Vec<u8> {
    let width = width.max(1) as usize;

    (0..pixels)
        .map(|i| {
            let x = (i % width) as u32;
            let y = (i / width) as u32;
            let spatial = x.wrapping_add(y) % 120;
            let frame_offset = (frame_index as u32 * 7) % 60;
            let seed = (i as u64) ^ (frame_index as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ plane_seed;
            let dither = (pseudo_random(seed) % 16) as u32;
            let value = 20 + spatial + frame_offset + dither;
            value.min(235) as u8
        })
        .collect()
}

fn make_planes(layout: &FrameLayout, frame_index: usize) -> Planes {
    let (chroma_width, _) = layout.chroma_dims();
    let y_plane = ramp_plane(layout.luma_pixels(), layout.width, frame_index, 1);
    let u_plane = ramp_plane(layout.chroma_pixels(), chroma_width, frame_index, 2);
    let v_plane = ramp_plane(layout.chroma_pixels(), chroma_width, frame_index, 3);

    Planes {
        y: y_plane,
        u: u_plane,
        v: v_plane,
    }
}

/// `count` frames with distinct content, so sliding windows are cut from it without regenerating
/// frames per call.
fn make_clip(layout: &FrameLayout, count: usize) -> Vec<Planes> {
    (0..count).map(|i| make_planes(layout, i)).collect()
}

/// The accelerator a real denoiser would pick, read from a throwaway probe.
///
/// `PlanarDenoiser` may own up to three inner denoisers and exposes no single accelerator getter.
fn selected_accelerator(accelerators: &[Accelerator], device: &Device) -> Result<Accelerator, anyhow::Error> {
    let probe_options = DenoiserOptions::builder()
        .channel_mode(ChannelMode::Luma)
        .mode(DenoisingMode::Spacial)
        .algorithm(Algorithm::default())
        .build();
    let probe = HostDenoiser::create(accelerators, device, 4, 4, probe_options)?;
    Ok(probe.selected_accelerator())
}

/// The `2r+1` frames centred on `clip[centre]`.
fn window_at(clip: &[Planes], centre: usize, radius: usize) -> Vec<Planes> {
    (0..(2 * radius + 1))
        .map(|i| clip[centre + i - radius].clone())
        .collect()
}

struct BenchResult {
    name: String,
    accelerator: Accelerator,
    iterations: usize,
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

impl BenchResult {
    fn print(&self) {
        println!(
            "[{:<8?}] {:<12} {:>4} iters  {:>9.3} ms/frame  (min: {:>7.3}, max: {:>7.3})",
            self.accelerator, self.name, self.iterations, self.mean_ms, self.min_ms, self.max_ms,
        );
    }
}

fn summarise(name: &str, accelerator: Accelerator, times: &[Duration]) -> BenchResult {
    let total: Duration = times.iter().sum();
    let min = times.iter().min().copied().unwrap_or_default();
    let max = times.iter().max().copied().unwrap_or_default();
    let mean = total / times.len().max(1) as u32;

    BenchResult {
        name: name.to_string(),
        accelerator,
        iterations: times.len(),
        mean_ms: mean.as_secs_f64() * 1000.0,
        min_ms: min.as_secs_f64() * 1000.0,
        max_ms: max.as_secs_f64() * 1000.0,
    }
}

/// Times `push` plus `recv` per output frame once the window is primed and the stream is steady.
fn bench_sequential(accelerators: &[Accelerator], device: &Device) -> Result<BenchResult, anyhow::Error> {
    let layout = layout();
    let options = plane_options(accelerators, device);
    let mut denoiser = PlanarDenoiser::create(&options, layout)?;
    let accelerator = selected_accelerator(accelerators, device)?;

    let radius = denoiser.temporal_radius();
    let window = 2 * radius + 1;
    let clip = make_clip(&layout, window as usize + WARMUP + ITERS);

    // Priming happens outside the timed region, so from here on every push has one recv.
    for frame in &clip[..window.saturating_sub(1) as usize] {
        let pushed = denoiser.push(frame);
        if push_needs_retry(pushed)? {
            let _ = denoiser.recv()?;
            denoiser.push(frame)?;
        }
    }

    while denoiser.recv()?.is_some() {}

    let mut next_frame = window.saturating_sub(1) as usize;

    for _ in 0..WARMUP {
        denoiser.push(&clip[next_frame])?;
        let _ = denoiser.recv()?;
        next_frame += 1;
    }

    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        denoiser.push(&clip[next_frame])?;
        let _received = denoiser.recv()?;
        times.push(start.elapsed());
        next_frame += 1;
    }

    // Drain the trailing temporal frames so every pushed frame is accounted for. An unpolled
    // `Pending` is free to drop, so this is bookkeeping rather than a safety requirement.
    denoiser.flush(|_| {})?;

    let result = summarise("sequential", accelerator, &times);
    Ok(result)
}

/// Times one `reseed` call over a fresh window per iteration.
///
/// The denoiser and every window are built before timing starts, so only `reseed` is on the clock.
fn bench_reseed(accelerators: &[Accelerator], device: &Device) -> Result<BenchResult, anyhow::Error> {
    let layout = layout();
    let options = plane_options(accelerators, device);
    let mut denoiser = PlanarDenoiser::create(&options, layout)?;
    let accelerator = selected_accelerator(accelerators, device)?;

    let radius = denoiser.temporal_radius() as usize;
    let clip = make_clip(&layout, 2 * radius + WARMUP + ITERS);

    let windows: Vec<Vec<Planes>> = (0..(WARMUP + ITERS))
        .map(|i| window_at(&clip, radius + i, radius))
        .collect();

    for window in &windows[..WARMUP] {
        denoiser.reseed(window)?;
    }

    let mut times = Vec::with_capacity(ITERS);
    for window in &windows[WARMUP..] {
        let start = Instant::now();
        let _reseeded = denoiser.reseed(window)?;
        times.push(start.elapsed());
    }

    denoiser.flush(|_| {})?;

    let result = summarise("reseed", accelerator, &times);
    Ok(result)
}

fn main() {
    // SAFETY: single-threaded at entry, no race possible.
    unsafe { av_denoise::raise_codegen_stack_limit() };

    let cli = Cli::parse();

    println!("Reseed cost benchmark - {WIDTH}×{HEIGHT}, temporal radius {RADIUS}");
    println!("  warmup={WARMUP}, timed={ITERS}");
    println!("  device:        {:?}", cli.device);
    println!("  accelerators:  {:?}", cli.accelerators);
    println!();

    let sequential_outcome = bench_sequential(&cli.accelerators, &cli.device);
    let sequential = match sequential_outcome {
        Ok(result) => result,
        Err(error) => {
            eprintln!("[sequential] failed: {error:?}");
            return;
        },
    };
    sequential.print();

    let reseed_outcome = bench_reseed(&cli.accelerators, &cli.device);
    let reseed = match reseed_outcome {
        Ok(result) => result,
        Err(error) => {
            eprintln!("[reseed] failed: {error:?}");
            return;
        },
    };
    reseed.print();

    let ratio = reseed.mean_ms / sequential.mean_ms;
    println!();
    println!("reseed / sequential ratio: {ratio:.2}x");
}

use std::time::{Duration, Instant};

use av_denoise_core::accelerate::Accelerator;
use av_denoise_core::{
    Algorithm,
    ChannelIntent,
    ChannelMode,
    Denoiser,
    DenoiserOptions,
    DenoisingMode,
    Depth,
    Device,
    FrameLayout,
    PlanarDenoiser,
    PlaneOptions,
    Planes,
    Subsampling,
    push_needs_retry,
};

const W: u32 = 1920;
const H: u32 = 1080;
const RADIUS: u32 = 2;

const WARMUP: usize = 5;
const ITERS: usize = 100;

#[derive(clap::Parser, Debug)]
#[command(about = "Cost of a reseed relative to a sequential frame", long_about = None)]
struct Cli {
    /// GPU device to bind to. Format: `default`, `discrete[:N]`,
    /// `integrated[:N]`, `virtual[:N]`, or `cpu`.
    #[arg(long, default_value = "default")]
    device: Device,

    /// Accelerator priority list (comma-delimited). Defaults to all
    /// compiled-in accelerators.
    #[arg(long, value_delimiter = ',', default_values_t = av_denoise_core::accelerate::get_default_accelerators())]
    accelerators: Vec<Accelerator>,

    /// Swallowed: cargo passes this when invoking the bench binary.
    #[arg(long, hide = true)]
    bench: bool,
}

fn layout() -> FrameLayout {
    FrameLayout {
        width: W,
        height: H,
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
        luma_mismatch_scale: None,
        chroma_mismatch_scale: None,
    }
}

/// A small xorshift generator, deterministic across runs so the synthetic
/// clip does not vary between executions.
fn pseudo_random(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// One plane's wire bytes for frame `frame_idx`: a spatial ramp across the
/// plane plus a per-frame offset and a deterministic dither, so a temporal
/// filter sees real signal and real noise to work with.
fn ramp_plane(pixels: usize, width: u32, frame_idx: usize, plane_seed: u64) -> Vec<u8> {
    let width = width.max(1) as usize;

    (0..pixels)
        .map(|i| {
            let x = (i % width) as u32;
            let y = (i / width) as u32;
            let spatial = x.wrapping_add(y) % 120;
            let frame_offset = (frame_idx as u32 * 7) % 60;
            let seed = (i as u64) ^ (frame_idx as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ plane_seed;
            let dither = (pseudo_random(seed) % 16) as u32;
            let value = 20 + spatial + frame_offset + dither;
            value.min(235) as u8
        })
        .collect()
}

fn make_planes(layout: &FrameLayout, frame_idx: usize) -> Planes {
    let (chroma_w, _) = layout.chroma_dims();
    Planes {
        y: ramp_plane(layout.luma_pixels(), layout.width, frame_idx, 1),
        u: ramp_plane(layout.chroma_pixels(), chroma_w, frame_idx, 2),
        v: ramp_plane(layout.chroma_pixels(), chroma_w, frame_idx, 3),
    }
}

/// `count` frames, each with distinct content, for building sliding
/// windows out of without re-generating a window's frames per call.
fn make_clip(layout: &FrameLayout, count: usize) -> Vec<Planes> {
    (0..count).map(|i| make_planes(layout, i)).collect()
}

/// The accelerator a real denoiser would pick for `accelerators` and
/// `device`, read from a throwaway probe since `PlanarDenoiser` may own
/// up to three inner denoisers and exposes no single accelerator getter.
fn selected_accelerator(accelerators: &[Accelerator], device: &Device) -> Result<Accelerator, anyhow::Error> {
    let opts = DenoiserOptions::builder()
        .channel_mode(ChannelMode::Luma)
        .mode(DenoisingMode::Spacial)
        .algorithm(Algorithm::default())
        .build();
    let probe = Denoiser::create(accelerators, device, 4, 4, opts)?;
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

/// Times `push` plus `recv` per output frame once the window is primed and
/// the stream is in steady state.
fn bench_sequential(accelerators: &[Accelerator], device: &Device) -> Result<BenchResult, anyhow::Error> {
    let layout = layout();
    let opts = plane_options(accelerators, device);
    let mut denoiser = PlanarDenoiser::create(&opts, layout)?;
    let accelerator = selected_accelerator(accelerators, device)?;

    let radius = denoiser.temporal_radius();
    let window = 2 * radius + 1;
    let clip = make_clip(&layout, window as usize + WARMUP + ITERS);

    // Prime the window outside the timed region so steady-state push/recv
    // lines up: one recv for every push from here on.
    for frame in &clip[..window.saturating_sub(1) as usize] {
        if push_needs_retry(denoiser.push(frame))? {
            let _ = denoiser.recv()?;
            denoiser.push(frame)?;
        }
    }
    while denoiser.recv()?.is_some() {}

    let mut idx = window.saturating_sub(1) as usize;

    for _ in 0..WARMUP {
        denoiser.push(&clip[idx])?;
        let _ = denoiser.recv()?;
        idx += 1;
    }

    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let start = Instant::now();
        denoiser.push(&clip[idx])?;
        let _out = denoiser.recv()?;
        times.push(start.elapsed());
        idx += 1;
    }

    // Drain the trailing temporal frames before the denoiser drops, so no
    // pending readback dies in flight with its GPU buffer still mapped.
    denoiser.flush(|_| {})?;

    Ok(summarise("sequential", accelerator, &times))
}

/// Times one `reseed` call over a fresh window per iteration. The
/// denoiser is built once, before timing starts, and every window is
/// built ahead of time too, so only `reseed` itself is on the clock.
fn bench_reseed(accelerators: &[Accelerator], device: &Device) -> Result<BenchResult, anyhow::Error> {
    let layout = layout();
    let opts = plane_options(accelerators, device);
    let mut denoiser = PlanarDenoiser::create(&opts, layout)?;
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
        let _out = denoiser.reseed(window)?;
        times.push(start.elapsed());
    }

    denoiser.flush(|_| {})?;

    Ok(summarise("reseed", accelerator, &times))
}

fn main() {
    // SAFETY: single-threaded at entry, no race possible.
    unsafe { av_denoise_core::raise_codegen_stack_limit() };

    use clap::Parser;
    let cli = Cli::parse();

    println!("Reseed cost benchmark - {W}×{H}, temporal radius {RADIUS}");
    println!("  warmup={WARMUP}, timed={ITERS}");
    println!("  device:        {:?}", cli.device);
    println!("  accelerators:  {:?}", cli.accelerators);
    println!();

    let sequential = match bench_sequential(&cli.accelerators, &cli.device) {
        Ok(result) => result,
        Err(err) => {
            eprintln!("[sequential] failed: {err:?}");
            return;
        },
    };
    sequential.print();

    let reseed = match bench_reseed(&cli.accelerators, &cli.device) {
        Ok(result) => result,
        Err(err) => {
            eprintln!("[reseed] failed: {err:?}");
            return;
        },
    };
    reseed.print();

    let ratio = reseed.mean_ms / sequential.mean_ms;
    println!();
    println!("reseed / sequential ratio: {ratio:.2}x");
}

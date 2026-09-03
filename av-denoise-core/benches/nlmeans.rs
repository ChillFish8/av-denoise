use std::hint::black_box;
use std::time::{Duration, Instant};

use av_denoise_core::nlmeans::kernels::{nlm_accumulate, nlm_bilateral, nlm_dist_2d_weight, nlm_finish};
use av_denoise_core::nlmeans::prefilter::bilateral_radius;
use av_denoise_core::nlmeans::{
    BLOCK_X,
    BLOCK_Y,
    ChannelMode,
    NlmDenoiser,
    NlmParams,
    Pending,
    PrefilterMode,
};
use cubecl::prelude::*;

const W: u32 = 1920;
const H: u32 = 1080;

const WARMUP_KERNEL: usize = 5;
const ITERS_KERNEL: usize = 100;

const WARMUP_PIPELINE: usize = 2;
const ITERS_PIPELINE: usize = 500;

fn stored_channels(ch: u32) -> u32 {
    match ch {
        1 => 1,
        2 => 2,
        _ => 4, // YUV: 3 logical, 4 stored (vec3 -> vec4 padding)
    }
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

/// Pad to next-pow2 lane count (matches NlmDenoiser internal storage).
fn make_padded_frame(w: u32, h: u32, ch: u32) -> Vec<f32> {
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
            "[{:<7}] {:<60} {:>4} iters  {:>9.2} fps  {:>7.2} ms/frame  \
             (min: {:>6.2}, max: {:>6.2})",
            self.backend, self.name, self.iterations, self.fps, self.mean_ms, self.min_ms, self.max_ms,
        );
    }
}

fn run_bench<R: Runtime>(
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

fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

fn bench_dist_2d_weight<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    ch: u32,
    ch_name: &str,
) -> BenchResult {
    let pixels = (W * H) as usize;
    let stored_ch = stored_channels(ch);
    let frame = make_padded_frame(W, H, ch);
    let input = client.create_from_slice(f32::as_bytes(&frame));
    let output = client.empty(pixels * size_of::<f32>());

    let params = NlmParams {
        patch_radius: 4,
        channels: match ch {
            1 => ChannelMode::Luma,
            2 => ChannelMode::Chroma,
            _ => ChannelMode::Yuv,
        },
        ..NlmParams::default()
    };
    let h2_inv_norm = params.h2_inv_norm();

    let grid_x = div_ceil(W, BLOCK_X);
    let grid_y = div_ceil(H, BLOCK_Y);
    let cube_count = CubeCount::new_2d(grid_x, grid_y);
    let cube_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);

    let name = format!("dist_2d_weight_1080p_{ch_name}");

    run_bench(&name, backend, client, WARMUP_KERNEL, ITERS_KERNEL, || unsafe {
        nlm_dist_2d_weight::launch_unchecked::<R>(
            client,
            cube_count.clone(),
            cube_dim,
            stored_ch as usize,
            ArrayArg::from_raw_parts(input.clone(), frame.len()),
            ArrayArg::from_raw_parts(output.clone(), pixels),
            0u32,
            0u32,
            1i32,
            0i32,
            h2_inv_norm,
            0.0f32,
            W,
            H,
            ch,
            params.patch_radius,
            BLOCK_X,
            BLOCK_Y,
        );
    })
}

fn bench_accumulate<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    ch: u32,
    ch_name: &str,
) -> BenchResult {
    let pixels = (W * H) as usize;
    let stored_ch = stored_channels(ch);
    let frame = make_padded_frame(W, H, ch);
    let input = client.create_from_slice(f32::as_bytes(&frame));

    let weights_data = vec![0.5f32; pixels];
    let weights = client.create_from_slice(f32::as_bytes(&weights_data));

    let accum = client.empty(pixels * stored_ch as usize * size_of::<f32>());
    let weight_sum = client.empty(pixels * size_of::<f32>());
    let max_weight = client.empty(pixels * size_of::<f32>());

    let grid_x = div_ceil(W, BLOCK_X);
    let grid_y = div_ceil(H, BLOCK_Y);
    let cube_count = CubeCount::new_2d(grid_x, grid_y);
    let cube_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);

    let name = format!("accumulate_1080p_{ch_name}");

    run_bench(&name, backend, client, WARMUP_KERNEL, ITERS_KERNEL, || unsafe {
        nlm_accumulate::launch_unchecked::<R>(
            client,
            cube_count.clone(),
            cube_dim,
            stored_ch as usize,
            ArrayArg::from_raw_parts(input.clone(), frame.len()),
            ArrayArg::from_raw_parts(accum.clone(), pixels * stored_ch as usize),
            ArrayArg::from_raw_parts(weight_sum.clone(), pixels),
            ArrayArg::from_raw_parts(weights.clone(), pixels),
            ArrayArg::from_raw_parts(weights.clone(), pixels),
            ArrayArg::from_raw_parts(max_weight.clone(), pixels),
            0u32,
            0u32,
            1i32,
            0i32,
            W,
            H,
        );
    })
}

fn bench_finish<R: Runtime>(client: &ComputeClient<R>, backend: &str, ch: u32, ch_name: &str) -> BenchResult {
    let pixels = (W * H) as usize;
    let stored_ch = stored_channels(ch);
    let frame = make_padded_frame(W, H, ch);
    let input = client.create_from_slice(f32::as_bytes(&frame));

    let accum_data = vec![0.25f32; pixels * stored_ch as usize];
    let accum = client.create_from_slice(f32::as_bytes(&accum_data));

    let ws_data = vec![1.0f32; pixels];
    let weight_sum = client.create_from_slice(f32::as_bytes(&ws_data));

    let mw_data = vec![0.8f32; pixels];
    let max_weight = client.create_from_slice(f32::as_bytes(&mw_data));

    let output = client.empty(pixels * stored_ch as usize * size_of::<f32>());

    let grid_x = div_ceil(W, BLOCK_X);
    let grid_y = div_ceil(H, BLOCK_Y);
    let cube_count = CubeCount::new_2d(grid_x, grid_y);
    let cube_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);

    let name = format!("finish_1080p_{ch_name}");

    run_bench(&name, backend, client, WARMUP_KERNEL, ITERS_KERNEL, || unsafe {
        nlm_finish::launch_unchecked::<R>(
            client,
            cube_count.clone(),
            cube_dim,
            stored_ch as usize,
            ArrayArg::from_raw_parts(input.clone(), frame.len()),
            ArrayArg::from_raw_parts(output.clone(), pixels * stored_ch as usize),
            ArrayArg::from_raw_parts(accum.clone(), pixels * stored_ch as usize),
            ArrayArg::from_raw_parts(weight_sum.clone(), pixels),
            ArrayArg::from_raw_parts(max_weight.clone(), pixels),
            0u32,
            0u32,
            1.0f32,
            W,
            H,
            ch,
        );
    })
}

fn bench_bilateral<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    ch: u32,
    ch_name: &str,
) -> BenchResult {
    let pixels = (W * H) as usize;
    let stored_ch = stored_channels(ch);
    let frame = make_padded_frame(W, H, ch);
    let input = client.create_from_slice(f32::as_bytes(&frame));
    let output = client.empty(pixels * stored_ch as usize * size_of::<f32>());

    let radius = bilateral_radius(BILATERAL_SIGMA_S);

    let grid_x = div_ceil(W, BLOCK_X);
    let grid_y = div_ceil(H, BLOCK_Y);
    let cube_count = CubeCount::new_2d(grid_x, grid_y);
    let cube_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);

    let name = format!("bilateral_1080p_{ch_name}");

    run_bench(&name, backend, client, WARMUP_KERNEL, ITERS_KERNEL, || unsafe {
        nlm_bilateral::launch_unchecked::<R>(
            client,
            cube_count.clone(),
            cube_dim,
            stored_ch as usize,
            ArrayArg::from_raw_parts(input.clone(), frame.len()),
            ArrayArg::from_raw_parts(output.clone(), pixels * stored_ch as usize),
            0u32,
            1.0 / (2.0 * BILATERAL_SIGMA_S * BILATERAL_SIGMA_S),
            1.0 / (2.0 * BILATERAL_SIGMA_R * BILATERAL_SIGMA_R),
            W,
            H,
            ch,
            radius,
            BLOCK_X,
            BLOCK_Y,
        );
    })
}

fn denoise_params(channels: ChannelMode, temporal_radius: u32, prefilter: PrefilterMode) -> NlmParams {
    NlmParams {
        temporal_radius,
        search_radius: 2,
        patch_radius: 4,
        strength: 1.2,
        self_weight: 1.0,
        channels,
        prefilter,
        ..NlmParams::default()
    }
}

/// Push a frame (and, when needed, a matching reference) for the
/// configured prefilter mode. Used by the streaming pipeline benches so
/// the same push pattern works for `External` and non-`External` modes.
fn push_frame_for_prefilter<R: Runtime>(
    denoiser: &mut NlmDenoiser<R>,
    frame: &[f32],
    supply_reference: bool,
) {
    if supply_reference {
        denoiser.push_frame_with_reference(frame, frame);
    } else {
        denoiser.push_frame(frame);
    }
}

/// Steady-state streaming bench: every iteration pushes a fresh frame
/// (the real per-frame cost: upload plus optional prefilter) and then
/// calls the synchronous `denoise()` which waits for the readback. This
/// is the cost a caller pays if they push and wait in lockstep.
fn bench_denoise_spatial<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    channels: ChannelMode,
    ch_name: &str,
    prefilter: PrefilterMode,
    tag: &str,
) -> BenchResult {
    let ch = channels.count();
    let params = denoise_params(channels, 0, prefilter);
    let frame = make_synthetic_frame(W, H, ch);
    let supply_reference = matches!(prefilter, PrefilterMode::External);
    let name = format!("denoise_spatial{tag}_1080p_{ch_name}");

    let mut denoiser = NlmDenoiser::<R>::new(client, params, W, H);
    futures::executor::block_on(client.sync()).unwrap();

    run_bench(&name, backend, client, WARMUP_PIPELINE, ITERS_PIPELINE, || {
        push_frame_for_prefilter(&mut denoiser, &frame, supply_reference);
        let result = denoiser
            .denoise()
            .unwrap()
            .unwrap()
            .as_f32()
            .expect("f32 denoiser");
        black_box(&result);
    })
}

/// Steady-state temporal streaming bench. The window is pre-filled
/// outside the timer (a one-off cost in real usage), then every measured
/// iteration pushes one fresh frame and waits for that frame's denoise.
fn bench_denoise_temporal<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    channels: ChannelMode,
    ch_name: &str,
    prefilter: PrefilterMode,
    tag: &str,
) -> BenchResult {
    let ch = channels.count();
    let params = denoise_params(channels, 1, prefilter);
    let frame = make_synthetic_frame(W, H, ch);
    let total_frames = 1 + 2 * params.temporal_radius as usize;
    let supply_reference = matches!(prefilter, PrefilterMode::External);
    let name = format!("denoise_temporal{tag}_1080p_{ch_name}");

    let mut denoiser = NlmDenoiser::<R>::new(client, params, W, H);
    for _ in 0..total_frames - 1 {
        push_frame_for_prefilter(&mut denoiser, &frame, supply_reference);
    }
    futures::executor::block_on(client.sync()).unwrap();

    run_bench(&name, backend, client, WARMUP_PIPELINE, ITERS_PIPELINE, || {
        push_frame_for_prefilter(&mut denoiser, &frame, supply_reference);
        let result = denoiser
            .denoise()
            .unwrap()
            .unwrap()
            .as_f32()
            .expect("f32 denoiser");
        black_box(&result);
    })
}

/// Pipelined variant: each iteration pushes a fresh frame, submits its
/// denoise kernels (no wait), then blocks on the *previous* frame's
/// readback. With double-buffered output handles, frame N+1's kernels
/// run on the GPU while frame N's host readback is still in flight.
fn bench_denoise_temporal_pipelined<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    channels: ChannelMode,
    ch_name: &str,
    prefilter: PrefilterMode,
    tag: &str,
) -> BenchResult {
    let ch = channels.count();
    let params = denoise_params(channels, 1, prefilter);
    let frame = make_synthetic_frame(W, H, ch);
    let total_frames = 1 + 2 * params.temporal_radius as usize;
    let supply_reference = matches!(prefilter, PrefilterMode::External);
    let name = format!("denoise_temporal_pipelined{tag}_1080p_{ch_name}");

    let mut denoiser = NlmDenoiser::<R>::new(client, params, W, H);
    for _ in 0..total_frames - 1 {
        push_frame_for_prefilter(&mut denoiser, &frame, supply_reference);
    }
    futures::executor::block_on(client.sync()).unwrap();

    // Prime the pipeline with one outstanding `Pending` so every measured
    // iteration has previous work to wait on.
    push_frame_for_prefilter(&mut denoiser, &frame, supply_reference);
    let mut in_flight: Option<Pending<R>> = Some(denoiser.denoise_submit().unwrap().unwrap());

    let result = run_bench(&name, backend, client, WARMUP_PIPELINE, ITERS_PIPELINE, || {
        push_frame_for_prefilter(&mut denoiser, &frame, supply_reference);
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

const BILATERAL_SIGMA_S: f32 = 3.0;
const BILATERAL_SIGMA_R: f32 = 0.02;

const DENOISE_VARIANTS: &[(PrefilterMode, &str)] = &[
    (PrefilterMode::None, ""),
    (PrefilterMode::External, "_rclip_external"),
    (
        PrefilterMode::Bilateral {
            sigma_s: BILATERAL_SIGMA_S,
            sigma_r: BILATERAL_SIGMA_R,
        },
        "_rclip_bilateral",
    ),
    (PrefilterMode::NlmSpatial { strength_scale: 1.0 }, "_nlm_pilot"),
];

fn run_all_benches<R: Runtime>(backend: &str, device: &R::Device) {
    let client = R::client(device);

    println!("--- {backend} ---");
    println!();

    let channels = [
        (1u32, "luma", ChannelMode::Luma),
        (2, "chroma", ChannelMode::Chroma),
        (3, "yuv", ChannelMode::Yuv),
    ];

    for &(ch, ch_name, _) in &channels {
        bench_dist_2d_weight::<R>(&client, backend, ch, ch_name).print();
    }
    println!();

    for &(ch, ch_name, _) in &channels {
        bench_accumulate::<R>(&client, backend, ch, ch_name).print();
    }
    println!();

    for &(ch, ch_name, _) in &channels {
        bench_finish::<R>(&client, backend, ch, ch_name).print();
    }
    println!();

    for &(ch, ch_name, _) in &channels {
        bench_bilateral::<R>(&client, backend, ch, ch_name).print();
    }
    println!();

    // Group each channel mode's baseline and rclip variants together so
    // before/after comparisons land on adjacent rows.
    for &(_, ch_name, mode) in &channels {
        for &(prefilter, tag) in DENOISE_VARIANTS {
            bench_denoise_spatial::<R>(&client, backend, mode, ch_name, prefilter, tag).print();
        }
    }
    println!();

    for &(_, ch_name, mode) in &channels {
        for &(prefilter, tag) in DENOISE_VARIANTS {
            if matches!(prefilter, PrefilterMode::External) {
                continue;
            }
            bench_denoise_temporal::<R>(&client, backend, mode, ch_name, prefilter, tag).print();
            bench_denoise_temporal_pipelined::<R>(&client, backend, mode, ch_name, prefilter, tag).print();
        }
    }
    println!();
}

/// Bench-harness CLI. `cargo bench --bench nlmeans -- --device discrete:1`
/// selects the second discrete GPU.
#[derive(clap::Parser, Debug)]
#[command(about = "NLMeans benchmarks", long_about = None)]
struct Cli {
    /// GPU device to bind to. Format: `default`, `discrete[:N]`,
    /// `integrated[:N]`, `virtual[:N]`, or `cpu`.
    #[arg(long, default_value = "default")]
    device: av_denoise_core::Device,

    /// Swallowed: cargo passes this when invoking the bench binary.
    #[arg(long, hide = true)]
    bench: bool,
}

fn main() {
    use clap::Parser;
    let cli = Cli::parse();

    println!("NLMeans Benchmarks - 1920x1080");
    println!("  kernel:   warmup={WARMUP_KERNEL}, timed={ITERS_KERNEL}");
    println!("  pipeline: warmup={WARMUP_PIPELINE}, timed={ITERS_PIPELINE}");

    #[cfg(feature = "vulkan")]
    {
        let device = cli.device.to_wgpu().expect("wgpu device conversion failed");
        println!("  device:   {device:?}");
        println!();
        run_all_benches::<cubecl::wgpu::WgpuRuntime>("vulkan", &device);
    }

    #[cfg(not(feature = "vulkan"))]
    {
        let _ = cli;
        eprintln!("No GPU backend enabled. Run with --features vulkan");
        std::process::exit(1);
    }
}

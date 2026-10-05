use std::hint::black_box;
use std::time::{Duration, Instant};

use av_denoise_core::bench_api::kernels::{nlm_accumulate, nlm_bilateral, nlm_dist_2d_weight, nlm_finish};
use av_denoise_core::bench_api::prefilter::bilateral_radius;
use av_denoise_core::bench_api::{
    BLOCK_X,
    BLOCK_Y,
    Device,
    HostIo,
    NlmDenoiser,
    NlmParams,
    start_read,
    wait_read,
};
use av_denoise_core::{ChannelMode, PrefilterMode};
use clap::Parser;
use cubecl::prelude::*;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

const WARMUP_KERNEL: usize = 5;
const ITERS_KERNEL: usize = 100;

const WARMUP_PIPELINE: usize = 2;
const ITERS_PIPELINE: usize = 500;

const BILATERAL_SIGMA_S: f32 = 3.0;
const BILATERAL_SIGMA_R: f32 = 0.02;

const DENOISE_VARIANTS: &[(PrefilterMode, &str)] = &[
    (PrefilterMode::None, ""),
    (
        PrefilterMode::Bilateral {
            sigma_s: BILATERAL_SIGMA_S,
            sigma_r: BILATERAL_SIGMA_R,
        },
        "_rclip_bilateral",
    ),
    (PrefilterMode::NlmSpatial { strength_scale: 1.0 }, "_nlm_pilot"),
];

fn stored_channels(channels: u32) -> u32 {
    match channels {
        1 => 1,
        2 => 2,
        _ => 4, // YUV has 3 logical channels stored as 4, padding vec3 to vec4.
    }
}

fn make_synthetic_frame(width: u32, height: u32, channels: u32) -> Vec<f32> {
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
fn make_padded_frame(width: u32, height: u32, channels: u32) -> Vec<f32> {
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

fn div_ceil(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

fn bench_dist_2d_weight<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    channels: u32,
    channel_name: &str,
) -> BenchResult {
    let pixels = (WIDTH * HEIGHT) as usize;
    let stored_ch = stored_channels(channels);
    let frame = make_padded_frame(WIDTH, HEIGHT, channels);
    let frame_bytes = f32::as_bytes(&frame);
    let input = client.create_from_slice(frame_bytes);
    let output = client.empty(pixels * size_of::<f32>());

    let channel_mode = match channels {
        1 => ChannelMode::Luma,
        2 => ChannelMode::Chroma,
        _ => ChannelMode::Yuv,
    };
    let params = NlmParams {
        patch_radius: 4,
        channels: channel_mode,
        ..NlmParams::default()
    };
    let h2_inv_norm = params.h2_inv_norm();

    let grid_x = div_ceil(WIDTH, BLOCK_X);
    let grid_y = div_ceil(HEIGHT, BLOCK_Y);
    let cube_count = CubeCount::new_2d(grid_x, grid_y);
    let cube_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);

    let name = format!("dist_2d_weight_1080p_{channel_name}");

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
            WIDTH,
            HEIGHT,
            channels,
            params.patch_radius,
            BLOCK_X,
            BLOCK_Y,
        );
    })
}

fn bench_accumulate<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    channels: u32,
    channel_name: &str,
) -> BenchResult {
    let pixels = (WIDTH * HEIGHT) as usize;
    let stored_ch = stored_channels(channels);
    let frame = make_padded_frame(WIDTH, HEIGHT, channels);
    let frame_bytes = f32::as_bytes(&frame);
    let input = client.create_from_slice(frame_bytes);

    let weights_data = vec![0.5f32; pixels];
    let weights_bytes = f32::as_bytes(&weights_data);
    let weights = client.create_from_slice(weights_bytes);

    let accum = client.empty(pixels * stored_ch as usize * size_of::<f32>());
    let weight_sum = client.empty(pixels * size_of::<f32>());
    let max_weight = client.empty(pixels * size_of::<f32>());

    let grid_x = div_ceil(WIDTH, BLOCK_X);
    let grid_y = div_ceil(HEIGHT, BLOCK_Y);
    let cube_count = CubeCount::new_2d(grid_x, grid_y);
    let cube_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);

    let name = format!("accumulate_1080p_{channel_name}");

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
            WIDTH,
            HEIGHT,
        );
    })
}

fn bench_finish<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    channels: u32,
    channel_name: &str,
) -> BenchResult {
    let pixels = (WIDTH * HEIGHT) as usize;
    let stored_ch = stored_channels(channels);
    let frame = make_padded_frame(WIDTH, HEIGHT, channels);
    let frame_bytes = f32::as_bytes(&frame);
    let input = client.create_from_slice(frame_bytes);

    let accum_data = vec![0.25f32; pixels * stored_ch as usize];
    let accum_bytes = f32::as_bytes(&accum_data);
    let accum = client.create_from_slice(accum_bytes);

    let weight_sum_data = vec![1.0f32; pixels];
    let weight_sum_bytes = f32::as_bytes(&weight_sum_data);
    let weight_sum = client.create_from_slice(weight_sum_bytes);

    let max_weight_data = vec![0.8f32; pixels];
    let max_weight_bytes = f32::as_bytes(&max_weight_data);
    let max_weight = client.create_from_slice(max_weight_bytes);

    let output = client.empty(pixels * stored_ch as usize * size_of::<f32>());

    let grid_x = div_ceil(WIDTH, BLOCK_X);
    let grid_y = div_ceil(HEIGHT, BLOCK_Y);
    let cube_count = CubeCount::new_2d(grid_x, grid_y);
    let cube_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);

    let name = format!("finish_1080p_{channel_name}");

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
            WIDTH,
            HEIGHT,
            channels,
        );
    })
}

fn bench_bilateral<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    channels: u32,
    channel_name: &str,
) -> BenchResult {
    let pixels = (WIDTH * HEIGHT) as usize;
    let stored_ch = stored_channels(channels);
    let frame = make_padded_frame(WIDTH, HEIGHT, channels);
    let frame_bytes = f32::as_bytes(&frame);
    let input = client.create_from_slice(frame_bytes);
    let output = client.empty(pixels * stored_ch as usize * size_of::<f32>());

    let radius = bilateral_radius(BILATERAL_SIGMA_S);

    let grid_x = div_ceil(WIDTH, BLOCK_X);
    let grid_y = div_ceil(HEIGHT, BLOCK_Y);
    let cube_count = CubeCount::new_2d(grid_x, grid_y);
    let cube_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);

    let name = format!("bilateral_1080p_{channel_name}");

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
            WIDTH,
            HEIGHT,
            channels,
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

/// The steady-state spatial streaming cost.
///
/// Every iteration pushes a fresh frame, the real per-frame upload and optional prefilter cost, then
/// calls the synchronous `denoise()` which waits for the readback. This is the cost a caller pays
/// when pushing and waiting in lockstep.
fn bench_denoise_spatial<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    channels: ChannelMode,
    channel_name: &str,
    prefilter: PrefilterMode,
    tag: &str,
) -> BenchResult {
    let channel_count = channels.count();
    let params = denoise_params(channels, 0, prefilter);
    let frame = make_synthetic_frame(WIDTH, HEIGHT, channel_count);
    let name = format!("denoise_spatial{tag}_1080p_{channel_name}");

    let mut denoiser = NlmDenoiser::<R>::new(client, params, WIDTH, HEIGHT);
    let sync = client.sync();
    futures::executor::block_on(sync).unwrap();

    run_bench(&name, backend, client, WARMUP_PIPELINE, ITERS_PIPELINE, || {
        denoiser.push_frame(&frame);
        let result = denoiser.denoise().unwrap().unwrap();
        black_box(&result);
    })
}

/// The steady-state temporal streaming cost.
///
/// The window is pre-filled outside the timer because that is a one-off cost in real use. Every
/// measured iteration then pushes one fresh frame and waits for that frame's denoise.
fn bench_denoise_temporal<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    channels: ChannelMode,
    channel_name: &str,
    prefilter: PrefilterMode,
    tag: &str,
) -> BenchResult {
    let channel_count = channels.count();
    let params = denoise_params(channels, 1, prefilter);
    let frame = make_synthetic_frame(WIDTH, HEIGHT, channel_count);
    let total_frames = 1 + 2 * params.temporal_radius as usize;
    let name = format!("denoise_temporal{tag}_1080p_{channel_name}");

    let mut denoiser = NlmDenoiser::<R>::new(client, params, WIDTH, HEIGHT);
    for _ in 0..total_frames - 1 {
        denoiser.push_frame(&frame);
    }

    let sync = client.sync();
    futures::executor::block_on(sync).unwrap();

    run_bench(&name, backend, client, WARMUP_PIPELINE, ITERS_PIPELINE, || {
        denoiser.push_frame(&frame);
        let result = denoiser.denoise().unwrap().unwrap();
        black_box(&result);
    })
}

/// The pipelined temporal streaming cost.
///
/// Each iteration pushes a fresh frame, submits its denoise kernels without waiting, then blocks on
/// the previous frame's readback. With double-buffered output handles, frame N+1's kernels run on
/// the GPU while frame N's host readback is still in flight.
fn bench_denoise_temporal_pipelined<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    channels: ChannelMode,
    channel_name: &str,
    prefilter: PrefilterMode,
    tag: &str,
) -> BenchResult {
    let channel_count = channels.count();
    let params = denoise_params(channels, 1, prefilter);
    let frame = make_synthetic_frame(WIDTH, HEIGHT, channel_count);
    let total_frames = 1 + 2 * params.temporal_radius as usize;
    let name = format!("denoise_temporal_pipelined{tag}_1080p_{channel_name}");

    let mut denoiser = NlmDenoiser::<R>::new(client, params, WIDTH, HEIGHT);
    for _ in 0..total_frames - 1 {
        denoiser.push_frame(&frame);
    }

    let sync = client.sync();
    futures::executor::block_on(sync).unwrap();

    // Prime the pipeline with one outstanding readback so every measured iteration has previous
    // work to wait on.
    denoiser.push_frame(&frame);
    let first = denoiser.denoise_submit_gpu().unwrap().unwrap();
    let first_read = start_read(client, first.handle);
    let mut in_flight = Some(first_read);

    let result = run_bench(&name, backend, client, WARMUP_PIPELINE, ITERS_PIPELINE, || {
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

fn run_all_benches<R: Runtime>(backend: &str, device: &R::Device) {
    let client = R::client(device);

    println!("--- {backend} ---");
    println!();

    let channel_modes = [
        (1u32, "luma", ChannelMode::Luma),
        (2, "chroma", ChannelMode::Chroma),
        (3, "yuv", ChannelMode::Yuv),
    ];

    for &(channels, channel_name, _) in &channel_modes {
        let result = bench_dist_2d_weight::<R>(&client, backend, channels, channel_name);
        result.print();
    }

    println!();

    for &(channels, channel_name, _) in &channel_modes {
        let result = bench_accumulate::<R>(&client, backend, channels, channel_name);
        result.print();
    }

    println!();

    for &(channels, channel_name, _) in &channel_modes {
        let result = bench_finish::<R>(&client, backend, channels, channel_name);
        result.print();
    }

    println!();

    for &(channels, channel_name, _) in &channel_modes {
        let result = bench_bilateral::<R>(&client, backend, channels, channel_name);
        result.print();
    }

    println!();

    // Group each channel mode's baseline and rclip variants together so before/after comparisons
    // land on adjacent rows.
    for &(_, channel_name, mode) in &channel_modes {
        for &(prefilter, tag) in DENOISE_VARIANTS {
            let result = bench_denoise_spatial::<R>(&client, backend, mode, channel_name, prefilter, tag);
            result.print();
        }
    }

    println!();

    for &(_, channel_name, mode) in &channel_modes {
        for &(prefilter, tag) in DENOISE_VARIANTS {
            let eager = bench_denoise_temporal::<R>(&client, backend, mode, channel_name, prefilter, tag);
            eager.print();

            let pipelined =
                bench_denoise_temporal_pipelined::<R>(&client, backend, mode, channel_name, prefilter, tag);
            pipelined.print();
        }
    }

    println!();
}

/// Bench-harness CLI.
///
/// `cargo bench --bench nlmeans -- --device discrete:1` selects the second discrete GPU.
#[derive(clap::Parser, Debug)]
#[command(about = "NLMeans benchmarks", long_about = None)]
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

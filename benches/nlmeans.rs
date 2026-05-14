use std::hint::black_box;
use std::time::{Duration, Instant};

use av_denoise::nlmeans::kernels::{nlm_accumulate, nlm_dist_2d_weight, nlm_finish};
use av_denoise::nlmeans::{BLOCK_X, BLOCK_Y, ChannelMode, NlmDenoiser, NlmParams};
use cubecl::prelude::*;

const W: u32 = 1920;
const H: u32 = 1080;

const WARMUP_KERNEL: usize = 5;
const ITERS_KERNEL: usize = 100;

const WARMUP_PIPELINE: usize = 2;
const ITERS_PIPELINE: usize = 500;

// --- Synthetic image ---

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

// --- Bench harness ---

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
            "[{:<7}] {:<40} {:>3} iters  {:>8.2} fps  {:>8.2} ms/frame  \
             (min: {:.2}, max: {:.2})",
            self.backend,
            self.name,
            self.iterations,
            self.fps,
            self.mean_ms,
            self.min_ms,
            self.max_ms,
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

// --- Helper: div_ceil ---

fn div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

// --- Individual kernel benchmarks ---

fn bench_dist_2d_weight<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    ch: u32,
    ch_name: &str,
) -> BenchResult {
    let pixels = (W * H) as usize;
    let num_frames = 1u32;
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

    run_bench(&name, backend, client, WARMUP_KERNEL, ITERS_KERNEL, || {
        nlm_dist_2d_weight::launch::<R>(
            client,
            cube_count.clone(),
            cube_dim,
            unsafe {
                ArrayArg::from_raw_parts::<f32>(&input, frame.len(), stored_ch as usize)
            },
            unsafe { ArrayArg::from_raw_parts::<f32>(&output, pixels, 1) },
            ScalarArg::new(0u32),
            ScalarArg::new(1i32),
            ScalarArg::new(0i32),
            ScalarArg::new(0i32),
            ScalarArg::new(h2_inv_norm),
            W,
            H,
            ch,
            num_frames,
            params.patch_radius,
            BLOCK_X,
            BLOCK_Y,
        )
        .unwrap();
    })
}

fn bench_accumulate<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    ch: u32,
    ch_name: &str,
) -> BenchResult {
    let pixels = (W * H) as usize;
    let num_frames = 1u32;
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

    run_bench(&name, backend, client, WARMUP_KERNEL, ITERS_KERNEL, || {
        nlm_accumulate::launch::<R>(
            client,
            cube_count.clone(),
            cube_dim,
            unsafe {
                ArrayArg::from_raw_parts::<f32>(&input, frame.len(), stored_ch as usize)
            },
            unsafe {
                ArrayArg::from_raw_parts::<f32>(
                    &accum,
                    pixels * stored_ch as usize,
                    stored_ch as usize,
                )
            },
            unsafe { ArrayArg::from_raw_parts::<f32>(&weight_sum, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&weights, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&weights, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&max_weight, pixels, 1) },
            ScalarArg::new(0u32),
            ScalarArg::new(1i32),
            ScalarArg::new(0i32),
            ScalarArg::new(0i32),
            W,
            H,
            ch,
            num_frames,
        )
        .unwrap();
    })
}

fn bench_finish<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    ch: u32,
    ch_name: &str,
) -> BenchResult {
    let pixels = (W * H) as usize;
    let num_frames = 1u32;
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

    run_bench(&name, backend, client, WARMUP_KERNEL, ITERS_KERNEL, || {
        nlm_finish::launch::<R>(
            client,
            cube_count.clone(),
            cube_dim,
            unsafe {
                ArrayArg::from_raw_parts::<f32>(&input, frame.len(), stored_ch as usize)
            },
            unsafe {
                ArrayArg::from_raw_parts::<f32>(
                    &output,
                    pixels * stored_ch as usize,
                    stored_ch as usize,
                )
            },
            unsafe {
                ArrayArg::from_raw_parts::<f32>(
                    &accum,
                    pixels * stored_ch as usize,
                    stored_ch as usize,
                )
            },
            unsafe { ArrayArg::from_raw_parts::<f32>(&weight_sum, pixels, 1) },
            unsafe { ArrayArg::from_raw_parts::<f32>(&max_weight, pixels, 1) },
            ScalarArg::new(0u32),
            ScalarArg::new(1.0f32),
            W,
            H,
            ch,
            num_frames,
        )
        .unwrap();
    })
}

// --- Full pipeline benchmarks ---

fn bench_denoise_spatial<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    channels: ChannelMode,
    ch_name: &str,
) -> BenchResult {
    let ch = channels.count();
    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 4,
        strength: 1.2,
        self_weight: 1.0,
        channels,
    };

    let frame = make_synthetic_frame(W, H, ch);

    let name = format!("denoise_spatial_1080p_{ch_name}");

    run_bench(
        &name,
        backend,
        client,
        WARMUP_PIPELINE,
        ITERS_PIPELINE,
        || {
            let mut denoiser = NlmDenoiser::<R>::new(client, params.clone(), W, H);
            denoiser.push_frame(&frame);
            let result = denoiser.denoise().unwrap().unwrap();
            black_box(&result);
        },
    )
}

fn bench_denoise_temporal<R: Runtime>(
    client: &ComputeClient<R>,
    backend: &str,
    channels: ChannelMode,
    ch_name: &str,
) -> BenchResult {
    let ch = channels.count();
    let params = NlmParams {
        temporal_radius: 1,
        search_radius: 2,
        patch_radius: 4,
        strength: 1.2,
        self_weight: 1.0,
        channels,
    };

    let frame0 = make_synthetic_frame(W, H, ch);
    let frame1 = make_synthetic_frame(W, H, ch);
    let frame2 = make_synthetic_frame(W, H, ch);

    let name = format!("denoise_temporal_1080p_{ch_name}");

    run_bench(
        &name,
        backend,
        client,
        WARMUP_PIPELINE,
        ITERS_PIPELINE,
        || {
            let mut denoiser = NlmDenoiser::<R>::new(client, params.clone(), W, H);
            denoiser.push_frame(&frame0);
            denoiser.push_frame(&frame1);
            denoiser.push_frame(&frame2);
            let result = denoiser.denoise().unwrap().unwrap();
            black_box(&result);
        },
    )
}

// --- Runner ---

fn run_all_benches<R: Runtime>(backend: &str) {
    let device = <R as Runtime>::Device::default();
    let client = R::client(&device);

    println!("--- {backend} ---");
    println!();

    let channels = [
        (1u32, "luma", ChannelMode::Luma),
        (2, "chroma", ChannelMode::Chroma),
        (3, "yuv", ChannelMode::Yuv),
    ];

    // Individual kernel benchmarks.
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

    // Full pipeline benchmarks.
    for &(_, ch_name, mode) in &channels {
        bench_denoise_spatial::<R>(&client, backend, mode, ch_name).print();
    }
    println!();

    for &(_, ch_name, mode) in &channels {
        bench_denoise_temporal::<R>(&client, backend, mode, ch_name).print();
    }
    println!();
}

fn main() {
    println!("NLMeans Benchmarks - 1920x1080");
    println!("  kernel:   warmup={WARMUP_KERNEL}, timed={ITERS_KERNEL}");
    println!("  pipeline: warmup={WARMUP_PIPELINE}, timed={ITERS_PIPELINE}");
    println!();

    #[cfg(feature = "vulkan")]
    run_all_benches::<cubecl::wgpu::WgpuRuntime>("vulkan");

    #[cfg(not(feature = "vulkan"))]
    {
        eprintln!("No GPU backend enabled. Run with --features vulkan");
        std::process::exit(1);
    }
}

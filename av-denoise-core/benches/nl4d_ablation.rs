//! Per-frame cost of the three collab kernels at the geometry the
//! pipeline actually runs them at.
//!
//! A separate denoiser runs for luma and for chroma, and at 4:2:0 the
//! chroma planes are half-size on each axis. A bench that runs chroma at
//! full resolution would report four times the real work. Both planes are
//! measured here and summed, so the total is one frame's kernel cost.

use av_denoise_core::collab::geometry::{fused_cubes_x, ref_count, refs_along};
use av_denoise_core::collab::kernels::aggregate::{
    collab_normalise,
    collab_zero_accum,
    cross_frame_accum_scale,
    weight_scale,
};
use av_denoise_core::collab::kernels::fused::collab_fused;
use av_denoise_core::collab::kernels::transforms::dct_noise_profile;
use av_denoise_core::nlmeans::{BLOCK_X, BLOCK_Y};
use cubecl::benchmark::{Benchmark, BenchmarkComputations, TimingMethod};
use cubecl::prelude::*;
use cubecl::server::Handle;

#[derive(Clone, Copy)]
struct Geom {
    w: u32,
    h: u32,
    ch: u32,
    stored: u32,
    label: &'static str,
}

const PLANES: &[Geom] = &[
    Geom {
        w: 1920,
        h: 1080,
        ch: 1,
        stored: 1,
        label: "luma   1920x1080 c1",
    },
    Geom {
        w: 960,
        h: 540,
        ch: 2,
        stored: 2,
        label: "chroma  960x540  c2",
    },
];

const RADIUS: u32 = 2;
const REFINE: u32 = 2;
const SPATIAL_RADIUS: u32 = 9;
const K_MAX: u32 = 8;
const BLK_STEP: u32 = 8;
const BLKSIZE: u32 = 16;
const THSAD: f32 = (BLKSIZE * BLKSIZE) as f32 * 0.02;
const N_FRAMES: u32 = 2 * RADIUS + 1;
const CENTRE_SLOT: u32 = RADIUS;
const NEIGHBOUR_SLOTS: [u32; 4] = [0, 1, 3, 4];
const SIGMA: f32 = 0.02;
const LAMBDA_HT: f32 = 5.3;

fn frame_data(g: Geom) -> Vec<f32> {
    let mut data = Vec::with_capacity((g.w * g.h * g.stored) as usize);
    for y in 0..g.h {
        for x in 0..g.w {
            let base = 0.5 + 0.2 * (x as f32 * 0.05).sin() * (y as f32 * 0.03).cos();
            for c in 0..g.stored {
                let seed = (y * g.w + x) * g.stored + c;
                let hash = seed
                    .wrapping_mul(2654435761)
                    .wrapping_add(seed.wrapping_mul(340573321));
                let noise = (hash as f32 / u32::MAX as f32 - 0.5) * 0.1;
                data.push(
                    if c < g.ch {
                        (base + noise).clamp(0.0, 1.0)
                    } else {
                        0.0
                    },
                );
            }
        }
    }
    data
}

fn block_sync<R: Runtime>(client: &ComputeClient<R>) {
    cubecl::future::block_on(client.sync()).unwrap();
}

struct Rig<R: Runtime> {
    client: ComputeClient<R>,
    g: Geom,
    ring: Handle,
    ring_len: usize,
    mv_field: Handle,
    confidence: Handle,
    neighbour_slots: Handle,
    accum: Handle,
    wsum: Handle,
    output: Handle,
    group_weight: Handle,
    sigma: Handle,
    dct_profile: Handle,
    mv_len: usize,
    conf_len: usize,
    blocks_x: u32,
    blocks_y: u32,
    mv_stride: u32,
    conf_stride: u32,
}

impl<R: Runtime> Rig<R> {
    fn new(client: ComputeClient<R>, g: Geom) -> Self {
        let mut ring_data = Vec::new();
        for _ in 0..N_FRAMES {
            ring_data.extend(frame_data(g));
        }
        let ring = client.create_from_slice(f32::as_bytes(&ring_data));

        let blocks_x = g.w.div_ceil(BLK_STEP);
        let blocks_y = g.h.div_ceil(BLK_STEP);
        // `MotionCtx` pads each neighbour's slice of the motion and
        // confidence buffers up to the runtime's binding alignment, and
        // passes the padded element count as the kernel's stride. The
        // rig pads the same way so the strides the kernels compile
        // against here are the ones the pipeline compiles against.
        let align = client.properties().memory.alignment;
        let blocks = (blocks_x * blocks_y) as u64;
        let pad = |bytes: u64| bytes.next_multiple_of(align);
        let mv_stride = (pad(blocks * 2 * size_of::<i32>() as u64) / size_of::<i32>() as u64) as u32;
        let conf_stride = (pad(blocks * size_of::<f32>() as u64) / size_of::<f32>() as u64) as u32;
        let mv_len = (2 * RADIUS * mv_stride) as usize;
        let conf_len = (2 * RADIUS * conf_stride) as usize;

        let refs = ref_count(g.w, g.h);
        let pixels = (g.w * g.h) as usize;
        let frame_len = pixels * g.stored as usize;

        let mut sigma_host = vec![0.0f32; g.stored as usize];
        sigma_host[..g.ch as usize].fill(SIGMA);

        Self {
            mv_field: client.create_from_slice(i32::as_bytes(&vec![0i32; mv_len])),
            confidence: client.create_from_slice(f32::as_bytes(&vec![1.0f32; conf_len])),
            neighbour_slots: client.create_from_slice(u32::as_bytes(&NEIGHBOUR_SLOTS)),
            accum: client.empty(frame_len * N_FRAMES as usize * size_of::<i32>()),
            wsum: client.empty(pixels * N_FRAMES as usize * size_of::<i32>()),
            output: client.empty(frame_len * size_of::<f32>()),
            group_weight: client.empty(refs * size_of::<f32>()),
            sigma: client.create_from_slice(f32::as_bytes(&sigma_host)),
            dct_profile: client.create_from_slice(f32::as_bytes(&dct_noise_profile(0.0))),
            ring_len: ring_data.len(),
            ring,
            mv_len,
            conf_len,
            blocks_x,
            blocks_y,
            mv_stride,
            conf_stride,
            g,
            client,
        }
    }

    /// The fused kernel, launched exactly as `Nl4dDenoiser` launches
    /// it. Eight references share one 64-lane cube, so the grid is an
    /// eighth as wide along x as the reference grid and the cube is 1D.
    /// One row covers matching, filtering, and scatter together.
    fn fused(&self) {
        let g = self.g;
        let refs = ref_count(g.w, g.h);
        let refs_x = refs_along(g.w);
        let pixels = (g.w * g.h) as usize;
        let frame_len = pixels * g.stored as usize;
        unsafe {
            collab_fused::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_2d(fused_cubes_x(g.w), refs_along(g.h)),
                CubeDim::new_1d(64),
                g.stored as usize,
                ArrayArg::from_raw_parts(self.ring.clone(), self.ring_len),
                ArrayArg::from_raw_parts(self.mv_field.clone(), self.mv_len),
                ArrayArg::from_raw_parts(self.confidence.clone(), self.conf_len),
                ArrayArg::from_raw_parts(self.neighbour_slots.clone(), NEIGHBOUR_SLOTS.len()),
                ArrayArg::from_raw_parts(self.sigma.clone(), g.stored as usize),
                ArrayArg::from_raw_parts(self.dct_profile.clone(), 8),
                ArrayArg::from_raw_parts(self.accum.clone(), frame_len * N_FRAMES as usize),
                ArrayArg::from_raw_parts(self.wsum.clone(), pixels * N_FRAMES as usize),
                ArrayArg::from_raw_parts(self.group_weight.clone(), refs),
                CENTRE_SLOT,
                0.0f32,
                0.0f32,
                THSAD,
                LAMBDA_HT,
                weight_scale(SIGMA, &dct_noise_profile(0.0)),
                cross_frame_accum_scale(SPATIAL_RADIUS, RADIUS),
                true,
                RADIUS,
                REFINE,
                self.mv_stride,
                self.conf_stride,
                BLK_STEP,
                BLKSIZE,
                self.blocks_x,
                self.blocks_y,
                g.w,
                g.h,
                g.ch,
                K_MAX,
                g.stored,
                SPATIAL_RADIUS,
                refs_x,
            );
        }
    }

    fn normalise(&self) {
        let g = self.g;
        let pixels = (g.w * g.h) as usize;
        let frame_len = pixels * g.stored as usize;
        unsafe {
            collab_normalise::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_2d(g.w.div_ceil(BLOCK_X), g.h.div_ceil(BLOCK_Y)),
                CubeDim::new_2d(BLOCK_X, BLOCK_Y),
                g.stored as usize,
                ArrayArg::from_raw_parts(self.accum.clone(), frame_len * N_FRAMES as usize),
                ArrayArg::from_raw_parts(self.wsum.clone(), pixels * N_FRAMES as usize),
                ArrayArg::from_raw_parts(self.output.clone(), frame_len),
                0u32,
                g.w,
                g.h,
                g.ch,
                g.stored,
            );
        }
    }

    fn zero(&self) {
        let g = self.g;
        let pixels = (g.w * g.h) as usize;
        let frame_len = pixels * g.stored as usize;
        let dim = 256u32;
        let grid = (frame_len as u32).div_ceil(dim).min(65_535);
        unsafe {
            collab_zero_accum::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(dim),
                ArrayArg::from_raw_parts(self.accum.clone(), frame_len * N_FRAMES as usize),
                ArrayArg::from_raw_parts(self.wsum.clone(), pixels * N_FRAMES as usize),
                0u32,
                pixels as u32,
                g.stored,
                grid * dim,
            );
        }
    }
}

struct Arm<'a, R: Runtime> {
    rig: &'a Rig<R>,
    kernel: &'static str,
    prime: bool,
}

impl<R: Runtime> Benchmark for Arm<'_, R> {
    type Input = ();
    type Output = ();

    fn prepare(&self) -> Self::Input {
        if self.prime {
            self.rig.fused();
            block_sync(&self.rig.client);
        }
    }

    fn execute(&self, _: Self::Input) -> Result<(), String> {
        match self.kernel {
            "fused" => self.rig.fused(),
            "normalise" => self.rig.normalise(),
            _ => self.rig.zero(),
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!("{:<15} {}", self.kernel, self.rig.g.label)
    }

    fn sync(&self) {
        block_sync(&self.rig.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![
            self.rig.g.w as usize,
            self.rig.g.h as usize,
            self.rig.g.ch as usize,
        ]]
    }
}

#[derive(clap::Parser, Debug)]
struct Cli {
    #[arg(long, default_value = "default")]
    device: av_denoise_core::Device,
    #[arg(long, hide = true)]
    bench: bool,
}

fn main() {
    use clap::Parser;
    let cli = Cli::parse();

    #[cfg(feature = "vulkan")]
    {
        let device = cli.device.to_wgpu().expect("wgpu device conversion failed");
        let client = cubecl::wgpu::WgpuRuntime::client(&device);
        println!("\ncollab kernels at real per-frame geometry, TimingMethod::Device");
        println!("  device: {device:?}");
        println!(
            "  buffer alignment: {} bytes\n",
            client.properties().memory.alignment
        );

        // (name, prime). A primed arm runs `fused` once before it is
        // timed, so `normalise` reads real accumulator contents rather
        // than an empty buffer.
        let kernels = [("zero_accum", false), ("fused", false), ("normalise", true)];
        let mut totals = vec![0.0f64; kernels.len()];

        for g in PLANES {
            let rig = Rig::<cubecl::wgpu::WgpuRuntime>::new(client.clone(), *g);
            for (i, (k, prime)) in kernels.iter().enumerate() {
                let arm = Arm {
                    rig: &rig,
                    kernel: k,
                    prime: *prime,
                };
                let name = arm.name();
                match arm.run(TimingMethod::Device) {
                    Ok(d) => {
                        let c = BenchmarkComputations::new(&d);
                        let ms = c.median.as_secs_f64() * 1000.0;
                        totals[i] += ms;
                        println!("  {name:<40} {ms:>8.3} ms");
                    },
                    Err(e) => println!("  {name:<40}  error: {e}"),
                }
            }
            println!();
        }

        println!("  --- per frame, both planes summed ---");
        let mut grand = 0.0;
        for (i, (k, _)) in kernels.iter().enumerate() {
            grand += totals[i];
            println!("  {:<40} {:>8.3} ms", *k, totals[i]);
        }
        println!("  {:<40} {:>8.3} ms", "COLLAB TOTAL", grand);
        println!();
    }

    #[cfg(not(feature = "vulkan"))]
    {
        let _ = cli;
        eprintln!("No GPU backend enabled. Run with --features vulkan");
        std::process::exit(1);
    }
}

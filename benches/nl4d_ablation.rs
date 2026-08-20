//! Per-frame cost of the four collab kernels at the geometry the pipeline
//! actually runs them at.
//!
//! A separate denoiser runs for luma and for chroma, and at 4:2:0 the
//! chroma planes are half-size on each axis. A bench that runs chroma at
//! full resolution would report four times the real work. Both planes are
//! measured here and summed, so the total is one frame's kernel cost.

use av_denoise::collab::geometry::{member_buf_len, ref_count, refs_along};
use av_denoise::collab::kernels::aggregate::{
    collab_normalise,
    collab_zero_accum,
    cross_frame_accum_scale,
    weight_scale,
};
use av_denoise::collab::kernels::filter_ht::collab_filter_ht;
use av_denoise::collab::kernels::group_temporal::collab_group_temporal;
use av_denoise::collab::kernels::transforms::dct_noise_profile;
use av_denoise::nlmeans::{BLOCK_X, BLOCK_Y};
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
    member_pos: Handle,
    member_frame: Handle,
    member_count: Handle,
    member_sig2: Handle,
    accum: Handle,
    wsum: Handle,
    output: Handle,
    filtered_dummy: Handle,
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
        let mv_stride = blocks_x * blocks_y * 2;
        let conf_stride = blocks_x * blocks_y;
        let mv_len = (2 * RADIUS * mv_stride) as usize;
        let conf_len = (2 * RADIUS * conf_stride) as usize;

        let pos_len = member_buf_len(g.w, g.h, K_MAX);
        let refs = ref_count(g.w, g.h);
        let pixels = (g.w * g.h) as usize;
        let frame_len = pixels * g.stored as usize;

        let mut sigma_host = vec![0.0f32; g.stored as usize];
        sigma_host[..g.ch as usize].fill(SIGMA);

        Self {
            mv_field: client.create_from_slice(i32::as_bytes(&vec![0i32; mv_len])),
            confidence: client.create_from_slice(f32::as_bytes(&vec![1.0f32; conf_len])),
            neighbour_slots: client.create_from_slice(u32::as_bytes(&NEIGHBOUR_SLOTS)),
            member_pos: client.empty(pos_len * size_of::<u32>()),
            member_frame: client.empty(pos_len * size_of::<u32>()),
            member_count: client.empty(refs * size_of::<u32>()),
            member_sig2: client.empty(pos_len * size_of::<f32>()),
            accum: client.empty(frame_len * N_FRAMES as usize * size_of::<i32>()),
            wsum: client.empty(pixels * N_FRAMES as usize * size_of::<i32>()),
            output: client.empty(frame_len * size_of::<f32>()),
            filtered_dummy: client.empty(size_of::<f32>()),
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

    fn group(&self) {
        let g = self.g;
        let pos_len = member_buf_len(g.w, g.h, K_MAX);
        let refs = ref_count(g.w, g.h);
        let refs_x = refs_along(g.w);
        unsafe {
            collab_group_temporal::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_2d(refs_x, refs_along(g.h)),
                CubeDim::new_2d(8, 8),
                g.stored as usize,
                ArrayArg::from_raw_parts(self.ring.clone(), self.ring_len),
                ArrayArg::from_raw_parts(self.mv_field.clone(), self.mv_len),
                ArrayArg::from_raw_parts(self.confidence.clone(), self.conf_len),
                ArrayArg::from_raw_parts(self.member_pos.clone(), pos_len),
                ArrayArg::from_raw_parts(self.member_frame.clone(), pos_len),
                ArrayArg::from_raw_parts(self.member_count.clone(), refs),
                ArrayArg::from_raw_parts(self.member_sig2.clone(), pos_len),
                CENTRE_SLOT,
                ArrayArg::from_raw_parts(self.neighbour_slots.clone(), NEIGHBOUR_SLOTS.len()),
                0.0f32,
                0.0f32,
                THSAD,
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
                SPATIAL_RADIUS,
                refs_x,
            );
        }
    }

    fn filter(&self) {
        let g = self.g;
        let pos_len = member_buf_len(g.w, g.h, K_MAX);
        let refs = ref_count(g.w, g.h);
        let refs_x = refs_along(g.w);
        let pixels = (g.w * g.h) as usize;
        let frame_len = pixels * g.stored as usize;
        unsafe {
            collab_filter_ht::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_2d(refs_x, refs_along(g.h)),
                CubeDim::new_2d(8, 8),
                g.stored as usize,
                ArrayArg::from_raw_parts(self.ring.clone(), self.ring_len),
                ArrayArg::from_raw_parts(self.member_pos.clone(), pos_len),
                ArrayArg::from_raw_parts(self.member_frame.clone(), pos_len),
                ArrayArg::from_raw_parts(self.member_count.clone(), refs),
                ArrayArg::from_raw_parts(self.member_sig2.clone(), pos_len),
                ArrayArg::from_raw_parts(self.accum.clone(), frame_len * N_FRAMES as usize),
                ArrayArg::from_raw_parts(self.wsum.clone(), pixels * N_FRAMES as usize),
                ArrayArg::from_raw_parts(self.filtered_dummy.clone(), 1),
                ArrayArg::from_raw_parts(self.group_weight.clone(), refs),
                CENTRE_SLOT,
                ArrayArg::from_raw_parts(self.sigma.clone(), g.stored as usize),
                ArrayArg::from_raw_parts(self.dct_profile.clone(), 8),
                LAMBDA_HT,
                weight_scale(SIGMA, &dct_noise_profile(0.0)),
                cross_frame_accum_scale(SPATIAL_RADIUS, RADIUS),
                true,
                false,
                false,
                true,
                g.w,
                g.h,
                g.ch,
                K_MAX,
                g.stored,
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
            self.rig.group();
            block_sync(&self.rig.client);
        }
    }

    fn execute(&self, _: Self::Input) -> Result<(), String> {
        match self.kernel {
            "group_temporal" => self.rig.group(),
            "filter_ht" => self.rig.filter(),
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
    device: av_denoise::Device,
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
        println!("  device: {device:?}\n");

        let kernels = [
            ("zero_accum", false),
            ("group_temporal", false),
            ("filter_ht", true),
            ("normalise", true),
        ];
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

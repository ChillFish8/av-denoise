//! THROWAWAY. Decomposes the two nl4d kernels' time by sweeping the
//! comptime values that drive their inner loops. Delete once the
//! optimisation targets are picked.

use av_denoise::collab::geometry::{member_buf_len, ref_count, refs_along};
use av_denoise::collab::kernels::aggregate::{cross_frame_accum_scale, weight_scale};
use av_denoise::collab::kernels::filter_ht::collab_filter_ht;
use av_denoise::collab::kernels::group_temporal::collab_group_temporal;
use av_denoise::collab::kernels::transforms::dct_noise_profile;
use cubecl::benchmark::{Benchmark, BenchmarkComputations, TimingMethod};
use cubecl::prelude::*;
use cubecl::server::Handle;

const W: u32 = 1920;
const H: u32 = 1080;
const CH: u32 = 3;
const STORED_CH: u32 = 4;
const RADIUS: u32 = 2;
const REFINE: u32 = 2;
const BLK_STEP: u32 = 8;
const BLKSIZE: u32 = 16;
const THSAD: f32 = (BLKSIZE * BLKSIZE) as f32 * 0.02;
const N_FRAMES: u32 = 2 * RADIUS + 1;
const CENTRE_SLOT: u32 = RADIUS;
const NEIGHBOUR_SLOTS: [u32; 4] = [0, 1, 3, 4];
const SIGMA: f32 = 0.02;
const LAMBDA_HT: f32 = 5.3;

fn frame_data() -> Vec<f32> {
    let mut data = Vec::with_capacity((W * H * STORED_CH) as usize);
    for y in 0..H {
        for x in 0..W {
            let base = 0.5 + 0.2 * (x as f32 * 0.05).sin() * (y as f32 * 0.03).cos();
            for c in 0..STORED_CH {
                let seed = (y * W + x) * STORED_CH + c;
                let hash = seed
                    .wrapping_mul(2654435761)
                    .wrapping_add(seed.wrapping_mul(340573321));
                let noise = (hash as f32 / u32::MAX as f32 - 0.5) * 0.1;
                let v = if c < CH { (base + noise).clamp(0.0, 1.0) } else { 0.0 };
                data.push(v);
            }
        }
    }
    data
}

fn block_sync<R: Runtime>(client: &ComputeClient<R>) {
    cubecl::future::block_on(client.sync()).unwrap();
}

/// Everything both kernels read, sized for the largest `k_max` swept so
/// one allocation serves every arm.
struct Rig<R: Runtime> {
    client: ComputeClient<R>,
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
    fn new(client: ComputeClient<R>) -> Self {
        let mut ring_data = Vec::new();
        for _ in 0..N_FRAMES {
            ring_data.extend(frame_data());
        }
        let ring = client.create_from_slice(f32::as_bytes(&ring_data));

        let blocks_x = W.div_ceil(BLK_STEP);
        let blocks_y = H.div_ceil(BLK_STEP);
        let mv_stride = blocks_x * blocks_y * 2;
        let conf_stride = blocks_x * blocks_y;
        let mv_len = (2 * RADIUS * mv_stride) as usize;
        let conf_len = (2 * RADIUS * conf_stride) as usize;

        let mv_field = client.create_from_slice(i32::as_bytes(&vec![0i32; mv_len]));
        let confidence = client.create_from_slice(f32::as_bytes(&vec![1.0f32; conf_len]));
        let neighbour_slots = client.create_from_slice(u32::as_bytes(&NEIGHBOUR_SLOTS));

        let max_pos_len = member_buf_len(W, H, 8);
        let refs = ref_count(W, H);
        let pixels = (W * H) as usize;
        let frame_len = pixels * STORED_CH as usize;

        let mut sigma_host = vec![0.0f32; STORED_CH as usize];
        sigma_host[..CH as usize].fill(SIGMA);

        Self {
            member_pos: client.empty(max_pos_len * size_of::<u32>()),
            member_frame: client.empty(max_pos_len * size_of::<u32>()),
            member_count: client.empty(refs * size_of::<u32>()),
            member_sig2: client.empty(max_pos_len * size_of::<f32>()),
            accum: client.empty(frame_len * N_FRAMES as usize * size_of::<i32>()),
            wsum: client.empty(pixels * N_FRAMES as usize * size_of::<i32>()),
            filtered_dummy: client.empty(size_of::<f32>()),
            group_weight: client.empty(refs * size_of::<f32>()),
            sigma: client.create_from_slice(f32::as_bytes(&sigma_host)),
            dct_profile: client.create_from_slice(f32::as_bytes(&dct_noise_profile(0.0))),
            ring_len: ring_data.len(),
            ring,
            mv_field,
            confidence,
            neighbour_slots,
            mv_len,
            conf_len,
            blocks_x,
            blocks_y,
            mv_stride,
            conf_stride,
            client,
        }
    }

    fn launch_group(&self, k_max: u32, spatial_radius: u32) {
        let pos_len = member_buf_len(W, H, k_max);
        let refs = ref_count(W, H);
        let refs_x = refs_along(W);
        let refs_y = refs_along(H);
        unsafe {
            collab_group_temporal::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_2d(refs_x, refs_y),
                CubeDim::new_2d(8, 8),
                STORED_CH as usize,
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
                W,
                H,
                CH,
                k_max,
                spatial_radius,
                refs_x,
            );
        }
    }

    fn launch_filter(&self, k_max: u32) {
        let pos_len = member_buf_len(W, H, k_max);
        let refs = ref_count(W, H);
        let refs_x = refs_along(W);
        let refs_y = refs_along(H);
        let pixels = (W * H) as usize;
        let frame_len = pixels * STORED_CH as usize;
        unsafe {
            collab_filter_ht::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_2d(refs_x, refs_y),
                CubeDim::new_2d(8, 8),
                STORED_CH as usize,
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
                ArrayArg::from_raw_parts(self.sigma.clone(), STORED_CH as usize),
                ArrayArg::from_raw_parts(self.dct_profile.clone(), 8),
                LAMBDA_HT,
                weight_scale(SIGMA, &dct_noise_profile(0.0)),
                cross_frame_accum_scale(9, RADIUS),
                true,
                false,
                false,
                true,
                W,
                H,
                CH,
                k_max,
                STORED_CH,
                refs_x,
            );
        }
    }
}

struct GroupArm<'a, R: Runtime> {
    rig: &'a Rig<R>,
    k_max: u32,
    spatial_radius: u32,
}

impl<R: Runtime> Benchmark for GroupArm<'_, R> {
    type Input = ();
    type Output = ();

    fn prepare(&self) -> Self::Input {}

    fn execute(&self, _: Self::Input) -> Result<(), String> {
        self.rig.launch_group(self.k_max, self.spatial_radius);
        Ok(())
    }

    fn name(&self) -> String {
        let n_spatial = (2 * self.spatial_radius + 1) * (2 * self.spatial_radius + 1);
        let n_cand = n_spatial + 2 * RADIUS * (2 * REFINE + 1) * (2 * REFINE + 1);
        format!(
            "group_temporal k={} sr={} (cand={n_cand}, rounds={})",
            self.k_max,
            self.spatial_radius,
            self.k_max - 1,
        )
    }

    fn sync(&self) {
        block_sync(&self.rig.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![W as usize, H as usize, CH as usize]]
    }
}

struct FilterArm<'a, R: Runtime> {
    rig: &'a Rig<R>,
    k_max: u32,
}

impl<R: Runtime> Benchmark for FilterArm<'_, R> {
    type Input = ();
    type Output = ();

    fn prepare(&self) -> Self::Input {
        // Members must match the `k_max` the filter is about to run at.
        self.rig.launch_group(self.k_max, 9);
        block_sync(&self.rig.client);
    }

    fn execute(&self, _: Self::Input) -> Result<(), String> {
        self.rig.launch_filter(self.k_max);
        Ok(())
    }

    fn name(&self) -> String {
        format!("filter_ht    k={}", self.k_max)
    }

    fn sync(&self) {
        block_sync(&self.rig.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![W as usize, H as usize, CH as usize]]
    }
}

fn report<B: Benchmark>(bench: B) {
    let name = bench.name();
    match bench.run(TimingMethod::Device) {
        Ok(d) => {
            let c = BenchmarkComputations::new(&d);
            println!(
                "  {:<52}  {:>10.3} ms  (median {:>8.3} ms)",
                name,
                c.mean.as_secs_f64() * 1000.0,
                c.median.as_secs_f64() * 1000.0,
            );
        },
        Err(e) => println!("  {name:<52}  error: {e}"),
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
        let rig = Rig::<cubecl::wgpu::WgpuRuntime>::new(client);

        println!("\nnl4d ablation - 1920x1080 yuv, TimingMethod::Device");
        println!("  device: {device:?}\n");

        println!("-- k_max sweep (scoring fixed, selection rounds = k-1) --");
        for k in [2u32, 4, 8] {
            report(GroupArm { rig: &rig, k_max: k, spatial_radius: 9 });
        }
        println!("\n-- spatial_radius sweep (selection fixed at 7 rounds) --");
        for sr in [4u32, 6, 9] {
            report(GroupArm { rig: &rig, k_max: 8, spatial_radius: sr });
        }
        println!("\n-- filter_ht k_max sweep --");
        for k in [2u32, 4, 8] {
            report(FilterArm { rig: &rig, k_max: k });
        }
        println!();
    }

    #[cfg(not(feature = "vulkan"))]
    {
        let _ = cli;
        eprintln!("No GPU backend enabled. Run with --features vulkan");
        std::process::exit(1);
    }
}

//! Per-frame cost of the three collab kernels at the geometry the pipeline runs them at
//!
//! Luma and chroma run in separate denoisers, and at 4:2:0 the chroma planes are half-size on each
//! axis. A bench that ran chroma at full resolution would report four times the real work, so both
//! planes are measured and summed into one frame's kernel cost.

use av_denoise_core::bench_api::collab::geometry::{fused_cubes_x, ref_count, refs_along, strength_map_dims};
use av_denoise_core::bench_api::collab::kernels::aggregate::{
    collab_normalise,
    collab_zero_accum,
    cross_frame_accum_scale,
    kaiser_window,
    weight_scale,
};
use av_denoise_core::bench_api::collab::kernels::fused::{STRENGTH_MAP_OFF, collab_fused};
use av_denoise_core::bench_api::collab::kernels::transforms::dct_noise_profile;
use av_denoise_core::bench_api::collab::{
    COLLAB_GROUPS,
    PATCH_SIZE,
    grid_frames,
    needs_warp_uniform_search,
    supports_f16_search,
};
use av_denoise_core::bench_api::{BLOCK_X, BLOCK_Y, Device, NOISE_CURVE_BINS};
use clap::Parser;
use cubecl::benchmark::{Benchmark, BenchmarkComputations, TimingMethod};
use cubecl::prelude::*;
use cubecl::server::Handle;

#[derive(Clone, Copy)]
struct PlaneGeometry {
    width: u32,
    height: u32,
    channels: u32,
    stored_channels: u32,
    label: &'static str,
}

const PLANES: &[PlaneGeometry] = &[
    PlaneGeometry {
        width: 1920,
        height: 1080,
        channels: 1,
        stored_channels: 1,
        label: "luma   1920x1080 c1",
    },
    PlaneGeometry {
        width: 960,
        height: 540,
        channels: 2,
        stored_channels: 2,
        label: "chroma  960x540  c2",
    },
];

const RADIUS: u32 = 2;
const REFINE: u32 = 2;
const SPATIAL_RADIUS: u32 = 9;
const K_MAX: u32 = 8;
const BLK_STEP: u32 = 8;
const BLKSIZE: u32 = 16;
const N_FRAMES: u32 = 2 * RADIUS + 1;
const CENTRE_SLOT: u32 = RADIUS;
const NEIGHBOUR_SLOTS: [u32; 4] = [0, 1, 3, 4];
const SIGMA: f32 = 0.02;
/// `Nl4dParams::default().lambda_ht`.
const LAMBDA_HT: f32 = 4.158;
/// The pooled threshold over `LAMBDA_HT`, as `nl4d_pool_ratio` gives it at the default lambda.
const POOL_RATIO: f32 = 2.42 / LAMBDA_HT;

fn frame_data(geometry: PlaneGeometry) -> Vec<f32> {
    let mut data = Vec::with_capacity((geometry.width * geometry.height * geometry.stored_channels) as usize);
    for y in 0..geometry.height {
        for x in 0..geometry.width {
            let base = 0.5 + 0.2 * (x as f32 * 0.05).sin() * (y as f32 * 0.03).cos();
            for channel in 0..geometry.stored_channels {
                let seed = (y * geometry.width + x) * geometry.stored_channels + channel;
                let hash = seed
                    .wrapping_mul(2654435761)
                    .wrapping_add(seed.wrapping_mul(340573321));
                let noise = (hash as f32 / u32::MAX as f32 - 0.5) * 0.1;
                let sample = if channel < geometry.channels {
                    (base + noise).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                data.push(sample);
            }
        }
    }

    data
}

fn block_sync<R: Runtime>(client: &ComputeClient<R>) {
    let sync = client.sync();
    cubecl::future::block_on(sync).unwrap();
}

struct Rig<R: Runtime> {
    client: ComputeClient<R>,
    geometry: PlaneGeometry,
    ring: Handle,
    ring_len: usize,
    /// An f16 copy of `ring`, which the `fused_prod_f16` row searches.
    search_ring: Handle,
    mv_field: Handle,
    confidence: Handle,
    neighbour_slots: Handle,
    accum: Handle,
    wsum: Handle,
    output: Handle,
    group_weight: Handle,
    sigma: Handle,
    dct_profile: Handle,
    /// An all-zero noise curve, passed with `curve_valid = 0` where the kernel never applies it.
    zero_curve: Handle,
    /// A strength map of ones, passed with the map off.
    unit_map: Handle,
    map_len: usize,
    /// The uniform aggregation window, which the `fused` row runs with.
    kaiser_off: Handle,
    /// A `beta = 2` window, which the `fused_kaiser` and `fused_prod` rows run with.
    ///
    /// The kernel loads and applies the window taps for every scattered pixel in the `fused` and
    /// `fused_kaiser` rows, so this row checks that the taper's values add no cost over the uniform
    /// window.
    kaiser_on: Handle,
    mv_len: usize,
    conf_len: usize,
    blocks_x: u32,
    blocks_y: u32,
    mv_stride: u32,
    conf_stride: u32,
}

impl<R: Runtime> Rig<R> {
    fn new(client: ComputeClient<R>, geometry: PlaneGeometry) -> Self {
        let mut ring_data = Vec::new();
        for _ in 0..N_FRAMES {
            let frame = frame_data(geometry);
            ring_data.extend(frame);
        }

        let ring_bytes = f32::as_bytes(&ring_data);
        let ring = client.create_from_slice(ring_bytes);
        let search_values: Vec<half::f16> = ring_data
            .iter()
            .map(|value| half::f16::from_f32(*value))
            .collect();
        let search_bytes = half::f16::as_bytes(&search_values);
        let search_ring = client.create_from_slice(search_bytes);

        let blocks_x = geometry.width.div_ceil(BLK_STEP);
        let blocks_y = geometry.height.div_ceil(BLK_STEP);

        // `MotionCtx` pads each neighbour's slice of the motion and confidence buffers up to the
        // runtime's binding alignment and passes the padded count as the kernel's stride. The rig
        // pads the same way so the kernels compile against the strides the pipeline uses.
        let align = client.properties().memory.alignment;
        let blocks = (blocks_x * blocks_y) as u64;
        let pad = |bytes: u64| bytes.next_multiple_of(align);
        let mv_stride_bytes = pad(blocks * 2 * size_of::<i32>() as u64);
        let mv_stride = (mv_stride_bytes / size_of::<i32>() as u64) as u32;
        let conf_stride_bytes = pad(blocks * size_of::<f32>() as u64);
        let conf_stride = (conf_stride_bytes / size_of::<f32>() as u64) as u32;
        let mv_len = (2 * RADIUS * mv_stride) as usize;
        let conf_len = (2 * RADIUS * conf_stride) as usize;

        let refs = ref_count(geometry.width, geometry.height);
        let pixels = (geometry.width * geometry.height) as usize;
        let frame_len = pixels * geometry.stored_channels as usize;

        let mut sigma_host = vec![0.0f32; geometry.stored_channels as usize];
        sigma_host[..geometry.channels as usize].fill(SIGMA);

        let (map_cols, map_rows) = strength_map_dims(geometry.width, geometry.height);
        let map_len = (map_cols * map_rows) as usize;
        let unit_map_host = vec![1.0f32; map_len];

        let mv_host = vec![0i32; mv_len];
        let mv_bytes = i32::as_bytes(&mv_host);
        let mv_field = client.create_from_slice(mv_bytes);
        let conf_host = vec![1.0f32; conf_len];
        let conf_bytes = f32::as_bytes(&conf_host);
        let confidence = client.create_from_slice(conf_bytes);
        let slots_bytes = u32::as_bytes(&NEIGHBOUR_SLOTS);
        let neighbour_slots = client.create_from_slice(slots_bytes);
        let accum = client.empty(frame_len * N_FRAMES as usize * size_of::<i32>());
        let wsum = client.empty(pixels * N_FRAMES as usize * size_of::<i32>());
        let output = client.empty(frame_len * size_of::<f32>());
        let group_weight = client.empty(refs * size_of::<f32>());
        let sigma_bytes = f32::as_bytes(&sigma_host);
        let sigma = client.create_from_slice(sigma_bytes);
        let profile_host = dct_noise_profile(0.0);
        let profile_bytes = f32::as_bytes(&profile_host);
        let dct_profile = client.create_from_slice(profile_bytes);
        let zero_curve_host = [0.0f32; NOISE_CURVE_BINS];
        let zero_curve_bytes = f32::as_bytes(&zero_curve_host);
        let zero_curve = client.create_from_slice(zero_curve_bytes);
        let unit_map_bytes = f32::as_bytes(&unit_map_host);
        let unit_map = client.create_from_slice(unit_map_bytes);
        let kaiser_off_host = kaiser_window(0.0);
        let kaiser_off_bytes = f32::as_bytes(&kaiser_off_host);
        let kaiser_off = client.create_from_slice(kaiser_off_bytes);
        let kaiser_on_host = kaiser_window(2.0);
        let kaiser_on_bytes = f32::as_bytes(&kaiser_on_host);
        let kaiser_on = client.create_from_slice(kaiser_on_bytes);

        Self {
            mv_field,
            confidence,
            neighbour_slots,
            accum,
            wsum,
            output,
            group_weight,
            sigma,
            dct_profile,
            zero_curve,
            unit_map,
            map_len,
            kaiser_off,
            kaiser_on,
            ring_len: ring_data.len(),
            ring,
            search_ring,
            mv_len,
            conf_len,
            blocks_x,
            blocks_y,
            mv_stride,
            conf_stride,
            geometry,
            client,
        }
    }

    /// The fused kernel, launched exactly as `Nl4dDenoiser` launches it.
    ///
    /// Eight references share one 64-lane cube, so the grid is an eighth as wide along x as the
    /// reference grid and the cube is 1D. One row covers matching, filtering and scatter together.
    fn fused(&self) {
        self.fused_with(&self.kaiser_off, false);
    }

    /// [Self::fused] with the aggregation window on.
    fn fused_kaiser(&self) {
        self.fused_with(&self.kaiser_on, false);
    }

    /// The program production launches, with the aggregation window and the pooled threshold on.
    fn fused_prod(&self) {
        self.fused_with(&self.kaiser_on, true);
    }

    /// [Self::fused_prod] with candidate distances read from the f16 search ring.
    fn fused_prod_f16(&self) {
        self.fused_launch::<half::f16>(&self.kaiser_on, true, &self.search_ring, self.ring_len, true);
    }

    /// The f32 search, with the f32 ring bound as its placeholder search ring.
    fn fused_with(&self, kaiser: &Handle, pooled: bool) {
        let placeholder_len = self.geometry.stored_channels as usize;
        self.fused_launch::<f32>(kaiser, pooled, &self.ring, placeholder_len, false);
    }

    fn fused_launch<S: Float>(
        &self,
        kaiser: &Handle,
        pooled: bool,
        search_ring: &Handle,
        search_len: usize,
        f16_search: bool,
    ) {
        let geometry = self.geometry;
        let refs = ref_count(geometry.width, geometry.height);
        let refs_x = refs_along(geometry.width);
        let pixels = (geometry.width * geometry.height) as usize;
        let frame_len = pixels * geometry.stored_channels as usize;
        let (map_cols, map_rows) = strength_map_dims(geometry.width, geometry.height);

        let cubes_x = fused_cubes_x(geometry.width);
        let refs_y = refs_along(geometry.height);
        let dct_profile = dct_noise_profile(0.0);
        let group_weight_scale = weight_scale(SIGMA, &dct_profile);
        let accum_scale = cross_frame_accum_scale(SPATIAL_RADIUS, RADIUS);
        let uniform_search = needs_warp_uniform_search(&self.client);
        let frames_per_volume = grid_frames(RADIUS);

        unsafe {
            collab_fused::launch_unchecked::<S, R>(
                &self.client,
                CubeCount::new_2d(cubes_x, refs_y),
                CubeDim::new_1d(64),
                geometry.stored_channels as usize,
                ArrayArg::from_raw_parts(self.ring.clone(), self.ring_len),
                ArrayArg::from_raw_parts(search_ring.clone(), search_len),
                ArrayArg::from_raw_parts(self.mv_field.clone(), self.mv_len),
                ArrayArg::from_raw_parts(self.confidence.clone(), self.conf_len),
                ArrayArg::from_raw_parts(self.neighbour_slots.clone(), NEIGHBOUR_SLOTS.len()),
                ArrayArg::from_raw_parts(self.sigma.clone(), geometry.stored_channels as usize),
                ArrayArg::from_raw_parts(self.zero_curve.clone(), NOISE_CURVE_BINS),
                ArrayArg::from_raw_parts(self.unit_map.clone(), self.map_len),
                ArrayArg::from_raw_parts(self.dct_profile.clone(), 8),
                ArrayArg::from_raw_parts(kaiser.clone(), PATCH_SIZE as usize),
                ArrayArg::from_raw_parts(self.accum.clone(), frame_len * N_FRAMES as usize),
                ArrayArg::from_raw_parts(self.wsum.clone(), pixels * N_FRAMES as usize),
                ArrayArg::from_raw_parts(self.group_weight.clone(), refs),
                CENTRE_SLOT,
                0.0f32,
                LAMBDA_HT,
                0u32,
                STRENGTH_MAP_OFF,
                group_weight_scale,
                accum_scale,
                uniform_search,
                f16_search,
                RADIUS,
                frames_per_volume,
                REFINE,
                self.mv_stride,
                self.conf_stride,
                BLK_STEP,
                BLKSIZE,
                self.blocks_x,
                self.blocks_y,
                geometry.width,
                geometry.height,
                geometry.channels,
                K_MAX,
                geometry.stored_channels,
                SPATIAL_RADIUS,
                refs_x,
                map_cols,
                map_rows,
                POOL_RATIO,
                pooled,
                COLLAB_GROUPS,
                false,
                false,
                0,
            );
        }
    }

    fn normalise(&self) {
        let geometry = self.geometry;
        let pixels = (geometry.width * geometry.height) as usize;
        let frame_len = pixels * geometry.stored_channels as usize;
        let cubes_x = geometry.width.div_ceil(BLOCK_X);
        let cubes_y = geometry.height.div_ceil(BLOCK_Y);

        unsafe {
            collab_normalise::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_2d(cubes_x, cubes_y),
                CubeDim::new_2d(BLOCK_X, BLOCK_Y),
                geometry.stored_channels as usize,
                ArrayArg::from_raw_parts(self.accum.clone(), frame_len * N_FRAMES as usize),
                ArrayArg::from_raw_parts(self.wsum.clone(), pixels * N_FRAMES as usize),
                ArrayArg::from_raw_parts(self.output.clone(), frame_len),
                0u32,
                geometry.width,
                geometry.height,
                geometry.channels,
                geometry.stored_channels,
            );
        }
    }

    fn zero(&self) {
        let geometry = self.geometry;
        let pixels = (geometry.width * geometry.height) as usize;
        let frame_len = pixels * geometry.stored_channels as usize;
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
                geometry.stored_channels,
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
            "fused_kaiser" => self.rig.fused_kaiser(),
            "fused_prod" => self.rig.fused_prod(),
            "fused_prod_f16" => self.rig.fused_prod_f16(),
            "normalise" => self.rig.normalise(),
            _ => self.rig.zero(),
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!("{:<15} {}", self.kernel, self.rig.geometry.label)
    }

    fn sync(&self) {
        block_sync(&self.rig.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        let geometry = self.rig.geometry;
        vec![vec![
            geometry.width as usize,
            geometry.height as usize,
            geometry.channels as usize,
        ]]
    }
}

#[derive(clap::Parser, Debug)]
struct Cli {
    #[arg(long, default_value = "default")]
    device: Device,
    #[arg(long, hide = true)]
    bench: bool,
}

fn main() {
    let cli = Cli::parse();

    #[cfg(feature = "vulkan")]
    {
        let device = cli.device.to_wgpu().expect("wgpu device conversion failed");
        let client = cubecl::wgpu::WgpuRuntime::client(&device);
        let alignment = client.properties().memory.alignment;
        println!("\ncollab kernels at real per-frame geometry, TimingMethod::Device");
        println!("  device: {device:?}");
        println!("  buffer alignment: {} bytes\n", alignment);

        // (name, prime). A primed arm runs `fused` once before it is timed, so `normalise` reads
        // real accumulator contents rather than an empty buffer.
        let kernels = [
            ("zero_accum", false),
            ("fused", false),
            ("fused_kaiser", false),
            ("fused_prod", false),
            ("fused_prod_f16", false),
            ("normalise", true),
        ];
        let mut totals = vec![0.0f64; kernels.len()];
        let f16_supported = supports_f16_search(&client);

        for plane in PLANES {
            let rig = Rig::<cubecl::wgpu::WgpuRuntime>::new(client.clone(), *plane);
            for (index, (kernel, prime)) in kernels.iter().enumerate() {
                if *kernel == "fused_prod_f16" && !f16_supported {
                    println!("  {kernel:<15} skipped, the device has no f16 search");
                    continue;
                }

                let arm = Arm {
                    rig: &rig,
                    kernel,
                    prime: *prime,
                };
                let name = arm.name();
                match arm.run(TimingMethod::Device) {
                    Ok(durations) => {
                        let computations = BenchmarkComputations::new(&durations);
                        let median_ms = computations.median.as_secs_f64() * 1000.0;
                        totals[index] += median_ms;
                        println!("  {name:<40} {median_ms:>8.3} ms");
                    },
                    Err(err) => println!("  {name:<40}  error: {err}"),
                }
            }

            println!();
        }

        println!("  --- per frame, both planes summed ---");
        let mut grand_total = 0.0;
        for (index, (kernel, _)) in kernels.iter().enumerate() {
            grand_total += totals[index];
            println!("  {:<40} {:>8.3} ms", *kernel, totals[index]);
        }

        println!("  {:<40} {:>8.3} ms", "COLLAB TOTAL", grand_total);
        println!();
    }

    #[cfg(not(feature = "vulkan"))]
    {
        let _ = cli;
        eprintln!("No GPU backend enabled. Run with --features vulkan");
        std::process::exit(1);
    }
}

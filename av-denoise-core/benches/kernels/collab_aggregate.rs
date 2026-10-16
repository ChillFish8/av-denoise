use av_denoise_core::bench_api::collab::kernels::aggregate::{collab_normalise, collab_zero_accum};
use cubecl::benchmark::{Benchmark, Instant, ProfileDuration};
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{BLOCK_X, BLOCK_Y, HEIGHT, WIDTH, block_sync, stored_channels};

/// The 65,535 workgroups per dimension GPU limit the library clamps to.
///
/// The library's own constant is crate-private, so a bench target spells it out.
const MAX_GRID_1D: u32 = 65_535;

/// Copies `length` values of `source` over `target`.
///
/// The loop is strided, so a grid clamped to [MAX_GRID_1D] still reaches every slot.
#[cube(launch_unchecked)]
fn refill(
    source: &Array<i32>,
    target: &mut Array<i32>,
    #[comptime] length: u32,
    #[comptime] total_threads: u32,
) {
    let mut idx = ABSOLUTE_POS_X;
    while idx < length {
        target[idx as usize] = source[idx as usize];
        idx += total_threads;
    }
}

/// Divides the fixed-point accumulators back out to a finished 1080p frame plane.
///
/// Cost scales with `stored_ch`, since one accumulator slot per channel is read for every pixel.
/// The kernel zeroes what it reads, so the accumulators are refilled before every sample, outside
/// the timed span, and each sample times the divide.
pub struct CollabNormaliseBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub channels: u32,
    pub channel_name: &'static str,
}

#[derive(Clone)]
pub struct CollabNormaliseInput {
    pub accum: Handle,
    pub wsum: Handle,
    pub output: Handle,
}

/// The buffers one normalise sample reads, plus the untouched copies it is refilled from.
#[derive(Clone)]
pub struct CollabNormaliseSample {
    pub buffers: CollabNormaliseInput,
    pub accum_source: Handle,
    pub wsum_source: Handle,
}

impl<R: Runtime> CollabNormaliseBench<R> {
    /// Copies `length` values of `source` over `target` on the device.
    fn refill_from(&self, source: &Handle, target: &Handle, length: usize) {
        let dim = 256u32;
        let grid = (length as u32).div_ceil(dim).min(MAX_GRID_1D);
        let cube_count = CubeCount::new_1d(grid);
        let cube_dim = CubeDim::new_1d(dim);
        let source_handle = source.clone();
        let target_handle = target.clone();

        unsafe {
            let source_arg = ArrayArg::from_raw_parts(source_handle, length);
            let target_arg = ArrayArg::from_raw_parts(target_handle, length);

            refill::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                source_arg,
                target_arg,
                length as u32,
                grid * dim,
            );
        }
    }
}

impl<R: Runtime> Benchmark for CollabNormaliseBench<R> {
    type Input = CollabNormaliseSample;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let stored_ch = stored_channels(self.channels) as usize;
        let pixels = (WIDTH * HEIGHT) as usize;

        // Shaped like a real pass, with a few dozen contributions per pixel, each already scaled
        // into the accumulator's fixed point.
        let accum_data: Vec<i32> = (0..pixels * stored_ch)
            .map(|index| (index % 97) as i32 * 8192)
            .collect();
        let wsum_data: Vec<i32> = (0..pixels)
            .map(|index| ((index % 31) + 20) as i32 * 8192)
            .collect();

        let accum_bytes = i32::as_bytes(&accum_data);
        let accum = self.client.create_from_slice(accum_bytes);
        let accum_source = self.client.create_from_slice(accum_bytes);
        let wsum_bytes = i32::as_bytes(&wsum_data);
        let wsum = self.client.create_from_slice(wsum_bytes);
        let wsum_source = self.client.create_from_slice(wsum_bytes);
        let output = self.client.empty(pixels * stored_ch * size_of::<f32>());

        let buffers = CollabNormaliseInput { accum, wsum, output };

        CollabNormaliseSample {
            buffers,
            accum_source,
            wsum_source,
        }
    }

    /// Refills the accumulators, then times one launch with the refill outside the span.
    fn profile_full(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        let stored_ch = stored_channels(self.channels) as usize;
        let pixels = (WIDTH * HEIGHT) as usize;
        self.refill_from(&args.accum_source, &args.buffers.accum, pixels * stored_ch);
        self.refill_from(&args.wsum_source, &args.buffers.wsum, pixels);

        self.sync();
        let start_time = Instant::now();
        self.execute(args)?;
        self.sync();

        let end_time = Instant::now();
        let duration = ProfileDuration::new_system_time(start_time, end_time);
        Ok(duration)
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let stored_ch = stored_channels(self.channels) as usize;
        let pixels = (WIDTH * HEIGHT) as usize;
        let buffers = args.buffers;
        let cubes_x = WIDTH.div_ceil(BLOCK_X);
        let cubes_y = HEIGHT.div_ceil(BLOCK_Y);
        let cube_count = CubeCount::new_2d(cubes_x, cubes_y);
        let cube_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);

        unsafe {
            let accum_arg = ArrayArg::from_raw_parts(buffers.accum, pixels * stored_ch);
            let wsum_arg = ArrayArg::from_raw_parts(buffers.wsum, pixels);
            let output_arg = ArrayArg::from_raw_parts(buffers.output, pixels * stored_ch);

            collab_normalise::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                stored_ch,
                accum_arg,
                wsum_arg,
                output_arg,
                // One frame's region, since this bench measures a single frame's normalisation.
                0u32,
                WIDTH,
                HEIGHT,
                self.channels,
                stored_ch as u32,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!("collab_normalise_1080p_{}", self.channel_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![WIDTH as usize, HEIGHT as usize, self.channels as usize]]
    }
}

/// Clears both accumulators for one region, which a stream's first pass runs over the whole ring.
pub struct CollabZeroAccumBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub channels: u32,
    pub channel_name: &'static str,
}

impl<R: Runtime> Benchmark for CollabZeroAccumBench<R> {
    type Input = CollabNormaliseInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let stored_ch = stored_channels(self.channels) as usize;
        let pixels = (WIDTH * HEIGHT) as usize;
        let accum = self.client.empty(pixels * stored_ch * size_of::<i32>());
        let wsum = self.client.empty(pixels * size_of::<i32>());
        let output = self.client.empty(size_of::<f32>());

        CollabNormaliseInput { accum, wsum, output }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let stored_ch = stored_channels(self.channels) as usize;
        let pixels = (WIDTH * HEIGHT) as usize;
        let dim = 256u32;

        // `collab_zero_accum` strides, so the clamped grid still reaches every slot.
        let grid = ((pixels * stored_ch) as u32).div_ceil(dim).min(MAX_GRID_1D);

        unsafe {
            collab_zero_accum::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(dim),
                ArrayArg::from_raw_parts(args.accum.clone(), pixels * stored_ch),
                ArrayArg::from_raw_parts(args.wsum.clone(), pixels),
                // A single-frame region, as a single-frame caller passes.
                0u32,
                pixels as u32,
                stored_ch as u32,
                grid * dim,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!("collab_zero_accum_1080p_{}", self.channel_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![WIDTH as usize, HEIGHT as usize, self.channels as usize]]
    }
}

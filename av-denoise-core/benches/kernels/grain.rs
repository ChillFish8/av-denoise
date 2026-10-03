use av_denoise_core::nl4d::kernels::{grain_measure, grain_reduce_partials, grain_save_vectors};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{H, W, block_sync, make_synthetic_frame};

const STEP: u32 = 8;
const NEIGHBOURS: u32 = 4;
const RING: u32 = 5;
const SAVE_THREADS: u32 = 256;
const CELL: u32 = 8;
/// Two histograms of 16 luma bins by 64 std buckets.
const HIST_TOTAL: usize = 2 * 16 * 64;
/// 46 lag sums and a pixel count, the lanes of one autocovariance record.
const RECORD_LANES: usize = 47;
/// One record and a strength group per cell.
const PARTIAL_LANES: usize = RECORD_LANES + 1;
const STRENGTH_GROUPS: usize = 16;
const REDUCE_THREADS: u32 = 128;
const EDGE_COUNT: usize = 65;

fn blocks() -> u32 {
    W.div_ceil(STEP) * H.div_ceil(STEP)
}

/// Saves one neighbour's 1080p motion field into the grain export's vector ring.
pub struct GrainSaveVectorsBench<R: Runtime> {
    pub client: ComputeClient<R>,
}

#[derive(Clone)]
pub struct GrainSaveVectorsInput {
    pub mv: Handle,
    pub conf: Handle,
    pub saved_mv: Handle,
    pub saved_conf: Handle,
}

impl<R: Runtime> Benchmark for GrainSaveVectorsBench<R> {
    type Input = GrainSaveVectorsInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let blocks = blocks() as usize;
        let mv_host = vec![1i32; NEIGHBOURS as usize * blocks * 2];
        let conf_host = vec![0.9f32; NEIGHBOURS as usize * blocks];
        let mv = self.client.create_from_slice(i32::as_bytes(&mv_host));
        let conf = self.client.create_from_slice(f32::as_bytes(&conf_host));
        let saved_mv = self.client.empty(RING as usize * blocks * 2 * size_of::<i32>());
        let saved_conf = self.client.empty(RING as usize * blocks * size_of::<f32>());

        GrainSaveVectorsInput {
            mv,
            conf,
            saved_mv,
            saved_conf,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let blocks = blocks();
        let grid = blocks.div_ceil(SAVE_THREADS);

        unsafe {
            grain_save_vectors::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(SAVE_THREADS),
                ArrayArg::from_raw_parts(args.mv, (NEIGHBOURS * blocks * 2) as usize),
                ArrayArg::from_raw_parts(args.conf, (NEIGHBOURS * blocks) as usize),
                ArrayArg::from_raw_parts(args.saved_mv, (RING * blocks * 2) as usize),
                ArrayArg::from_raw_parts(args.saved_conf, (RING * blocks) as usize),
                2 * blocks * 2,
                2 * blocks,
                1u32,
                blocks,
                grid * SAVE_THREADS,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        "grain_save_vectors_1080p".to_string()
    }

    fn sync(&self) {
        block_sync(&self.client);
    }
}

/// Measures the source and kept grain of one 1080p luma frame.
pub struct GrainMeasureBench<R: Runtime> {
    pub client: ComputeClient<R>,
}

#[derive(Clone)]
pub struct GrainMeasureInput {
    pub input: Handle,
    pub out_t: Handle,
    pub out_prev: Handle,
    pub saved_mv: Handle,
    pub saved_conf: Handle,
    pub edges: Handle,
    pub hist: Handle,
    pub partials: Handle,
}

fn cells() -> usize {
    ((W / CELL) * (H / CELL)) as usize
}

impl<R: Runtime> Benchmark for GrainMeasureBench<R> {
    type Input = GrainMeasureInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (W * H) as usize;
        let blocks = blocks() as usize;

        // Two ring slots, a frame and a copy moved by one pixel, so every cell has per-pixel grain.
        let frame = make_synthetic_frame(W, H, 1);
        let mut ring = frame.clone();
        ring.extend_from_slice(&frame[1..]);
        ring.push(frame[0]);

        // Flat outputs with a small per-pixel ripple, so the kept grain is not zero either.
        let out_prev = vec![0.5f32; pixels];
        let out_t: Vec<f32> = (0..pixels)
            .map(|index| 0.5 + 0.001 * ((index % 7) as f32 - 3.0))
            .collect();
        let saved_mv_host = vec![0i32; 2 * blocks * 2];
        let saved_conf_host = vec![1.0f32; 2 * blocks];
        let edges_host: Vec<f32> = (0..EDGE_COUNT).map(|index| index as f32 * 0.002).collect();
        let hist_host = vec![0i32; HIST_TOTAL];

        GrainMeasureInput {
            input: self.client.create_from_slice(f32::as_bytes(&ring)),
            out_t: self.client.create_from_slice(f32::as_bytes(&out_t)),
            out_prev: self.client.create_from_slice(f32::as_bytes(&out_prev)),
            saved_mv: self.client.create_from_slice(i32::as_bytes(&saved_mv_host)),
            saved_conf: self.client.create_from_slice(f32::as_bytes(&saved_conf_host)),
            edges: self.client.create_from_slice(f32::as_bytes(&edges_host)),
            hist: self.client.create_from_slice(i32::as_bytes(&hist_host)),
            partials: self.client.empty(cells() * PARTIAL_LANES * size_of::<f32>()),
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (W * H) as usize;
        let blocks = blocks() as usize;

        unsafe {
            grain_measure::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_2d(W / CELL, H / CELL),
                CubeDim::new_2d(CELL, CELL),
                1usize,
                ArrayArg::from_raw_parts(args.input, 2 * pixels),
                ArrayArg::from_raw_parts(args.out_t, pixels),
                ArrayArg::from_raw_parts(args.out_prev, pixels),
                ArrayArg::from_raw_parts(args.saved_mv, 2 * blocks * 2),
                ArrayArg::from_raw_parts(args.saved_conf, 2 * blocks),
                ArrayArg::from_raw_parts(args.edges, EDGE_COUNT),
                ArrayArg::from_raw_parts(args.hist, HIST_TOTAL),
                ArrayArg::from_raw_parts(args.partials, cells() * PARTIAL_LANES),
                0u32,
                1u32,
                0u32,
                1u32,
                1u32,
                1u32,
                W,
                H,
                1u32,
                W.div_ceil(STEP),
                H.div_ceil(STEP),
                STEP,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        "grain_measure_1080p".to_string()
    }

    fn sync(&self) {
        block_sync(&self.client);
    }
}

/// Adds every cell's partials of one 1080p frame into its strength group's chunk record.
pub struct GrainReducePartialsBench<R: Runtime> {
    pub client: ComputeClient<R>,
}

#[derive(Clone)]
pub struct GrainReducePartialsInput {
    pub partials: Handle,
    pub chunk: Handle,
}

impl<R: Runtime> Benchmark for GrainReducePartialsBench<R> {
    type Input = GrainReducePartialsInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        // Cells spread over every strength group.
        let mut partials_host = Vec::with_capacity(cells() * PARTIAL_LANES);
        for cell in 0..cells() {
            partials_host.extend_from_slice(&[0.5f32; RECORD_LANES]);
            partials_host.push((cell % STRENGTH_GROUPS) as f32);
        }

        let chunk_host = vec![0.0f32; STRENGTH_GROUPS * RECORD_LANES];

        GrainReducePartialsInput {
            partials: self.client.create_from_slice(f32::as_bytes(&partials_host)),
            chunk: self.client.create_from_slice(f32::as_bytes(&chunk_host)),
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        unsafe {
            grain_reduce_partials::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(RECORD_LANES as u32),
                CubeDim::new_1d(REDUCE_THREADS),
                ArrayArg::from_raw_parts(args.partials, cells() * PARTIAL_LANES),
                ArrayArg::from_raw_parts(args.chunk, STRENGTH_GROUPS * RECORD_LANES),
                cells() as u32,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        "grain_reduce_partials_1080p".to_string()
    }

    fn sync(&self) {
        block_sync(&self.client);
    }
}

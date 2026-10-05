use av_denoise_core::bench_api::nl4d_kernels::{grain_measure, grain_reduce_partials, grain_save_vectors};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{HEIGHT, WIDTH, block_sync, make_synthetic_frame};

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

/// A frame size the grain benches run at.
#[derive(Clone, Copy)]
pub struct GrainSize {
    pub width: u32,
    pub height: u32,
    pub label: &'static str,
}

impl GrainSize {
    fn pixels(&self) -> usize {
        (self.width * self.height) as usize
    }

    fn blocks(&self) -> u32 {
        self.width.div_ceil(STEP) * self.height.div_ceil(STEP)
    }

    fn cells(&self) -> usize {
        ((self.width / CELL) * (self.height / CELL)) as usize
    }
}

pub const GRAIN_SIZES: &[GrainSize] = &[
    GrainSize {
        width: WIDTH,
        height: HEIGHT,
        label: "1080p",
    },
    GrainSize {
        width: 3840,
        height: 2160,
        label: "4k",
    },
];

/// Saves one neighbour's motion field into the grain export's vector ring.
pub struct GrainSaveVectorsBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub size: GrainSize,
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
        let blocks = self.size.blocks() as usize;
        let mv_host = vec![1i32; NEIGHBOURS as usize * blocks * 2];
        let conf_host = vec![0.9f32; NEIGHBOURS as usize * blocks];
        let mv_bytes = i32::as_bytes(&mv_host);
        let mv = self.client.create_from_slice(mv_bytes);
        let conf_bytes = f32::as_bytes(&conf_host);
        let conf = self.client.create_from_slice(conf_bytes);
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
        let blocks = self.size.blocks();
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
        format!("grain_save_vectors_{}", self.size.label)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }
}

/// Measures the source and kept grain of one luma frame.
pub struct GrainMeasureBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub size: GrainSize,
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

impl<R: Runtime> Benchmark for GrainMeasureBench<R> {
    type Input = GrainMeasureInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = self.size.pixels();
        let blocks = self.size.blocks() as usize;
        let cells = self.size.cells();

        // Two ring slots, a frame and a copy moved by one pixel, so every cell has per-pixel grain.
        let frame = make_synthetic_frame(self.size.width, self.size.height, 1);
        let mut ring = frame.clone();
        ring.extend_from_slice(&frame[1..]);
        ring.push(frame[0]);

        // Flat outputs with a small per-pixel ripple, so the kept grain is not zero either.
        let out_prev_host = vec![0.5f32; pixels];
        let out_t_host: Vec<f32> = (0..pixels)
            .map(|index| 0.5 + 0.001 * ((index % 7) as f32 - 3.0))
            .collect();
        let saved_mv_host = vec![0i32; 2 * blocks * 2];
        let saved_conf_host = vec![1.0f32; 2 * blocks];
        let edges_host: Vec<f32> = (0..EDGE_COUNT).map(|index| index as f32 * 0.002).collect();
        let hist_host = vec![0i32; HIST_TOTAL];

        let ring_bytes = f32::as_bytes(&ring);
        let input = self.client.create_from_slice(ring_bytes);
        let out_t_bytes = f32::as_bytes(&out_t_host);
        let out_t = self.client.create_from_slice(out_t_bytes);
        let out_prev_bytes = f32::as_bytes(&out_prev_host);
        let out_prev = self.client.create_from_slice(out_prev_bytes);
        let saved_mv_bytes = i32::as_bytes(&saved_mv_host);
        let saved_mv = self.client.create_from_slice(saved_mv_bytes);
        let saved_conf_bytes = f32::as_bytes(&saved_conf_host);
        let saved_conf = self.client.create_from_slice(saved_conf_bytes);
        let edges_bytes = f32::as_bytes(&edges_host);
        let edges = self.client.create_from_slice(edges_bytes);
        let hist_bytes = i32::as_bytes(&hist_host);
        let hist = self.client.create_from_slice(hist_bytes);
        let partials = self.client.empty(cells * PARTIAL_LANES * size_of::<f32>());

        GrainMeasureInput {
            input,
            out_t,
            out_prev,
            saved_mv,
            saved_conf,
            edges,
            hist,
            partials,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let width = self.size.width;
        let height = self.size.height;
        let pixels = self.size.pixels();
        let blocks = self.size.blocks() as usize;
        let cells = self.size.cells();
        let blocks_x = width.div_ceil(STEP);
        let blocks_y = height.div_ceil(STEP);

        unsafe {
            grain_measure::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_2d(width / CELL, height / CELL),
                CubeDim::new_2d(CELL, CELL),
                1usize,
                ArrayArg::from_raw_parts(args.input, 2 * pixels),
                ArrayArg::from_raw_parts(args.out_t, pixels),
                ArrayArg::from_raw_parts(args.out_prev, pixels),
                ArrayArg::from_raw_parts(args.saved_mv, 2 * blocks * 2),
                ArrayArg::from_raw_parts(args.saved_conf, 2 * blocks),
                ArrayArg::from_raw_parts(args.edges, EDGE_COUNT),
                ArrayArg::from_raw_parts(args.hist, HIST_TOTAL),
                ArrayArg::from_raw_parts(args.partials, cells * PARTIAL_LANES),
                0u32,
                1u32,
                0u32,
                1u32,
                1u32,
                1u32,
                width,
                height,
                1u32,
                blocks_x,
                blocks_y,
                STEP,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!("grain_measure_{}", self.size.label)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }
}

/// Adds every cell's partials of one frame into its strength group's chunk record.
pub struct GrainReducePartialsBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub size: GrainSize,
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
        let cells = self.size.cells();
        let mut partials_host = Vec::with_capacity(cells * PARTIAL_LANES);
        for cell in 0..cells {
            partials_host.extend_from_slice(&[0.5f32; RECORD_LANES]);
            partials_host.push((cell % STRENGTH_GROUPS) as f32);
        }

        let chunk_host = vec![0.0f32; STRENGTH_GROUPS * RECORD_LANES];

        let partials_bytes = f32::as_bytes(&partials_host);
        let partials = self.client.create_from_slice(partials_bytes);
        let chunk_bytes = f32::as_bytes(&chunk_host);
        let chunk = self.client.create_from_slice(chunk_bytes);

        GrainReducePartialsInput { partials, chunk }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let cells = self.size.cells();

        unsafe {
            grain_reduce_partials::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(RECORD_LANES as u32),
                CubeDim::new_1d(REDUCE_THREADS),
                ArrayArg::from_raw_parts(args.partials, cells * PARTIAL_LANES),
                ArrayArg::from_raw_parts(args.chunk, STRENGTH_GROUPS * RECORD_LANES),
                cells as u32,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!("grain_reduce_partials_{}", self.size.label)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }
}

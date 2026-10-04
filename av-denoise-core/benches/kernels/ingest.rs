use av_denoise_core::bench_api::engine_kernels::{ingest_f32, ingest_words};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{BLOCK_1D, block_sync, stored_channels};

/// How the planes under test are stored.
#[derive(Clone, Copy, Debug)]
pub enum IngestFormat {
    U8,
    U16Ten,
    F32,
}

impl IngestFormat {
    fn samples_per_word(self) -> u32 {
        match self {
            IngestFormat::U8 => 4,
            IngestFormat::U16Ten => 2,
            IngestFormat::F32 => 1,
        }
    }

    fn max(self) -> f32 {
        match self {
            IngestFormat::U8 => 255.0,
            IngestFormat::U16Ten => 1023.0,
            IngestFormat::F32 => 1.0,
        }
    }
}

#[derive(Clone)]
pub struct IngestInput {
    planes: Vec<Handle>,
    ring: Handle,
}

pub struct IngestBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub width: u32,
    pub height: u32,
    pub ch: u32,
    pub ch_name: &'static str,
    pub format: IngestFormat,
}

impl<R: Runtime> IngestBench<R> {
    fn pixels(&self) -> u32 {
        self.width * self.height
    }

    fn words(&self) -> u32 {
        self.pixels().div_ceil(self.format.samples_per_word())
    }

    fn ring_len(&self) -> usize {
        self.pixels() as usize * stored_channels(self.ch) as usize
    }
}

impl<R: Runtime> Benchmark for IngestBench<R> {
    type Input = IngestInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let plane_bytes = self.words() as usize * size_of::<u32>();
        let plane_data = vec![0x5Au8; plane_bytes];
        let planes = (0..self.ch)
            .map(|_| self.client.create_from_slice(&plane_data))
            .collect();
        let ring = self.client.empty(self.ring_len() * size_of::<f32>());
        IngestInput { planes, ring }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = self.pixels();
        let stored_ch = stored_channels(self.ch);
        let groups = pixels.div_ceil(BLOCK_1D).min(65535);
        let total_threads = groups * BLOCK_1D;
        let words = self.words() as usize;
        let ring_len = self.ring_len();
        let samples_per_word = self.format.samples_per_word();
        let max = self.format.max();

        // Planes past `ch` are placeholders the kernel never reads.
        let plane_0 = args.planes[0].clone();
        let plane_1 = args.planes.get(1).unwrap_or(&args.planes[0]).clone();
        let plane_2 = args.planes.get(2).unwrap_or(&args.planes[0]).clone();

        unsafe {
            match self.format {
                IngestFormat::F32 => ingest_f32::launch_unchecked::<R>(
                    &self.client,
                    CubeCount::new_1d(groups),
                    CubeDim::new_1d(BLOCK_1D),
                    ArrayArg::from_raw_parts(plane_0, pixels as usize),
                    ArrayArg::from_raw_parts(plane_1, pixels as usize),
                    ArrayArg::from_raw_parts(plane_2, pixels as usize),
                    ArrayArg::from_raw_parts(args.ring.clone(), ring_len),
                    0u32,
                    pixels,
                    self.ch,
                    stored_ch,
                    total_threads,
                ),
                IngestFormat::U8 | IngestFormat::U16Ten => ingest_words::launch_unchecked::<R>(
                    &self.client,
                    CubeCount::new_1d(groups),
                    CubeDim::new_1d(BLOCK_1D),
                    ArrayArg::from_raw_parts(plane_0, words),
                    ArrayArg::from_raw_parts(plane_1, words),
                    ArrayArg::from_raw_parts(plane_2, words),
                    ArrayArg::from_raw_parts(args.ring.clone(), ring_len),
                    max,
                    0u32,
                    pixels,
                    self.ch,
                    stored_ch,
                    samples_per_word,
                    total_threads,
                ),
            }
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "ingest_{}x{}_{:?}_{}",
            self.width, self.height, self.format, self.ch_name
        )
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![self.width as usize, self.height as usize, self.ch as usize]]
    }
}

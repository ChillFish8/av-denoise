use av_denoise_core::engine_kernels::{egress_f32, egress_words};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{BLOCK_1D, block_sync, make_padded_frame, stored_channels};

/// How the planes under test are stored.
#[derive(Clone, Copy, Debug)]
pub enum EgressFormat {
    U8,
    U16Ten,
    F32,
}

impl EgressFormat {
    fn samples_per_word(self) -> u32 {
        match self {
            EgressFormat::U8 => 4,
            EgressFormat::U16Ten => 2,
            EgressFormat::F32 => 1,
        }
    }

    fn max(self) -> f32 {
        match self {
            EgressFormat::U8 => 255.0,
            EgressFormat::U16Ten => 1023.0,
            EgressFormat::F32 => 1.0,
        }
    }
}

#[derive(Clone)]
pub struct EgressInput {
    frame: Handle,
    planes: Vec<Handle>,
    placeholder: Handle,
}

pub struct EgressBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub width: u32,
    pub height: u32,
    pub ch: u32,
    pub ch_name: &'static str,
    pub format: EgressFormat,
}

impl<R: Runtime> EgressBench<R> {
    fn pixels(&self) -> u32 {
        self.width * self.height
    }

    fn words(&self) -> u32 {
        self.pixels().div_ceil(self.format.samples_per_word())
    }

    fn frame_len(&self) -> usize {
        self.pixels() as usize * stored_channels(self.ch) as usize
    }
}

impl<R: Runtime> Benchmark for EgressBench<R> {
    type Input = EgressInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let frame_data = make_padded_frame(self.width, self.height, self.ch);
        let frame = self.client.create_from_slice(f32::as_bytes(&frame_data));
        let plane_bytes = self.words() as usize * size_of::<u32>();
        let planes = (0..self.ch).map(|_| self.client.empty(plane_bytes)).collect();
        let placeholder = self.client.empty(size_of::<u32>());
        EgressInput {
            frame,
            planes,
            placeholder,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = self.pixels();
        let stored_ch = stored_channels(self.ch);
        let frame_len = self.frame_len();
        let samples_per_word = self.format.samples_per_word();
        let max = self.format.max();
        let words = self.words();
        let plane_len = match self.format {
            EgressFormat::F32 => pixels as usize,
            _ => words as usize,
        };
        let threads = match self.format {
            EgressFormat::F32 => pixels,
            _ => words,
        };
        let groups = threads.div_ceil(BLOCK_1D).clamp(1, 65535);
        let total_threads = groups * BLOCK_1D;

        // Planes past `ch` bind a distinct placeholder the kernel never writes.
        let plane_0 = args.planes[0].clone();
        let plane_1 = args.planes.get(1).unwrap_or(&args.placeholder).clone();
        let plane_2 = args.planes.get(2).unwrap_or(&args.placeholder).clone();

        unsafe {
            match self.format {
                EgressFormat::F32 => egress_f32::launch_unchecked::<R>(
                    &self.client,
                    CubeCount::new_1d(groups),
                    CubeDim::new_1d(BLOCK_1D),
                    ArrayArg::from_raw_parts(args.frame.clone(), frame_len),
                    ArrayArg::from_raw_parts(plane_0, plane_len),
                    ArrayArg::from_raw_parts(plane_1, plane_len),
                    ArrayArg::from_raw_parts(plane_2, plane_len),
                    pixels,
                    self.ch,
                    stored_ch,
                    total_threads,
                ),
                EgressFormat::U8 | EgressFormat::U16Ten => egress_words::launch_unchecked::<R>(
                    &self.client,
                    CubeCount::new_1d(groups),
                    CubeDim::new_1d(BLOCK_1D),
                    ArrayArg::from_raw_parts(args.frame.clone(), frame_len),
                    ArrayArg::from_raw_parts(plane_0, plane_len),
                    ArrayArg::from_raw_parts(plane_1, plane_len),
                    ArrayArg::from_raw_parts(plane_2, plane_len),
                    max,
                    pixels,
                    self.ch,
                    stored_ch,
                    samples_per_word,
                    words,
                    total_threads,
                ),
            }
        }

        Ok(())
    }

    fn name(&self) -> String {
        format!(
            "egress_{}x{}_{:?}_{}",
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

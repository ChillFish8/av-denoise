use av_denoise_core::bench_api::kernels::motion::{mc_fit_bands, mc_fit_finish, mc_fit_sums};
use av_denoise_core::bench_api::motion::{
    FIT_BAND_WORDS,
    FIT_INLIER_CAP,
    FIT_LANES,
    FIT_SCALES,
    FIT_STAGE_POLISH,
    FIT_STAGE_SCORE,
    FIT_STAGE_SOLVE,
    FIT_STAGE_WEIGH,
    FIT_STATE_WORDS,
    FIT_THREADS,
    FitScratch,
    ModelRing,
};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{HEIGHT, WIDTH, block_sync};

const FIT_BLKSIZE: u32 = 16;
const FIT_STEP: u32 = 8;
const FIT_RING_DEPTH: u32 = 4;
const FIT_SLOT: u32 = 1;
const FIT_SCALE: f32 = 4.0;

/// The launches one bench row times.
#[derive(Debug, Clone, Copy)]
pub enum FitKernel {
    /// One solve pass of the sums kernel.
    Sums,
    /// One solve pass of the finish kernel.
    Finish,
    /// Every pass of one fit, in the order the library queues them.
    Whole,
}

impl FitKernel {
    pub const ALL: &[Self] = &[Self::Sums, Self::Finish, Self::Whole];
}

/// The camera-model fit over the 1080p luma block grid.
///
/// The field is a synthetic 3 px pan with a SAD per block. Each fit finds the bars, then runs a
/// weigh pass, a solve pass per reweighting scale, a polish pass and a score pass, each a sums
/// launch and a finish launch. A prior adds a gate pass.
pub struct FitModelBench {
    pub client: Client,
    pub kernel: FitKernel,
}

#[derive(Clone)]
pub struct FitModelInput {
    pub field: Handle,
    pub sad: Handle,
    pub scratch: FitScratch,
    pub ring: ModelRing,
}

fn blocks_x() -> u32 {
    WIDTH.div_ceil(FIT_STEP)
}

fn blocks_y() -> u32 {
    HEIGHT.div_ceil(FIT_STEP)
}

impl FitModelBench {
    fn launch_pass(&self, args: &FitModelInput, stage: u32, scale: f32) {
        self.launch_sums(args, stage, scale);
        self.launch_finish(args, stage);
    }

    fn launch_whole(&self, args: &FitModelInput) {
        self.launch_bands(args);
        self.launch_pass(args, FIT_STAGE_WEIGH, FIT_INLIER_CAP);

        for scale in FIT_SCALES {
            self.launch_pass(args, FIT_STAGE_SOLVE, scale);
        }

        self.launch_pass(args, FIT_STAGE_POLISH, FIT_INLIER_CAP);
        self.launch_pass(args, FIT_STAGE_SCORE, FIT_INLIER_CAP);
    }

    fn launch_bands(&self, args: &FitModelInput) {
        let blocks = args.scratch.blocks() as usize;
        let cube_count = CubeCount::new_1d(1);
        let cube_dim = CubeDim::new_1d(FIT_THREADS);

        let field = args.field.clone();
        let sad = args.sad.clone();
        let bands = args.scratch.bands().clone();

        unsafe {
            let field_arg = BufferArg::from_raw_parts(field, 2 * blocks);
            let sad_arg = BufferArg::from_raw_parts(sad, blocks);
            let bands_arg = BufferArg::from_raw_parts(bands, FIT_BAND_WORDS as usize);

            mc_fit_bands::launch_unchecked(
                &self.client,
                cube_count,
                cube_dim,
                field_arg,
                sad_arg,
                bands_arg,
                false,
                0,
                WIDTH,
                HEIGHT,
                FIT_BLKSIZE,
                FIT_STEP,
                blocks_x(),
                blocks_y(),
            );
        }
    }

    fn launch_sums(&self, args: &FitModelInput, stage: u32, scale: f32) {
        let blocks = args.scratch.blocks() as usize;
        let partial_count = args.scratch.partial_count();
        let partial_words = (partial_count * FIT_LANES) as usize;
        let cube_count = CubeCount::new_1d(partial_count);
        let cube_dim = CubeDim::new_1d(FIT_THREADS);

        let field = args.field.clone();
        let sad = args.sad.clone();
        let bands = args.scratch.bands().clone();
        let weights = args.scratch.weights().clone();
        let state = args.scratch.state().clone();
        let models = args.ring.handle().clone();
        let partials = args.scratch.partials().clone();

        unsafe {
            let field_arg = BufferArg::from_raw_parts(field, 2 * blocks);
            let sad_arg = BufferArg::from_raw_parts(sad, blocks);
            let bands_arg = BufferArg::from_raw_parts(bands, FIT_BAND_WORDS as usize);
            let weights_arg = BufferArg::from_raw_parts(weights, blocks);
            let state_arg = BufferArg::from_raw_parts(state, FIT_STATE_WORDS as usize);
            let models_arg = BufferArg::from_raw_parts(models, args.ring.words());
            let partials_arg = BufferArg::from_raw_parts(partials, partial_words);

            mc_fit_sums::launch_unchecked(
                &self.client,
                cube_count,
                cube_dim,
                field_arg,
                sad_arg,
                bands_arg,
                weights_arg,
                state_arg,
                models_arg,
                partials_arg,
                scale,
                0,
                stage,
                true,
                false,
                0,
                true,
                WIDTH,
                HEIGHT,
                FIT_BLKSIZE,
                FIT_STEP,
                blocks_x(),
                blocks_y(),
            );
        }
    }

    fn launch_finish(&self, args: &FitModelInput, stage: u32) {
        let partial_count = args.scratch.partial_count();
        let partial_words = (partial_count * FIT_LANES) as usize;
        let cube_count = CubeCount::new_1d(1);
        let cube_dim = CubeDim::new_1d(FIT_THREADS);

        let partials = args.scratch.partials().clone();
        let state = args.scratch.state().clone();
        let models = args.ring.handle().clone();

        unsafe {
            let partials_arg = BufferArg::from_raw_parts(partials, partial_words);
            let state_arg = BufferArg::from_raw_parts(state, FIT_STATE_WORDS as usize);
            let models_arg = BufferArg::from_raw_parts(models, args.ring.words());

            mc_fit_finish::launch_unchecked(
                &self.client,
                cube_count,
                cube_dim,
                partials_arg,
                state_arg,
                models_arg,
                partial_count,
                0,
                FIT_SLOT,
                stage,
                WIDTH,
                HEIGHT,
            );
        }
    }
}

impl Benchmark for FitModelBench {
    type Input = FitModelInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let blocks = blocks_x() * blocks_y();
        let vectors: Vec<i32> = (0..blocks).flat_map(|_| [3, 0]).collect();
        let vector_bytes = i32::as_bytes(&vectors);
        let field = self.client.create_from_slice(vector_bytes);
        let sad_values: Vec<f32> = (0..blocks).map(|block| 1.0 + (block % 7) as f32 * 0.1).collect();
        let sad_bytes = f32::as_bytes(&sad_values);
        let sad = self.client.create_from_slice(sad_bytes);
        let scratch = FitScratch::new(&self.client, blocks);
        let ring = ModelRing::new(&self.client, FIT_RING_DEPTH);
        let input = FitModelInput {
            field,
            sad,
            scratch,
            ring,
        };

        // The bar and weigh stages fill what a solve pass reads, and one solve pass leaves partials
        // for the finish kernel.
        self.launch_bands(&input);
        self.launch_pass(&input, FIT_STAGE_WEIGH, FIT_INLIER_CAP);
        self.launch_sums(&input, FIT_STAGE_SOLVE, FIT_SCALE);

        input
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        match self.kernel {
            FitKernel::Sums => self.launch_sums(&args, FIT_STAGE_SOLVE, FIT_SCALE),
            FitKernel::Finish => self.launch_finish(&args, FIT_STAGE_SOLVE),
            FitKernel::Whole => self.launch_whole(&args),
        }

        Ok(())
    }

    fn name(&self) -> String {
        match self.kernel {
            FitKernel::Sums => "mc_fit_sums_1080p".to_string(),
            FitKernel::Finish => "mc_fit_finish_1080p".to_string(),
            FitKernel::Whole => "mc_fit_model_1080p".to_string(),
        }
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        let blocks = (blocks_x() * blocks_y()) as usize;
        vec![vec![blocks, FIT_LANES as usize]]
    }
}

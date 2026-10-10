use av_denoise_core::bench_api::tune::{RegulariseLaunch, RegulariseShape, labels};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{HEIGHT, WIDTH, block_sync, make_synthetic_frame, shapes_with_channels};

const BLKSIZE: u32 = 16;
const STEP: u32 = 8;
const THSAD_PIXEL: f32 = 0.02;
const FIELD_LAMBDA: f32 = 1.0;

/// The nl4d field regularisation pass over one neighbour at 1080p.
pub struct MvRegulariseBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub candidate: usize,
}

#[derive(Clone)]
pub struct RegulariseInput {
    pub centre: Handle,
    pub neighbour: Handle,
    pub mv_in: Handle,
    pub mv_out: Handle,
    pub confidence: Handle,
}

impl<R: Runtime> Benchmark for MvRegulariseBench<R> {
    type Input = RegulariseInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let blocks_x = WIDTH.div_ceil(STEP);
        let blocks_y = HEIGHT.div_ceil(STEP);
        let blocks = (blocks_x * blocks_y) as usize;

        let centre_frame = make_synthetic_frame(WIDTH, HEIGHT, 1);
        let centre_bytes = f32::as_bytes(&centre_frame);
        let centre = self.client.create_from_slice(centre_bytes);
        let neighbour_frame = make_synthetic_frame(WIDTH, HEIGHT, 1);
        let neighbour_bytes = f32::as_bytes(&neighbour_frame);
        let neighbour = self.client.create_from_slice(neighbour_bytes);

        let field: Vec<i32> = (0..2 * blocks).map(|index| (index % 7) as i32 - 3).collect();
        let field_bytes = i32::as_bytes(&field);
        let mv_in = self.client.create_from_slice(field_bytes);
        let mv_out = self.client.empty(2 * blocks * size_of::<i32>());
        let confidence = self.client.empty(blocks * size_of::<f32>());

        RegulariseInput {
            centre,
            neighbour,
            mv_in,
            mv_out,
            confidence,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let blocks_x = WIDTH.div_ceil(STEP);
        let blocks_y = HEIGHT.div_ceil(STEP);
        let block_area = (BLKSIZE * BLKSIZE) as f32;
        let lambda = FIELD_LAMBDA * block_area * THSAD_PIXEL;
        let thsad = block_area * THSAD_PIXEL;

        let shape = RegulariseShape {
            level_width: WIDTH,
            level_height: HEIGHT,
            blksize: BLKSIZE,
            step: STEP,
            blocks_x,
            blocks_y,
        };
        let launch = RegulariseLaunch {
            client: self.client.clone(),
            centre: args.centre,
            neighbour: args.neighbour,
            level_len: (WIDTH * HEIGHT) as usize,
            mv_in: args.mv_in,
            mv_out: args.mv_out,
            conf_out: args.confidence,
            lambda_pixel: lambda,
            sad_noise_floor: 0.0,
            thsad,
            shape,
        };

        launch.launch_candidate(self.candidate)
    }

    fn name(&self) -> String {
        let label = labels::regularise(self.candidate);
        format!("nl4d_mv_regularise_1080p_luma_{label}")
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(1)
    }
}

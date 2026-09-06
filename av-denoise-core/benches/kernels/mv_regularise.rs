use av_denoise_core::nl4d::kernels::nl4d_mv_regularise;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{H, W, block_sync, make_synthetic_frame, shapes_with_ch};

const BLKSIZE: u32 = 16;
const STEP: u32 = 8;
const THSAD_PIXEL: f32 = 0.02;
const FIELD_LAMBDA: f32 = 1.0;

/// The nl4d field regularisation pass over one neighbour at 1080p.
pub struct MvRegulariseBench<R: Runtime> {
    pub client: ComputeClient<R>,
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
        let blocks = (W.div_ceil(STEP) * H.div_ceil(STEP)) as usize;
        let centre = self
            .client
            .create_from_slice(f32::as_bytes(&make_synthetic_frame(W, H, 1)));
        let neighbour = self
            .client
            .create_from_slice(f32::as_bytes(&make_synthetic_frame(W, H, 1)));
        let field: Vec<i32> = (0..2 * blocks).map(|i| (i % 7) as i32 - 3).collect();
        let mv_in = self.client.create_from_slice(i32::as_bytes(&field));
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
        let blocks_x = W.div_ceil(STEP);
        let blocks_y = H.div_ceil(STEP);
        let blocks = (blocks_x * blocks_y) as usize;
        let block_area = (BLKSIZE * BLKSIZE) as f32;
        unsafe {
            nl4d_mv_regularise::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_2d(blocks_x, blocks_y),
                CubeDim::new_2d(8, 8),
                ArrayArg::from_raw_parts(args.centre.clone(), (W * H) as usize),
                ArrayArg::from_raw_parts(args.neighbour.clone(), (W * H) as usize),
                ArrayArg::from_raw_parts(args.mv_in.clone(), 2 * blocks),
                ArrayArg::from_raw_parts(args.mv_out.clone(), 2 * blocks),
                ArrayArg::from_raw_parts(args.confidence.clone(), blocks),
                FIELD_LAMBDA * block_area * THSAD_PIXEL,
                0.0,
                block_area * THSAD_PIXEL,
                W,
                H,
                BLKSIZE,
                STEP,
                blocks_x,
                blocks_y,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        "nl4d_mv_regularise_1080p_luma".to_string()
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_ch(1)
    }
}

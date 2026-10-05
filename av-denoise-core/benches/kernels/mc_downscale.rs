use av_denoise_core::bench_api::kernels::motion::nlm_mc_downscale;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{HEIGHT, WIDTH, block_sync, make_synthetic_frame, shapes_with_channels};

/// A 2x2 box downsample of a full-resolution luma frame into a half-resolution slot.
///
/// It builds the coarse pyramid level for motion estimation.
pub struct DownscaleBench<R: Runtime> {
    pub client: ComputeClient<R>,
}

#[derive(Clone)]
pub struct DownscaleInput {
    pub src: Handle,
    pub dst: Handle,
}

impl<R: Runtime> Benchmark for DownscaleBench<R> {
    type Input = DownscaleInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let src_frame = make_synthetic_frame(WIDTH, HEIGHT, 1);
        let src_bytes = f32::as_bytes(&src_frame);
        let src = self.client.create_from_slice(src_bytes);
        let dst_width = WIDTH / 2;
        let dst_height = HEIGHT / 2;
        let dst = self
            .client
            .empty((dst_width * dst_height) as usize * size_of::<f32>());

        DownscaleInput { src, dst }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let dst_width = WIDTH / 2;
        let dst_height = HEIGHT / 2;
        let block_x = 16u32;
        let block_y = 16u32;
        let cubes_x = dst_width.div_ceil(block_x);
        let cubes_y = dst_height.div_ceil(block_y);
        let grid = CubeCount::new_2d(cubes_x, cubes_y);
        let dim = CubeDim::new_2d(block_x, block_y);

        unsafe {
            nlm_mc_downscale::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                ArrayArg::from_raw_parts(args.src.clone(), (WIDTH * HEIGHT) as usize),
                ArrayArg::from_raw_parts(args.dst.clone(), (dst_width * dst_height) as usize),
                0u32,
                0u32,
                WIDTH,
                HEIGHT,
                dst_width,
                dst_height,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        "mc_downscale_1080p_to_540p_luma".to_string()
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(1)
    }
}

use av_denoise_core::nlmeans::kernels::motion::nlm_mc_downscale;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{H, W, block_sync, make_synthetic_frame, shapes_with_ch};

/// 2x2 box downsample over a full-res luma frame into a `/2` slot.
/// Used to build the coarse pyramid level for motion estimation.
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
        let src_frame = make_synthetic_frame(W, H, 1);
        let src = self.client.create_from_slice(f32::as_bytes(&src_frame));
        let dst_w = W / 2;
        let dst_h = H / 2;
        let dst = self.client.empty((dst_w * dst_h) as usize * size_of::<f32>());
        DownscaleInput { src, dst }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let dst_w = W / 2;
        let dst_h = H / 2;
        let block_x = 16u32;
        let block_y = 16u32;
        let grid = CubeCount::new_2d(dst_w.div_ceil(block_x), dst_h.div_ceil(block_y));
        let dim = CubeDim::new_2d(block_x, block_y);

        unsafe {
            nlm_mc_downscale::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                ArrayArg::from_raw_parts(args.src.clone(), (W * H) as usize),
                ArrayArg::from_raw_parts(args.dst.clone(), (dst_w * dst_h) as usize),
                0u32,
                0u32,
                W,
                H,
                dst_w,
                dst_h,
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
        shapes_with_ch(1)
    }
}

use av_denoise::nlmeans::kernels::nlm_weight_ratio_partial;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{BLOCK_1D, COPY_GRID_1D, H, W, block_sync};

const SELF_WEIGHT: f32 = 1.0;

/// Reduces the per-pixel residual-noise ratio down to one partial sum per
/// block, at the library's fixed 1080p grid.
pub struct CollabWeightRatioBench<R: Runtime> {
    pub client: ComputeClient<R>,
}

#[derive(Clone)]
pub struct CollabWeightRatioInput {
    weight_sum: Handle,
    weight_sq_sum: Handle,
    max_weight: Handle,
    partials: Handle,
}

impl<R: Runtime> Benchmark for CollabWeightRatioBench<R> {
    type Input = CollabWeightRatioInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let pixels = (W * H) as usize;
        let ws_data = vec![4.0f32; pixels];
        let wsq_data = vec![2.0f32; pixels];
        let mw_data = vec![0.8f32; pixels];
        let weight_sum = self.client.create_from_slice(f32::as_bytes(&ws_data));
        let weight_sq_sum = self.client.create_from_slice(f32::as_bytes(&wsq_data));
        let max_weight = self.client.create_from_slice(f32::as_bytes(&mw_data));
        let partials = self.client.empty(COPY_GRID_1D as usize * size_of::<f32>());
        CollabWeightRatioInput {
            weight_sum,
            weight_sq_sum,
            max_weight,
            partials,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let pixels = (W * H) as u32;
        let total_threads = COPY_GRID_1D * BLOCK_1D;
        unsafe {
            nlm_weight_ratio_partial::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(COPY_GRID_1D),
                CubeDim::new_1d(BLOCK_1D),
                ArrayArg::from_raw_parts(args.weight_sum.clone(), pixels as usize),
                ArrayArg::from_raw_parts(args.weight_sq_sum.clone(), pixels as usize),
                ArrayArg::from_raw_parts(args.max_weight.clone(), pixels as usize),
                ArrayArg::from_raw_parts(args.partials.clone(), COPY_GRID_1D as usize),
                SELF_WEIGHT,
                pixels,
                total_threads,
                BLOCK_1D,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        "collab_weight_ratio_partial_1080p".to_string()
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![W as usize, H as usize, 1]]
    }
}

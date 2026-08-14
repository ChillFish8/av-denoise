use av_denoise::collab::geometry::{filtered_buf_len, ref_count, refs_along};
use av_denoise::collab::kernels::aggregate::collab_aggregate;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{BLOCK_X, BLOCK_Y, H, W, block_sync, stored_channels};

/// Blends a synthetic set of filtered patches back onto one 1080p frame
/// plane, at the library's default reference grid, for each channel
/// mode. Cost scales with `stored_ch`, since every covering candidate
/// reads and writes a full `Vector<f32, N>` line.
pub struct CollabAggregateBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub ch: u32,
    pub ch_name: &'static str,
}

#[derive(Clone)]
pub struct CollabAggregateInput {
    pub filtered: Handle,
    pub group_weight: Handle,
    pub output: Handle,
}

impl<R: Runtime> Benchmark for CollabAggregateBench<R> {
    type Input = CollabAggregateInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let stored = stored_channels(self.ch) as usize;
        let refs = ref_count(W, H);
        let filt_len = filtered_buf_len(W, H);
        let out_len = (W * H) as usize;

        let filtered_data: Vec<f32> = (0..filt_len * stored).map(|i| (i % 97) as f32 * 0.01).collect();
        let weight_data: Vec<f32> = (0..refs).map(|i| 1.0 + (i % 11) as f32 * 0.1).collect();

        let filtered = self.client.create_from_slice(f32::as_bytes(&filtered_data));
        let group_weight = self.client.create_from_slice(f32::as_bytes(&weight_data));
        let output = self.client.empty(out_len * stored * size_of::<f32>());

        CollabAggregateInput {
            filtered,
            group_weight,
            output,
        }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let stored = stored_channels(self.ch) as usize;
        let refs = ref_count(W, H);
        let filt_len = filtered_buf_len(W, H);
        let out_len = (W * H) as usize;
        let refs_x = refs_along(W);
        let refs_y = refs_along(H);

        let grid = CubeCount::new_2d(W.div_ceil(BLOCK_X), H.div_ceil(BLOCK_Y));
        let dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);

        unsafe {
            collab_aggregate::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                stored,
                ArrayArg::from_raw_parts(args.filtered.clone(), filt_len * stored),
                ArrayArg::from_raw_parts(args.group_weight.clone(), refs),
                ArrayArg::from_raw_parts(args.output.clone(), out_len * stored),
                W,
                H,
                refs_x,
                refs_y,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        format!("collab_aggregate_1080p_{}", self.ch_name)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![W as usize, H as usize, self.ch as usize]]
    }
}

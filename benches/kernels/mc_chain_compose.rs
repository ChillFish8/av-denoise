use av_denoise::nlmeans::kernels::motion::nlm_mc_chain_compose;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{H, W};

const CHAIN_STEP: u32 = 8;
const CHAIN_RADIUS: u32 = 4;
const CHAIN_PAIR_RING_SLOTS: u32 = 2 * CHAIN_RADIUS;

/// Chained motion-composition kernel at 1080p geometry, walking a
/// full `R = 4` chain (the deepest hop the default temporal radius
/// exercises). The pair ring holds synthetic (never-analysed) data;
/// the kernel's cost is dominated by the per-block hop walk, not the
/// values it reads, so the content doesn't need to be meaningful.
pub struct ChainComposeBench<R: Runtime> {
    pub client: ComputeClient<R>,
}

#[derive(Clone)]
pub struct ChainComposeInput {
    pub pair_ring: Handle,
    pub mv_field: Handle,
}

impl<R: Runtime> Benchmark for ChainComposeBench<R> {
    type Input = ChainComposeInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let blocks_x = W.div_ceil(CHAIN_STEP);
        let blocks_y = H.div_ceil(CHAIN_STEP);
        let dir_len = (blocks_x * blocks_y * 2) as usize;
        let pair_ring_len = CHAIN_PAIR_RING_SLOTS as usize * 2 * dir_len;

        let pair_ring_data = vec![0i32; pair_ring_len];
        let pair_ring = self.client.create_from_slice(i32::as_bytes(&pair_ring_data));
        let mv_field = self.client.empty(dir_len * size_of::<i32>());

        ChainComposeInput { pair_ring, mv_field }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let blocks_x = W.div_ceil(CHAIN_STEP);
        let blocks_y = H.div_ceil(CHAIN_STEP);
        let dir_len = (blocks_x * blocks_y * 2) as usize;
        let pair_ring_len = CHAIN_PAIR_RING_SLOTS as usize * 2 * dir_len;

        // One thread per output block, matching `run_chain_compose`'s
        // production launch shape.
        let grid = CubeCount::new_2d(blocks_x, blocks_y);
        let dim = CubeDim::new_2d(1, 1);

        unsafe {
            nlm_mc_chain_compose::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                ArrayArg::from_raw_parts(args.pair_ring.clone(), pair_ring_len),
                ArrayArg::from_raw_parts(args.mv_field.clone(), dir_len),
                0u32,
                true,
                CHAIN_RADIUS,
                CHAIN_PAIR_RING_SLOTS,
                CHAIN_STEP,
                W,
                H,
                blocks_x,
                blocks_y,
            );
        }
        Ok(())
    }

    fn name(&self) -> String {
        "mc_chain_compose_1080p_r4".to_string()
    }

    fn sync(&self) {
        super::block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        super::shapes_with_ch(1)
    }
}

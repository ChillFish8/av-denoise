use av_denoise_core::bench_api::kernels::motion::nlm_mc_chain_compose;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{HEIGHT, WIDTH, block_sync, shapes_with_channels};

const CHAIN_STEP: u32 = 8;
const CHAIN_RADIUS: u32 = 4;
const CHAIN_PAIR_RING_SLOTS: u32 = 2 * CHAIN_RADIUS;

/// The padded per-direction pair-ring stride in `i32` elements.
///
/// It mirrors `MotionCtx::pair_direction_stride`, which is crate-private. Each block holds one
/// `(dx, dy)` pair, and the total is rounded up to a 32-byte storage-buffer offset alignment.
fn padded_pair_direction_stride(blocks_x: u32, blocks_y: u32) -> u32 {
    let unpadded_bytes = (blocks_x as u64) * (blocks_y as u64) * 2 * size_of::<i32>() as u64;
    let padded_bytes = unpadded_bytes.next_multiple_of(32);

    (padded_bytes / size_of::<i32>() as u64) as u32
}

/// The chained motion-composition kernel at 1080p, walking a full `R = 4` chain.
///
/// The pair ring holds synthetic data that was never analysed. The cost is dominated by the
/// per-block hop walk rather than the values it reads, so the content does not need to be
/// meaningful.
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
        let blocks_x = WIDTH.div_ceil(CHAIN_STEP);
        let blocks_y = HEIGHT.div_ceil(CHAIN_STEP);
        let dir_len = padded_pair_direction_stride(blocks_x, blocks_y);
        let slot_len = 2 * dir_len;
        let pair_ring_len = CHAIN_PAIR_RING_SLOTS as usize * slot_len as usize;

        let pair_ring_data = vec![0i32; pair_ring_len];
        let pair_ring_bytes = i32::as_bytes(&pair_ring_data);
        let pair_ring = self.client.create_from_slice(pair_ring_bytes);
        let mv_field = self
            .client
            .empty((blocks_x * blocks_y * 2) as usize * size_of::<i32>());

        ChainComposeInput { pair_ring, mv_field }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let blocks_x = WIDTH.div_ceil(CHAIN_STEP);
        let blocks_y = HEIGHT.div_ceil(CHAIN_STEP);
        let dir_len = padded_pair_direction_stride(blocks_x, blocks_y);
        let slot_len = 2 * dir_len;
        let pair_ring_len = CHAIN_PAIR_RING_SLOTS as usize * slot_len as usize;

        // One thread per output block, the same launch shape the library uses.
        let grid = CubeCount::new_2d(blocks_x, blocks_y);
        let dim = CubeDim::new_2d(1, 1);

        unsafe {
            nlm_mc_chain_compose::launch_unchecked::<R>(
                &self.client,
                grid,
                dim,
                ArrayArg::from_raw_parts(args.pair_ring.clone(), pair_ring_len),
                ArrayArg::from_raw_parts(args.mv_field.clone(), (blocks_x * blocks_y * 2) as usize),
                0u32,
                true,
                CHAIN_RADIUS,
                CHAIN_PAIR_RING_SLOTS,
                dir_len,
                slot_len,
                CHAIN_STEP,
                WIDTH,
                HEIGHT,
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
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(1)
    }
}

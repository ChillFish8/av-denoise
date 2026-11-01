use av_denoise_core::bench_api::kernels::motion::mc_compose_models;
use av_denoise_core::bench_api::motion::{AFFINE_WORDS, ModelRing, compose_gaps};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::block_sync;

const COMPOSE_RADIUS: u32 = 4;
const COMPOSE_NEIGHBOURS: u32 = 2 * COMPOSE_RADIUS;
const COMPOSE_RING_DEPTH: u32 = 16;

/// The camera-model composition kernel over a full `R = 4` window.
///
/// The ring holds zeroed models. Every gap walks its full chain whatever the values, so the
/// content does not change the cost.
pub struct ComposeModelsBench {
    pub client: Client,
}

#[derive(Clone)]
pub struct ComposeModelsInput {
    pub ring: ModelRing,
    pub gaps: Handle,
    pub out: Handle,
}

impl Benchmark for ComposeModelsBench {
    type Input = ComposeModelsInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let ring = ModelRing::new(&self.client, COMPOSE_RING_DEPTH);
        let slot_frames: Vec<u32> = (100..100 + COMPOSE_NEIGHBOURS + 1).collect();
        let centre = COMPOSE_RADIUS as usize;
        let radius = COMPOSE_RADIUS as usize;
        let gap_table = compose_gaps(&slot_frames, centre, radius, &ring);
        let gap_bytes = u32::as_bytes(&gap_table);
        let gaps = self.client.create_from_slice(gap_bytes);
        let out_bytes = (COMPOSE_NEIGHBOURS * AFFINE_WORDS) as usize * size_of::<f32>();
        let out = self.client.empty(out_bytes);

        ComposeModelsInput { ring, gaps, out }
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let gaps_len = (COMPOSE_NEIGHBOURS * COMPOSE_RADIUS) as usize;
        let out_len = (COMPOSE_NEIGHBOURS * AFFINE_WORDS) as usize;

        // One cube with a thread per neighbour, the same launch shape the library uses.
        let grid = CubeCount::new_1d(1);
        let dim = CubeDim::new_1d(COMPOSE_NEIGHBOURS);

        let models = args.ring.handle().clone();
        let models_len = args.ring.words();

        unsafe {
            let models_arg = BufferArg::from_raw_parts(models, models_len);
            let gaps_arg = BufferArg::from_raw_parts(args.gaps, gaps_len);
            let out_arg = BufferArg::from_raw_parts(args.out, out_len);

            mc_compose_models::launch_unchecked(
                &self.client,
                grid,
                dim,
                models_arg,
                gaps_arg,
                out_arg,
                COMPOSE_NEIGHBOURS,
                COMPOSE_RADIUS,
                2.0f32,
                2.0f32,
            );
        }

        Ok(())
    }

    fn name(&self) -> String {
        "mc_compose_models_r4".to_string()
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![vec![COMPOSE_NEIGHBOURS as usize, AFFINE_WORDS as usize]]
    }
}

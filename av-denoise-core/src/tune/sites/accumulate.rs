use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{AccumOutputs, NLM_ACCUMULATE, allocate_accum, clear_accum, compare_accum};
use crate::nlmeans::kernels::nlm_accumulate;
use crate::tune::fidelity::{CompareResult, Fidelity};
use crate::tune::gpu::bind;
use crate::tune::key::SiteKey;
use crate::tune::site::{LaunchShape, SiteCandidate, VariantSite};

/// A cube shape for the per-pixel `nlm_accumulate` kernel, which has no shared-memory tile.
///
/// Every pixel's arithmetic is independent of the cube shape, so every shape is bit identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulateCandidate {
    /// The baseline.
    ///
    /// On RDNA it runs 10 to 25% faster than 32x8 because the extra threads hide memory latency.
    Thin32x16,
    Thin32x8,
}

impl AccumulateCandidate {
    /// The cube's width and height in threads.
    pub fn block(self) -> (u32, u32) {
        match self {
            AccumulateCandidate::Thin32x16 => (32, 16),
            AccumulateCandidate::Thin32x8 => (32, 8),
        }
    }
}

impl SiteCandidate for AccumulateCandidate {
    const SITE: &'static str = NLM_ACCUMULATE;
    const ALL: &'static [Self] = &[AccumulateCandidate::Thin32x16, AccumulateCandidate::Thin32x8];

    fn name(self) -> &'static str {
        match self {
            AccumulateCandidate::Thin32x16 => "32x16",
            AccumulateCandidate::Thin32x8 => "32x8",
        }
    }

    fn fidelity(self) -> Fidelity {
        Fidelity::Exact
    }
}

/// One `nlm_accumulate` launch for the frame against itself at search offset `(q_x, q_y)`.
///
/// The weight map is symmetric there, so `weights` serves both the forward and backward lookups and
/// both neighbours come from `frame`.
#[derive(Clone)]
pub struct AccumulateSite<R: Runtime> {
    pub client: ComputeClient<R>,
    pub input: Handle,
    /// Elements in `input`.
    pub ring_len: usize,
    pub weights: Handle,
    pub outputs: AccumOutputs,
    pub frame: u32,
    pub q_x: i32,
    pub q_y: i32,
    pub width: u32,
    pub height: u32,
    pub stored_ch: u32,
}

impl<R: Runtime> VariantSite<R> for AccumulateSite<R> {
    type Candidate = AccumulateCandidate;
    type Outputs = AccumOutputs;

    fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    fn key(&self) -> SiteKey {
        SiteKey::builder(NLM_ACCUMULATE, self.width, self.height)
            .param("stored", self.stored_ch)
            .build()
    }

    fn shape(&self, candidate: AccumulateCandidate) -> LaunchShape {
        let (x, y) = candidate.block();
        LaunchShape { x, y }
    }

    fn shared_bytes(&self, _candidate: AccumulateCandidate) -> usize {
        0
    }

    fn launch(&self, candidate: AccumulateCandidate, outputs: &AccumOutputs) {
        let (block_x, block_y) = candidate.block();
        let cubes_x = self.width.div_ceil(block_x);
        let cubes_y = self.height.div_ceil(block_y);
        let cube_count = CubeCount::new_2d(cubes_x, cubes_y);
        let cube_dim = CubeDim::new_2d(block_x, block_y);
        let vector_size = self.stored_ch as usize;

        let input = bind::<R>(&self.input, self.ring_len);
        let accum = bind::<R>(&outputs.accum, outputs.accum_len);
        let weight_sum = bind::<R>(&outputs.weight_sum, outputs.pixels);
        let weights_fwd = bind::<R>(&self.weights, outputs.pixels);
        let weights_bwd = bind::<R>(&self.weights, outputs.pixels);
        let max_weight = bind::<R>(&outputs.max_weight, outputs.pixels);

        unsafe {
            nlm_accumulate::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                vector_size,
                input,
                accum,
                weight_sum,
                weights_fwd,
                weights_bwd,
                max_weight,
                self.frame,
                self.frame,
                self.q_x,
                self.q_y,
                self.width,
                self.height,
            );
        }
    }

    fn real_outputs(&self) -> AccumOutputs {
        self.outputs.clone()
    }

    fn scratch_outputs(&self) -> AccumOutputs {
        allocate_accum(&self.client, self.outputs.accum_len, self.outputs.pixels)
    }

    fn clear_outputs(&self, outputs: &AccumOutputs) {
        clear_accum(&self.client, outputs);
    }

    fn compare(
        &self,
        baseline: &AccumOutputs,
        candidate: &AccumOutputs,
        fidelity: Fidelity,
    ) -> Result<CompareResult, anyhow::Error> {
        compare_accum(&self.client, baseline, candidate, fidelity)
    }
}

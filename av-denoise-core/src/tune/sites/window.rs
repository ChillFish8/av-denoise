use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{AccumOutputs, NLM_WINDOW, allocate_accum, clear_accum, compare_accum};
use crate::nlmeans::kernels::{
    nlm_fused_pair_accumulate_window,
    nlm_fused_pair_accumulate_window_ref,
    nlm_fused_single_window,
    nlm_fused_single_window_ref,
};
use crate::tune::fidelity::{CompareResult, Fidelity};
use crate::tune::gpu::bind;
use crate::tune::key::SiteKey;
use crate::tune::site::{LaunchShape, SiteCandidate, VariantSite};

/// A cube shape for the windowed NLM kernels.
///
/// Each pixel walks the same search window in the same order whatever the cube shape, so every
/// shape is bit identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowCandidate {
    Block32x8,
    Block32x16,
    Block16x16,
}

impl WindowCandidate {
    /// The cube's width and height in threads.
    pub fn block(self) -> (u32, u32) {
        match self {
            WindowCandidate::Block32x8 => (32, 8),
            WindowCandidate::Block32x16 => (32, 16),
            WindowCandidate::Block16x16 => (16, 16),
        }
    }
}

impl SiteCandidate for WindowCandidate {
    const SITE: &'static str = NLM_WINDOW;
    const ALL: &'static [Self] = &[
        WindowCandidate::Block32x8,
        WindowCandidate::Block32x16,
        WindowCandidate::Block16x16,
    ];

    fn name(self) -> &'static str {
        match self {
            WindowCandidate::Block32x8 => "32x8",
            WindowCandidate::Block32x16 => "32x16",
            WindowCandidate::Block16x16 => "16x16",
        }
    }

    fn fidelity(self) -> Fidelity {
        Fidelity::Exact
    }
}

/// Which windowed kernel a launch runs.
#[derive(Debug, Clone)]
pub enum WindowPass {
    /// The frame against itself, with one noise offset per search candidate in `offsets`.
    Single { offsets: Handle, offsets_len: usize },
    /// A forward and a backward temporal neighbour.
    Pair {
        frame_fwd: u32,
        frame_bwd: u32,
        noise_offset: f32,
        confidence: ConfidenceBinding,
    },
}

/// The per-block confidence slices a temporal pair weights by.
///
/// With `enabled` off the kernel never reads `forward` or `backward`.
#[derive(Debug, Clone)]
pub struct ConfidenceBinding {
    pub enabled: bool,
    pub forward: Handle,
    pub backward: Handle,
    /// Elements in each slice, one per motion block.
    pub len: usize,
    pub step: u32,
    pub blocks_x: u32,
    pub blocks_y: u32,
}

/// One windowed NLM launch over the whole search window.
#[derive(Clone)]
pub struct WindowSite<R: Runtime> {
    pub client: ComputeClient<R>,
    pub pass: WindowPass,
    pub input: Handle,
    /// The prefiltered ring the distances read, when the pipeline has one.
    pub reference: Option<Handle>,
    /// Elements in `input`, and in `reference` when it is set.
    pub ring_len: usize,
    pub outputs: AccumOutputs,
    pub frame_t: u32,
    pub h2_inv_norm: f32,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub stored_ch: u32,
    pub patch_radius: u32,
    pub search_radius: u32,
}

/// The grid and cube one candidate launches.
struct WindowGrid {
    cube_count: CubeCount,
    cube_dim: CubeDim,
    block_x: u32,
    block_y: u32,
}

impl<R: Runtime> WindowSite<R> {
    fn grid(&self, candidate: WindowCandidate) -> WindowGrid {
        let (block_x, block_y) = candidate.block();
        let cubes_x = self.width.div_ceil(block_x);
        let cubes_y = self.height.div_ceil(block_y);

        WindowGrid {
            cube_count: CubeCount::new_2d(cubes_x, cubes_y),
            cube_dim: CubeDim::new_2d(block_x, block_y),
            block_x,
            block_y,
        }
    }

    fn launch_single(&self, grid: WindowGrid, offsets: &Handle, offsets_len: usize, outputs: &AccumOutputs) {
        let vector_size = self.stored_ch as usize;
        let input = bind::<R>(&self.input, self.ring_len);
        let accum = bind::<R>(&outputs.accum, outputs.accum_len);
        let weight_sum = bind::<R>(&outputs.weight_sum, outputs.pixels);
        let max_weight = bind::<R>(&outputs.max_weight, outputs.pixels);
        let offsets = bind::<R>(offsets, offsets_len);

        match &self.reference {
            Some(reference) => {
                let reference = bind::<R>(reference, self.ring_len);

                unsafe {
                    nlm_fused_single_window_ref::launch_unchecked::<R>(
                        &self.client,
                        grid.cube_count,
                        grid.cube_dim,
                        vector_size,
                        input,
                        reference,
                        accum,
                        weight_sum,
                        max_weight,
                        self.frame_t,
                        self.h2_inv_norm,
                        offsets,
                        self.width,
                        self.height,
                        self.channels,
                        self.patch_radius,
                        self.search_radius,
                        grid.block_x,
                        grid.block_y,
                    );
                }
            },
            None => unsafe {
                nlm_fused_single_window::launch_unchecked::<R>(
                    &self.client,
                    grid.cube_count,
                    grid.cube_dim,
                    vector_size,
                    input,
                    accum,
                    weight_sum,
                    max_weight,
                    self.frame_t,
                    self.h2_inv_norm,
                    offsets,
                    self.width,
                    self.height,
                    self.channels,
                    self.patch_radius,
                    self.search_radius,
                    grid.block_x,
                    grid.block_y,
                );
            },
        }
    }

    fn launch_pair(
        &self,
        grid: WindowGrid,
        frames: (u32, u32),
        noise_offset: f32,
        confidence: &ConfidenceBinding,
        outputs: &AccumOutputs,
    ) {
        let (frame_fwd, frame_bwd) = frames;
        let vector_size = self.stored_ch as usize;
        let input = bind::<R>(&self.input, self.ring_len);
        let accum = bind::<R>(&outputs.accum, outputs.accum_len);
        let weight_sum = bind::<R>(&outputs.weight_sum, outputs.pixels);
        let max_weight = bind::<R>(&outputs.max_weight, outputs.pixels);
        let conf_fwd = bind::<R>(&confidence.forward, confidence.len);
        let conf_bwd = bind::<R>(&confidence.backward, confidence.len);

        match &self.reference {
            Some(reference) => {
                let reference = bind::<R>(reference, self.ring_len);

                unsafe {
                    nlm_fused_pair_accumulate_window_ref::launch_unchecked::<R>(
                        &self.client,
                        grid.cube_count,
                        grid.cube_dim,
                        vector_size,
                        input,
                        reference,
                        accum,
                        weight_sum,
                        max_weight,
                        conf_fwd,
                        conf_bwd,
                        confidence.enabled,
                        self.frame_t,
                        frame_fwd,
                        frame_bwd,
                        self.h2_inv_norm,
                        noise_offset,
                        self.width,
                        self.height,
                        self.channels,
                        self.patch_radius,
                        self.search_radius,
                        grid.block_x,
                        grid.block_y,
                        confidence.step,
                        confidence.blocks_x,
                        confidence.blocks_y,
                    );
                }
            },
            None => unsafe {
                nlm_fused_pair_accumulate_window::launch_unchecked::<R>(
                    &self.client,
                    grid.cube_count,
                    grid.cube_dim,
                    vector_size,
                    input,
                    accum,
                    weight_sum,
                    max_weight,
                    conf_fwd,
                    conf_bwd,
                    confidence.enabled,
                    self.frame_t,
                    frame_fwd,
                    frame_bwd,
                    self.h2_inv_norm,
                    noise_offset,
                    self.width,
                    self.height,
                    self.channels,
                    self.patch_radius,
                    self.search_radius,
                    grid.block_x,
                    grid.block_y,
                    confidence.step,
                    confidence.blocks_x,
                    confidence.blocks_y,
                );
            },
        }
    }
}

impl<R: Runtime> VariantSite<R> for WindowSite<R> {
    type Candidate = WindowCandidate;
    type Outputs = AccumOutputs;

    fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    fn key(&self) -> SiteKey {
        let (kernel, confidence, step) = match &self.pass {
            WindowPass::Single { .. } => ("single", false, 0),
            WindowPass::Pair { confidence, .. } => ("pair", confidence.enabled, confidence.step),
        };
        let has_reference = self.reference.is_some();

        SiteKey::builder(NLM_WINDOW, self.width, self.height)
            .param("kernel", kernel)
            .param("ref", u8::from(has_reference))
            .param("conf", u8::from(confidence))
            .param("step", step)
            .param("ch", self.channels)
            .param("stored", self.stored_ch)
            .param("pr", self.patch_radius)
            .param("sr", self.search_radius)
            .build()
    }

    fn shape(&self, candidate: WindowCandidate) -> LaunchShape {
        let (x, y) = candidate.block();
        LaunchShape { x, y }
    }

    fn shared_bytes(&self, candidate: WindowCandidate) -> usize {
        let (block_x, block_y) = candidate.block();
        let halo = 2 * self.patch_radius;
        let reach = halo + 2 * self.search_radius;
        let expanded = ((block_x + reach) * (block_y + reach)) as usize;
        let tile = ((block_x + halo) * (block_y + halo)) as usize;
        let vector_bytes = self.stored_ch as usize * size_of::<f32>();
        let distance_tiles = match self.pass {
            WindowPass::Single { .. } => 1,
            WindowPass::Pair { .. } => 2,
        };

        expanded * vector_bytes + distance_tiles * tile * size_of::<f32>()
    }

    fn launch(&self, candidate: WindowCandidate, outputs: &AccumOutputs) {
        let grid = self.grid(candidate);

        match &self.pass {
            WindowPass::Single { offsets, offsets_len } => {
                self.launch_single(grid, offsets, *offsets_len, outputs);
            },
            WindowPass::Pair {
                frame_fwd,
                frame_bwd,
                noise_offset,
                confidence,
            } => {
                let frames = (*frame_fwd, *frame_bwd);
                self.launch_pair(grid, frames, *noise_offset, confidence, outputs);
            },
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

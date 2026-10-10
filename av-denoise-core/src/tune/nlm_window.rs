use cubecl::AutotuneKey;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};
use serde::{Deserialize, Serialize};

use super::{TuneId, alternatives, fits, zeroed};
use crate::nlmeans::kernels::{nlm_fused_pair_accumulate_window, nlm_fused_single_window};
use crate::nlmeans::{BLOCK_X, BLOCK_Y};

/// Every fused window cube shape the tuner tries. The first is the default launch.
pub const WINDOW_CANDIDATES: [(u32, u32); 3] = [(BLOCK_X, BLOCK_Y), (32, 16), (16, 16)];

pub fn candidate_label(index: usize) -> String {
    let (block_x, block_y) = WINDOW_CANDIDATES[index];
    format!("c{index}_{block_x}x{block_y}")
}

#[derive(AutotuneKey, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct WindowKey {
    pair: bool,
    channels: u32,
    patch_radius: u32,
    search_radius: u32,
    use_confidence: bool,
    #[autotune(anchor)]
    width: usize,
    #[autotune(anchor)]
    height: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct WindowShape {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub stored_ch: u32,
    pub patch_radius: u32,
    pub search_radius: u32,
}

/// Per-block confidence for a pair pass. With `use_confidence` off the kernel never reads it.
#[derive(Clone)]
pub struct WindowConfidence {
    pub use_confidence: bool,
    pub conf_fwd: Handle,
    pub conf_bwd: Handle,
    pub conf_len: usize,
    pub step: u32,
    pub blocks_x: u32,
    pub blocks_y: u32,
}

#[derive(Clone)]
pub enum WindowPass {
    Single {
        frame_t: u32,
        offset_lut: Handle,
        offset_lut_len: usize,
    },
    Pair {
        frame_t: u32,
        frame_fwd: u32,
        frame_bwd: u32,
        noise_offset: f32,
        confidence: WindowConfidence,
    },
}

/// One fused window launch, with everything a candidate needs to run it.
#[derive(Clone)]
pub struct WindowLaunch<R: Runtime> {
    pub client: ComputeClient<R>,
    pub input: Handle,
    pub input_len: usize,
    pub accum: Handle,
    pub frame_size: usize,
    pub weight_sum: Handle,
    pub max_weight: Handle,
    pub pixels: usize,
    pub h2_inv_norm: f32,
    pub pass: WindowPass,
    pub shape: WindowShape,
}

impl<R: Runtime> WindowLaunch<R> {
    fn is_pair(&self) -> bool {
        matches!(self.pass, WindowPass::Pair { .. })
    }

    pub(crate) fn key(&self) -> WindowKey {
        let use_confidence = match &self.pass {
            WindowPass::Pair { confidence, .. } => confidence.use_confidence,
            WindowPass::Single { .. } => false,
        };
        let shape = self.shape;

        WindowKey::new(
            self.is_pair(),
            shape.channels,
            shape.patch_radius,
            shape.search_radius,
            use_confidence,
            shape.width as usize,
            shape.height as usize,
        )
    }

    /// Shared memory a `block_x` by `block_y` cube of this launch allocates.
    ///
    /// The centre tile carries the search apron and every stored channel. A single pass adds one
    /// patch tile of distances and a pair pass adds two.
    pub(crate) fn shared_bytes(&self, block_x: u32, block_y: u32) -> usize {
        let shape = self.shape;
        let patch_apron = 2 * shape.patch_radius;
        let search_apron = 2 * shape.search_radius;
        let expanded_x = block_x + patch_apron + search_apron;
        let expanded_y = block_y + patch_apron + search_apron;
        let expanded = (expanded_x * expanded_y * shape.stored_ch) as usize;
        let tile = ((block_x + patch_apron) * (block_y + patch_apron)) as usize;
        let tiles = if self.is_pair() { 2 } else { 1 };

        (expanded + tiles * tile) * size_of::<f32>()
    }

    /// A copy whose accumulators are fresh zeroed scratch.
    pub(crate) fn with_scratch(&self) -> Self {
        let accum_bytes = self.frame_size * size_of::<f32>();
        let pixel_bytes = self.pixels * size_of::<f32>();

        Self {
            accum: zeroed(&self.client, accum_bytes),
            weight_sum: zeroed(&self.client, pixel_bytes),
            max_weight: zeroed(&self.client, pixel_bytes),
            ..self.clone()
        }
    }

    pub fn launch_candidate(&self, index: usize) -> Result<(), String> {
        let (block_x, block_y) = WINDOW_CANDIDATES[index];
        let hardware = &self.client.properties().hardware;
        let shared_bytes = self.shared_bytes(block_x, block_y);
        if index != 0 && !fits(hardware, block_x * block_y, shared_bytes) {
            let label = candidate_label(index);
            return Err(format!("{label} does not fit this device"));
        }

        let shape = self.shape;
        let cubes_x = shape.width.div_ceil(block_x);
        let cubes_y = shape.height.div_ceil(block_y);
        let grid = CubeCount::new_2d(cubes_x, cubes_y);
        let dim = CubeDim::new_2d(block_x, block_y);
        let vector_size = shape.stored_ch as usize;
        let separable = std::env::var("AVD_PROBE_SEPARABLE").is_ok();

        match &self.pass {
            WindowPass::Single {
                frame_t,
                offset_lut,
                offset_lut_len,
            } => unsafe {
                nlm_fused_single_window::launch_unchecked::<R>(
                    &self.client,
                    grid,
                    dim,
                    vector_size,
                    ArrayArg::from_raw_parts(self.input.clone(), self.input_len),
                    ArrayArg::from_raw_parts(self.accum.clone(), self.frame_size),
                    ArrayArg::from_raw_parts(self.weight_sum.clone(), self.pixels),
                    ArrayArg::from_raw_parts(self.max_weight.clone(), self.pixels),
                    *frame_t,
                    self.h2_inv_norm,
                    ArrayArg::from_raw_parts(offset_lut.clone(), *offset_lut_len),
                    shape.width,
                    shape.height,
                    shape.channels,
                    shape.patch_radius,
                    shape.search_radius,
                    block_x,
                    block_y,
                    separable,
                );
            },
            WindowPass::Pair {
                frame_t,
                frame_fwd,
                frame_bwd,
                noise_offset,
                confidence,
            } => unsafe {
                nlm_fused_pair_accumulate_window::launch_unchecked::<R>(
                    &self.client,
                    grid,
                    dim,
                    vector_size,
                    ArrayArg::from_raw_parts(self.input.clone(), self.input_len),
                    ArrayArg::from_raw_parts(self.accum.clone(), self.frame_size),
                    ArrayArg::from_raw_parts(self.weight_sum.clone(), self.pixels),
                    ArrayArg::from_raw_parts(self.max_weight.clone(), self.pixels),
                    ArrayArg::from_raw_parts(confidence.conf_fwd.clone(), confidence.conf_len),
                    ArrayArg::from_raw_parts(confidence.conf_bwd.clone(), confidence.conf_len),
                    confidence.use_confidence,
                    *frame_t,
                    *frame_fwd,
                    *frame_bwd,
                    self.h2_inv_norm,
                    *noise_offset,
                    shape.width,
                    shape.height,
                    shape.channels,
                    shape.patch_radius,
                    shape.search_radius,
                    block_x,
                    block_y,
                    confidence.step,
                    confidence.blocks_x,
                    confidence.blocks_y,
                    separable,
                );
            },
        }

        Ok(())
    }
}

/// Runs the fastest fused window candidate for this device, tuning on the first call for a key.
pub(crate) fn launch<R: Runtime>(window: WindowLaunch<R>) {
    static TUNER: LocalTuner<WindowKey, TuneId> = local_tuner!("nlm-window");

    let tunables = TUNER.init(|| {
        let group = alternatives();
        let mut set = TunableSet::new(
            |window: &WindowLaunch<R>| window.key(),
            |_key: &WindowKey, window: &WindowLaunch<R>| window.with_scratch(),
        );

        for index in 0..WINDOW_CANDIDATES.len() {
            let label = candidate_label(index);
            let tunable = Tunable::new(&label, move |window: WindowLaunch<R>| {
                window.launch_candidate(index)
            });

            set = match index {
                0 => set.with(tunable),
                _ => set.with(tunable.group(&group, |_key| 0)),
            };
        }

        set
    });

    let client = window.client.clone();
    let tune_id = TuneId::new(&client);
    TUNER.execute(&tune_id, &client, tunables, window);
}

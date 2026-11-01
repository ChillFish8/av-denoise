use cubecl::prelude::*;
use cubecl::server::Handle;

use super::ModelRing;
use crate::motion::MotionCtx;
use crate::motion::kernels::{mc_fit_bands, mc_fit_finish, mc_fit_sums};
use crate::tune::gpu::{bind, zeroed_words};

/// Threads in every fit workgroup, one block each in the sums kernel.
pub const FIT_THREADS: u32 = 256;

/// The weighted sums each fit workgroup reduces.
pub const FIT_LANES: u32 = 8;

/// The `f32` words of the fit's running state.
pub const FIT_STATE_WORDS: u32 = 8;

/// The stage that weighs each block and measures the mean SAD.
pub const FIT_STAGE_WEIGH: u32 = 0;

/// The stage that scores the prior and starts the fit from it when it would be accepted.
pub const FIT_STAGE_GATE: u32 = 1;

/// The stage that solves one reweighted least-squares pass.
pub const FIT_STAGE_SOLVE: u32 = 2;

/// The stage that solves one equal-weight least-squares pass over the previous model's inliers.
pub const FIT_STAGE_POLISH: u32 = 3;

/// The stage that scores the fit against no motion and writes the model.
pub const FIT_STAGE_SCORE: u32 = 4;

/// The `u32` words of the bar bounds [mc_fit_bands](crate::motion::kernels::mc_fit_bands()) writes.
pub const FIT_BAND_WORDS: u32 = 4;

/// The residual scale of each reweighting pass, in luma pixels.
///
/// Each pass weights a block by `1 / (1 + (residual / scale)^2)^2` against the previous pass's
/// model, so blocks moving on their own fade out as the scale shrinks.
pub const FIT_SCALES: [f32; 4] = [16.0, 8.0, 4.0, 2.0];

/// The residual, in luma pixels, within which a block counts as an inlier when polishing and
/// scoring.
pub const FIT_INLIER_CAP: f32 = 2.0;

/// The tangent of the largest roll a model may carry, 15 degrees.
pub const MAX_ROLL_TAN: f32 = 0.267_949_2;

/// One direction of a pair field to fit a camera model to.
pub(crate) struct FitField<'a> {
    /// One pair slot, with both directions laid out as [MotionCtx::pair_direction_bytes] describes.
    pub pairs: &'a Handle,
    /// 0 fits the older-to-newer direction and 1 the newer-to-older one.
    pub direction: u32,
    /// The winning SAD of each block in that direction, one `f32` per block in block-row-major
    /// order.
    ///
    /// Only each SAD's size relative to the field's mean matters, so any noise level works. Without
    /// it every block counts as an equally good match.
    pub sad: Option<&'a Handle>,
    /// Whether `sad` also holds each block's window mean SAD and then its mean sample, each one
    /// [MotionCtx::conf_stride] after the last.
    ///
    /// A block whose winner barely stands out from its window, such as a flat wall, then counts
    /// for little in the solve and in the score. Only flat, dark still blocks can form bars.
    pub has_contrast: bool,
}

/// The device buffers one fit works in, reusable from fit to fit.
#[derive(Debug, Clone)]
pub struct FitScratch {
    bands: Handle,
    weights: Handle,
    state: Handle,
    partials: Handle,
    sad_placeholder: Handle,
    blocks: u32,
}

impl FitScratch {
    /// Scratch for a block grid of `blocks` blocks.
    pub fn new(client: &Client, blocks: u32) -> Self {
        let partial_count = blocks.div_ceil(FIT_THREADS);
        let partial_words = (partial_count * FIT_LANES) as usize;
        let bands = client.empty(FIT_BAND_WORDS as usize * size_of::<u32>());
        let weights = client.empty(blocks as usize * size_of::<f32>());
        let state = client.empty(FIT_STATE_WORDS as usize * size_of::<f32>());
        let partials = client.empty(partial_words * size_of::<f32>());
        let sad_placeholder = zeroed_words(client, 1);

        Self {
            bands,
            weights,
            state,
            partials,
            sad_placeholder,
            blocks,
        }
    }

    /// The sums workgroups one pass launches, one per [FIT_THREADS] blocks.
    pub fn partial_count(&self) -> u32 {
        self.blocks.div_ceil(FIT_THREADS)
    }

    pub fn blocks(&self) -> u32 {
        self.blocks
    }

    pub fn bands(&self) -> &Handle {
        &self.bands
    }

    pub fn weights(&self) -> &Handle {
        &self.weights
    }

    pub fn state(&self) -> &Handle {
        &self.state
    }

    pub fn partials(&self) -> &Handle {
        &self.partials
    }
}

/// Fits a similarity camera model to one pair field and writes it to `ring` slot `slot`.
///
/// The model maps the older frame onto the newer one whichever direction `field` names. Static
/// letterbox and pillarbox bars are found first and left out. The fit starts from the model in
/// ring slot `prior` when it is valid and would be accepted on this field, and from an unweighted
/// fit otherwise. Each of [FIT_SCALES] then reweights it, and a last pass refits the blocks within
/// [FIT_INLIER_CAP] of it. The model must cut the outlier-capped residual of no motion by a clear
/// margin and stay within the roll, zoom and corner bounds, or identity is written with `valid`
/// at 0.
///
/// Every step runs on the device in a fixed order, so the same field always gives the same bits.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernels bind"
)]
pub(crate) fn run_model_fit(
    client: &Client,
    ctx: &MotionCtx,
    width: u32,
    height: u32,
    field: FitField<'_>,
    scratch: &FitScratch,
    prior: Option<u32>,
    ring: &ModelRing,
    slot: u32,
) {
    debug_assert_eq!(
        scratch.blocks,
        ctx.blocks_x * ctx.blocks_y,
        "the scratch must match the block grid"
    );
    debug_assert!(slot < ring.depth(), "the fit slot must lie in the ring");

    let launch = FitLaunch {
        client,
        ctx,
        width,
        height,
        field: &field,
        scratch,
        ring,
        prior,
        slot,
    };

    launch.bands();

    launch.sums(FIT_STAGE_WEIGH, FIT_INLIER_CAP);
    launch.finish(FIT_STAGE_WEIGH);

    if prior.is_some() {
        launch.sums(FIT_STAGE_GATE, FIT_INLIER_CAP);
        launch.finish(FIT_STAGE_GATE);
    }

    for scale in FIT_SCALES {
        launch.sums(FIT_STAGE_SOLVE, scale);
        launch.finish(FIT_STAGE_SOLVE);
    }

    launch.sums(FIT_STAGE_POLISH, FIT_INLIER_CAP);
    launch.finish(FIT_STAGE_POLISH);

    launch.sums(FIT_STAGE_SCORE, FIT_INLIER_CAP);
    launch.finish(FIT_STAGE_SCORE);
}

/// The buffers, shapes and slots every dispatch of one fit binds.
struct FitLaunch<'a> {
    client: &'a Client,
    ctx: &'a MotionCtx,
    width: u32,
    height: u32,
    field: &'a FitField<'a>,
    scratch: &'a FitScratch,
    ring: &'a ModelRing,
    prior: Option<u32>,
    slot: u32,
}

impl FitLaunch<'_> {
    fn prior_slot(&self) -> u32 {
        self.prior.unwrap_or(0)
    }

    fn direction(&self) -> Handle {
        let direction_offset = self.field.direction as u64 * self.ctx.pair_direction_bytes();

        self.field.pairs.clone().offset_start(direction_offset)
    }

    fn has_contrast(&self) -> bool {
        self.field.sad.is_some() && self.field.has_contrast
    }

    /// The SAD buffer bound over the words the kernels read, or a placeholder without one.
    fn sad_arg(&self) -> BufferArg {
        let blocks = self.scratch.blocks as usize;
        let means_offset = self.ctx.conf_stride() as usize;
        let sad_len = if self.has_contrast() {
            2 * means_offset + blocks
        } else {
            blocks
        };

        match self.field.sad {
            Some(sad) => bind(sad, sad_len),
            None => bind(&self.scratch.sad_placeholder, 1),
        }
    }

    fn bands(&self) {
        let direction_len = self.ctx.pair_direction_len() as usize;
        let direction = self.direction();
        let has_contrast = self.has_contrast();
        let means_offset = self.ctx.conf_stride();
        let cube_count = CubeCount::new_1d(1);
        let cube_dim = CubeDim::new_1d(FIT_THREADS);

        let field_arg = bind(&direction, direction_len);
        let sad_arg = self.sad_arg();
        let bands_arg = bind(&self.scratch.bands, FIT_BAND_WORDS as usize);

        // SAFETY: each binding covers the words its buffer holds, and the kernel reads only blocks
        // inside the grid.
        unsafe {
            mc_fit_bands::launch_unchecked(
                self.client,
                cube_count,
                cube_dim,
                field_arg,
                sad_arg,
                bands_arg,
                has_contrast,
                means_offset,
                self.width,
                self.height,
                self.ctx.blksize,
                self.ctx.step,
                self.ctx.blocks_x,
                self.ctx.blocks_y,
            );
        }
    }

    fn sums(&self, stage: u32, scale: f32) {
        let blocks = self.scratch.blocks as usize;
        let partial_count = self.scratch.partial_count();
        let partial_words = (partial_count * FIT_LANES) as usize;
        let direction_len = self.ctx.pair_direction_len() as usize;
        let direction = self.direction();
        let older_to_newer = self.field.direction == 0;
        let has_sad = self.field.sad.is_some();
        let has_contrast = self.has_contrast();
        let means_offset = self.ctx.conf_stride();
        let cube_count = CubeCount::new_1d(partial_count);
        let cube_dim = CubeDim::new_1d(FIT_THREADS);

        let field_arg = bind(&direction, direction_len);
        let sad_arg = self.sad_arg();
        let bands_arg = bind(&self.scratch.bands, FIT_BAND_WORDS as usize);
        let weights_arg = bind(&self.scratch.weights, blocks);
        let state_arg = bind(&self.scratch.state, FIT_STATE_WORDS as usize);
        let models_arg = bind(self.ring.handle(), self.ring.words());
        let partials_arg = bind(&self.scratch.partials, partial_words);

        // SAFETY: each binding covers the words its buffer holds, and the kernel guards every block
        // index against the grid.
        unsafe {
            mc_fit_sums::launch_unchecked(
                self.client,
                cube_count,
                cube_dim,
                field_arg,
                sad_arg,
                bands_arg,
                weights_arg,
                state_arg,
                models_arg,
                partials_arg,
                scale,
                self.prior_slot(),
                stage,
                has_sad,
                has_contrast,
                means_offset,
                older_to_newer,
                self.width,
                self.height,
                self.ctx.blksize,
                self.ctx.step,
                self.ctx.blocks_x,
                self.ctx.blocks_y,
            );
        }
    }

    fn finish(&self, stage: u32) {
        let partial_count = self.scratch.partial_count();
        let partial_words = (partial_count * FIT_LANES) as usize;
        let cube_count = CubeCount::new_1d(1);
        let cube_dim = CubeDim::new_1d(FIT_THREADS);

        let partials_arg = bind(&self.scratch.partials, partial_words);
        let state_arg = bind(&self.scratch.state, FIT_STATE_WORDS as usize);
        let models_arg = bind(self.ring.handle(), self.ring.words());

        // SAFETY: each binding covers the words its buffer holds, and the prior and fit slots lie in
        // the ring.
        unsafe {
            mc_fit_finish::launch_unchecked(
                self.client,
                cube_count,
                cube_dim,
                partials_arg,
                state_arg,
                models_arg,
                partial_count,
                self.prior_slot(),
                self.slot,
                stage,
                self.width,
                self.height,
            );
        }
    }
}

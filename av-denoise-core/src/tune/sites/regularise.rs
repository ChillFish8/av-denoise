use cubecl::prelude::*;
use cubecl::server::Handle;

use super::NL4D_MV_REGULARISE;
use crate::nl4d::kernels::{REGULARISE_CANDIDATES, nl4d_mv_regularise, nl4d_mv_regularise_coop};
use crate::tune::fidelity::{CompareResult, Fidelity, PlaneDiff, diff_integers, diff_plane};
use crate::tune::gpu::{bind, read_f32, read_i32, zero_words};
use crate::tune::key::SiteKey;
use crate::tune::site::{LaunchShape, SiteCandidate, VariantSite};

/// Words of shared scratch the regularise kernels hold beside the centre tile.
///
/// That is two per candidate for the vector list, two for the median, and one per candidate for
/// its SAD and one for its cost.
const SCRATCH_WORDS: u32 = 2 * REGULARISE_CANDIDATES + 2 + 2 * REGULARISE_CANDIDATES;

/// The largest confidence difference the cooperative kernels may show against the serial one.
pub const COOP_MAX_ABS: f32 = 1.0e-4;

/// The lowest confidence PSNR the cooperative kernels may show against the serial one.
pub const COOP_MIN_PSNR: f64 = 80.0;

/// The share of blocks whose winning vector may differ from the serial kernel's.
///
/// A near tie between two different candidates can resolve either way once the SADs are summed in
/// another order.
pub const COOP_MAX_FLIPPED_SHARE: f64 = 0.001;

const COOP_FIDELITY: Fidelity = Fidelity::Rounding {
    min_psnr: COOP_MIN_PSNR,
    max_abs: COOP_MAX_ABS,
};

/// How closely two regularised fields agree.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FieldAgreement {
    /// Blocks whose winning vector differs.
    pub(crate) flipped: usize,
    /// The confidence difference over the blocks whose vectors match.
    pub(crate) confidence: PlaneDiff,
}

/// Counts the blocks whose vectors differ and diffs the confidence of the blocks whose vectors match.
pub(crate) fn compare_fields(
    expected_vectors: &[i32],
    actual_vectors: &[i32],
    expected_confidence: &[f32],
    actual_confidence: &[f32],
) -> FieldAgreement {
    let blocks = expected_confidence.len();
    let mut flipped = 0usize;
    let mut kept_expected = Vec::with_capacity(blocks);
    let mut kept_actual = Vec::with_capacity(blocks);

    for block in 0..blocks {
        let expected_pair = &expected_vectors[2 * block..2 * block + 2];
        let actual_pair = &actual_vectors[2 * block..2 * block + 2];

        if expected_pair == actual_pair {
            kept_expected.push(expected_confidence[block]);
            kept_actual.push(actual_confidence[block]);
        } else {
            flipped += 1;
        }
    }

    let confidence = diff_plane(&kept_expected, &kept_actual);

    FieldAgreement { flipped, confidence }
}

/// How one candidate spreads a block's work over its cube.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegulariseLayout {
    /// The serial kernel on an `x` by `y` cube, one thread per scored vector.
    Serial { x: u32, y: u32 },
    /// The cooperative kernel on a `threads` by 1 cube.
    Cooperative { threads: u32 },
}

/// A way to run the regularise pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegulariseCandidate {
    Serial8x8,
    Serial8x4,
    Serial4x4,
    Coop32,
    Coop64,
}

impl RegulariseCandidate {
    pub fn layout(self) -> RegulariseLayout {
        match self {
            RegulariseCandidate::Serial8x8 => RegulariseLayout::Serial { x: 8, y: 8 },
            RegulariseCandidate::Serial8x4 => RegulariseLayout::Serial { x: 8, y: 4 },
            RegulariseCandidate::Serial4x4 => RegulariseLayout::Serial { x: 4, y: 4 },
            RegulariseCandidate::Coop32 => RegulariseLayout::Cooperative { threads: 32 },
            RegulariseCandidate::Coop64 => RegulariseLayout::Cooperative { threads: 64 },
        }
    }
}

impl SiteCandidate for RegulariseCandidate {
    const SITE: &'static str = NL4D_MV_REGULARISE;
    const ALL: &'static [Self] = &[
        RegulariseCandidate::Serial8x8,
        RegulariseCandidate::Serial8x4,
        RegulariseCandidate::Serial4x4,
        RegulariseCandidate::Coop32,
        RegulariseCandidate::Coop64,
    ];

    fn name(self) -> &'static str {
        match self {
            RegulariseCandidate::Serial8x8 => "serial8x8",
            RegulariseCandidate::Serial8x4 => "serial8x4",
            RegulariseCandidate::Serial4x4 => "serial4x4",
            RegulariseCandidate::Coop32 => "coop32",
            RegulariseCandidate::Coop64 => "coop64",
        }
    }

    fn fidelity(self) -> Fidelity {
        match self.layout() {
            RegulariseLayout::Serial { .. } => Fidelity::Exact,
            RegulariseLayout::Cooperative { .. } => COOP_FIDELITY,
        }
    }
}

/// The regularised field and confidence one neighbour's pass writes.
#[derive(Debug, Clone)]
pub struct RegulariseOutputs {
    /// Two `i32` components per block.
    pub vectors: Handle,
    /// One `f32` per block.
    pub confidence: Handle,
    pub blocks: usize,
}

/// One neighbour's regularisation pass.
#[derive(Clone)]
pub struct RegulariseSite<R: Runtime> {
    pub client: ComputeClient<R>,
    /// Level 0 of the centre frame's luma pyramid.
    pub centre: Handle,
    /// Level 0 of the neighbour frame's luma pyramid.
    pub neighbour: Handle,
    /// Elements in one level 0 plane.
    pub level_len: usize,
    pub mv_in: Handle,
    pub outputs: RegulariseOutputs,
    pub lambda_pixel: f32,
    pub sad_noise_floor: f32,
    pub thsad: f32,
    pub width: u32,
    pub height: u32,
    pub blksize: u32,
    pub step: u32,
    pub blocks_x: u32,
    pub blocks_y: u32,
}

impl<R: Runtime> RegulariseSite<R> {
    fn launch_serial(&self, x: u32, y: u32, outputs: &RegulariseOutputs) {
        let blocks = outputs.blocks;
        let cube_count = CubeCount::new_2d(self.blocks_x, self.blocks_y);
        let cube_dim = CubeDim::new_2d(x, y);
        let centre = bind::<R>(&self.centre, self.level_len);
        let neighbour = bind::<R>(&self.neighbour, self.level_len);
        let mv_in = bind::<R>(&self.mv_in, 2 * blocks);
        let vectors = bind::<R>(&outputs.vectors, 2 * blocks);
        let confidence = bind::<R>(&outputs.confidence, blocks);

        unsafe {
            nl4d_mv_regularise::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                centre,
                neighbour,
                mv_in,
                vectors,
                confidence,
                self.lambda_pixel,
                self.sad_noise_floor,
                self.thsad,
                self.width,
                self.height,
                self.blksize,
                self.step,
                self.blocks_x,
                self.blocks_y,
            );
        }
    }

    fn launch_cooperative(&self, threads: u32, outputs: &RegulariseOutputs) {
        let blocks = outputs.blocks;
        let cube_count = CubeCount::new_2d(self.blocks_x, self.blocks_y);
        let cube_dim = CubeDim::new_1d(threads);
        let centre = bind::<R>(&self.centre, self.level_len);
        let neighbour = bind::<R>(&self.neighbour, self.level_len);
        let mv_in = bind::<R>(&self.mv_in, 2 * blocks);
        let vectors = bind::<R>(&outputs.vectors, 2 * blocks);
        let confidence = bind::<R>(&outputs.confidence, blocks);

        unsafe {
            nl4d_mv_regularise_coop::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                centre,
                neighbour,
                mv_in,
                vectors,
                confidence,
                self.lambda_pixel,
                self.sad_noise_floor,
                self.thsad,
                self.width,
                self.height,
                self.blksize,
                self.step,
                self.blocks_x,
                self.blocks_y,
                threads,
            );
        }
    }
}

impl<R: Runtime> VariantSite<R> for RegulariseSite<R> {
    type Candidate = RegulariseCandidate;
    type Outputs = RegulariseOutputs;

    const PRESCREEN: bool = true;

    fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    fn key(&self) -> SiteKey {
        SiteKey::builder(NL4D_MV_REGULARISE, self.width, self.height)
            .param("blksize", self.blksize)
            .param("step", self.step)
            .build()
    }

    fn shape(&self, candidate: RegulariseCandidate) -> LaunchShape {
        match candidate.layout() {
            RegulariseLayout::Serial { x, y } => LaunchShape { x, y },
            RegulariseLayout::Cooperative { threads } => LaunchShape { x: threads, y: 1 },
        }
    }

    fn shared_bytes(&self, candidate: RegulariseCandidate) -> usize {
        let centre_floats = self.blksize * self.blksize;
        let partial_floats = match candidate.layout() {
            RegulariseLayout::Serial { .. } => 0,
            RegulariseLayout::Cooperative { threads } => REGULARISE_CANDIDATES * threads,
        };
        let words = centre_floats + SCRATCH_WORDS + partial_floats;

        words as usize * size_of::<f32>()
    }

    fn launch(&self, candidate: RegulariseCandidate, outputs: &RegulariseOutputs) {
        match candidate.layout() {
            RegulariseLayout::Serial { x, y } => self.launch_serial(x, y, outputs),
            RegulariseLayout::Cooperative { threads } => self.launch_cooperative(threads, outputs),
        }
    }

    fn real_outputs(&self) -> RegulariseOutputs {
        self.outputs.clone()
    }

    fn scratch_outputs(&self) -> RegulariseOutputs {
        let blocks = self.outputs.blocks;
        let vectors = self.client.empty(2 * blocks * size_of::<i32>());
        let confidence = self.client.empty(blocks * size_of::<f32>());
        let scratch = RegulariseOutputs {
            vectors,
            confidence,
            blocks,
        };

        self.clear_outputs(&scratch);
        scratch
    }

    fn clear_outputs(&self, outputs: &RegulariseOutputs) {
        zero_words(&self.client, &outputs.vectors, 2 * outputs.blocks);
        zero_words(&self.client, &outputs.confidence, outputs.blocks);
    }

    fn compare(
        &self,
        baseline: &RegulariseOutputs,
        candidate: &RegulariseOutputs,
        fidelity: Fidelity,
    ) -> Result<CompareResult, anyhow::Error> {
        let blocks = baseline.blocks;
        let expected_vectors = read_i32(&self.client, &baseline.vectors, 2 * blocks)?;
        let actual_vectors = read_i32(&self.client, &candidate.vectors, 2 * blocks)?;
        let expected_confidence = read_f32(&self.client, &baseline.confidence, blocks)?;
        let actual_confidence = read_f32(&self.client, &candidate.confidence, blocks)?;

        if fidelity == Fidelity::Exact {
            let vector_diff = diff_integers(&expected_vectors, &actual_vectors);
            let confidence_diff = diff_plane(&expected_confidence, &actual_confidence);
            let verdict = fidelity.judge(&[vector_diff, confidence_diff]);
            return Ok(verdict);
        }

        let agreement = compare_fields(
            &expected_vectors,
            &actual_vectors,
            &expected_confidence,
            &actual_confidence,
        );
        let flipped = agreement.flipped;

        let allowed = (blocks as f64 * COOP_MAX_FLIPPED_SHARE).floor() as usize;
        if flipped > allowed {
            let reason = format!("{flipped} of {blocks} vectors differ where at most {allowed} may");
            return Ok(CompareResult::Fail(reason));
        }

        let verdict = fidelity.judge(&[agreement.confidence]);

        Ok(verdict)
    }
}

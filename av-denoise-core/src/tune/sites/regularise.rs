use cubecl::prelude::*;
use cubecl::server::Handle;

use super::NL4D_MV_REGULARISE;
use crate::nl4d::kernels::{REGULARISE_CANDIDATES, nl4d_mv_regularise, nl4d_mv_regularise_coop};
use crate::noise::DERIVED_WORDS;
use crate::tune::family::{Families, FamilyId, FlatFamily};
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
///
/// The buffers hold every neighbour of `layout`, each `mv_stride` and `conf_stride` values apart, and
/// the padding between neighbours is skipped.
pub(crate) fn compare_fields(
    layout: &RegulariseOutputs,
    expected_vectors: &[i32],
    actual_vectors: &[i32],
    expected_confidence: &[f32],
    actual_confidence: &[f32],
) -> FieldAgreement {
    let blocks = layout.blocks;
    let neighbours = layout.neighbours as usize;
    let mv_stride = layout.mv_stride as usize;
    let conf_stride = layout.conf_stride as usize;
    let mut flipped = 0usize;
    let mut kept_expected = Vec::with_capacity(neighbours * blocks);
    let mut kept_actual = Vec::with_capacity(neighbours * blocks);

    for neighbour in 0..neighbours {
        for block in 0..blocks {
            let vector = neighbour * mv_stride + 2 * block;
            let score = neighbour * conf_stride + block;
            let expected_pair = &expected_vectors[vector..vector + 2];
            let actual_pair = &actual_vectors[vector..vector + 2];

            if expected_pair == actual_pair {
                kept_expected.push(expected_confidence[score]);
                kept_actual.push(actual_confidence[score]);
            } else {
                flipped += 1;
            }
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

/// The serial layouts, led by 8x8, and the cooperative layouts, led by 32 threads.
const REGULARISE_FAMILIES: &[FlatFamily<RegulariseCandidate>] = &[
    FlatFamily {
        id: FamilyId("serial"),
        representative: RegulariseCandidate::Serial8x8,
        members: &[
            RegulariseCandidate::Serial8x8,
            RegulariseCandidate::Serial8x4,
            RegulariseCandidate::Serial4x4,
        ],
    },
    FlatFamily {
        id: FamilyId("coop"),
        representative: RegulariseCandidate::Coop32,
        members: &[RegulariseCandidate::Coop32, RegulariseCandidate::Coop64],
    },
];

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

    fn families() -> Families<Self> {
        Families::Flat(REGULARISE_FAMILIES)
    }
}

/// The regularised field and confidence of every neighbour.
#[derive(Debug, Clone)]
pub struct RegulariseOutputs {
    /// Two `i32` components per block, `mv_stride` values per neighbour.
    pub vectors: Handle,
    /// One `f32` per block, `conf_stride` values per neighbour.
    pub confidence: Handle,
    /// Blocks in one neighbour's field.
    pub blocks: usize,
    pub neighbours: u32,
    pub mv_stride: u32,
    pub conf_stride: u32,
}

impl RegulariseOutputs {
    /// `i32` values the vector buffer holds, padding between neighbours included.
    pub fn vectors_len(&self) -> usize {
        self.neighbours as usize * self.mv_stride as usize
    }

    /// `f32` values the confidence buffer holds, padding between neighbours included.
    pub fn confidence_len(&self) -> usize {
        self.neighbours as usize * self.conf_stride as usize
    }
}

/// The regularisation pass over every neighbour in one launch.
#[derive(Clone)]
pub struct RegulariseSite {
    pub client: Client,
    /// The whole luma pyramid. Level 0 of slot `s` starts `s * slot_pixels` values in.
    pub pyramid: Handle,
    /// `f32` values `pyramid` holds.
    pub pyramid_len: usize,
    pub centre_slot: u32,
    /// One `u32` ring slot per neighbour, in motion-field order.
    pub neighbour_slots: Handle,
    pub slot_pixels: u32,
    /// The motion field of every neighbour, laid out as the outputs' vectors.
    pub mv_in: Handle,
    pub outputs: RegulariseOutputs,
    pub lambda_pixel: f32,
    /// The derived noise values, whose SAD floor the confidence is scored against.
    pub noise: Handle,
    pub thsad: f32,
    pub width: u32,
    pub height: u32,
    pub blksize: u32,
    pub step: u32,
    pub blocks_x: u32,
    pub blocks_y: u32,
}

impl RegulariseSite {
    fn launch_serial(&self, x: u32, y: u32, outputs: &RegulariseOutputs) {
        let cube_count = CubeCount::new_3d(self.blocks_x, self.blocks_y, outputs.neighbours);
        let cube_dim = CubeDim::new_2d(x, y);
        let slots_len = outputs.neighbours as usize;
        let pyramid = bind(&self.pyramid, self.pyramid_len);
        let neighbour_slots = bind(&self.neighbour_slots, slots_len);
        let vectors_len = outputs.vectors_len();
        let confidence_len = outputs.confidence_len();
        let mv_in = bind(&self.mv_in, vectors_len);
        let vectors = bind(&outputs.vectors, vectors_len);
        let confidence = bind(&outputs.confidence, confidence_len);
        let noise = bind(&self.noise, DERIVED_WORDS as usize);

        unsafe {
            nl4d_mv_regularise::launch_unchecked(
                &self.client,
                cube_count,
                cube_dim,
                pyramid,
                neighbour_slots,
                mv_in,
                vectors,
                confidence,
                self.centre_slot,
                self.lambda_pixel,
                noise,
                self.thsad,
                self.width,
                self.height,
                self.slot_pixels,
                self.blksize,
                self.step,
                self.blocks_x,
                self.blocks_y,
                outputs.mv_stride,
                outputs.conf_stride,
            );
        }
    }

    fn launch_cooperative(&self, threads: u32, outputs: &RegulariseOutputs) {
        let cube_count = CubeCount::new_3d(self.blocks_x, self.blocks_y, outputs.neighbours);
        let cube_dim = CubeDim::new_1d(threads);
        let slots_len = outputs.neighbours as usize;
        let pyramid = bind(&self.pyramid, self.pyramid_len);
        let neighbour_slots = bind(&self.neighbour_slots, slots_len);
        let vectors_len = outputs.vectors_len();
        let confidence_len = outputs.confidence_len();
        let mv_in = bind(&self.mv_in, vectors_len);
        let vectors = bind(&outputs.vectors, vectors_len);
        let confidence = bind(&outputs.confidence, confidence_len);
        let noise = bind(&self.noise, DERIVED_WORDS as usize);

        unsafe {
            nl4d_mv_regularise_coop::launch_unchecked(
                &self.client,
                cube_count,
                cube_dim,
                pyramid,
                neighbour_slots,
                mv_in,
                vectors,
                confidence,
                self.centre_slot,
                self.lambda_pixel,
                noise,
                self.thsad,
                self.width,
                self.height,
                self.slot_pixels,
                self.blksize,
                self.step,
                self.blocks_x,
                self.blocks_y,
                outputs.mv_stride,
                outputs.conf_stride,
                threads,
            );
        }
    }
}

impl VariantSite for RegulariseSite {
    type Candidate = RegulariseCandidate;
    type Outputs = RegulariseOutputs;

    const PRESCREEN: bool = true;

    fn client(&self) -> &Client {
        &self.client
    }

    fn key(&self) -> SiteKey {
        SiteKey::builder(NL4D_MV_REGULARISE, self.width, self.height)
            .param("blksize", self.blksize)
            .param("step", self.step)
            .param("neighbours", self.outputs.neighbours)
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

    fn output_handles(&self, outputs: &RegulariseOutputs) -> Vec<Handle> {
        vec![outputs.vectors.clone(), outputs.confidence.clone()]
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
        let vectors_bytes = self.outputs.vectors_len() * size_of::<i32>();
        let confidence_bytes = self.outputs.confidence_len() * size_of::<f32>();
        let vectors = self.client.empty(vectors_bytes);
        let confidence = self.client.empty(confidence_bytes);
        let scratch = RegulariseOutputs {
            vectors,
            confidence,
            blocks: self.outputs.blocks,
            neighbours: self.outputs.neighbours,
            mv_stride: self.outputs.mv_stride,
            conf_stride: self.outputs.conf_stride,
        };

        self.clear_outputs(&scratch);
        scratch
    }

    fn clear_outputs(&self, outputs: &RegulariseOutputs) {
        let vectors_len = outputs.vectors_len();
        let confidence_len = outputs.confidence_len();
        zero_words(&self.client, &outputs.vectors, vectors_len);
        zero_words(&self.client, &outputs.confidence, confidence_len);
    }

    fn compare(
        &self,
        baseline: &RegulariseOutputs,
        candidate: &RegulariseOutputs,
        fidelity: Fidelity,
    ) -> Result<CompareResult, anyhow::Error> {
        let vectors_len = baseline.vectors_len();
        let confidence_len = baseline.confidence_len();
        let expected_vectors = read_i32(&self.client, &baseline.vectors, vectors_len)?;
        let actual_vectors = read_i32(&self.client, &candidate.vectors, vectors_len)?;
        let expected_confidence = read_f32(&self.client, &baseline.confidence, confidence_len)?;
        let actual_confidence = read_f32(&self.client, &candidate.confidence, confidence_len)?;

        if fidelity == Fidelity::Exact {
            let vector_diff = diff_integers(&expected_vectors, &actual_vectors);
            let confidence_diff = diff_plane(&expected_confidence, &actual_confidence);
            let verdict = fidelity.judge(&[vector_diff, confidence_diff]);
            return Ok(verdict);
        }

        let agreement = compare_fields(
            baseline,
            &expected_vectors,
            &actual_vectors,
            &expected_confidence,
            &actual_confidence,
        );
        let flipped = agreement.flipped;

        let vectors = baseline.neighbours as usize * baseline.blocks;
        let allowed = (vectors as f64 * COOP_MAX_FLIPPED_SHARE).floor() as usize;
        if flipped > allowed {
            let reason = format!("{flipped} of {vectors} vectors differ where at most {allowed} may");
            return Ok(CompareResult::Fail(reason));
        }

        let verdict = fidelity.judge(&[agreement.confidence]);

        Ok(verdict)
    }
}

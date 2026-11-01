use cubecl::prelude::*;
use cubecl::server::Handle;

use super::MC_TRUEMOTION;
use crate::motion::AFFINE_WORDS;
use crate::motion::truemotion::kernels::{
    LIST_CAPACITY,
    META_WORDS,
    SCORE_WORDS,
    STAGED_FLAG_WORDS,
    STAGED_WORDS,
    mc_truemotion,
    mc_truemotion_coop,
};
use crate::noise::DERIVED_WORDS;
use crate::tune::family::{Families, FamilyId, FlatFamily};
use crate::tune::fidelity::{CompareResult, Fidelity, PlaneDiff, diff_integers, diff_plane};
use crate::tune::gpu::{bind, read_f32, read_i32, zero_words};
use crate::tune::key::SiteKey;
use crate::tune::site::{LaunchShape, SiteCandidate, VariantSite};

/// Words of shared scratch every truemotion layout holds beside the centre tile and its sum buffers.
///
/// That is two per list entry for the vectors and one for their SADs, the integer and float state
/// words, and the staged neighbourhood and change flags.
const SCRATCH_WORDS: u32 = 3 * LIST_CAPACITY + META_WORDS + SCORE_WORDS + STAGED_WORDS + STAGED_FLAG_WORDS;

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

/// How closely two truemotion fields agree.
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
    layout: &TruemotionOutputs,
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
pub enum TruemotionLayout {
    /// The serial kernel on an `x` by `y` cube, one thread per scored vector.
    Serial { x: u32, y: u32 },
    /// The cooperative kernel on a `threads` by 1 cube.
    Cooperative { threads: u32 },
}

/// A way to run one truemotion pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruemotionCandidate {
    Serial8x8,
    Serial8x4,
    Serial4x4,
    Coop32,
}

impl TruemotionCandidate {
    pub fn layout(self) -> TruemotionLayout {
        match self {
            TruemotionCandidate::Serial8x8 => TruemotionLayout::Serial { x: 8, y: 8 },
            TruemotionCandidate::Serial8x4 => TruemotionLayout::Serial { x: 8, y: 4 },
            TruemotionCandidate::Serial4x4 => TruemotionLayout::Serial { x: 4, y: 4 },
            TruemotionCandidate::Coop32 => TruemotionLayout::Cooperative { threads: 32 },
        }
    }
}

/// The serial layouts, led by 8x8, and the cooperative layout on 32 threads.
const TRUEMOTION_FAMILIES: &[FlatFamily<TruemotionCandidate>] = &[
    FlatFamily {
        id: FamilyId("serial"),
        representative: TruemotionCandidate::Serial8x8,
        members: &[
            TruemotionCandidate::Serial8x8,
            TruemotionCandidate::Serial8x4,
            TruemotionCandidate::Serial4x4,
        ],
    },
    FlatFamily {
        id: FamilyId("coop"),
        representative: TruemotionCandidate::Coop32,
        members: &[TruemotionCandidate::Coop32],
    },
];

impl SiteCandidate for TruemotionCandidate {
    const SITE: &'static str = MC_TRUEMOTION;
    const ALL: &'static [Self] = &[
        TruemotionCandidate::Serial8x8,
        TruemotionCandidate::Serial8x4,
        TruemotionCandidate::Serial4x4,
        TruemotionCandidate::Coop32,
    ];

    fn name(self) -> &'static str {
        match self {
            TruemotionCandidate::Serial8x8 => "serial8x8",
            TruemotionCandidate::Serial8x4 => "serial8x4",
            TruemotionCandidate::Serial4x4 => "serial4x4",
            TruemotionCandidate::Coop32 => "coop32",
        }
    }

    fn fidelity(self) -> Fidelity {
        match self.layout() {
            TruemotionLayout::Serial { .. } => Fidelity::Exact,
            TruemotionLayout::Cooperative { .. } => COOP_FIDELITY,
        }
    }

    fn families() -> Families<Self> {
        Families::Flat(TRUEMOTION_FAMILIES)
    }
}

/// The field, confidence and SADs one pass writes for every neighbour.
#[derive(Debug, Clone)]
pub struct TruemotionOutputs {
    /// Two `i32` components per block, `mv_stride` values per neighbour.
    pub vectors: Handle,
    /// `i32` values `vectors` holds.
    pub vectors_len: usize,
    /// One `f32` per block, `conf_stride` values per neighbour, or a placeholder when
    /// `writes_confidence` is unset.
    pub confidence: Handle,
    /// `f32` values `confidence` holds.
    pub confidence_len: usize,
    pub writes_confidence: bool,
    /// Each block's winning SAD in the confidence layout, then room for as many values again.
    pub sad: Option<Handle>,
    /// Flags each block that left its own vector, one `u32` per block in the confidence layout.
    pub changes: Option<Handle>,
    /// Blocks in one neighbour's field.
    pub blocks: usize,
    pub neighbours: u32,
    pub mv_stride: u32,
    pub conf_stride: u32,
}

impl TruemotionOutputs {
    pub fn vectors_len(&self) -> usize {
        self.vectors_len
    }

    pub fn confidence_len(&self) -> usize {
        self.confidence_len
    }

    /// `u32` values a change flag buffer holds.
    pub fn changes_len(&self) -> usize {
        self.neighbours as usize * self.conf_stride as usize
    }

    /// `f32` values the SAD buffer holds.
    pub fn sad_len(&self) -> usize {
        2 * self.neighbours as usize * self.conf_stride as usize
    }
}

/// The values a pass scores every block with, as
/// [mc_truemotion](crate::motion::truemotion::kernels::mc_truemotion) reads them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TruemotionValues {
    pub lambda_base: f32,
    pub lambda_noise: f32,
    pub surcharge: f32,
    pub fade_level: f32,
    pub thsad: f32,
    pub sad_field: u32,
}

/// One truemotion pass over every neighbour in one launch.
#[derive(Clone)]
pub struct TruemotionSite {
    pub client: Client,
    /// The whole luma pyramid. Level 0 of slot `s` starts `s * slot_pixels` values in.
    pub pyramid: Handle,
    /// `f32` values `pyramid` holds.
    pub pyramid_len: usize,
    pub centre_slot: u32,
    /// One `u32` ring slot per neighbour, in motion-field order.
    pub neighbour_slots: Handle,
    pub slot_pixels: u32,
    /// The field the pass reads, laid out as the outputs' vectors.
    pub mv_in: Handle,
    /// The search's seeds, laid out as `mv_in`.
    pub seeds: Option<Handle>,
    /// The camera model predictions, [AFFINE_WORDS] words per neighbour.
    pub predictions: Option<Handle>,
    /// The change flags the pass before wrote, laid out as the outputs' flags.
    pub changes_in: Option<Handle>,
    pub outputs: TruemotionOutputs,
    pub values: TruemotionValues,
    /// The derived noise values, whose SAD floor scores the confidence and scales the penalty.
    pub noise: Handle,
    /// Whether a positive SAD floor in `noise` scales the penalty and fade.
    pub reads_floor: bool,
    pub refine_radius: u32,
    pub width: u32,
    pub height: u32,
    pub blksize: u32,
    pub step: u32,
    pub blocks_x: u32,
    pub blocks_y: u32,
}

/// Every buffer one launch binds, with placeholders for the unused ones.
struct Bindings {
    pyramid: BufferArg,
    neighbour_slots: BufferArg,
    mv_in: BufferArg,
    seeds: BufferArg,
    predictions: BufferArg,
    changes_in: BufferArg,
    vectors: BufferArg,
    confidence: BufferArg,
    sad: BufferArg,
    changes_out: BufferArg,
    noise: BufferArg,
}

impl TruemotionSite {
    /// Binds every buffer a launch into `outputs` reads or writes.
    ///
    /// An unread seed field binds `mv_in`, unread predictions bind `noise`, an unwritten SAD output
    /// binds the confidence buffer, and unused change flags bind the neighbour slots. The kernel
    /// never indexes any of them.
    fn bindings(&self, outputs: &TruemotionOutputs) -> Bindings {
        let vectors_len = outputs.vectors_len();
        let confidence_len = outputs.confidence_len();
        let slots_len = outputs.neighbours as usize;
        let changes_len = outputs.changes_len();
        let (changes_in, changes_in_len) = match self.changes_in.as_ref() {
            Some(handle) => (handle, changes_len),
            None => (&self.neighbour_slots, slots_len),
        };
        let (changes_out, changes_out_len) = match outputs.changes.as_ref() {
            Some(handle) => (handle, changes_len),
            None => (&self.neighbour_slots, slots_len),
        };
        let seeds = self.seeds.as_ref().unwrap_or(&self.mv_in);
        let (predictions, predictions_len) = match self.predictions.as_ref() {
            Some(handle) => (handle, (outputs.neighbours * AFFINE_WORDS) as usize),
            None => (&self.noise, DERIVED_WORDS as usize),
        };
        let (sad, sad_len) = match outputs.sad.as_ref() {
            Some(handle) => (handle, outputs.sad_len()),
            None => (&outputs.confidence, confidence_len),
        };

        Bindings {
            pyramid: bind(&self.pyramid, self.pyramid_len),
            neighbour_slots: bind(&self.neighbour_slots, slots_len),
            mv_in: bind(&self.mv_in, vectors_len),
            seeds: bind(seeds, vectors_len),
            predictions: bind(predictions, predictions_len),
            changes_in: bind(changes_in, changes_in_len),
            vectors: bind(&outputs.vectors, vectors_len),
            confidence: bind(&outputs.confidence, confidence_len),
            sad: bind(sad, sad_len),
            changes_out: bind(changes_out, changes_out_len),
            noise: bind(&self.noise, DERIVED_WORDS as usize),
        }
    }

    fn launch_serial(&self, x: u32, y: u32, outputs: &TruemotionOutputs) {
        let cube_count = CubeCount::new_3d(self.blocks_x, self.blocks_y, outputs.neighbours);
        let cube_dim = CubeDim::new_2d(x, y);
        let buffers = self.bindings(outputs);
        let values = self.values;

        unsafe {
            mc_truemotion::launch_unchecked(
                &self.client,
                cube_count,
                cube_dim,
                buffers.pyramid,
                buffers.neighbour_slots,
                buffers.mv_in,
                buffers.seeds,
                buffers.predictions,
                buffers.changes_in,
                buffers.vectors,
                buffers.confidence,
                buffers.sad,
                buffers.changes_out,
                buffers.noise,
                self.centre_slot,
                values.lambda_base,
                values.lambda_noise,
                values.surcharge,
                values.fade_level,
                values.thsad,
                values.sad_field,
                self.seeds.is_some(),
                self.predictions.is_some(),
                self.reads_floor,
                outputs.writes_confidence,
                outputs.sad.is_some(),
                self.changes_in.is_some(),
                outputs.changes.is_some(),
                self.refine_radius,
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

    fn launch_cooperative(&self, threads: u32, outputs: &TruemotionOutputs) {
        let cube_count = CubeCount::new_3d(self.blocks_x, self.blocks_y, outputs.neighbours);
        let cube_dim = CubeDim::new_1d(threads);
        let buffers = self.bindings(outputs);
        let values = self.values;

        unsafe {
            mc_truemotion_coop::launch_unchecked(
                &self.client,
                cube_count,
                cube_dim,
                buffers.pyramid,
                buffers.neighbour_slots,
                buffers.mv_in,
                buffers.seeds,
                buffers.predictions,
                buffers.changes_in,
                buffers.vectors,
                buffers.confidence,
                buffers.sad,
                buffers.changes_out,
                buffers.noise,
                self.centre_slot,
                values.lambda_base,
                values.lambda_noise,
                values.surcharge,
                values.fade_level,
                values.thsad,
                values.sad_field,
                self.seeds.is_some(),
                self.predictions.is_some(),
                self.reads_floor,
                outputs.writes_confidence,
                outputs.sad.is_some(),
                self.changes_in.is_some(),
                outputs.changes.is_some(),
                self.refine_radius,
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

impl VariantSite for TruemotionSite {
    type Candidate = TruemotionCandidate;
    type Outputs = TruemotionOutputs;

    const PRESCREEN: bool = true;

    fn client(&self) -> &Client {
        &self.client
    }

    fn key(&self) -> SiteKey {
        let seeds = u32::from(self.seeds.is_some());
        let model = u32::from(self.predictions.is_some());
        let reads_changes = u32::from(self.changes_in.is_some());
        let writes_changes = u32::from(self.outputs.changes.is_some());
        let changes = reads_changes * 2 + writes_changes;
        let floor = u32::from(self.reads_floor);
        let writes_confidence = u32::from(self.outputs.writes_confidence);
        let writes_sad = u32::from(self.outputs.sad.is_some());
        let writes = writes_confidence * 2 + writes_sad;

        SiteKey::builder(MC_TRUEMOTION, self.width, self.height)
            .param("blksize", self.blksize)
            .param("step", self.step)
            .param("neighbours", self.outputs.neighbours)
            .param("seeds", seeds)
            .param("model", model)
            .param("refine", self.refine_radius)
            .param("changes", changes)
            .param("floor", floor)
            .param("writes", writes)
            .build()
    }

    fn shape(&self, candidate: TruemotionCandidate) -> LaunchShape {
        match candidate.layout() {
            TruemotionLayout::Serial { x, y } => LaunchShape { x, y },
            TruemotionLayout::Cooperative { threads } => LaunchShape { x: threads, y: 1 },
        }
    }

    fn shared_bytes(&self, candidate: TruemotionCandidate) -> usize {
        let centre_floats = self.blksize * self.blksize;
        // The serial kernels hold a row sum per row of every listed vector and one partial word,
        // the cooperative ones a partial per vector per thread and one row sum word.
        let sum_words = match candidate.layout() {
            TruemotionLayout::Serial { .. } => LIST_CAPACITY * self.blksize + 1,
            TruemotionLayout::Cooperative { threads } => LIST_CAPACITY * threads + 1,
        };
        let words = centre_floats + SCRATCH_WORDS + sum_words;

        words as usize * size_of::<f32>()
    }

    fn output_handles(&self, outputs: &TruemotionOutputs) -> Vec<Handle> {
        let mut handles = vec![outputs.vectors.clone()];
        if outputs.writes_confidence {
            handles.push(outputs.confidence.clone());
        }

        if let Some(sad) = outputs.sad.as_ref() {
            handles.push(sad.clone());
        }

        if let Some(changes) = outputs.changes.as_ref() {
            handles.push(changes.clone());
        }

        handles
    }

    fn launch(&self, candidate: TruemotionCandidate, outputs: &TruemotionOutputs) {
        match candidate.layout() {
            TruemotionLayout::Serial { x, y } => self.launch_serial(x, y, outputs),
            TruemotionLayout::Cooperative { threads } => self.launch_cooperative(threads, outputs),
        }
    }

    fn real_outputs(&self) -> TruemotionOutputs {
        self.outputs.clone()
    }

    fn scratch_outputs(&self) -> TruemotionOutputs {
        let vectors_bytes = self.outputs.vectors_len() * size_of::<i32>();
        let confidence_bytes = self.outputs.confidence_len() * size_of::<f32>();
        let vectors = self.client.empty(vectors_bytes);
        let confidence = self.client.empty(confidence_bytes);
        let sad = self.outputs.sad.as_ref().map(|_| {
            let sad_bytes = self.outputs.sad_len() * size_of::<f32>();
            self.client.empty(sad_bytes)
        });
        let changes = self.outputs.changes.as_ref().map(|_| {
            let changes_bytes = self.outputs.changes_len() * size_of::<u32>();
            self.client.empty(changes_bytes)
        });
        let scratch = TruemotionOutputs {
            vectors,
            confidence,
            sad,
            changes,
            ..self.outputs.clone()
        };

        self.clear_outputs(&scratch);
        scratch
    }

    fn clear_outputs(&self, outputs: &TruemotionOutputs) {
        let vectors_len = outputs.vectors_len();
        let confidence_len = outputs.confidence_len();
        zero_words(&self.client, &outputs.vectors, vectors_len);
        zero_words(&self.client, &outputs.confidence, confidence_len);

        if let Some(sad) = outputs.sad.as_ref() {
            let sad_len = outputs.sad_len();
            zero_words(&self.client, sad, sad_len);
        }

        if let Some(changes) = outputs.changes.as_ref() {
            let changes_len = outputs.changes_len();
            zero_words(&self.client, changes, changes_len);
        }
    }

    fn compare(
        &self,
        baseline: &TruemotionOutputs,
        candidate: &TruemotionOutputs,
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
            let mut diffs = vec![vector_diff, confidence_diff];

            if let (Some(expected), Some(actual)) = (baseline.sad.as_ref(), candidate.sad.as_ref()) {
                let sad_len = baseline.sad_len();
                let expected_sads = read_f32(&self.client, expected, sad_len)?;
                let actual_sads = read_f32(&self.client, actual, sad_len)?;
                let sad_diff = diff_plane(&expected_sads, &actual_sads);
                diffs.push(sad_diff);
            }

            if let (Some(expected), Some(actual)) = (baseline.changes.as_ref(), candidate.changes.as_ref()) {
                let changes_len = baseline.changes_len();
                let expected_flags = read_i32(&self.client, expected, changes_len)?;
                let actual_flags = read_i32(&self.client, actual, changes_len)?;
                let flag_diff = diff_integers(&expected_flags, &actual_flags);
                diffs.push(flag_diff);
            }

            let verdict = fidelity.judge(&diffs);
            return Ok(verdict);
        }

        let vectors = baseline.neighbours as usize * baseline.blocks;
        let agreement = if baseline.writes_confidence {
            compare_fields(
                baseline,
                &expected_vectors,
                &actual_vectors,
                &expected_confidence,
                &actual_confidence,
            )
        } else {
            let zeroed = vec![0.0f32; baseline.neighbours as usize * baseline.conf_stride as usize];
            compare_fields(baseline, &expected_vectors, &actual_vectors, &zeroed, &zeroed)
        };
        let flipped = agreement.flipped;

        let allowed = (vectors as f64 * COOP_MAX_FLIPPED_SHARE).floor() as usize;
        if flipped > allowed {
            let reason = format!("{flipped} of {vectors} vectors differ where at most {allowed} may");
            return Ok(CompareResult::Fail(reason));
        }

        let verdict = fidelity.judge(&[agreement.confidence]);

        Ok(verdict)
    }
}

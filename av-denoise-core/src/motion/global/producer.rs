use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{
    AFFINE_WORDS,
    FIT_LANES,
    FIT_THREADS,
    FitField,
    FitScratch,
    MODEL_WORDS,
    ModelRing,
    run_compose_models,
    run_model_fit,
};
use crate::engine::align::StorageAlign;
use crate::engine::{BufferSize, DevicePlane, IngestTarget, SampleFormat, ingest};
use crate::motion::kernels::mc_model_identity;
use crate::motion::pyramid::{pyramid_pixels_per_frame, pyramid_slot_byte_offset, run_pyramid_downscale};
use crate::motion::{
    MAX_PYRAMID_LEVELS,
    MAX_SEARCH_RADIUS,
    MatchTargets,
    MotionCompensationMode,
    MotionCtx,
    run_analyse,
};
use crate::noise::{DERIVED_SAD_FLOOR, DERIVED_WORDS};
use crate::tune::gpu::{bind, zeroed_words};

/// Frames the producer's pyramid holds, the newest and the one before it.
const PYRAMID_SLOTS: u32 = 2;

/// The deep search a pair's first match runs, on the block grid of `mode`.
///
/// It searches the most pyramid levels at the widest radius, so a pan of about 40 luma pixels is
/// found without a model to start from.
fn acquisition_ctx(
    mode: &MotionCompensationMode,
    luma_width: u32,
    luma_height: u32,
    align: StorageAlign,
) -> Option<MotionCtx> {
    let MotionCompensationMode::Mvtools {
        blksize,
        overlap,
        estimation,
        ..
    } = *mode
    else {
        return None;
    };

    let deep = MotionCompensationMode::Mvtools {
        blksize,
        overlap,
        search_radius: MAX_SEARCH_RADIUS,
        pyramid_levels: MAX_PYRAMID_LEVELS,
        estimation,
    };
    MotionCtx::new(deep, luma_width, luma_height, align)
}

/// The camera models every pass reads, with the scale from a pass's pixels to luma pixels.
#[derive(Debug, Clone)]
#[expect(dead_code, reason = "task 5 seeds the pass searches from these models")]
pub(crate) struct SharedMotion {
    pub ring: ModelRing,
    pub scale_x: f32,
    pub scale_y: f32,
}

/// The consecutive stream frames taken since the producer last started over.
#[derive(Debug, Clone, Copy)]
struct Run {
    first: u32,
    last: u32,
}

impl Run {
    /// Whether `frame` follows on from the run, as the next frame or as the last one again.
    ///
    /// A frame comes again when a pass refused its push after the producer took it.
    fn continues_to(&self, frame: u32) -> bool {
        let next = self.last.checked_add(1);
        let repeats = self.last == frame;

        self.first < frame && (repeats || next == Some(frame))
    }
}

/// Fits one camera model per adjacent frame pair from the luma plane, into a [ModelRing].
///
/// Each push matches the new frame against the one before it, fits a model to the block vectors,
/// matches again around that model and fits once more. The first match runs a deep search, so a
/// fast pan is found from a run's start, and also searches around the previous pair's model when
/// there is one. The first frame of a run writes identity.
pub struct PairModelProducer {
    client: Client,
    /// The search of `mode`, which the refit match runs.
    ctx: MotionCtx,
    /// The deep search of the first match, on the same block grid as `ctx`.
    deep_ctx: MotionCtx,
    width: u32,
    height: u32,
    input: SampleFormat,
    ring: ModelRing,
    scratch: FitScratch,
    pyramid: Handle,
    /// One single-slot table per pyramid slot.
    slot_tables: Vec<Handle>,
    /// One single-entry gap table per ring slot.
    gap_tables: Vec<Handle>,
    predictions: Handle,
    field: Handle,
    sad: Handle,
    confidence_placeholder: Handle,
    /// Zeroed noise values, so the search prefers its existing seed only on exact ties.
    noise: Handle,
    plane_placeholder: Handle,
    run: Option<Run>,
}

impl PairModelProducer {
    /// A producer fitting `luma_width` by `luma_height` frames into a ring `depth` slots deep.
    ///
    /// It searches with the block geometry of `mode`, and is `None` when `mode` does not search.
    pub fn new(
        client: &Client,
        luma_width: u32,
        luma_height: u32,
        mode: &MotionCompensationMode,
        input: SampleFormat,
        depth: u32,
    ) -> Option<Self> {
        let align = StorageAlign::from_client(client);
        let ctx = MotionCtx::new(*mode, luma_width, luma_height, align)?;
        let deep_ctx = acquisition_ctx(mode, luma_width, luma_height, align)?;
        let blocks = ctx.blocks_x * ctx.blocks_y;

        // Each level spans every frame slot before the next starts, so the shallow search reads the
        // deep pyramid's leading levels as its own.
        let levels = deep_ctx.pyramid_levels;
        let pyramid_pixels = pyramid_pixels_per_frame(luma_width, luma_height, levels, align);
        let pyramid_bytes = pyramid_pixels * PYRAMID_SLOTS as usize * size_of::<f32>();
        let pyramid = client.empty(pyramid_bytes);

        let slot_tables = (0..PYRAMID_SLOTS)
            .map(|slot| single_word_table(client, slot))
            .collect();
        let gap_tables = (0..depth).map(|slot| single_word_table(client, slot)).collect();

        let field_bytes = ctx.pair_direction_bytes() as usize;
        let sad_bytes = ctx.confidence_bytes_per_neighbour() as usize;
        let predictions_bytes = AFFINE_WORDS as usize * size_of::<f32>();
        let field = client.empty(field_bytes);
        let sad = client.empty(sad_bytes);
        let predictions = client.empty(predictions_bytes);
        let confidence_placeholder = zeroed_words(client, 1);
        let noise = zeroed_words(client, DERIVED_WORDS as usize);
        let plane_placeholder = client.empty(size_of::<u32>());
        let ring = ModelRing::new(client, depth);
        let scratch = FitScratch::new(client, blocks);

        Some(Self {
            client: client.clone(),
            ctx,
            deep_ctx,
            width: luma_width,
            height: luma_height,
            input,
            ring,
            scratch,
            pyramid,
            slot_tables,
            gap_tables,
            predictions,
            field,
            sad,
            confidence_placeholder,
            noise,
            plane_placeholder,
            run: None,
        })
    }

    pub fn ring(&self) -> &ModelRing {
        &self.ring
    }

    /// Starts over, so the next frame pushed writes identity.
    pub fn reset(&mut self) {
        self.run = None;
    }

    /// Fits the pair from stream frame `frame - 1` to `frame` into ring slot `frame`.
    ///
    /// `luma` is the frame's luma plane in the input sample format. A frame that does not follow
    /// the last one pushed starts a new run and writes identity.
    pub fn push(&mut self, luma: &DevicePlane<'_>, frame: u32) -> Result<(), anyhow::Error> {
        let newer_slot = frame % PYRAMID_SLOTS;
        self.build_pyramid(luma, newer_slot);

        let run = self.run.filter(|run| run.continues_to(frame));
        let Some(run) = run else {
            self.run = Some(Run {
                first: frame,
                last: frame,
            });
            self.write_identity(frame);
            return Ok(());
        };

        self.run = Some(Run {
            first: run.first,
            last: frame,
        });

        let older_slot = (frame - 1) % PYRAMID_SLOTS;
        let slot = self.ring.slot(frame);
        let has_prior = run.first + 1 < frame;
        let prior = has_prior.then(|| self.ring.slot(frame - 1));

        if let Some(prior_slot) = prior {
            self.predict(prior_slot);
        }

        self.match_pair(&self.deep_ctx, older_slot, newer_slot, has_prior)?;
        self.fit(prior, slot);

        self.predict(slot);
        self.match_pair(&self.ctx, older_slot, newer_slot, true)?;
        self.fit(Some(slot), slot);

        Ok(())
    }

    /// Normalises `luma` into pyramid slot `slot` and builds its coarser levels.
    fn build_pyramid(&self, luma: &DevicePlane<'_>, slot: u32) {
        let align = self.deep_ctx.align;
        let levels = self.deep_ctx.pyramid_levels;
        let pyramid_pixels = pyramid_pixels_per_frame(self.width, self.height, levels, align);
        let offset_bytes = pyramid_slot_byte_offset(self.width, self.height, PYRAMID_SLOTS, 0, slot, align);
        let offset = offset_bytes / size_of::<f32>() as u64;
        let target = IngestTarget {
            ring: &self.pyramid,
            ring_len: pyramid_pixels * PYRAMID_SLOTS as usize,
            offset: offset as u32,
            pixels: self.width * self.height,
            channels: 1,
        };
        let planes = [*luma];

        ingest(&self.client, &planes, self.input, &self.plane_placeholder, target);
        run_pyramid_downscale(
            &self.client,
            &self.deep_ctx,
            self.width,
            self.height,
            PYRAMID_SLOTS,
            slot,
            &self.pyramid,
        );
    }

    /// Composes the one-step prediction of the model in ring slot `slot`.
    fn predict(&self, slot: u32) {
        let gaps = &self.gap_tables[slot as usize];

        run_compose_models(&self.client, &self.ring, gaps, &self.predictions, 1, 1, 1.0, 1.0);
    }

    /// Matches the older frame's blocks into the newer frame with the search of `ctx`, also around
    /// the prediction when `seeded`.
    fn match_pair(
        &self,
        ctx: &MotionCtx,
        older_slot: u32,
        newer_slot: u32,
        seeded: bool,
    ) -> Result<(), anyhow::Error> {
        let predictions = seeded.then_some(&self.predictions);
        let targets = MatchTargets {
            pyramid: &self.pyramid,
            frame_count: PYRAMID_SLOTS,
            centre_slot: older_slot,
            neighbour_slots: &self.slot_tables[newer_slot as usize],
            neighbours: 1,
            mv_field: &self.field,
            mv_len: ctx.pair_direction_len() as usize,
            confidence: &self.confidence_placeholder,
            conf_len: 1,
            write_confidence: false,
            predictions,
            sad: Some(&self.sad),
        };

        run_analyse(
            &self.client,
            ctx,
            self.width,
            self.height,
            &targets,
            &self.noise,
            DERIVED_SAD_FLOOR,
            1.0,
        )
    }

    /// Fits the matched field into ring slot `slot`, starting from ring slot `prior` when given.
    fn fit(&self, prior: Option<u32>, slot: u32) {
        let field = FitField {
            pairs: &self.field,
            direction: 0,
            sad: Some(&self.sad),
        };

        run_model_fit(
            &self.client,
            &self.ctx,
            self.width,
            self.height,
            field,
            &self.scratch,
            prior,
            &self.ring,
            slot,
        );
    }

    fn write_identity(&self, frame: u32) {
        let slot = self.ring.slot(frame);
        let cube_count = CubeCount::new_1d(1);
        let cube_dim = CubeDim::new_1d(1);
        let models_arg = bind(self.ring.handle(), self.ring.words());

        // SAFETY: the binding covers the ring and the slot lies inside it.
        unsafe {
            mc_model_identity::launch_unchecked(&self.client, cube_count, cube_dim, models_arg, slot);
        }
    }
}

/// A one-word `u32` table holding `value`.
fn single_word_table(client: &Client, value: u32) -> Handle {
    let table = [value];
    let bytes = u32::as_bytes(&table);

    client.create_from_slice(bytes)
}

fn checked_times(count: Option<u64>, factor: u64) -> Option<u64> {
    count.and_then(|count| count.checked_mul(factor))
}

/// The element count of every buffer [PairModelProducer::new] sizes from the frame.
pub(crate) fn producer_buffer_sizes(
    client: &Client,
    luma_width: u32,
    luma_height: u32,
    mode: &MotionCompensationMode,
    depth: u32,
) -> Vec<BufferSize> {
    let align = StorageAlign::from_client(client);
    let Some(ctx) = acquisition_ctx(mode, luma_width, luma_height, align) else {
        return Vec::new();
    };

    let slots = u64::from(PYRAMID_SLOTS);
    let per_frame = pyramid_pixels_per_frame(luma_width, luma_height, ctx.pyramid_levels, align);
    let per_frame = u64::try_from(per_frame).ok();
    let pyramid = checked_times(per_frame, slots);

    let blocks = u64::from(ctx.blocks_x).checked_mul(u64::from(ctx.blocks_y));
    let field_bytes = checked_times(blocks, 2 * size_of::<i32>() as u64);
    let field_bytes = field_bytes.and_then(|bytes| align.checked_pad_bytes(bytes));
    let field = field_bytes.map(|bytes| bytes / size_of::<i32>() as u64);
    let sad_bytes = checked_times(blocks, size_of::<f32>() as u64);
    let sad_bytes = sad_bytes.and_then(|bytes| align.checked_pad_bytes(bytes));
    let sad = sad_bytes.map(|bytes| bytes / size_of::<f32>() as u64);

    let partial_count = blocks.map(|blocks| blocks.div_ceil(u64::from(FIT_THREADS)));
    let partials = checked_times(partial_count, u64::from(FIT_LANES));
    let ring = u64::from(depth) * u64::from(MODEL_WORDS);

    vec![
        ("pair model pyramid", pyramid),
        ("pair model field", field),
        ("pair model SAD", sad),
        ("pair model fit weights", blocks),
        ("pair model fit partials", partials),
        ("pair model ring", Some(ring)),
    ]
}

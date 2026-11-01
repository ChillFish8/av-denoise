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
use crate::motion::kernels::{mc_block_means, mc_box_rows, mc_high_pass_cols, mc_model_identity};
use crate::motion::pyramid::{pyramid_pixels_per_frame, pyramid_slot_byte_offset, run_pyramid_downscale};
use crate::motion::{MAX_PYRAMID_LEVELS, MatchTargets, MotionCompensationMode, MotionCtx, run_analyse};
use crate::noise::{DERIVED_SAD_FLOOR, DERIVED_WORDS};
use crate::tune::gpu::{bind, zeroed_words};

/// Frames the producer's pyramid holds, the newest and the one before it.
const PYRAMID_SLOTS: u32 = 2;

/// How far either side, in luma pixels, the local mean each frame loses before matching reaches.
///
/// A flickering lamp or a flare changes the brightness of whole regions between two frames, which
/// swamps their SADs. Removing the local mean keeps the detail the match needs.
const HIGH_PASS_RADIUS: u32 = 8;

/// The side of the square workgroup the high-pass kernels run.
const HIGH_PASS_TILE: u32 = 16;

/// The coarse search radius of a pair's first match, on its quarter-resolution level.
const ACQUISITION_COARSE_RADIUS: u32 = 12;

/// How many times wider a coarse block of a pair's first match is than its content-matched size.
///
/// A quarter-resolution block of 4 pixels holds too little detail to tell matches apart.
const ACQUISITION_COARSE_BLOCK_FACTOR: u32 = 2;

/// The deep search a pair's first match runs, on the block grid and fine radius of `mode`.
///
/// It searches the most pyramid levels with a wide coarse radius, so a pan of about 50 luma pixels
/// is found without a model to start from.
fn acquisition_ctx(
    mode: &MotionCompensationMode,
    luma_width: u32,
    luma_height: u32,
    align: StorageAlign,
) -> Option<MotionCtx> {
    let MotionCompensationMode::Mvtools {
        blksize,
        overlap,
        search_radius,
        estimation,
        ..
    } = *mode
    else {
        return None;
    };

    let deep = MotionCompensationMode::Mvtools {
        blksize,
        overlap,
        search_radius,
        pyramid_levels: MAX_PYRAMID_LEVELS,
        estimation,
    };
    let ctx = MotionCtx::new(deep, luma_width, luma_height, align)?;
    let wide = ctx.with_coarse_search(ACQUISITION_COARSE_RADIUS, ACQUISITION_COARSE_BLOCK_FACTOR);

    Some(wide)
}

/// The camera models every pass reads, with the scale from a pass's pixels to luma pixels.
#[derive(Debug, Clone)]
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
/// Each frame loses its local mean first, so a change of lighting does not swamp the match. Each
/// push matches the new frame against the one before it, fits a model to the block vectors,
/// matches again around that model and fits once more. The first match runs a deep search, so a
/// fast pan is found without a model to start from. Each model depends on its own pair alone, and
/// the first frame of a run writes identity.
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
    /// Each block's winning SAD, window mean SAD and mean sample, one confidence stride apart.
    ///
    /// The mean sample is of the newer frame, which matches the older one on every still block.
    sad: Handle,
    /// The newest frame's normalised luma before the high-pass.
    luma: Handle,
    /// The newest frame's luma averaged along each row.
    luma_rows: Handle,
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
        let sad_bytes = 3 * ctx.confidence_bytes_per_neighbour() as usize;
        let predictions_bytes = AFFINE_WORDS as usize * size_of::<f32>();
        let field = client.empty(field_bytes);
        let sad = client.empty(sad_bytes);
        let plane_bytes = (luma_width * luma_height) as usize * size_of::<f32>();
        let luma = client.empty(plane_bytes);
        let luma_rows = client.empty(plane_bytes);
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
            luma,
            luma_rows,
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

        self.match_pair(&self.deep_ctx, older_slot, newer_slot, false)?;
        self.fit(None, slot);

        self.predict(slot);
        self.match_pair(&self.ctx, older_slot, newer_slot, true)?;
        self.fit(Some(slot), slot);

        Ok(())
    }

    /// Normalises `luma`, measures its block means, and writes it high-passed into pyramid slot
    /// `slot` with its coarser levels.
    fn build_pyramid(&self, luma: &DevicePlane<'_>, slot: u32) {
        let pixels = self.width * self.height;
        let target = IngestTarget {
            ring: &self.luma,
            ring_len: pixels as usize,
            offset: 0,
            pixels,
            channels: 1,
        };
        let planes = [*luma];

        ingest(&self.client, &planes, self.input, &self.plane_placeholder, target);
        self.measure_block_means();
        self.high_pass_into(slot);
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

    /// Writes the block means of the newest luma to the third confidence stride of `sad`.
    fn measure_block_means(&self) {
        let pixels = (self.width * self.height) as usize;
        let blocks = self.ctx.blocks_x * self.ctx.blocks_y;
        let means_offset = 2 * self.ctx.confidence_bytes_per_neighbour();
        let means = self.sad.clone().offset_start(means_offset);
        let cube_count = CubeCount::new_1d(blocks.div_ceil(FIT_THREADS));
        let cube_dim = CubeDim::new_1d(FIT_THREADS);

        let luma_arg = bind(&self.luma, pixels);
        let means_arg = bind(&means, blocks as usize);

        // SAFETY: the bindings cover the luma plane and one stride of `sad`, and the kernel guards
        // every block index.
        unsafe {
            mc_block_means::launch_unchecked(
                &self.client,
                cube_count,
                cube_dim,
                luma_arg,
                means_arg,
                self.width,
                self.height,
                self.ctx.blksize,
                self.ctx.step,
                self.ctx.blocks_x,
                self.ctx.blocks_y,
            );
        }
    }

    /// Writes the newest luma minus its local mean to level 0 of pyramid slot `slot`.
    fn high_pass_into(&self, slot: u32) {
        let pixels = (self.width * self.height) as usize;
        let align = self.deep_ctx.align;
        let offset_bytes = pyramid_slot_byte_offset(self.width, self.height, PYRAMID_SLOTS, 0, slot, align);
        let level0 = self.pyramid.clone().offset_start(offset_bytes);
        let cubes_x = self.width.div_ceil(HIGH_PASS_TILE);
        let cubes_y = self.height.div_ceil(HIGH_PASS_TILE);

        let luma_arg = bind(&self.luma, pixels);
        let rows_arg = bind(&self.luma_rows, pixels);
        let grid = CubeCount::new_2d(cubes_x, cubes_y);
        let dim = CubeDim::new_2d(HIGH_PASS_TILE, HIGH_PASS_TILE);

        // SAFETY: both bindings cover a whole luma plane, and the kernel guards every pixel.
        unsafe {
            mc_box_rows::launch_unchecked(
                &self.client,
                grid,
                dim,
                luma_arg,
                rows_arg,
                self.width,
                self.height,
                HIGH_PASS_RADIUS,
            );
        }

        let luma_arg = bind(&self.luma, pixels);
        let rows_arg = bind(&self.luma_rows, pixels);
        let level0_arg = bind(&level0, pixels);
        let grid = CubeCount::new_2d(cubes_x, cubes_y);
        let dim = CubeDim::new_2d(HIGH_PASS_TILE, HIGH_PASS_TILE);

        // SAFETY: every binding covers one luma plane, and level 0 of a slot holds a whole plane.
        unsafe {
            mc_high_pass_cols::launch_unchecked(
                &self.client,
                grid,
                dim,
                luma_arg,
                rows_arg,
                level0_arg,
                self.width,
                self.height,
                HIGH_PASS_RADIUS,
            );
        }
    }

    /// Composes the one-step prediction of the model in ring slot `slot`.
    fn predict(&self, slot: u32) {
        let gaps = &self.gap_tables[slot as usize];

        run_compose_models(
            &self.client,
            &self.ring,
            gaps,
            &self.predictions,
            1,
            0,
            1,
            1.0,
            1.0,
        );
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
            has_contrast: true,
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
    let sad_stride = sad_bytes.map(|bytes| bytes / size_of::<f32>() as u64);
    let sad = checked_times(sad_stride, 3);
    let plane = u64::from(luma_width) * u64::from(luma_height);

    let partial_count = blocks.map(|blocks| blocks.div_ceil(u64::from(FIT_THREADS)));
    let partials = checked_times(partial_count, u64::from(FIT_LANES));
    let ring = u64::from(depth) * u64::from(MODEL_WORDS);

    vec![
        ("pair model pyramid", pyramid),
        ("pair model field", field),
        ("pair model SAD", sad),
        ("pair model luma", Some(plane)),
        ("pair model luma rows", Some(plane)),
        ("pair model fit weights", blocks),
        ("pair model fit partials", partials),
        ("pair model ring", Some(ring)),
    ]
}

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::align::StorageAlign;
use crate::motion::truemotion::{TruemotionParams, TruemotionTargets, run_truemotion};
use crate::motion::{
    MatchTargets,
    MotionCompensationMode,
    MotionCtx,
    MotionEstimation,
    THSAD_PIXEL,
    level_dims,
    pyramid_slot_pixels,
    run_analyse,
};
use crate::noise::DERIVED_SAD_FLOOR;
use crate::noise::tests::sad_floor_buffer;
use crate::tune::KernelSelector;
use crate::tune::gpu::{read_f32, read_i32};
use crate::tune::tests::fixtures::upload;

pub(super) const BLKSIZE: u32 = 16;
pub(super) const STEP: u32 = 8;

pub(super) fn thsad() -> f32 {
    (BLKSIZE * BLKSIZE) as f32 * THSAD_PIXEL
}

/// The SAD two copies of a block with independent noise of `sigma` score by chance.
pub(super) fn noise_floor(sigma: f32) -> f32 {
    (BLKSIZE * BLKSIZE) as f32 * 2.0 * sigma / std::f32::consts::PI.sqrt()
}

/// A hashed value between 0 and 1 for `index` under `seed`.
pub(super) fn hashed(index: u32, seed: u32) -> f32 {
    let seed_offset = seed.wrapping_mul(0x9E37_79B9);
    let mut hash = index.wrapping_add(seed_offset);
    hash = hash.wrapping_mul(2_654_435_761);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x85EB_CA6B);
    hash ^= hash >> 13;
    hash as f32 / u32::MAX as f32
}

/// A frame with hashed values between 0.2 and 0.8 everywhere.
pub(super) fn textured(width: u32, height: u32, seed: u32) -> Vec<f32> {
    (0..width * height)
        .map(|index| 0.2 + 0.6 * hashed(index, seed))
        .collect()
}

/// Adds roughly Gaussian noise of `sigma` to every value of `frame`, hashed from `seed`.
pub(super) fn add_noise(frame: &mut [f32], sigma: f32, seed: u32) {
    // The sum of four uniform values has a variance of a third, so scaling by `sqrt(3)` gives one.
    let scale = sigma * 3.0f32.sqrt();
    for (index, value) in frame.iter_mut().enumerate() {
        let draws: f32 = (0..4u32).map(|draw| hashed(index as u32 * 4 + draw, seed)).sum();
        let unit = draws - 2.0;
        *value += unit * scale;
    }
}

/// `neighbour(x, y) = centre(x - dx, y - dy)`, so the true vector is `(dx, dy)`.
pub(super) fn shifted(centre: &[f32], width: u32, height: u32, dx: i32, dy: i32) -> Vec<f32> {
    let mut frame = vec![0.0f32; (width * height) as usize];
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let source_x = (x - dx).clamp(0, width as i32 - 1) as u32;
            let source_y = (y - dy).clamp(0, height as i32 - 1) as u32;
            frame[(y as u32 * width + x as u32) as usize] = centre[(source_y * width + source_x) as usize];
        }
    }

    frame
}

pub(super) fn uniform_field(blocks: usize, vector: [i32; 2]) -> Vec<i32> {
    let mut field = Vec::with_capacity(2 * blocks);
    for _ in 0..blocks {
        field.push(vector[0]);
        field.push(vector[1]);
    }

    field
}

/// A pure translation by `(tx, ty)` in the camera model's six words.
pub(super) fn translation(tx: f32, ty: f32) -> [f32; 6] {
    [1.0, 0.0, 0.0, 1.0, tx, ty]
}

/// A search geometry over `width` x `height` with `pyramid_levels` levels and the stage at `params`.
pub(super) fn stage_ctx(
    client: &Client,
    width: u32,
    height: u32,
    search_radius: u32,
    pyramid_levels: u32,
    params: TruemotionParams,
) -> MotionCtx {
    let mode = MotionCompensationMode::Mvtools {
        blksize: BLKSIZE,
        overlap: BLKSIZE - STEP,
        search_radius,
        max_search_escalation: search_radius,
        pyramid_levels,
        estimation: MotionEstimation::Direct,
    };
    let align = StorageAlign::from_client(client);
    let ctx = MotionCtx::new(mode, width, height, align).expect("the geometry is valid");

    MotionCtx {
        truemotion: params,
        ..ctx
    }
}

/// One centre frame matched against one neighbour.
pub(super) struct StageInput<'a> {
    pub width: u32,
    pub height: u32,
    pub centre: &'a [f32],
    pub neighbour: &'a [f32],
    /// The field the stage starts from, two values per block.
    pub field: &'a [i32],
    /// The search's seeds, two values per block.
    pub seeds: Option<&'a [i32]>,
    pub prediction: Option<[f32; 6]>,
    pub sad_floor: f32,
    pub reads_floor: bool,
}

/// The field and confidence a stage or search wrote, without padding.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct StageOutput {
    pub field: Vec<i32>,
    pub confidence: Vec<f32>,
}

impl StageOutput {
    pub fn vector(&self, block: usize) -> [i32; 2] {
        [self.field[2 * block], self.field[2 * block + 1]]
    }

    /// The confidence as bit patterns, so a comparison is exact.
    pub fn confidence_bits(&self) -> Vec<u32> {
        self.confidence.iter().map(|value| value.to_bits()).collect()
    }
}

/// The buffers one test search binds.
struct SearchBuffers {
    pyramid: Handle,
    neighbour_slots: Handle,
    mv_field: Handle,
    confidence: Handle,
    predictions: Option<Handle>,
    noise: Handle,
    seeds: Handle,
    field: Handle,
    changes: Handle,
}

/// Halves a frame by averaging each 2x2 square, as the pyramid build does.
fn half_level(frame: &[f32], width: u32, height: u32) -> Vec<f32> {
    let (half_width, half_height) = level_dims(width, height, 1);
    let mut level = Vec::with_capacity((half_width * half_height) as usize);

    for y in 0..half_height {
        for x in 0..half_width {
            let top = (2 * y * width + 2 * x) as usize;
            let bottom = top + width as usize;
            let sum = frame[top] + frame[top + 1] + frame[bottom] + frame[bottom + 1];
            level.push(sum * 0.25);
        }
    }

    level
}

/// The pyramid of `input`'s two frames with `ctx`'s levels, each level spanning both frame slots.
fn pyramid_values(ctx: &MotionCtx, input: &StageInput<'_>) -> Vec<f32> {
    let frames = [input.centre.to_vec(), input.neighbour.to_vec()];
    let mut pyramid = Vec::new();

    for level in 0..ctx.pyramid_levels {
        let slot_pixels = pyramid_slot_pixels(input.width, input.height, level, ctx.align) as usize;
        for frame in &frames {
            let level_frame = if level == 0 {
                frame.clone()
            } else {
                half_level(frame, input.width, input.height)
            };
            let padding = slot_pixels - level_frame.len();
            let padding_values = std::iter::repeat_n(0.0, padding);

            pyramid.extend_from_slice(&level_frame);
            pyramid.extend(padding_values);
        }
    }

    pyramid
}

/// Uploads `input` into the layout a search reads, with the stage's own buffers holding
/// `scratch_fill` everywhere except the seeds `input` gives.
fn upload_search(
    client: &Client,
    ctx: &MotionCtx,
    input: &StageInput<'_>,
    scratch_fill: i32,
) -> SearchBuffers {
    assert!(ctx.pyramid_levels <= 2, "the harness builds at most two levels");

    let frames = pyramid_values(ctx, input);
    let pyramid = upload(client, &frames);

    let table = [1u32];
    let table_bytes = u32::as_bytes(&table);
    let neighbour_slots = client.create_from_slice(table_bytes);

    let mv_len = ctx.mv_stride() as usize;
    let field_len = input.field.len();
    let mut field_words = vec![scratch_fill; mv_len];
    field_words[..field_len].copy_from_slice(input.field);
    let field_bytes = i32::as_bytes(&field_words);
    let mv_field = client.create_from_slice(field_bytes);

    let mut seed_words = vec![scratch_fill; mv_len];
    if let Some(seeds) = input.seeds {
        seed_words[..seeds.len()].copy_from_slice(seeds);
    }

    let seed_bytes = i32::as_bytes(&seed_words);
    let seeds = client.create_from_slice(seed_bytes);
    let stage_words = vec![scratch_fill; mv_len];
    let stage_bytes = i32::as_bytes(&stage_words);
    let field = client.create_from_slice(stage_bytes);

    let conf_len = ctx.conf_stride() as usize;
    let confidence = client.empty(conf_len * size_of::<f32>());
    let change_words = vec![scratch_fill as u32; 2 * conf_len];
    let change_bytes = u32::as_bytes(&change_words);
    let changes = client.create_from_slice(change_bytes);
    let predictions = input.prediction.map(|words| upload(client, &words));
    let noise = sad_floor_buffer(client, DERIVED_SAD_FLOOR, input.sad_floor);

    SearchBuffers {
        pyramid,
        neighbour_slots,
        mv_field,
        confidence,
        predictions,
        noise,
        seeds,
        field,
        changes,
    }
}

/// Reads the unpadded field and confidence of `ctx`'s blocks back.
fn read_output(client: &Client, ctx: &MotionCtx, buffers: &SearchBuffers) -> StageOutput {
    let blocks = (ctx.blocks_x * ctx.blocks_y) as usize;
    let mv_len = ctx.mv_stride() as usize;
    let conf_len = ctx.conf_stride() as usize;
    let field = read_i32(client, &buffers.mv_field, mv_len).expect("the field reads back");
    let confidence = read_f32(client, &buffers.confidence, conf_len).expect("the confidence reads back");

    StageOutput {
        field: field[..2 * blocks].to_vec(),
        confidence: confidence[..blocks].to_vec(),
    }
}

/// The targets of a one-neighbour search over `buffers`.
fn search_targets<'a>(
    ctx: &MotionCtx,
    buffers: &'a SearchBuffers,
    stage: Option<TruemotionTargets<'a>>,
) -> MatchTargets<'a> {
    MatchTargets {
        pyramid: &buffers.pyramid,
        frame_count: 2,
        centre_slot: 0,
        neighbour_slots: &buffers.neighbour_slots,
        neighbours: 1,
        mv_field: &buffers.mv_field,
        mv_len: ctx.mv_stride() as usize,
        confidence: &buffers.confidence,
        conf_len: ctx.conf_stride() as usize,
        write_confidence: true,
        predictions: buffers.predictions.as_ref(),
        sad: None,
        escalation: None,
        truemotion: stage,
    }
}

/// Runs the stage at `params` over `input`, with its own buffers starting at `scratch_fill`.
pub(super) fn run_stage_with(
    client: &Client,
    input: &StageInput<'_>,
    params: TruemotionParams,
    scratch_fill: i32,
) -> StageOutput {
    let ctx = stage_ctx(client, input.width, input.height, 1, 1, params);
    let buffers = upload_search(client, &ctx, input, scratch_fill);
    let kernels = KernelSelector::baseline(client);
    let stage = TruemotionTargets {
        kernels: &kernels,
        seeds: &buffers.seeds,
        field: &buffers.field,
        changes: &buffers.changes,
        reads_floor: input.reads_floor,
    };
    let targets = search_targets(&ctx, &buffers, Some(stage));
    let has_seeds = input.seeds.is_some();

    run_truemotion(
        &ctx,
        input.width,
        input.height,
        &targets,
        &stage,
        has_seeds,
        &buffers.noise,
        DERIVED_SAD_FLOOR,
        thsad(),
    )
    .expect("the stage runs");

    read_output(client, &ctx, &buffers)
}

/// [run_stage_with] with the stage's own buffers zeroed.
pub(super) fn run_stage(client: &Client, input: &StageInput<'_>, params: TruemotionParams) -> StageOutput {
    run_stage_with(client, input, params, 0)
}

/// How a test search runs.
#[derive(Debug, Clone, Copy)]
pub(super) struct SearchShape {
    pub search_radius: u32,
    /// One or two.
    pub pyramid_levels: u32,
    pub params: TruemotionParams,
    /// What the stage's own buffers hold before the search.
    pub scratch_fill: i32,
}

/// Runs a search of `shape` over `input`, ending with the stage.
///
/// The search finds its own seeds, so `input`'s field and seeds are unread.
pub(super) fn run_search(client: &Client, input: &StageInput<'_>, shape: SearchShape) -> StageOutput {
    let ctx = stage_ctx(
        client,
        input.width,
        input.height,
        shape.search_radius,
        shape.pyramid_levels,
        shape.params,
    );
    let scratch_fill = shape.scratch_fill;
    let buffers = upload_search(client, &ctx, input, scratch_fill);
    let kernels = KernelSelector::baseline(client);
    let stage = TruemotionTargets {
        kernels: &kernels,
        seeds: &buffers.seeds,
        field: &buffers.field,
        changes: &buffers.changes,
        reads_floor: input.reads_floor,
    };
    let targets = search_targets(&ctx, &buffers, Some(stage));

    run_analyse(
        client,
        &ctx,
        input.width,
        input.height,
        &targets,
        &buffers.noise,
        DERIVED_SAD_FLOOR,
        thsad(),
    )
    .expect("the search runs");

    read_output(client, &ctx, &buffers)
}

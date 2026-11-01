use cubecl::server::Handle;

use crate::engine::copy_words_into;
use crate::motion::{
    MatchTargets,
    MotionCtx,
    THSAD_PIXEL,
    level_dims,
    pyramid_pixels_per_frame,
    pyramid_slot_pixels,
};
use crate::tune::KernelSelector;
use crate::tune::sites::{TruemotionOutputs, TruemotionSite, TruemotionValues};

/// Where a search's truemotion stage keeps its working fields, and what launches its passes.
#[derive(Clone, Copy)]
pub(crate) struct TruemotionTargets<'a> {
    pub kernels: &'a KernelSelector,
    /// Holds as many `i32` values as the search's motion field, for the search's seeds.
    pub seeds: &'a Handle,
    /// Holds as many `i32` values as the search's motion field, for the field between passes.
    pub field: &'a Handle,
    /// Holds two of the search's confidence layouts of `u32`, for the blocks each pass changed.
    pub changes: &'a Handle,
    /// Whether a positive derived word of the search's noise at its `sad_field` scales the penalty.
    ///
    /// A search whose noise is zeroed or does not belong to the window keeps the unscaled penalty.
    pub reads_floor: bool,
}

/// Runs the truemotion stage over the field a search left in `targets`.
///
/// Each pass picks every block's vector from its candidates under a penalty toward its
/// neighbourhood's median, reading only the field the pass before it wrote, so a vector one block
/// found spreads to its neighbours. A pass after the first rescores only the blocks next to one the
/// pass before changed. The candidates and penalty follow the
/// [TruemotionParams](crate::motion::truemotion::TruemotionParams) of `motion_ctx`. With
/// `has_seeds`, the seeds the search started from are candidates too. The final field lands back
/// in `targets`, with its confidence and SADs written as the search writes them, scored against the
/// derived word of `noise` at `sad_field` and `thsad`. It runs whatever the penalty, so the search
/// decides whether a stage at its strength runs at all.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
pub(crate) fn run_truemotion(
    motion_ctx: &MotionCtx,
    width: u32,
    height: u32,
    targets: &MatchTargets<'_>,
    stage: &TruemotionTargets<'_>,
    has_seeds: bool,
    noise: &Handle,
    sad_field: u32,
    thsad: f32,
) -> Result<(), anyhow::Error> {
    let params = motion_ctx.truemotion;
    let block_area = (motion_ctx.blksize * motion_ctx.blksize) as f32;
    let fade_level = params.fade_scale * thsad.max(0.0);
    let values = TruemotionValues {
        lambda_base: params.field_lambda * block_area * THSAD_PIXEL,
        lambda_noise: params.field_lambda * params.noise_weight,
        surcharge: params.surcharge,
        fade_level,
        thsad,
        sad_field,
    };

    let frame_pixels = pyramid_pixels_per_frame(width, height, motion_ctx.pyramid_levels, motion_ctx.align);
    let pyramid_len = frame_pixels * targets.frame_count as usize;
    let slot_pixels = pyramid_slot_pixels(width, height, 0, motion_ctx.align);
    let (level_width, level_height) = level_dims(width, height, 0);
    let blocks = (motion_ctx.blocks_x * motion_ctx.blocks_y) as usize;
    let seeds = has_seeds.then(|| stage.seeds.clone());
    let predictions = targets.predictions.cloned();
    let conf_stride = motion_ctx.conf_stride();
    let changes_len = targets.neighbours as u64 * conf_stride as u64;
    let changes_bytes = changes_len * size_of::<u32>() as u64;
    let change_halves = [
        stage.changes.clone(),
        stage.changes.clone().offset_start(changes_bytes),
    ];

    for pass in 0..params.passes {
        // Passes alternate between the search's field and the stage's own, so none reads what it
        // writes.
        let (field_in, field_out) = if pass % 2 == 0 {
            (targets.mv_field, stage.field)
        } else {
            (stage.field, targets.mv_field)
        };

        // Each pass flags its changes for the next one in the half the pass before did not use.
        let has_next = pass + 1 < params.passes;
        let changes_in = (pass > 0).then(|| change_halves[(pass as usize + 1) % 2].clone());
        let changes_out = has_next.then(|| change_halves[pass as usize % 2].clone());

        let outputs = TruemotionOutputs {
            vectors: field_out.clone(),
            vectors_len: targets.mv_len,
            confidence: targets.confidence.clone(),
            confidence_len: targets.conf_len,
            writes_confidence: targets.write_confidence,
            sad: targets.sad.cloned(),
            changes: changes_out,
            blocks,
            neighbours: targets.neighbours,
            mv_stride: motion_ctx.mv_stride(),
            conf_stride,
        };
        let site = TruemotionSite {
            client: stage.kernels.client().clone(),
            pyramid: targets.pyramid.clone(),
            pyramid_len,
            centre_slot: targets.centre_slot,
            neighbour_slots: targets.neighbour_slots.clone(),
            slot_pixels,
            mv_in: field_in.clone(),
            seeds: seeds.clone(),
            predictions: predictions.clone(),
            changes_in,
            outputs,
            values,
            noise: noise.clone(),
            reads_floor: stage.reads_floor,
            refine_radius: params.refine_radius,
            width: level_width,
            height: level_height,
            blksize: motion_ctx.blksize,
            step: motion_ctx.step,
            blocks_x: motion_ctx.blocks_x,
            blocks_y: motion_ctx.blocks_y,
        };

        stage.kernels.run(&site)?;
    }

    // An odd pass count leaves the final field in the stage's own buffer.
    if params.passes % 2 == 1 {
        let client = stage.kernels.client();
        copy_words_into(client, stage.field, targets.mv_field, targets.mv_len);
    }

    Ok(())
}

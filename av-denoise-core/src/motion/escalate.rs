use cubecl::prelude::*;
use cubecl::server::Handle;

use super::MotionCtx;
use super::analyse::{Escalation, MatchTargets, coarse_level, escalation_state_len, prediction_binding};
use super::kernels::{
    BLOCK_MATCH_THREADS,
    FLAG_THREADS,
    FLOOR_THREADS,
    mc_escalate_search,
    mc_flag_failures,
    mc_measure_floor,
};
use super::pyramid::{level_dims, pyramid_pixels_per_frame, pyramid_slot_pixels};
use crate::noise::DERIVED_WORDS;

/// Searches again, with wider windows, every block whose fine first pass failed.
///
/// It runs after a fine pass that recorded its escalation state in `targets`. It first writes the
/// SAD floor every failure test and margin scales by, as
/// [mc_measure_floor](super::kernels::mc_measure_floor) describes.
/// Each stage of the schedule then flags the blocks that still fail and searches them again around
/// the centre of their first-pass window, so the schedule's ceiling bounds the total reach. With
/// `widen_coarse`, a stage wider than the coarse pass's radius searches the coarse level again too.
/// A block keeps a wider result only when it beats the kept SAD by a floor-scaled margin. The
/// vectors, SADs and confidence in `targets` always describe the kept result, with confidence
/// scored against the derived word of `noise` at `sad_field` as in
/// [run_analyse](crate::motion::run_analyse).
///
/// Every launch covers the whole grid and unflagged blocks exit at once, so nothing is read back.
/// A schedule that does not escalate launches nothing. An escalating schedule needs a positive
/// `thsad`, since a `thsad` of 0 fails every block on confidence.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernels bind"
)]
pub(crate) fn run_escalation(
    client: &Client,
    motion_ctx: &MotionCtx,
    width: u32,
    height: u32,
    targets: &MatchTargets<'_>,
    noise: &Handle,
    sad_field: u32,
    thsad: f32,
    widen_coarse: bool,
) -> Result<(), anyhow::Error> {
    if !motion_ctx.schedule.escalates() {
        return Ok(());
    }

    let Some(escalation) = targets.escalation else {
        anyhow::bail!("an escalating search schedule needs escalation state in its targets");
    };
    let state = escalation.state;

    if thsad.is_nan() || thsad <= 0.0 {
        anyhow::bail!("an escalating search schedule needs a positive thsad, got {thsad}");
    }

    let scoring = Scoring {
        noise,
        sad_field,
        thsad,
    };
    launch_floor(client, motion_ctx, targets, &escalation, &scoring);

    for stage_radius in motion_ctx.schedule.stage_radii() {
        launch_flags(client, motion_ctx, targets, state, thsad);

        let stage = Stage {
            radius: stage_radius,
            widen_coarse,
        };
        launch_escalate(client, motion_ctx, width, height, targets, state, stage, &scoring);
    }

    Ok(())
}

/// How a search scores the confidence it writes.
pub(super) struct Scoring<'a> {
    pub(super) noise: &'a Handle,
    pub(super) sad_field: u32,
    pub(super) thsad: f32,
}

/// One widening stage of an escalation.
#[derive(Clone, Copy)]
struct Stage {
    radius: u32,
    widen_coarse: bool,
}

/// Launches the SAD floor measure over every neighbour in `targets`.
pub(super) fn launch_floor(
    client: &Client,
    motion_ctx: &MotionCtx,
    targets: &MatchTargets<'_>,
    escalation: &Escalation<'_>,
    scoring: &Scoring<'_>,
) {
    let grid = CubeCount::new_1d(1);
    let dim = CubeDim::new_1d(FLOOR_THREADS);
    let state_len = escalation_state_len(motion_ctx, targets.neighbours);
    let blocks = motion_ctx.blocks_x * motion_ctx.blocks_y;
    let state = escalation.state.clone();
    let noise = scoring.noise.clone();

    unsafe {
        let state_arg = BufferArg::from_raw_parts(state, state_len);
        let noise_arg = BufferArg::from_raw_parts(noise, DERIVED_WORDS as usize);

        mc_measure_floor::launch_unchecked(
            client,
            grid,
            dim,
            state_arg,
            noise_arg,
            scoring.sad_field,
            targets.neighbours,
            escalation.reads_floor,
            blocks,
            motion_ctx.conf_stride(),
        );
    }
}

/// Launches the failure flags over every block of every neighbour in `targets`.
pub(super) fn launch_flags(
    client: &Client,
    motion_ctx: &MotionCtx,
    targets: &MatchTargets<'_>,
    state: &Handle,
    thsad: f32,
) {
    let cubes_x = motion_ctx.blocks_x.div_ceil(FLAG_THREADS);
    let grid = CubeCount::new_3d(cubes_x, motion_ctx.blocks_y, targets.neighbours);
    let dim = CubeDim::new_1d(FLAG_THREADS);
    let state_len = escalation_state_len(motion_ctx, targets.neighbours);
    let mv_field = targets.mv_field.clone();
    let state = state.clone();

    unsafe {
        let mv_field_arg = BufferArg::from_raw_parts(mv_field, targets.mv_len);
        let state_arg = BufferArg::from_raw_parts(state, state_len);

        mc_flag_failures::launch_unchecked(
            client,
            grid,
            dim,
            mv_field_arg,
            state_arg,
            thsad,
            motion_ctx.blocks_x,
            motion_ctx.blocks_y,
            motion_ctx.mv_stride(),
            motion_ctx.conf_stride(),
        );
    }
}

/// Launches one escalation stage over every block of every neighbour.
///
/// The stage searches the coarse level again only when asked, the pyramid has one, and the stage
/// is wider than the coarse pass's own radius.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
fn launch_escalate(
    client: &Client,
    motion_ctx: &MotionCtx,
    width: u32,
    height: u32,
    targets: &MatchTargets<'_>,
    state: &Handle,
    stage: Stage,
    scoring: &Scoring<'_>,
) {
    let pyramid_pixels = pyramid_pixels_per_frame(width, height, motion_ctx.pyramid_levels, motion_ctx.align);
    let pyramid_len = pyramid_pixels * targets.frame_count as usize;
    let slots_len = targets.neighbours as usize;
    let (fine_width, fine_height) = level_dims(width, height, 0);
    let slot_pixels = pyramid_slot_pixels(width, height, 0, motion_ctx.align);
    let conf_stride = motion_ctx.conf_stride();
    let state_len = escalation_state_len(motion_ctx, targets.neighbours);
    let grid = CubeCount::new_3d(motion_ctx.blocks_x, motion_ctx.blocks_y, targets.neighbours);
    let dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);
    let write_sad = targets.sad.is_some();
    // An unwritten SAD output binds the confidence buffer, which the kernel then never indexes.
    let (sad, sad_len) = match targets.sad {
        Some(handle) => (handle.clone(), (2 * targets.neighbours * conf_stride) as usize),
        None => (targets.confidence.clone(), targets.conf_len),
    };
    let pyramid = targets.pyramid.clone();
    let neighbour_slots = targets.neighbour_slots.clone();
    let mv_field = targets.mv_field.clone();
    let confidence = targets.confidence.clone();
    let state = state.clone();
    let noise = scoring.noise.clone();
    let (predictions, predictions_len) = prediction_binding(targets, scoring.noise);
    let model_candidate = targets.predictions.is_some();

    let coarse = coarse_level(motion_ctx, width, height, targets.frame_count);
    let reaches_further = stage.radius > motion_ctx.coarse_search_radius;
    let coarse = coarse.filter(|_| stage.widen_coarse && reaches_further);
    let widen_coarse = coarse.is_some();
    // A stage that keeps to the fine level never reads these, so they take the fine level's shape.
    let (level_width, level_height, level_offset, level_slot_pixels) = match coarse.as_ref() {
        Some(level) => (level.width, level.height, level.level_offset, level.slot_pixels),
        None => (fine_width, fine_height, 0, slot_pixels),
    };
    let (coarse_blksize, level_scale, floor_scale) = match coarse.as_ref() {
        Some(level) => (level.blksize, level.scale, level.floor_scale),
        None => (motion_ctx.blksize, 1, 1.0),
    };

    unsafe {
        let pyramid_arg = BufferArg::from_raw_parts(pyramid, pyramid_len);
        let neighbour_slots_arg = BufferArg::from_raw_parts(neighbour_slots, slots_len);
        let mv_field_arg = BufferArg::from_raw_parts(mv_field, targets.mv_len);
        let confidence_arg = BufferArg::from_raw_parts(confidence, targets.conf_len);
        let sad_arg = BufferArg::from_raw_parts(sad, sad_len);
        let state_arg = BufferArg::from_raw_parts(state, state_len);
        let predictions_arg = BufferArg::from_raw_parts(predictions, predictions_len);
        let noise_arg = BufferArg::from_raw_parts(noise, DERIVED_WORDS as usize);

        mc_escalate_search::launch_unchecked(
            client,
            grid,
            dim,
            pyramid_arg,
            neighbour_slots_arg,
            mv_field_arg,
            confidence_arg,
            sad_arg,
            state_arg,
            predictions_arg,
            targets.centre_slot,
            targets.write_confidence,
            write_sad,
            model_candidate,
            noise_arg,
            scoring.sad_field,
            scoring.thsad,
            level_offset,
            level_slot_pixels,
            floor_scale,
            widen_coarse,
            level_width,
            level_height,
            coarse_blksize,
            level_scale,
            fine_width,
            fine_height,
            slot_pixels,
            motion_ctx.blksize,
            motion_ctx.step,
            stage.radius,
            motion_ctx.blocks_x,
            motion_ctx.mv_stride(),
            conf_stride,
        );
    }
}

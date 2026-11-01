use cubecl::prelude::*;
use cubecl::server::Handle;

use super::MotionCtx;
use super::analyse::{MatchTargets, escalation_state_len};
use super::kernels::{BLOCK_MATCH_THREADS, FLAG_THREADS, mc_escalate_search, mc_flag_failures};
use super::pyramid::{level_dims, pyramid_pixels_per_frame, pyramid_slot_pixels};
use crate::noise::DERIVED_WORDS;

/// Searches again, with wider windows, every block whose fine first pass failed.
///
/// It runs after a fine pass that recorded its escalation state in `targets`. Each stage of the
/// schedule flags the blocks that still fail, then searches them again around the centre of their
/// first-pass window, so the schedule's ceiling bounds the total reach. A block keeps a wider
/// result only when it beats the kept SAD by a noise-scaled margin. The vectors, SADs and confidence
/// in `targets` always describe the kept result, with confidence scored as in
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
) -> Result<(), anyhow::Error> {
    if !motion_ctx.schedule.escalates() {
        return Ok(());
    }

    let Some(state) = targets.escalation else {
        anyhow::bail!("an escalating search schedule needs escalation state in its targets");
    };

    if thsad.is_nan() || thsad <= 0.0 {
        anyhow::bail!("an escalating search schedule needs a positive thsad, got {thsad}");
    }

    for stage_radius in motion_ctx.schedule.stage_radii() {
        launch_flags(client, motion_ctx, targets, state, noise, sad_field, thsad);
        launch_escalate(
            client,
            motion_ctx,
            width,
            height,
            targets,
            state,
            stage_radius,
            noise,
            sad_field,
            thsad,
        );
    }

    Ok(())
}

/// Launches the failure flags over every block of every neighbour in `targets`.
pub(super) fn launch_flags(
    client: &Client,
    motion_ctx: &MotionCtx,
    targets: &MatchTargets<'_>,
    state: &Handle,
    noise: &Handle,
    sad_field: u32,
    thsad: f32,
) {
    let cubes_x = motion_ctx.blocks_x.div_ceil(FLAG_THREADS);
    let grid = CubeCount::new_3d(cubes_x, motion_ctx.blocks_y, targets.neighbours);
    let dim = CubeDim::new_1d(FLAG_THREADS);
    let state_len = escalation_state_len(motion_ctx, targets.neighbours);
    let mv_field = targets.mv_field.clone();
    let state = state.clone();
    let noise = noise.clone();

    unsafe {
        let mv_field_arg = BufferArg::from_raw_parts(mv_field, targets.mv_len);
        let state_arg = BufferArg::from_raw_parts(state, state_len);
        let noise_arg = BufferArg::from_raw_parts(noise, DERIVED_WORDS as usize);

        mc_flag_failures::launch_unchecked(
            client,
            grid,
            dim,
            mv_field_arg,
            state_arg,
            noise_arg,
            sad_field,
            thsad,
            motion_ctx.blocks_x,
            motion_ctx.blocks_y,
            motion_ctx.mv_stride(),
            motion_ctx.conf_stride(),
        );
    }
}

/// Launches one escalation stage at `stage_radius` over every block of every neighbour.
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
    stage_radius: u32,
    noise: &Handle,
    sad_field: u32,
    thsad: f32,
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
    let noise = noise.clone();

    unsafe {
        let pyramid_arg = BufferArg::from_raw_parts(pyramid, pyramid_len);
        let neighbour_slots_arg = BufferArg::from_raw_parts(neighbour_slots, slots_len);
        let mv_field_arg = BufferArg::from_raw_parts(mv_field, targets.mv_len);
        let confidence_arg = BufferArg::from_raw_parts(confidence, targets.conf_len);
        let sad_arg = BufferArg::from_raw_parts(sad, sad_len);
        let state_arg = BufferArg::from_raw_parts(state, state_len);
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
            targets.centre_slot,
            targets.write_confidence,
            write_sad,
            noise_arg,
            sad_field,
            thsad,
            fine_width,
            fine_height,
            slot_pixels,
            motion_ctx.blksize,
            motion_ctx.step,
            stage_radius,
            motion_ctx.blocks_x,
            motion_ctx.mv_stride(),
            conf_stride,
        );
    }
}

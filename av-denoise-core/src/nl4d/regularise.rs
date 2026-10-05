use cubecl::prelude::*;
use cubecl::server::Handle;

use super::kernels::nl4d_mv_regularise;
use crate::nlmeans::RingView;
use crate::nlmeans::motion::{
    MotionCtx,
    THSAD_PIXEL,
    confidence_byte_offset,
    level_dims,
    mv_field_byte_offset,
    pyramid_slot_byte_offset,
};

/// Runs the field regularisation pass over every neighbour of `view`.
///
/// Each block's vector is re-scored against the median of its neighbourhood. The results land in
/// `mv_out` and `conf_out`, which share the front end's per-neighbour layout.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
pub(super) fn run_regularise<R: Runtime>(
    client: &ComputeClient<R>,
    motion_ctx: &MotionCtx,
    view: &RingView,
    width: u32,
    height: u32,
    field_lambda: f32,
    sad_noise_floor: f32,
    thsad: f32,
    mv_out: &Handle,
    conf_out: &Handle,
) -> Result<(), anyhow::Error> {
    let (level_width, level_height) = level_dims(width, height, 0);
    let level_len = (level_width * level_height) as usize;
    let blocks = (motion_ctx.blocks_x * motion_ctx.blocks_y) as usize;
    let block_area = (motion_ctx.blksize * motion_ctx.blksize) as f32;
    let lambda_pixel = field_lambda * block_area * THSAD_PIXEL;
    let centre_offset = pyramid_slot_byte_offset(
        width,
        height,
        view.frame_count,
        0,
        view.centre_slot,
        motion_ctx.align,
    );
    let centre = view.pyramid.clone().offset_start(centre_offset);

    for (t, &slot) in view.neighbour_slots.iter().enumerate() {
        let t = t as u32;
        let neighbour_offset =
            pyramid_slot_byte_offset(width, height, view.frame_count, 0, slot, motion_ctx.align);
        let neighbour = view.pyramid.clone().offset_start(neighbour_offset);
        let mv_offset = mv_field_byte_offset(motion_ctx, t);
        let conf_offset = confidence_byte_offset(motion_ctx, t);
        let mv_in = view.mv_field.clone().offset_start(mv_offset);
        let mv_dst = mv_out.clone().offset_start(mv_offset);
        let conf_dst = conf_out.clone().offset_start(conf_offset);

        unsafe {
            nl4d_mv_regularise::launch_unchecked::<R>(
                client,
                CubeCount::new_2d(motion_ctx.blocks_x, motion_ctx.blocks_y),
                CubeDim::new_2d(8, 8),
                ArrayArg::from_raw_parts(centre.clone(), level_len),
                ArrayArg::from_raw_parts(neighbour, level_len),
                ArrayArg::from_raw_parts(mv_in, 2 * blocks),
                ArrayArg::from_raw_parts(mv_dst, 2 * blocks),
                ArrayArg::from_raw_parts(conf_dst, blocks),
                lambda_pixel,
                sad_noise_floor,
                thsad,
                level_width,
                level_height,
                motion_ctx.blksize,
                motion_ctx.step,
                motion_ctx.blocks_x,
                motion_ctx.blocks_y,
            );
        }
    }

    Ok(())
}

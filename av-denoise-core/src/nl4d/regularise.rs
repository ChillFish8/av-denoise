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

/// Runs the field regularisation pass over every neighbour of `view`,
/// writing the result into `mv_out` and `conf_out`, which share the
/// front end's per-neighbour layout.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
pub(super) fn run_regularise<R: Runtime>(
    client: &ComputeClient<R>,
    mc: &MotionCtx,
    view: &RingView,
    width: u32,
    height: u32,
    field_lambda: f32,
    sad_noise_floor: f32,
    thsad: f32,
    mv_out: &Handle,
    conf_out: &Handle,
) -> Result<(), anyhow::Error> {
    let (fw, fh) = level_dims(width, height, 0);
    let level_len = (fw * fh) as usize;
    let blocks = (mc.blocks_x * mc.blocks_y) as usize;
    let lambda_pixel = field_lambda * (mc.blksize * mc.blksize) as f32 * THSAD_PIXEL;
    let centre = view.pyramid.clone().offset_start(pyramid_slot_byte_offset(
        width,
        height,
        view.frame_count,
        0,
        view.centre_slot,
        mc.align,
    ));

    for (t, &slot) in view.neighbour_slots.iter().enumerate() {
        let t = t as u32;
        let neighbour = view.pyramid.clone().offset_start(pyramid_slot_byte_offset(
            width,
            height,
            view.frame_count,
            0,
            slot,
            mc.align,
        ));
        let mv_in = view.mv_field.clone().offset_start(mv_field_byte_offset(mc, t));
        let mv_dst = mv_out.clone().offset_start(mv_field_byte_offset(mc, t));
        let conf_dst = conf_out.clone().offset_start(confidence_byte_offset(mc, t));

        unsafe {
            nl4d_mv_regularise::launch_unchecked::<R>(
                client,
                CubeCount::new_2d(mc.blocks_x, mc.blocks_y),
                CubeDim::new_2d(8, 8),
                ArrayArg::from_raw_parts(centre.clone(), level_len),
                ArrayArg::from_raw_parts(neighbour, level_len),
                ArrayArg::from_raw_parts(mv_in, 2 * blocks),
                ArrayArg::from_raw_parts(mv_dst, 2 * blocks),
                ArrayArg::from_raw_parts(conf_dst, blocks),
                lambda_pixel,
                sad_noise_floor,
                thsad,
                fw,
                fh,
                mc.blksize,
                mc.step,
                mc.blocks_x,
                mc.blocks_y,
            );
        }
    }

    Ok(())
}

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::nlmeans::RingView;
use crate::nlmeans::motion::{
    MotionCtx,
    THSAD_PIXEL,
    level_dims,
    pyramid_pixels_per_frame,
    pyramid_slot_pixels,
};
use crate::tune::KernelSelector;
use crate::tune::sites::{RegulariseOutputs, RegulariseSite};

/// Runs the field regularisation pass over every neighbour of `view` in one launch, through `kernels`.
///
/// Each block's vector is re-scored against the median of its neighbourhood. The results land in
/// `mv_out` and `conf_out`, which share the front end's per-neighbour layout.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
pub(super) fn run_regularise<R: Runtime>(
    kernels: &KernelSelector<R>,
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
    let blocks = (motion_ctx.blocks_x * motion_ctx.blocks_y) as usize;
    let block_area = (motion_ctx.blksize * motion_ctx.blksize) as f32;
    let lambda_pixel = field_lambda * block_area * THSAD_PIXEL;
    let neighbours = view.neighbour_slots.len() as u32;
    let slot_pixels = pyramid_slot_pixels(width, height, 0, motion_ctx.align);
    let frame_pixels = pyramid_pixels_per_frame(width, height, motion_ctx.pyramid_levels, motion_ctx.align);
    let pyramid_len = frame_pixels * view.frame_count as usize;

    let outputs = RegulariseOutputs {
        vectors: mv_out.clone(),
        confidence: conf_out.clone(),
        blocks,
        neighbours,
        mv_stride: view.mv_stride,
        conf_stride: view.conf_stride,
    };
    let site = RegulariseSite {
        client: kernels.client().clone(),
        pyramid: view.pyramid.clone(),
        pyramid_len,
        centre_slot: view.centre_slot,
        neighbour_slots: view.neighbour_slots_buf.clone(),
        slot_pixels,
        mv_in: view.mv_field.clone(),
        outputs,
        lambda_pixel,
        sad_noise_floor,
        thsad,
        width: level_width,
        height: level_height,
        blksize: motion_ctx.blksize,
        step: motion_ctx.step,
        blocks_x: motion_ctx.blocks_x,
        blocks_y: motion_ctx.blocks_y,
    };

    kernels.run(&site)
}

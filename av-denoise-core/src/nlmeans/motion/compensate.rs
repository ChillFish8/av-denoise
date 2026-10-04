use cubecl::prelude::*;
use cubecl::server::Handle;

use super::MotionCtx;
use super::analyse::mv_field_byte_offset;
use crate::nlmeans::kernels::motion::nlm_mc_warp;

/// Shifts one neighbour frame into line with the centre frame, into its slot of `dst`.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
pub(crate) fn run_compensate<R: Runtime>(
    client: &ComputeClient<R>,
    motion_ctx: &MotionCtx,
    stored_ch: u32,
    width: u32,
    height: u32,
    frame_count: u32,
    neighbour_slot: u32,
    neighbour_idx: u32,
    src: &Handle,
    dst: &Handle,
    mv_field: &Handle,
) -> Result<(), anyhow::Error> {
    let block_x = 16u32;
    let block_y = 16u32;
    let cubes_x = width.div_ceil(block_x);
    let cubes_y = height.div_ceil(block_y);
    let grid = CubeCount::new_2d(cubes_x, cubes_y);
    let dim = CubeDim::new_2d(block_x, block_y);

    let total_pixels = (frame_count * height * width * stored_ch) as usize;
    let mv_slice_len = (motion_ctx.blocks_x as usize) * (motion_ctx.blocks_y as usize) * 2;
    let mv_offset = mv_field_byte_offset(motion_ctx, neighbour_idx);
    let mv_slice = mv_field.clone().offset_start(mv_offset);

    unsafe {
        nlm_mc_warp::launch_unchecked::<R>(
            client,
            grid,
            dim,
            stored_ch as usize,
            ArrayArg::from_raw_parts(src.clone(), total_pixels),
            ArrayArg::from_raw_parts(dst.clone(), total_pixels),
            ArrayArg::from_raw_parts(mv_slice, mv_slice_len),
            neighbour_slot,
            neighbour_slot,
            motion_ctx.step,
            motion_ctx.blocks_x,
            motion_ctx.blocks_y,
            width,
            height,
        );
    }

    Ok(())
}

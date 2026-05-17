//! Compensate (warp) pass dispatch. Takes the MV field produced by
//! `analyse` and produces an aligned copy of the neighbour frame in
//! the compensated ring buffer.

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::super::kernels::motion::nlm_mc_warp;
use super::analyse::mv_field_byte_offset;
use super::MotionCtx;

/// Run the compensate kernel for one neighbour, warping it toward the
/// centre frame. Produces a full-frame `Vector<f32, N>` write into
/// `compensated[neighbour_slot]`.
pub(crate) fn run_compensate<R: Runtime>(
    client: &ComputeClient<R>,
    mc: &MotionCtx,
    channels: u32,
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
    let _ = channels;
    let block_x = 16u32;
    let block_y = 16u32;
    let grid = CubeCount::new_2d(width.div_ceil(block_x), height.div_ceil(block_y));
    let dim = CubeDim::new_2d(block_x, block_y);

    let total_pixels = (frame_count * height * width * stored_ch) as usize;
    let mv_slice_len = (mc.blocks_x as usize) * (mc.blocks_y as usize) * 2;
    let mv_slice = mv_field
        .clone()
        .offset_start(mv_field_byte_offset(mc, neighbour_idx));

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
            mc.step,
            mc.blocks_x,
            mc.blocks_y,
            width,
            height,
        );
    }

    Ok(())
}

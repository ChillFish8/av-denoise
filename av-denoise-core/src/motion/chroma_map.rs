use cubecl::prelude::*;
use cubecl::server::Handle;

use super::MotionCtx;
use super::kernels::mc_map_luma_fields;

/// Threads along each side of the square workgroup that maps chroma blocks.
const MAP_THREADS: u32 = 8;

/// The chroma subsampling of a frame, as a power of two per axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Subsampling {
    pub shift_x: u32,
    pub shift_y: u32,
}

/// A motion field's block grid and the `i32` stride between its fields.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FieldLayout {
    pub blocks_x: u32,
    pub blocks_y: u32,
    pub step: u32,
    pub blksize: u32,
    pub stride: u32,
}

impl FieldLayout {
    /// The layout of the neighbour fields of a motion window.
    pub(crate) fn window(ctx: &MotionCtx) -> Self {
        Self {
            blocks_x: ctx.blocks_x,
            blocks_y: ctx.blocks_y,
            step: ctx.step,
            blksize: ctx.blksize,
            stride: ctx.mv_stride(),
        }
    }

    /// The layout of the two directions of a look-ahead pair.
    pub(crate) fn pair(ctx: &MotionCtx) -> Self {
        Self {
            stride: ctx.pair_direction_stride(),
            ..Self::window(ctx)
        }
    }
}

/// Maps `fields` luma motion fields onto the chroma block grid.
pub(crate) fn map_luma_fields(
    client: &Client,
    luma_field: &Handle,
    luma: FieldLayout,
    chroma_field: &Handle,
    chroma: FieldLayout,
    subsampling: Subsampling,
    fields: u32,
) {
    let area_x = chroma.blksize << subsampling.shift_x;
    let area_y = chroma.blksize << subsampling.shift_y;
    let covers_x = area_x.saturating_sub(luma.blksize) / luma.step + 1;
    let covers_y = area_y.saturating_sub(luma.blksize) / luma.step + 1;

    let cubes_x = chroma.blocks_x.div_ceil(MAP_THREADS);
    let cubes_y = chroma.blocks_y.div_ceil(MAP_THREADS);
    let grid = CubeCount::new_3d(cubes_x, cubes_y, fields);
    let dim = CubeDim::new_2d(MAP_THREADS, MAP_THREADS);
    let luma_len = (fields * luma.stride) as usize;
    let chroma_len = (fields * chroma.stride) as usize;

    unsafe {
        mc_map_luma_fields::launch_unchecked(
            client,
            grid,
            dim,
            BufferArg::from_raw_parts(luma_field.clone(), luma_len),
            BufferArg::from_raw_parts(chroma_field.clone(), chroma_len),
            luma.blocks_x,
            luma.blocks_y,
            luma.step,
            luma.stride,
            chroma.blocks_x,
            chroma.blocks_y,
            chroma.step,
            chroma.stride,
            subsampling.shift_x,
            subsampling.shift_y,
            covers_x,
            covers_y,
        );
    }
}

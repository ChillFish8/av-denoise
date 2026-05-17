//! Motion-vector analyse pass dispatch. Hierarchical block-matching:
//! coarse pass on the `/2` pyramid level seeds a fine pass at full
//! resolution; result is a per-block i16x2 MV field per neighbour.

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::super::kernels::motion::{nlm_mc_block_match_coarse, nlm_mc_block_match_fine};
use super::pyramid::{level_dims, pyramid_slot_byte_offset};
use super::MotionCtx;

/// Byte offset of the MV-field slice for a given neighbour index.
/// The MV-field buffer is laid out as `[neighbour][block_y][block_x][2]`
/// of `i32` (2 components: dx, dy).
pub(crate) fn mv_field_byte_offset(mc: &MotionCtx, neighbour_idx: u32) -> u64 {
    let per_neighbour = (mc.blocks_x as u64) * (mc.blocks_y as u64) * 2;
    (neighbour_idx as u64) * per_neighbour * (size_of::<i32>() as u64)
}

/// Total number of `i32` slots needed for `n` neighbours' MV fields.
pub(crate) fn mv_field_total_i32(mc: &MotionCtx, neighbours: u32) -> usize {
    (neighbours as usize) * (mc.blocks_x as usize) * (mc.blocks_y as usize) * 2
}

/// Run analyse for one (centre, neighbour) pair: coarse pass on the
/// pyramid top level, fine refinement at full resolution. Writes the
/// resulting MV field into `mv_field` at the slot reserved for this
/// neighbour.
pub(crate) fn run_analyse<R: Runtime>(
    client: &ComputeClient<R>,
    mc: &MotionCtx,
    width: u32,
    height: u32,
    frame_count: u32,
    centre_slot: u32,
    neighbour_slot: u32,
    neighbour_idx: u32,
    pyramid: &Handle,
    mv_field: &Handle,
) -> Result<(), anyhow::Error> {
    let mv_offset = mv_field_byte_offset(mc, neighbour_idx);
    let mv_slot = mv_field.clone().offset_start(mv_offset);
    let mv_slot_len = (mc.blocks_x as usize) * (mc.blocks_y as usize) * 2;

    // Coarse pass on the top pyramid level (if pyramid_levels > 1).
    if mc.pyramid_levels > 1 {
        let coarse_level = mc.pyramid_levels - 1;
        let (cw, ch) = level_dims(width, height, coarse_level);
        let coarse_centre =
            pyramid
                .clone()
                .offset_start(pyramid_slot_byte_offset(width, height, frame_count, coarse_level, centre_slot));
        let coarse_neighbour =
            pyramid
                .clone()
                .offset_start(pyramid_slot_byte_offset(
                    width,
                    height,
                    frame_count,
                    coarse_level,
                    neighbour_slot,
                ));
        let level_len = (cw * ch) as usize;
        let coarse_scale = 1u32 << coarse_level;
        // Coarse blocks correspond to fine blocks downscaled by 2^coarse_level.
        let coarse_blksize = (mc.blksize / coarse_scale).max(2);
        let coarse_step = (mc.step / coarse_scale).max(1);
        let coarse_blocks_x = cw.div_ceil(coarse_step).max(1);
        let coarse_blocks_y = ch.div_ceil(coarse_step).max(1);
        let grid = CubeCount::new_2d(coarse_blocks_x, coarse_blocks_y);
        // One cube per block; pick a small cube dim that fits typical
        // coarse blocks (8x8). Threads collaborate on SAD reduction.
        let dim = CubeDim::new_2d(8, 8);

        unsafe {
            nlm_mc_block_match_coarse::launch_unchecked::<R>(
                client,
                grid,
                dim,
                ArrayArg::from_raw_parts(coarse_centre, level_len),
                ArrayArg::from_raw_parts(coarse_neighbour, level_len),
                ArrayArg::from_raw_parts(mv_slot.clone(), mv_slot_len),
                cw,
                ch,
                coarse_blksize,
                coarse_step,
                mc.search_radius,
                coarse_scale,
                mc.blocks_x,
                mc.blocks_y,
            );
        }
    } else {
        // Pyramid disabled: zero the MV field so the fine pass starts
        // from a (0, 0) seed.
        // We don't have a dedicated zero kernel for i32 here; the fine
        // pass kernel itself treats an out-of-band seed of (0, 0) when
        // pyramid_levels == 1.
    }

    // Fine pass at full resolution.
    let (fw, fh) = level_dims(width, height, 0);
    let fine_centre = pyramid.clone().offset_start(pyramid_slot_byte_offset(
        width, height, frame_count, 0, centre_slot,
    ));
    let fine_neighbour = pyramid.clone().offset_start(pyramid_slot_byte_offset(
        width, height, frame_count, 0, neighbour_slot,
    ));
    let level_len = (fw * fh) as usize;
    let grid = CubeCount::new_2d(mc.blocks_x, mc.blocks_y);
    let dim = CubeDim::new_2d(8, 8);
    let seeded = if mc.pyramid_levels > 1 { 1u32 } else { 0u32 };

    unsafe {
        nlm_mc_block_match_fine::launch_unchecked::<R>(
            client,
            grid,
            dim,
            ArrayArg::from_raw_parts(fine_centre, level_len),
            ArrayArg::from_raw_parts(fine_neighbour, level_len),
            ArrayArg::from_raw_parts(mv_slot, mv_slot_len),
            fw,
            fh,
            mc.blksize,
            mc.step,
            mc.search_radius,
            seeded,
            mc.blocks_x,
            mc.blocks_y,
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::MotionCompensationMode;

    fn mc(blksize: u32, overlap: u32) -> MotionCtx {
        MotionCtx::new(
            MotionCompensationMode::Mvtools {
                blksize,
                overlap,
                search_radius: 4,
                pyramid_levels: 2,
            },
            64,
            64,
        )
        .unwrap()
    }

    #[test]
    fn mv_field_offset_zero_for_first_neighbour() {
        assert_eq!(mv_field_byte_offset(&mc(16, 8), 0), 0);
    }

    #[test]
    fn mv_field_offset_advances_by_blocks() {
        let m = mc(16, 8);
        let per = (m.blocks_x as u64) * (m.blocks_y as u64) * 2 * 4;
        assert_eq!(mv_field_byte_offset(&m, 3), 3 * per);
    }

    #[test]
    fn mv_field_total_scales_with_neighbours() {
        let m = mc(16, 8);
        let per = (m.blocks_x as usize) * (m.blocks_y as usize) * 2;
        assert_eq!(mv_field_total_i32(&m, 4), 4 * per);
    }
}

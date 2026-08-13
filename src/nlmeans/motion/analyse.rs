use cubecl::prelude::*;
use cubecl::server::Handle;

use super::MotionCtx;
use super::pyramid::{level_dims, pyramid_slot_byte_offset};
#[cfg(test)]
use crate::nlmeans::align::StorageAlign;
use crate::nlmeans::kernels::motion::{nlm_mc_block_match_coarse, nlm_mc_block_match_fine};

/// Byte offset of the MV-field slice for a given neighbour index.
/// The MV-field buffer is laid out as `[neighbour][block_y][block_x][2]`
/// of `i32` (2 components: dx, dy), with each neighbour's slice padded
/// up to a 32-byte boundary (see [`MotionCtx::mv_field_bytes_per_neighbour`]).
pub(crate) fn mv_field_byte_offset(mc: &MotionCtx, neighbour_idx: u32) -> u64 {
    (neighbour_idx as u64) * mc.mv_field_bytes_per_neighbour()
}

/// Byte offset of the confidence slice for a given neighbour index.
/// The confidence buffer mirrors the MV field's layout,
/// `[neighbour][block_y][block_x]` of `f32`, but stores one scalar per
/// block instead of the MV field's two `i32` components, and each
/// neighbour's slice is padded up to a 32-byte boundary (see
/// [`MotionCtx::confidence_bytes_per_neighbour`]).
pub(crate) fn confidence_byte_offset(mc: &MotionCtx, neighbour_idx: u32) -> u64 {
    (neighbour_idx as u64) * mc.confidence_bytes_per_neighbour()
}

/// Run analyse for one (centre, neighbour) pair. Coarse pass on the
/// pyramid top level, fine refinement at full resolution. Writes the
/// resulting MV field into `mv_field` at the slot reserved for this
/// neighbour. `sad_noise_floor` and `thsad` are the fine kernel's
/// confidence scalars (see
/// [`crate::nlmeans::kernels::motion::nlm_mc_block_match_fine`]).
///
/// When `write_confidence` is `true`, also writes a per-block
/// confidence score into `confidence` at the matching slot. When
/// `false`, `confidence` is never indexed, so callers that don't
/// consume the confidence output can pass a small placeholder buffer
/// and skip computing `sad_noise_floor`/`thsad` (`0.0` for both is
/// fine in that case).
#[allow(clippy::too_many_arguments)]
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
    confidence: &Handle,
    write_confidence: bool,
    sad_noise_floor: f32,
    thsad: f32,
) -> Result<(), anyhow::Error> {
    let mv_offset = mv_field_byte_offset(mc, neighbour_idx);
    let mv_slot = mv_field.clone().offset_start(mv_offset);
    let mv_slot_len = (mc.blocks_x as usize) * (mc.blocks_y as usize) * 2;

    // Only slice into `confidence` at its real per-neighbour offset
    // when the kernel will actually write it. Otherwise `confidence`
    // is a small placeholder buffer with no per-neighbour layout to
    // offset into.
    let (conf_slot, conf_slot_len) = if write_confidence {
        let conf_offset = confidence_byte_offset(mc, neighbour_idx);
        (
            confidence.clone().offset_start(conf_offset),
            (mc.blocks_x as usize) * (mc.blocks_y as usize),
        )
    } else {
        (confidence.clone(), 1)
    };

    // Coarse pass on the top pyramid level (if pyramid_levels > 1).
    if mc.pyramid_levels > 1 {
        let coarse_level = mc.pyramid_levels - 1;
        let (cw, ch) = level_dims(width, height, coarse_level);
        let coarse_centre = pyramid.clone().offset_start(pyramid_slot_byte_offset(
            width,
            height,
            frame_count,
            coarse_level,
            centre_slot,
            mc.align,
        ));
        let coarse_neighbour = pyramid.clone().offset_start(pyramid_slot_byte_offset(
            width,
            height,
            frame_count,
            coarse_level,
            neighbour_slot,
            mc.align,
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
                mc.step,
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
        width,
        height,
        frame_count,
        0,
        centre_slot,
        mc.align,
    ));
    let fine_neighbour = pyramid.clone().offset_start(pyramid_slot_byte_offset(
        width,
        height,
        frame_count,
        0,
        neighbour_slot,
        mc.align,
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
            ArrayArg::from_raw_parts(conf_slot, conf_slot_len),
            write_confidence,
            sad_noise_floor,
            thsad,
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

/// Seeded refinement pass used by chained motion estimation. Reads the
/// composed seed already sitting in `mv_field` at `neighbour_idx`'s
/// slot, searches a small `(2·refine_radius + 1)²` window around it,
/// and writes the corrected MV back to the same slot. Skips the coarse
/// pass entirely, unlike [`run_analyse`], since the composed seed
/// already carries the coarse displacement.
///
/// `refine_radius` is the comptime search radius for this pass, set
/// independently of `mc.search_radius` (the direct path's window).
/// Every other argument matches `run_analyse`'s fine-pass call exactly,
/// including the confidence-write convention.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_seeded_refine<R: Runtime>(
    client: &ComputeClient<R>,
    mc: &MotionCtx,
    width: u32,
    height: u32,
    frame_count: u32,
    centre_slot: u32,
    neighbour_slot: u32,
    neighbour_idx: u32,
    refine_radius: u32,
    pyramid: &Handle,
    mv_field: &Handle,
    confidence: &Handle,
    write_confidence: bool,
    sad_noise_floor: f32,
    thsad: f32,
) -> Result<(), anyhow::Error> {
    let mv_offset = mv_field_byte_offset(mc, neighbour_idx);
    let mv_slot = mv_field.clone().offset_start(mv_offset);
    let mv_slot_len = (mc.blocks_x as usize) * (mc.blocks_y as usize) * 2;

    let (conf_slot, conf_slot_len) = if write_confidence {
        let conf_offset = confidence_byte_offset(mc, neighbour_idx);
        (
            confidence.clone().offset_start(conf_offset),
            (mc.blocks_x as usize) * (mc.blocks_y as usize),
        )
    } else {
        (confidence.clone(), 1)
    };

    let (fw, fh) = level_dims(width, height, 0);
    let fine_centre = pyramid.clone().offset_start(pyramid_slot_byte_offset(
        width,
        height,
        frame_count,
        0,
        centre_slot,
        mc.align,
    ));
    let fine_neighbour = pyramid.clone().offset_start(pyramid_slot_byte_offset(
        width,
        height,
        frame_count,
        0,
        neighbour_slot,
        mc.align,
    ));
    let level_len = (fw * fh) as usize;
    let grid = CubeCount::new_2d(mc.blocks_x, mc.blocks_y);
    let dim = CubeDim::new_2d(8, 8);

    unsafe {
        nlm_mc_block_match_fine::launch_unchecked::<R>(
            client,
            grid,
            dim,
            ArrayArg::from_raw_parts(fine_centre, level_len),
            ArrayArg::from_raw_parts(fine_neighbour, level_len),
            ArrayArg::from_raw_parts(mv_slot, mv_slot_len),
            ArrayArg::from_raw_parts(conf_slot, conf_slot_len),
            write_confidence,
            sad_noise_floor,
            thsad,
            fw,
            fh,
            mc.blksize,
            mc.step,
            refine_radius,
            1u32,
            mc.blocks_x,
            mc.blocks_y,
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nlmeans::motion::{MotionCompensationMode, MotionEstimation};

    /// The alignment the Vulkan adapters we test against report.
    fn align() -> StorageAlign {
        StorageAlign::new(32)
    }

    fn mc(blksize: u32, overlap: u32) -> MotionCtx {
        MotionCtx::new(
            MotionCompensationMode::Mvtools {
                blksize,
                overlap,
                search_radius: 4,
                pyramid_levels: 2,
                estimation: MotionEstimation::Direct,
            },
            64,
            64,
            align(),
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
    fn confidence_offset_zero_for_first_neighbour() {
        assert_eq!(confidence_byte_offset(&mc(16, 8), 0), 0);
    }

    #[test]
    fn confidence_offset_advances_by_blocks() {
        let m = mc(16, 8);
        let per = (m.blocks_x as u64) * (m.blocks_y as u64) * 4;
        assert_eq!(confidence_byte_offset(&m, 3), 3 * per);
    }

    #[test]
    fn confidence_offset_is_one_component_not_two() {
        // Confidence stores one f32 per block. The MV field stores two
        // i32 components per block. Both scalars are 4 bytes, so equal
        // block counts should give the confidence stride exactly half
        // the MV field's, as long as the unpadded stride is already a
        // multiple of 32 (true for this fixture's 64 blocks).
        let m = mc(16, 8);
        assert_eq!(mv_field_byte_offset(&m, 1), 2 * confidence_byte_offset(&m, 1));
    }

    #[test]
    fn confidence_offset_pads_small_block_counts_to_32_bytes() {
        // A tiny frame with this geometry has only 1 block
        // (4x4, blksize=4, overlap=0 => step=4 => 1x1 blocks), so the
        // unpadded stride (1 block * 4 bytes = 4 bytes) would place
        // neighbour 1 at a non-32-aligned offset. `wgpu` rejects a
        // bind-group offset that isn't a multiple of its
        // `min_storage_buffer_offset_alignment`, so the stride must be
        // padded up to 32 bytes regardless of block count.
        let m = MotionCtx::new(
            MotionCompensationMode::Mvtools {
                blksize: 4,
                overlap: 0,
                search_radius: 1,
                pyramid_levels: 1,
                estimation: MotionEstimation::Direct,
            },
            4,
            4,
            align(),
        )
        .unwrap();
        assert_eq!(
            m.blocks_x * m.blocks_y,
            1,
            "fixture should have exactly one block"
        );
        assert_eq!(confidence_byte_offset(&m, 0), 0);
        assert_eq!(confidence_byte_offset(&m, 1), 32);
        assert_eq!(confidence_byte_offset(&m, 2), 64);
    }

    #[test]
    fn mv_field_offset_pads_small_block_counts_to_32_bytes() {
        // Same fixture as `confidence_offset_pads_small_block_counts_to_32_bytes`:
        // one block, so the unpadded MV-field stride (1 block * 2 i32
        // components * 4 bytes = 8 bytes) would place neighbour 1 at a
        // non-32-aligned offset.
        let m = MotionCtx::new(
            MotionCompensationMode::Mvtools {
                blksize: 4,
                overlap: 0,
                search_radius: 1,
                pyramid_levels: 1,
                estimation: MotionEstimation::Direct,
            },
            4,
            4,
            align(),
        )
        .unwrap();
        assert_eq!(
            m.blocks_x * m.blocks_y,
            1,
            "fixture should have exactly one block"
        );
        assert_eq!(mv_field_byte_offset(&m, 0), 0);
        assert_eq!(mv_field_byte_offset(&m, 1), 32);
        assert_eq!(mv_field_byte_offset(&m, 2), 64);
    }

    #[test]
    fn mv_field_offset_pads_the_1080_square_odd_block_count_case() {
        // 1080x1080 at the library defaults (blksize=16, overlap=8, so
        // step=8) gives 135x135 = 18225 blocks, an odd count. The
        // unpadded stride (18225 blocks * 8 bytes = 145800) sits 8
        // bytes past the preceding 32-byte boundary (145792), so it
        // must round up to 145824 rather than leave neighbour 1
        // misaligned. 1920x1080 (the harness's usual frame size)
        // happens to land on an even block count at this geometry, so
        // it never exercises this rounding.
        let m = MotionCtx::new(MotionCompensationMode::mvtools_default(), 1080, 1080, align()).unwrap();
        assert_eq!(
            m.blocks_x * m.blocks_y,
            18225,
            "test premise: this geometry gives an odd block count"
        );
        assert_eq!(
            145_800u64 % 32,
            8,
            "test premise: the unpadded stride is not 32-aligned"
        );
        assert_eq!(mv_field_byte_offset(&m, 0), 0);
        assert_eq!(mv_field_byte_offset(&m, 1), 145_824);
    }
}

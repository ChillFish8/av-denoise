use cubecl::prelude::*;
use cubecl::server::Handle;

use super::MotionCtx;
use super::pyramid::{level_dims, pyramid_slot_byte_offset};
#[cfg(test)]
use crate::nlmeans::align::StorageAlign;
use crate::nlmeans::kernels::motion::{nlm_mc_block_match_coarse, nlm_mc_block_match_fine};

/// Where a neighbour's slice of the motion field starts.
///
/// The buffer is indexed by neighbour, then block, then component, with
/// two `i32` components per block. Each neighbour's slice is padded up
/// to an alignment boundary. See
/// [`MotionCtx::mv_field_bytes_per_neighbour`].
pub(crate) fn mv_field_byte_offset(mc: &MotionCtx, neighbour_idx: u32) -> u64 {
    (neighbour_idx as u64) * mc.mv_field_bytes_per_neighbour()
}

/// Where a neighbour's slice of the confidence buffer starts.
///
/// The layout mirrors the motion field, indexed by neighbour and then
/// block, but stores one `f32` per block rather than two `i32`
/// components.
///
/// Each neighbour's slice is padded up to an alignment boundary. See
/// [`MotionCtx::confidence_bytes_per_neighbour`].
pub(crate) fn confidence_byte_offset(mc: &MotionCtx, neighbour_idx: u32) -> u64 {
    (neighbour_idx as u64) * mc.confidence_bytes_per_neighbour()
}

/// Works out how one neighbour frame moved relative to the centre
/// frame.
///
/// A coarse pass runs on the smallest pyramid level, then a fine pass
/// refines its answer at full resolution. The result goes into
/// `mv_field` at the slot reserved for this neighbour.
///
/// `sad_noise_floor` and `thsad` are the fine kernel's confidence
/// scalars. See
/// [`crate::nlmeans::kernels::motion::nlm_mc_block_match_fine`].
///
/// When `write_confidence` is true, a per-block confidence score also
/// goes into `confidence` at the matching slot.
///
/// When it is false, `confidence` is never indexed. Callers that do not
/// need the score can pass a small placeholder buffer and leave both
/// scalars at 0.0.
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
    // when the kernel is going to write it. Otherwise `confidence` is a
    // small placeholder buffer with no per-neighbour layout to offset
    // into.
    let (conf_slot, conf_slot_len) = if write_confidence {
        let conf_offset = confidence_byte_offset(mc, neighbour_idx);
        (
            confidence.clone().offset_start(conf_offset),
            (mc.blocks_x as usize) * (mc.blocks_y as usize),
        )
    } else {
        (confidence.clone(), 1)
    };

    // The coarse pass, which only runs with more than one pyramid level.
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
        // A coarse block covers the same content as a fine block scaled
        // down by 2 raised to the coarse level.
        let coarse_blksize = (mc.blksize / coarse_scale).max(2);
        let coarse_step = (mc.step / coarse_scale).max(1);
        let coarse_blocks_x = cw.div_ceil(coarse_step).max(1);
        let coarse_blocks_y = ch.div_ceil(coarse_step).max(1);
        let grid = CubeCount::new_2d(coarse_blocks_x, coarse_blocks_y);
        // One block of threads per image block, sized to suit the 8x8
        // blocks a coarse level typically has. Those threads share the
        // scoring work between them.
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
        // With the pyramid disabled the fine pass has to start from a
        // zero seed. There is no dedicated zeroing kernel for `i32`
        // here, because the fine pass treats a missing seed as zero
        // when there is only one pyramid level.
    }

    // The fine pass, which runs at full resolution.
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

/// Cleans up the seed that chained motion estimation produced.
///
/// The joined seed already sits in `mv_field` at this neighbour's slot.
/// This searches a small window around it and writes the corrected
/// vector back to the same place.
///
/// Unlike [`run_analyse`] there is no coarse pass, because the joined
/// seed already carries the large movement.
///
/// `refine_radius` is this pass's own search radius, set independently
/// of the direct path's `mc.search_radius`. Every other argument matches
/// the fine-pass call in `run_analyse`, including how confidence is
/// written.
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

    /// The alignment the Vulkan adapters these tests run on report.
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
        // Confidence stores one `f32` per block and the motion field
        // stores two `i32` components. Both are 4 bytes, so at the same
        // block count the confidence stride should be exactly half the
        // motion field's, as long as the unpadded stride already lands
        // on a 32-byte boundary. This fixture's 64 blocks do.
        let m = mc(16, 8);
        assert_eq!(mv_field_byte_offset(&m, 1), 2 * confidence_byte_offset(&m, 1));
    }

    #[test]
    fn confidence_offset_pads_small_block_counts_to_32_bytes() {
        // A 4x4 frame at this geometry has a single block, so the
        // unpadded stride is only 4 bytes and would leave neighbour 1
        // at an offset that is not 32-aligned.
        //
        // wgpu rejects a bind-group offset that is not a multiple of
        // its `min_storage_buffer_offset_alignment`, so the stride has
        // to pad up to 32 bytes whatever the block count.
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
        // The same fixture as
        // `confidence_offset_pads_small_block_counts_to_32_bytes`, with
        // one block. The unpadded motion-field stride is 8 bytes, which
        // would leave neighbour 1 at an offset that is not 32-aligned.
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
        // A 1080x1080 frame at the library defaults gives 135x135
        // blocks, an odd count of 18,225. The unpadded stride of
        // 145,800 bytes sits 8 past the preceding 32-byte boundary at
        // 145,792, so it has to round up to 145,824 rather than leave
        // neighbour 1 misaligned.
        //
        // The harness's usual 1920x1080 happens to land on an even
        // block count at this geometry, so it never reaches this case.
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

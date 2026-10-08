use cubecl::prelude::*;
use cubecl::server::Handle;

use super::MotionCtx;
use super::pyramid::{level_dims, pyramid_slot_byte_offset};
#[cfg(test)]
use crate::nlmeans::align::StorageAlign;
use crate::nlmeans::kernels::motion::{
    BLOCK_MATCH_THREADS,
    nlm_mc_block_match_coarse,
    nlm_mc_block_match_fine,
};

/// Where a neighbour's motion-field slice starts.
///
/// The field is indexed by neighbour, block and then component, with two `i32` per block. Each
/// slice is padded as [MotionCtx::mv_field_bytes_per_neighbour] describes.
pub(crate) fn mv_field_byte_offset(motion_ctx: &MotionCtx, neighbour_idx: u32) -> u64 {
    (neighbour_idx as u64) * motion_ctx.mv_field_bytes_per_neighbour()
}

/// Where a neighbour's confidence slice starts.
///
/// It mirrors the motion field's layout with one `f32` per block, padded as
/// [MotionCtx::confidence_bytes_per_neighbour] describes.
pub(crate) fn confidence_byte_offset(motion_ctx: &MotionCtx, neighbour_idx: u32) -> u64 {
    (neighbour_idx as u64) * motion_ctx.confidence_bytes_per_neighbour()
}

/// Estimates how one neighbour frame moved relative to the centre frame.
///
/// A coarse pass on the smallest pyramid level seeds a fine pass at full resolution, which writes
/// this neighbour's `mv_field` slot. With `write_confidence` set, a per-block score also lands in
/// its `confidence` slot. Otherwise `confidence` is never indexed, so a placeholder buffer works
/// and `sad_noise_floor` and `thsad` may stay at 0.0.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
pub(crate) fn run_analyse<R: Runtime>(
    client: &ComputeClient<R>,
    motion_ctx: &MotionCtx,
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
    let mv_offset = mv_field_byte_offset(motion_ctx, neighbour_idx);
    let mv_slot = mv_field.clone().offset_start(mv_offset);
    let mv_slot_len = (motion_ctx.blocks_x as usize) * (motion_ctx.blocks_y as usize) * 2;

    // A placeholder `confidence` has no per-neighbour layout to offset into.
    let (conf_slot, conf_slot_len) = if write_confidence {
        let conf_offset = confidence_byte_offset(motion_ctx, neighbour_idx);
        (
            confidence.clone().offset_start(conf_offset),
            (motion_ctx.blocks_x as usize) * (motion_ctx.blocks_y as usize),
        )
    } else {
        (confidence.clone(), 1)
    };

    if motion_ctx.pyramid_levels > 1 {
        let coarse_level = motion_ctx.pyramid_levels - 1;
        let (coarse_width, coarse_height) = level_dims(width, height, coarse_level);
        let coarse_centre_offset = pyramid_slot_byte_offset(
            width,
            height,
            frame_count,
            coarse_level,
            centre_slot,
            motion_ctx.align,
        );
        let coarse_centre = pyramid.clone().offset_start(coarse_centre_offset);
        let coarse_neighbour_offset = pyramid_slot_byte_offset(
            width,
            height,
            frame_count,
            coarse_level,
            neighbour_slot,
            motion_ctx.align,
        );
        let coarse_neighbour = pyramid.clone().offset_start(coarse_neighbour_offset);
        let level_len = (coarse_width * coarse_height) as usize;
        let coarse_scale = 1u32 << coarse_level;
        // A coarse block covers the same content as a fine block scaled down by `coarse_scale`.
        let coarse_blksize = (motion_ctx.blksize / coarse_scale).max(2);
        let coarse_step = (motion_ctx.step / coarse_scale).max(1);
        let coarse_blocks_x = coarse_width.div_ceil(coarse_step).max(1);
        let coarse_blocks_y = coarse_height.div_ceil(coarse_step).max(1);
        let grid = CubeCount::new_2d(coarse_blocks_x, coarse_blocks_y);
        // One cube per image block, with its threads sharing the scoring work.
        let dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);

        unsafe {
            nlm_mc_block_match_coarse::launch_unchecked::<R>(
                client,
                grid,
                dim,
                ArrayArg::from_raw_parts(coarse_centre, level_len),
                ArrayArg::from_raw_parts(coarse_neighbour, level_len),
                ArrayArg::from_raw_parts(mv_slot.clone(), mv_slot_len),
                coarse_width,
                coarse_height,
                coarse_blksize,
                coarse_step,
                motion_ctx.search_radius,
                coarse_scale,
                motion_ctx.blocks_x,
                motion_ctx.blocks_y,
                motion_ctx.step,
            );
        }
    } else {
        // A single level has no coarse seed, and the fine pass then treats the seed as zero.
    }

    let (fine_width, fine_height) = level_dims(width, height, 0);
    let fine_centre_offset =
        pyramid_slot_byte_offset(width, height, frame_count, 0, centre_slot, motion_ctx.align);
    let fine_centre = pyramid.clone().offset_start(fine_centre_offset);
    let fine_neighbour_offset =
        pyramid_slot_byte_offset(width, height, frame_count, 0, neighbour_slot, motion_ctx.align);
    let fine_neighbour = pyramid.clone().offset_start(fine_neighbour_offset);
    let level_len = (fine_width * fine_height) as usize;
    let grid = CubeCount::new_2d(motion_ctx.blocks_x, motion_ctx.blocks_y);
    let dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);
    let seeded = if motion_ctx.pyramid_levels > 1 { 1u32 } else { 0u32 };

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
            fine_width,
            fine_height,
            motion_ctx.blksize,
            motion_ctx.step,
            motion_ctx.search_radius,
            seeded,
            motion_ctx.blocks_x,
        );
    }

    Ok(())
}

/// Refines the chained seed already sitting in this neighbour's `mv_field` slot.
///
/// It searches `refine_radius` around the seed and writes the corrected vector back. There is no
/// coarse pass, because the joined seed already carries the large movement. Confidence is written
/// the same way as in [run_analyse].
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
pub(crate) fn run_seeded_refine<R: Runtime>(
    client: &ComputeClient<R>,
    motion_ctx: &MotionCtx,
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
    let mv_offset = mv_field_byte_offset(motion_ctx, neighbour_idx);
    let mv_slot = mv_field.clone().offset_start(mv_offset);
    let mv_slot_len = (motion_ctx.blocks_x as usize) * (motion_ctx.blocks_y as usize) * 2;

    let (conf_slot, conf_slot_len) = if write_confidence {
        let conf_offset = confidence_byte_offset(motion_ctx, neighbour_idx);
        (
            confidence.clone().offset_start(conf_offset),
            (motion_ctx.blocks_x as usize) * (motion_ctx.blocks_y as usize),
        )
    } else {
        (confidence.clone(), 1)
    };

    let (fine_width, fine_height) = level_dims(width, height, 0);
    let fine_centre_offset =
        pyramid_slot_byte_offset(width, height, frame_count, 0, centre_slot, motion_ctx.align);
    let fine_centre = pyramid.clone().offset_start(fine_centre_offset);
    let fine_neighbour_offset =
        pyramid_slot_byte_offset(width, height, frame_count, 0, neighbour_slot, motion_ctx.align);
    let fine_neighbour = pyramid.clone().offset_start(fine_neighbour_offset);
    let level_len = (fine_width * fine_height) as usize;
    let grid = CubeCount::new_2d(motion_ctx.blocks_x, motion_ctx.blocks_y);
    let dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);

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
            fine_width,
            fine_height,
            motion_ctx.blksize,
            motion_ctx.step,
            refine_radius,
            1u32,
            motion_ctx.blocks_x,
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

    fn motion_ctx(blksize: u32, overlap: u32) -> MotionCtx {
        let mode = MotionCompensationMode::Mvtools {
            blksize,
            overlap,
            search_radius: 4,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        };
        let align = align();

        MotionCtx::new(mode, 64, 64, align).unwrap()
    }

    /// A 4x4 frame at a geometry that leaves exactly one block.
    fn single_block_ctx() -> MotionCtx {
        let mode = MotionCompensationMode::Mvtools {
            blksize: 4,
            overlap: 0,
            search_radius: 1,
            pyramid_levels: 1,
            estimation: MotionEstimation::Direct,
        };
        let align = align();

        MotionCtx::new(mode, 4, 4, align).unwrap()
    }

    #[test]
    fn mv_field_offset_zero_for_first_neighbour() {
        let ctx = motion_ctx(16, 8);
        let offset = mv_field_byte_offset(&ctx, 0);

        assert_eq!(offset, 0);
    }

    #[test]
    fn mv_field_offset_advances_by_blocks() {
        let ctx = motion_ctx(16, 8);
        let stride = (ctx.blocks_x as u64) * (ctx.blocks_y as u64) * 2 * 4;
        let offset = mv_field_byte_offset(&ctx, 3);

        assert_eq!(offset, 3 * stride);
    }

    #[test]
    fn confidence_offset_zero_for_first_neighbour() {
        let ctx = motion_ctx(16, 8);
        let offset = confidence_byte_offset(&ctx, 0);

        assert_eq!(offset, 0);
    }

    #[test]
    fn confidence_offset_advances_by_blocks() {
        let ctx = motion_ctx(16, 8);
        let stride = (ctx.blocks_x as u64) * (ctx.blocks_y as u64) * 4;
        let offset = confidence_byte_offset(&ctx, 3);

        assert_eq!(offset, 3 * stride);
    }

    #[test]
    fn confidence_offset_is_one_component_not_two() {
        // Both strides are 4 bytes per component, so confidence is exactly half the motion field
        // while the unpadded stride is 32-byte aligned, which this fixture's 64 blocks are.
        let ctx = motion_ctx(16, 8);
        let mv_offset = mv_field_byte_offset(&ctx, 1);
        let confidence_offset = confidence_byte_offset(&ctx, 1);

        assert_eq!(mv_offset, 2 * confidence_offset);
    }

    #[test]
    fn confidence_offset_pads_small_block_counts_to_32_bytes() {
        // One block gives an unpadded stride of 4 bytes, which would leave neighbour 1 off a
        // 32-byte boundary.
        let ctx = single_block_ctx();
        assert_eq!(
            ctx.blocks_x * ctx.blocks_y,
            1,
            "fixture should have exactly one block"
        );

        let first = confidence_byte_offset(&ctx, 0);
        let second = confidence_byte_offset(&ctx, 1);
        let third = confidence_byte_offset(&ctx, 2);

        assert_eq!(first, 0);
        assert_eq!(second, 32);
        assert_eq!(third, 64);
    }

    #[test]
    fn mv_field_offset_pads_small_block_counts_to_32_bytes() {
        // One block gives an unpadded stride of 8 bytes, which would leave neighbour 1 off a
        // 32-byte boundary.
        let ctx = single_block_ctx();
        assert_eq!(
            ctx.blocks_x * ctx.blocks_y,
            1,
            "fixture should have exactly one block"
        );

        let first = mv_field_byte_offset(&ctx, 0);
        let second = mv_field_byte_offset(&ctx, 1);
        let third = mv_field_byte_offset(&ctx, 2);

        assert_eq!(first, 0);
        assert_eq!(second, 32);
        assert_eq!(third, 64);
    }

    #[test]
    fn mv_field_offset_pads_the_1080_square_odd_block_count_case() {
        // 1080x1080 at the defaults gives 135x135 blocks. The unpadded stride of 145,800 bytes sits
        // 8 past a 32-byte boundary, so it must round up to 145,824.
        let mode = MotionCompensationMode::mvtools_default();
        let align = align();
        let ctx = MotionCtx::new(mode, 1080, 1080, align).unwrap();
        assert_eq!(
            ctx.blocks_x * ctx.blocks_y,
            18225,
            "test premise: this geometry gives an odd block count"
        );
        assert_eq!(
            145_800u64 % 32,
            8,
            "test premise: the unpadded stride is not 32-aligned"
        );

        let first = mv_field_byte_offset(&ctx, 0);
        let second = mv_field_byte_offset(&ctx, 1);

        assert_eq!(first, 0);
        assert_eq!(second, 145_824);
    }
}

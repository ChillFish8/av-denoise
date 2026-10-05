use cubecl::prelude::*;
use cubecl::server::Handle;

use super::MotionCtx;
use crate::nlmeans::align::StorageAlign;
use crate::nlmeans::kernels::motion::{nlm_mc_downscale, nlm_mc_extract_luma};

/// One frame's luma pixel count at `level`, padded to whole alignment boundaries.
///
/// Slot offsets are sums of whole strides, so padding the stride keeps every offset aligned. wgpu
/// rejects a bind-group offset that is not a multiple of its `min_storage_buffer_offset_alignment`,
/// which an unpadded level such as 180x137 would break on every odd slot. Kernels only read a
/// slot's leading `width * height` pixels, so the padding is never touched.
fn level_slot_pixels(width: u32, height: u32, level: u32, align: StorageAlign) -> usize {
    let (level_width, level_height) = level_dims(width, height, level);
    let level_pixels = (level_width as usize) * (level_height as usize);
    align.pad_elems::<f32>(level_pixels)
}

/// One frame's luma pixel count across every pyramid level, padded the way
/// [pyramid_slot_byte_offset] addresses it.
pub fn pyramid_pixels_per_frame(width: u32, height: u32, levels: u32, align: StorageAlign) -> usize {
    (0..levels)
        .map(|level| level_slot_pixels(width, height, level, align))
        .sum()
}

/// Where a level and frame slot starts in the flat pyramid buffer, always on an alignment boundary.
pub fn pyramid_slot_byte_offset(
    width: u32,
    height: u32,
    frame_count: u32,
    level: u32,
    frame: u32,
    align: StorageAlign,
) -> u64 {
    let mut offset_pixels: usize = 0;
    for lower_level in 0..level {
        offset_pixels += (frame_count as usize) * level_slot_pixels(width, height, lower_level, align);
    }

    offset_pixels += (frame as usize) * level_slot_pixels(width, height, level, align);
    (offset_pixels * size_of::<f32>()) as u64
}

/// The pixel dimensions at `level`, where level 0 is full resolution.
pub fn level_dims(width: u32, height: u32, level: u32) -> (u32, u32) {
    let mut level_width = width;
    let mut level_height = height;
    for _ in 0..level {
        level_width = (level_width / 2).max(1);
        level_height = (level_height / 2).max(1);
    }

    (level_width, level_height)
}

/// Builds every pyramid level for the slot just uploaded.
///
/// Level 0 is the luma plane alone, and each further level is the one before averaged 2x2 at half
/// size.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
pub(crate) fn run_pyramid_build<R: Runtime>(
    client: &ComputeClient<R>,
    motion_ctx: &MotionCtx,
    width: u32,
    height: u32,
    frame_count: u32,
    slot: u32,
    full_res: &Handle,
    pyramid: &Handle,
    stored_ch: u32,
) -> Result<(), anyhow::Error> {
    extract_luma::<R>(
        client,
        full_res,
        pyramid,
        slot,
        width,
        height,
        frame_count,
        stored_ch,
        motion_ctx.align,
    );

    for level in 1..motion_ctx.pyramid_levels {
        downscale_level::<R>(
            client,
            pyramid,
            slot,
            width,
            height,
            frame_count,
            level,
            motion_ctx.align,
        );
    }

    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
fn extract_luma<R: Runtime>(
    client: &ComputeClient<R>,
    full_res: &Handle,
    pyramid: &Handle,
    slot: u32,
    width: u32,
    height: u32,
    frame_count: u32,
    stored_ch: u32,
    align: StorageAlign,
) {
    let block_x = 16u32;
    let block_y = 16u32;
    let cubes_x = width.div_ceil(block_x);
    let cubes_y = height.div_ceil(block_y);
    let grid = CubeCount::new_2d(cubes_x, cubes_y);
    let dim = CubeDim::new_2d(block_x, block_y);
    let full_len = (frame_count * height * width * stored_ch) as usize;
    let level0_offset = pyramid_slot_byte_offset(width, height, frame_count, 0, slot, align);
    let level0_dst = pyramid.clone().offset_start(level0_offset);
    let level0_len = (frame_count * height * width) as usize;

    unsafe {
        nlm_mc_extract_luma::launch_unchecked::<R>(
            client,
            grid,
            dim,
            stored_ch as usize,
            ArrayArg::from_raw_parts(full_res.clone(), full_len),
            ArrayArg::from_raw_parts(level0_dst, level0_len),
            slot,
            0u32,
            width,
            height,
        );
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
fn downscale_level<R: Runtime>(
    client: &ComputeClient<R>,
    pyramid: &Handle,
    slot: u32,
    width: u32,
    height: u32,
    frame_count: u32,
    level: u32,
    align: StorageAlign,
) {
    let (src_w, src_h) = level_dims(width, height, level - 1);
    let (dst_w, dst_h) = level_dims(width, height, level);
    let block_x = 16u32;
    let block_y = 16u32;
    let cubes_x = dst_w.div_ceil(block_x);
    let cubes_y = dst_h.div_ceil(block_y);
    let grid = CubeCount::new_2d(cubes_x, cubes_y);
    let dim = CubeDim::new_2d(block_x, block_y);

    let src_offset = pyramid_slot_byte_offset(width, height, frame_count, level - 1, slot, align);
    let src = pyramid.clone().offset_start(src_offset);
    let dst_offset = pyramid_slot_byte_offset(width, height, frame_count, level, slot, align);
    let dst = pyramid.clone().offset_start(dst_offset);
    let src_len = (src_w * src_h) as usize;
    let dst_len = (dst_w * dst_h) as usize;

    unsafe {
        nlm_mc_downscale::launch_unchecked::<R>(
            client,
            grid,
            dim,
            ArrayArg::from_raw_parts(src, src_len),
            ArrayArg::from_raw_parts(dst, dst_len),
            0u32,
            0u32,
            src_w,
            src_h,
            dst_w,
            dst_h,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nlmeans::motion::MAX_PYRAMID_LEVELS;

    /// The alignment the Vulkan adapters these tests run on report.
    fn align() -> StorageAlign {
        StorageAlign::new(32)
    }

    #[test]
    fn pyramid_pixels_single_level_matches_image() {
        let align = align();
        let pixels = pyramid_pixels_per_frame(64, 32, 1, align);
        assert_eq!(pixels, 64 * 32);
    }

    #[test]
    fn pyramid_pixels_two_levels_sums_levels() {
        // Level 0 is 64x32, so 2048 pixels, and level 1 is 32x16, so 512.
        let align = align();
        let pixels = pyramid_pixels_per_frame(64, 32, 2, align);
        assert_eq!(pixels, 2048 + 512);
    }

    #[test]
    fn level_dims_halve() {
        let full = level_dims(64, 32, 0);
        let half = level_dims(64, 32, 1);
        let quarter = level_dims(64, 32, 2);

        assert_eq!(full, (64, 32));
        assert_eq!(half, (32, 16));
        assert_eq!(quarter, (16, 8));
    }

    #[test]
    fn slot_byte_offsets_respect_every_alignment_a_runtime_can_report() {
        // Backends report alignments between 4 and 256 bytes. Each size has a level whose unpadded
        // stride falls short, such as 180x137, the half-size level of a 720x548 frame's chroma
        // plane, at 16 bytes short of a 32-byte boundary.
        for bytes in [4u64, 16, 32, 64, 256] {
            let align = StorageAlign::new(bytes);
            for (width, height) in [(360, 274), (720, 548), (722, 546), (66, 66), (42, 28)] {
                for level in 0..MAX_PYRAMID_LEVELS {
                    for frame in 0..5 {
                        let offset = pyramid_slot_byte_offset(width, height, 5, level, frame, align);
                        assert_eq!(
                            offset % bytes,
                            0,
                            "align {bytes}: {width}x{height} level={level} frame={frame} lands at byte {offset}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn pixels_per_frame_covers_the_last_slot_of_every_level() {
        let (width, height, frames, levels) = (360u32, 274u32, 5u32, 3u32);

        for bytes in [4u64, 16, 32, 64, 256] {
            let align = StorageAlign::new(bytes);
            let total_bytes =
                pyramid_pixels_per_frame(width, height, levels, align) * frames as usize * size_of::<f32>();

            for level in 0..levels {
                let (level_width, level_height) = level_dims(width, height, level);
                let last = pyramid_slot_byte_offset(width, height, frames, level, frames - 1, align) as usize;
                let end = last + (level_width * level_height) as usize * size_of::<f32>();
                assert!(
                    end <= total_bytes,
                    "align {bytes}: level {level} slot {} ends at {end}, past the {total_bytes}-byte buffer",
                    frames - 1
                );
            }
        }
    }

    #[test]
    fn slot_byte_offset_advances_past_full_levels() {
        // Level 1, frame 2 skips all of level 0 at 8192 pixels, then two level 1 frames at 512 each.
        let align = align();
        let bytes = pyramid_slot_byte_offset(64, 32, 4, 1, 2, align);
        assert_eq!(bytes as usize, (8192 + 1024) * 4);
    }
}

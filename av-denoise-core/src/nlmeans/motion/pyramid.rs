use cubecl::prelude::*;
use cubecl::server::Handle;

use super::MotionCtx;
use crate::nlmeans::align::StorageAlign;
use crate::nlmeans::kernels::motion::{nlm_mc_downscale, nlm_mc_extract_luma};

/// How many luma pixels one frame takes up at `level`, padded up to a
/// whole number of alignment boundaries.
///
/// Slot offsets are sums of whole slot strides, so padding the stride is
/// what keeps every offset aligned.
///
/// wgpu rejects a bind-group offset that is not a multiple of its
/// `min_storage_buffer_offset_alignment`. A level whose pixel count does
/// not fill whole boundaries, such as a 180x137 chroma level, would
/// otherwise leave every odd slot short of one.
///
/// Kernels only ever read a slot's leading `width * height` pixels, so
/// the padding is never touched.
fn level_slot_pixels(width: u32, height: u32, level: u32, align: StorageAlign) -> usize {
    let (w, h) = level_dims(width, height, level);
    align.pad_elems::<f32>((w as usize) * (h as usize))
}

/// How many luma pixels each frame takes up across every pyramid level.
///
/// Level 0 contributes the full pixel count, and each level after that
/// halves both axes.
///
/// Every level's contribution is padded to the alignment, which matches
/// the layout [`pyramid_slot_byte_offset`] addresses.
pub fn pyramid_pixels_per_frame(width: u32, height: u32, levels: u32, align: StorageAlign) -> usize {
    (0..levels)
        .map(|level| level_slot_pixels(width, height, level, align))
        .sum()
}

/// Where a given level and frame slot starts inside the flat pyramid
/// buffer.
///
/// The result is always a multiple of the alignment. See
/// [`level_slot_pixels`].
pub fn pyramid_slot_byte_offset(
    width: u32,
    height: u32,
    frame_count: u32,
    level: u32,
    frame: u32,
    align: StorageAlign,
) -> u64 {
    let mut offset_pixels: usize = 0;
    for l in 0..level {
        offset_pixels += (frame_count as usize) * level_slot_pixels(width, height, l, align);
    }
    offset_pixels += (frame as usize) * level_slot_pixels(width, height, level, align);
    (offset_pixels * size_of::<f32>()) as u64
}

/// The pixel dimensions at `level`, where level 0 is full resolution.
pub fn level_dims(width: u32, height: u32, level: u32) -> (u32, u32) {
    let mut w = width;
    let mut h = height;
    for _ in 0..level {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    (w, h)
}

/// Builds every pyramid level for the slot that was just uploaded,
/// starting from the packed full-resolution input.
///
/// Level 0 is the luma plane on its own. Each level after that is the
/// one before it at half size, averaged 2x2.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
pub(crate) fn run_pyramid_build<R: Runtime>(
    client: &ComputeClient<R>,
    mc: &MotionCtx,
    width: u32,
    height: u32,
    frame_count: u32,
    slot: u32,
    full_res: &Handle,
    pyramid: &Handle,
    stored_ch: u32,
) -> Result<(), anyhow::Error> {
    let _ = mc;
    extract_luma::<R>(
        client,
        full_res,
        pyramid,
        slot,
        width,
        height,
        frame_count,
        stored_ch,
        mc.align,
    );
    for level in 1..mc.pyramid_levels {
        downscale_level::<R>(client, pyramid, slot, width, height, frame_count, level, mc.align);
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
    let grid = CubeCount::new_2d(width.div_ceil(block_x), height.div_ceil(block_y));
    let dim = CubeDim::new_2d(block_x, block_y);
    let full_len = (frame_count * height * width * stored_ch) as usize;
    let level0_dst = pyramid.clone().offset_start(pyramid_slot_byte_offset(
        width,
        height,
        frame_count,
        0,
        slot,
        align,
    ));
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
    let grid = CubeCount::new_2d(dst_w.div_ceil(block_x), dst_h.div_ceil(block_y));
    let dim = CubeDim::new_2d(block_x, block_y);

    let src = pyramid.clone().offset_start(pyramid_slot_byte_offset(
        width,
        height,
        frame_count,
        level - 1,
        slot,
        align,
    ));
    let dst = pyramid.clone().offset_start(pyramid_slot_byte_offset(
        width,
        height,
        frame_count,
        level,
        slot,
        align,
    ));
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

    /// The alignment the Vulkan adapters these tests run on report.
    fn align() -> StorageAlign {
        StorageAlign::new(32)
    }

    #[test]
    fn pyramid_pixels_single_level_matches_image() {
        assert_eq!(pyramid_pixels_per_frame(64, 32, 1, align()), 64 * 32);
    }

    #[test]
    fn pyramid_pixels_two_levels_sums_levels() {
        // Level 0 is 64x32, so 2048 pixels. Level 1 is 32x16, so 512.
        // That gives 2560 in total.
        assert_eq!(pyramid_pixels_per_frame(64, 32, 2, align()), 2048 + 512);
    }

    #[test]
    fn level_dims_halve() {
        assert_eq!(level_dims(64, 32, 0), (64, 32));
        assert_eq!(level_dims(64, 32, 1), (32, 16));
        assert_eq!(level_dims(64, 32, 2), (16, 8));
    }

    #[test]
    fn slot_byte_offsets_respect_every_alignment_a_runtime_can_report() {
        // A GPU rejects a bind-group offset that is not a multiple of
        // `min_storage_buffer_offset_alignment`, which backends report
        // anywhere from 4 to 256 bytes.
        //
        // Every dimension pair here has at least one level whose
        // unpadded slot stride falls short. 360x274 is the chroma plane
        // of a 720x548 frame, and its half-size level of 180x137 comes
        // to 98,640 bytes, 16 short of a 32-byte boundary.
        for bytes in [4u64, 16, 32, 64, 256] {
            let align = StorageAlign::new(bytes);
            for (w, h) in [(360, 274), (720, 548), (722, 546), (66, 66), (42, 28)] {
                for level in 0..super::super::MAX_PYRAMID_LEVELS {
                    for frame in 0..5 {
                        let offset = pyramid_slot_byte_offset(w, h, 5, level, frame, align);
                        assert_eq!(
                            offset % bytes,
                            0,
                            "align {bytes}: {w}x{h} level={level} frame={frame} lands at byte {offset}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn pixels_per_frame_covers_the_last_slot_of_every_level() {
        // The allocation `pyramid_pixels_per_frame` sizes has to hold
        // every slot `pyramid_slot_byte_offset` addresses, padding
        // included, whatever alignment the runtime reports.
        let (w, h, frames, levels) = (360u32, 274u32, 5u32, 3u32);

        for bytes in [4u64, 16, 32, 64, 256] {
            let align = StorageAlign::new(bytes);
            let total_bytes =
                pyramid_pixels_per_frame(w, h, levels, align) * frames as usize * size_of::<f32>();

            for level in 0..levels {
                let (lw, lh) = level_dims(w, h, level);
                let last = pyramid_slot_byte_offset(w, h, frames, level, frames - 1, align) as usize;
                let end = last + (lw * lh) as usize * size_of::<f32>();
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
        // 4 frames of a 64x32 image across 2 levels. Reaching level 1,
        // frame 2 means skipping all of level 0, which is 8192 pixels,
        // then two frames of level 1 at 512 pixels each, so 1024 more.
        let bytes = pyramid_slot_byte_offset(64, 32, 4, 1, 2, align());
        assert_eq!(bytes as usize, (8192 + 1024) * 4);
    }
}

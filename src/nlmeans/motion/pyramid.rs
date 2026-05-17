//! Pyramid-build dispatch for the motion-estimation hierarchy. Lives
//! alongside `analyse` and `compensate` so each step's Rust-side glue
//! is colocated with its kernel-launch pattern.

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::super::kernels::motion::{nlm_mc_downscale, nlm_mc_extract_luma};
use super::MotionCtx;

/// Number of luma pixels stored per frame across every pyramid level
/// for an image of `(width, height)`. Level 0 contributes `w*h`; each
/// subsequent level halves both axes.
pub fn pyramid_pixels_per_frame(width: u32, height: u32, levels: u32) -> usize {
    let mut total: usize = 0;
    let mut w = width;
    let mut h = height;
    for _ in 0..levels {
        total += (w as usize) * (h as usize);
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    total
}

/// Byte offset of a given `(level, frame)` slot inside the flat pyramid
/// buffer.
pub fn pyramid_slot_byte_offset(
    width: u32,
    height: u32,
    frame_count: u32,
    level: u32,
    frame: u32,
) -> u64 {
    let mut offset_pixels: u64 = 0;
    let mut w = width as u64;
    let mut h = height as u64;
    for l in 0..level {
        offset_pixels += (frame_count as u64) * w * h;
        w = (w / 2).max(1);
        h = (h / 2).max(1);
        let _ = l;
    }
    offset_pixels += (frame as u64) * w * h;
    offset_pixels * (size_of::<f32>() as u64)
}

/// Pixel dimensions at `level` (level 0 = full res).
pub fn level_dims(width: u32, height: u32, level: u32) -> (u32, u32) {
    let mut w = width;
    let mut h = height;
    for _ in 0..level {
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    (w, h)
}

/// Build every pyramid level for the freshly-uploaded slot, starting
/// from the packed full-resolution input. Level 0 is the extracted
/// luma plane; each subsequent level is a 2x box downsample of the one
/// before it.
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
    extract_luma::<R>(client, full_res, pyramid, slot, width, height, frame_count, stored_ch);
    for level in 1..mc.pyramid_levels {
        downscale_level::<R>(client, pyramid, slot, width, height, frame_count, level);
    }
    Ok(())
}

fn extract_luma<R: Runtime>(
    client: &ComputeClient<R>,
    full_res: &Handle,
    pyramid: &Handle,
    slot: u32,
    width: u32,
    height: u32,
    frame_count: u32,
    stored_ch: u32,
) {
    let block_x = 16u32;
    let block_y = 16u32;
    let grid = CubeCount::new_2d(width.div_ceil(block_x), height.div_ceil(block_y));
    let dim = CubeDim::new_2d(block_x, block_y);
    let full_len = (frame_count * height * width * stored_ch) as usize;
    let level0_dst = pyramid.clone().offset_start(pyramid_slot_byte_offset(
        width, height, frame_count, 0, slot,
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

fn downscale_level<R: Runtime>(
    client: &ComputeClient<R>,
    pyramid: &Handle,
    slot: u32,
    width: u32,
    height: u32,
    frame_count: u32,
    level: u32,
) {
    let (src_w, src_h) = level_dims(width, height, level - 1);
    let (dst_w, dst_h) = level_dims(width, height, level);
    let block_x = 16u32;
    let block_y = 16u32;
    let grid = CubeCount::new_2d(dst_w.div_ceil(block_x), dst_h.div_ceil(block_y));
    let dim = CubeDim::new_2d(block_x, block_y);

    let src = pyramid.clone().offset_start(pyramid_slot_byte_offset(
        width, height, frame_count, level - 1, slot,
    ));
    let dst = pyramid.clone().offset_start(pyramid_slot_byte_offset(
        width, height, frame_count, level, slot,
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

    #[test]
    fn pyramid_pixels_single_level_matches_image() {
        assert_eq!(pyramid_pixels_per_frame(64, 32, 1), 64 * 32);
    }

    #[test]
    fn pyramid_pixels_two_levels_sums_levels() {
        // Level 0: 64x32 = 2048; level 1: 32x16 = 512. Total 2560.
        assert_eq!(pyramid_pixels_per_frame(64, 32, 2), 2048 + 512);
    }

    #[test]
    fn level_dims_halve() {
        assert_eq!(level_dims(64, 32, 0), (64, 32));
        assert_eq!(level_dims(64, 32, 1), (32, 16));
        assert_eq!(level_dims(64, 32, 2), (16, 8));
    }

    #[test]
    fn slot_byte_offset_advances_past_full_levels() {
        // 4 frames, 64x32 image, 2 levels. Offset to (level=1, frame=2):
        //   skip level 0 entirely: 4 * 64 * 32 = 8192 pixels
        //   plus 2 frames at level 1 (32x16 = 512 each): 1024 pixels
        let bytes = pyramid_slot_byte_offset(64, 32, 4, 1, 2);
        assert_eq!(bytes as usize, (8192 + 1024) * 4);
    }
}

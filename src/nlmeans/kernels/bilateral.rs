use cubecl::prelude::*;
use cubecl::terminate;

use super::helpers::{read_clamped_line, read_line};

/// Joint spatial+range Gaussian bilateral prefilter. Each thread loads
/// a `(block + 2·radius)²` tile of source pixels into shared memory as
/// `Line<f32>` (one vectorised entry per pixel), then convolves over
/// the patch using
///     `w = exp(-(dx² + dy²) · inv_two_sigma_s_sq)`
///     `  · exp(-||Δc||² · inv_two_sigma_r_sq)`
/// against the centre pixel. The output keeps the same channel layout
/// as the input (padding lanes copied through unchanged) so it can
/// stand in for `input` in the `_ref` distance kernels.
#[cube(launch)]
pub fn nlm_bilateral(
    input: &Array<Line<f32>>,
    output: &mut Array<Line<f32>>,
    frame: u32,
    inv_two_sigma_s_sq: f32,
    inv_two_sigma_r_sq: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
    #[comptime] radius: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let tile_width = comptime!(block_x + 2 * radius);
    let tile_elems = comptime!((block_x + 2 * radius) * (block_y + 2 * radius));
    let mut smem =
        SharedMemory::<f32>::new_lined(tile_elems as usize, stored_ch as usize);

    let local_x = UNIT_POS_X;
    let local_y = UNIT_POS_Y;
    let global_x = CUBE_POS_X * block_x + local_x;
    let global_y = CUBE_POS_Y * block_y + local_y;

    let tile_start_x = CUBE_POS_X as i32 * block_x as i32 - radius as i32;
    let tile_start_y = CUBE_POS_Y as i32 * block_y as i32 - radius as i32;
    let tile_end_x = tile_start_x + comptime!((block_x + 2 * radius) as i32);
    let tile_end_y = tile_start_y + comptime!((block_y + 2 * radius) as i32);

    let interior =
        tile_start_x >= 0 && tile_end_x <= width as i32 && tile_start_y >= 0 && tile_end_y <= height as i32;

    let threads = block_x * block_y;
    let thread_id = local_y * block_x + local_x;
    let mut idx = thread_id;
    if interior {
        while idx < tile_elems {
            let tile_x = idx % tile_width;
            let tile_y = idx / tile_width;
            let src_x = (tile_start_x + tile_x as i32) as u32;
            let src_y = (tile_start_y + tile_y as i32) as u32;
            smem[idx as usize] = read_line(input, src_x, src_y, frame, width, height);
            idx += threads;
        }
    } else {
        while idx < tile_elems {
            let tile_x = idx % tile_width;
            let tile_y = idx / tile_width;
            let src_x = tile_start_x + tile_x as i32;
            let src_y = tile_start_y + tile_y as i32;
            smem[idx as usize] = read_clamped_line(input, src_x, src_y, frame, width, height);
            idx += threads;
        }
    }

    sync_cube();

    if global_x >= width || global_y >= height {
        terminate!();
    }

    let center_tile_x = local_x + radius;
    let center_tile_y = local_y + radius;
    let center = smem[(center_tile_y * tile_width + center_tile_x) as usize];

    let patch_size = 2 * radius + 1;
    let mut weight_sum = 0.0f32;
    let mut acc = Line::<f32>::empty(input.line_size()).fill(0.0f32);
    for offset_y in 0..patch_size {
        for offset_x in 0..patch_size {
            let dy = offset_y as i32 - radius as i32;
            let dx = offset_x as i32 - radius as i32;
            let smem_idx = ((center_tile_y + offset_y - radius) * tile_width + center_tile_x
                + offset_x
                - radius) as usize;
            let neighbor = smem[smem_idx];

            let diff = neighbor - center;
            let mut range_sq = 0.0f32;
            #[unroll]
            for c in 0..channels {
                range_sq += diff[c as usize] * diff[c as usize];
            }
            let spatial = (dx * dx + dy * dy) as f32 * inv_two_sigma_s_sq;
            let w = f32::exp(-(spatial + range_sq * inv_two_sigma_r_sq));

            let line_w = Line::<f32>::empty(input.line_size()).fill(w);
            acc += neighbor * line_w;
            weight_sum += w;
        }
    }

    let inv = 1.0f32 / weight_sum;
    let line_inv = Line::<f32>::empty(input.line_size()).fill(inv);
    output[((frame * height + global_y) * width + global_x) as usize] = acc * line_inv;
}

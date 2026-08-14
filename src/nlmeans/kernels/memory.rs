use cubecl::prelude::*;

/// Copies `length` elements from `src[src_offset..]` into
/// `dst[dst_offset..]`, entirely on the GPU.
///
/// The loop is strided so the grid can stay under the 65,535 dispatch
/// limit.
///
/// The offsets are kernel arguments rather than byte offsets on the
/// bound handles, which lets a caller address any slot of a ring buffer
/// whatever its stride. The GPU only accepts a buffer bound at a
/// multiple of its `min_storage_buffer_offset_alignment`, and a
/// `width * height * stored_ch` frame stride rarely lands on one.
#[cube(launch_unchecked)]
pub fn gpu_copy(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_offset: u32,
    dst_offset: u32,
    #[comptime] length: u32,
    #[comptime] total_threads: u32,
) {
    let mut idx = ABSOLUTE_POS_X;
    while idx < length {
        dst[(dst_offset + idx) as usize] = src[(src_offset + idx) as usize];
        idx += total_threads;
    }
}

/// Zeroes `accum`, `weight_sum`, and `max_weight` in one dispatch.
///
/// The main loop covers all three up to `weight_len`, then a tail loop
/// finishes the channel-padded remainder of `accum`, which is always at
/// least as long as the other two.
#[cube(launch_unchecked)]
pub fn gpu_zero_buffers(
    accum: &mut Array<f32>,
    weight_sum: &mut Array<f32>,
    max_weight: &mut Array<f32>,
    #[comptime] accum_len: u32,
    #[comptime] weight_len: u32,
    #[comptime] total_threads: u32,
) {
    let mut idx = ABSOLUTE_POS_X;

    while idx < weight_len {
        accum[idx as usize] = 0.0f32;
        weight_sum[idx as usize] = 0.0f32;
        max_weight[idx as usize] = 0.0f32;
        idx += total_threads;
    }

    while idx < accum_len {
        accum[idx as usize] = 0.0f32;
        idx += total_threads;
    }
}

/// Zeroes one plain `f32` buffer.
///
/// This covers a plane `gpu_zero_buffers` does not already clear, such
/// as the optional weight-squared accumulator, without folding it into
/// that kernel's fixed three-buffer shape.
#[cube(launch_unchecked)]
pub fn gpu_zero_one(dst: &mut Array<f32>, #[comptime] length: u32, #[comptime] total_threads: u32) {
    let mut idx = ABSOLUTE_POS_X;
    while idx < length {
        dst[idx as usize] = 0.0f32;
        idx += total_threads;
    }
}

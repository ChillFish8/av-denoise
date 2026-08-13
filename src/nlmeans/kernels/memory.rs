use cubecl::prelude::*;

/// GPU→GPU buffer copy of `length` elements from `src[src_offset..]`
/// into `dst[dst_offset..]`. Uses a strided loop so the grid can be
/// capped under the 65 535 1D dispatch limit.
///
/// The offsets are kernel arguments rather than byte offsets on the
/// bound handles, so a caller can address one slot of a ring buffer
/// whatever its stride. The GPU only accepts a buffer bound at a
/// multiple of its `min_storage_buffer_offset_alignment`, which a
/// `width * height * stored_ch` frame stride meets only by luck.
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

/// Zero `accum`, `weight_sum`, `max_weight` in one dispatch. The hot
/// loop covers all three up to `weight_len`; a tail loop finishes the
/// channel-padded remainder of `accum` (which is always at least as
/// long as `weight_sum` and `max_weight`).
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

use cubecl::prelude::*;

/// GPU→GPU buffer copy. Uses a strided loop so the grid can be capped
/// under the 65 535 1D dispatch limit.
#[cube(launch)]
pub fn gpu_copy(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    #[comptime] length: u32,
    #[comptime] total_threads: u32,
) {
    let mut idx = ABSOLUTE_POS_X;
    while idx < length {
        dst[idx as usize] = src[idx as usize];
        idx += total_threads;
    }
}

/// Zero `accum`, `weight_sum`, `max_weight` in one dispatch. The hot
/// loop covers all three up to `weight_len`; a tail loop finishes the
/// channel-padded remainder of `accum` (which is always at least as
/// long as `weight_sum` and `max_weight`).
#[cube(launch)]
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

use cubecl::prelude::*;

/// Copies `length` elements from `src[src_offset..]` into `dst[dst_offset..]`.
///
/// The loop is strided so the grid stays under the 65,535 workgroup limit. The offsets are kernel
/// arguments because a buffer can only be bound at a multiple of
/// `min_storage_buffer_offset_alignment`, and a ring slot's stride rarely lands on one.
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

/// Writes `length` values of `src` from `offset` into the same positions of `dst` as f16.
///
/// The loop is strided by `total_threads`, so a grid capped below the value count still covers it.
#[cube(launch_unchecked)]
pub fn gpu_cast_f16(
    src: &Array<f32>,
    dst: &mut Array<half::f16>,
    offset: u32,
    #[comptime] length: u32,
    #[comptime] total_threads: u32,
) {
    let mut idx = ABSOLUTE_POS_X;
    while idx < length {
        let position = (offset + idx) as usize;
        dst[position] = half::f16::cast_from(src[position]);
        idx += total_threads;
    }
}

/// Writes `length` values of `src` from `offset` into the same positions of `dst` as integer codes.
///
/// Each value is scaled by `max_code`, rounded to the nearest code and clamped between 0 and
/// `max_code`, so a malformed sample still lands on a valid code. The loop is strided by
/// `total_threads`, so a grid capped below the value count still covers it.
#[cube(launch_unchecked)]
pub fn gpu_cast_codes<C: Int>(
    src: &Array<f32>,
    dst: &mut Array<C>,
    max_code: f32,
    offset: u32,
    #[comptime] length: u32,
    #[comptime] total_threads: u32,
) {
    let mut idx = ABSOLUTE_POS_X;
    while idx < length {
        let position = (offset + idx) as usize;
        let rounded = src[position] * max_code + 0.5;
        let bounded = f32::clamp(rounded, 0.0, max_code);
        dst[position] = C::cast_from(bounded);
        idx += total_threads;
    }
}

/// Zeroes `accum`, `weight_sum` and `max_weight` in one dispatch.
///
/// `accum_len` must be at least `weight_len`, because a tail loop finishes the channel-padded
/// remainder of `accum`.
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

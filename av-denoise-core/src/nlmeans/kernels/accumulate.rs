use cubecl::prelude::*;
use cubecl::terminate;

use super::helpers::{accumulate_pair, clamp_coord};

/// Adds the forward and backward neighbour contributions at every pixel from a weight map.
///
/// `weights_fwd` and `weights_bwd` may be the same buffer, as in the symmetric case at temporal
/// offset 0. The backward weight is read at the clamped neighbour so border pixels still read a
/// valid weight.
#[cube(launch_unchecked)]
pub fn nlm_accumulate<N: Size>(
    input: &Array<Vector<f32, N>>,
    accum: &mut Array<Vector<f32, N>>,
    weight_sum: &mut Array<f32>,
    weights_fwd: &Array<f32>,
    weights_bwd: &Array<f32>,
    max_weight: &mut Array<f32>,
    frame_fwd: u32,
    frame_bwd: u32,
    q_x: i32,
    q_y: i32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    if x >= width || y >= height {
        terminate!();
    }

    let pixel_idx = (y * width + x) as usize;
    let weight_fwd = weights_fwd[pixel_idx];

    let clamped_bwd_x = clamp_coord(x as i32 - q_x, width);
    let clamped_bwd_y = clamp_coord(y as i32 - q_y, height);
    let weight_bwd = weights_bwd[(clamped_bwd_y * width + clamped_bwd_x) as usize];

    accumulate_pair(
        input, accum, weight_sum, max_weight, x, y, q_x, q_y, frame_fwd, frame_bwd, weight_fwd, weight_bwd,
        width, height,
    );
}

/// Turns the accumulated sums into the denoised output.
///
/// The result is `(original * self_weight + accum) / (self_weight + weight_sum)`, where
/// `self_weight` is `wref * max_weight`. When the denominator is close to zero, no usable match was
/// found and the original pixel is kept.
///
/// `center_frame` and `output_frame` index into the whole bound rings, because a buffer can only
/// be bound at the GPU's offset alignment and a ring slot rarely lands on it.
#[cube(launch_unchecked)]
pub fn nlm_finish<N: Size>(
    input: &Array<Vector<f32, N>>,
    output: &mut Array<Vector<f32, N>>,
    accum: &Array<Vector<f32, N>>,
    weight_sum: &Array<f32>,
    max_weight: &Array<f32>,
    center_frame: u32,
    output_frame: u32,
    wref: f32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    if x >= width || y >= height {
        terminate!();
    }

    let pixel_idx = (y * width + x) as usize;
    let frame_idx = ((center_frame * height + y) * width + x) as usize;
    let output_idx = ((output_frame * height + y) * width + x) as usize;

    let self_weight = wref * max_weight[pixel_idx];
    let denominator = self_weight + weight_sum[pixel_idx];

    let original = input[frame_idx];
    let accumulated = accum[pixel_idx];

    // `Vector::empty` zeroes its lanes, so the padding lanes of a 3-channel frame stay 0.
    let mut out = Vector::<f32, N>::empty();

    if denominator > 1e-30f32 {
        let inv_denominator = 1.0f32 / denominator;

        #[unroll]
        for channel in 0..channels {
            out[channel as usize] =
                (original[channel as usize] * self_weight + accumulated[channel as usize]) * inv_denominator;
        }
    } else {
        #[unroll]
        for channel in 0..channels {
            out[channel as usize] = original[channel as usize];
        }
    }

    output[output_idx] = out;
}

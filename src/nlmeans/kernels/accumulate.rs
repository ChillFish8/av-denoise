use cubecl::prelude::*;
use cubecl::terminate;

use super::helpers::{accumulate_pair, clamp_coord};

/// Apply the `+q` and `−q` contributions at every pixel using a single
/// weight map. `weights_fwd` and `weights_bwd` may point to the same
/// buffer for the symmetric (k=0) case. The backward lookup uses the
/// clamped neighbour index so border pixels read a valid weight.
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

/// Normalise the accumulated sums into the denoised output:
///     `out = (original × m + acc) / (m + weight_sum)`  where  `m = wref × max_weight`.
/// When the denominator is near zero (no usable matches across the
/// search window) the original pixel value is preserved unchanged.
///
/// Reads the centre pixel from `input`'s frame `center_frame` and
/// writes to `output`'s frame `output_frame`, so a caller writing into
/// a ring buffer slot passes the whole ring rather than binding it at
/// the slot's byte offset (which the GPU only accepts on 32-byte
/// boundaries).
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

    let m = wref * max_weight[pixel_idx];
    let denominator = m + weight_sum[pixel_idx];

    let original = input[frame_idx];
    let accumulated = accum[pixel_idx];

    // `Vector::empty` zero-initialises so any padding lanes (vec3 → vec4)
    // stay 0 regardless of which branch runs below.
    let mut out = Vector::<f32, N>::empty();

    if denominator > 1e-30f32 {
        let inv_denominator = 1.0f32 / denominator;
        #[unroll]
        for c in 0..channels {
            out[c as usize] = (original[c as usize] * m + accumulated[c as usize]) * inv_denominator;
        }
    } else {
        #[unroll]
        for c in 0..channels {
            out[c as usize] = original[c as usize];
        }
    }

    output[output_idx] = out;
}

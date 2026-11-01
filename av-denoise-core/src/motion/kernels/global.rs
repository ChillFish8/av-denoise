use cubecl::prelude::*;
use cubecl::terminate;

use crate::motion::global::{AFFINE_WORDS, MODEL_WORDS, NO_PAIR};

/// Composes the camera model from the centre frame to each neighbour, in the pass's own pixels.
///
/// One thread handles one neighbour, in motion-field order with past neighbours first. Its
/// `max_steps` gap entries name the model-ring slots between the centre and that neighbour,
/// listed from the centre outward. A future neighbour applies them in order. A past neighbour
/// multiplies them into the motion from the neighbour to the centre and writes its inverse.
/// A `NO_PAIR` entry, or a model whose `valid` word is zero, counts as no motion.
///
/// Models are in luma pixels and each output is `m00, m01, m10, m11, tx, ty` with
/// `x_luma = x_pass * scale` per axis.
#[cube(launch_unchecked)]
pub fn mc_compose_models(
    models: &[f32],
    gaps: &[u32],
    out: &mut [f32],
    #[comptime] neighbours: u32,
    #[comptime] max_steps: u32,
    scale_x: f32,
    scale_y: f32,
) {
    let neighbour = ABSOLUTE_POS_X;
    if neighbour >= neighbours {
        terminate!();
    }

    let past = neighbour < neighbours / 2;
    let mut acc_a = 1.0f32;
    let mut acc_b = 0.0f32;
    let mut acc_tx = 0.0f32;
    let mut acc_ty = 0.0f32;

    for step in 0..max_steps {
        let gap = gaps[(neighbour * max_steps + step) as usize];
        let mut pair_a = 1.0f32;
        let mut pair_b = 0.0f32;
        let mut pair_tx = 0.0f32;
        let mut pair_ty = 0.0f32;

        if gap != NO_PAIR {
            let base = gap * MODEL_WORDS;
            let valid = models[(base + 4) as usize];
            if valid != 0.0 {
                pair_a = models[base as usize];
                pair_b = models[(base + 1) as usize];
                pair_tx = models[(base + 2) as usize];
                pair_ty = models[(base + 3) as usize];
            }
        }

        // Future steps apply after the motion so far, past steps apply before it.
        let mut outer_a = pair_a;
        let mut outer_b = pair_b;
        let mut outer_tx = pair_tx;
        let mut outer_ty = pair_ty;
        let mut inner_a = acc_a;
        let mut inner_b = acc_b;
        let mut inner_tx = acc_tx;
        let mut inner_ty = acc_ty;

        if past {
            outer_a = acc_a;
            outer_b = acc_b;
            outer_tx = acc_tx;
            outer_ty = acc_ty;
            inner_a = pair_a;
            inner_b = pair_b;
            inner_tx = pair_tx;
            inner_ty = pair_ty;
        }

        acc_a = outer_a * inner_a - outer_b * inner_b;
        acc_b = outer_a * inner_b + outer_b * inner_a;
        acc_tx = outer_a * inner_tx - outer_b * inner_ty + outer_tx;
        acc_ty = outer_b * inner_tx + outer_a * inner_ty + outer_ty;
    }

    if past {
        let norm = acc_a * acc_a + acc_b * acc_b;
        let mut inverse_a = 1.0f32;
        let mut inverse_b = 0.0f32;
        let mut inverse_tx = 0.0f32;
        let mut inverse_ty = 0.0f32;

        if norm > 0.0 {
            let reciprocal = 1.0f32 / norm;
            inverse_a = acc_a * reciprocal;
            inverse_b = -acc_b * reciprocal;
            inverse_tx = -(inverse_a * acc_tx - inverse_b * acc_ty);
            inverse_ty = -(inverse_b * acc_tx + inverse_a * acc_ty);
        }

        acc_a = inverse_a;
        acc_b = inverse_b;
        acc_tx = inverse_tx;
        acc_ty = inverse_ty;
    }

    let base = (neighbour * AFFINE_WORDS) as usize;
    out[base] = acc_a;
    out[base + 1] = -acc_b * scale_y / scale_x;
    out[base + 2] = acc_b * scale_x / scale_y;
    out[base + 3] = acc_a;
    out[base + 4] = acc_tx / scale_x;
    out[base + 5] = acc_ty / scale_y;
}

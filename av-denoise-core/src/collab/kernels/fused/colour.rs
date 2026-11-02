use cubecl::prelude::*;

use crate::collab::PATCH_SIZE;
use crate::noise::TEMPORAL_NOISE_BLOCK;
use crate::noise::colour::{COLOUR_CELLS, COLOUR_CHROMA_CENTRE, COLOUR_MEAN_FIELDS};
use crate::noise::kernels::{colour_gain, colour_gain_pair};

/// Where a reference patch reads the colour gains of its thresholds.
///
/// `block` is the block of the colour means grid that holds the patch's centre. `scale` turns a
/// normalised sample into the 8-bit scale the gain tables use. The gains apply only when `on` is
/// not 0.
#[derive(CubeType, Clone, Copy)]
pub(crate) struct PatchColour {
    pub(crate) block: u32,
    pub(crate) scale: f32,
    pub(crate) on: u32,
}

/// The colour lookup of the patch whose top-left is `(ref_x, ref_y)` in a plane `width` wide.
#[cube]
pub(crate) fn patch_colour(
    ref_x: u32,
    ref_y: u32,
    scale: f32,
    on: u32,
    #[comptime] width: u32,
) -> PatchColour {
    let blocks_x = comptime!(width.div_ceil(TEMPORAL_NOISE_BLOCK));
    let half_patch = comptime!(PATCH_SIZE / 2);
    let block_x = (ref_x + half_patch) / TEMPORAL_NOISE_BLOCK;
    let block_y = (ref_y + half_patch) / TEMPORAL_NOISE_BLOCK;

    PatchColour {
        block: block_y * blocks_x + block_x,
        scale,
        on,
    }
}

/// The gains of channel 0's and channel 1's thresholds at a patch of mean `patch_first` and
/// `patch_second`, both normalised.
///
/// A luma patch reads its own mean luma and its block's Cb and Cr from `colour_means`. A chroma
/// patch reads its own Cb and Cr and its block's luma, and each channel reads its own plane's table
/// in `colour_gains`. Every gain is 1 while `colour.on` is 0.
#[cube]
pub(crate) fn threshold_gains(
    colour_gains: &[f32],
    colour_means: &[f32],
    colour: &PatchColour,
    patch_first: f32,
    patch_second: f32,
    #[comptime] channels: u32,
) -> (f32, f32) {
    let mut first_gain = 1.0f32;
    let mut second_gain = 1.0f32;

    if colour.on != 0u32 {
        let mean_base = colour.block * COLOUR_MEAN_FIELDS;
        let block_luma = colour_means[mean_base as usize];
        let block_cb = colour_means[(mean_base + 1u32) as usize];
        let block_cr = colour_means[(mean_base + 2u32) as usize];

        if comptime!(channels == 1) {
            let patch_luma = patch_first * colour.scale;
            first_gain = colour_gain(colour_gains, 0u32, patch_luma, block_cb, block_cr);
        }

        if comptime!(channels == 2) {
            let scaled_cb = patch_first * colour.scale;
            let scaled_cr = patch_second * colour.scale;
            let patch_cb = scaled_cb - COLOUR_CHROMA_CENTRE;
            let patch_cr = scaled_cr - COLOUR_CHROMA_CENTRE;
            let (cb_gain, cr_gain) =
                colour_gain_pair(colour_gains, 0u32, COLOUR_CELLS, block_luma, patch_cb, patch_cr);
            first_gain = cb_gain;
            second_gain = cr_gain;
        }
    }

    (first_gain, second_gain)
}

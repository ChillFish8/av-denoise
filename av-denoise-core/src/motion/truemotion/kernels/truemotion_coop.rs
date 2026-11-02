use cubecl::prelude::*;

use super::truemotion::{PassValues, truemotion_block};

/// Runs one truemotion pass with every thread of the cube sharing every vector's SAD.
///
/// It picks the same vectors as [mc_truemotion](crate::motion::truemotion::kernels::mc_truemotion)
/// up to rounding, with the same inputs and outputs. Launch one cube of `threads` threads along x
/// per block per neighbour, where `threads` is a power of two of at least 8. Each thread sums the
/// absolute differences of every `threads`-th pixel of the block for every listed vector, so
/// neighbouring threads read neighbouring pixels. The partial sums meet in shared memory through a
/// tree with one barrier per level.
///
/// Vectors whose differences match pixel for pixel tie exactly, because every vector goes through
/// the same tree.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, value or comptime shape the kernel binds"
)]
pub fn mc_truemotion_coop(
    pyramid: &[f32],
    neighbour_slots: &[u32],
    mv_in: &[i32],
    seeds: &[i32],
    predictions: &[f32],
    changes_in: &[u32],
    mv_out: &mut [i32],
    confidence_out: &mut [f32],
    sad_out: &mut [f32],
    changes_out: &mut [u32],
    noise: &[f32],
    centre_slot: u32,
    lambda_base: f32,
    lambda_noise: f32,
    surcharge: f32,
    fade_level: f32,
    thsad: f32,
    sad_field: u32,
    #[comptime] has_seeds: bool,
    #[comptime] model_candidate: bool,
    #[comptime] reads_floor: bool,
    #[comptime] write_confidence: bool,
    #[comptime] write_sad: bool,
    #[comptime] reads_changes: bool,
    #[comptime] writes_changes: bool,
    #[comptime] refine_radius: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] slot_pixels: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
    #[comptime] mv_stride: u32,
    #[comptime] conf_stride: u32,
    #[comptime] threads: u32,
) {
    let values = PassValues {
        lambda_base,
        lambda_noise,
        surcharge,
        fade_level,
        thsad,
        sad_field,
    };

    truemotion_block(
        pyramid,
        neighbour_slots,
        mv_in,
        seeds,
        predictions,
        changes_in,
        mv_out,
        confidence_out,
        sad_out,
        changes_out,
        noise,
        centre_slot,
        &values,
        has_seeds,
        model_candidate,
        reads_floor,
        write_confidence,
        write_sad,
        reads_changes,
        writes_changes,
        refine_radius,
        width,
        height,
        slot_pixels,
        blksize,
        step,
        blocks_x,
        blocks_y,
        mv_stride,
        conf_stride,
        threads,
    );
}

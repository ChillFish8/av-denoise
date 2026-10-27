use cubecl::prelude::*;
use cubecl::server::Handle;

use super::MotionCtx;
use super::analyse::{MatchTargets, run_analyse};
use super::kernels::mc_pair_zero;
use crate::engine::{BLOCK_1D, MAX_GRID_1D};
use crate::noise::DERIVED_SAD_FLOOR;

/// Where one slot and direction of the pair ring starts.
///
/// The ring is indexed by slot, direction, block and then component, and direction 0 runs from the
/// older frame to the newer one. Each direction is padded as [MotionCtx::pair_direction_bytes]
/// describes. A slot is keyed by the newer frame's push index modulo the ring size, which
/// [pair_ring_slot_count](crate::motion::pair_ring_slot_count) makes safe.
/// [mc_chain_compose](crate::motion::kernels::mc_chain_compose()) steps through the ring with the
/// same padded strides, so the two must change together.
pub(crate) fn pair_byte_offset(motion_ctx: &MotionCtx, pair_slot: u32, direction: u32) -> u64 {
    (pair_slot as u64) * motion_ctx.pair_slot_bytes() + (direction as u64) * motion_ctx.pair_direction_bytes()
}

/// Measures motion between a pushed frame and the one before it, both ways, into the pair ring.
///
/// `older_slot` and `newer_slot` are physical input-ring slots, and `slot_tables` holds one
/// single-slot table per ring slot. Nothing reads confidence at the pair level, so both directions
/// turn it off and target `confidence_dummy`, and `noise` is bound only to fill the kernel's
/// derived noise values.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
pub(crate) fn run_pair_analyse(
    client: &Client,
    motion_ctx: &MotionCtx,
    width: u32,
    height: u32,
    frame_count: u32,
    older_slot: u32,
    newer_slot: u32,
    pair_slot: u32,
    pyramid: &Handle,
    slot_tables: &[Handle],
    pair_ring: &Handle,
    confidence_dummy: &Handle,
    noise: &Handle,
) -> Result<(), anyhow::Error> {
    let direction_len = motion_ctx.pair_direction_len() as usize;

    let older_to_newer_offset = pair_byte_offset(motion_ctx, pair_slot, 0);
    let older_to_newer = pair_ring.clone().offset_start(older_to_newer_offset);
    let older_to_newer_targets = MatchTargets {
        pyramid,
        frame_count,
        centre_slot: older_slot,
        neighbour_slots: &slot_tables[newer_slot as usize],
        neighbours: 1,
        mv_field: &older_to_newer,
        mv_len: direction_len,
        confidence: confidence_dummy,
        conf_len: 1,
        write_confidence: false,
    };
    run_analyse(
        client,
        motion_ctx,
        width,
        height,
        &older_to_newer_targets,
        noise,
        DERIVED_SAD_FLOOR,
        1.0,
    )?;

    let newer_to_older_offset = pair_byte_offset(motion_ctx, pair_slot, 1);
    let newer_to_older = pair_ring.clone().offset_start(newer_to_older_offset);
    let newer_to_older_targets = MatchTargets {
        pyramid,
        frame_count,
        centre_slot: newer_slot,
        neighbour_slots: &slot_tables[older_slot as usize],
        neighbours: 1,
        mv_field: &newer_to_older,
        mv_len: direction_len,
        confidence: confidence_dummy,
        conf_len: 1,
        write_confidence: false,
    };
    run_analyse(
        client,
        motion_ctx,
        width,
        height,
        &newer_to_older_targets,
        noise,
        DERIVED_SAD_FLOOR,
        1.0,
    )?;

    Ok(())
}

/// Fills both directions of one pair-ring slot with zeroes.
///
/// Duplicated ring slots, which appear while priming and during the end-of-stream flush, hold the
/// same frame twice, so their motion is zero. Each direction gets its own dispatch because padding
/// separates the two.
pub(crate) fn zero_pair_slot(client: &Client, motion_ctx: &MotionCtx, pair_ring: &Handle, pair_slot: u32) {
    let length = motion_ctx.pair_direction_len();
    let grid = length.div_ceil(BLOCK_1D).min(MAX_GRID_1D);
    let total_threads = grid * BLOCK_1D;

    for direction in 0..2u32 {
        let offset = pair_byte_offset(motion_ctx, pair_slot, direction);
        let direction_slice = pair_ring.clone().offset_start(offset);
        let cube_count = CubeCount::new_1d(grid);
        let cube_dim = CubeDim::new_1d(BLOCK_1D);

        unsafe {
            let direction_arg = BufferArg::from_raw_parts(direction_slice, length as usize);

            mc_pair_zero::launch_unchecked(
                client,
                cube_count,
                cube_dim,
                direction_arg,
                length,
                total_threads,
            );
        }
    }
}

/// Maps a nonzero temporal offset onto its motion-field neighbour index.
///
/// Negative offsets take indices 0 up to the radius minus 1 and positive offsets follow, which is
/// the order the analyse, confidence and compose passes fill their buffers in.
pub(crate) fn neighbour_idx_for_k(radius: u32, k: i32) -> u32 {
    debug_assert_ne!(k, 0, "k=0 is the spatial pair, it has no neighbour index");
    debug_assert!(
        k.unsigned_abs() <= radius,
        "k={k} outside the temporal window ±{radius}"
    );

    if k < 0 {
        (k + radius as i32) as u32
    } else {
        (radius as i32 - 1 + k) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::align::StorageAlign;
    use crate::motion::{MotionCompensationMode, MotionEstimation};

    /// A 4-pixel-high frame cut into 4x4 blocks with no overlap.
    fn small_block_ctx(width: u32) -> MotionCtx {
        let mode = MotionCompensationMode::Mvtools {
            blksize: 4,
            overlap: 0,
            search_radius: 1,
            pyramid_levels: 1,
            estimation: MotionEstimation::Direct,
        };
        let align = StorageAlign::new(32);

        MotionCtx::new(mode, width, 4, align).unwrap()
    }

    #[test]
    fn neighbour_idx_for_k_matches_dispatch_convention() {
        // Negative offsets take indices 0 up to radius minus 1, then the positive ones follow.
        let furthest_back = neighbour_idx_for_k(2, -2);
        let nearest_back = neighbour_idx_for_k(2, -1);
        let nearest_forward = neighbour_idx_for_k(2, 1);
        let furthest_forward = neighbour_idx_for_k(2, 2);

        assert_eq!(furthest_back, 0);
        assert_eq!(nearest_back, 1);
        assert_eq!(nearest_forward, 2);
        assert_eq!(furthest_forward, 3);
    }

    #[test]
    fn pair_byte_offset_pads_small_block_counts_to_32_bytes() {
        // One block gives an unpadded direction stride of 8 bytes, which would leave direction 1
        // off a 32-byte boundary.
        let ctx = small_block_ctx(4);
        assert_eq!(
            ctx.blocks_x * ctx.blocks_y,
            1,
            "fixture should have exactly one block"
        );

        let slot_0_forward = pair_byte_offset(&ctx, 0, 0);
        let slot_0_backward = pair_byte_offset(&ctx, 0, 1);
        let slot_1_forward = pair_byte_offset(&ctx, 1, 0);
        let slot_1_backward = pair_byte_offset(&ctx, 1, 1);

        assert_eq!(slot_0_forward, 0);
        assert_eq!(slot_0_backward, 32);
        assert_eq!(slot_1_forward, 64);
        assert_eq!(slot_1_backward, 96);
    }

    #[test]
    fn pair_byte_offset_direction_one_pads_even_when_slot_base_is_aligned() {
        // Two blocks give an aligned 32-byte unpadded slot stride, but the 16-byte direction stride
        // inside it still needs padding of its own.
        let ctx = small_block_ctx(8);
        assert_eq!(
            ctx.blocks_x * ctx.blocks_y,
            2,
            "fixture should have exactly two blocks"
        );

        let slot_0_forward = pair_byte_offset(&ctx, 0, 0);
        let slot_0_backward = pair_byte_offset(&ctx, 0, 1);
        let slot_1_forward = pair_byte_offset(&ctx, 1, 0);
        let slot_1_backward = pair_byte_offset(&ctx, 1, 1);

        assert_eq!(slot_0_forward, 0);
        assert_eq!(slot_0_backward, 32);
        assert_eq!(slot_1_forward, 64);
        assert_eq!(slot_1_backward, 96);
    }
}

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::MotionCtx;
use super::analyse::{mv_field_byte_offset, run_analyse};
use crate::nlmeans::denoiser::NlmDenoiser;
use crate::nlmeans::kernels::motion::{nlm_mc_chain_compose, nlm_mc_pair_zero};
use crate::nlmeans::{BLOCK_1D, MAX_GRID_1D};

/// Where one slot and direction of the pair ring starts.
///
/// The ring is indexed by slot, direction, block and then component, and direction 0 runs from the
/// older frame to the newer one. Each direction is padded as [MotionCtx::pair_direction_bytes]
/// describes. A slot is keyed by the newer frame's push index modulo the ring size, which
/// [pair_ring_slot_count](crate::nlmeans::motion::pair_ring_slot_count) makes safe.
/// `nlm_mc_chain_compose` steps through the ring with the same padded strides, so the two must
/// change together.
pub(crate) fn pair_byte_offset(motion_ctx: &MotionCtx, pair_slot: u32, direction: u32) -> u64 {
    (pair_slot as u64) * motion_ctx.pair_slot_bytes() + (direction as u64) * motion_ctx.pair_direction_bytes()
}

/// Measures motion between a pushed frame and the one before it, both ways, into the pair ring.
///
/// `older_slot` and `newer_slot` are physical input-ring slots. Nothing reads confidence at the pair
/// level, so both directions turn it off and target `confidence_dummy`.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
pub(crate) fn run_pair_analyse<R: Runtime>(
    client: &ComputeClient<R>,
    motion_ctx: &MotionCtx,
    width: u32,
    height: u32,
    frame_count: u32,
    older_slot: u32,
    newer_slot: u32,
    pair_slot: u32,
    pyramid: &Handle,
    pair_ring: &Handle,
    confidence_dummy: &Handle,
) -> Result<(), anyhow::Error> {
    let older_to_newer_offset = pair_byte_offset(motion_ctx, pair_slot, 0);
    let older_to_newer = pair_ring.clone().offset_start(older_to_newer_offset);
    run_analyse::<R>(
        client,
        motion_ctx,
        width,
        height,
        frame_count,
        older_slot,
        newer_slot,
        0,
        pyramid,
        &older_to_newer,
        confidence_dummy,
        false,
        0.0,
        1.0,
    )?;

    let newer_to_older_offset = pair_byte_offset(motion_ctx, pair_slot, 1);
    let newer_to_older = pair_ring.clone().offset_start(newer_to_older_offset);
    run_analyse::<R>(
        client,
        motion_ctx,
        width,
        height,
        frame_count,
        newer_slot,
        older_slot,
        0,
        pyramid,
        &newer_to_older,
        confidence_dummy,
        false,
        0.0,
        1.0,
    )?;

    Ok(())
}

/// Fills both directions of one pair-ring slot with zeroes.
///
/// Duplicated ring slots, which appear while priming and during the end-of-stream flush, hold the
/// same frame twice, so their motion is zero. Each direction gets its own dispatch because padding
/// separates the two.
pub(crate) fn zero_pair_slot<R: Runtime>(
    client: &ComputeClient<R>,
    motion_ctx: &MotionCtx,
    pair_ring: &Handle,
    pair_slot: u32,
) {
    let length = motion_ctx.pair_direction_len();
    let grid = length.div_ceil(BLOCK_1D).min(MAX_GRID_1D);
    let total_threads = grid * BLOCK_1D;

    for direction in 0..2u32 {
        let offset = pair_byte_offset(motion_ctx, pair_slot, direction);
        let direction_slice = pair_ring.clone().offset_start(offset);

        unsafe {
            nlm_mc_pair_zero::launch_unchecked::<R>(
                client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(BLOCK_1D),
                ArrayArg::from_raw_parts(direction_slice, length as usize),
                length,
                total_threads,
            );
        }
    }
}

/// Launches `nlm_mc_chain_compose` into this neighbour's `mv_field` slot.
///
/// The padded strides are passed explicitly, because the kernel reads the whole pair ring as one
/// array and must step through it as [pair_byte_offset] does.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
fn dispatch_chain_compose<R: Runtime>(
    client: &ComputeClient<R>,
    motion_ctx: &MotionCtx,
    width: u32,
    height: u32,
    pair_ring_slots: u32,
    pair_ring_len: usize,
    start_pair_slot: u32,
    forward: bool,
    steps: u32,
    pair_ring: &Handle,
    mv_field: &Handle,
    neighbour_idx: u32,
) -> Result<(), anyhow::Error> {
    let mv_offset = mv_field_byte_offset(motion_ctx, neighbour_idx);
    let mv_slot = mv_field.clone().offset_start(mv_offset);
    let mv_slot_len = (motion_ctx.blocks_x as usize) * (motion_ctx.blocks_y as usize) * 2;

    // One single-thread cube per block, since there is no per-candidate work to share.
    let grid = CubeCount::new_2d(motion_ctx.blocks_x, motion_ctx.blocks_y);
    let dim = CubeDim::new_2d(1, 1);

    unsafe {
        nlm_mc_chain_compose::launch_unchecked::<R>(
            client,
            grid,
            dim,
            ArrayArg::from_raw_parts(pair_ring.clone(), pair_ring_len),
            ArrayArg::from_raw_parts(mv_slot, mv_slot_len),
            start_pair_slot,
            forward,
            steps,
            pair_ring_slots,
            motion_ctx.pair_direction_stride(),
            motion_ctx.pair_slot_stride(),
            motion_ctx.step,
            width,
            height,
            motion_ctx.blocks_x,
            motion_ctx.blocks_y,
        );
    }

    Ok(())
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

impl<R: Runtime> NlmDenoiser<R> {
    /// Joins the adjacent-frame fields into one motion field for the neighbour at offset `k`.
    ///
    /// `k` must be nonzero and land inside the
    /// ring, at most twice the temporal radius either way. The walk takes one hop per step out from
    /// the centre `center_t`, following the older-to-newer field for a positive `k` and the
    /// newer-to-older field for a negative one. It does nothing unless `Chained` estimation is
    /// active.
    pub(crate) fn run_chain_compose(
        &self,
        center_t: u32,
        k: i32,
        neighbour_idx: u32,
    ) -> Result<(), anyhow::Error> {
        let Some(motion_ctx) = self.mc_ctx.as_ref() else {
            return Ok(());
        };

        if !self.is_chained() || k == 0 {
            return Ok(());
        }

        let radius = self.params.temporal_radius;
        let far_edge = 2 * radius as i32;
        debug_assert!(
            (0..=far_edge).contains(&(center_t as i32 + k)),
            "center_t={center_t} k={k} lands outside the ring 0..={far_edge}"
        );
        debug_assert!(
            neighbour_idx < 2 * radius,
            "neighbour_idx={neighbour_idx} outside the field, which holds {} neighbours",
            2 * radius,
        );

        let pair_ring = self
            .pair_ring_buf
            .as_ref()
            .expect("pair_ring allocated when Chained is active");
        let mv_field = self
            .mv_field_buf
            .as_ref()
            .expect("mv_field allocated when mc_ctx is Some");

        let forward = k > 0;
        let steps = k.unsigned_abs();
        let start_gap = if forward {
            center_t as i32
        } else {
            center_t as i32 - 1
        };
        let start_pair_slot = self.pair_slot(start_gap);
        let pair_ring_slots = super::pair_ring_slot_count(radius);
        let pair_ring_len = pair_ring_slots as usize * motion_ctx.pair_slot_stride() as usize;

        dispatch_chain_compose::<R>(
            &self.client,
            motion_ctx,
            self.width,
            self.height,
            pair_ring_slots,
            pair_ring_len,
            start_pair_slot,
            forward,
            steps,
            pair_ring,
            mv_field,
            neighbour_idx,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nlmeans::align::StorageAlign;
    use crate::nlmeans::motion::{MotionCompensationMode, MotionEstimation};

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

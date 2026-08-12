use cubecl::prelude::*;
use cubecl::server::Handle;

use super::MotionCtx;
use super::analyse::{mv_field_byte_offset, run_analyse};
use crate::nlmeans::denoiser::NlmDenoiser;
use crate::nlmeans::kernels::motion::{nlm_mc_chain_compose, nlm_mc_pair_zero};
use crate::nlmeans::{BLOCK_1D, MAX_GRID_1D};

/// Byte offset of a `(pair_slot, direction)` sub-array inside the pair
/// ring, laid out `[pair_slot][direction][block_y][block_x][2]` of
/// `i32` (direction 0 = older→newer, 1 = newer→older), with each
/// direction's slice padded up to a 32-byte boundary (see
/// [`MotionCtx::pair_direction_bytes`]). `pair_slot` is keyed by the
/// newer frame's position in the push sequence, reduced modulo
/// `2 * temporal_radius` (see [`super::pair_ring_slot_count`]'s doc
/// comment for the lifetime this sizing guarantees). The
/// `nlm_mc_chain_compose` kernel reads the whole pair ring unsliced
/// and strides through it with the same padded values
/// (`MotionCtx::pair_direction_stride`/`pair_slot_stride`), so a
/// change here must stay in sync with that kernel's own stride
/// arguments.
pub(crate) fn pair_byte_offset(mc: &MotionCtx, pair_slot: u32, direction: u32) -> u64 {
    (pair_slot as u64) * mc.pair_slot_bytes() + (direction as u64) * mc.pair_direction_bytes()
}

/// Run the adjacent-frame pair analyse for one pushed frame, storing
/// both directions' motion fields into `pair_ring` at `pair_slot`.
/// `older_slot`/`newer_slot` are physical input-ring slots, the same
/// addressing `run_analyse` always uses. `pyramid` is the analyse
/// pyramid `run_motion_compensation` also reads from (the reference
/// ring when a prefilter is active, otherwise the raw input ring).
///
/// Confidence is never consumed at the pair level, so both directions
/// disable the fine kernel's `write_confidence` output and pass
/// `confidence_dummy` as its placeholder target.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_pair_analyse<R: Runtime>(
    client: &ComputeClient<R>,
    mc: &MotionCtx,
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
    let older_to_newer = pair_ring.clone().offset_start(pair_byte_offset(mc, pair_slot, 0));
    run_analyse::<R>(
        client,
        mc,
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

    let newer_to_older = pair_ring.clone().offset_start(pair_byte_offset(mc, pair_slot, 1));
    run_analyse::<R>(
        client,
        mc,
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

/// Zero-fill both directions of `pair_slot`. Used for duplicated ring
/// slots (stream priming and end-of-stream flush), whose pair is zero
/// motion by definition since both "frames" are identical content.
/// Zeroes each direction with its own dispatch, at its own padded
/// offset, since the two directions no longer sit contiguously once
/// `pair_direction_bytes` pads between them.
pub(crate) fn zero_pair_slot<R: Runtime>(
    client: &ComputeClient<R>,
    mc: &MotionCtx,
    pair_ring: &Handle,
    pair_slot: u32,
) {
    let length = mc.pair_direction_len();
    let grid = length.div_ceil(BLOCK_1D).min(MAX_GRID_1D);
    let total_threads = grid * BLOCK_1D;

    for direction in 0..2u32 {
        let offset = pair_byte_offset(mc, pair_slot, direction);
        let dst = pair_ring.clone().offset_start(offset);

        unsafe {
            nlm_mc_pair_zero::launch_unchecked::<R>(
                client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(BLOCK_1D),
                ArrayArg::from_raw_parts(dst, length as usize),
                length,
                total_threads,
            );
        }
    }
}

/// Launch `nlm_mc_chain_compose` once, writing the composed motion
/// field into `mv_field` at `neighbour_idx`'s slot (the same slot
/// convention `run_motion_compensation` uses for the direct path).
/// Passes the padded per-direction/per-slot pair-ring strides through
/// explicitly, since the kernel reads the whole pair ring unsliced and
/// must stride through it the same way `pair_byte_offset` does.
#[allow(clippy::too_many_arguments)]
fn dispatch_chain_compose<R: Runtime>(
    client: &ComputeClient<R>,
    mc: &MotionCtx,
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
    let mv_slot = mv_field
        .clone()
        .offset_start(mv_field_byte_offset(mc, neighbour_idx));
    let mv_slot_len = (mc.blocks_x as usize) * (mc.blocks_y as usize) * 2;

    // One thread per output block: a cube per block, a single thread
    // per cube (unlike the SAD-reduction kernels, there's no
    // per-candidate work to split across a cube's threads here).
    let grid = CubeCount::new_2d(mc.blocks_x, mc.blocks_y);
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
            mc.pair_direction_stride(),
            mc.pair_slot_stride(),
            mc.step,
            width,
            height,
            mc.blocks_x,
            mc.blocks_y,
        );
    }

    Ok(())
}

/// Maps a nonzero temporal offset `k` to the neighbour index the
/// analyse/compose passes use inside `mv_field_buf`. Mirrors
/// `dispatch::neighbour_idx_for_k` exactly, duplicated here rather than
/// shared since that helper is private to the direct-path dispatch
/// module. `dispatch`'s own tests cover this same mapping. `pub(crate)`
/// so tests can resolve the same neighbour index the compose path
/// writes to, instead of re-deriving the formula themselves.
pub(crate) fn neighbour_idx_for_k(radius: u32, k: i32) -> u32 {
    debug_assert_ne!(k, 0);
    debug_assert!(k.unsigned_abs() <= radius);
    if k < 0 {
        (k + radius as i32) as u32
    } else {
        (radius as i32 - 1 + k) as u32
    }
}

impl<R: Runtime> NlmDenoiser<R> {
    /// Compose the chained motion field for neighbour offset `k`
    /// (`k != 0`, `|k| <= temporal_radius`) into `mv_field_buf`, at the
    /// same slot the direct path fills for this neighbour.
    ///
    /// `center_t` is the window-relative index of the centre frame.
    /// Every caller elsewhere in the codebase passes `temporal_radius`
    /// (see `dispatch::run_motion_compensation`'s convention). Walks
    /// `|k|` adjacent-pair hops out from the centre, following the
    /// older→newer field when `k > 0` or the newer→older field when
    /// `k < 0`.
    ///
    /// No-op when `Chained` estimation isn't active. When it is
    /// active, `dispatch::run_motion_compensation` calls this once per
    /// neighbour on every submit, then corrects the composed seed with
    /// `run_seeded_refine`. Tests also call it directly.
    pub(crate) fn run_chain_compose(&self, center_t: u32, k: i32) -> Result<(), anyhow::Error> {
        let Some(mc) = self.mc_ctx.as_ref() else {
            return Ok(());
        };
        if !self.is_chained() || k == 0 {
            return Ok(());
        }

        let radius = self.params.temporal_radius;
        debug_assert!(
            k.unsigned_abs() <= radius,
            "k={k} outside the temporal window ±{radius}"
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
        let pair_ring_len = pair_ring_slots as usize * mc.pair_slot_stride() as usize;
        let neighbour_idx = neighbour_idx_for_k(radius, k);

        dispatch_chain_compose::<R>(
            &self.client,
            mc,
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
    use crate::nlmeans::motion::{MotionCompensationMode, MotionEstimation};

    #[test]
    fn neighbour_idx_for_k_matches_dispatch_convention() {
        // Same walk `dispatch::neighbour_idx_for_k`'s own tests check:
        // k = -radius..-1 first (indices 0..radius-1), then k = 1..radius.
        assert_eq!(neighbour_idx_for_k(2, -2), 0);
        assert_eq!(neighbour_idx_for_k(2, -1), 1);
        assert_eq!(neighbour_idx_for_k(2, 1), 2);
        assert_eq!(neighbour_idx_for_k(2, 2), 3);
    }

    #[test]
    fn pair_byte_offset_pads_small_block_counts_to_32_bytes() {
        // One block (4x4 frame, blksize=4, overlap=0 => step=4 => 1x1
        // blocks), so the unpadded direction stride (1 block * 2 i32
        // components * 4 bytes = 8 bytes) would place direction 1 at a
        // non-32-aligned offset.
        let m = MotionCtx::new(
            MotionCompensationMode::Mvtools {
                blksize: 4,
                overlap: 0,
                search_radius: 1,
                pyramid_levels: 1,
                estimation: MotionEstimation::Direct,
            },
            4,
            4,
        )
        .unwrap();
        assert_eq!(
            m.blocks_x * m.blocks_y,
            1,
            "fixture should have exactly one block"
        );
        assert_eq!(pair_byte_offset(&m, 0, 0), 0);
        assert_eq!(pair_byte_offset(&m, 0, 1), 32);
        assert_eq!(pair_byte_offset(&m, 1, 0), 64);
        assert_eq!(pair_byte_offset(&m, 1, 1), 96);
    }

    #[test]
    fn pair_byte_offset_direction_one_pads_even_when_slot_base_is_aligned() {
        // Two blocks (8x4 frame, blksize=4, overlap=0 => step=4 =>
        // 2x1 blocks). The unpadded per-slot stride (2 blocks * 2
        // directions * 2 i32 components * 4 bytes = 32 bytes) is
        // already 32-aligned, so every slot's own base offset is fine
        // on its own. But the unpadded per-direction stride within
        // that slot (2 blocks * 2 i32 components * 4 bytes = 16 bytes)
        // is not, so direction 1 still needs its own padding even
        // though direction 0's slot base never did.
        let m = MotionCtx::new(
            MotionCompensationMode::Mvtools {
                blksize: 4,
                overlap: 0,
                search_radius: 1,
                pyramid_levels: 1,
                estimation: MotionEstimation::Direct,
            },
            8,
            4,
        )
        .unwrap();
        assert_eq!(
            m.blocks_x * m.blocks_y,
            2,
            "fixture should have exactly two blocks"
        );
        assert_eq!(pair_byte_offset(&m, 0, 0), 0);
        assert_eq!(pair_byte_offset(&m, 0, 1), 32);
        assert_eq!(pair_byte_offset(&m, 1, 0), 64);
        assert_eq!(pair_byte_offset(&m, 1, 1), 96);
    }
}

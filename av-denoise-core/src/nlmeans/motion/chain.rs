use cubecl::prelude::*;
use cubecl::server::Handle;

use super::MotionCtx;
use super::analyse::{mv_field_byte_offset, run_analyse};
use crate::nlmeans::denoiser::NlmDenoiser;
use crate::nlmeans::kernels::motion::{nlm_mc_chain_compose, nlm_mc_pair_zero};
use crate::nlmeans::{BLOCK_1D, MAX_GRID_1D};

/// Where one slot and direction of the pair ring starts.
///
/// The ring is indexed by slot, then direction, then block, then
/// component. Direction 0 runs from the older frame to the newer one,
/// and direction 1 the other way. Each direction's slice is padded up to
/// an alignment boundary. See [`MotionCtx::pair_direction_bytes`].
///
/// The slot is keyed by the newer frame's place in the push sequence,
/// reduced modulo the ring size. [`super::pair_ring_slot_count`]
/// explains why that sizing is safe.
///
/// The `nlm_mc_chain_compose` kernel reads the whole ring as one array
/// and steps through it with the same padded strides, so any change here
/// has to stay in step with that kernel's own stride arguments.
pub(crate) fn pair_byte_offset(mc: &MotionCtx, pair_slot: u32, direction: u32) -> u64 {
    (pair_slot as u64) * mc.pair_slot_bytes() + (direction as u64) * mc.pair_direction_bytes()
}

/// Measures motion between one pushed frame and the one before it, in
/// both directions, and stores the result in the pair ring.
///
/// `older_slot` and `newer_slot` are physical input-ring slots, the same
/// addressing `run_analyse` uses.
///
/// `pyramid` is the same one `run_motion_compensation` reads, which is
/// the reference ring when a prefilter is active and the raw input ring
/// otherwise.
///
/// Nothing reads confidence at the pair level, so both directions turn
/// the fine kernel's confidence output off and pass `confidence_dummy`
/// as the placeholder target.
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

/// Fills both directions of one pair-ring slot with zeroes.
///
/// Duplicated ring slots, which appear while priming the stream and
/// again during the end-of-stream flush, hold the same content in both
/// frames, so their motion is zero by definition.
///
/// Each direction gets its own dispatch at its own padded offset,
/// because the two directions no longer sit next to each other once
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

/// Launches `nlm_mc_chain_compose` once, writing the joined motion field
/// into `mv_field` at this neighbour's slot.
///
/// That is the same slot the direct path fills for the neighbour.
///
/// The padded per-direction and per-slot strides are passed through
/// explicitly, because the kernel reads the whole pair ring as one array
/// and has to step through it the same way `pair_byte_offset` does.
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

    // One thread per output block, so one block of threads per image
    // block with a single thread in it. Unlike the scoring kernels
    // there is no per-candidate work to spread across threads here.
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

/// Maps a nonzero temporal offset onto the neighbour index the analyse
/// and compose passes use inside the motion-field buffer.
///
/// This mirrors `dispatch::neighbour_idx_for_k` exactly. It is
/// duplicated rather than shared because that helper is private to the
/// direct-path dispatch module, and `dispatch`'s own tests already cover
/// the mapping.
///
/// It is `pub(crate)` so tests can look up the same neighbour index the
/// compose path writes to, rather than working out the formula
/// themselves.
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
    /// Joins the adjacent-frame fields into one motion field for the
    /// neighbour at temporal offset `k`, writing it to the same slot the
    /// direct path fills.
    ///
    /// `k` must be nonzero and no further out than the temporal radius.
    ///
    /// The walk takes one hop per step out from the centre, following
    /// the older-to-newer field for a positive `k` and the
    /// newer-to-older field for a negative one.
    ///
    /// `center_t` is the centre frame's index within the window. Every
    /// caller passes the temporal radius, which is the convention
    /// `dispatch::run_motion_compensation` sets.
    ///
    /// This does nothing unless `Chained` estimation is active. When it
    /// is, `dispatch::run_motion_compensation` calls it once per
    /// neighbour on every submit and then cleans up the seed with
    /// `run_seeded_refine`. Tests call it directly as well.
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
    use crate::nlmeans::align::StorageAlign;
    use crate::nlmeans::motion::{MotionCompensationMode, MotionEstimation};

    #[test]
    fn neighbour_idx_for_k_matches_dispatch_convention() {
        // The same walk `dispatch::neighbour_idx_for_k`'s own tests
        // check. The negative offsets come first, taking indices 0 up
        // to radius minus 1, then the positive ones follow.
        assert_eq!(neighbour_idx_for_k(2, -2), 0);
        assert_eq!(neighbour_idx_for_k(2, -1), 1);
        assert_eq!(neighbour_idx_for_k(2, 1), 2);
        assert_eq!(neighbour_idx_for_k(2, 2), 3);
    }

    #[test]
    fn pair_byte_offset_pads_small_block_counts_to_32_bytes() {
        // A 4x4 frame at this geometry has a single block, so the
        // unpadded direction stride is only 8 bytes and would leave
        // direction 1 at an offset that is not 32-aligned.
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
            StorageAlign::new(32),
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
        // An 8x4 frame at this geometry has two blocks. The unpadded
        // per-slot stride comes to 32 bytes, which is already aligned,
        // so every slot's own base offset would be fine.
        //
        // The per-direction stride inside that slot is only 16 bytes
        // though, so direction 1 still needs padding of its own even
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
            StorageAlign::new(32),
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

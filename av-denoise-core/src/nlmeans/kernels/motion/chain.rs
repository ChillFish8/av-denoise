use cubecl::prelude::*;
use cubecl::terminate;

/// Joins a run of adjacent-frame motion fields into one motion vector
/// per block, reaching from the centre frame to a distant neighbour.
///
/// Motion is only ever measured between neighbouring frames. To reach a
/// frame several steps away, this kernel follows the picture from one
/// frame to the next and adds up what it finds.
///
/// The result goes into `mv_field` in the layout `nlm_mc_warp` and the
/// analyse kernels share.
///
/// # The walk
///
/// One thread handles one output block.
///
/// Each hop finds the block nearest the walking position, clamped at the
/// edges exactly the way `nlm_mc_warp` maps a pixel to a block. It reads
/// that block's motion, adds it to the running total, and moves the
/// walking position by the same amount before the next hop.
///
/// Only the block lookup clamps. The position itself keeps its true
/// value, so a long chain can wander outside the frame without
/// corrupting the lookups that follow.
///
/// The walk starts at `start_pair_slot` and takes `steps` hops. Going
/// forward it reads direction 0 and moves to the next slot. Going
/// backward it reads direction 1 and moves to the previous one.
///
/// Consecutive hops land on consecutive slots, because the pair ring is
/// keyed by the newer frame's place in the push sequence. See
/// `crate::nlmeans::motion::pair_ring_slot_count`. The caller therefore
/// only has to work out the first hop's slot.
///
/// # Ring layout
///
/// `pair_ring` holds every live adjacent-pair field, indexed by slot,
/// then direction, then block, then component. Direction 0 runs from the
/// older frame to the newer one, and direction 1 the other way.
///
/// Each direction's slice is padded up to a 32-byte boundary, which is 8
/// `i32` elements.
///
/// `dir_len` and `slot_len` are those padded strides in elements, taken
/// from `MotionCtx::pair_direction_stride` and `pair_slot_stride`, not
/// the raw block count. The host writes at the padded offsets, so the
/// reads here have to use the same ones.
#[cube(launch_unchecked)]
pub fn nlm_mc_chain_compose(
    pair_ring: &Array<i32>,
    mv_field: &mut Array<i32>,
    start_pair_slot: u32,
    #[comptime] forward: bool,
    #[comptime] steps: u32,
    #[comptime] pair_ring_slots: u32,
    #[comptime] dir_len: u32,
    #[comptime] slot_len: u32,
    #[comptime] step: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
) {
    let bx = ABSOLUTE_POS_X;
    let by = ABSOLUTE_POS_Y;

    if bx >= blocks_x || by >= blocks_y {
        terminate!();
    }

    let direction = comptime!(if forward { 0u32 } else { 1u32 });

    let mut pos_x = (bx * step + step / 2) as i32;
    let mut pos_y = (by * step + step / 2) as i32;
    let mut acc_x = 0i32;
    let mut acc_y = 0i32;

    for i in 0..steps {
        let slot = if forward {
            (start_pair_slot + i) % pair_ring_slots
        } else {
            (start_pair_slot + pair_ring_slots - i) % pair_ring_slots
        };

        let cx = clamp_i32(pos_x, width as i32) as u32;
        let cy = clamp_i32(pos_y, height as i32) as u32;
        let bxi = (cx / step).min(blocks_x - 1);
        let byi = (cy / step).min(blocks_y - 1);

        let base = slot * slot_len + direction * dir_len + (byi * blocks_x + bxi) * 2;
        let fx = pair_ring[base as usize];
        let fy = pair_ring[(base + 1) as usize];

        acc_x += fx;
        acc_y += fy;
        pos_x += fx;
        pos_y += fy;
    }

    let out_idx = ((by * blocks_x + bx) * 2) as usize;
    mv_field[out_idx] = acc_x;
    mv_field[out_idx + 1] = acc_y;
}

/// Fills both directions of one pair-ring slot with zeroes.
///
/// Duplicated ring slots, which appear while priming the stream and
/// again during the end-of-stream flush, hold the same content in both
/// frames. Their motion is zero by definition, so writing the zeroes is
/// both cheaper and exactly right compared with analysing identical
/// input.
#[cube(launch_unchecked)]
pub fn nlm_mc_pair_zero(dst: &mut Array<i32>, #[comptime] length: u32, #[comptime] total_threads: u32) {
    let mut idx = ABSOLUTE_POS_X;
    while idx < length {
        dst[idx as usize] = 0i32;
        idx += total_threads;
    }
}

#[cube]
fn clamp_i32(value: i32, limit: i32) -> i32 {
    let mut result = value;
    if value < 0 {
        result = 0;
    } else if value >= limit {
        result = limit - 1;
    }
    result
}

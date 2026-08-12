use cubecl::prelude::*;
use cubecl::terminate;

/// Compose a chain of adjacent-frame motion fields into a single
/// `centre → k` motion vector per block, writing the result into
/// `mv_field` at the layout `nlm_mc_warp` and the analyse kernels
/// share (`[block_y][block_x][2]` of `i32`).
///
/// `pair_ring` holds every currently-live adjacent-pair field, laid out
/// `[pair_slot][direction][block_y][block_x][2]` of `i32` (direction 0
/// = older→newer, 1 = newer→older), with each direction's slice padded
/// up to a 32-byte (8-`i32`-element) boundary. `dir_len`/`slot_len` are
/// this padded per-direction/per-slot stride in `i32` elements
/// (`MotionCtx::pair_direction_stride`/`pair_slot_stride` on the host
/// side), not the raw `blocks_x * blocks_y * 2` element count, since
/// the host-side write offsets (`pair_byte_offset`) use the padded
/// value too and this kernel's reads must land on the same bytes. The
/// walk starts at `start_pair_slot` and takes `steps` hops, reading
/// direction 0 and incrementing the slot (mod `pair_ring_slots`) when
/// `forward` is true, or reading direction 1 and decrementing the slot
/// otherwise. Consecutive hops land on consecutive pair slots because
/// the pair ring is keyed by the newer frame's position in the push
/// sequence (see `crate::nlmeans::motion::pair_ring_slot_count`), so
/// the caller only ever has to resolve the first hop's slot.
///
/// One thread per output block. Each hop resolves the walking
/// position's current block (nearest block, edge-clamped exactly like
/// `nlm_mc_warp`'s pixel→block lookup), reads that block's motion
/// field from the pair ring, adds it to the running total, and
/// advances the walking position by the same amount before the next
/// hop. The block lookup at every hop reads a clamped copy of the
/// position. The position itself accumulates unclamped, so a long
/// chain can walk outside the frame without corrupting later lookups.
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

/// Zero-fill both directions of one pair-ring slot. Used for duplicated
/// ring slots (stream priming and end-of-stream flush), whose pair
/// field is zero motion by definition since both "frames" are the same
/// content, cheaper and exact compared to running analyse on identical
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

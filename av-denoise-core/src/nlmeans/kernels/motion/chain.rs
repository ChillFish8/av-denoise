use cubecl::prelude::*;
use cubecl::terminate;

/// Chains adjacent-frame motion fields into one vector per block, from the centre frame to a
/// distant neighbour.
///
/// One thread walks one block for `steps` hops from `start_pair_slot`. Each hop reads the motion of
/// the block under the walking position and moves the position by it. Forward walks read
/// direction 0 and step to the next slot, backward walks read direction 1 and step to the previous
/// one. Consecutive hops use consecutive slots because the pair ring is keyed by the newer frame's
/// push order. Only the block lookup is clamped, so a chain can leave the frame without corrupting
/// later lookups.
///
/// `pair_ring` is indexed by slot, direction, block, then component. Direction 0 runs from the
/// older frame to the newer one. `dir_len` and `slot_len` are the padded strides in elements, since
/// each direction is padded to a 32-byte boundary.
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
    let block_col = ABSOLUTE_POS_X;
    let block_row = ABSOLUTE_POS_Y;

    if block_col >= blocks_x || block_row >= blocks_y {
        terminate!();
    }

    let direction = comptime!(if forward { 0u32 } else { 1u32 });

    let mut pos_x = (block_col * step + step / 2) as i32;
    let mut pos_y = (block_row * step + step / 2) as i32;
    let mut acc_x = 0i32;
    let mut acc_y = 0i32;

    for i in 0..steps {
        let slot = if forward {
            (start_pair_slot + i) % pair_ring_slots
        } else {
            (start_pair_slot + pair_ring_slots - i) % pair_ring_slots
        };

        let clamped_x = clamp_i32(pos_x, width as i32) as u32;
        let clamped_y = clamp_i32(pos_y, height as i32) as u32;
        let hop_col = (clamped_x / step).min(blocks_x - 1);
        let hop_row = (clamped_y / step).min(blocks_y - 1);

        let base = slot * slot_len + direction * dir_len + (hop_row * blocks_x + hop_col) * 2;
        let hop_x = pair_ring[base as usize];
        let hop_y = pair_ring[(base + 1) as usize];

        acc_x += hop_x;
        acc_y += hop_y;
        pos_x += hop_x;
        pos_y += hop_y;
    }

    let out_idx = ((block_row * blocks_x + block_col) * 2) as usize;
    mv_field[out_idx] = acc_x;
    mv_field[out_idx + 1] = acc_y;
}

/// Fills both directions of one pair-ring slot with zeroes.
///
/// A duplicated ring slot holds the same frame twice, so its motion is exactly zero.
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

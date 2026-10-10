use cubecl::prelude::*;

use crate::collab::PATCH_SIZE;
use crate::collab::kernels::fused::search::{candidate_partial, pack_reference};
use crate::collab::kernels::fused::tile::tile_partial;
use crate::collab::kernels::group::{clamp_top_left, pack_pos_t};
use crate::collab::kernels::plane_ops::shift_insert8;

/// Candidates scored per batch, one per lane of the group.
const BATCH: u32 = 8;

/// Sums each of 8 per-lane partials across the group and leaves candidate `sub`'s total in lane
/// `sub`.
///
/// Each stage keeps the half of the candidates that matches the lane's bit and swaps the other
/// half with its partner, so 8 sums take 7 shuffles. The stages pair lanes by XOR 1, 2 and 4 in
/// the order [plane_ssd_reduce8](crate::collab::kernels::plane_ops::plane_ssd_reduce8) does, so
/// every total matches it bit for bit.
#[cube]
fn reduce_scatter8(partials: &Array<f32>, sub: u32) -> f32 {
    let odd_lane = (sub & 1u32) == 1u32;
    let mut pairs = Array::<f32>::new(4usize);

    #[unroll]
    for k in 0..4u32 {
        let even = partials[comptime!(2 * k) as usize];
        let odd = partials[comptime!(2 * k + 1) as usize];
        let mine = select(odd_lane, odd, even);
        let other = select(odd_lane, even, odd);
        let received = plane_shuffle_xor(other, 1u32);
        pairs[k as usize] = mine + received;
    }

    let upper_pair = ((sub >> 1u32) & 1u32) == 1u32;
    let mut quads = Array::<f32>::new(2usize);

    #[unroll]
    for m in 0..2u32 {
        let low = pairs[comptime!(2 * m) as usize];
        let high = pairs[comptime!(2 * m + 1) as usize];
        let mine = select(upper_pair, high, low);
        let other = select(upper_pair, low, high);
        let received = plane_shuffle_xor(other, 2u32);
        quads[m as usize] = mine + received;
    }

    let upper_quad = ((sub >> 2u32) & 1u32) == 1u32;
    let mine = select(upper_quad, quads[1usize], quads[0usize]);
    let other = select(upper_quad, quads[0usize], quads[1usize]);
    let received = plane_shuffle_xor(other, 4u32);
    mine + received
}

/// [spatial_search](crate::collab::kernels::fused::search::spatial_search) scored 8 candidates at
/// a time.
///
/// Candidates walk the same rectangle in the same order and insert in that order, so it keeps the
/// same eight slots. Every branch around a shuffle is decided by a warp-wide vote, so it is safe
/// with or without `warp_uniform`.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub(crate) fn spatial_search_batched<S: Float, N: Size>(
    ring: &Array<Vector<f32, N>>,
    search_ring: &Array<Vector<S, N>>,
    tile: &SharedMemory<S>,
    current: &Array<f32>,
    rx: u32,
    ry: u32,
    centre_slot: u32,
    sub: u32,
    base: u32,
    scale: f32,
    best_d: &mut f32,
    best_pos: &mut u32,
    tile_x: u32,
    tile_y: u32,
    #[comptime] warp_uniform: bool,
    #[comptime] f16_search: bool,
    #[comptime] tiled: bool,
    #[comptime] tile_w: u32,
    #[comptime] spatial_radius: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
) -> u32 {
    let max_x = comptime!(width - PATCH_SIZE);
    let max_y = comptime!(height - PATCH_SIZE);
    let reference = pack_reference::<S>(current, f16_search, channels, stored_ch);

    let s_left = clamp_top_left(rx as i32 - spatial_radius as i32, max_x);
    let s_right = clamp_top_left(rx as i32 + spatial_radius as i32, max_x);
    let s_top = clamp_top_left(ry as i32 - spatial_radius as i32, max_y);
    let s_bot = clamp_top_left(ry as i32 + spatial_radius as i32, max_y);
    let rect_w = s_right - s_left + 1u32;
    let rect_h = s_bot - s_top + 1u32;

    // The uniform walk covers the unclipped span so every group in the warp takes the same turns.
    let span = comptime!(2 * spatial_radius + 1);
    let mut walk_w = rect_w;
    let mut walk_count = rect_w * rect_h;
    if comptime!(warp_uniform) {
        walk_w = span;
        walk_count = comptime!(span * span);
    }

    let batches = walk_count.div_ceil(BATCH);
    let mut batch = 0u32;
    while batch < batches {
        let mut partials = Array::<f32>::new(BATCH as usize);

        #[unroll]
        for j in 0..BATCH {
            let walk_index = batch * BATCH + j;
            let candidate_x = u32::min(s_left + walk_index % walk_w, s_right);
            let candidate_y = u32::min(s_top + walk_index / walk_w, s_bot);

            if comptime!(tiled) {
                let column = candidate_x - tile_x + sub;
                let row = candidate_y - tile_y;
                partials[j as usize] = tile_partial(
                    tile, current, &reference, column, row, f16_search, tile_w, channels, stored_ch,
                );
            } else {
                partials[j as usize] = candidate_partial(
                    ring,
                    search_ring,
                    current,
                    &reference,
                    candidate_x,
                    candidate_y,
                    centre_slot,
                    sub,
                    f16_search,
                    width,
                    height,
                    channels,
                    stored_ch,
                );
            }
        }

        let total = reduce_scatter8(&partials, sub);

        // Lane `sub` now owns candidate `sub` of the batch.
        let walk_index = batch * BATCH + sub;
        let wanted_x = s_left + walk_index % walk_w;
        let wanted_y = s_top + walk_index / walk_w;
        let live = walk_index < walk_count && wanted_x <= s_right && wanted_y <= s_bot;
        let candidate_x = u32::min(wanted_x, s_right);
        let candidate_y = u32::min(wanted_y, s_bot);
        let packed = pack_pos_t(candidate_x, candidate_y, 0u32);

        let mut dist = select(live, total * scale, 3.0e38f32);
        if live && candidate_x == rx && candidate_y == ry {
            dist = -1.0e38f32;
        }

        let worst = plane_shuffle(*best_d, base + 7u32);
        if plane_any(dist < worst) {
            #[unroll]
            for j in 0..BATCH {
                let candidate_d = plane_shuffle(dist, base + j);
                let candidate_pos = plane_shuffle(packed, base + j);
                let current_worst = plane_shuffle(*best_d, base + 7u32);
                if plane_any(candidate_d < current_worst) {
                    shift_insert8(best_d, best_pos, candidate_d, candidate_pos, sub);
                }
            }
        }

        batch += 1u32;
    }

    rect_w * rect_h
}

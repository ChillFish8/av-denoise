use cubecl::prelude::*;

use crate::collab::PATCH_SIZE;
use crate::collab::kernels::plane_ops::plane_ssd_reduce8;
use crate::nlmeans::kernels::helpers::read_line;

/// Packs the reference column into `S` pairs for [packed_partial].
///
/// A single stored channel pairs row `r` with row `r + 4`. Wider storage pairs neighbouring
/// channels within each row, and lanes past `channels` stay zero. Without `f16_search` it returns
/// one unfilled pair.
#[cube]
pub(crate) fn pack_reference<S: Float>(
    current: &Array<f32>,
    #[comptime] f16_search: bool,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
) -> Array<Vector<S, Const<2>>> {
    let pair_count = comptime!(
        if !f16_search {
            1
        } else if stored_ch == 1 {
            PATCH_SIZE / 2
        } else {
            PATCH_SIZE * stored_ch / 2
        }
    );
    let mut packed = Array::<Vector<S, Const<2>>>::new(pair_count as usize);

    if comptime!(f16_search && stored_ch == 1) {
        #[unroll]
        for r in 0..comptime!(PATCH_SIZE / 2) {
            let mut pair = Vector::<f32, Const<2>>::new(0.0f32);
            pair[0] = current[r as usize];
            pair[1] = current[comptime!(r + PATCH_SIZE / 2) as usize];
            packed[r as usize] = Vector::<S, Const<2>>::cast_from(pair);
        }
    } else if comptime!(f16_search) {
        let pairs_per_row = comptime!(stored_ch / 2);
        #[unroll]
        for r in 0..PATCH_SIZE {
            #[unroll]
            for p in 0..pairs_per_row {
                let first = comptime!(2 * p);
                let second = comptime!(2 * p + 1);
                let mut pair = Vector::<f32, Const<2>>::new(0.0f32);
                if comptime!(first < channels) {
                    pair[0] = current[comptime!(r * channels + first) as usize];
                }
                if comptime!(second < channels) {
                    pair[1] = current[comptime!(r * channels + second) as usize];
                }
                packed[comptime!(r * pairs_per_row + p) as usize] = Vector::<S, Const<2>>::cast_from(pair);
            }
        }
    }

    packed
}

/// One lane's squared difference against the candidate column in the search ring.
///
/// Both sides pair up as [pack_reference] lays them out, so each packed instruction covers two
/// values in the ring's precision `S`. The two lanes of the sum widen to f32 before they add.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the read needs"
)]
pub(crate) fn packed_partial<S: Float, N: Size>(
    search_ring: &Array<Vector<S, N>>,
    reference: &Array<Vector<S, Const<2>>>,
    x: u32,
    y: u32,
    slot: u32,
    sub: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] stored_ch: u32,
) -> f32 {
    let mut sum = Vector::<S, Const<2>>::new(S::new(0.0f32));

    if comptime!(stored_ch == 1) {
        #[unroll]
        for r in 0..comptime!(PATCH_SIZE / 2) {
            let top = read_line(search_ring, x + sub, y + r, slot, width, height);
            let bottom_row = y + r + comptime!(PATCH_SIZE / 2);
            let bottom = read_line(search_ring, x + sub, bottom_row, slot, width, height);
            let mut pixel = Vector::<S, Const<2>>::empty();
            pixel[0] = top[0];
            pixel[1] = bottom[0];
            let diff = reference[r as usize] - pixel;
            sum += diff * diff;
        }
    } else {
        let pairs_per_row = comptime!(stored_ch / 2);
        #[unroll]
        for r in 0..PATCH_SIZE {
            let line = read_line(search_ring, x + sub, y + r, slot, width, height);
            #[unroll]
            for p in 0..pairs_per_row {
                let mut pixel = Vector::<S, Const<2>>::empty();
                pixel[0] = line[comptime!(2 * p) as usize];
                pixel[1] = line[comptime!(2 * p + 1) as usize];
                let diff = reference[comptime!(r * pairs_per_row + p) as usize] - pixel;
                sum += diff * diff;
            }
        }
    }

    let low = f32::cast_from(sum[0]);
    let high = f32::cast_from(sum[1]);
    low + high
}

/// One lane's share of the distance from the reference patch to the candidate with top-left
/// `(x, y)` in frame `slot`.
///
/// The lane sums its own column. With `f16_search` the column comes from `search_ring` and is scored
/// by [packed_partial].
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the read needs"
)]
pub(crate) fn candidate_partial<S: Float, N: Size>(
    ring: &Array<Vector<f32, N>>,
    search_ring: &Array<Vector<S, N>>,
    current: &Array<f32>,
    reference: &Array<Vector<S, Const<2>>>,
    x: u32,
    y: u32,
    slot: u32,
    sub: u32,
    #[comptime] f16_search: bool,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
) -> f32 {
    let mut partial = 0.0f32;

    if comptime!(f16_search) {
        partial = packed_partial(search_ring, reference, x, y, slot, sub, width, height, stored_ch);
    } else {
        #[unroll]
        for r in 0..PATCH_SIZE {
            let pixel = read_line(ring, x + sub, y + r, slot, width, height);
            #[unroll]
            for c in 0..channels {
                let diff = current[(r * channels + c) as usize] - pixel[c as usize];
                partial += diff * diff;
            }
        }
    }

    partial
}

/// The distance from the reference patch to the candidate with top-left `(x, y)` in frame `slot`.
///
/// Each lane sums its own column and `plane_ssd_reduce8` completes the distance with shuffles, so
/// every lane of the group must call this, even when the result is discarded.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the read needs"
)]
pub(crate) fn candidate_distance<S: Float, N: Size>(
    ring: &Array<Vector<f32, N>>,
    search_ring: &Array<Vector<S, N>>,
    current: &Array<f32>,
    reference: &Array<Vector<S, Const<2>>>,
    x: u32,
    y: u32,
    slot: u32,
    sub: u32,
    scale: f32,
    #[comptime] f16_search: bool,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
) -> f32 {
    let partial = candidate_partial(
        ring,
        search_ring,
        current,
        reference,
        x,
        y,
        slot,
        sub,
        f16_search,
        width,
        height,
        channels,
        stored_ch,
    );
    let distance = plane_ssd_reduce8(partial);

    distance * scale
}

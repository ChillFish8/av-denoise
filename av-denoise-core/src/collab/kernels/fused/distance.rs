use cubecl::prelude::*;

use super::shape::SearchShape;
use crate::collab::PATCH_SIZE;
use crate::collab::kernels::plane_ops::plane_ssd_reduce8;
use crate::engine::kernels::read_line;

/// Packs the reference column into `S` pairs for [packed_partial].
///
/// A single stored channel pairs row `r` with row `r + 4`. Wider storage pairs neighbouring
/// channels within each row, and lanes past `channels` stay zero. Without `f16_search` it returns
/// one unfilled pair.
#[cube]
pub(crate) fn pack_reference<S: Float>(
    current: &[f32],
    #[comptime] shape: SearchShape,
) -> Array<Vector<S, Const<2>>> {
    let f16_search = comptime!(shape.f16_search);
    let channels = comptime!(shape.channels);
    let stored_ch = comptime!(shape.stored_ch);
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
            pair.insert(0usize, current[r as usize]);
            pair.insert(1usize, current[comptime!(r + PATCH_SIZE / 2) as usize]);
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
                    pair.insert(0usize, current[comptime!(r * channels + first) as usize]);
                }
                if comptime!(second < channels) {
                    pair.insert(1usize, current[comptime!(r * channels + second) as usize]);
                }
                packed[comptime!(r * pairs_per_row + p) as usize] = Vector::<S, Const<2>>::cast_from(pair);
            }
        }
    }

    packed
}

/// The squared difference between reference pair `pair` and the candidate values `first` and
/// `second`, in the ring's precision `S`.
#[cube]
pub(crate) fn pair_squared_difference<S: Float>(
    reference: &[Vector<S, Const<2>>],
    first: S,
    second: S,
    #[comptime] pair: u32,
) -> Vector<S, Const<2>> {
    let mut pixel = Vector::<S, Const<2>>::empty();
    pixel.insert(0usize, first);
    pixel.insert(1usize, second);
    let diff = reference[pair as usize] - pixel;
    diff * diff
}

/// Widens both lanes of a pair sum to f32 and adds them.
#[cube]
pub(crate) fn pair_sum_total<S: Float>(sum: Vector<S, Const<2>>) -> f32 {
    let low_lane = sum.extract(0usize);
    let high_lane = sum.extract(1usize);
    let low = f32::cast_from(low_lane);
    let high = f32::cast_from(high_lane);
    low + high
}

/// One lane's squared difference against the candidate column in the search ring.
///
/// Both sides pair up as [pack_reference] lays them out, so each packed instruction covers two
/// values in the ring's precision `S`. The two lanes of the sum widen to f32 before they add.
#[cube]
pub(crate) fn packed_partial<S: Float, N: Size>(
    search_ring: &[Vector<S, N>],
    reference: &[Vector<S, Const<2>>],
    x: u32,
    y: u32,
    slot: u32,
    sub: u32,
    #[comptime] shape: SearchShape,
) -> f32 {
    let width = comptime!(shape.width);
    let height = comptime!(shape.height);
    let stored_ch = comptime!(shape.stored_ch);
    let mut sum = Vector::<S, Const<2>>::new(S::new(0.0f32));

    if comptime!(stored_ch == 1) {
        #[unroll]
        for r in 0..comptime!(PATCH_SIZE / 2) {
            let top = read_line(search_ring, x + sub, y + r, slot, width, height);
            let bottom_row = y + r + comptime!(PATCH_SIZE / 2);
            let bottom = read_line(search_ring, x + sub, bottom_row, slot, width, height);
            let top_lane = top.extract(0usize);
            let bottom_lane = bottom.extract(0usize);
            let squared = pair_squared_difference(reference, top_lane, bottom_lane, r);
            sum += squared;
        }
    } else {
        let pairs_per_row = comptime!(stored_ch / 2);
        #[unroll]
        for r in 0..PATCH_SIZE {
            let line = read_line(search_ring, x + sub, y + r, slot, width, height);
            #[unroll]
            for p in 0..pairs_per_row {
                let first = line.extract(comptime!(2 * p) as usize);
                let second = line.extract(comptime!(2 * p + 1) as usize);
                let pair = comptime!(r * pairs_per_row + p);
                let squared = pair_squared_difference(reference, first, second, pair);
                sum += squared;
            }
        }
    }

    pair_sum_total(sum)
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
    ring: &[Vector<f32, N>],
    search_ring: &[Vector<S, N>],
    current: &[f32],
    reference: &[Vector<S, Const<2>>],
    x: u32,
    y: u32,
    slot: u32,
    sub: u32,
    #[comptime] shape: SearchShape,
) -> f32 {
    let width = comptime!(shape.width);
    let height = comptime!(shape.height);
    let channels = comptime!(shape.channels);
    let f16_search = comptime!(shape.f16_search);
    let mut partial = 0.0f32;

    if f16_search {
        partial = packed_partial(search_ring, reference, x, y, slot, sub, shape);
    } else {
        #[unroll]
        for r in 0..PATCH_SIZE {
            let pixel = read_line(ring, x + sub, y + r, slot, width, height);
            #[unroll]
            for c in 0..channels {
                let diff = current[(r * channels + c) as usize] - pixel.extract(c as usize);
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
    ring: &[Vector<f32, N>],
    search_ring: &[Vector<S, N>],
    current: &[f32],
    reference: &[Vector<S, Const<2>>],
    x: u32,
    y: u32,
    slot: u32,
    sub: u32,
    scale: f32,
    #[comptime] shape: SearchShape,
) -> f32 {
    let partial = candidate_partial(ring, search_ring, current, reference, x, y, slot, sub, shape);
    let distance = plane_ssd_reduce8(partial);

    distance * scale
}

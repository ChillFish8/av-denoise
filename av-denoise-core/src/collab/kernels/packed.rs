use cubecl::prelude::*;

use super::aggregate::WEIGHT_GAIN;

/// The value of one unit of the high word, `2^32`.
pub const PACK_UNIT: i64 = 1 << 32;

/// The packed word for one contribution, `accum_fixed * 2^32 + weight_fixed`.
///
/// `weight_fixed` is never negative and a region's weight total stays below `2^31`, so the low
/// word never carries into the high word. A negative `accum_fixed` makes the word negative, and a
/// signed 64-bit add keeps the high word the exact sum.
#[cube]
pub fn pack_accum<W: Int>(accum_fixed: i32, weight_fixed: i32) -> W {
    let high = W::cast_from(accum_fixed);
    let low = W::cast_from(weight_fixed);
    let unit = W::new(PACK_UNIT);

    high * unit + low
}

/// The weight total a packed word holds, its low 32 bits.
#[cube]
pub fn unpack_weight(word: i64) -> i32 {
    let low = word & 0xFFFF_FFFFi64;
    i32::cast_from(low)
}

/// The accumulator total a packed word holds.
///
/// The word is `accum * 2^32 + weight` with `weight` between 0 and `2^32`, so removing the low word
/// leaves an exact multiple of `2^32`.
#[cube]
pub fn unpack_accum(word: i64) -> i32 {
    let low = word & 0xFFFF_FFFFi64;
    let high = (word - low) / PACK_UNIT;
    i32::cast_from(high)
}

/// Host mirror of `pack_accum`.
pub fn pack_accum_host(accum_fixed: i32, weight_fixed: i32) -> i64 {
    i64::from(accum_fixed) * PACK_UNIT + i64::from(weight_fixed)
}

/// Host mirror of `unpack_weight`.
pub fn unpack_weight_host(word: i64) -> i32 {
    (word & 0xFFFF_FFFF) as i32
}

/// Host mirror of `unpack_accum`.
pub fn unpack_accum_host(word: i64) -> i32 {
    let low = word & 0xFFFF_FFFF;
    ((word - low) / PACK_UNIT) as i32
}

/// One channel's weighted mean, with the arithmetic of
/// [collab_normalise](super::aggregate::collab_normalise).
#[cube]
fn normalised(accum_fixed: i32, weight_total: i32) -> f32 {
    let accumulated = accum_fixed as f32;
    let mut value = accumulated;
    if weight_total != 0i32 {
        value = accumulated * WEIGHT_GAIN / (weight_total as f32);
    }

    value
}

/// [collab_normalise](super::aggregate::collab_normalise) for the packed layout.
///
/// Channel 0 and the weight come from `packed`, the other channels from `rest` at
/// `pixel * (stored_ch - 1) + channel - 1`. The region is written back as zeros for its next use.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub fn collab_normalise_packed<N: Size>(
    packed: &mut Array<i64>,
    rest: &mut Array<i32>,
    output: &mut Array<Vector<f32, N>>,
    frame_offset: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
) {
    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    if x >= width || y >= height {
        terminate!();
    }

    let rest_channels = comptime!(stored_ch - 1);
    let local_pixel = y * width + x;
    let pixel = frame_offset + local_pixel;
    let word = packed[pixel as usize];
    let weight_total = unpack_weight(word);
    let first_accum = unpack_accum(word);

    let mut pixel_out = Vector::<f32, N>::empty();
    pixel_out[0usize] = normalised(first_accum, weight_total);

    #[unroll]
    for channel in 1..channels {
        let rest_index = pixel * rest_channels + channel - 1u32;
        let accumulated = rest[rest_index as usize];
        pixel_out[channel as usize] = normalised(accumulated, weight_total);
    }

    output[local_pixel as usize] = pixel_out;

    // A later frame reuses this region and expects it empty, padding lanes included.
    packed[pixel as usize] = 0i64;

    #[unroll]
    for channel in 0..rest_channels {
        rest[(pixel * rest_channels + channel) as usize] = 0i32;
    }
}

/// Zeroes `pixels` pixels of both packed buffers, starting `frame_offset` pixels in.
///
/// [collab_zero_accum](super::aggregate::collab_zero_accum) for the packed layout. The loop strides
/// so the grid stays under the workgroup limit.
#[cube(launch_unchecked)]
#[expect(
    clippy::collapsible_if,
    reason = "the outer condition is comptime, so luma compiles no access to the other channels"
)]
pub fn collab_zero_packed(
    packed: &mut Array<Atomic<i64>>,
    rest: &mut Array<Atomic<i32>>,
    frame_offset: u32,
    #[comptime] pixels: u32,
    #[comptime] rest_channels: u32,
    #[comptime] total_threads: u32,
) {
    let rest_span = comptime!(pixels * rest_channels);
    let span = comptime!(u32::max(pixels, rest_span));
    let mut idx = ABSOLUTE_POS_X;

    while idx < span {
        if idx < pixels {
            Atomic::store(&packed[(frame_offset + idx) as usize], 0i64);
        }

        if comptime!(rest_channels > 0) {
            if idx < rest_span {
                Atomic::store(&rest[(frame_offset * rest_channels + idx) as usize], 0i32);
            }
        }

        idx += total_threads;
    }
}

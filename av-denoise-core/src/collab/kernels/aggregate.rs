use cubecl::prelude::*;

use super::transforms::RECIPROCAL_FLOOR;
use crate::collab::{MAX_K, PATCH_AREA, PATCH_SIZE, STEP};

/// The fixed-point scale a single-frame accumulator counts in.
///
/// The accumulators hold fixed-point integers because integer atomics are much faster than float
/// atomics on most GPUs. `2^19` is the largest power of two that keeps the worst case well inside
/// `i32`. One pass covers a pixel with at most 392 member patches (49 step-grid references times
/// `MAX_K`), each weighted by at most 1 and clamped to [ACCUM_CLAMP], so the accumulator peaks
/// near `1.03e9`, about half of `i32::MAX`. One unit is `1.9e-6`, well under the `2.4e-4` one
/// 12-bit code level spans. `wsum` peaks at the same place because [WEIGHT_GAIN] trades the
/// weight's smaller bound for exactly that much scale.
///
/// A cross-frame ring uses [cross_frame_accum_scale] instead. Either scale cancels in
/// `collab_normalise`, which divides one accumulator by the other.
pub const ACCUM_SCALE: f32 = 524_288.0;

/// The headroom `cross_frame_accum_scale` leaves under `i32::MAX`, matching the two-fold margin
/// of [ACCUM_SCALE].
const CROSS_FRAME_SAFETY_FACTOR: f64 = 2.0;

/// The fixed-point scale for the cross-frame accumulator ring, in place of [ACCUM_SCALE].
///
/// Large radii push the worst case past `i32::MAX`, and one constant small enough for them would
/// waste precision at common radii, so the scale is derived per configuration. Along one axis at
/// most `((PATCH_SIZE - 1) + 2 * spatial_radius) / STEP + 1` step-grid references can reach a
/// pixel, and each brings `MAX_K` members. A region collects up to `4 * temporal_radius + 1`
/// passes, the steady `2 * temporal_radius + 1` plus `temporal_radius` head and tail passes when a
/// short scene puts one frame in both edge rings.
///
/// Each contribution is weighted by at most 1 and clamped to [ACCUM_CLAMP], and `wsum` fits the
/// same budget through [WEIGHT_CLAMP] and [WEIGHT_GAIN]. The result is the largest power of two
/// that keeps the worst case under half of `i32::MAX`.
pub fn cross_frame_accum_scale(spatial_radius: u32, temporal_radius: u32) -> f32 {
    let refs_per_axis = ((PATCH_SIZE - 1) + 2 * spatial_radius) / STEP + 1;
    let contribs_per_pass = refs_per_axis as f64 * refs_per_axis as f64 * MAX_K as f64;
    let passes = (4 * temporal_radius + 1) as f64;
    let max_raw_value = contribs_per_pass * passes * ACCUM_CLAMP as f64;

    let budget = i32::MAX as f64 / CROSS_FRAME_SAFETY_FACTOR;
    let exponent = (budget / max_raw_value).log2().floor();
    let scale = 2f64.powf(exponent);

    debug_assert!(
        scale * max_raw_value <= budget,
        "cross_frame_accum_scale({spatial_radius}, {temporal_radius}) picked {scale}, which \
         does not keep the worst-case accumulator value under the safety budget",
    );

    scale as f32
}

/// The magnitude a filtered value is clamped to before it enters the accumulator.
///
/// The filter shrinks coefficients of input already between 0 and 1, so a value this large means
/// the filter has gone wrong. The clamp makes the accumulator saturate instead of overflowing
/// `i32` into a wildly wrong pixel.
pub const ACCUM_CLAMP: f32 = 5.0;

/// The constant every group weight is multiplied by so it lands in a fixed-point-friendly band.
///
/// A group weight is `1 / sum(retained coefficient variance)`, which tracks `1 / sigma^2` and
/// spans too wide a range for fixed point. Aggregation computes `sum(w * x) / sum(w)`, so a factor
/// common to every weight cancels exactly.
///
/// The filter always keeps the group DC, whose variance `sigma^2 * g_max^2` is the smallest
/// retained sum, so dividing by it bounds the weight above by 1, the bound [WEIGHT_CLAMP] relies
/// on. A group of 512 coefficients sums to at most `512 * sigma^2 * g_max^2`, so the weight lies
/// between `1/512` and 1. Below [RECIPROCAL_FLOOR] every weight saturates at
/// `1 / RECIPROCAL_FLOOR`, so taking the larger of the two keeps the bound of 1.
pub fn weight_scale(sigma: f32, dct_profile: &[f32; 8]) -> f32 {
    let g_max = dct_profile.iter().copied().fold(0.0f32, f32::max);
    let norm = sigma * sigma * g_max * g_max;
    if norm.is_finite() && norm > RECIPROCAL_FLOOR {
        norm
    } else {
        RECIPROCAL_FLOOR
    }
}

/// The zeroth-order modified Bessel function of the first kind, `sum_k ((x / 2)^k / k!)^2`.
///
/// The series converges to `f64` precision well inside 32 terms for every `beta` that
/// [kaiser_window] accepts.
fn bessel_i0(x: f64) -> f64 {
    let half = x / 2.0;
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    for k in 1..32 {
        term *= half / k as f64;
        sum += term * term;
    }
    sum
}

/// The separable 8-tap Kaiser window `scatter_patch` tapers a patch's contribution with.
///
/// Weighting a patch's edges less than its centre blends the threshold decisions of the many
/// patches covering a pixel, as BM3D's aggregation window does. Pixel `(i, j)` takes
/// `w[i] * w[j]`, with `w[i] = I0(beta * sqrt(1 - (2i / 7 - 1)^2)) / I0(beta)` so the peak is 1.
/// Larger `beta` tapers harder, BM3D uses 2.0, and `beta = 0` returns exactly all ones.
///
/// Every tap is above 0 and at most 1, so the accumulator bounds behind [ACCUM_SCALE] and
/// [cross_frame_accum_scale] still hold. The smallest weight shrinks by `w[0]^2`, `0.193` at
/// `beta = 2`, which leaves the weight floor around 124 units of `wsum` at the default geometry,
/// still above rounding. Only the geometry moves that floor, not match quality.
pub fn kaiser_window(beta: f32) -> [f32; PATCH_SIZE as usize] {
    let denom = bessel_i0(beta as f64);
    let last = (PATCH_SIZE - 1) as f64;
    let mut window = [0.0f32; PATCH_SIZE as usize];
    for (i, tap) in window.iter_mut().enumerate() {
        let position = 2.0 * i as f64 / last - 1.0;
        let radius = (1.0 - position * position).sqrt();
        let numerator = bessel_i0(beta as f64 * radius);
        *tap = (numerator / denom) as f32;
    }
    window
}

/// The magnitude a group weight is clamped to before it enters `wsum`.
///
/// [weight_scale] bounds a normalised weight by 1, a fifth of [ACCUM_CLAMP], and [WEIGHT_GAIN]
/// turns that difference into resolution.
pub const WEIGHT_CLAMP: f32 = 1.0;

/// The extra fixed-point resolution `wsum` gets over `accum`.
///
/// A weight is bounded by [WEIGHT_CLAMP] and a value by the larger [ACCUM_CLAMP], so counting
/// weights at this multiple spends the same `i32` budget while resolving them this many times
/// finer. A weight can fall to `1/512`, and one below half a unit contributes nothing, so the gain
/// keeps poorly matched groups above that point. `collab_normalise` multiplies it back out.
pub const WEIGHT_GAIN: f32 = ACCUM_CLAMP / WEIGHT_CLAMP;

/// Converts one weighted value into fixed point at `scale`.
///
/// `scale` is [ACCUM_SCALE] or a [cross_frame_accum_scale] result. It rounds rather than
/// truncates, because filtered values are never negative and truncation would pull the weighted
/// mean toward black where weights are smallest.
#[cube]
pub fn to_fixed(value: f32, scale: f32) -> i32 {
    let clamped = f32::clamp(value, -ACCUM_CLAMP, ACCUM_CLAMP);
    f32::round(clamped * scale) as i32
}

/// Converts one group weight into `wsum`'s fixed point, at [WEIGHT_GAIN] times `scale`.
///
/// It rounds for the same reason as `to_fixed`.
#[cube]
pub fn to_fixed_weight(weight: f32, scale: f32) -> i32 {
    let clamped = f32::clamp(weight, 0.0f32, WEIGHT_CLAMP);
    f32::round(clamped * scale * WEIGHT_GAIN) as i32
}

/// Adds this thread's pixel of one filtered patch to the accumulators.
///
/// Each of the cube's 64 threads owns the patch pixel `tid` picks, so one call per member scatters
/// the whole patch. `kaiser` holds [kaiser_window]'s 8 taps, and eight ones disable the taper.
/// `accum` and `wsum` hold one `frame_pixels` region per frame in ring-slot order, and
/// `frame_slot` picks this member's region.
///
/// `write_weight` adds the weight to `wsum`, and only the first channel's pass sets it because
/// aggregation needs one weight per patch. `accum_scale` is [ACCUM_SCALE] or a
/// [cross_frame_accum_scale] result, and `wsum` counts at [WEIGHT_GAIN] times it.
#[cube]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub fn scatter_patch(
    accum: &mut Array<Atomic<i32>>,
    wsum: &mut Array<Atomic<i32>>,
    kaiser: &Array<f32>,
    value: f32,
    weight: f32,
    patch_x: u32,
    patch_y: u32,
    tid: u32,
    write_weight: bool,
    #[comptime] channel: u32,
    #[comptime] width: u32,
    #[comptime] stored_ch: u32,
    frame_slot: u32,
    #[comptime] frame_pixels: u32,
    accum_scale: f32,
) {
    let row = tid / PATCH_SIZE;
    let col = tid % PATCH_SIZE;
    let local_pixel = (patch_y + row) * width + patch_x + col;
    let pixel = frame_slot * frame_pixels + local_pixel;

    // The window scales the value and the weight alike, so it cancels where coverage is uniform
    // and only reweights the blend where it is not.
    let window = kaiser[row as usize] * kaiser[col as usize];
    let weight = weight * window;
    Atomic::fetch_add(
        &accum[(pixel * stored_ch + channel) as usize],
        to_fixed(value * weight, accum_scale),
    );
    if write_weight {
        Atomic::fetch_add(&wsum[pixel as usize], to_fixed_weight(weight, accum_scale));
    }
}

/// Zeroes `pixels` pixels of both accumulators, starting `frame_offset` pixels in.
///
/// `client.empty` memory is not zeroed, so the accumulators must be cleared before a pass, and a
/// reused ring region must be cleared again before a later pass writes into it. The
/// `accum` region is `pixels * stored_ch` slots and `wsum`'s is `pixels`, so the weight write is
/// masked past its end. The loop is strided so the grid stays under the 65,535 workgroup limit,
/// which a 4K or 8K frame exceeds at 256 threads per block.
#[cube(launch_unchecked)]
pub fn collab_zero_accum(
    accum: &mut Array<Atomic<i32>>,
    wsum: &mut Array<Atomic<i32>>,
    frame_offset: u32,
    #[comptime] pixels: u32,
    #[comptime] stored_ch: u32,
    #[comptime] total_threads: u32,
) {
    let mut idx = ABSOLUTE_POS_X;
    while idx < pixels * stored_ch {
        Atomic::store(&accum[(frame_offset * stored_ch + idx) as usize], 0i32);
        if idx < pixels {
            Atomic::store(&wsum[(frame_offset + idx) as usize], 0i32);
        }
        idx += total_threads;
    }
}

/// Writes `accum / wsum` for one frame's region of the accumulators into `output`.
///
/// Each pixel becomes the weighted mean of every filtered patch covering it. The fixed-point scale
/// cancels, so only [WEIGHT_GAIN] is multiplied back. `frame_offset` (in pixels) picks the region,
/// and `output` is one frame wide.
///
/// The weight sum is never zero, because the references cover every pixel between one and nine
/// times and [WEIGHT_GAIN] keeps every group weight from rounding to nothing. A zero sum returns
/// the accumulator untouched rather than a NaN.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer or comptime shape the kernel binds"
)]
pub fn collab_normalise<N: Size>(
    accum: &Array<i32>,
    wsum: &Array<i32>,
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

    let local_pixel = y * width + x;
    let pixel = frame_offset + local_pixel;
    let weight_total = wsum[pixel as usize];

    let mut pixel_out = Vector::<f32, N>::empty();
    #[unroll]
    for channel in 0..channels {
        let accumulated = accum[(pixel * stored_ch + channel) as usize] as f32;
        let mut value = accumulated;
        if weight_total != 0i32 {
            value = accumulated * WEIGHT_GAIN / (weight_total as f32);
        }
        pixel_out[channel as usize] = value;
    }
    output[local_pixel as usize] = pixel_out;
}

// Recheck ACCUM_SCALE's headroom if the patch area or group size moves, since both raise the
// number of contributions one pass adds to a pixel.
const _: () = assert!(
    PATCH_AREA == 64 && MAX_K == 8,
    "recheck ACCUM_SCALE's headroom, the per-pass contribution bound moved"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl4d::MAX_KAISER_BETA;

    #[test]
    fn to_fixed_rounds_rather_than_truncating() {
        let rounded_up = to_fixed(0.6, 1.0);
        let rounded_down = to_fixed(0.4, 1.0);
        let negative = to_fixed(-0.6, 1.0);

        assert_eq!(rounded_up, 1);
        assert_eq!(rounded_down, 0);
        assert_eq!(negative, -1);
    }

    #[test]
    fn to_fixed_weight_resolves_finer_than_to_fixed() {
        let scale = 65_536.0f32;
        let weight = 0.3 / scale;
        let as_value = to_fixed(weight, scale);
        let as_weight = to_fixed_weight(weight, scale);

        assert_eq!(as_value, 0);
        assert_eq!(as_weight, 2);
    }

    #[test]
    fn to_fixed_weight_clamps_at_one() {
        let scale = 1_024.0f32;
        let above = to_fixed_weight(4.0, scale);
        let at_clamp = to_fixed_weight(WEIGHT_CLAMP, scale);
        let negative = to_fixed_weight(-1.0, scale);

        assert_eq!(above, at_clamp);
        assert_eq!(negative, 0);
    }

    // The ranges `Nl4dParams::validate` enforces, as plain numbers so the test keeps covering the
    // true worst case if those ranges narrow.
    const SPATIAL_RADIUS_RANGE: std::ops::RangeInclusive<u32> = 1..=16;
    const TEMPORAL_RADIUS_RANGE: std::ops::RangeInclusive<u32> = 1..=8;

    #[test]
    fn every_spatial_and_temporal_radius_stays_under_the_safety_budget() {
        let budget = i32::MAX as f64 / CROSS_FRAME_SAFETY_FACTOR;

        for spatial_radius in SPATIAL_RADIUS_RANGE {
            for temporal_radius in TEMPORAL_RADIUS_RANGE {
                let refs_per_axis = ((PATCH_SIZE - 1) + 2 * spatial_radius) / STEP + 1;
                let contribs_per_pass = refs_per_axis as f64 * refs_per_axis as f64 * MAX_K as f64;
                let passes = (4 * temporal_radius + 1) as f64;
                let max_raw_value = contribs_per_pass * passes * ACCUM_CLAMP as f64;

                let scale = cross_frame_accum_scale(spatial_radius, temporal_radius) as f64;
                let worst_case_value = max_raw_value * scale;

                assert!(
                    worst_case_value <= budget,
                    "spatial_radius={spatial_radius} temporal_radius={temporal_radius}: \
                     scale={scale} gives worst-case value {worst_case_value}, over the \
                     budget {budget}",
                );
                assert!(
                    scale > 0.0 && scale.is_finite(),
                    "spatial_radius={spatial_radius} temporal_radius={temporal_radius}: \
                     scale={scale} is not a usable fixed-point scale",
                );
            }
        }
    }

    #[test]
    fn defaults_keep_at_least_a_2_15_scale() {
        let floor = 32_768.0f32;
        let derived = cross_frame_accum_scale(9, 2);

        assert!(
            derived >= floor,
            "derived scale {derived} at the defaults (spatial_radius=9, temporal_radius=2) \
             should be at least {floor}",
        );
    }

    #[test]
    fn kaiser_window_at_beta_zero_is_exactly_one_everywhere() {
        let window = kaiser_window(0.0);
        assert_eq!(window, [1.0f32; PATCH_SIZE as usize]);
    }

    /// An even tap count puts the centre between taps 3 and 4, so the rise is checked up to that
    /// pair.
    #[test]
    fn kaiser_window_is_symmetric_and_rises_to_the_centre() {
        for beta in [1.0f32, 2.0, 4.0, MAX_KAISER_BETA] {
            let window = kaiser_window(beta);
            for i in 0..4 {
                assert!(
                    (window[i] - window[7 - i]).abs() < 1e-6,
                    "beta {beta}: tap {i} is {} and its mirror {}",
                    window[i],
                    window[7 - i],
                );

                if i < 3 {
                    assert!(
                        window[i + 1] > window[i],
                        "beta {beta}: tap {} is not above tap {i}",
                        i + 1,
                    );
                }
            }

            assert!(
                window.iter().all(|&tap| tap > 0.0 && tap <= 1.0),
                "beta {beta}: a tap is not above zero and at most 1, {window:?}",
            );
        }
    }

    #[test]
    fn kaiser_window_end_taps_are_the_bessel_ratio() {
        for beta in [1.0f32, 2.0, 4.0] {
            let window = kaiser_window(beta);
            let expected = (1.0 / bessel_i0(beta as f64)) as f32;
            assert!(
                (window[0] - expected).abs() < 1e-6,
                "beta {beta}: end tap {} against the ratio {expected}",
                window[0],
            );
            assert!((window[7] - expected).abs() < 1e-6);
        }

        // Pins the end tap behind the `w[0]^2 = 0.193` at `beta = 2` quoted in `kaiser_window`'s doc.
        let beta_two_window = kaiser_window(2.0);
        assert!((beta_two_window[0] - 0.4388).abs() < 1e-3);
    }

    #[test]
    fn contribution_model_reproduces_the_documented_392_at_the_default_spatial_radius() {
        let spatial_radius = 9u32;
        let refs_per_axis = ((PATCH_SIZE - 1) + 2 * spatial_radius) / STEP + 1;
        assert_eq!(refs_per_axis, 7);
        assert_eq!(refs_per_axis * refs_per_axis * MAX_K, 392);
    }
}

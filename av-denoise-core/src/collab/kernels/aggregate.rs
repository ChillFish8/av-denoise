use cubecl::prelude::*;

use crate::collab::kernels::transforms::RECIPROCAL_FLOOR;
use crate::collab::{MAX_K, PATCH_AREA, PATCH_SIZE, STEP};

/// The fixed-point scale a single-frame accumulator counts in.
///
/// Aggregation adds one weighted value per covering patch, one atomic add
/// each. Integer atomics are much faster than float atomics on most GPUs,
/// so the accumulators hold fixed-point integers.
///
/// `2^19` is the largest power of two that keeps the worst case well
/// inside `i32`. One pass covers a pixel with at most 392 member patches
/// (49 references on the step grid, each with at most `MAX_K` members),
/// each weighted by at most 1 (see [`weight_scale`]) and clamped to
/// [`ACCUM_CLAMP`], so the accumulator peaks near `1.03e9`, about half of
/// `i32::MAX`. One unit is `1.9e-6`, well under the `2.4e-4` that one
/// 12-bit code level spans. `wsum` peaks at the same place, because
/// [`WEIGHT_GAIN`] trades a weight's smaller bound for exactly that much
/// more scale.
///
/// A cross-frame ring collects several passes before it is read back and
/// needs [`cross_frame_accum_scale`] instead. [`scatter_patch`] takes the
/// scale as an argument so a single-frame caller keeps this one's full
/// precision. Either way the scale cancels in [`collab_normalise`], which
/// divides one accumulator by the other.
pub const ACCUM_SCALE: f32 = 524_288.0;

/// The headroom [`cross_frame_accum_scale`] leaves under `i32::MAX`,
/// matching the roughly two-fold margin [`ACCUM_SCALE`] leaves for the
/// single-pass case.
const CROSS_FRAME_SAFETY_FACTOR: f64 = 2.0;

/// The fixed-point scale [`crate::nl4d::Nl4dDenoiser`]'s cross-frame
/// accumulator ring counts in, in place of [`ACCUM_SCALE`].
///
/// One fixed constant will not do. A large `spatial_radius` or
/// `temporal_radius` pushes the worst-case accumulator value past
/// `i32::MAX`, and a scale small enough to survive that would throw away
/// precision at the radii people actually use. Deriving it per
/// configuration avoids both.
///
/// The worst case is how many member patches can cover one pixel. A
/// member covering pixel `x` came from a reference whose own top-left is
/// within `(PATCH_SIZE - 1) + 2 * spatial_radius` pixels of `x`, and
/// references sit on the `STEP` grid, so along one axis at most
///
/// ```text
/// refs_per_axis = ((PATCH_SIZE - 1) + 2 * spatial_radius) / STEP + 1
/// ```
///
/// of them can reach it. Squaring for both axes and taking every one of
/// `MAX_K` members gives one pass's contribution count, and a cross-frame
/// ring collects that from `2 * temporal_radius + 1` passes before
/// [`crate::nl4d::Nl4dDenoiser::run_collab_stage`] reads it back. Each
/// contribution is weighted by at most 1 (see [`weight_scale`]) and
/// clamped to [`ACCUM_CLAMP`]. The same figure sizes `wsum`, whose
/// contributions are bounded by [`WEIGHT_CLAMP`] instead but counted at
/// [`WEIGHT_GAIN`] times the scale.
///
/// The result is the largest power of two that keeps that worst case
/// under `i32::MAX` divided by [`CROSS_FRAME_SAFETY_FACTOR`].
pub fn cross_frame_accum_scale(spatial_radius: u32, temporal_radius: u32) -> f32 {
    let refs_per_axis = ((PATCH_SIZE - 1) + 2 * spatial_radius) / STEP + 1;
    let contribs_per_pass = refs_per_axis as f64 * refs_per_axis as f64 * MAX_K as f64;
    let passes = (2 * temporal_radius + 1) as f64;
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

/// The magnitude a single filtered value is clamped to before it enters
/// the accumulator.
///
/// The filter shrinks DCT coefficients of input already inside `[0, 1]`,
/// so a value this large means the filter has already gone wrong. The
/// clamp exists so that when it does, the accumulator saturates
/// gracefully rather than overflowing `i32` and wrapping into a wildly
/// wrong pixel. See [`ACCUM_SCALE`] for how this bound sizes the scale.
pub const ACCUM_CLAMP: f32 = 5.0;

/// The constant every group weight is multiplied by before it reaches
/// the accumulator, so the weights land in a fixed-point-friendly band.
///
/// A group's weight is `1 / sum(retained coefficient variance)`. That
/// sum is a multiple of the caller's `sigma^2`, so the weight's absolute
/// magnitude tracks `1 / sigma^2` and spans far too wide a range for a
/// fixed-point accumulator to hold directly.
///
/// Scaling every weight by the same constant is free, because
/// aggregation computes `sum(w * x) / sum(w)` and any factor common to
/// every weight cancels exactly. This returns that constant.
///
/// `sigma^2 * g_max^2` is the smallest retained sum any group can have.
/// The filter always keeps the group's DC coefficient, whose variance is
/// `sigma^2 * g[0]^2`, and `g[0]` is the profile's largest entry.
/// Dividing by it therefore bounds the normalised weight above by 1,
/// whatever `sigma` and whatever correlation shaping is in use, which is
/// what [`WEIGHT_CLAMP`] relies on.
///
/// The bound below is not `1/512`, the figure a group of 512 coefficients
/// each carrying the plain `sigma^2` would give. `collab_fused` also
/// gives a temporal member the extra variance its motion block's
/// confidence implies, and that has no relation to `sigma`, so the sum
/// runs above `512 * sigma^2 * g_max^2` by however much that extra
/// variance is worth. What keeps the bound finite is
/// [`crate::collab::kernels::fused::MEMBER_SIGMA2_CAP`], which holds a
/// member's own variance to a fixed multiple of the channel's, putting
/// the weight in `[1 / (512 * (1 + cap)), 1]`.
///
/// That range still does not fit `accum`'s fixed point with room to
/// spare, which is why `wsum` counts at [`WEIGHT_GAIN`] times its scale.
/// Without both the cap and the gain, a poorly matched group rounds away
/// to nothing and takes its pixel with it.
///
/// The [`RECIPROCAL_FLOOR`] fallback covers a `sigma` small enough that
/// `sigma^2 * g_max^2` falls under it, zero included. The filter builds
/// its weight with `safe_reciprocal(sum, RECIPROCAL_FLOOR)`, so below
/// that floor every weight saturates at `1 / RECIPROCAL_FLOOR` instead of
/// following the sum. Taking the larger of the two tracks whichever bound
/// the weight is actually against, so the upper bound of 1 holds either
/// way.
pub fn weight_scale(sigma: f32, dct_profile: &[f32; 8]) -> f32 {
    let g_max = dct_profile.iter().copied().fold(0.0f32, f32::max);
    let norm = sigma * sigma * g_max * g_max;
    if norm.is_finite() && norm > RECIPROCAL_FLOOR {
        norm
    } else {
        RECIPROCAL_FLOOR
    }
}

/// The zeroth-order modified Bessel function of the first kind, from its own power series.
///
/// `I0(x) = sum_k ((x / 2)^k / k!)^2`.
///
/// The terms fall off by more than a factor of `x^2 / (4 k^2)` each time, so the series
/// is short and exact to `f64` well before the loop bound for every `beta`
/// [`kaiser_window`] accepts.
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

/// The separable 8-tap Kaiser window [`scatter_patch`] tapers a patch's contribution with.
///
/// A patch is one of many covering each pixel, and every one of them
/// made its own threshold decision. Weighting a patch's edge pixels less
/// than its centre blends those decisions together instead of letting
/// each patch's own decision reach its boundary at full strength, which
/// is what BM3D's aggregation window is for. The window is separable, so
/// eight taps cover the whole 8x8 patch: pixel `(i, j)` takes `w[i] * w[j]`.
///
/// `w[i] = I0(beta * sqrt(1 - (2i / 7 - 1)^2)) / I0(beta)`, the standard
/// Kaiser window over 8 points, normalised so its peak is 1. Larger
/// `beta` tapers harder. BM3D uses 2.0.
///
/// `beta = 0` returns all ones, because the numerator and denominator
/// are both `I0(0)`. That is the off switch, and it is exactly uniform
/// rather than nearly so, so a caller that wants no window gets the same
/// arithmetic the kernel did before this existed.
///
/// Every tap is above zero and at most 1, so the window can only shrink a
/// contribution. [`WEIGHT_CLAMP`]'s bound of 1 on a weight and the
/// worst-case accumulator value both still hold, and neither
/// [`ACCUM_SCALE`] nor [`cross_frame_accum_scale`] needs rederiving.
///
/// What the window does narrow is the other end of the range. The
/// smallest weight the fixed point has to resolve is scaled by the
/// smallest tap product, `w[0]^2`, which is `0.193` at `beta = 2`. A
/// badly matched group's weight lands around 9.8 units of `wsum` before
/// the window and around 1.9 after it, so it still survives the rounding
/// [`WEIGHT_GAIN`] exists to keep it above, with about a fifth of the
/// margin.
pub fn kaiser_window(beta: f32) -> [f32; PATCH_SIZE as usize] {
    let denom = bessel_i0(beta as f64);
    let last = (PATCH_SIZE - 1) as f64;
    let mut window = [0.0f32; PATCH_SIZE as usize];
    for (i, tap) in window.iter_mut().enumerate() {
        let position = 2.0 * i as f64 / last - 1.0;
        *tap = (bessel_i0(beta as f64 * (1.0 - position * position).sqrt()) / denom) as f32;
    }
    window
}

/// The magnitude a group weight is clamped to before it enters `wsum`.
///
/// A normalised weight is `weight_scale / sum`, and [`weight_scale`]
/// returns the smallest `sum` any group can have, so the weight never
/// exceeds 1. That is a fifth of the bound a filtered value needs, and
/// [`WEIGHT_GAIN`] is what turns the difference into resolution.
pub const WEIGHT_CLAMP: f32 = 1.0;

/// The extra fixed-point resolution `wsum` gets over `accum`.
///
/// Both accumulators are sized by the same worst case, how many
/// contributions can reach one pixel multiplied by the largest a single
/// contribution can be. A value is bounded by [`ACCUM_CLAMP`] and a
/// weight by the much smaller [`WEIGHT_CLAMP`], so counting the weight at
/// this multiple of the value's scale spends exactly the same `i32`
/// budget while resolving weights this many times finer.
///
/// The resolution matters because a group weight spans a far wider range
/// than [`weight_scale`]'s own doc used to claim, see the note there. A
/// weight that falls below half a fixed-point unit contributes nothing at
/// all to either accumulator, and this is part of what keeps a poorly
/// matched group above that point.
///
/// [`collab_normalise`] multiplies it back out, so it never reaches a
/// finished pixel.
pub const WEIGHT_GAIN: f32 = ACCUM_CLAMP / WEIGHT_CLAMP;

/// Converts one weighted value into the accumulator's fixed point, at
/// `scale` ([`ACCUM_SCALE`] for a single-frame accumulator, or
/// [`cross_frame_accum_scale`]'s return value for a cross-frame ring).
///
/// Rounds rather than truncating. A filtered value is never negative, so
/// a toward-zero cast would bias every contribution the same way, and it
/// biases the value more than the weight it is divided by: at a weight of
/// three fixed-point units a value of 0.2 truncates to nothing while its
/// weight still counts three. The result is a weighted mean pulled toward
/// black, worst exactly where the weights are smallest.
#[cube]
pub fn to_fixed(value: f32, scale: f32) -> i32 {
    let clamped = f32::clamp(value, -ACCUM_CLAMP, ACCUM_CLAMP);
    f32::round(clamped * scale) as i32
}

/// Converts one group weight into `wsum`'s fixed point, which counts at
/// [`WEIGHT_GAIN`] times the `scale` [`to_fixed`] uses.
///
/// Rounds for the same reason [`to_fixed`] does.
#[cube]
pub fn to_fixed_weight(weight: f32, scale: f32) -> i32 {
    let clamped = f32::clamp(weight, 0.0f32, WEIGHT_CLAMP);
    f32::round(clamped * scale * WEIGHT_GAIN) as i32
}

/// Adds one filtered patch to the accumulators at its own position,
/// inside whichever frame's region of the accumulators it belongs to.
///
/// `value` is this thread's pixel of the patch, and `weight` the
/// normalised weight of the group the patch came from. Every thread in
/// the cube owns one of the patch's 64 pixels, so one call per member
/// scatters the whole patch.
///
/// `kaiser` holds [`kaiser_window`]'s 8 taps, which taper the patch's
/// contribution toward its edges. A caller that wants no taper passes
/// eight ones, which [`kaiser_window`] returns at `beta = 0`.
///
/// `accum`/`wsum` hold one region per frame in a caller's window, each
/// `frame_pixels` (`width * height`) pixels wide, laid out back to back
/// in ring-slot order. `frame_slot` selects the region this member's own
/// frame owns. A single-frame caller passes `frame_slot = 0`, which folds
/// the frame offset away to nothing.
///
/// `write_weight` adds the weight itself to `wsum`. Aggregation needs one
/// weight per covering patch, not one per channel, so only the pass over
/// the first channel sets this.
///
/// `accum_scale` is the fixed-point scale [`to_fixed`] converts into,
/// [`ACCUM_SCALE`] for a single-frame accumulator or
/// [`cross_frame_accum_scale`]'s return value for a cross-frame ring.
/// Either cancels in [`collab_normalise`], so a caller picks whichever
/// matches how many passes can write into its accumulator. `wsum` counts
/// at [`WEIGHT_GAIN`] times that scale, which [`collab_normalise`]
/// multiplies back out.
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
    // The window multiplies the value and the weight by the same factor.
    // `collab_normalise` divides one accumulator by the other, so it
    // cancels wherever the coverage is uniform and reweights the blend
    // where it is not, rather than shifting the pixel's level.
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

/// Clears both accumulators ahead of a filter pass, starting
/// `frame_offset` pixels into them.
///
/// `accum` and `wsum` may hold more than one frame's worth of pixels,
/// laid out back to back in ring-slot order the way [`scatter_patch`]
/// addresses them, so `frame_offset` (in pixels, the same unit `pixels`
/// is) picks out which region this call zeroes. `frame_offset = 0` with
/// `pixels` covering the whole buffer zeroes it all in one call.
///
/// The `accum` region is `pixels * stored_ch` slots wide and `wsum`'s is
/// `pixels` wide, so the loop is sized for the larger one and the weight
/// write is masked off past its end.
///
/// Each thread steps forward by the whole grid's thread count rather than
/// owning one slot. A caller's dispatch grid is clamped to the GPU's
/// 65,535-workgroups-per-dimension limit, which a 4K or 8K frame can
/// exceed at this kernel's 256-thread block size, and striding is what
/// still reaches every slot past that clamp.
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

/// Turns one frame's region of the accumulators into a finished frame
/// plane.
///
/// Each pixel's output is the weighted mean of every filtered patch that
/// covered it, which is `accum / wsum`.
///
/// Both sides carry whatever fixed-point scale the caller's own
/// [`scatter_patch`] calls used, [`ACCUM_SCALE`] or a
/// [`cross_frame_accum_scale`] result, so that scale divides out and
/// never appears here. The weight sum counts at [`WEIGHT_GAIN`] times
/// that scale, which is the one factor that does not cancel, so the
/// ratio is multiplied by it.
///
/// `accum`/`wsum` may hold more than one frame's worth of pixels, laid
/// out back to back in ring-slot order the way [`scatter_patch`]
/// addresses them, and `frame_offset` (in pixels) picks out which region
/// this call reads. `output` is always exactly one frame wide, so it is
/// indexed by the plain, offset-free pixel position.
///
/// The weight sum is never zero. A group always contains its own
/// reference patch, and the references alone cover every pixel between
/// one and nine times over, since they sit on a grid of stride `STEP` and
/// are `PATCH_SIZE` wide. Coverage alone is not enough, because a weight
/// small enough to round to nothing would leave a covered pixel with an
/// empty weight sum. [`WEIGHT_GAIN`] and
/// [`crate::collab::kernels::fused::MEMBER_SIGMA2_CAP`] together keep
/// every group's weight above that point.
///
/// If the weight sum ever were to be zero anyway, the guard below returns
/// the accumulator untouched rather than a NaN or an infinity.
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
    let w = wsum[pixel as usize];

    let mut out = Vector::<f32, N>::empty();
    #[unroll]
    for c in 0..channels {
        let a = accum[(pixel * stored_ch + c) as usize] as f32;
        let mut v = a;
        if w != 0i32 {
            v = a * WEIGHT_GAIN / (w as f32);
        }
        out[c as usize] = v;
    }
    output[local_pixel as usize] = out;
}

// Ties the fixed-point headroom argument in `ACCUM_SCALE`'s docs to the
// constants it is actually derived from. A larger patch or a larger
// group both raise the number of contributions one pass can add to a
// pixel, and the scale has to come down to match either moving.
const _: () = assert!(
    PATCH_AREA == 64 && MAX_K == 8,
    "recheck ACCUM_SCALE's headroom, the per-pass contribution bound moved"
);

// `cross_frame_accum_scale` needs no equivalent assertion. It re-derives
// its scale from `PATCH_SIZE`, `STEP`, and `MAX_K` on every call, so
// there is no baked-in number for a compile-time check to protect. Its
// arithmetic is guarded instead by a `debug_assert!` inside the function
// and by the exhaustive test below.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl4d::MAX_KAISER_BETA;

    /// A filtered value is never negative, so a toward-zero cast biases
    /// every contribution the same way. Rounding is what keeps the
    /// weighted mean `collab_normalise` computes centred on the value the
    /// filter actually produced.
    #[test]
    fn to_fixed_rounds_rather_than_truncating() {
        assert_eq!(to_fixed(0.6, 1.0), 1);
        assert_eq!(to_fixed(0.4, 1.0), 0);
        assert_eq!(to_fixed(-0.6, 1.0), -1);
    }

    /// A weight small enough to round away in `accum`'s fixed point still
    /// reaches `wsum`, because `WEIGHT_GAIN` buys back exactly the
    /// resolution the weight's smaller bound leaves unused.
    #[test]
    fn to_fixed_weight_resolves_finer_than_to_fixed() {
        let scale = 65_536.0f32;
        let weight = 0.3 / scale;
        assert_eq!(to_fixed(weight, scale), 0);
        assert_eq!(to_fixed_weight(weight, scale), 2);
    }

    /// The weight's own bound, which is what lets `WEIGHT_GAIN` spend the
    /// same `i32` budget as a value clamped to `ACCUM_CLAMP`.
    #[test]
    fn to_fixed_weight_clamps_at_one() {
        let scale = 1_024.0f32;
        assert_eq!(to_fixed_weight(4.0, scale), to_fixed_weight(WEIGHT_CLAMP, scale),);
        assert_eq!(to_fixed_weight(-1.0, scale), 0);
    }

    // `Nl4dParams::validate` in `nl4d/params.rs` is what actually enforces
    // these ranges; they are repeated here as plain numbers rather than
    // imported, so this test does not depend on `nl4d` at all and keeps
    // exercising the true worst case even if that module's ranges ever
    // narrow.
    const SPATIAL_RADIUS_RANGE: std::ops::RangeInclusive<u32> = 1..=16;
    const TEMPORAL_RADIUS_RANGE: std::ops::RangeInclusive<u32> = 1..=8;

    /// Recomputes the worst-case accumulator value the same way
    /// [`cross_frame_accum_scale`]'s own `debug_assert!` does, for every
    /// `(spatial_radius, temporal_radius)` pair the validated parameter
    /// ranges allow, rather than only the pair a debug build happens to
    /// exercise at runtime.
    #[test]
    fn every_spatial_and_temporal_radius_stays_under_the_safety_budget() {
        let budget = i32::MAX as f64 / CROSS_FRAME_SAFETY_FACTOR;

        for spatial_radius in SPATIAL_RADIUS_RANGE {
            for temporal_radius in TEMPORAL_RADIUS_RANGE {
                let refs_per_axis = ((PATCH_SIZE - 1) + 2 * spatial_radius) / STEP + 1;
                let contribs_per_pass = refs_per_axis as f64 * refs_per_axis as f64 * MAX_K as f64;
                let passes = (2 * temporal_radius + 1) as f64;
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

    /// Deriving the scale per configuration, rather than sizing one
    /// constant for the widest configuration allowed, is what lets a
    /// typical configuration keep more fixed-point precision. The
    /// defaults should clear `2^15`.
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
        assert_eq!(kaiser_window(0.0), [1.0f32; PATCH_SIZE as usize]);
    }

    /// An even tap count puts the centre of the span between taps 3 and
    /// 4, so those two are equal rather than one being above the other,
    /// and the rise is checked up to that pair.
    #[test]
    fn kaiser_window_is_symmetric_and_rises_to_the_centre() {
        for beta in [1.0f32, 2.0, 4.0, MAX_KAISER_BETA] {
            let w = kaiser_window(beta);
            for i in 0..4 {
                assert!(
                    (w[i] - w[7 - i]).abs() < 1e-6,
                    "beta {beta}: tap {i} is {} and its mirror {}",
                    w[i],
                    w[7 - i],
                );
                if i < 3 {
                    assert!(w[i + 1] > w[i], "beta {beta}: tap {} is not above tap {i}", i + 1,);
                }
            }
            assert!(
                w.iter().all(|&t| t > 0.0 && t <= 1.0),
                "beta {beta}: a tap is not above zero and at most 1, {w:?}",
            );
        }
    }

    #[test]
    fn kaiser_window_end_taps_are_the_bessel_ratio() {
        for beta in [1.0f32, 2.0, 4.0] {
            let w = kaiser_window(beta);
            let expected = (1.0 / bessel_i0(beta as f64)) as f32;
            assert!(
                (w[0] - expected).abs() < 1e-6,
                "beta {beta}: end tap {} against the ratio {expected}",
                w[0],
            );
            assert!((w[7] - expected).abs() < 1e-6);
        }
        // The figure the doc's margin arithmetic uses.
        assert!((kaiser_window(2.0)[0] - 0.4388).abs() < 1e-3);
    }

    /// Ties the `392`-contribution figure in [`ACCUM_SCALE`]'s docs to
    /// the model `cross_frame_accum_scale` is built on.
    #[test]
    fn contribution_model_reproduces_the_documented_392_at_the_default_spatial_radius() {
        let spatial_radius = 9u32;
        let refs_per_axis = ((PATCH_SIZE - 1) + 2 * spatial_radius) / STEP + 1;
        assert_eq!(refs_per_axis, 7);
        assert_eq!(refs_per_axis * refs_per_axis * MAX_K, 392);
    }
}

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
/// 12-bit code level spans.
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
/// clamped to [`ACCUM_CLAMP`].
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
/// `sigma^2 * g[0]^2`, and `g[0]` is the profile's largest entry. The sum
/// therefore sits in `[sigma^2 * g_max^2, 512 * sigma^2 * g_max^2]`, so
/// dividing by it puts the normalised weight in `[1/512, 1]` whatever
/// `sigma` and whatever correlation shaping is in use.
///
/// The [`RECIPROCAL_FLOOR`] fallback covers a `sigma` small enough that
/// `sigma^2 * g_max^2` falls under it, zero included. The filter builds
/// its weight with `safe_reciprocal(sum, RECIPROCAL_FLOOR)`, so below
/// that floor every weight saturates at `1 / RECIPROCAL_FLOOR` instead of
/// following the sum. Taking the larger of the two tracks whichever bound
/// the weight is actually against, and keeps the normalised weight in
/// `[1/512, 1]` either way.
pub fn weight_scale(sigma: f32, dct_profile: &[f32; 8]) -> f32 {
    let g_max = dct_profile.iter().copied().fold(0.0f32, f32::max);
    let norm = sigma * sigma * g_max * g_max;
    if norm.is_finite() && norm > RECIPROCAL_FLOOR {
        norm
    } else {
        RECIPROCAL_FLOOR
    }
}

/// Converts one weighted value into the accumulator's fixed point, at
/// `scale` ([`ACCUM_SCALE`] for a single-frame accumulator, or
/// [`cross_frame_accum_scale`]'s return value for a cross-frame ring).
#[cube]
pub fn to_fixed(value: f32, scale: f32) -> i32 {
    let clamped = f32::clamp(value, -ACCUM_CLAMP, ACCUM_CLAMP);
    (clamped * scale) as i32
}

/// Adds one filtered patch to the accumulators at its own position,
/// inside whichever frame's region of the accumulators it belongs to.
///
/// `value` is this thread's pixel of the patch, and `weight` the
/// normalised weight of the group the patch came from. Every thread in
/// the cube owns one of the patch's 64 pixels, so one call per member
/// scatters the whole patch.
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
/// matches how many passes can write into its accumulator.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn scatter_patch(
    accum: &mut Array<Atomic<i32>>,
    wsum: &mut Array<Atomic<i32>>,
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
    let local_pixel = (patch_y + tid / PATCH_SIZE) * width + patch_x + tid % PATCH_SIZE;
    let pixel = frame_slot * frame_pixels + local_pixel;
    Atomic::fetch_add(
        &accum[(pixel * stored_ch + channel) as usize],
        to_fixed(value * weight, accum_scale),
    );
    if write_weight {
        Atomic::fetch_add(&wsum[pixel as usize], to_fixed(weight, accum_scale));
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
/// never appears here.
///
/// `accum`/`wsum` may hold more than one frame's worth of pixels, laid
/// out back to back in ring-slot order the way [`scatter_patch`]
/// addresses them, and `frame_offset` (in pixels) picks out which region
/// this call reads. `output` is always exactly one frame wide, so it is
/// indexed by the plain, offset-free pixel position.
///
/// The weight sum is never zero in practice. A group always contains its
/// own reference patch, and the references alone cover every pixel
/// between one and nine times over, since they sit on a grid of stride
/// `STEP` and are `PATCH_SIZE` wide.
///
/// If the weight sum ever were to be zero, the guard below returns the
/// accumulator untouched rather than a NaN or an infinity.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
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
            v = a / (w as f32);
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

use cubecl::prelude::*;

use crate::collab::kernels::transforms::RECIPROCAL_FLOOR;
use crate::collab::{MAX_K, PATCH_AREA, PATCH_SIZE, STEP};

/// The fixed-point scale the accumulators count in.
///
/// Aggregation sums one weighted value per covering patch into a shared
/// accumulator, which means one atomic add per contribution. Floating
/// point atomics are emulated through compare-and-swap retry loops on
/// several GPUs and measured roughly 50 times slower than integer ones
/// on the hardware this was developed against, so the accumulators hold
/// fixed-point integers instead.
///
/// `2^19` is chosen from three bounds. The normalised group weight lands
/// in `[1/512, 1]` (see [`weight_scale`]), a filtered pixel is clamped to
/// [`ACCUM_CLAMP`], and a pixel is covered by at most 392 member patches
/// in one filter pass (49 reference patches within reach on the step
/// grid, each holding at most `MAX_K` members). The largest accumulator
/// value is therefore `392 * 1 * 5 * 2^19`, about `1.03e9`, which leaves
/// a factor of two under `i32::MAX`. One unit is `1.9e-6`, well under the
/// `2.4e-4` one code level of 12-bit output spans.
///
/// This is the scale every single-frame caller passes to
/// [`scatter_patch`], where an accumulator only ever sees one pass's
/// worth of contributions. A
/// cross-frame accumulator, [`crate::nl4d::Nl4dDenoiser`]'s ring of
/// per-frame regions, sees contributions from several passes before it
/// is read back and needs [`cross_frame_accum_scale`] instead, a smaller
/// scale sized for that larger worst case. `scatter_patch` takes the
/// scale as a runtime argument rather than reading either constant
/// itself, precisely so a caller with only one pass's worth of headroom
/// to spend keeps this constant's full precision rather than paying for
/// headroom it never uses.
///
/// The scale cancels in `collab_normalise`, which divides one
/// accumulator by the other, so nothing downstream has to know it.
pub const ACCUM_SCALE: f32 = 524_288.0;

/// The factor of headroom [`cross_frame_accum_scale`] leaves under
/// `i32::MAX`, matching the roughly two-fold margin [`ACCUM_SCALE`]'s
/// own fixed derivation leaves for the single-pass case.
const CROSS_FRAME_SAFETY_FACTOR: f64 = 2.0;

/// Derives the fixed-point scale [`crate::nl4d::Nl4dDenoiser`]'s
/// cross-frame accumulator ring should count in, in place of
/// [`ACCUM_SCALE`], from the `spatial_radius` and `temporal_radius` a
/// particular denoiser was actually built with.
///
/// # Why a fixed constant was not safe
///
/// An earlier version of this scale was a single constant, `2^15`,
/// derived from `392 contributions/pass * 17 passes * 5 * 2^15`, about
/// `1.09e9`, a little under half of `i32::MAX`. That derivation mixed a
/// worst case from each axis without checking the two are reachable
/// together. `17 passes` is the worst case on the temporal axis, `2 *
/// MAX_TEMPORAL_RADIUS + 1` at the validated ceiling
/// `MAX_TEMPORAL_RADIUS = 8`. But `392` contributions per pass is only
/// the default `spatial_radius = 9`, not that axis's own worst case.
/// [`crate::nl4d::Nl4dParams::validate`] permits `spatial_radius` up to
/// `16`, and a larger radius raises the contribution count. At
/// `spatial_radius = 15` or `16` combined with `temporal_radius = 8`,
/// the true worst-case accumulator value clears `i32::MAX` outright, and
/// the fixed `2^15` scale silently overflows and wraps the accumulator
/// into a wildly wrong pixel, with no crash and nothing in the output to
/// flag it.
///
/// # The real bound
///
/// A member patch that ends up covering pixel `x` has a top-left `P`
/// somewhere in `[x - (PATCH_SIZE - 1), x]`, and `P` lies within
/// `spatial_radius` pixels of whichever reference `R` produced it (see
/// [`crate::collab::kernels::group_temporal::collab_group_temporal`]'s
/// spatial search window). So every reference that could have scattered
/// a member covering `x` has its own top-left within `(PATCH_SIZE - 1) +
/// 2 * spatial_radius` pixels of `x`, and references sit on the `STEP`
/// grid, so at most
///
/// ```text
/// refs_per_axis = ((PATCH_SIZE - 1) + 2 * spatial_radius) / STEP + 1
/// ```
///
/// of them (integer division) lie in that span along one axis. This
/// reproduces the old `392` figure exactly at `spatial_radius = 9`
/// (`refs_per_axis = 7`, `7 * 7 * MAX_K = 392`), which is what validates
/// the model. Squaring for both axes and taking every one of `MAX_K`
/// members from each such reference gives the worst-case contribution
/// count for one pass,
///
/// ```text
/// contribs_per_pass = refs_per_axis^2 * MAX_K
/// ```
///
/// and a cross-frame accumulator can receive that many times over from
/// as many as `2 * temporal_radius + 1` passes before
/// [`crate::nl4d::Nl4dDenoiser::run_collab_stage`] reads it back, each
/// contribution clamped to [`ACCUM_CLAMP`] and weighted by at most `1.0`
/// (see [`weight_scale`]). The scale returned is the largest power of
/// two that keeps `contribs_per_pass * (2 * temporal_radius + 1) *
/// ACCUM_CLAMP * scale` under `i32::MAX / `[`CROSS_FRAME_SAFETY_FACTOR`],
/// so a power-of-two scale keeps the same fixed-point behaviour the
/// rest of this module relies on.
///
/// Deriving the scale from the configuration actually in use, rather
/// than the worst case every axis allows, also means a typical
/// configuration keeps more fixed-point precision than the worst case
/// would afford it. At the defaults, `spatial_radius = 9` and
/// `temporal_radius = 2`, this returns a scale several times larger than
/// the old fixed `2^15`.
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
/// The filters shrink DCT coefficients of input already inside `[0, 1]`,
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
/// `sigma^2 * g_max^2` is the right one, because the weight is the
/// reciprocal of the retained variance sum and that sum is bounded
/// below by exactly this quantity. Both filters always retain the
/// group's DC coefficient, whose propagated variance is exactly
/// `sigma^2 * g[0]^2` for a profile `g`, and `g[0]` is that profile's
/// maximum for every correlation it is defined for. The retained sum
/// therefore sits in `[sigma^2 * g_max^2, 512 * sigma^2 * g_max^2]`, so
/// the normalised weight sits in `[1/512, 1]` whatever `sigma` and
/// whatever correlation shaping is in use.
///
/// A `sigma` small enough that `sigma^2 * g_max^2` falls under
/// [`RECIPROCAL_FLOOR`], `sigma` of exactly zero included, is why the
/// floor appears here too. The filters build their weight with
/// `safe_reciprocal(sum, RECIPROCAL_FLOOR)`, so once the retained sum
/// drops under that floor every group's weight saturates at `1 /
/// RECIPROCAL_FLOOR` instead of following the sum. Taking the larger of
/// the two tracks whichever bound the weight is actually against, and
/// the normalised weight stays inside `[1/512, 1]` either way. Without
/// it a `sigma` of zero would leave every weight at `1e12`, and the
/// accumulator's clamp would saturate the value and the weight alike and
/// return a flat 1.0 for every pixel.
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
/// in ring-slot order. `frame_slot` selects which region this member's
/// own frame owns, and `frame_pixels` is that region's width. A caller
/// with only one frame always passes `frame_slot = 0`, which folds the
/// frame offset away to nothing and reproduces the single-frame
/// addressing this function
/// used before cross-frame aggregation existed, whatever `frame_pixels`
/// happens to be.
///
/// `write_weight` adds the weight itself to `wsum`. Aggregation needs
/// one weight per covering patch, not one per channel, so only the pass
/// over the first channel sets this.
///
/// `accum_scale` is the fixed-point scale [`to_fixed`] converts into,
/// [`ACCUM_SCALE`] for a single-frame accumulator or
/// [`cross_frame_accum_scale`]'s return value for a cross-frame ring. It
/// cancels in `collab_normalise` the same way either scale does on its
/// own, so a caller is free to pick whichever one matches how many
/// passes its own accumulator can receive contributions from before it
/// is read back.
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
/// One thread per accumulator slot within the `pixels`-wide region this
/// call clears. `accum` and `wsum` may hold more than one frame's worth
/// of pixels, laid out back to back in ring-slot order the way
/// [`scatter_patch`] addresses them, so `frame_offset` (in pixels, the
/// same unit `pixels` is) picks out which region this call zeroes.
/// `frame_offset = 0` together with `pixels` covering the whole buffer
/// zeroes it all in one call, the way a single-frame caller always
/// does, and the way a fresh cross-frame ring is cleared in full before
/// its first pass.
///
/// The `accum` region is `pixels * stored_ch` slots wide and `wsum`'s is
/// `pixels` wide, so the loop is sized for the larger one and the
/// weight write is masked off past its end.
///
/// Grid-strided rather than one thread per slot, because the caller's
/// dispatch grid is clamped to the GPU's 65,535-workgroups-per-dimension
/// limit and a 4K or 8K frame's `pixels * stored_ch` can need more
/// workgroups than that at the 256-thread block size this launches
/// with. A one-thread-per-slot launch would leave the tail past the
/// clamp point holding whatever was already there instead of zero, so
/// each thread instead steps forward by the whole grid's thread count
/// until it has covered every slot.
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
/// covered it, which is `accum / wsum`. Both sides carry whatever fixed-
/// point scale the caller's own [`scatter_patch`] calls used,
/// [`ACCUM_SCALE`] or a [`cross_frame_accum_scale`] result, so that scale
/// divides out and never appears here.
///
/// `accum`/`wsum` may hold more than one frame's worth of pixels, laid
/// out back to back in ring-slot order the way [`scatter_patch`]
/// addresses them, and `frame_offset` (in pixels) picks out which
/// region this call reads. `output` is always exactly one frame wide,
/// `width * height` pixels, regardless of how big `accum`/`wsum` are,
/// so it is indexed by the plain, offset-free pixel position. A
/// single-frame caller always passes `frame_offset = 0`, which
/// reproduces the addressing this function used before cross-frame
/// aggregation existed.
///
/// # Why the weight sum is never zero
///
/// A group always contains its own reference patch as member 0, and the
/// reference patches alone cover every pixel of the frame between one
/// and nine times over, because they sit on a grid of stride `STEP` and
/// are `PATCH_SIZE` wide. Every pixel therefore receives at least one
/// contribution with a positive weight. A `wsum` of zero would still
/// have to come from somewhere for this to divide by it, so the guard
/// below returns the accumulator untouched rather than a NaN or an
/// infinity.
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

// `cross_frame_accum_scale` has no equivalent compile-time assertion,
// unlike `ACCUM_SCALE` above. That assertion exists because `ACCUM_SCALE`
// is a constant baked in ahead of time from `PATCH_AREA` and `MAX_K`, so
// nothing else re-checks it if either constant moves. `cross_frame_accum_
// scale` reads `PATCH_SIZE`, `STEP`, and `MAX_K` itself and re-derives
// the scale from whatever they currently are, plus the caller's own
// `spatial_radius` and `temporal_radius`, every time it runs, so there is
// no baked-in number left for a compile-time check to protect. This is
// exactly the bug this module used to have: a scale derived once from a
// worst case and then trusted for every configuration, including ones
// the derivation never actually covered. What needs guarding now is the
// derivation's own arithmetic, which a `debug_assert!` inside
// `cross_frame_accum_scale` itself checks on every call, and which this
// module's own test `every_spatial_and_temporal_radius_stays_under_the_
// safety_budget` checks exhaustively across every `spatial_radius` and
// `temporal_radius` the validated parameter ranges allow, independent of
// whether debug assertions are compiled in.

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

    /// This is the test that would have caught the original bug: it
    /// recomputes the worst-case accumulator value the same way
    /// [`cross_frame_accum_scale`]'s own `debug_assert!` does, for every
    /// `(spatial_radius, temporal_radius)` pair the validated parameter
    /// ranges allow, rather than only the one pair a debug build happens
    /// to exercise at runtime.
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

    /// The point of deriving the scale per configuration, rather than
    /// sizing one constant for the worst case every configuration is
    /// allowed to reach, is that a typical configuration keeps more
    /// fixed-point precision than the worst case would afford it. At the
    /// defaults this should give a scale at or above the old fixed
    /// `2^15`.
    #[test]
    fn defaults_keep_at_least_the_old_fixed_scale() {
        let old_fixed_scale = 32_768.0f32;
        let derived = cross_frame_accum_scale(9, 2);

        assert!(
            derived >= old_fixed_scale,
            "derived scale {derived} at the defaults (spatial_radius=9, temporal_radius=2) \
             should be at least the old fixed scale {old_fixed_scale}",
        );
    }

    /// Ties the `392`-contribution figure in this module's docs to the
    /// model `cross_frame_accum_scale` is built on, the same way the
    /// original derivation's own worked example did.
    #[test]
    fn contribution_model_reproduces_the_documented_392_at_the_default_spatial_radius() {
        let spatial_radius = 9u32;
        let refs_per_axis = ((PATCH_SIZE - 1) + 2 * spatial_radius) / STEP + 1;
        assert_eq!(refs_per_axis, 7);
        assert_eq!(refs_per_axis * refs_per_axis * MAX_K, 392);
    }
}

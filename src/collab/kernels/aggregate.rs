use cubecl::prelude::*;

use crate::collab::kernels::transforms::RECIPROCAL_FLOOR;
use crate::collab::{PATCH_AREA, PATCH_SIZE};

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
/// (49 reference patches within reach on the step grid, each holding at
/// most `MAX_K` members). The largest accumulator value is therefore
/// `392 * 1 * 5 * 2^19`, about `1.03e9`, which leaves a factor of two
/// under `i32::MAX`. One unit is `1.9e-6`, well under the `2.4e-4` one
/// code level of 12-bit output spans.
///
/// The scale cancels in `collab_normalise`, which divides one
/// accumulator by the other, so nothing downstream has to know it.
pub const ACCUM_SCALE: f32 = 524_288.0;

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

/// Converts one weighted value into the accumulator's fixed point.
#[cube]
pub fn to_fixed(value: f32) -> i32 {
    let clamped = f32::clamp(value, -ACCUM_CLAMP, ACCUM_CLAMP);
    (clamped * ACCUM_SCALE) as i32
}

/// Adds one filtered patch to the accumulators at its own position in
/// the frame.
///
/// `value` is this thread's pixel of the patch, and `weight` the
/// normalised weight of the group the patch came from. Every thread in
/// the cube owns one of the patch's 64 pixels, so one call per member
/// scatters the whole patch.
///
/// `write_weight` adds the weight itself to `wsum`. Aggregation needs
/// one weight per covering patch, not one per channel, so only the pass
/// over the first channel sets this.
#[cube]
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
) {
    let pixel = (patch_y + tid / PATCH_SIZE) * width + patch_x + tid % PATCH_SIZE;
    Atomic::fetch_add(
        &accum[(pixel * stored_ch + channel) as usize],
        to_fixed(value * weight),
    );
    if write_weight {
        Atomic::fetch_add(&wsum[pixel as usize], to_fixed(weight));
    }
}

/// Clears both accumulators ahead of a filter pass.
///
/// One thread per accumulator slot. `accum` holds `width * height *
/// stored_ch` slots and `wsum` holds `width * height`, so the launch is
/// sized for the larger one and the weight write is masked off past its
/// end.
#[cube(launch_unchecked)]
pub fn collab_zero_accum(
    accum: &mut Array<Atomic<i32>>,
    wsum: &mut Array<Atomic<i32>>,
    #[comptime] pixels: u32,
    #[comptime] stored_ch: u32,
) {
    let idx = ABSOLUTE_POS_X;
    if idx < pixels * stored_ch {
        Atomic::store(&accum[idx as usize], 0i32);
    }
    if idx < pixels {
        Atomic::store(&wsum[idx as usize], 0i32);
    }
}

/// Turns the accumulators into the finished frame plane.
///
/// Each pixel's output is the weighted mean of every filtered patch that
/// covered it, which is `accum / wsum`. Both sides carry the same
/// [`ACCUM_SCALE`], so the scale divides out and never appears here.
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
pub fn collab_normalise<N: Size>(
    accum: &Array<i32>,
    wsum: &Array<i32>,
    output: &mut Array<Vector<f32, N>>,
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

    let pixel = y * width + x;
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
    output[pixel as usize] = out;
}

// Ties the fixed-point headroom argument in `ACCUM_SCALE`'s docs to the
// constants it is actually derived from. A larger patch or a larger
// group both raise the number of contributions a pixel can receive, and
// the scale has to come down to match.
const _: () = assert!(
    PATCH_AREA == 64 && super::super::MAX_K == 8,
    "recheck ACCUM_SCALE's headroom, the per-pixel contribution bound moved"
);

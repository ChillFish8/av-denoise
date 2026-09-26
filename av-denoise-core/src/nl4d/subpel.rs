use std::str::FromStr;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::kernels::nl4d_phase_planes;
use crate::nlmeans::{BLOCK_X, BLOCK_Y};

/// How finely nl4d aligns a temporal match between whole pixels.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Hash)]
pub enum SubpelPrecision {
    #[default]
    Off,
    Half,
    Quarter,
}

impl SubpelPrecision {
    /// The comptime selector the fused kernel specialises on.
    pub(crate) fn kernel_mode(self) -> u32 {
        match self {
            SubpelPrecision::Off => 0,
            SubpelPrecision::Half => 1,
            SubpelPrecision::Quarter => 2,
        }
    }
}

impl FromStr for SubpelPrecision {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "off" => Ok(SubpelPrecision::Off),
            "half" => Ok(SubpelPrecision::Half),
            "quarter" => Ok(SubpelPrecision::Quarter),
            other => Err(format!(
                "subpel must be one of off, half or quarter, got '{other}'"
            )),
        }
    }
}

/// The HEVC half-sample filter.
///
/// Applied to the pixels from `x - 3` to `x + 4`, it gives the sample at
/// `x + 1/2`.
pub const HALF_PEL_TAPS: [f32; 8] = [
    -1.0 / 64.0,
    4.0 / 64.0,
    -11.0 / 64.0,
    40.0 / 64.0,
    40.0 / 64.0,
    -11.0 / 64.0,
    4.0 / 64.0,
    -1.0 / 64.0,
];

/// Quarter-pel phases on both axes, `4 * 4`.
pub const PHASE_COUNT: usize = 16;

/// The side of the weight map [phase_gains] builds, wide enough for every
/// tap a phase can reach.
const WEIGHT_MAP_SIDE: usize = 12;

/// Where integer offset 0 sits inside the weight map.
const WEIGHT_MAP_ORIGIN: usize = 3;

/// The integer-pixel weights behind one half-grid coordinate on one axis.
///
/// An even coordinate is the integer pixel itself. An odd one is the
/// filter centred between that pixel and the next.
fn half_weights_1d(half_coord: u32) -> [f32; WEIGHT_MAP_SIDE] {
    let mut weights = [0.0f32; WEIGHT_MAP_SIDE];
    let pixel = (half_coord >> 1) as usize + WEIGHT_MAP_ORIGIN;
    if half_coord & 1 == 0 {
        weights[pixel] = 1.0;
        return weights;
    }

    for (k, &tap) in HALF_PEL_TAPS.iter().enumerate() {
        weights[pixel + k - 3] = tap;
    }
    weights
}

/// The two half-grid points a quarter-grid sample averages, as
/// `(ax, ay, bx, by)` in half-pel units.
///
/// An axis on the half grid uses its own point twice. An odd axis takes
/// the points either side. When both axes are odd the pair lies on the
/// anti-diagonal.
pub(crate) fn half_pair(qx: u32, qy: u32) -> (u32, u32, u32, u32) {
    let odd_x = qx & 1;
    let odd_y = qy & 1;
    let both_odd = odd_x & odd_y == 1;

    let low_y = qy >> 1;
    let high_y = (qy + odd_y) >> 1;
    let (ay, by) = if both_odd {
        (high_y, low_y)
    } else {
        (low_y, high_y)
    };

    (qx >> 1, ay, (qx + odd_x) >> 1, by)
}

/// The noise-variance gain of every quarter-pel phase, relative to one
/// integer pixel.
///
/// Each phase's sample is a fixed linear mix of integer pixels. For
/// independent noise its variance is the sum of the squared weights. A
/// quarter phase averages two half-grid samples that share pixels, and
/// building the combined weights first accounts for that overlap.
pub fn phase_gains() -> [f32; PHASE_COUNT] {
    let mut gains = [0.0f32; PHASE_COUNT];

    for py in 0..4u32 {
        for px in 0..4u32 {
            let (ax, ay, bx, by) = half_pair(px, py);
            let first_x = half_weights_1d(ax);
            let first_y = half_weights_1d(ay);
            let second_x = half_weights_1d(bx);
            let second_y = half_weights_1d(by);

            let mut sum_sq = 0.0f64;
            for row in 0..WEIGHT_MAP_SIDE {
                for col in 0..WEIGHT_MAP_SIDE {
                    let first = first_x[col] as f64 * first_y[row] as f64;
                    let second = second_x[col] as f64 * second_y[row] as f64;
                    let weight = 0.5 * (first + second);
                    sum_sq += weight * weight;
                }
            }
            gains[(py * 4 + px) as usize] = sum_sq as f32;
        }
    }

    gains
}

/// The factor [weight_scale](crate::collab::kernels::aggregate::weight_scale)
/// is multiplied by, so a group of low-noise fractional members still
/// normalises to a weight of at most 1.
pub(crate) fn weight_floor_gain(precision: SubpelPrecision) -> f32 {
    match precision {
        SubpelPrecision::Off => 1.0,
        SubpelPrecision::Half | SubpelPrecision::Quarter => {
            phase_gains().into_iter().fold(f32::MAX, f32::min)
        },
    }
}

/// The buffers and shape one phase-plane dispatch binds.
pub(crate) struct PhasePlaneCtx<'a> {
    pub ring: &'a Handle,
    pub phase_ring: &'a Handle,
    pub taps: &'a Handle,
    pub total_frames: u32,
    pub width: u32,
    pub height: u32,
    pub stored_ch: u32,
}

/// Rebuilds the four phase planes of one ring slot from that slot's
/// current contents.
pub(crate) fn run_phase_planes<R: Runtime>(client: &ComputeClient<R>, ctx: &PhasePlaneCtx<'_>, slot: u32) {
    let ring_len = (ctx.total_frames * ctx.width * ctx.height * ctx.stored_ch) as usize;
    let grid = CubeCount::new_2d(ctx.width.div_ceil(BLOCK_X), ctx.height.div_ceil(BLOCK_Y));

    unsafe {
        nl4d_phase_planes::launch_unchecked::<R>(
            client,
            grid,
            CubeDim::new_2d(BLOCK_X, BLOCK_Y),
            ctx.stored_ch as usize,
            ArrayArg::from_raw_parts(ctx.ring.clone(), ring_len),
            ArrayArg::from_raw_parts(ctx.phase_ring.clone(), ring_len * 4),
            ArrayArg::from_raw_parts(ctx.taps.clone(), 8),
            slot,
            ctx.width,
            ctx.height,
        );
    }
}

/// The four phase planes of one frame, computed on the host.
///
/// Edges clamp the same way the kernel's reads do, and HV filters the
/// clamped rows of the unrounded H result.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) fn phase_planes_host(frame: &[f32], width: u32, height: u32, stored_ch: u32) -> [Vec<f32>; 4] {
    let ch = stored_ch as usize;
    let at = |x: i32, y: i32, c: usize| -> f32 {
        let cx = x.clamp(0, width as i32 - 1) as usize;
        let cy = y.clamp(0, height as i32 - 1) as usize;
        frame[(cy * width as usize + cx) * ch + c]
    };
    let horizontal_at = |x: i32, y: i32, c: usize| -> f32 {
        let mut sum = 0.0f32;
        for (k, &tap) in HALF_PEL_TAPS.iter().enumerate() {
            sum += tap * at(x + k as i32 - 3, y, c);
        }
        sum
    };

    let len = frame.len();
    let mut planes = [frame.to_vec(), vec![0.0; len], vec![0.0; len], vec![0.0; len]];
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            for c in 0..ch {
                let idx = (y as usize * width as usize + x as usize) * ch + c;
                let mut vertical = 0.0f32;
                let mut diagonal = 0.0f32;
                for (k, &tap) in HALF_PEL_TAPS.iter().enumerate() {
                    let row = (y + k as i32 - 3).clamp(0, height as i32 - 1);
                    vertical += tap * at(x, row, c);
                    diagonal += tap * horizontal_at(x, row, c);
                }
                planes[1][idx] = horizontal_at(x, y, c);
                planes[2][idx] = vertical;
                planes[3][idx] = diagonal;
            }
        }
    }
    planes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gain_at(px: usize, py: usize) -> f32 {
        phase_gains()[py * 4 + px]
    }

    #[test]
    fn integer_phase_gain_is_exactly_one() {
        assert_eq!(gain_at(0, 0), 1.0);
    }

    #[test]
    fn half_phase_gains_match_the_filter_taps() {
        let half = 3476.0 / 4096.0;
        assert!((gain_at(2, 0) - half).abs() < 1e-6);
        assert!((gain_at(0, 2) - half).abs() < 1e-6);
        assert!((gain_at(2, 2) - half * half).abs() < 1e-6);
    }

    /// Averaging the integer sample with the half sample beside it gives
    /// weights of `taps / 2` plus one half at offset 0.
    #[test]
    fn quarter_phase_gain_matches_the_averaged_taps() {
        let expected = 12692.0 / 16384.0;
        assert!((gain_at(1, 0) - expected).abs() < 1e-6);
    }

    #[test]
    fn phase_gains_are_symmetric_across_axes_and_directions() {
        for (px, py) in [(1, 0), (3, 0), (0, 1), (0, 3)] {
            assert!((gain_at(px, py) - gain_at(1, 0)).abs() < 1e-6, "({px}, {py})");
        }
        for (px, py) in [(1, 1), (3, 3), (1, 3), (3, 1)] {
            assert!(gain_at(px, py) < 1.0, "({px}, {py})");
        }
    }

    #[test]
    fn every_phase_gain_is_positive_and_at_most_one() {
        for gain in phase_gains() {
            assert!(gain > 0.0 && gain <= 1.0, "gain {gain}");
        }
    }

    #[test]
    fn weight_floor_gain_is_exactly_one_when_off() {
        assert_eq!(weight_floor_gain(SubpelPrecision::Off), 1.0);
    }

    #[test]
    fn weight_floor_gain_is_the_smallest_phase_gain_when_on() {
        let smallest = phase_gains().into_iter().fold(f32::MAX, f32::min);
        assert_eq!(weight_floor_gain(SubpelPrecision::Half), smallest);
        assert_eq!(weight_floor_gain(SubpelPrecision::Quarter), smallest);
    }

    #[test]
    fn subpel_precision_parses_case_insensitively() {
        assert_eq!("off".parse::<SubpelPrecision>(), Ok(SubpelPrecision::Off));
        assert_eq!("Half".parse::<SubpelPrecision>(), Ok(SubpelPrecision::Half));
        assert_eq!("QUARTER".parse::<SubpelPrecision>(), Ok(SubpelPrecision::Quarter));
    }

    #[test]
    fn subpel_precision_rejects_unknown_names() {
        let error = "eighth".parse::<SubpelPrecision>().unwrap_err();
        assert!(error.contains("off, half or quarter"), "{error}");
    }
}

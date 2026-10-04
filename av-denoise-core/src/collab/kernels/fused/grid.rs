use cubecl::prelude::*;

use crate::collab::{MAX_K, PATCH_SIZE};

/// One forward Haar level over `len` values of `vals`, spaced `stride` apart from `start`.
///
/// Approximations land in the first half of the span and details in the second.
#[cube]
fn haar_strided_level_fwd(
    vals: &mut Array<f32>,
    #[comptime] start: u32,
    #[comptime] stride: u32,
    #[comptime] len: u32,
) {
    let half = comptime!(len / 2);
    let mut snapshot = Array::<f32>::new(MAX_K as usize);
    #[unroll]
    for k in 0..len {
        snapshot[k as usize] = vals[comptime!(start + k * stride) as usize];
    }

    #[unroll]
    for p in 0..half {
        let first = snapshot[comptime!(2 * p) as usize];
        let second = snapshot[comptime!(2 * p + 1) as usize];
        vals[comptime!(start + p * stride) as usize] = (first + second) * std::f32::consts::FRAC_1_SQRT_2;
        vals[comptime!(start + (half + p) * stride) as usize] =
            (first - second) * std::f32::consts::FRAC_1_SQRT_2;
    }
}

/// The inverse of `haar_strided_level_fwd`.
#[cube]
fn haar_strided_level_inv(
    vals: &mut Array<f32>,
    #[comptime] start: u32,
    #[comptime] stride: u32,
    #[comptime] len: u32,
) {
    let half = comptime!(len / 2);
    let mut snapshot = Array::<f32>::new(MAX_K as usize);
    #[unroll]
    for k in 0..len {
        snapshot[k as usize] = vals[comptime!(start + k * stride) as usize];
    }

    #[unroll]
    for p in 0..half {
        let low = snapshot[p as usize];
        let high = snapshot[comptime!(half + p) as usize];
        vals[comptime!(start + 2 * p * stride) as usize] = (low + high) * std::f32::consts::FRAC_1_SQRT_2;
        vals[comptime!(start + (2 * p + 1) * stride) as usize] =
            (low - high) * std::f32::consts::FRAC_1_SQRT_2;
    }
}

/// Propagates independent variances through one Haar level.
///
/// Both outputs of a Haar pair carry the mean variance of its inputs.
#[cube]
fn variance_strided_level(
    vals: &mut Array<f32>,
    #[comptime] start: u32,
    #[comptime] stride: u32,
    #[comptime] len: u32,
) {
    let half = comptime!(len / 2);
    let mut snapshot = Array::<f32>::new(MAX_K as usize);
    #[unroll]
    for k in 0..len {
        snapshot[k as usize] = vals[comptime!(start + k * stride) as usize];
    }

    #[unroll]
    for p in 0..half {
        let first = snapshot[comptime!(2 * p) as usize];
        let second = snapshot[comptime!(2 * p + 1) as usize];
        let mean = (first + second) * 0.5f32;
        vals[comptime!(start + p * stride) as usize] = mean;
        vals[comptime!(start + (half + p) * stride) as usize] = mean;
    }
}

#[cube]
fn haar_strided_fwd(
    vals: &mut Array<f32>,
    #[comptime] start: u32,
    #[comptime] stride: u32,
    #[comptime] len: u32,
) {
    if comptime!(len >= 4) {
        haar_strided_level_fwd(vals, start, stride, 4u32);
    }
    if comptime!(len >= 2) {
        haar_strided_level_fwd(vals, start, stride, 2u32);
    }
}

#[cube]
fn haar_strided_inv(
    vals: &mut Array<f32>,
    #[comptime] start: u32,
    #[comptime] stride: u32,
    #[comptime] len: u32,
) {
    if comptime!(len >= 2) {
        haar_strided_level_inv(vals, start, stride, 2u32);
    }
    if comptime!(len >= 4) {
        haar_strided_level_inv(vals, start, stride, 4u32);
    }
}

#[cube]
fn variance_strided(
    vals: &mut Array<f32>,
    #[comptime] start: u32,
    #[comptime] stride: u32,
    #[comptime] len: u32,
) {
    if comptime!(len >= 4) {
        variance_strided_level(vals, start, stride, 4u32);
    }
    if comptime!(len >= 2) {
        variance_strided_level(vals, start, stride, 2u32);
    }
}

/// The separable time-by-volume Haar over a group of `MAX_K / grid_frames` volumes.
///
/// Member `s * grid_frames + t` is frame `t` of volume `s`. Each volume is transformed along time,
/// then every temporal coefficient is transformed across the volumes.
#[cube]
pub(crate) fn grid_fwd(stack: &mut Array<f32>, #[comptime] grid_frames: u32) {
    let volumes = comptime!(MAX_K / grid_frames);

    #[unroll]
    for pos in 0..PATCH_SIZE {
        let mut vals = Array::<f32>::new(MAX_K as usize);
        #[unroll]
        for m in 0..MAX_K {
            vals[m as usize] = stack[(m * PATCH_SIZE + pos) as usize];
        }

        #[unroll]
        for volume in 0..volumes {
            haar_strided_fwd(&mut vals, comptime!(volume * grid_frames), 1u32, grid_frames);
        }

        #[unroll]
        for frame in 0..grid_frames {
            haar_strided_fwd(&mut vals, frame, grid_frames, volumes);
        }

        #[unroll]
        for m in 0..MAX_K {
            stack[(m * PATCH_SIZE + pos) as usize] = vals[m as usize];
        }
    }
}

/// The inverse of `grid_fwd`.
#[cube]
pub(crate) fn grid_inv(stack: &mut Array<f32>, #[comptime] grid_frames: u32) {
    let volumes = comptime!(MAX_K / grid_frames);

    #[unroll]
    for pos in 0..PATCH_SIZE {
        let mut vals = Array::<f32>::new(MAX_K as usize);
        #[unroll]
        for m in 0..MAX_K {
            vals[m as usize] = stack[(m * PATCH_SIZE + pos) as usize];
        }

        #[unroll]
        for frame in 0..grid_frames {
            haar_strided_inv(&mut vals, frame, grid_frames, volumes);
        }

        #[unroll]
        for volume in 0..volumes {
            haar_strided_inv(&mut vals, comptime!(volume * grid_frames), 1u32, grid_frames);
        }

        #[unroll]
        for m in 0..MAX_K {
            stack[(m * PATCH_SIZE + pos) as usize] = vals[m as usize];
        }
    }
}

/// Propagates each member's variance to the coefficient it lands on under `grid_fwd`.
#[cube]
pub(crate) fn grid_variance(v: &mut Array<f32>, #[comptime] grid_frames: u32) {
    let volumes = comptime!(MAX_K / grid_frames);

    #[unroll]
    for volume in 0..volumes {
        variance_strided(v, comptime!(volume * grid_frames), 1u32, grid_frames);
    }

    #[unroll]
    for frame in 0..grid_frames {
        variance_strided(v, frame, grid_frames, volumes);
    }
}

/// One Haar level over `len` values spaced `stride` apart from `start`.
///
/// Approximations land in the first half of the span and details in the second.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
fn haar_level_host(vals: &mut [f32; 8], start: usize, stride: usize, len: usize, inverse: bool) {
    let half = len / 2;
    let snapshot: Vec<f32> = (0..len).map(|k| vals[start + k * stride]).collect();
    for p in 0..half {
        if inverse {
            let low = snapshot[p];
            let high = snapshot[half + p];
            vals[start + 2 * p * stride] = (low + high) * std::f32::consts::FRAC_1_SQRT_2;
            vals[start + (2 * p + 1) * stride] = (low - high) * std::f32::consts::FRAC_1_SQRT_2;
        } else {
            let first = snapshot[2 * p];
            let second = snapshot[2 * p + 1];
            vals[start + p * stride] = (first + second) * std::f32::consts::FRAC_1_SQRT_2;
            vals[start + (half + p) * stride] = (first - second) * std::f32::consts::FRAC_1_SQRT_2;
        }
    }
}

/// The dyadic levels of a span of `len`, in forward order.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
fn levels(len: usize) -> Vec<usize> {
    [4usize, 2].into_iter().filter(|&level| level <= len).collect()
}

#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) fn grid_fwd_host(column: &[f32; 8], grid_frames: u32) -> [f32; 8] {
    let frames = grid_frames as usize;
    let volumes = 8 / frames;
    let mut vals = *column;

    for volume in 0..volumes {
        for level in levels(frames) {
            haar_level_host(&mut vals, volume * frames, 1, level, false);
        }
    }

    for frame in 0..frames {
        for level in levels(volumes) {
            haar_level_host(&mut vals, frame, frames, level, false);
        }
    }

    vals
}

#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) fn grid_inv_host(column: &[f32; 8], grid_frames: u32) -> [f32; 8] {
    let frames = grid_frames as usize;
    let volumes = 8 / frames;
    let mut vals = *column;

    for frame in 0..frames {
        for level in levels(volumes).into_iter().rev() {
            haar_level_host(&mut vals, frame, frames, level, true);
        }
    }

    for volume in 0..volumes {
        for level in levels(frames).into_iter().rev() {
            haar_level_host(&mut vals, volume * frames, 1, level, true);
        }
    }

    vals
}

/// Propagates independent per-member variances through `grid_fwd_host`.
///
/// Both outputs of a Haar pair carry the mean variance of its inputs.
#[cfg(all(test, any(feature = "vulkan", feature = "metal")))]
pub(crate) fn grid_variance_host(variances: &[f32; 8], grid_frames: u32) -> [f32; 8] {
    let frames = grid_frames as usize;
    let volumes = 8 / frames;
    let mut vals = *variances;
    let average = |vals: &mut [f32; 8], start: usize, stride: usize, len: usize| {
        let half = len / 2;
        let snapshot: Vec<f32> = (0..len).map(|k| vals[start + k * stride]).collect();
        for p in 0..half {
            let mean = (snapshot[2 * p] + snapshot[2 * p + 1]) * 0.5;
            vals[start + p * stride] = mean;
            vals[start + (half + p) * stride] = mean;
        }
    };

    for volume in 0..volumes {
        for level in levels(frames) {
            average(&mut vals, volume * frames, 1, level);
        }
    }

    for frame in 0..frames {
        for level in levels(volumes) {
            average(&mut vals, frame, frames, level);
        }
    }

    vals
}

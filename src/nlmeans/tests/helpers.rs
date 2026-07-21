use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

pub(super) type R = WgpuRuntime;

pub(super) fn make_client() -> ComputeClient<R> {
    let device = <R as Runtime>::Device::default();
    R::client(&device)
}

pub(super) fn make_uniform_frame(w: u32, h: u32, ch: u32, val: f32) -> Vec<f32> {
    vec![val; (w * h * ch) as usize]
}

/// Creates a frame with a patch of noise (not just a single pixel)
/// so that NLMeans has matching noisy patches to work with.
#[allow(clippy::too_many_arguments)]
pub(super) fn make_frame_with_noisy_region(
    w: u32,
    h: u32,
    ch: u32,
    base: f32,
    cx: u32,
    cy: u32,
    radius: u32,
    noise_val: f32,
) -> Vec<f32> {
    let mut frame = vec![base; (w * h * ch) as usize];

    for dy in 0..=radius * 2 {
        for dx in 0..=radius * 2 {
            let x = cx + dx - radius;
            let y = cy + dy - radius;

            if x < w && y < h {
                for c in 0..ch {
                    frame[((y * w + x) * ch + c) as usize] = noise_val;
                }
            }
        }
    }

    frame
}

/// Flat base value plus deterministic pseudo-Gaussian noise, densely
/// packed as `pixels * ch` (no channel padding). Each sample sums four
/// hash-derived uniforms in `[-0.5, 0.5]` (Irwin-Hall) and rescales to
/// the requested per-channel standard deviation, so the same arguments
/// always reproduce the same frame. `sigmas` is indexed per channel and
/// wraps if shorter than `ch`.
pub(super) fn make_noisy_gaussian_frame(w: u32, h: u32, ch: u32, base: f32, sigmas: &[f32]) -> Vec<f32> {
    let mut frame = vec![0.0f32; (w * h * ch) as usize];
    // Sum of 4 independent Uniform(-0.5, 0.5) samples has variance 4/12 = 1/3.
    let unit_std = (1.0f32 / 3.0f32).sqrt();

    for idx in 0..(w * h * ch) {
        let mut sum = 0.0f32;
        for k in 0..4u32 {
            let mut hash = (idx * 4 + k).wrapping_mul(2654435761).wrapping_add(0x9E3779B9);
            hash ^= hash >> 15;
            hash = hash.wrapping_mul(0x85EBCA6B);
            hash ^= hash >> 13;
            sum += (hash as f32 / u32::MAX as f32) - 0.5;
        }
        let c = (idx % ch) as usize;
        let sigma = sigmas[c % sigmas.len()];
        frame[idx as usize] = (base + (sum / unit_std) * sigma).clamp(0.0, 1.0);
    }

    frame
}

/// Independent pseudo-Gaussian noise sample at `(idx, seed)`, decorrelated
/// across `seed` values so two calls with the same `size`/`base`/`sigma`
/// but different `seed`s produce two independently-noisy copies of the
/// same clean signal. Same Irwin-Hall construction as
/// `make_noisy_gaussian_frame`, which is deterministic in position alone
/// and so can't produce a *second*, decorrelated sample.
pub(super) fn noisy_copy(size: u32, base: f32, sigma: f32, seed: u32) -> Vec<f32> {
    let unit_std = (1.0f32 / 3.0f32).sqrt();
    let mut frame = vec![0.0f32; (size * size) as usize];
    for idx in 0..(size * size) {
        let mut sum = 0.0f32;
        for k in 0..4u32 {
            let mut hash = (idx * 4 + k)
                .wrapping_mul(2654435761)
                .wrapping_add(seed.wrapping_mul(0x9E37_79B9).wrapping_add(k));
            hash ^= hash >> 15;
            hash = hash.wrapping_mul(0x85EB_CA6B);
            hash ^= hash >> 13;
            sum += (hash as f32 / u32::MAX as f32) - 0.5;
        }
        frame[idx as usize] = (base + (sum / unit_std) * sigma).clamp(0.0, 1.0);
    }
    frame
}

/// Smooth horizontal luma gradient from `lo` to `hi` inclusive,
/// replicated down every row.
pub(super) fn make_gradient_frame(w: u32, h: u32, lo: f32, hi: f32) -> Vec<f32> {
    let mut frame = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let t = x as f32 / (w - 1).max(1) as f32;
            frame[(y * w + x) as usize] = lo + (hi - lo) * t;
        }
    }
    frame
}

/// Expands a densely packed `pixels * ch` frame into the padded
/// `pixels * stored_ch` GPU storage layout used by `Vector`-typed
/// kernel buffers (extra lanes zeroed). Mirrors the padding
/// `NlmDenoiser::upload_into_slot` applies internally.
pub(super) fn pad_channels(dense: &[f32], pixels: usize, ch: u32, stored_ch: u32) -> Vec<f32> {
    if ch == stored_ch {
        return dense.to_vec();
    }
    let ch = ch as usize;
    let stored_ch = stored_ch as usize;
    let mut out = vec![0.0f32; pixels * stored_ch];
    for p in 0..pixels {
        out[p * stored_ch..p * stored_ch + ch].copy_from_slice(&dense[p * ch..p * ch + ch]);
    }
    out
}

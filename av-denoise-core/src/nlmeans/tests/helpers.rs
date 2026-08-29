use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

pub(super) use crate::nlmeans::align::StorageAlign;
#[cfg(feature = "vulkan")]
use crate::{ChannelMode, Denoiser, DenoiserOptions, DenoisingMode, accelerate::Accelerator, device::Device};

pub(super) type R = WgpuRuntime;

pub(super) fn make_client() -> ComputeClient<R> {
    let device = <R as Runtime>::Device::default();
    R::client(&device)
}

/// Buffer-binding alignment the test runtime reports, the same value
/// `NlmDenoiser` lays its per-slot buffers out against.
pub(super) fn test_align() -> StorageAlign {
    StorageAlign::from_client(&make_client())
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

/// Adds independent pseudo-Gaussian noise to an arbitrary `w * h`
/// clean field, clamping once after. Each sample sums four
/// hash-derived uniforms in `[-0.5, 0.5]` (Irwin-Hall) and rescales to
/// the requested standard deviation, decorrelated across `seed` values
/// so two calls with different `seed`s over the *same* `clean` field
/// produce two independently noisy copies of it.
pub(super) fn noisy_field_over(clean: &[f32], w: u32, h: u32, sigma: f32, seed: u32) -> Vec<f32> {
    let unit_std = (1.0f32 / 3.0f32).sqrt();
    let mut frame = vec![0.0f32; (w * h) as usize];
    for idx in 0..(w * h) {
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
        frame[idx as usize] = (clean[idx as usize] + (sum / unit_std) * sigma).clamp(0.0, 1.0);
    }
    frame
}

/// Independent pseudo-Gaussian noise sample at `(idx, seed)` over a
/// flat `base` field, decorrelated across `seed` values so two calls
/// with the same `size`/`base`/`sigma` but different `seed`s produce
/// two independently-noisy copies of the same clean signal. A thin
/// wrapper over [`noisy_field_over`] for the common flat-field case.
pub(super) fn noisy_copy(size: u32, base: f32, sigma: f32, seed: u32) -> Vec<f32> {
    noisy_field_over(&vec![base; (size * size) as usize], size, size, sigma, seed)
}

/// Builds a frame of grain that is correlated between neighbouring
/// pixels.
///
/// An independent white noise field is blurred horizontally with a
/// `[0.25, 0.5, 0.25]` kernel, clamped at the edges, then added to
/// `base` and clamped once more.
///
/// The blur is identical every call, so two calls with different seeds
/// are independent of each other while each carries the blur's own
/// spatial correlation.
///
/// That works out at a lag-1 correlation of two thirds for any input
/// distribution, with the variance scaled by 0.375, the sum of the taps
/// squared.
pub(super) fn correlated_noisy_frame(w: u32, h: u32, base: f32, sigma_pre: f32, seed: u32) -> Vec<f32> {
    let unit_std = (1.0f32 / 3.0f32).sqrt();
    let mut raw = vec![0.0f32; (w * h) as usize];
    for idx in 0..(w * h) {
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
        raw[idx as usize] = (sum / unit_std) * sigma_pre;
    }

    let mut out = vec![0.0f32; raw.len()];
    for y in 0..h {
        for x in 0..w {
            let xl = x.saturating_sub(1);
            let xr = (x + 1).min(w - 1);
            let l = raw[(y * w + xl) as usize];
            let c = raw[(y * w + x) as usize];
            let r = raw[(y * w + xr) as usize];
            let blurred = 0.25 * l + 0.5 * c + 0.25 * r;
            out[(y * w + x) as usize] = (base + blurred).clamp(0.0, 1.0);
        }
    }
    out
}

/// Builds grain correlated the same way [`correlated_noisy_frame`] does,
/// with a tunable horizontal blur tap `a` (kernel `[a, 1 - 2a, a]`)
/// instead of that function's fixed `0.25`.
///
/// For taps `(a, b, a)` applied to unit-variance white noise, the
/// lag-1 correlation along x works out to `2*a*b / (2*a^2 + b^2)`.
/// `a = 0.25` reduces to `correlated_noisy_frame`'s two thirds; `a =
/// 0.125` gives a predicted correlation of about `0.316`, an
/// intermediate value between the uncorrelated and two-thirds cases.
/// Both are confirmed empirically, not just derived, in
/// `residual_correlation.rs`.
pub(super) fn correlated_noisy_frame_with_tap(
    w: u32,
    h: u32,
    base: f32,
    sigma_pre: f32,
    seed: u32,
    a: f32,
) -> Vec<f32> {
    let b = 1.0 - 2.0 * a;
    let unit_std = (1.0f32 / 3.0f32).sqrt();
    let mut raw = vec![0.0f32; (w * h) as usize];
    for idx in 0..(w * h) {
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
        raw[idx as usize] = (sum / unit_std) * sigma_pre;
    }

    let mut out = vec![0.0f32; raw.len()];
    for y in 0..h {
        for x in 0..w {
            let xl = x.saturating_sub(1);
            let xr = (x + 1).min(w - 1);
            let l = raw[(y * w + xl) as usize];
            let c = raw[(y * w + x) as usize];
            let r = raw[(y * w + xr) as usize];
            let blurred = a * l + b * c + a * r;
            out[(y * w + x) as usize] = (base + blurred).clamp(0.0, 1.0);
        }
    }
    out
}

/// A deterministic frame with real spatial structure at more than one
/// scale, rather than a flat field or a smooth gradient.
///
/// Two out-of-phase sine waves plus a finer third one give NLMeans
/// patches with genuinely varying content to match against, so a test
/// built over this frame exercises the same weight spread real footage
/// produces instead of the uniform, always-maximal weights a flat frame
/// hands every candidate.
pub(super) fn make_textured_frame(w: u32, h: u32) -> Vec<f32> {
    let mut frame = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            let v = 0.5
                + 0.2 * (fx * 8.0 * std::f32::consts::PI).sin() * (fy * 6.0 * std::f32::consts::PI).cos()
                + 0.1 * (fx * 20.0 * std::f32::consts::PI).sin();
            frame[(y * w + x) as usize] = v.clamp(0.05, 0.95);
        }
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
/// Builds a top-level [`Denoiser`] at the given temporal radius over
/// Luma, with every other option left at its default, on the Vulkan
/// accelerator.
#[cfg(feature = "vulkan")]
pub(super) fn test_denoiser(radius: u32, w: u32, h: u32) -> Denoiser {
    let opts = DenoiserOptions::builder()
        .channel_mode(ChannelMode::Luma)
        .mode(DenoisingMode::Temporal { radius })
        .build();
    Denoiser::create(&[Accelerator::Vulkan], &Device::Default, w, h, opts)
        .expect("denoiser construction failed")
}

/// A deterministic frame whose pixels ramp from `0.2` to `0.8` across
/// the row and shift a little with `i`, so a sequence built from
/// increasing `i` gives every frame in the window distinct content.
pub(super) fn ramp_frame(w: u32, h: u32, i: usize) -> Vec<f32> {
    let mut frame = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let t = (x as f32 + y as f32 * w as f32) / (w * h) as f32;
            frame[(y * w + x) as usize] = (0.2 + 0.6 * t + i as f32 * 0.01).clamp(0.0, 1.0);
        }
    }
    frame
}

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

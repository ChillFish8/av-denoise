use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

pub(super) type R = WgpuRuntime;

pub(super) fn make_client() -> ComputeClient<R> {
    let device = <R as Runtime>::Device::default();
    R::client(&device)
}

/// Adds independent pseudo-Gaussian noise to a flat `base` field.
///
/// Each sample sums four hash-derived uniforms in `[-0.5, 0.5]`
/// (Irwin-Hall) and rescales to the requested standard deviation, so the
/// same arguments always reproduce the same frame. Ported from
/// `src/nlmeans/tests/helpers.rs`, keeping only the flat-field case this
/// tree needs.
pub(super) fn noisy_field_over(w: u32, h: u32, base: f32, sigma: f32) -> Vec<f32> {
    let unit_std = (1.0f32 / 3.0f32).sqrt();
    let mut frame = vec![0.0f32; (w * h) as usize];
    for idx in 0..(w * h) {
        let mut sum = 0.0f32;
        for k in 0..4u32 {
            let mut hash = (idx * 4 + k).wrapping_mul(2654435761).wrapping_add(0x9E3779B9);
            hash ^= hash >> 15;
            hash = hash.wrapping_mul(0x85EBCA6B);
            hash ^= hash >> 13;
            sum += (hash as f32 / u32::MAX as f32) - 0.5;
        }
        frame[idx as usize] = base + (sum / unit_std) * sigma;
    }
    frame
}

/// A flat base value with a per-pixel hash offset that never repeats
/// across the frame.
///
/// Any two distinct 8x8 windows into this frame differ in most of their
/// 64 pixels, so a tiny admission threshold rejects every candidate but
/// the reference patch itself. The horizontal ramp on its own would
/// still leave two vertically-shifted patches identical, since the
/// frame would otherwise repeat down every row, so the hash term is
/// what actually makes every position unique.
pub(super) fn make_unique_frame(w: u32, h: u32) -> Vec<f32> {
    let mut frame = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let mut hash = idx.wrapping_mul(2654435761).wrapping_add(0x9E3779B9);
            hash ^= hash >> 15;
            hash = hash.wrapping_mul(0x85EBCA6B);
            hash ^= hash >> 13;
            let offset = hash as f32 / u32::MAX as f32;
            frame[idx as usize] = x as f32 * 10.0 + offset * 10.0;
        }
    }
    frame
}

/// Writes an 8x8 patch into `frame` with its top-left corner at `(px,
/// py)`.
pub(super) fn plant_patch(frame: &mut [f32], w: u32, px: u32, py: u32, patch: &[f32; 64]) {
    for row in 0..8u32 {
        for col in 0..8u32 {
            let idx = (py + row) * w + (px + col);
            frame[idx as usize] = patch[(row * 8 + col) as usize];
        }
    }
}

/// A deterministic 8x8 texture with values well clear of the flat
/// backgrounds these tests plant it over.
pub(super) fn deterministic_texture(seed: u32) -> [f32; 64] {
    let mut out = [0.0f32; 64];
    for (idx, v) in out.iter_mut().enumerate() {
        let mut hash = (idx as u32)
            .wrapping_mul(2654435761)
            .wrapping_add(seed.wrapping_mul(0x9E37_79B9));
        hash ^= hash >> 15;
        hash = hash.wrapping_mul(0x85EBCA6B);
        hash ^= hash >> 13;
        *v = 0.6 + (hash as f32 / u32::MAX as f32) * 0.3;
    }
    out
}

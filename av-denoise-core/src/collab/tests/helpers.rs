use cubecl::prelude::*;

pub(super) fn make_client() -> Client {
    crate::tests::compilation_cache::install_test_compilation_cache();
    crate::testing::test_client()
}

/// Adds independent pseudo-Gaussian noise to a flat `base` field.
///
/// Each sample sums four hash-derived uniforms between -0.5 and 0.5 (Irwin-Hall) and rescales to
/// the requested standard deviation, so the same arguments always reproduce the same frame.
pub(super) fn noisy_flat_field(width: u32, height: u32, base: f32, sigma: f32) -> Vec<f32> {
    let unit_std = (1.0f32 / 3.0f32).sqrt();
    let mut frame = vec![0.0f32; (width * height) as usize];
    for idx in 0..(width * height) {
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

/// A horizontal ramp plus a per-pixel hash offset, so no 8x8 window repeats anywhere in the frame.
///
/// Any two distinct windows differ in most of their 64 pixels, so a tiny admission threshold rejects
/// every candidate but the reference patch itself. The ramp alone repeats down every row, so the hash
/// term is what makes vertically shifted patches differ.
pub(super) fn make_unique_frame(width: u32, height: u32) -> Vec<f32> {
    let mut frame = vec![0.0f32; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
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

/// Writes an 8x8 patch into `frame` with its top-left corner at `(left, top)`.
pub(super) fn plant_patch(frame: &mut [f32], width: u32, left: u32, top: u32, patch: &[f32; 64]) {
    for row in 0..8u32 {
        for col in 0..8u32 {
            let idx = (top + row) * width + (left + col);
            frame[idx as usize] = patch[(row * 8 + col) as usize];
        }
    }
}

/// A deterministic 8x8 texture with values well clear of the flat backgrounds the tests plant it over.
pub(super) fn deterministic_texture(seed: u32) -> [f32; 64] {
    let mut texture = [0.0f32; 64];
    for (idx, value) in texture.iter_mut().enumerate() {
        let mut hash = (idx as u32)
            .wrapping_mul(2654435761)
            .wrapping_add(seed.wrapping_mul(0x9E37_79B9));
        hash ^= hash >> 15;
        hash = hash.wrapping_mul(0x85EBCA6B);
        hash ^= hash >> 13;
        *value = 0.6 + (hash as f32 / u32::MAX as f32) * 0.3;
    }

    texture
}

/// A frame where every pixel holds `value`.
pub(super) fn constant_frame(width: u32, height: u32, value: f32) -> Vec<f32> {
    vec![value; (width * height) as usize]
}

/// [make_unique_frame] content, rescaled to between 0 and 1, between black bars of `bar_rows` rows at
/// the top and bottom.
///
/// Every candidate inside a bar ties at distance zero, which is where search order decides members.
pub(super) fn letterboxed_frame(width: u32, height: u32, bar_rows: u32) -> Vec<f32> {
    let raw = make_unique_frame(width, height);
    let peak = raw.iter().copied().fold(0.0f32, f32::max);
    let mut frame = vec![0.0f32; (width * height) as usize];

    for y in bar_rows..height - bar_rows {
        for x in 0..width {
            let index = (y * width + x) as usize;
            frame[index] = raw[index] / peak;
        }
    }

    frame
}

/// Flat cells of `cell` pixels in four grey levels, like flat-shaded animation.
pub(super) fn cel_frame(width: u32, height: u32, cell: u32) -> Vec<f32> {
    let levels = [0.2f32, 0.45, 0.7, 0.9];
    let mut frame = vec![0.0f32; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let cell_index = (y / cell) * 7 + (x / cell) * 3;
            let level = levels[(cell_index % 4) as usize];
            frame[(y * width + x) as usize] = level;
        }
    }

    frame
}

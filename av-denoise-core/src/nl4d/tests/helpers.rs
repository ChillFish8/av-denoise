use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use crate::nlmeans::motion::neighbour_idx_for_k;

/// A non-flat luma field, built from two out-of-phase sine waves rather
/// than noise, so it carries real spatial structure a denoiser can
/// either preserve or destroy.
pub(super) fn textured_base(w: u32, h: u32) -> Vec<f32> {
    let mut frame = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            let v = 0.5
                + 0.15 * (fx * 6.0 * std::f32::consts::PI).sin() * (fy * 4.0 * std::f32::consts::PI).cos();
            frame[(y * w + x) as usize] = v.clamp(0.05, 0.95);
        }
    }
    frame
}

/// Adds independent pseudo-Gaussian noise to `base`, decorrelated across
/// `seed` so different seeds over the same base give independently
/// noisy copies of the same clean content.
pub(super) fn noisy_copy_of(base: &[f32], w: u32, h: u32, sigma: f32, seed: u32) -> Vec<f32> {
    let unit_std = (1.0f32 / 3.0f32).sqrt();
    let mut frame = vec![0.0f32; base.len()];
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
        frame[idx as usize] = (base[idx as usize] + (sum / unit_std) * sigma).clamp(0.0, 1.0);
    }
    frame
}

/// PSNR between two equal-length planes, in dB.
pub(super) fn psnr(a: &[f32], b: &[f32]) -> f64 {
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    if mse <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (1.0f64 / mse).log10()
}

pub(super) type R = WgpuRuntime;

pub(super) fn make_client() -> ComputeClient<R> {
    let device = <R as Runtime>::Device::default();
    R::client(&device)
}

/// The block step and grid this tree's fixtures use, matching
/// [`crate::collab::PATCH_SIZE`] so a block boundary always lines up
/// with a patch boundary.
pub(super) const BLK_STEP: u32 = 8;

/// A ring of `2 * radius + 1` frames, one physical slot per logical
/// temporal offset from `-radius` to `radius`, with the centre frame at
/// physical slot `radius`.
///
/// Every field is already shaped the way
/// [`crate::collab::kernels::fused::collab_fused`] expects to read it,
/// so a test only has to upload each `Vec` and launch.
pub(super) struct RingFixture {
    pub ring: Vec<f32>,
    pub mv_field: Vec<i32>,
    pub confidence: Vec<f32>,
    pub neighbour_slots: Vec<u32>,
    pub centre_slot: u32,
    pub radius: u32,
    pub blocks_x: u32,
    pub blocks_y: u32,
    pub mv_stride: u32,
    pub conf_stride: u32,
    pub width: u32,
    pub height: u32,
}

/// A deterministic 8x8 texture with values well clear of the flat
/// background these fixtures plant it over.
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

/// Writes an 8x8 patch into `frame` with its top-left corner at `(px,
/// py)`.
fn plant_patch(frame: &mut [f32], w: u32, px: u32, py: u32, patch: &[f32; 64]) {
    for row in 0..8u32 {
        for col in 0..8u32 {
            let idx = (py + row) * w + (px + col);
            frame[idx as usize] = patch[(row * 8 + col) as usize];
        }
    }
}

/// Builds a ring whose centre frame carries a distinctive 8x8 patch at
/// `ref_pos`, and whose neighbour frame for logical offset `k` carries
/// the same patch shifted by `shift_per_k * k` pixels on the x axis.
///
/// The motion field is seeded to predict exactly that shift, at the
/// block covering `ref_pos`, so a correct search recovers the planted
/// patch through the motion prediction, not through luck.
///
/// `conf` gives the per-neighbour confidence written into every block of
/// that neighbour's plane, keyed by the same logical offset `k` the
/// shift is keyed by.
#[expect(clippy::too_many_arguments)]
pub(super) fn planted_ring(
    w: u32,
    h: u32,
    radius: u32,
    ref_pos: (u32, u32),
    shift_per_k: i32,
    patch: &[f32; 64],
    background: f32,
    conf: impl Fn(i32) -> f32,
) -> RingFixture {
    let n_frames = 2 * radius + 1;
    let centre_slot = radius;
    let blocks_x = w.div_ceil(BLK_STEP);
    let blocks_y = h.div_ceil(BLK_STEP);
    let mv_stride = blocks_x * blocks_y * 2;
    let conf_stride = blocks_x * blocks_y;

    let (rx, ry) = ref_pos;
    let bx = rx / BLK_STEP;
    let by = ry / BLK_STEP;
    let block = by * blocks_x + bx;

    let mut ring = vec![0.0f32; (n_frames * w * h) as usize];
    for slot in 0..n_frames {
        let k = slot as i32 - radius as i32;
        let frame = &mut ring[(slot * w * h) as usize..((slot + 1) * w * h) as usize];
        frame.fill(background);
        let px = (rx as i32 + shift_per_k * k) as u32;
        plant_patch(frame, w, px, ry, patch);
    }

    let mut mv_field = vec![0i32; (2 * radius * mv_stride) as usize];
    let mut confidence = vec![0.0f32; (2 * radius * conf_stride) as usize];
    let mut neighbour_slots = vec![0u32; (2 * radius) as usize];
    for k in -(radius as i32)..=(radius as i32) {
        if k == 0 {
            continue;
        }
        let t = neighbour_idx_for_k(radius, k);
        let slot = (k + radius as i32) as u32;
        neighbour_slots[t as usize] = slot;

        let mv_base = (t * mv_stride + block * 2) as usize;
        mv_field[mv_base] = shift_per_k * k;
        mv_field[mv_base + 1] = 0;

        let c_base = t * conf_stride;
        confidence[c_base as usize..(c_base + conf_stride) as usize].fill(conf(k));
    }

    RingFixture {
        ring,
        mv_field,
        confidence,
        neighbour_slots,
        centre_slot,
        radius,
        blocks_x,
        blocks_y,
        mv_stride,
        conf_stride,
        width: w,
        height: h,
    }
}

/// A ring of independent pseudo-random frames, with a zeroed motion
/// field and uniform confidence.
///
/// No 8x8 window into this ring resembles any other, on any frame, so
/// every candidate a search finds is a poor match. It exists for the
/// no-admission-gate test, where the point is that the group still
/// fills to `k_max` despite that.
pub(super) fn noisy_ring(w: u32, h: u32, radius: u32, confidence_value: f32) -> RingFixture {
    let n_frames = 2 * radius + 1;
    let centre_slot = radius;
    let blocks_x = w.div_ceil(BLK_STEP);
    let blocks_y = h.div_ceil(BLK_STEP);
    let mv_stride = blocks_x * blocks_y * 2;
    let conf_stride = blocks_x * blocks_y;

    let mut ring = vec![0.0f32; (n_frames * w * h) as usize];
    for (idx, v) in ring.iter_mut().enumerate() {
        let mut hash = (idx as u32).wrapping_mul(2654435761).wrapping_add(0x9E3779B9);
        hash ^= hash >> 15;
        hash = hash.wrapping_mul(0x85EBCA6B);
        hash ^= hash >> 13;
        *v = hash as f32 / u32::MAX as f32;
    }

    let mv_field = vec![0i32; (2 * radius * mv_stride) as usize];
    let confidence = vec![confidence_value; (2 * radius * conf_stride) as usize];
    let mut neighbour_slots = vec![0u32; (2 * radius) as usize];
    for k in -(radius as i32)..=(radius as i32) {
        if k == 0 {
            continue;
        }
        let t = neighbour_idx_for_k(radius, k);
        neighbour_slots[t as usize] = (k + radius as i32) as u32;
    }

    RingFixture {
        ring,
        mv_field,
        confidence,
        neighbour_slots,
        centre_slot,
        radius,
        blocks_x,
        blocks_y,
        mv_stride,
        conf_stride,
        width: w,
        height: h,
    }
}

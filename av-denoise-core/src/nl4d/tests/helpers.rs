use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use crate::nl4d::Nl4dParams;
use crate::nlmeans::motion::neighbour_idx_for_k;
use crate::nlmeans::tests::helpers::noisy_field_over;
use crate::nlmeans::{
    ChannelMode,
    HqParams,
    MotionCompensationMode,
    MotionEstimation,
    NlmParams,
    PrefilterMode,
};

pub(super) type R = WgpuRuntime;

pub(super) const SIGMA: f32 = 6.0 / 255.0;
pub(super) const SPATIAL_RADIUS: u32 = 9;
pub(super) const REFINE: u32 = 2;
pub(super) const C_MIN: f32 = 0.05;
pub(super) const LAMBDA_HT: f32 = 2.7;

/// The motion block step, equal to [PATCH_SIZE](crate::collab::PATCH_SIZE) so a block boundary
/// always lines up with a patch boundary.
pub(super) const BLK_STEP: u32 = 8;

/// Parameters for a still clip with sigma pinned to [SIGMA].
pub(super) fn static_clip_params(temporal_radius: u32) -> Nl4dParams {
    let motion_compensation = MotionCompensationMode::Mvtools {
        blksize: 16,
        overlap: 8,
        search_radius: 4,
        pyramid_levels: 2,
        estimation: MotionEstimation::Auto,
    };
    let hq = HqParams::with_sigma(SIGMA);
    let nlm = NlmParams {
        temporal_radius,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation,
        hq: Some(hq),
    };

    Nl4dParams {
        nlm,
        temporal_radius,
        refine: REFINE,
        spatial_radius: SPATIAL_RADIUS,
        lambda_ht: LAMBDA_HT,
        c_min: C_MIN,
        // The shipped default, so these run the aggregation a real caller gets.
        kaiser_beta: 2.0,
        field_lambda: 0.0,
        // No effect here, since sigma is pinned.
        noise_map: true,
        flat_boost: 1.5,
        chroma_flat_boost: 1.5,
        shadow_soften: 0.65,
        // Off, so the pipeline tests keep the flat map their expectations were recorded against.
        flat_texture_cut: 1.0,
        // Off, so the pipeline tests keep the per-coefficient kernel their expectations were
        // recorded against.
        pooled_threshold: false,
        grain_export: false,
    }
}

/// A non-flat luma field built from two out-of-phase sine waves.
///
/// It carries real spatial structure rather than noise, which a denoiser can either preserve or
/// destroy.
pub(crate) fn textured_base(width: u32, height: u32) -> Vec<f32> {
    let mut frame = vec![0.0f32; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let x_fraction = x as f32 / width as f32;
            let y_fraction = y as f32 / height as f32;
            let value = 0.5
                + 0.15
                    * (x_fraction * 6.0 * std::f32::consts::PI).sin()
                    * (y_fraction * 4.0 * std::f32::consts::PI).cos();
            frame[(y * width + x) as usize] = value.clamp(0.05, 0.95);
        }
    }

    frame
}

/// PSNR between two equal-length planes, in dB.
pub(super) fn psnr(output: &[f32], reference: &[f32]) -> f64 {
    let mse: f64 = output
        .iter()
        .zip(reference.iter())
        .map(|(&out, &expected)| (out as f64 - expected as f64).powi(2))
        .sum::<f64>()
        / output.len() as f64;
    if mse <= 0.0 {
        return f64::INFINITY;
    }

    10.0 * (1.0f64 / mse).log10()
}

pub(super) fn make_client() -> ComputeClient<R> {
    let device = <R as Runtime>::Device::default();
    R::client(&device)
}

/// A ring of `2 * radius + 1` frames with the centre frame at physical slot `radius`.
///
/// Every field is already shaped the way
/// [collab_fused](crate::collab::kernels::fused::collab_fused) reads it, so a test only uploads
/// each `Vec` and launches.
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

/// A deterministic 8x8 texture with values well clear of the flat background it is planted over.
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

/// Writes an 8x8 patch into `frame` with its top-left corner at `(x, y)`.
fn plant_patch(frame: &mut [f32], width: u32, x: u32, y: u32, patch: &[f32; 64]) {
    for row in 0..8u32 {
        for col in 0..8u32 {
            let idx = (y + row) * width + (x + col);
            frame[idx as usize] = patch[(row * 8 + col) as usize];
        }
    }
}

/// Builds a ring whose centre frame carries `patch` at `ref_pos`, and whose neighbour at logical
/// offset `k` carries it shifted by `shift_per_k * k` pixels along x.
///
/// The motion field predicts exactly that shift at the block covering `ref_pos`, so a correct
/// search recovers the patch through the prediction, not through luck. `confidence_for(k)` is
/// written into every block of neighbour `k`'s confidence plane.
#[expect(
    clippy::too_many_arguments,
    reason = "the test helper takes the full set of parameters its cases vary"
)]
pub(super) fn planted_ring(
    width: u32,
    height: u32,
    radius: u32,
    ref_pos: (u32, u32),
    shift_per_k: i32,
    patch: &[f32; 64],
    background: f32,
    confidence_for: impl Fn(i32) -> f32,
) -> RingFixture {
    let frame_count = 2 * radius + 1;
    let centre_slot = radius;
    let blocks_x = width.div_ceil(BLK_STEP);
    let blocks_y = height.div_ceil(BLK_STEP);
    let mv_stride = blocks_x * blocks_y * 2;
    let conf_stride = blocks_x * blocks_y;

    let (ref_x, ref_y) = ref_pos;
    let block_x = ref_x / BLK_STEP;
    let block_y = ref_y / BLK_STEP;
    let block = block_y * blocks_x + block_x;

    let mut ring = vec![0.0f32; (frame_count * width * height) as usize];
    for slot in 0..frame_count {
        let k = slot as i32 - radius as i32;
        let frame = &mut ring[(slot * width * height) as usize..((slot + 1) * width * height) as usize];
        frame.fill(background);
        let patch_x = (ref_x as i32 + shift_per_k * k) as u32;
        plant_patch(frame, width, patch_x, ref_y, patch);
    }

    let mut mv_field = vec![0i32; (2 * radius * mv_stride) as usize];
    let mut confidence = vec![0.0f32; (2 * radius * conf_stride) as usize];
    let mut neighbour_slots = vec![0u32; (2 * radius) as usize];
    for k in -(radius as i32)..=(radius as i32) {
        if k == 0 {
            continue;
        }

        let neighbour = neighbour_idx_for_k(radius, k);
        let slot = (k + radius as i32) as u32;
        neighbour_slots[neighbour as usize] = slot;

        let mv_base = (neighbour * mv_stride + block * 2) as usize;
        mv_field[mv_base] = shift_per_k * k;
        mv_field[mv_base + 1] = 0;

        let conf_base = neighbour * conf_stride;
        let neighbour_confidence = confidence_for(k);
        confidence[conf_base as usize..(conf_base + conf_stride) as usize].fill(neighbour_confidence);
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
        width,
        height,
    }
}

/// A ring of independent pseudo-random frames, with a zeroed motion field and uniform confidence.
///
/// No 8x8 window into this ring resembles any other, on any frame, so every candidate a search
/// finds is a poor match.
pub(super) fn noisy_ring(width: u32, height: u32, radius: u32, confidence_value: f32) -> RingFixture {
    let frame_count = 2 * radius + 1;
    let centre_slot = radius;
    let blocks_x = width.div_ceil(BLK_STEP);
    let blocks_y = height.div_ceil(BLK_STEP);
    let mv_stride = blocks_x * blocks_y * 2;
    let conf_stride = blocks_x * blocks_y;

    let mut ring = vec![0.0f32; (frame_count * width * height) as usize];
    for (idx, value) in ring.iter_mut().enumerate() {
        let mut hash = (idx as u32).wrapping_mul(2654435761).wrapping_add(0x9E3779B9);
        hash ^= hash >> 15;
        hash = hash.wrapping_mul(0x85EBCA6B);
        hash ^= hash >> 13;
        *value = hash as f32 / u32::MAX as f32;
    }

    let mv_field = vec![0i32; (2 * radius * mv_stride) as usize];
    let confidence = vec![confidence_value; (2 * radius * conf_stride) as usize];
    let mut neighbour_slots = vec![0u32; (2 * radius) as usize];
    for k in -(radius as i32)..=(radius as i32) {
        if k == 0 {
            continue;
        }

        let neighbour = neighbour_idx_for_k(radius, k);
        neighbour_slots[neighbour as usize] = (k + radius as i32) as u32;
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
        width,
        height,
    }
}

/// `count` interleaved frames of a drifting texture with independent noise per frame and channel.
pub(crate) fn noisy_frames(width: u32, height: u32, channels: u32, count: usize) -> Vec<Vec<f32>> {
    let base = textured_base(width + count as u32, height);
    let mut frames = Vec::with_capacity(count);

    for index in 0..count {
        let mut frame = vec![0.0f32; (width * height * channels) as usize];

        for channel in 0..channels {
            let mut clean = Vec::with_capacity((width * height) as usize);
            for row in 0..height {
                let start = (row * (width + count as u32) + index as u32) as usize;
                clean.extend_from_slice(&base[start..start + width as usize]);
            }

            let seed = (index as u32) * 3 + channel;
            let noisy = noisy_field_over(&clean, width, height, 0.03, seed);
            for pixel in 0..(width * height) as usize {
                frame[pixel * channels as usize + channel as usize] = noisy[pixel];
            }
        }

        frames.push(frame);
    }

    frames
}

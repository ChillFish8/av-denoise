use cubecl::cpu::CpuRuntime;
use cubecl::prelude::*;

use super::*;

type R = CpuRuntime;

fn make_client() -> ComputeClient<R> {
    let device = <R as Runtime>::Device::default();
    R::client(&device)
}

fn make_uniform_frame(w: u32, h: u32, ch: u32, val: f32) -> Vec<f32> {
    vec![val; (w * h * ch) as usize]
}

/// Creates a frame with a patch of noise (not just a single pixel)
/// so that NLMeans has matching noisy patches to work with.
fn make_frame_with_noisy_region(
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

// ==================== Single-frame (spatial) tests ====================

#[test]
fn uniform_image_passthrough() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
    };

    let w = 16;
    let h = 16;
    let frame = make_uniform_frame(w, h, 1, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();

    for (i, &v) in result.iter().enumerate() {
        assert!((v - 0.5).abs() < 1e-5, "pixel {i}: expected 0.5, got {v}");
    }
}

#[test]
fn uniform_yuv_passthrough() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Yuv,
    };

    let w = 16;
    let h = 16;
    let frame = make_uniform_frame(w, h, 3, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();
    assert_eq!(result.len(), (w * h * 3) as usize);

    for (i, &v) in result.iter().enumerate() {
        assert!((v - 0.5).abs() < 1e-5, "pixel {i}: expected 0.5, got {v}");
    }
}

#[test]
fn uniform_chroma_passthrough() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Chroma,
    };

    let w = 16;
    let h = 16;
    let frame = make_uniform_frame(w, h, 2, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();
    assert_eq!(result.len(), (w * h * 2) as usize);

    for (i, &v) in result.iter().enumerate() {
        assert!((v - 0.5).abs() < 1e-5, "pixel {i}: expected ~0.5, got {v}");
    }
}

#[test]
fn noisy_region_suppressed() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 3,
        patch_radius: 1,
        strength: 50.0,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
    };

    let w = 32;
    let h = 32;
    let mut frame = vec![0.5f32; (w * h) as usize];
    frame[(16 * w + 16) as usize] = 0.8;

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();

    let noisy_idx = (16 * w + 16) as usize;
    let denoised = result[noisy_idx];

    assert!(
        denoised < 0.8,
        "noisy pixel should be somewhat suppressed, got {denoised}"
    );
}

#[test]
fn high_strength_smooths_heavily() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 1,
        strength: 10000.0,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
    };

    let w = 16;
    let h = 16;

    let mut frame = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        let val = if y % 2 == 0 { 0.3 } else { 0.7 };
        for x in 0..w {
            frame[(y * w + x) as usize] = val;
        }
    }

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();

    let center = result[(8 * w + 8) as usize];
    assert!(
        (center - 0.5).abs() < 0.15,
        "high strength should smooth toward ~0.5, got {center}"
    );
}

#[test]
fn low_strength_preserves_original() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 2,
        strength: 0.001,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
    };

    let w = 16;
    let h = 16;

    let mut frame = vec![0.5f32; (w * h) as usize];
    frame[(8 * w + 8) as usize] = 0.8;

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();

    let pixel = result[(8 * w + 8) as usize];
    assert!(
        (pixel - 0.8).abs() < 0.05,
        "low strength should preserve original ~0.8, got {pixel}"
    );
}

#[test]
fn self_weight_zero_uniform() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 0.0,
        channels: ChannelMode::Luma,
    };

    let w = 16;
    let h = 16;

    let frame = make_uniform_frame(w, h, 1, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();

    for (i, &v) in result.iter().enumerate() {
        assert!((v - 0.5).abs() < 1e-5, "pixel {i}: expected ~0.5, got {v}");
    }
}

#[test]
fn spatial_only_no_delay() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 0,
        ..NlmParams::default()
    };

    let w = 8;
    let h = 8;
    let frame = make_uniform_frame(w, h, 3, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap();
    assert!(result.is_some(), "d=0 should not delay output");
}

// ==================== Temporal tests ====================

#[test]
fn temporal_requires_full_window() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 1,
        channels: ChannelMode::Luma,
        ..NlmParams::default()
    };

    let w = 8;
    let h = 8;
    let frame = make_uniform_frame(w, h, 1, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);

    denoiser.push_frame(&frame);
    assert!(
        denoiser.denoise().unwrap().is_none(),
        "should not output with only 1 of 3 frames"
    );

    denoiser.push_frame(&frame);
    assert!(
        denoiser.denoise().unwrap().is_none(),
        "should not output with only 2 of 3 frames"
    );

    denoiser.push_frame(&frame);
    let result = denoiser.denoise().unwrap();
    assert!(result.is_some(), "should output with 3 of 3 frames");
}

#[test]
fn temporal_denoise_uniform() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 1,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
    };

    let w = 8;
    let h = 8;

    let frame = make_uniform_frame(w, h, 1, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);
    denoiser.push_frame(&frame);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();

    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - 0.5).abs() < 1e-4,
            "temporal uniform: pixel {i} expected ~0.5, got {v}"
        );
    }
}

#[test]
fn temporal_with_noisy_center_frame() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 1,
        search_radius: 2,
        patch_radius: 1,
        strength: 10.0,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
    };

    let w = 16;
    let h = 16;

    let clean = make_uniform_frame(w, h, 1, 0.5);
    let noisy = make_frame_with_noisy_region(w, h, 1, 0.5, 8, 8, 1, 0.8);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&clean);
    denoiser.push_frame(&noisy);
    denoiser.push_frame(&clean);

    let result = denoiser.denoise().unwrap().unwrap();

    let center_val = result[(8 * w + 8) as usize];
    assert!(
        center_val < 0.8,
        "temporal denoising should suppress noise, got {center_val}"
    );
}

#[test]
fn temporal_asymmetric_frames_correct_weights() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 1,
        search_radius: 1,
        patch_radius: 1,
        strength: 5.0,
        self_weight: 0.0,
        channels: ChannelMode::Luma,
    };

    let w = 16;
    let h = 16;

    let mut frame0 = vec![0.5f32; (w * h) as usize];
    for y in 6..10 {
        for x in 6..10 {
            frame0[(y * w + x) as usize] = 0.9;
        }
    }

    let frame1 = vec![0.5f32; (w * h) as usize];
    let frame2 = vec![0.5f32; (w * h) as usize];

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame0);
    denoiser.push_frame(&frame1);
    denoiser.push_frame(&frame2);

    let result = denoiser.denoise().unwrap().unwrap();

    let center_val = result[(8 * w + 8) as usize];
    assert!(
        (center_val - 0.5).abs() < 0.1,
        "temporal asymmetric: center should stay near 0.5 \
         (past frame de-weighted), got {center_val}"
    );
}

#[test]
fn flush_produces_remaining_frames() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 1,
        channels: ChannelMode::Luma,
        ..NlmParams::default()
    };

    let w = 8;
    let h = 8;

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);

    for _ in 0..4 {
        let frame = make_uniform_frame(w, h, 1, 0.5);
        denoiser.push_frame(&frame);
        let _ = denoiser.denoise().unwrap();
    }

    let remaining = denoiser.flush().unwrap();
    assert_eq!(
        remaining.len(),
        1,
        "flush should produce 1 remaining frame for d=1"
    );

    for frame in &remaining {
        assert_eq!(frame.len(), (w * h) as usize);
    }
}

// ==================== Symmetry and edge tests ====================

#[test]
fn symmetry_preserved() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
    };

    let w = 16;
    let h = 16;

    let mut frame = vec![0.5f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..(w / 2) {
            let val = 0.3 + 0.4 * (x as f32 / w as f32);
            frame[(y * w + x) as usize] = val;
            frame[(y * w + (w - 1 - x)) as usize] = val;
        }
    }

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();

    for y in 0..h {
        for x in 0..(w / 2) {
            let left = result[(y * w + x) as usize];
            let right = result[(y * w + (w - 1 - x)) as usize];
            assert!(
                (left - right).abs() < 1e-5,
                "symmetry broken at ({x},{y}): \
                 left={left}, right={right}"
            );
        }
    }
}

#[test]
fn clamp_to_edge_no_darkening() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 2,
        strength: 100.0,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
    };

    let w = 8;
    let h = 8;
    let frame = make_uniform_frame(w, h, 1, 0.7);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();

    let corner = result[0];
    assert!(
        (corner - 0.7).abs() < 0.05,
        "corner pixel should not darken with clamp-to-edge, \
         got {corner}"
    );

    let edge = result[4];
    assert!(
        (edge - 0.7).abs() < 0.05,
        "edge pixel should not darken with clamp-to-edge, \
         got {edge}"
    );
}

// ==================== Normalization tests ====================

#[test]
fn normalization_u8_roundtrip() {
    let original: Vec<u8> = (0..=255).collect();
    let normalized = normalize_u8(&original);
    let restored = denormalize_u8(&normalized);

    assert_eq!(original, restored);
}

#[test]
fn normalization_u16_roundtrip() {
    let original: Vec<u16> = (0..1024).chain(64000..=65535).collect();
    let normalized = normalize_u16(&original);
    let restored = denormalize_u16(&normalized);

    assert_eq!(original, restored);
}

// ==================== Separable filter tests ====================
// These use patch_radius > 2 to trigger the separable path.

#[test]
fn separable_uniform_passthrough() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 4, // > SEPARABLE_THRESHOLD
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
    };

    let w = 16;
    let h = 16;
    let frame = make_uniform_frame(w, h, 1, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    assert!(
        denoiser.use_separable,
        "should use separable for patch_radius=4"
    );
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();

    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - 0.5).abs() < 1e-4,
            "separable: pixel {i}: expected 0.5, got {v}"
        );
    }
}

#[test]
fn separable_yuv_passthrough() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 3, // > SEPARABLE_THRESHOLD
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Yuv,
    };

    let w = 16;
    let h = 16;
    let frame = make_uniform_frame(w, h, 3, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    assert!(denoiser.use_separable);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();
    assert_eq!(result.len(), (w * h * 3) as usize);

    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - 0.5).abs() < 1e-4,
            "separable yuv: pixel {i}: expected 0.5, got {v}"
        );
    }
}

#[test]
fn separable_symmetry_preserved() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 4,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
    };

    let w = 16;
    let h = 16;

    let mut frame = vec![0.5f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..(w / 2) {
            let val = 0.3 + 0.4 * (x as f32 / w as f32);
            frame[(y * w + x) as usize] = val;
            frame[(y * w + (w - 1 - x)) as usize] = val;
        }
    }

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();

    for y in 0..h {
        for x in 0..(w / 2) {
            let left = result[(y * w + x) as usize];
            let right = result[(y * w + (w - 1 - x)) as usize];
            assert!(
                (left - right).abs() < 1e-4,
                "separable symmetry broken at ({x},{y}): \
                 left={left}, right={right}"
            );
        }
    }
}

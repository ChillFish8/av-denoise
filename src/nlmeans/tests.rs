use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use super::*;

type R = WgpuRuntime;

fn make_client() -> ComputeClient<R> {
    let device = <R as Runtime>::Device::default();
    R::client(&device)
}

fn make_uniform_frame(w: u32, h: u32, ch: u32, val: f32) -> Vec<f32> {
    vec![val; (w * h * ch) as usize]
}

/// Creates a frame with a patch of noise (not just a single pixel)
/// so that NLMeans has matching noisy patches to work with.
#[allow(clippy::too_many_arguments)]
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
        prefilter: PrefilterMode::None,
    };

    let w = 16;
    let h = 16;
    let frame = make_uniform_frame(w, h, 1, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap().to_vec();

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
        prefilter: PrefilterMode::None,
    };

    let w = 16;
    let h = 16;
    let frame = make_uniform_frame(w, h, 3, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap().to_vec();
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
        prefilter: PrefilterMode::None,
    };

    let w = 16;
    let h = 16;
    let frame = make_uniform_frame(w, h, 2, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap().to_vec();
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
        prefilter: PrefilterMode::None,
    };

    let w = 32;
    let h = 32;
    let mut frame = vec![0.5f32; (w * h) as usize];
    frame[(16 * w + 16) as usize] = 0.8;

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap().to_vec();

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
        prefilter: PrefilterMode::None,
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

    let result = denoiser.denoise().unwrap().unwrap().to_vec();

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
        prefilter: PrefilterMode::None,
    };

    let w = 16;
    let h = 16;

    let mut frame = vec![0.5f32; (w * h) as usize];
    frame[(8 * w + 8) as usize] = 0.8;

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap().to_vec();

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
        prefilter: PrefilterMode::None,
    };

    let w = 16;
    let h = 16;

    let frame = make_uniform_frame(w, h, 1, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap().to_vec();

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

#[test]
fn temporal_requires_full_window() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 1,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
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
        prefilter: PrefilterMode::None,
    };

    let w = 8;
    let h = 8;

    let frame = make_uniform_frame(w, h, 1, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);
    denoiser.push_frame(&frame);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap().to_vec();

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
        prefilter: PrefilterMode::None,
    };

    let w = 16;
    let h = 16;

    let clean = make_uniform_frame(w, h, 1, 0.5);
    let noisy = make_frame_with_noisy_region(w, h, 1, 0.5, 8, 8, 1, 0.8);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&clean);
    denoiser.push_frame(&noisy);
    denoiser.push_frame(&clean);

    let result = denoiser.denoise().unwrap().unwrap().to_vec();

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
        prefilter: PrefilterMode::None,
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

    let result = denoiser.denoise().unwrap().unwrap().to_vec();

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
        prefilter: PrefilterMode::None,
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

    let mut remaining: Vec<Vec<f32>> = Vec::new();
    denoiser.flush(|frame| remaining.push(frame.to_vec())).unwrap();
    assert_eq!(
        remaining.len(),
        1,
        "flush should produce 1 remaining frame for d=1"
    );

    for frame in &remaining {
        assert_eq!(frame.len(), (w * h) as usize);
    }
}

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
        prefilter: PrefilterMode::None,
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

    let result = denoiser.denoise().unwrap().unwrap().to_vec();

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
        prefilter: PrefilterMode::None,
    };

    let w = 8;
    let h = 8;
    let frame = make_uniform_frame(w, h, 1, 0.7);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap().to_vec();

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

/// The following tests use `patch_radius > SEPARABLE_THRESHOLD` to trigger
/// the separable path.
#[test]
fn separable_uniform_passthrough() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 9, // > SEPARABLE_THRESHOLD
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
    };

    let w = 32;
    let h = 32;
    let frame = make_uniform_frame(w, h, 1, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    assert!(denoiser.use_separable, "should use separable for patch_radius=9");
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap().to_vec();

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
        patch_radius: 9, // > SEPARABLE_THRESHOLD
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Yuv,
        prefilter: PrefilterMode::None,
    };

    let w = 32;
    let h = 32;
    let frame = make_uniform_frame(w, h, 3, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    assert!(denoiser.use_separable);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap().to_vec();
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
        prefilter: PrefilterMode::None,
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

    let result = denoiser.denoise().unwrap().unwrap().to_vec();

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

/// Aliasing the reference to the input must reproduce the no-prefilter
/// baseline exactly. Sanity check on the `_ref` kernel variants.
#[test]
fn external_reference_equals_input_matches_baseline() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frame = make_frame_with_noisy_region(w, h, 1, 0.3, 8, 8, 2, 0.7);

    let baseline = {
        let params = NlmParams {
            temporal_radius: 0,
            search_radius: 2,
            patch_radius: 2,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::None,
        };
        let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
        d.push_frame(&frame);
        d.denoise().unwrap().unwrap().to_vec()
    };

    let with_ref = {
        let params = NlmParams {
            temporal_radius: 0,
            search_radius: 2,
            patch_radius: 2,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::External,
        };
        let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
        d.push_frame_with_reference(&frame, &frame);
        d.denoise().unwrap().unwrap().to_vec()
    };

    assert_eq!(baseline.len(), with_ref.len());
    for (i, (a, b)) in baseline.iter().zip(with_ref.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "pixel {i}: baseline={a}, with_ref={b}");
    }
}

/// Separable path (patch_radius > 2) variant of the identity check.
#[test]
fn external_reference_separable_matches_baseline() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frame = make_frame_with_noisy_region(w, h, 1, 0.3, 8, 8, 2, 0.7);

    let baseline = {
        let params = NlmParams {
            temporal_radius: 0,
            search_radius: 2,
            patch_radius: 4,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::None,
        };
        let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
        d.push_frame(&frame);
        d.denoise().unwrap().unwrap().to_vec()
    };

    let with_ref = {
        let params = NlmParams {
            temporal_radius: 0,
            search_radius: 2,
            patch_radius: 4,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::External,
        };
        let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
        d.push_frame_with_reference(&frame, &frame);
        d.denoise().unwrap().unwrap().to_vec()
    };

    for (i, (a, b)) in baseline.iter().zip(with_ref.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "pixel {i}: baseline={a}, with_ref={b}");
    }
}

/// Temporal rclip with `reference == input` for every pushed frame
/// must also reproduce baseline.
#[test]
fn external_reference_temporal_matches_baseline() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frames = [
        make_frame_with_noisy_region(w, h, 1, 0.3, 8, 8, 2, 0.7),
        make_frame_with_noisy_region(w, h, 1, 0.3, 7, 8, 2, 0.65),
        make_frame_with_noisy_region(w, h, 1, 0.3, 9, 8, 2, 0.75),
    ];

    let baseline = {
        let params = NlmParams {
            temporal_radius: 1,
            search_radius: 2,
            patch_radius: 2,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::None,
        };
        let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
        for f in &frames {
            d.push_frame(f);
        }
        d.denoise().unwrap().unwrap().to_vec()
    };

    let with_ref = {
        let params = NlmParams {
            temporal_radius: 1,
            search_radius: 2,
            patch_radius: 2,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::External,
        };
        let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
        for f in &frames {
            d.push_frame_with_reference(f, f);
        }
        d.denoise().unwrap().unwrap().to_vec()
    };

    for (i, (a, b)) in baseline.iter().zip(with_ref.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "pixel {i}: baseline={a}, with_ref={b}");
    }
}

/// Bilateral prefilter on a uniform image must reproduce the uniform
/// value exactly (weights sum to anything, but the weighted average of
/// identical values is itself).
#[test]
fn bilateral_uniform_image_passthrough() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frame = make_uniform_frame(w, h, 1, 0.5);

    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 1.0,
            sigma_r: 0.1,
        },
    };

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&frame);
    let result = d.denoise().unwrap().unwrap().to_vec();

    for (i, &v) in result.iter().enumerate() {
        assert!((v - 0.5).abs() < 1e-4, "pixel {i}: expected 0.5, got {v}");
    }
}

/// Bilateral smoke test on noisy input. Verifies the kernel produces
/// finite, in-range outputs (we trust the kernel correctness from the
/// uniform-image and identity tests).
#[test]
fn bilateral_noisy_image_finite() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frame = make_frame_with_noisy_region(w, h, 1, 0.4, 8, 8, 3, 0.8);

    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 2.0,
            sigma_r: 0.05,
        },
    };

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&frame);
    let result = d.denoise().unwrap().unwrap().to_vec();

    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "pixel {i}: non-finite output {v}");
        assert!((-0.01..=1.01).contains(&v), "pixel {i}: out-of-range output {v}");
    }
}

#[test]
fn validate_rejects_oversized_patch_radius() {
    let params = NlmParams {
        patch_radius: MAX_PATCH_RADIUS + 1,
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn validate_rejects_oversized_search_radius() {
    let params = NlmParams {
        search_radius: MAX_SEARCH_RADIUS + 1,
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn validate_rejects_oversized_temporal_radius() {
    let params = NlmParams {
        temporal_radius: MAX_TEMPORAL_RADIUS + 1,
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn validate_rejects_non_positive_strength() {
    let zero = NlmParams {
        strength: 0.0,
        ..NlmParams::default()
    };
    assert!(zero.validate().is_err());

    let nan = NlmParams {
        strength: f32::NAN,
        ..NlmParams::default()
    };
    assert!(nan.validate().is_err());

    let inf = NlmParams {
        strength: f32::INFINITY,
        ..NlmParams::default()
    };
    assert!(inf.validate().is_err());
}

#[test]
fn validate_rejects_negative_self_weight() {
    let params = NlmParams {
        self_weight: -0.1,
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn validate_accepts_defaults() {
    assert!(NlmParams::default().validate().is_ok());
}

/// Drives most weights toward zero by combining a near-maximum search
/// radius with extremely low strength (large `h2_inv_norm`), and on
/// noisy content so denominators land near the underflow guard in
/// `nlm_finish`. The output must contain no `inf`/`nan` regardless.
#[test]
fn extreme_params_produce_finite_output() {
    let client = make_client();
    let w = 32;
    let h = 32;
    let frame = make_frame_with_noisy_region(w, h, 1, 0.1, 16, 16, 5, 0.9);

    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 2,
        strength: 0.1,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
    };

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&frame);
    let result = d.denoise().unwrap().unwrap().to_vec();

    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "pixel {i}: non-finite output {v}");
        assert!((-0.01..=1.01).contains(&v), "pixel {i}: out-of-range output {v}");
    }
}

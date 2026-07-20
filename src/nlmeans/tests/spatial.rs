use super::helpers::*;
use crate::nlmeans::*;

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
        motion_compensation: MotionCompensationMode::None,
        hq: None,
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
        motion_compensation: MotionCompensationMode::None,
        hq: None,
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
        motion_compensation: MotionCompensationMode::None,
        hq: None,
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
        motion_compensation: MotionCompensationMode::None,
        hq: None,
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
        motion_compensation: MotionCompensationMode::None,
        hq: None,
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
        motion_compensation: MotionCompensationMode::None,
        hq: None,
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
        motion_compensation: MotionCompensationMode::None,
        hq: None,
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
        motion_compensation: MotionCompensationMode::None,
        hq: None,
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
        motion_compensation: MotionCompensationMode::None,
        hq: None,
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

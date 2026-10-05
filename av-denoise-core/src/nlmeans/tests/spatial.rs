use super::helpers::*;
use crate::bench_api::HostIo;
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

    let width = 16;
    let height = 16;
    let frame = make_uniform_frame(width, height, 1, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();

    for (i, &value) in result.iter().enumerate() {
        assert!((value - 0.5).abs() < 1e-5, "pixel {i}: expected 0.5, got {value}");
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

    let width = 16;
    let height = 16;
    let frame = make_uniform_frame(width, height, 3, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();
    assert_eq!(result.len(), (width * height * 3) as usize);

    for (i, &value) in result.iter().enumerate() {
        assert!((value - 0.5).abs() < 1e-5, "pixel {i}: expected 0.5, got {value}");
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

    let width = 16;
    let height = 16;
    let frame = make_uniform_frame(width, height, 2, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();
    assert_eq!(result.len(), (width * height * 2) as usize);

    for (i, &value) in result.iter().enumerate() {
        assert!(
            (value - 0.5).abs() < 1e-5,
            "pixel {i}: expected ~0.5, got {value}"
        );
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

    let width = 32;
    let height = 32;
    let mut frame = vec![0.5f32; (width * height) as usize];
    frame[(16 * width + 16) as usize] = 0.8;

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();

    let noisy_index = (16 * width + 16) as usize;
    let denoised = result[noisy_index];

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

    let width = 16;
    let height = 16;

    let mut frame = vec![0.0f32; (width * height) as usize];
    for y in 0..height {
        let row_value = if y % 2 == 0 { 0.3 } else { 0.7 };
        for x in 0..width {
            frame[(y * width + x) as usize] = row_value;
        }
    }

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();

    let center = result[(8 * width + 8) as usize];
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

    let width = 16;
    let height = 16;

    let mut frame = vec![0.5f32; (width * height) as usize];
    frame[(8 * width + 8) as usize] = 0.8;

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();

    let pixel = result[(8 * width + 8) as usize];
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

    let width = 16;
    let height = 16;

    let frame = make_uniform_frame(width, height, 1, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();

    for (i, &value) in result.iter().enumerate() {
        assert!(
            (value - 0.5).abs() < 1e-5,
            "pixel {i}: expected ~0.5, got {value}"
        );
    }
}

#[test]
fn spatial_only_no_delay() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 0,
        ..NlmParams::default()
    };

    let width = 8;
    let height = 8;
    let frame = make_uniform_frame(width, height, 3, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
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

    let width = 16;
    let height = 16;

    let mut frame = vec![0.5f32; (width * height) as usize];
    for y in 0..height {
        for x in 0..(width / 2) {
            let value = 0.3 + 0.4 * (x as f32 / width as f32);
            frame[(y * width + x) as usize] = value;
            frame[(y * width + (width - 1 - x)) as usize] = value;
        }
    }

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();

    for y in 0..height {
        for x in 0..(width / 2) {
            let left = result[(y * width + x) as usize];
            let right = result[(y * width + (width - 1 - x)) as usize];
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

    let width = 8;
    let height = 8;
    let frame = make_uniform_frame(width, height, 1, 0.7);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
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

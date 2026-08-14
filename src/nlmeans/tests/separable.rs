use super::helpers::*;
use crate::nlmeans::*;

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
        motion_compensation: MotionCompensationMode::None,
        track_weight_sq: false,
        hq: None,
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
        motion_compensation: MotionCompensationMode::None,
        track_weight_sq: false,
        hq: None,
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
        motion_compensation: MotionCompensationMode::None,
        track_weight_sq: false,
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
                (left - right).abs() < 1e-4,
                "separable symmetry broken at ({x},{y}): \
                 left={left}, right={right}"
            );
        }
    }
}

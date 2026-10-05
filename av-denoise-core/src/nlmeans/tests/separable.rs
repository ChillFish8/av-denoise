use super::helpers::*;
use crate::bench_api::HostIo;
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
        hq: None,
    };

    let width = 32;
    let height = 32;
    let frame = make_uniform_frame(width, height, 1, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    assert!(denoiser.use_separable, "should use separable for patch_radius=9");
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();

    for (i, &value) in result.iter().enumerate() {
        assert!(
            (value - 0.5).abs() < 1e-4,
            "separable: pixel {i}: expected 0.5, got {value}"
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
        hq: None,
    };

    let width = 32;
    let height = 32;
    let frame = make_uniform_frame(width, height, 3, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    assert!(denoiser.use_separable);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();
    assert_eq!(result.len(), (width * height * 3) as usize);

    for (i, &value) in result.iter().enumerate() {
        assert!(
            (value - 0.5).abs() < 1e-4,
            "separable yuv: pixel {i}: expected 0.5, got {value}"
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
                (left - right).abs() < 1e-4,
                "separable symmetry broken at ({x},{y}): \
                 left={left}, right={right}"
            );
        }
    }
}

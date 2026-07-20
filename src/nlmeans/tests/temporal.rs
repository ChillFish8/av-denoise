use super::helpers::*;
use crate::nlmeans::*;

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

    // Leading-edge mirror fills R past slots from the very first push, so the
    // window only needs R+1 real pushes (= 2 for radius 1) before the first
    // submit produces output.
    denoiser.push_frame(&frame);
    assert!(
        denoiser.denoise().unwrap().is_none(),
        "should not output with only 1 real push (leading-mirror fills R, total still R+1 < 2R+1)"
    );

    denoiser.push_frame(&frame);
    let result = denoiser.denoise().unwrap();
    assert!(
        result.is_some(),
        "should output once R+1 real frames have been pushed (window now full via leading mirror)"
    );
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
        motion_compensation: MotionCompensationMode::None,
        hq: None,
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
        motion_compensation: MotionCompensationMode::None,
        hq: None,
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
        motion_compensation: MotionCompensationMode::None,
        hq: None,
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

/// `N` pushes at temporal radius `R` must produce exactly `N` total emissions
/// (during pushes + flush). Regression guard against the old bug where the
/// leading `R` logical frames were silently dropped (every scene lost its
/// first frame under `--temporal-radius >= 1`).
#[test]
fn temporal_push_flush_frame_count_matches() {
    let client = make_client();
    let w = 8;
    let h = 8;

    for radius in 1..=2 {
        let params = NlmParams {
            temporal_radius: radius,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::None,
            ..NlmParams::default()
        };
        let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);

        const PUSHES: usize = 10;
        let mut during_pushes = 0usize;
        for i in 0..PUSHES {
            // Distinct frames so the kernel can't accidentally satisfy a
            // count check by mis-pairing duplicate buffers.
            let value = 0.1 + (i as f32) * 0.05;
            let frame = make_uniform_frame(w, h, 1, value);
            denoiser.push_frame(&frame);
            if denoiser.denoise().unwrap().is_some() {
                during_pushes += 1;
            }
        }

        let mut flushed = 0usize;
        denoiser.flush(|_| flushed += 1).unwrap();

        assert_eq!(
            during_pushes + flushed,
            PUSHES,
            "radius {radius}: pushed {PUSHES} frames, got {during_pushes} during pushes + {flushed} from flush",
        );
    }
}

use cubecl::prelude::ComputeClient;

use super::helpers::*;
use crate::bench_api::HostIo;
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

    let width = 8;
    let height = 8;
    let frame = make_uniform_frame(width, height, 1, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);

    // The leading-edge mirror fills the R past slots, so the window needs only R+1 real pushes.
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

    let width = 8;
    let height = 8;

    let frame = make_uniform_frame(width, height, 1, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);
    denoiser.push_frame(&frame);
    denoiser.push_frame(&frame);

    let result = denoiser.denoise().unwrap().unwrap();

    for (i, &value) in result.iter().enumerate() {
        assert!(
            (value - 0.5).abs() < 1e-4,
            "temporal uniform: pixel {i} expected ~0.5, got {value}"
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

    let width = 16;
    let height = 16;

    let clean = make_uniform_frame(width, height, 1, 0.5);
    let noisy = make_frame_with_noisy_region(width, height, 1, 0.5, 8, 8, 1, 0.8);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&clean);
    denoiser.push_frame(&noisy);
    denoiser.push_frame(&clean);

    let result = denoiser.denoise().unwrap().unwrap();

    let center_value = result[(8 * width + 8) as usize];
    assert!(
        center_value < 0.8,
        "temporal denoising should suppress noise, got {center_value}"
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

    let width = 16;
    let height = 16;

    let mut frame0 = vec![0.5f32; (width * height) as usize];
    for y in 6..10 {
        for x in 6..10 {
            frame0[(y * width + x) as usize] = 0.9;
        }
    }

    let frame1 = vec![0.5f32; (width * height) as usize];
    let frame2 = vec![0.5f32; (width * height) as usize];

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame0);
    denoiser.push_frame(&frame1);
    denoiser.push_frame(&frame2);

    let result = denoiser.denoise().unwrap().unwrap();

    let center_value = result[(8 * width + 8) as usize];
    assert!(
        (center_value - 0.5).abs() < 0.1,
        "temporal asymmetric: center should stay near 0.5 \
         (past frame de-weighted), got {center_value}"
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

    let width = 8;
    let height = 8;

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);

    for _ in 0..4 {
        let frame = make_uniform_frame(width, height, 1, 0.5);
        denoiser.push_frame(&frame);
        let _ = denoiser.denoise().unwrap();
    }

    let mut remaining: Vec<Vec<f32>> = Vec::new();
    let collect = |frame: &[f32]| {
        let samples = frame.to_vec();
        remaining.push(samples);
    };
    denoiser.flush(collect).unwrap();
    assert_eq!(
        remaining.len(),
        1,
        "flush should produce 1 remaining frame for d=1"
    );

    for frame in &remaining {
        assert_eq!(frame.len(), (width * height) as usize);
    }
}

/// Pins the bug where the leading `R` frames of every scene were dropped.
#[test]
fn temporal_push_flush_frame_count_matches() {
    let client = make_client();
    let width = 8;
    let height = 8;

    for radius in 1..=2 {
        let params = NlmParams {
            temporal_radius: radius,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::None,
            ..NlmParams::default()
        };
        let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);

        const PUSHES: usize = 10;
        let mut during_pushes = 0usize;
        for i in 0..PUSHES {
            // Distinct frames stop mis-paired duplicate buffers from satisfying the count.
            let value = 0.1 + (i as f32) * 0.05;
            let frame = make_uniform_frame(width, height, 1, value);
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

fn psnr(reference: &[f32], test: &[f32]) -> f64 {
    let mse: f64 = reference
        .iter()
        .zip(test.iter())
        .map(|(&reference_value, &test_value)| {
            let difference = (reference_value as f64) - (test_value as f64);
            difference * difference
        })
        .sum::<f64>()
        / reference.len() as f64;

    if mse <= 1e-20 {
        return 999.0;
    }

    10.0 * (1.0f64 / mse).log10()
}

/// A gradient (a smooth region to average) with a block of another value (an edge to preserve).
fn structured_base(width: u32, height: u32) -> Vec<f32> {
    let mut base = make_gradient_frame(width, height, 0.2, 0.8);
    let block_left = width / 3;
    let block_top = height / 3;
    for y in block_top..block_top * 2 {
        for x in block_left..block_left * 2 {
            base[(y * width + x) as usize] = 0.15;
        }
    }

    base
}

/// Denoises `frames` through the windowed and separable dispatches, returning each PSNR against `base`.
///
/// The separable path shares no code with the windowed pair kernel, so it acts as an independent
/// reference.
fn windowed_vs_separable_psnr(
    client: &ComputeClient<R>,
    params: &NlmParams,
    width: u32,
    height: u32,
    base: &[f32],
    frames: &[Vec<f32>],
) -> (f64, f64) {
    let windowed_params = params.clone();
    let mut windowed = NlmDenoiser::<R>::new(client, windowed_params, width, height);
    for frame in frames {
        windowed.push_frame(frame);
    }

    let windowed_result = windowed.denoise().unwrap().unwrap();

    let separable_params = params.clone();
    let mut separable = NlmDenoiser::<R>::new(client, separable_params, width, height);
    separable.use_separable = true;
    for frame in frames {
        separable.push_frame(frame);
    }

    let separable_result = separable.denoise().unwrap().unwrap();

    let windowed_psnr = psnr(base, &windowed_result);
    let separable_psnr = psnr(base, &separable_result);

    (windowed_psnr, separable_psnr)
}

/// The windowed kernel's backward temporal weight must use the same centre patch as the value it
/// multiplies.
///
/// A weight measured against a shifted patch grows wrong with the search offset, which these search
/// radii are large enough to expose.
#[test]
fn temporal_windowed_matches_separable_at_search_5_and_6() {
    let client = make_client();
    let width = 128;
    let height = 128;
    let base = structured_base(width, height);

    for search_radius in [5u32, 6] {
        let params = NlmParams {
            temporal_radius: 4,
            search_radius,
            patch_radius: 4,
            strength: 0.35,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::None,
            motion_compensation: MotionCompensationMode::None,
            hq: Some(HqParams::with_sigma(16.0 / 255.0)),
        };

        let sigma = 16.0 / 255.0;
        let frames: Vec<Vec<f32>> = (0..9)
            .map(|seed| noisy_field_over(&base, width, height, sigma, seed))
            .collect();

        let (windowed_psnr, separable_psnr) =
            windowed_vs_separable_psnr(&client, &params, width, height, &base, &frames);

        assert!(
            (windowed_psnr - separable_psnr).abs() < 1.5,
            "search_radius={search_radius}: windowed ({windowed_psnr:.2} dB) should track \
             separable ({separable_psnr:.2} dB) within measurement noise"
        );
    }
}

/// A prefilter is active so both dispatches read patch distances from the prefiltered reference.
#[test]
fn temporal_windowed_ref_matches_separable_ref_at_search_5_and_6() {
    let client = make_client();
    let width = 128;
    let height = 128;
    let base = structured_base(width, height);

    for search_radius in [5u32, 6] {
        let params = NlmParams {
            temporal_radius: 4,
            search_radius,
            patch_radius: 4,
            strength: 0.35,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::Bilateral {
                sigma_s: 1.0,
                sigma_r: 0.1,
            },
            motion_compensation: MotionCompensationMode::None,
            hq: Some(HqParams::with_sigma(16.0 / 255.0)),
        };

        let sigma = 16.0 / 255.0;
        let frames: Vec<Vec<f32>> = (0..9)
            .map(|seed| noisy_field_over(&base, width, height, sigma, seed))
            .collect();

        let (windowed_psnr, separable_psnr) =
            windowed_vs_separable_psnr(&client, &params, width, height, &base, &frames);

        assert!(
            (windowed_psnr - separable_psnr).abs() < 1.5,
            "search_radius={search_radius}: windowed ({windowed_psnr:.2} dB) should track \
             separable ({separable_psnr:.2} dB) within measurement noise"
        );
    }
}

/// A debug build overflows its codegen stack on the fully unrolled window loop at this radius, even
/// at the stack size `.cargo/config.toml` sets.
#[test]
#[ignore = "debug build codegen overflows the stack at search_radius=8, run with --release"]
fn temporal_windowed_matches_separable_at_the_search_ceiling() {
    let client = make_client();
    let width = 128;
    let height = 128;
    let base = structured_base(width, height);

    let params = NlmParams {
        temporal_radius: 4,
        search_radius: MAX_SEARCH_RADIUS,
        patch_radius: 4,
        strength: 0.35,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        hq: Some(HqParams::with_sigma(16.0 / 255.0)),
    };

    let sigma = 16.0 / 255.0;
    let frames: Vec<Vec<f32>> = (0..9)
        .map(|seed| noisy_field_over(&base, width, height, sigma, seed))
        .collect();

    let (windowed_psnr, separable_psnr) =
        windowed_vs_separable_psnr(&client, &params, width, height, &base, &frames);

    assert!(
        (windowed_psnr - separable_psnr).abs() < 1.5,
        "search_radius={MAX_SEARCH_RADIUS}: windowed ({windowed_psnr:.2} dB) should track \
         separable ({separable_psnr:.2} dB) within measurement noise"
    );
}

/// Uniform input zeroes every patch distance, so this cannot catch a mis-centred weight.
///
/// It does catch a kernel reading or writing outside its region, which breaks the uniformity.
#[test]
fn temporal_uniform_passthrough_search_5_and_6() {
    let client = make_client();
    let width = 64;
    let height = 64;
    let frame = make_uniform_frame(width, height, 1, 0.5);

    for search_radius in [5u32, 6] {
        let params = NlmParams {
            temporal_radius: 2,
            search_radius,
            patch_radius: 4,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::None,
            motion_compensation: MotionCompensationMode::None,
            hq: None,
        };

        let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
        for _ in 0..5 {
            denoiser.push_frame(&frame);
        }

        let result = denoiser.denoise().unwrap().unwrap();

        for (i, &value) in result.iter().enumerate() {
            assert!(
                (value - 0.5).abs() < 1e-3,
                "search_radius={search_radius}: pixel {i} expected ~0.5, got {value}"
            );
        }
    }
}

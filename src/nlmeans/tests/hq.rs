use super::helpers::*;
use crate::nlmeans::*;

/// Shared baseline for the fast path. Each test overrides just the
/// field it's exercising via struct-update syntax.
fn base_params() -> NlmParams {
    NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        hq: None,
    }
}

/// With both HQ features off, `effective_strength` and `noise_offset`
/// degenerate to exactly what the fast path already computes, so the
/// two denoisers must agree bit-for-bit.
#[test]
fn hq_disabled_features_match_fast_mode() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frame = make_frame_with_noisy_region(w, h, 1, 0.5, 8, 8, 3, 0.9);

    let mut fast = NlmDenoiser::<R>::new(&client, base_params(), w, h);
    fast.push_frame(&frame);
    let fast_out = fast.denoise().unwrap().unwrap().to_vec();

    let hq_params = NlmParams {
        hq: Some(HqParams {
            auto_strength: false,
            noise_floor: false,
            sigma: 8.0 / 255.0,
        }),
        ..base_params()
    };
    let mut hq = NlmDenoiser::<R>::new(&client, hq_params, w, h);
    hq.push_frame(&frame);
    let hq_out = hq.denoise().unwrap().unwrap().to_vec();

    assert_eq!(
        fast_out, hq_out,
        "disabled HQ features should reproduce the fast path exactly"
    );
}

/// Turning on the noise floor shifts every patch distance by a
/// nonzero offset. Neighbours whose distance falls below that offset
/// get clamped to full weight instead of the fast path's decayed
/// weight. That changes their contribution to the weighted average
/// relative to neighbours that stay above the offset, so the two
/// outputs must differ somewhere while staying finite and in range.
///
/// `sigma` here is set well above the CLI's "heavy noise" guidance.
/// The synthetic frame is a solid block edge rather than real
/// per-pixel noise, so patch distances only take a few discrete
/// values, either zero or a multiple of one mismatched tap's
/// contribution, instead of the small continuum real sensor noise
/// would produce. A small, realistic sigma would sit below every
/// nonzero distance and never clamp anything. The larger sigma exists
/// purely to land the offset between two of those discrete steps.
#[test]
fn hq_noise_floor_changes_output() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frame = make_frame_with_noisy_region(w, h, 1, 0.5, 8, 8, 3, 0.9);

    let mut fast = NlmDenoiser::<R>::new(&client, base_params(), w, h);
    fast.push_frame(&frame);
    let fast_out = fast.denoise().unwrap().unwrap().to_vec();

    let hq_params = NlmParams {
        hq: Some(HqParams {
            auto_strength: false,
            noise_floor: true,
            sigma: 40.0 / 255.0,
        }),
        ..base_params()
    };
    let mut hq = NlmDenoiser::<R>::new(&client, hq_params, w, h);
    hq.push_frame(&frame);
    let hq_out = hq.denoise().unwrap().unwrap().to_vec();

    let mut max_diff = 0.0f32;
    for (i, (&f, &q)) in fast_out.iter().zip(hq_out.iter()).enumerate() {
        assert!(q.is_finite(), "pixel {i}: non-finite HQ output {q}");
        assert!((0.0..=1.0).contains(&q), "pixel {i}: out-of-range HQ output {q}");
        max_diff = max_diff.max((f - q).abs());
    }

    assert!(
        max_diff > 1e-3,
        "expected the noise floor to change the output somewhere, max diff was {max_diff}"
    );
}

/// Mirrors `spatial::uniform_image_passthrough`: every patch distance
/// is zero on a flat frame, so both HQ features are no-ops and the
/// output must stay unchanged.
#[test]
fn hq_uniform_input_passthrough() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frame = make_uniform_frame(w, h, 1, 0.5);

    let params = NlmParams {
        hq: Some(HqParams::with_sigma(8.0 / 255.0)),
        ..base_params()
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);
    let result = denoiser.denoise().unwrap().unwrap().to_vec();

    for (i, &v) in result.iter().enumerate() {
        assert!((v - 0.5).abs() < 1e-5, "pixel {i}: expected 0.5, got {v}");
    }
}

/// Smoke test: HQ with a live temporal window over a short synthetic
/// sequence must produce finite, in-range output for every frame it
/// emits, whether during pushes or from the trailing flush.
#[test]
fn hq_temporal_smoke() {
    let client = make_client();
    let w = 16;
    let h = 16;

    let params = NlmParams {
        temporal_radius: 1,
        hq: Some(HqParams::with_sigma(6.0 / 255.0)),
        ..base_params()
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);

    let frames: Vec<Vec<f32>> = (0..5)
        .map(|i| make_frame_with_noisy_region(w, h, 1, 0.5, 6 + i, 8, 2, 0.8))
        .collect();

    let mut emitted = 0usize;
    let check = |frame: &[f32]| {
        for (i, &v) in frame.iter().enumerate() {
            assert!(v.is_finite(), "pixel {i}: non-finite output {v}");
            assert!((0.0..=1.0).contains(&v), "pixel {i}: out-of-range output {v}");
        }
    };

    for frame in &frames {
        denoiser.push_frame(frame);
        if let Some(result) = denoiser.denoise().unwrap() {
            check(result);
            emitted += 1;
        }
    }

    denoiser
        .flush(|frame| {
            check(frame);
            emitted += 1;
        })
        .unwrap();

    assert_eq!(emitted, frames.len(), "expected one output per pushed frame");
}

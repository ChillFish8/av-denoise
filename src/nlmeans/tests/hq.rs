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
            sigma_override: Some(8.0 / 255.0),
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
            sigma_override: Some(40.0 / 255.0),
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

/// `sigma_override: None` measures the noise level from the pushed
/// frame instead of requiring a caller-supplied value, and the
/// measured sigma must still drive real denoising.
///
/// Uses per-pixel Gaussian noise rather than a solid noisy block. The
/// Immerkær estimator responds to genuine high-frequency variation. A
/// single flat block only disturbs its boundary ring, so it reads back
/// close to the noise floor and barely denoises anything.
#[test]
fn hq_auto_sigma_denoises() {
    let client = make_client();
    let w = 32;
    let h = 32;
    let frame = make_noisy_gaussian_frame(w, h, 1, 0.5, &[8.0 / 255.0]);

    let params = NlmParams {
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
        }),
        ..base_params()
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&frame);
    let result = denoiser.denoise().unwrap().unwrap().to_vec();

    let mut max_diff = 0.0f32;
    for (i, (&input, &output)) in frame.iter().zip(result.iter()).enumerate() {
        assert!(output.is_finite(), "pixel {i}: non-finite output {output}");
        assert!(
            (0.0..=1.0).contains(&output),
            "pixel {i}: out-of-range output {output}"
        );
        max_diff = max_diff.max((input - output).abs());
    }

    assert!(
        max_diff > 1e-3,
        "expected the auto-estimated sigma to actually denoise the input, max diff was {max_diff}"
    );
}

/// Mirrors `hq_temporal_smoke` but with the noise level measured
/// automatically instead of supplied up front.
#[test]
fn hq_auto_sigma_temporal_smoke() {
    let client = make_client();
    let w = 16;
    let h = 16;

    let params = NlmParams {
        temporal_radius: 1,
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
        }),
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

/// A `sigma_override` must skip automatic estimation entirely. Neither
/// scratch buffer is allocated, and (by construction) no estimate
/// kernel is ever launched.
#[test]
fn hq_override_skips_estimation() {
    let client = make_client();
    let w = 16;
    let h = 16;

    let params = NlmParams {
        hq: Some(HqParams::with_sigma(8.0 / 255.0)),
        ..base_params()
    };

    let denoiser = NlmDenoiser::<R>::new(&client, params, w, h);

    assert!(
        denoiser.noise_partials.is_none(),
        "sigma_override must skip allocating the partials scratch buffer"
    );
    assert!(
        denoiser.noise_results.is_none(),
        "sigma_override must skip allocating the results buffer"
    );
}

/// `reset_stream_state` (called by `flush`) must clear the noise
/// estimator's EMA so a new stream doesn't inherit the previous
/// stream's noise level. Verified by observable behaviour. Pushing a
/// low-noise frame right after a reset must derive exactly the same
/// `h2_inv_norm` / `noise_offset` as a brand-new denoiser that only
/// ever saw that frame, instead of a value blended with the earlier
/// high-noise estimate.
#[test]
fn hq_reset_clears_noise_state() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let noisy = make_frame_with_noisy_region(w, h, 1, 0.5, 8, 8, 3, 0.9);
    let low = make_uniform_frame(w, h, 1, 0.5);

    let params = NlmParams {
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
        }),
        ..base_params()
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params.clone(), w, h);
    denoiser.push_frame(&noisy);
    denoiser.denoise().unwrap();

    denoiser.reset_stream_state();
    denoiser.push_frame(&low);
    denoiser.denoise().unwrap();

    let mut fresh = NlmDenoiser::<R>::new(&client, params, w, h);
    fresh.push_frame(&low);
    fresh.denoise().unwrap();

    assert_eq!(
        denoiser.h2_inv_norm, fresh.h2_inv_norm,
        "reset should clear the EMA so the next estimate starts fresh, not blended with stale state"
    );
    assert_eq!(
        denoiser.noise_offset, fresh.noise_offset,
        "reset should clear the EMA so the next estimate starts fresh, not blended with stale state"
    );
}

/// HQ auto-σ combined with the nlm-spatial pilot over a short temporal
/// sequence. Mirrors `hq_auto_sigma_temporal_smoke` but with the pilot
/// enabled. Every produced frame must stay finite and in range, and the
/// pipeline must still emit exactly one output per pushed frame.
#[test]
fn hq_pilot_temporal_end_to_end() {
    let client = make_client();
    let w = 16;
    let h = 16;

    let params = NlmParams {
        temporal_radius: 1,
        prefilter: PrefilterMode::NlmSpatial { strength_scale: 1.0 },
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
        }),
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

/// The nlm-spatial pilot changes what the main pass reads as its
/// distance signal, so HQ with the pilot enabled must diverge from
/// plain HQ (`prefilter: None`) on the same input.
///
/// Uses per-pixel Gaussian noise rather than a solid noisy block (as
/// `hq_noise_floor_changes_output` explains, a flat block only
/// disturbs its boundary ring, leaving patch distances elsewhere
/// identical whether or not the pilot ran).
#[test]
fn hq_pilot_differs_from_unguided() {
    let client = make_client();
    let w = 32;
    let h = 32;
    let frame = make_noisy_gaussian_frame(w, h, 1, 0.5, &[10.0 / 255.0]);

    let hq_params = |prefilter: PrefilterMode| NlmParams {
        prefilter,
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
        }),
        ..base_params()
    };

    let mut unguided = NlmDenoiser::<R>::new(&client, hq_params(PrefilterMode::None), w, h);
    unguided.push_frame(&frame);
    let unguided_out = unguided.denoise().unwrap().unwrap().to_vec();

    let mut piloted = NlmDenoiser::<R>::new(
        &client,
        hq_params(PrefilterMode::NlmSpatial { strength_scale: 1.0 }),
        w,
        h,
    );
    piloted.push_frame(&frame);
    let piloted_out = piloted.denoise().unwrap().unwrap().to_vec();

    let mut max_diff = 0.0f32;
    for (i, (&a, &b)) in unguided_out.iter().zip(piloted_out.iter()).enumerate() {
        assert!(b.is_finite(), "pixel {i}: non-finite piloted output {b}");
        assert!(
            (0.0..=1.0).contains(&b),
            "pixel {i}: out-of-range piloted output {b}"
        );
        max_diff = max_diff.max((a - b).abs());
    }

    assert!(
        max_diff > 1e-4,
        "expected the pilot to change HQ output somewhere, max diff was {max_diff}"
    );
}

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

    let result = denoiser
        .denoise()
        .unwrap()
        .unwrap()
        .as_f32()
        .expect("f32 denoiser")
        .to_vec();

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

    let result = denoiser
        .denoise()
        .unwrap()
        .unwrap()
        .as_f32()
        .expect("f32 denoiser")
        .to_vec();

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

    let result = denoiser
        .denoise()
        .unwrap()
        .unwrap()
        .as_f32()
        .expect("f32 denoiser")
        .to_vec();

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
    denoiser
        .flush(|frame| remaining.push(frame.as_f32().expect("f32 denoiser").to_vec()))
        .unwrap();
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

/// Deterministic per-frame noisy copy of `base`, decorrelated across
/// `seed`. Same Irwin-Hall hash as `noisy_copy`, generalised to a
/// non-uniform base image instead of a flat value.
fn noisy_copy_of(base: &[f32], seed: u32, sigma: f32) -> Vec<f32> {
    let unit_std = (1.0f32 / 3.0f32).sqrt();
    base.iter()
        .enumerate()
        .map(|(idx, &b)| {
            let idx = idx as u32;
            let mut sum = 0.0f32;
            for k in 0..4u32 {
                let mut hash = (idx * 4 + k)
                    .wrapping_mul(2654435761)
                    .wrapping_add(seed.wrapping_mul(0x9E37_79B9).wrapping_add(k));
                hash ^= hash >> 15;
                hash = hash.wrapping_mul(0x85EB_CA6B);
                hash ^= hash >> 13;
                sum += (hash as f32 / u32::MAX as f32) - 0.5;
            }
            (b + (sum / unit_std) * sigma).clamp(0.0, 1.0)
        })
        .collect()
}

fn psnr(reference: &[f32], test: &[f32]) -> f64 {
    let mse: f64 = reference
        .iter()
        .zip(test.iter())
        .map(|(&r, &t)| {
            let d = (r as f64) - (t as f64);
            d * d
        })
        .sum::<f64>()
        / reference.len() as f64;
    if mse <= 1e-20 {
        return 999.0;
    }
    10.0 * (1.0f64 / mse).log10()
}

/// Structured content for the search-radius regression tests below.
/// Combines a gradient (a smooth region for NLM to average) with a
/// block of a different value (an edge NLM should preserve rather
/// than blur across).
fn structured_base(w: u32, h: u32) -> Vec<f32> {
    let mut base = make_gradient_frame(w, h, 0.2, 0.8);
    let bx0 = w / 3;
    let by0 = h / 3;
    for y in by0..by0 * 2 {
        for x in bx0..bx0 * 2 {
            base[(y * w + x) as usize] = 0.15;
        }
    }
    base
}

/// Runs `params` through the windowed (default) dispatch and again
/// through the separable dispatch (forced via the public
/// `use_separable` flag, an independently-implemented path that
/// doesn't share the windowed pair kernel's code), denoising `frames`
/// of noisy copies of `base` both times. Returns `(windowed_psnr,
/// separable_psnr)` against `base`.
fn windowed_vs_separable_psnr(
    client: &cubecl::prelude::ComputeClient<R>,
    params: &NlmParams,
    w: u32,
    h: u32,
    base: &[f32],
    frames: &[Vec<f32>],
) -> (f64, f64) {
    let mut windowed = NlmDenoiser::<R>::new(client, params.clone(), w, h);
    for frame in frames {
        windowed.push_frame(frame);
    }
    let windowed_result = windowed
        .denoise()
        .unwrap()
        .unwrap()
        .as_f32()
        .expect("f32 denoiser")
        .to_vec();

    let mut separable = NlmDenoiser::<R>::new(client, params.clone(), w, h);
    separable.use_separable = true;
    for frame in frames {
        separable.push_frame(frame);
    }
    let separable_result = separable
        .denoise()
        .unwrap()
        .unwrap()
        .as_f32()
        .expect("f32 denoiser")
        .to_vec();

    (psnr(base, &windowed_result), psnr(base, &separable_result))
}

/// The backward temporal weight in `nlm_fused_pair_accumulate_window[_ref]`
/// must be measured against the same centre patch as the value it
/// multiplies. A weight measured against a shifted patch instead grows
/// wrong with the search offset, so this checks the windowed dispatch
/// against the independent separable dispatch at a search radius large
/// enough to expose a shift.
#[test]
fn temporal_windowed_matches_separable_at_search_5_and_6() {
    let client = make_client();
    let w = 128;
    let h = 128;
    let base = structured_base(w, h);

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
        let frames: Vec<Vec<f32>> = (0..9).map(|i| noisy_copy_of(&base, i, sigma)).collect();

        let (windowed_psnr, separable_psnr) =
            windowed_vs_separable_psnr(&client, &params, w, h, &base, &frames);

        assert!(
            (windowed_psnr - separable_psnr).abs() < 1.5,
            "search_radius={search_radius}: windowed ({windowed_psnr:.2} dB) should track \
             separable ({separable_psnr:.2} dB) within measurement noise"
        );
    }
}

/// Same check as [`temporal_windowed_matches_separable_at_search_5_and_6`]
/// for `nlm_fused_pair_accumulate_window_ref`, the variant that reads
/// patch distances from a prefiltered reference clip instead of the raw
/// input. A prefilter is active so both the windowed and separable
/// dispatches route through their `_ref` kernels.
#[test]
fn temporal_windowed_ref_matches_separable_ref_at_search_5_and_6() {
    let client = make_client();
    let w = 128;
    let h = 128;
    let base = structured_base(w, h);

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
        let frames: Vec<Vec<f32>> = (0..9).map(|i| noisy_copy_of(&base, i, sigma)).collect();

        let (windowed_psnr, separable_psnr) =
            windowed_vs_separable_psnr(&client, &params, w, h, &base, &frames);

        assert!(
            (windowed_psnr - separable_psnr).abs() < 1.5,
            "search_radius={search_radius}: windowed ({windowed_psnr:.2} dB) should track \
             separable ({separable_psnr:.2} dB) within measurement noise"
        );
    }
}

/// Same check as [`temporal_windowed_matches_separable_at_search_5_and_6`]
/// at the maximum supported search radius. Ignored by default. The
/// windowed kernel's fully unrolled window loop at this size overflows a
/// debug build's codegen stack even at the stack size
/// `.cargo/config.toml` sets (the spatial windowed kernel hits the same
/// limit). Release builds compile it fine. Run with
/// `cargo test --release -- --ignored
/// temporal_windowed_matches_separable_at_the_search_ceiling`.
#[test]
#[ignore = "debug build codegen overflows the stack at search_radius=8, run with --release"]
fn temporal_windowed_matches_separable_at_the_search_ceiling() {
    let client = make_client();
    let w = 128;
    let h = 128;
    let base = structured_base(w, h);

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
    let frames: Vec<Vec<f32>> = (0..9).map(|i| noisy_copy_of(&base, i, sigma)).collect();

    let (windowed_psnr, separable_psnr) = windowed_vs_separable_psnr(&client, &params, w, h, &base, &frames);

    assert!(
        (windowed_psnr - separable_psnr).abs() < 1.5,
        "search_radius={MAX_SEARCH_RADIUS}: windowed ({windowed_psnr:.2} dB) should track \
         separable ({separable_psnr:.2} dB) within measurement noise"
    );
}

/// Uniform-content sanity check at the same search radii as
/// [`temporal_windowed_matches_separable_at_search_5_and_6`]. Uniform
/// input makes every patch distance zero regardless of which pixel a
/// kernel reads, so this cannot catch a mis-centred weight, but it does
/// catch a kernel reading or writing outside its intended memory region,
/// which would pull in unrelated data and break uniformity even here.
#[test]
fn temporal_uniform_passthrough_search_5_and_6() {
    let client = make_client();
    let w = 64;
    let h = 64;
    let frame = make_uniform_frame(w, h, 1, 0.5);

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

        let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
        for _ in 0..5 {
            denoiser.push_frame(&frame);
        }
        let result = denoiser
            .denoise()
            .unwrap()
            .unwrap()
            .as_f32()
            .expect("f32 denoiser")
            .to_vec();

        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - 0.5).abs() < 1e-3,
                "search_radius={search_radius}: pixel {i} expected ~0.5, got {v}"
            );
        }
    }
}

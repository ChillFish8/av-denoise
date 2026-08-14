use cubecl::prelude::*;

use super::helpers::*;
use crate::nlmeans::kernels::nlm_weight_ratio_partial;
use crate::nlmeans::*;

/// Turning `track_weight_sq` on adds a write to a buffer nothing else in
/// the pipeline reads, so the denoised pixels themselves must come out
/// bit-for-bit identical either way.
///
/// This is the safety property the whole tree of kernel edits rests on.
/// Every windowed and separable kernel gained a new argument and a new
/// register, and none of that may perturb `accum`, `weight_sum`, or
/// `max_weight`, which is exactly what `nlm_finish` still reads from.
#[test]
fn tracking_does_not_change_the_output() {
    let client = make_client();
    let w = 24;
    let h = 24;
    let frame_a = make_noisy_gaussian_frame(w, h, 1, 0.5, &[0.06]);
    let frame_b = make_noisy_gaussian_frame(w, h, 1, 0.5, &[0.06]);
    let frame_c = make_noisy_gaussian_frame(w, h, 1, 0.5, &[0.06]);

    let base_params = NlmParams {
        temporal_radius: 1,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        track_weight_sq: false,
        hq: None,
    };

    let mut off = NlmDenoiser::<R>::new(&client, base_params.clone(), w, h);
    off.push_frame(&frame_a);
    off.push_frame(&frame_b);
    off.push_frame(&frame_c);
    let off_out = off.denoise().unwrap().unwrap().to_vec();

    let on_params = NlmParams {
        track_weight_sq: true,
        ..base_params
    };
    let mut on = NlmDenoiser::<R>::new(&client, on_params, w, h);
    on.push_frame(&frame_a);
    on.push_frame(&frame_b);
    on.push_frame(&frame_c);
    let on_out = on.denoise().unwrap().unwrap().to_vec();

    assert_eq!(
        off_out, on_out,
        "track_weight_sq must not change a single denoised pixel"
    );
}

/// Shared parameters for the ratio tests below. A high strength keeps
/// every weight pinned near its Welsch ceiling regardless of what a
/// patch distance rounds to, though on the uniform frames these tests
/// use, every patch distance is exactly zero and the strength does not
/// actually matter.
fn ratio_params(temporal_radius: u32) -> NlmParams {
    NlmParams {
        temporal_radius,
        search_radius: 1,
        patch_radius: 1,
        strength: 50.0,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        track_weight_sq: true,
        hq: None,
    }
}

/// Pins `residual_ratio_sqrt` to an exact value on a uniform frame,
/// derived from the actual kernel path these parameters select rather
/// than assumed.
///
/// A `patch_radius` of 1 sits below `SEPARABLE_THRESHOLD`, so
/// `run_denoise_kernels` picks the windowed path. At a `temporal_radius`
/// of 0 the only dispatch is `dispatch_fused_single_window_iter`, which
/// launches `nlm_fused_single_window` (`use_reference` is false since no
/// prefilter is set).
///
/// That kernel walks the full `(2 * search_radius + 1)^2 = 9` offsets in
/// the search window and skips only the centre `(0, 0)` one, so it
/// accumulates 8 candidate weights. On a uniform frame every patch
/// distance is exactly 0, `noise_offset` is 0 with no HQ noise floor, so
/// `welsch_weight(0, h2_inv_norm, 0)` reduces to `exp(0)`, exactly 1.0,
/// for every one of those 8 candidates whatever `h2_inv_norm` comes out
/// to. `max_weight` is therefore also exactly 1.0.
///
/// `nlm_finish` adds the centre pixel back in as `m`, `wref *
/// max_weight`, which is `self_weight * 1.0`. With `self_weight` at
/// 1.0 that makes `m` exactly 1.0 too, so the finished denominator,
/// `weight_sum + m`, comes to `8 + 1 = 9`.
///
/// `nlm_weight_ratio_partial` computes `(wsq + m * m) / (ws + m)^2` per
/// pixel. `wsq` is the sum of the same 8 candidate weights squared,
/// which is also 8 since every weight is exactly 1. So the ratio comes
/// to `(8 + 1) / 9^2`, which is `9 / 81`, which is `1 / 9`, and
/// `residual_ratio_sqrt` takes its square root, `1 / 3`.
#[test]
fn uniform_window_ratio_is_exact() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frame = make_uniform_frame(w, h, 1, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, ratio_params(0), w, h);
    denoiser.push_frame(&frame);
    denoiser
        .denoise()
        .unwrap()
        .expect("temporal_radius 0 outputs on the first push");

    let ratio = denoiser.residual_ratio_sqrt().unwrap();
    let expected = 1.0 / 3.0;
    assert!(
        (ratio - expected).abs() < 1e-4,
        "expected {expected}, got {ratio}"
    );
}

/// A single pixel whose accumulated weight is small but genuinely
/// nonzero, comfortably above the kernel's own "too small to divide by"
/// guard, must still produce a finite, in-range ratio rather than `NaN`.
///
/// `denom = weight_sum + m` is chosen around `1e-24`, which is real and
/// plausible on actual footage (one very weakly matching neighbour out of
/// a wide search window), and sits many orders of magnitude above the
/// `1e-30` the guard historically checked. But the kernel's own next step
/// squares that denominator before dividing by it, and `(1e-24)^2 =
/// 1e-48` underflows to exactly `0.0` in `f32`, whose smallest positive
/// value is about `1.4e-45`. `weight_sq_sum`, built from the same tiny
/// weight squared, underflows to `0.0` too, so the division becomes `0.0
/// / 0.0`, which is `NaN`. Squaring `1e-24` is exactly what this guard
/// exists to protect against, so the guard must be sized for the
/// quantity that is actually divided by, not the pre-squared value.
#[test]
fn small_but_real_denominator_does_not_divide_by_an_underflowed_square() {
    let client = make_client();

    let denom_seed = 5.406_747e-25f32;
    let weight_sum = vec![denom_seed];
    let max_weight = vec![denom_seed];
    let weight_sq_sum = vec![0.0f32];
    let self_weight = 1.0f32;
    let pixels = 1u32;

    let ws_buf = client.create_from_slice(f32::as_bytes(&weight_sum));
    let mw_buf = client.create_from_slice(f32::as_bytes(&max_weight));
    let wsq_buf = client.create_from_slice(f32::as_bytes(&weight_sq_sum));
    let partials_buf = client.empty(size_of::<f32>());

    let block = 256u32;
    let total_threads = block;
    unsafe {
        nlm_weight_ratio_partial::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(block),
            ArrayArg::from_raw_parts(ws_buf, pixels as usize),
            ArrayArg::from_raw_parts(wsq_buf, pixels as usize),
            ArrayArg::from_raw_parts(mw_buf, pixels as usize),
            ArrayArg::from_raw_parts(partials_buf.clone(), 1),
            self_weight,
            pixels,
            total_threads,
            block,
        );
    }

    let bytes = client.read_one(partials_buf).unwrap();
    let ratio = f32::from_bytes(&bytes)[0];

    assert!(
        !ratio.is_nan(),
        "a small but real denominator must not produce NaN, got {ratio}"
    );
    assert!(
        (0.0..=1.0).contains(&ratio),
        "a denominator this close to zero represents essentially no real match, so the ratio \
         should land at or near the no-match fallback of 1.0, got {ratio}"
    );
}

/// The design's own specified check. On a frame with real spatial
/// structure, not a flat field, the noise NLMeans actually leaves in its
/// output should track `base_sigma * residual_ratio_sqrt()`, the same
/// quantity `nl3d` hands its collaborative stage as the residual sigma.
///
/// `uniform_window_ratio_is_exact` above pins the formula analytically
/// on a uniform frame, the one case where every candidate weight is
/// forced to exactly 1.0 and the formula's underlying independence
/// assumption is exact by construction. That validates the arithmetic,
/// not the assumption. This test instead measures the real thing the
/// formula is supposed to predict. It denoises a noisy copy of a
/// textured frame, reads the result back, and compares its actual
/// standard deviation from the known clean reference against the
/// formula's own prediction.
///
/// The two are not expected to match exactly. `residual_ratio_sqrt`
/// treats every accumulated match weight as an independent sample, but
/// real NLMeans patches overlap and real grain is spatially correlated,
/// both of which break that assumption, so the true residual should
/// come out larger than the formula predicts, not smaller and not
/// exactly equal. `sigma_override` pins the front end's own belief about
/// the noise level to the exact value the frame was generated with, so
/// this isolates the ratio formula itself from how well noise
/// *estimation* would have recovered that same value.
#[test]
fn residual_ratio_matches_measured_residual_within_the_known_independence_gap() {
    let client = make_client();
    let w = 96;
    let h = 96;
    let true_sigma = 0.05f32;

    let clean = make_textured_frame(w, h);
    let noisy = noisy_field_over(&clean, w, h, true_sigma, 7);

    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 3,
        patch_radius: 3,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        track_weight_sq: true,
        hq: Some(HqParams::with_sigma(true_sigma)),
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    denoiser.push_frame(&noisy);
    let output = denoiser
        .denoise()
        .unwrap()
        .expect("temporal_radius 0 outputs on the first push")
        .to_vec();

    let ratio = denoiser.residual_ratio_sqrt().unwrap();
    let base_sigma = denoiser.current_sigmas()[0];
    assert!(
        (base_sigma - true_sigma).abs() < 1e-6,
        "sigma_override should pin current_sigmas exactly to the frame's true sigma, got \
         {base_sigma}"
    );

    let predicted = base_sigma * ratio;

    let n = output.len() as f64;
    let sq_err: f64 = output
        .iter()
        .zip(clean.iter())
        .map(|(&o, &c)| (o as f64 - c as f64).powi(2))
        .sum();
    let measured = (sq_err / n).sqrt() as f32;
    let measured_over_predicted = measured / predicted;

    // Reported for calibration. This is the single number that says
    // whether `residual_sigma_scale` compensates for this known
    // modelling gap or for something else.
    eprintln!(
        "residual sigma check: predicted={predicted:.6} measured={measured:.6} \
         measured/predicted={measured_over_predicted:.4}"
    );

    assert!(
        measured_over_predicted > 0.9,
        "measured residual {measured} fell notably below the predicted {predicted} \
         (measured/predicted={measured_over_predicted:.4}); the independence assumption \
         should make the formula underestimate the true residual, not overestimate it"
    );
    assert!(
        measured_over_predicted < 3.0,
        "measured residual {measured} is more than 3x the predicted {predicted} \
         (measured/predicted={measured_over_predicted:.4}), a wider gap than overlapping \
         patches and correlated grain should plausibly explain"
    );
}

/// A wider temporal window gathers more matching samples, so it has to
/// leave behind less residual noise.
///
/// Both frames are uniform and identical across every pushed slot, so
/// every candidate weight is exactly 1.0 at either radius, the same
/// argument `uniform_window_ratio_is_exact` walks through in detail.
/// With every weight equal, the ratio reduces to `1 / N` for the total
/// count `N` of unit weights folded in, including the centre self
/// weight, and `N` only grows when the temporal window does. This test
/// asserts that relational fact directly rather than pinning `N` at
/// radius 1, since nothing here depends on its exact value.
#[test]
fn ratio_falls_as_the_window_grows() {
    let client = make_client();
    let w = 16;
    let h = 16;
    let frame = make_uniform_frame(w, h, 1, 0.5);

    let mut radius0 = NlmDenoiser::<R>::new(&client, ratio_params(0), w, h);
    radius0.push_frame(&frame);
    radius0
        .denoise()
        .unwrap()
        .expect("temporal_radius 0 outputs on the first push");
    let ratio0 = radius0.residual_ratio_sqrt().unwrap();

    // The leading-edge mirror fills the temporal window's other slot
    // from the first push alone, so radius 1 needs 2 real pushes before
    // its first output, the same count `temporal::temporal_requires_full_window`
    // pins down.
    let mut radius1 = NlmDenoiser::<R>::new(&client, ratio_params(1), w, h);
    radius1.push_frame(&frame);
    assert!(
        radius1.denoise().unwrap().is_none(),
        "one push should not fill a temporal_radius 1 window yet"
    );
    radius1.push_frame(&frame);
    radius1
        .denoise()
        .unwrap()
        .expect("temporal_radius 1 outputs once the window is full");
    let ratio1 = radius1.residual_ratio_sqrt().unwrap();

    assert!(
        (0.0..=1.0).contains(&ratio0) && (0.0..=1.0).contains(&ratio1),
        "both ratios must lie in [0, 1], got radius0={ratio0} radius1={ratio1}"
    );
    assert!(
        ratio1 < ratio0,
        "expected the wider temporal window to leave less residual noise, got \
         radius0={ratio0} radius1={ratio1}"
    );
}

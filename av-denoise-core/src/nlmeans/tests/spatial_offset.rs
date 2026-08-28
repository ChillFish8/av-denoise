use super::helpers::*;
use crate::nlmeans::*;

/// The shared parameters for these tests, running spatially only with a
/// fixed sigma so the noise offset is nonzero and steady.
///
/// A fixed sigma also keeps automatic estimation inactive, so nothing
/// but the test itself ever touches the correlation state.
fn attenuation_params(sigma: f32) -> NlmParams {
    NlmParams {
        temporal_radius: 0,
        search_radius: 4,
        patch_radius: 3,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        hq: Some(HqParams {
            auto_strength: false,
            noise_floor: true,
            sigma_override: Some(sigma),
            temporal_confidence: false,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
        }),
    }
}

/// Setting the correlation state directly, as the estimator would after
/// folding a temporal sample, has to change the spatial weighting on
/// correlated content.
///
/// Reducing the noise-floor offset for nearby candidates changes which
/// patch distances reach full weight, which changes the weighted
/// average.
#[test]
fn rho_attenuation_changes_spatial_weighting_on_correlated_content() {
    let client = make_client();
    let w = 48;
    let h = 48;
    let sigma_marginal = 20.0 / 255.0;
    let sigma_pre = sigma_marginal / 0.375f32.sqrt();

    let frame = correlated_noisy_frame(w, h, 0.5, sigma_pre, 7);
    let params = attenuation_params(sigma_marginal);

    let mut rho_zero = NlmDenoiser::<R>::new(&client, params.clone(), w, h);
    rho_zero.push_frame(&frame);
    let rho_zero_out = rho_zero.denoise().unwrap().unwrap().to_vec();

    let mut rho_high = NlmDenoiser::<R>::new(&client, params, w, h);
    rho_high.rho_smoothed = Some(0.65);
    rho_high.push_frame(&frame);
    let rho_high_out = rho_high.denoise().unwrap().unwrap().to_vec();

    let mut max_diff = 0.0f32;
    for (i, (&a, &b)) in rho_zero_out.iter().zip(rho_high_out.iter()).enumerate() {
        assert!(a.is_finite() && b.is_finite(), "pixel {i}: non-finite output");
        assert!(
            (0.0..=1.0).contains(&a),
            "pixel {i}: rho=0 output out of range: {a}"
        );
        assert!(
            (0.0..=1.0).contains(&b),
            "pixel {i}: rho=0.65 output out of range: {b}"
        );
        max_diff = max_diff.max((a - b).abs());
    }

    assert!(
        max_diff > 1e-4,
        "expected rho attenuation to change the spatial weighting somewhere, max diff was {max_diff}"
    );
}

/// The windowed and separable paths must agree once correlation is
/// taken into account.
///
/// The windowed kernel reads each candidate's offset from the table,
/// while the separable path computes the same value on the host for
/// each dispatched candidate. Both have to arrive at the same factor.
///
/// The comparison covers interior pixels only, keeping a margin clear of
/// every clamped read either path makes.
///
/// Forcing the separable path mirrors the cross-check in
/// `temporal::windowed_vs_separable_psnr`. Unlike that one, this
/// compares pixels directly rather than through PSNR.
///
/// The two paths already handle clamped borders differently, which this
/// change did not touch, so the test also asserts the same agreement
/// with no correlation. That separates the existing difference from
/// anything the table wiring could have introduced.
#[test]
fn windowed_and_separable_agree_under_rho_attenuation() {
    let client = make_client();
    let w = 48;
    let h = 48;
    let sigma_marginal = 20.0 / 255.0;
    let sigma_pre = sigma_marginal / 0.375f32.sqrt();
    let margin = 7usize; // search_radius (4) + patch_radius (3)

    let frame = correlated_noisy_frame(w, h, 0.5, sigma_pre, 11);
    let params = attenuation_params(sigma_marginal);

    for rho in [0.0f32, 0.65] {
        let mut windowed = NlmDenoiser::<R>::new(&client, params.clone(), w, h);
        windowed.rho_smoothed = Some(rho);
        windowed.push_frame(&frame);
        let windowed_out = windowed.denoise().unwrap().unwrap().to_vec();

        let mut separable = NlmDenoiser::<R>::new(&client, params.clone(), w, h);
        separable.use_separable = true;
        separable.rho_smoothed = Some(rho);
        separable.push_frame(&frame);
        let separable_out = separable.denoise().unwrap().unwrap().to_vec();

        let mut max_diff_interior = 0.0f32;
        for y in margin..(h as usize - margin) {
            for x in margin..(w as usize - margin) {
                let idx = y * w as usize + x;
                max_diff_interior = max_diff_interior.max((windowed_out[idx] - separable_out[idx]).abs());
            }
        }

        assert!(
            max_diff_interior < 1e-3,
            "windowed and separable k=0 paths disagree on interior pixels at rho={rho}, \
             max diff {max_diff_interior}"
        );
    }
}

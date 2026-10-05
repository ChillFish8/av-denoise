use super::helpers::*;
use crate::bench_api::HostIo;
use crate::nlmeans::*;

/// Spatial-only parameters with a fixed sigma, so the noise offset is nonzero and steady.
///
/// A fixed sigma also keeps automatic estimation off, so only the test touches the correlation state.
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
            windowed_noise_estimation: false,
        }),
    }
}

/// Sets the correlation state directly, as the estimator would after folding a temporal sample.
#[test]
fn rho_attenuation_changes_spatial_weighting_on_correlated_content() {
    let client = make_client();
    let width = 48;
    let height = 48;
    let sigma_marginal = 20.0 / 255.0;
    let sigma_pre = sigma_marginal / 0.375f32.sqrt();

    let frame = correlated_noisy_frame(width, height, 0.5, sigma_pre, 7);
    let params = attenuation_params(sigma_marginal);

    let mut rho_zero = NlmDenoiser::<R>::new(&client, params.clone(), width, height);
    rho_zero.push_frame(&frame);
    let rho_zero_out = rho_zero.denoise().unwrap().unwrap();

    let mut rho_high = NlmDenoiser::<R>::new(&client, params, width, height);
    rho_high.rho_smoothed = Some(0.65);
    rho_high.push_frame(&frame);
    let rho_high_out = rho_high.denoise().unwrap().unwrap();

    let mut max_diff = 0.0f32;
    let pairs = rho_zero_out.iter().zip(rho_high_out.iter()).enumerate();
    for (i, (&zero_value, &high_value)) in pairs {
        assert!(
            zero_value.is_finite() && high_value.is_finite(),
            "pixel {i}: non-finite output"
        );
        assert!(
            (0.0..=1.0).contains(&zero_value),
            "pixel {i}: rho=0 output out of range: {zero_value}"
        );
        assert!(
            (0.0..=1.0).contains(&high_value),
            "pixel {i}: rho=0.65 output out of range: {high_value}"
        );
        max_diff = max_diff.max((zero_value - high_value).abs());
    }

    assert!(
        max_diff > 1e-4,
        "expected rho attenuation to change the spatial weighting somewhere, max diff was {max_diff}"
    );
}

/// The windowed kernel reads each candidate's offset from the table, while the separable path
/// computes it on the host.
///
/// Only interior pixels are compared, clear of every clamped read. The two paths already differ on clamped borders, so rho 0 is checked too. That separates the
/// border difference from anything the offset table could introduce.
#[test]
fn windowed_and_separable_agree_under_rho_attenuation() {
    let client = make_client();
    let width = 48;
    let height = 48;
    let sigma_marginal = 20.0 / 255.0;
    let sigma_pre = sigma_marginal / 0.375f32.sqrt();
    let margin = 7usize; // search_radius (4) + patch_radius (3)

    let frame = correlated_noisy_frame(width, height, 0.5, sigma_pre, 11);
    let params = attenuation_params(sigma_marginal);

    for rho in [0.0f32, 0.65] {
        let mut windowed = NlmDenoiser::<R>::new(&client, params.clone(), width, height);
        windowed.rho_smoothed = Some(rho);
        windowed.push_frame(&frame);
        let windowed_out = windowed.denoise().unwrap().unwrap();

        let mut separable = NlmDenoiser::<R>::new(&client, params.clone(), width, height);
        separable.use_separable = true;
        separable.rho_smoothed = Some(rho);
        separable.push_frame(&frame);
        let separable_out = separable.denoise().unwrap().unwrap();

        let mut max_diff_interior = 0.0f32;
        for y in margin..(height as usize - margin) {
            for x in margin..(width as usize - margin) {
                let idx = y * width as usize + x;
                let diff = (windowed_out[idx] - separable_out[idx]).abs();
                max_diff_interior = max_diff_interior.max(diff);
            }
        }

        assert!(
            max_diff_interior < 1e-3,
            "windowed and separable k=0 paths disagree on interior pixels at rho={rho}, \
             max diff {max_diff_interior}"
        );
    }
}

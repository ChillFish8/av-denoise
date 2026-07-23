use super::helpers::*;
use crate::nlmeans::*;

/// Shared params for the spatial-offset attenuation tests: k=0 only,
/// a fixed `sigma_override` so `noise_offset` is nonzero and stable
/// (auto estimation stays inactive, so nothing but the test itself
/// ever touches `rho_smoothed`).
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

/// Setting `rho_smoothed` directly (as the correlated-grain estimator
/// would after folding a temporal sample) must change the spatial k=0
/// weighting on correlated content relative to the `rho_smoothed = 0`
/// default: attenuating the noise-floor offset for near candidates
/// changes which patch distances clamp to full weight in
/// `welsch_weight`, which changes the weighted average.
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
    rho_high.rho_smoothed = 0.65;
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

/// The windowed k=0 kernel reads its offset from the LUT, indexed by
/// its comptime `(dx, dy)`; the separable k=0 kernel gets the same
/// value computed directly on the CPU per dispatched `(q_x, q_y)`.
/// Both must derive the identical `spatial_offset_factor(dx, dy, rho)`
/// for the same candidate, so under a nonzero `rho_smoothed` the two
/// independently-coded dispatch paths must still agree on interior
/// pixels (margin `search_radius + patch_radius`, clear of every
/// clamped-read border case either path takes). Forcing
/// `use_separable` mirrors `temporal::windowed_vs_separable_psnr`'s
/// cross-check; unlike that helper this compares interior pixels
/// directly rather than through PSNR, since both paths' clamped-border
/// handling (not touched by this change) already diverges near the
/// edges regardless of `rho_smoothed` — asserted below at `rho = 0`
/// too, to isolate that pre-existing divergence from anything this LUT
/// wiring could have introduced.
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
        windowed.rho_smoothed = rho;
        windowed.push_frame(&frame);
        let windowed_out = windowed.denoise().unwrap().unwrap().to_vec();

        let mut separable = NlmDenoiser::<R>::new(&client, params.clone(), w, h);
        separable.use_separable = true;
        separable.rho_smoothed = rho;
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

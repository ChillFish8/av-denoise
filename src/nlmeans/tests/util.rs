use super::helpers::*;
use crate::nlmeans::*;

#[test]
fn normalization_u8_roundtrip() {
    let original: Vec<u8> = (0..=255).collect();
    let normalized = normalize_u8(&original);
    let restored = denormalize_u8(&normalized);

    assert_eq!(original, restored);
}

#[test]
fn normalization_u16_roundtrip() {
    let original: Vec<u16> = (0..1024).chain(64000..=65535).collect();
    let normalized = normalize_u16(&original);
    let restored = denormalize_u16(&normalized);

    assert_eq!(original, restored);
}

/// Drives most weights toward zero by combining a near-maximum search
/// radius with extremely low strength (large `h2_inv_norm`), and on
/// noisy content so denominators land near the underflow guard in
/// `nlm_finish`. The output must contain no `inf`/`nan` regardless.
#[test]
fn extreme_params_produce_finite_output() {
    let client = make_client();
    let w = 32;
    let h = 32;
    let frame = make_frame_with_noisy_region(w, h, 1, 0.1, 16, 16, 5, 0.9);

    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 2,
        strength: 0.1,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        hq: None,
    };

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&frame);
    let result = d.denoise().unwrap().unwrap().to_vec();

    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "pixel {i}: non-finite output {v}");
        assert!((-0.01..=1.01).contains(&v), "pixel {i}: out-of-range output {v}");
    }
}

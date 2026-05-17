use crate::nlmeans::*;

#[test]
fn validate_rejects_oversized_patch_radius() {
    let params = NlmParams {
        patch_radius: MAX_PATCH_RADIUS + 1,
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn validate_rejects_oversized_search_radius() {
    let params = NlmParams {
        search_radius: MAX_SEARCH_RADIUS + 1,
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn validate_rejects_oversized_temporal_radius() {
    let params = NlmParams {
        temporal_radius: MAX_TEMPORAL_RADIUS + 1,
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn validate_rejects_non_positive_strength() {
    let zero = NlmParams {
        strength: 0.0,
        ..NlmParams::default()
    };
    assert!(zero.validate().is_err());

    let nan = NlmParams {
        strength: f32::NAN,
        ..NlmParams::default()
    };
    assert!(nan.validate().is_err());

    let inf = NlmParams {
        strength: f32::INFINITY,
        ..NlmParams::default()
    };
    assert!(inf.validate().is_err());
}

#[test]
fn validate_rejects_negative_self_weight() {
    let params = NlmParams {
        self_weight: -0.1,
        ..NlmParams::default()
    };
    assert!(params.validate().is_err());
}

#[test]
fn validate_accepts_defaults() {
    assert!(NlmParams::default().validate().is_ok());
}

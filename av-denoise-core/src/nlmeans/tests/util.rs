use super::helpers::*;
use crate::bench_api::HostIo;
use crate::nlmeans::*;

/// Very low strength on noisy content drives most weights toward zero, so the weight sums land near
/// the underflow guard in `nlm_finish`.
#[test]
fn extreme_params_produce_finite_output() {
    let client = make_client();
    let width = 32;
    let height = 32;
    let frame = make_frame_with_noisy_region(width, height, 1, 0.1, 16, 16, 5, 0.9);

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

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);
    let result = denoiser.denoise().unwrap().unwrap();

    for (i, &value) in result.iter().enumerate() {
        assert!(value.is_finite(), "pixel {i}: non-finite output {value}");
        assert!(
            (-0.01..=1.01).contains(&value),
            "pixel {i}: out-of-range output {value}"
        );
    }
}

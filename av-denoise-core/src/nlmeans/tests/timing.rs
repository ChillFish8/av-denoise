use cubecl::profile::TimingMethod;

use super::helpers::{make_client, make_noisy_gaussian_frame};
use crate::bench_api::HostIo;
use crate::nlmeans::*;

const SIZE: u32 = 64;

/// HQ parameters at `radius` with auto sigma.
fn hq_params(radius: u32) -> NlmParams {
    let options = NlmeansOptions {
        mode: DenoisingMode::Temporal { radius },
        ..NlmeansOptions::default()
    };
    let hq_options = NlmeansHqOptions {
        nlm: options,
        hq: HqParams::default(),
    };
    let algorithm = NlmeansAlgorithm::Hq(hq_options);

    resolve_params(&algorithm, ChannelMode::Luma)
}

/// Runs a stream through `params` and returns every output frame.
fn stream_outputs(params: NlmParams, sample_every: u32) -> Vec<Vec<f32>> {
    let client = make_client();
    let mut denoiser = NlmDenoiser::new(&client, params, SIZE, SIZE);
    denoiser.sample_device_time(sample_every);

    let mut outputs = Vec::new();
    for index in 0..20 {
        let base = 0.3 + index as f32 * 0.01;
        let frame = make_noisy_gaussian_frame(SIZE, SIZE, 1, base, &[0.03]);
        denoiser.push_frame(&frame);

        let denoised = denoiser.denoise().expect("denoise");
        outputs.extend(denoised);
    }

    denoiser
        .flush(|frame| outputs.push(frame.to_vec()))
        .expect("flush");

    let properties = client.properties();
    let has_device_timing = properties.timing_method == TimingMethod::Device;
    if sample_every > 0 && has_device_timing {
        let timing = denoiser.timing();
        assert!(timing.device_samples > 0, "no frame was sampled");
    }

    outputs
}

fn assert_sampling_leaves_output_unchanged(params: NlmParams) {
    let untimed_params = params.clone();
    let untimed = stream_outputs(untimed_params, 0);
    let timed = stream_outputs(params, 1);

    assert_eq!(timed, untimed);
}

#[test]
fn sampling_leaves_chunked_motion_compensation_unchanged() {
    let mut params = hq_params(MAX_TEMPORAL_RADIUS);
    params.motion_compensation = MotionCompensationMode::Mvtools {
        blksize: 16,
        overlap: 8,
        search_radius: 4,
        pyramid_levels: 2,
        estimation: MotionEstimation::Direct,
    };

    assert_sampling_leaves_output_unchanged(params);
}

#[test]
fn sampling_leaves_the_chunked_separable_path_unchanged() {
    let mut params = hq_params(2);
    params.patch_radius = 9;
    params.motion_compensation = MotionCompensationMode::None;

    assert_sampling_leaves_output_unchanged(params);
}

#[test]
fn sampling_leaves_the_nlm_spatial_pilot_unchanged() {
    let mut params = hq_params(2);
    params.prefilter = PrefilterMode::NlmSpatial {
        strength_scale: DEFAULT_PILOT_STRENGTH_SCALE,
    };

    assert_sampling_leaves_output_unchanged(params);
}

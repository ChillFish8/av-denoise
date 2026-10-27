use super::helpers::{make_client, noisy_frames, static_clip_params};
use crate::bench_api::HostIo;
use crate::nl4d::Nl4dDenoiser;
use crate::nlmeans::HqParams;

const SIZE: u32 = 64;

/// Runs an auto-sigma nl4d stream and returns every output frame.
fn outputs(sample_every: u32) -> (Vec<Vec<f32>>, Nl4dDenoiser) {
    let client = make_client();
    let mut params = static_clip_params(1);
    params.nlm.hq = Some(HqParams::default());
    let mut denoiser = Nl4dDenoiser::new(&client, params, SIZE, SIZE).expect("valid params");
    denoiser.sample_device_time(sample_every);

    let frames = noisy_frames(SIZE, SIZE, 1, 8);
    let mut outputs = Vec::new();
    for frame in &frames {
        denoiser.push_frame(frame);

        let denoised = denoiser.denoise().expect("denoise");
        outputs.extend(denoised);
    }

    denoiser
        .flush(|frame| outputs.push(frame.to_vec()))
        .expect("flush");

    (outputs, denoiser)
}

#[test]
fn sampling_device_time_leaves_the_output_unchanged() {
    let (untimed, _) = outputs(0);
    let (timed, _) = outputs(1);

    assert_eq!(timed, untimed);
}

#[test]
fn sampled_frames_report_device_time() {
    let (_, mut denoiser) = outputs(1);

    let timing = denoiser.timing();

    // RADV and every desktop Vulkan driver this runs on expose timestamp queries.
    assert_eq!(timing.device_samples, 8);
    assert!(timing.device_time > std::time::Duration::ZERO);
}

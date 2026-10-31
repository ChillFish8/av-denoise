use av_denoise_core::{ChannelMode, DenoisingMode, NlmTuning, NlmeansOptions};

use crate::host::{Algorithm, DenoiserOptions, HostDenoiser, Pending};

/// A spatial luma denoiser with small radii, since wide kernels cost codegen stack.
fn spatial_luma(size: u32) -> HostDenoiser {
    let tuning = NlmTuning {
        search_radius: Some(3),
        patch_radius: Some(2),
        ..NlmTuning::default()
    };
    let nlmeans_options = NlmeansOptions {
        tuning,
        ..NlmeansOptions::default()
    };
    let algorithm = Algorithm::Nlmeans(nlmeans_options);
    let options = DenoiserOptions::builder()
        .channel_mode(ChannelMode::Luma)
        .mode(DenoisingMode::Spacial)
        .algorithm(algorithm)
        .build();

    let accelerator = crate::testing::test_accelerator();
    let device = crate::testing::test_device();
    HostDenoiser::create(&[accelerator], &device, size, size, options).expect("denoiser construction failed")
}

/// Pushes one uniform frame and takes its readback out of the denoiser.
fn submit(denoiser: &mut HostDenoiser, size: u32) -> Pending {
    let plane = vec![128u8; (size * size) as usize];
    denoiser.push(&[&plane]).expect("push failed");

    let pending = denoiser.pending.pop_front();
    pending.expect("spatial mode submits one readback per push")
}

#[test]
fn pending_survives_denoiser_drop() {
    let pending = {
        let mut denoiser = spatial_luma(16);
        submit(&mut denoiser, 16)
    };

    let planes = pending.wait().expect("wait failed");

    assert_eq!(planes.len(), 1);
    assert_eq!(planes[0].len(), 16 * 16);

    for (index, &sample) in planes[0].iter().enumerate() {
        assert_eq!(sample, 128, "pixel {index}");
    }
}

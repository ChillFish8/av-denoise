use av_denoise_core::DenoisingMode;

use crate::accelerate::Accelerator;
use crate::planar::{
    ChannelIntent,
    FrameLayout,
    PlanarDenoiser,
    PlaneOptions,
    Subsampling,
    plane_profile_keys,
};
use crate::tune::TuningMode;
use crate::{Algorithm, Depth, Device};

#[test]
fn every_planar_host_reports_the_profile_key_its_options_predict() {
    let options = PlaneOptions {
        accelerators: vec![Accelerator::Vulkan],
        device: Device::Default,
        intent: ChannelIntent::LumaChroma,
        mode: DenoisingMode::Temporal { radius: 1 },
        algorithm: Algorithm::default(),
        luma_strength: None,
        chroma_strength: None,
        luma_lambda_ht: None,
        chroma_lambda_ht: None,
        tuning: TuningMode::Off,
        deterministic: false,
    };
    let layout = FrameLayout {
        width: 64,
        height: 48,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Eight,
    };

    let planar = PlanarDenoiser::create(&options, layout).expect("the denoiser builds");
    let expected = plane_profile_keys(&options, layout);
    let reported: Vec<String> = planar
        .hosts()
        .iter()
        .map(|host| host.profile_key().to_string())
        .collect();

    assert_eq!(reported, expected);
    assert_eq!(reported.len(), 2);
}

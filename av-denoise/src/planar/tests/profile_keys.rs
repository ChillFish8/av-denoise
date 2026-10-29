use av_denoise_core::DenoisingMode;
use av_denoise_core::tune::profile_key;

use crate::planar::{ChannelIntent, FrameLayout, PlanarDenoiser, PlaneOptions, Subsampling, plane_specs};
use crate::tune::TuningMode;
use crate::{Algorithm, Depth};

#[test]
fn every_planar_host_reports_the_profile_key_its_options_predict() {
    let options = PlaneOptions {
        accelerators: vec![crate::testing::test_accelerator()],
        device: crate::testing::test_device(),
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
    let specs = plane_specs(&options, layout);
    let expected: Vec<String> = specs.iter().map(profile_key).collect();
    let reported: Vec<String> = planar
        .hosts()
        .iter()
        .map(|host| {
            let spec = host.spec();
            profile_key(&spec)
        })
        .collect();

    assert_eq!(reported, expected);
    assert_eq!(reported.len(), 2);
}

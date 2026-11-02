use av_denoise_core::DenoisingMode;

use crate::planar::{ChannelIntent, FrameLayout, PlanarDenoiser, PlaneOptions, Subsampling, plane_spec};
use crate::tune::TuningMode;
use crate::{Algorithm, ChromaSiting, Depth};

#[test]
fn the_planar_host_runs_the_spec_its_options_predict() {
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
        chroma_siting: ChromaSiting::Unknown,
    };

    let planar = PlanarDenoiser::create(&options, layout).expect("the denoiser builds");
    let host = planar.host().expect("a built denoiser has a host");
    let expected = plane_spec(&options, layout);

    let reported = format!("{:?}", host.spec());
    let predicted = format!("{expected:?}");
    assert_eq!(reported, predicted);
}

use std::sync::Arc;

use av_denoise_core::DenoisingMode;
use av_denoise_core::tune::{PipelineProfile, TuneLedger};

use crate::planar::{ChannelIntent, FrameLayout, PlanarDenoiser, PlaneOptions, Subsampling};
use crate::tune::driver::build_alternative;
use crate::{Algorithm, Depth, TuningMode};

fn options() -> PlaneOptions {
    PlaneOptions {
        accelerators: vec![crate::testing::test_accelerator()],
        device: crate::testing::test_device(),
        intent: ChannelIntent::LumaChroma,
        mode: DenoisingMode::Temporal { radius: 1 },
        algorithm: Algorithm::default(),
        luma_strength: None,
        chroma_strength: None,
        luma_lambda_ht: None,
        chroma_lambda_ht: None,
        tuning: TuningMode::Auto,
        deterministic: false,
    }
}

fn layout() -> FrameLayout {
    FrameLayout {
        width: 64,
        height: 48,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Eight,
    }
}

#[test]
fn an_alternative_profile_builds_on_the_own_device() {
    let plane_options = options();
    let frame_layout = layout();
    let planar =
        PlanarDenoiser::create_on_own_device(&plane_options, frame_layout).expect("the denoiser builds");
    let placement = planar.placement().clone();
    let opened = placement.opened().expect("the denoiser opened a device");
    let own_device = opened.device.clone();
    let built_hosts = placement.handed_out();

    let ledger = TuneLedger::new(false);
    let profile = PipelineProfile::Baseline;
    let alternative = build_alternative(&planar, &ledger, profile).expect("the alternative builds");
    let reopened = placement.opened().expect("the device is still open");
    let alternative_hosts = alternative.hosts().len();
    let shares_placement = Arc::ptr_eq(alternative.placement(), &placement);
    let handed_out = placement.handed_out();

    assert_eq!(built_hosts, 2);
    assert!(shares_placement);
    assert_eq!(reopened.device, own_device);
    assert_eq!(handed_out, built_hosts + alternative_hosts);
}

use std::sync::Arc;

use av_denoise_core::DenoisingMode;
use av_denoise_core::tune::{BudgetRequest, SearchMode, TuneOutcome, TuneRequest};

use crate::planar::{
    ChannelIntent,
    FrameLayout,
    PlanarDenoiser,
    PlaneOptions,
    Planes,
    ReseedWindow,
    Subsampling,
};
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
        tuning: TuningMode::Off,
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

fn grey_frame(frame_layout: FrameLayout) -> Planes {
    let luma_bytes = frame_layout.luma_bytes();
    let chroma_bytes = frame_layout.chroma_bytes();

    Planes {
        y: vec![100; luma_bytes],
        u: vec![128; chroma_bytes],
        v: vec![128; chroma_bytes],
    }
}

fn families_request() -> TuneRequest {
    TuneRequest {
        force: false,
        mode: SearchMode::Families,
    }
}

#[test]
fn tuning_rebuilds_an_own_device_denoiser_on_its_device() {
    let plane_options = PlaneOptions {
        tuning: TuningMode::Auto,
        ..options()
    };
    let frame_layout = layout();
    let mut planar =
        PlanarDenoiser::create_on_own_device(&plane_options, frame_layout).expect("the denoiser builds");
    let placement = planar.placement().clone();
    let opened = placement.opened().expect("the denoiser opened a device");
    let own_device = opened.device.clone();
    let built_hosts = placement.handed_out();

    let frames = vec![grey_frame(frame_layout)];
    let budget = BudgetRequest::Seconds(0.0);
    let request = families_request();
    planar
        .tune(&frames, budget, &request)
        .expect("the denoiser rebuilds");

    let reopened = placement.opened().expect("the device is still open");
    let shares_placement = Arc::ptr_eq(planar.placement(), &placement);
    let handed_out = placement.handed_out();

    assert_eq!(built_hosts, 1);
    assert!(shares_placement);
    assert_eq!(reopened.device, own_device);
    assert_eq!(handed_out, 2 * built_hosts);
}

#[test]
fn a_skipped_pass_keeps_the_host() {
    let plane_options = options();
    let frame_layout = layout();
    let mut planar =
        PlanarDenoiser::create_on_own_device(&plane_options, frame_layout).expect("the denoiser builds");
    let placement = planar.placement().clone();
    let built_hosts = placement.handed_out();

    let frames = vec![grey_frame(frame_layout)];
    let request = families_request();
    let report = planar
        .tune(&frames, BudgetRequest::Full, &request)
        .expect("a skip never fails");

    let handed_out = placement.handed_out();
    let skipped = TuneOutcome::Skipped("tuning is off".to_string());

    assert_eq!(report.outcome, skipped);
    assert_eq!(handed_out, built_hosts);
}

#[test]
fn a_denoiser_without_a_host_rejects_reseeds_instead_of_panicking() {
    let plane_options = options();
    let frame_layout = layout();
    let mut planar = PlanarDenoiser::create(&plane_options, frame_layout).expect("the denoiser builds");
    let span = planar.window_span();
    let frame = grey_frame(frame_layout);
    let frames = vec![frame; span.frame_count()];
    planar.drop_host();

    let reseeded = planar.reseed(&frames);
    let window = ReseedWindow {
        frames: &frames,
        target: span.behind,
        at_clip_start: false,
        at_clip_end: false,
    };
    let reseeded_window = planar.reseed_window(window);
    let pushed = planar.push(&frames[0]);
    let received = planar.recv();

    assert!(reseeded.is_err());
    assert!(reseeded_window.is_err());
    assert!(pushed.is_err());
    assert!(received.is_err());
}

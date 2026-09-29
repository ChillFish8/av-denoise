// `temporal_opts` and the one test that uses it are the only things
// naming `Accelerator::Vulkan`, `Algorithm`, `Device`, and
// `ChannelIntent`. Their imports are gated the same way to keep
// cpu-only builds free of unused-import warnings.
#[cfg(feature = "vulkan")]
use av_denoise::accelerate::Accelerator;
#[cfg(feature = "vulkan")]
use av_denoise::{Algorithm, ChannelIntent, DenoisingMode, Device, PlanarDenoiser, PlaneOptions};

#[cfg(feature = "vulkan")]
use super::{tiny_layout, tiny_planes};
#[cfg(feature = "vulkan")]
use crate::pipeline::coordinator::OutputMsg;
#[cfg(feature = "vulkan")]
use crate::pipeline::worker::flush_worker;

/// Gated because it names the `Vulkan` accelerator variant, which only
/// exists when the `vulkan` feature is enabled.
#[cfg(feature = "vulkan")]
fn temporal_opts() -> PlaneOptions {
    PlaneOptions {
        accelerators: vec![Accelerator::Vulkan],
        device: Device::Default,
        intent: ChannelIntent::LumaChroma,
        mode: DenoisingMode::Temporal { radius: 1 },
        algorithm: Algorithm::default(),
        luma_strength: None,
        chroma_strength: None,
        luma_lambda_ht: None,
        chroma_lambda_ht: None,
    }
}

/// Gated because it depends on `temporal_opts`, which names the
/// `Vulkan` accelerator variant and only builds when the `vulkan`
/// feature is enabled.
#[cfg(feature = "vulkan")]
#[test]
fn flush_worker_errors_when_coordinator_has_disconnected() {
    let layout = tiny_layout();
    let mut wd = PlanarDenoiser::create(&temporal_opts(), layout).expect("denoiser construction failed");
    let planes = tiny_planes(layout);

    // One push into a temporal window leaves a trailing tail that
    // `flush` will pad and emit.
    wd.push(&planes).expect("push failed");

    let mut pending: std::collections::VecDeque<u64> = std::collections::VecDeque::new();
    pending.push_back(0);

    let (tx, rx) = crossbeam_channel::unbounded::<OutputMsg>();
    drop(rx);

    let mut warm_up = None;
    let err = flush_worker(&mut wd, &mut warm_up, &mut pending, &tx)
        .expect_err("expected the coordinator disconnect to surface as an error");

    assert!(
        err.to_string().contains("disconnect"),
        "error should mention the coordinator disconnect: {err}"
    );
}

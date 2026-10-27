// The test, its helper and their imports are gated because the `Vulkan` accelerator variant only
// exists with the `vulkan` feature, and gating the imports keeps cpu-only builds free of warnings.
#[cfg(feature = "vulkan")]
use std::collections::VecDeque;

#[cfg(feature = "vulkan")]
use av_denoise::PlanarDenoiser;

#[cfg(feature = "vulkan")]
use super::{temporal_opts, tiny_layout, tiny_planes};
#[cfg(feature = "vulkan")]
use crate::pipeline::coordinator::OutputMsg;
use crate::pipeline::worker::WorkerSettings;
#[cfg(feature = "vulkan")]
use crate::pipeline::worker::flush_worker;

#[cfg(feature = "vulkan")]
#[test]
fn flush_worker_errors_when_coordinator_has_disconnected() {
    let layout = tiny_layout();
    let options = temporal_opts();
    let mut denoiser = PlanarDenoiser::create(&options, layout).expect("denoiser construction failed");
    let planes = tiny_planes(layout);

    // One push into a temporal window leaves a trailing tail that `flush` pads and emits.
    denoiser.push(&planes).expect("push failed");

    let mut pending: VecDeque<u64> = VecDeque::new();
    pending.push_back(0);

    let (output_tx, output_rx) = crossbeam_channel::unbounded::<OutputMsg>();
    drop(output_rx);

    let mut warm_up = None;
    let err = flush_worker(&mut denoiser, &mut warm_up, &mut pending, &output_tx)
        .expect_err("expected the coordinator disconnect to surface as an error");

    assert!(
        err.to_string().contains("disconnect"),
        "error should mention the coordinator disconnect: {err}"
    );
}

#[test]
fn only_several_workers_without_shared_device_open_own_devices() {
    let several = WorkerSettings::for_run(2, false, false);
    let several_shared = WorkerSettings::for_run(2, false, true);
    let one = WorkerSettings::for_run(1, false, false);
    let timed = WorkerSettings::for_run(3, true, false);

    assert!(several.own_device);
    assert!(!several_shared.own_device);
    assert!(!one.own_device);
    assert!(timed.own_device);
    assert!(timed.device_timing);
    assert!(!several.device_timing);
}

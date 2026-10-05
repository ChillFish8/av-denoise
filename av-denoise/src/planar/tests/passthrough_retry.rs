use super::*;
use crate::accelerate::Accelerator;
use crate::{Algorithm, DenoisingMode};

/// Chroma-only intent, so `luma` is the disabled passthrough half and `chroma` is the one that can
/// report `QueueFull`.
fn chroma_only_options() -> PlaneOptions {
    PlaneOptions {
        accelerators: vec![Accelerator::Vulkan],
        device: Device::Default,
        intent: ChannelIntent::Chroma,
        mode: DenoisingMode::Spacial,
        algorithm: Algorithm::default(),
        luma_strength: None,
        chroma_strength: None,
        luma_lambda_ht: None,
        chroma_lambda_ht: None,
    }
}

fn fake_planes(layout: FrameLayout) -> Planes {
    let luma_pixels = layout.luma_pixels();
    let neutral = layout.depth.neutral_chroma();

    Planes {
        y: fill_plane(luma_pixels, neutral, layout.depth),
        u: layout.neutral_chroma_plane(),
        v: layout.neutral_chroma_plane(),
    }
}

#[test]
fn queue_full_retry_does_not_double_queue_the_passthrough_plane() {
    let layout = FrameLayout {
        width: 16,
        height: 16,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Eight,
    };
    let options = chroma_only_options();
    let mut denoiser = PlanarDenoiser::create(&options, layout).expect("denoiser construction failed");
    let planes = fake_planes(layout);

    // Spatial mode runs a depth-2 pipeline, so the first two pushes land directly.
    denoiser.push(&planes).expect("first push should land");
    denoiser.push(&planes).expect("second push should land");

    // The third push hits QueueFull on the chroma half.
    let err = denoiser.push(&planes).expect_err("expected QueueFull");
    assert!(
        matches!(err, DenoiserError::QueueFull),
        "expected QueueFull, got {err:?}"
    );

    // Drain one output, then retry the whole push for the same frame.
    denoiser.recv().expect("recv after drain failed");
    denoiser
        .push(&planes)
        .expect("retry push should land after drain");

    // Chroma accepted three frames and `recv` popped one, so the luma passthrough queue must hold
    // two and must not count the frame whose first attempt hit `QueueFull` twice.
    assert_eq!(
        denoiser.luma_passthrough.len(),
        2,
        "expected exactly one passthrough entry per chroma frame actually accepted, got {}",
        denoiser.luma_passthrough.len()
    );
}

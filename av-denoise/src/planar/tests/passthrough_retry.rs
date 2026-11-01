use super::*;
use crate::{Algorithm, DenoisingMode};

/// Chroma-only intent, so luma is the uncleaned plane and chroma is the group that can report
/// `QueueFull`.
fn chroma_only_options() -> PlaneOptions {
    PlaneOptions {
        accelerators: vec![crate::testing::test_accelerator()],
        device: crate::testing::test_device(),
        intent: ChannelIntent::Chroma,
        mode: DenoisingMode::Spacial,
        algorithm: Algorithm::default(),
        luma_strength: None,
        chroma_strength: None,
        luma_lambda_ht: None,
        chroma_lambda_ht: None,
        tuning: TuningMode::Off,
        deterministic: false,
    }
}

/// A frame whose uniform luma plane carries `marker`, so the frame an output came from shows.
fn marked_planes(layout: FrameLayout, marker: u16) -> Planes {
    let luma_pixels = layout.luma_pixels();

    Planes {
        y: fill_plane(luma_pixels, marker, layout.depth),
        u: layout.neutral_chroma_plane(),
        v: layout.neutral_chroma_plane(),
    }
}

#[test]
fn queue_full_retry_does_not_repeat_the_passthrough_plane() {
    let layout = FrameLayout {
        width: 16,
        height: 16,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Eight,
    };
    let options = chroma_only_options();
    let mut denoiser = PlanarDenoiser::create(&options, layout).expect("denoiser construction failed");
    let mut outputs = Vec::new();

    // Spatial mode runs a depth-2 pipeline, so the first two pushes land directly.
    let first = marked_planes(layout, 10);
    let second = marked_planes(layout, 11);
    let third = marked_planes(layout, 12);
    denoiser.push(&first).expect("first push should land");
    denoiser.push(&second).expect("second push should land");

    // The third push hits QueueFull.
    let err = denoiser.push(&third).expect_err("expected QueueFull");
    assert!(
        matches!(err, DenoiserError::QueueFull),
        "expected QueueFull, got {err:?}"
    );

    // Drain one output, then retry the whole push for the same frame.
    let received = denoiser.recv().expect("recv after drain failed");
    outputs.extend(received);
    denoiser.push(&third).expect("retry push should land after drain");
    denoiser
        .flush(|planes| outputs.push(planes))
        .expect("flush failed");

    // Three frames were accepted, so three come out, each with its own luma plane and none repeated.
    let markers: Vec<u8> = outputs.iter().map(|planes| planes.y[0]).collect();
    assert_eq!(markers, [10, 11, 12]);

    for (planes, source) in outputs.iter().zip([&first, &second, &third]) {
        assert_eq!(planes.y, source.y, "luma should pass through unchanged");
    }
}

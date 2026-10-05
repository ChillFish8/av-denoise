use super::*;
use crate::accelerate::Accelerator;
use crate::{Algorithm, DenoisingMode};

/// Runs `luma` and `chroma` as two real `HostDenoiser`s in spatial mode.
///
/// Spatial mode passes a uniform-valued plane through unchanged, so each plane can carry its own
/// marker value and the two halves drifting apart shows up.
fn luma_chroma_options() -> PlaneOptions {
    PlaneOptions {
        accelerators: vec![Accelerator::Vulkan],
        device: Device::Default,
        intent: ChannelIntent::LumaChroma,
        mode: DenoisingMode::Spacial,
        algorithm: Algorithm::default(),
        luma_strength: None,
        chroma_strength: None,
        luma_lambda_ht: None,
        chroma_lambda_ht: None,
    }
}

/// A uniform-valued frame whose luma and chroma planes encode `frame_index` with different formulas.
///
/// Pairing luma from one push with chroma from another makes the two encodings disagree.
fn marked_planes(layout: FrameLayout, frame_index: u8) -> Planes {
    let luma_pixels = layout.luma_pixels();
    let chroma_pixels = layout.chroma_pixels();
    let luma_marker = 10 + frame_index;
    let chroma_marker = 200 - frame_index;

    Planes {
        y: fill_plane(luma_pixels, luma_marker as u16, layout.depth),
        u: fill_plane(chroma_pixels, chroma_marker as u16, layout.depth),
        v: fill_plane(chroma_pixels, chroma_marker as u16, layout.depth),
    }
}

#[test]
fn distinct_u_and_v_planes_come_back_in_order() {
    let layout = FrameLayout {
        width: 16,
        height: 16,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Eight,
    };
    let options = luma_chroma_options();
    let mut denoiser = PlanarDenoiser::create(&options, layout).expect("denoiser construction failed");

    let luma_pixels = layout.luma_pixels();
    let chroma_pixels = layout.chroma_pixels();
    let planes = Planes {
        y: fill_plane(luma_pixels, 100, layout.depth),
        u: fill_plane(chroma_pixels, 60, layout.depth),
        v: fill_plane(chroma_pixels, 190, layout.depth),
    };
    denoiser.push(&planes).expect("push failed");

    let received = denoiser.recv().expect("recv failed");
    let denoised = received.expect("spatial mode emits one frame per push");

    for &sample in &denoised.u {
        assert!(sample.abs_diff(60) <= 2, "U sample {sample}, expected about 60");
    }

    for &sample in &denoised.v {
        assert!(sample.abs_diff(190) <= 2, "V sample {sample}, expected about 190");
    }
}

#[test]
fn queue_full_retries_never_desync_luma_and_chroma() {
    let layout = FrameLayout {
        width: 16,
        height: 16,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Eight,
    };
    let options = luma_chroma_options();
    let mut denoiser = PlanarDenoiser::create(&options, layout).expect("denoiser construction failed");

    // More pushes than the depth-2 pipeline holds, so this drives several `QueueFull` retries.
    const FRAME_COUNT: u8 = 6;
    let mut outputs: Vec<Planes> = Vec::new();

    for frame_index in 0..FRAME_COUNT {
        let planes = marked_planes(layout, frame_index);

        // The push, drain and retry sequence a streaming caller runs.
        let pushed = denoiser.push(&planes);
        let needs_retry = push_needs_retry(pushed).expect("push_needs_retry");
        if needs_retry {
            if let Some(denoised) = denoiser.recv().expect("recv failed") {
                outputs.push(denoised);
            }

            denoiser
                .push(&planes)
                .expect("retry push should land after drain");
        }
    }

    denoiser
        .flush(|denoised| outputs.push(denoised))
        .expect("flush failed");

    assert_eq!(
        outputs.len(),
        FRAME_COUNT as usize,
        "expected exactly one output frame per input frame, got {}",
        outputs.len()
    );

    for (position, denoised) in outputs.iter().enumerate() {
        let luma_marker = denoised.y[0];
        let chroma_marker = denoised.u[0];
        let index_from_luma = luma_marker - 10;
        let index_from_chroma = 200 - chroma_marker;

        assert_eq!(
            index_from_luma, index_from_chroma,
            "luma marker {luma_marker} (frame {index_from_luma}) and chroma marker {chroma_marker} \
             (frame {index_from_chroma}) disagree, so the luma and chroma pushes have drifted apart"
        );
        assert_eq!(
            index_from_luma as usize, position,
            "output {position} carries frame {index_from_luma}, so frames came back out of order"
        );
    }
}

#[test]
fn a_wrong_length_u_plane_is_rejected_without_advancing_either_half() {
    let layout = FrameLayout {
        width: 16,
        height: 16,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Eight,
    };
    let options = luma_chroma_options();
    let mut denoiser = PlanarDenoiser::create(&options, layout).expect("denoiser construction failed");

    let mut short_u = marked_planes(layout, 0);
    short_u.u.pop();

    let rejected = denoiser.push(&short_u);
    let is_plane_mismatch = matches!(
        rejected,
        Err(DenoiserError::Engine(av_denoise_core::Error::PlaneMismatch(_)))
    );
    assert!(is_plane_mismatch, "got {rejected:?}");

    const FRAME_COUNT: u8 = 4;
    let mut outputs: Vec<Planes> = Vec::new();

    for frame_index in 0..FRAME_COUNT {
        let planes = marked_planes(layout, frame_index);

        let pushed = denoiser.push(&planes);
        let needs_retry = push_needs_retry(pushed).expect("push_needs_retry");
        if needs_retry {
            if let Some(denoised) = denoiser.recv().expect("recv failed") {
                outputs.push(denoised);
            }

            denoiser
                .push(&planes)
                .expect("retry push should land after drain");
        }
    }

    denoiser
        .flush(|denoised| outputs.push(denoised))
        .expect("flush failed");

    assert_eq!(outputs.len(), FRAME_COUNT as usize);

    for (position, denoised) in outputs.iter().enumerate() {
        let index_from_luma = denoised.y[0] - 10;
        let index_from_chroma = 200 - denoised.u[0];

        assert_eq!(index_from_luma as usize, position, "luma came back out of order");
        assert_eq!(
            index_from_chroma as usize, position,
            "chroma came back out of order"
        );
    }
}

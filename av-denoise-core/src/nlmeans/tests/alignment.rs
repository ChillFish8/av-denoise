use super::helpers::*;
use crate::bench_api::HostIo;
use crate::nlmeans::*;

fn luma_params(motion_compensation: MotionCompensationMode) -> NlmParams {
    NlmParams {
        temporal_radius: 2,
        search_radius: 2,
        patch_radius: 1,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation,
        hq: None,
    }
}

/// Pushes five uniform frames and asserts the centre one denoises to the same uniform value.
///
/// The sizes passed in give per-slot byte strides off the 32-byte buffer-offset alignment of the
/// test adapters, and a view bound at such an offset is rejected outright.
fn denoise_uniform(width: u32, height: u32, params: NlmParams) {
    let client = make_client();
    let frame = make_uniform_frame(width, height, 1, 0.5);

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    for _ in 0..5 {
        denoiser.push_frame(&frame);
    }

    let result = denoiser.denoise().unwrap().unwrap();

    assert_eq!(result.len(), (width * height) as usize);
    for (i, &value) in result.iter().enumerate() {
        assert!(
            (value - 0.5).abs() < 1e-3,
            "pixel {i}: expected 0.5 (uniform input passthrough), got {value}"
        );
    }
}

#[test]
fn denoises_when_the_frame_ring_slot_stride_is_unaligned() {
    // A 34x34 luma slot is 4624 bytes, a multiple of 16 but not 32, so every odd ring slot starts
    // 16 bytes short of a boundary.
    let params = luma_params(MotionCompensationMode::None);
    denoise_uniform(34, 34, params);
}

#[test]
fn denoises_the_nlm_spatial_pilot_when_the_reference_ring_slot_stride_is_unaligned() {
    // The pilot writes into the reference ring, so at 34x34 it targets a slot 4624 bytes in, 16
    // short of a boundary. Only a temporal radius above 0 reaches a slot past the first.
    let mut params = luma_params(MotionCompensationMode::None);
    params.prefilter = PrefilterMode::NlmSpatial {
        strength_scale: DEFAULT_PILOT_STRENGTH_SCALE,
    };
    denoise_uniform(34, 34, params);
}

#[test]
fn denoises_with_motion_compensation_when_the_pyramid_slot_stride_is_unaligned() {
    // A 42x28 luma slot is 4704 bytes, a clean 32-byte multiple, so only the motion pyramid is at
    // stake. Its 21x14 half-size level is 1176 bytes, 24 short of a boundary. This matches the
    // shape of a 720x548 chroma plane.
    let motion_compensation = MotionCompensationMode::Mvtools {
        blksize: 8,
        overlap: 4,
        search_radius: 2,
        pyramid_levels: 2,
        estimation: MotionEstimation::Direct,
    };
    let params = luma_params(motion_compensation);
    denoise_uniform(42, 28, params);
}

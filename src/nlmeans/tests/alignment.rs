//! Frame dimensions whose per-slot byte strides are not multiples of
//! the GPU's storage-buffer offset alignment (32 bytes). A buffer view
//! bound at such an offset is rejected outright, so these sizes used to
//! abort the whole pipeline rather than denoise.

use super::helpers::*;
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

/// Pushes five uniform frames and denoises the centre one, asserting
/// the result is the same uniform value it went in as.
fn denoise_uniform(w: u32, h: u32, params: NlmParams) {
    let client = make_client();
    let frame = make_uniform_frame(w, h, 1, 0.5);

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    for _ in 0..5 {
        d.push_frame(&frame);
    }
    let result = d.denoise().unwrap().unwrap().to_vec();

    assert_eq!(result.len(), (w * h) as usize);
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - 0.5).abs() < 1e-3,
            "pixel {i}: expected 0.5 (uniform input passthrough), got {v}"
        );
    }
}

#[test]
fn denoises_when_the_frame_ring_slot_stride_is_unaligned() {
    // 34x34 luma is 1156 f32 per ring slot, 4624 bytes: a multiple of
    // 16 but not of 32, so every odd ring slot starts 16 bytes short of
    // an alignment boundary.
    denoise_uniform(34, 34, luma_params(MotionCompensationMode::None));
}

#[test]
fn denoises_the_nlm_spatial_pilot_when_the_reference_ring_slot_stride_is_unaligned() {
    // The pilot writes its own output into the reference ring, so at
    // 34x34 it targets a slot starting 4624 bytes in, 16 short of an
    // alignment boundary. Only a temporal radius above 0 ever reaches
    // a slot past the first.
    let mut params = luma_params(MotionCompensationMode::None);
    params.prefilter = PrefilterMode::NlmSpatial {
        strength_scale: DEFAULT_PILOT_STRENGTH_SCALE,
    };
    denoise_uniform(34, 34, params);
}

#[test]
fn denoises_with_motion_compensation_when_the_pyramid_slot_stride_is_unaligned() {
    // 42x28 luma sizes the ring slot at 1176 f32 (4704 bytes, a clean
    // 32-byte multiple), so only the motion pyramid is at stake: its
    // /2 level is 21x14 = 294 f32 = 1176 bytes, 24 bytes short of a
    // 32-byte multiple. Same shape as the 720x548 chroma plane that
    // first hit this.
    denoise_uniform(
        42,
        28,
        luma_params(MotionCompensationMode::Mvtools {
            blksize: 8,
            overlap: 4,
            search_radius: 2,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        }),
    );
}

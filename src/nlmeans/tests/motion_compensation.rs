//! Motion-compensation end-to-end smoke tests. Verifies that the
//! pyramid build, analyse, and warp dispatches run without crashing
//! and that uniform input passes through unchanged.

use super::super::*;
use super::helpers::*;

/// Smoke test: a temporal denoiser with motion compensation enabled
/// must allocate, run the pyramid build, analyse, and warp dispatches
/// on every pushed frame, and produce a denoised frame that preserves
/// a uniform input. Catches plumbing mistakes (buffer allocation,
/// pyramid offsets, MC dispatch ordering) even before quality is
/// evaluated.
#[test]
fn motion_compensation_uniform_passthrough() {
    let client = make_client();
    let w = 32;
    let h = 32;
    let frame = make_uniform_frame(w, h, 1, 0.5);

    let params = NlmParams {
        temporal_radius: 1,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::Mvtools {
            blksize: 8,
            overlap: 4,
            search_radius: 2,
            pyramid_levels: 2,
        },
    };

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&frame);
    d.push_frame(&frame);
    d.push_frame(&frame);
    let result = d.denoise().unwrap().unwrap().to_vec();

    assert_eq!(result.len(), (w * h) as usize);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "pixel {i}: non-finite output {v}");
        assert!(
            (v - 0.5).abs() < 1e-3,
            "pixel {i}: expected 0.5 (uniform input passthrough), got {v}"
        );
    }
}

/// MC + bilateral prefilter compound test: the reference clip ring
/// must also be warped, and the denoise must remain finite/in-range.
#[test]
fn motion_compensation_with_bilateral_finite() {
    let client = make_client();
    let w = 32;
    let h = 32;
    let frame = make_uniform_frame(w, h, 1, 0.5);

    let params = NlmParams {
        temporal_radius: 1,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 1.0,
            sigma_r: 0.1,
        },
        motion_compensation: MotionCompensationMode::Mvtools {
            blksize: 8,
            overlap: 4,
            search_radius: 2,
            pyramid_levels: 2,
        },
    };

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&frame);
    d.push_frame(&frame);
    d.push_frame(&frame);
    let result = d.denoise().unwrap().unwrap().to_vec();

    assert_eq!(result.len(), (w * h) as usize);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "pixel {i}: non-finite output {v}");
        assert!((-0.01..=1.01).contains(&v), "pixel {i}: out-of-range output {v}");
    }
}

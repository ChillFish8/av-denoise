//! Regression guard for `denoise_submit`'s `Pending` outliving its
//! `NlmDenoiser`. The readback future is constructed via an `async move`
//! that owns a cloned `ComputeClient`; this test pins that contract.
//!
//! If a future cubecl release changes `read_async` so the returned
//! future borrows from the client, or if a refactor here re-introduces
//! a lifetime-erasing `transmute`, this test will use-after-free under
//! Miri / ASan and may fail on stock builds too.

use super::helpers::*;
use crate::nlmeans::*;

#[test]
fn pending_survives_denoiser_drop() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 0,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        hq: None,
    };

    let w = 16;
    let h = 16;
    let frame = make_uniform_frame(w, h, 1, 0.5);

    let pending = {
        let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
        denoiser.push_frame(&frame);
        denoiser
            .denoise_submit()
            .expect("submit failed")
            .expect("expected a pending readback for spatial mode")
        // `denoiser` is dropped here; `pending` must remain valid.
    };

    let out = pending.wait().expect("wait failed");
    assert_eq!(out.len(), (w * h) as usize);
    for (i, &v) in out.iter().enumerate() {
        assert!((v - 0.5).abs() < 1e-5, "pixel {i}: expected 0.5, got {v}");
    }
}

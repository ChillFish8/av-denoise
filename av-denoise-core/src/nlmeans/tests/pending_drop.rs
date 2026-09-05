//! Dropping a `Pending` that was polled but has not landed must settle its readback
//! rather than abandon it.
//!
//! On the wgpu backends the first poll maps a staging buffer that only the finished
//! readback unmaps.
//!
//! A buffer handed back to the device's staging pool while still mapped makes the next
//! submit that touches it fail on cubecl's device thread, and every later call on that
//! device then errors.

use super::helpers::*;
use crate::nlmeans::*;

/// Large enough that the GPU cannot have finished by the time the
/// first poll runs, a few microseconds after submit. The readback is
/// large too, so its staging page is not one a smaller read would pick.
/// The radii stay small because a wide kernel is what costs codegen
/// stack, and this test is about the readback, not the kernel.
const SIZE: u32 = 2048;

fn params() -> NlmParams {
    NlmParams {
        temporal_radius: 0,
        search_radius: 3,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        hq: None,
    }
}

fn submit(client: &cubecl::client::ComputeClient<R>, frame: &[f32]) -> Pending<R> {
    let mut denoiser = NlmDenoiser::<R>::new(client, params(), SIZE, SIZE);
    denoiser.push_frame(frame);
    denoiser
        .denoise_submit()
        .expect("submit failed")
        .expect("spatial mode submits one readback per push")
}

#[test]
fn dropping_a_polled_pending_settles_its_readback() {
    let client = make_client();
    let frame = make_uniform_frame(SIZE, SIZE, 1, 0.5);

    let pending = submit(&client, &frame);
    let not_ready = match pending.try_wait().expect("poll failed") {
        TryWait::NotReady(pending) => pending,
        TryWait::Ready(_) => {
            panic!("the first poll landed, so the drop path cannot be exercised at this size")
        },
    };
    drop(not_ready);

    let out = submit(&client, &frame)
        .wait()
        .expect("a readback after a dropped polled Pending must still work")
        .into_f32()
        .expect("f32 output");
    assert_eq!(out.len(), (SIZE * SIZE) as usize);
}

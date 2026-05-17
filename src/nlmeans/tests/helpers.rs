use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

pub(super) type R = WgpuRuntime;

pub(super) fn make_client() -> ComputeClient<R> {
    let device = <R as Runtime>::Device::default();
    R::client(&device)
}

pub(super) fn make_uniform_frame(w: u32, h: u32, ch: u32, val: f32) -> Vec<f32> {
    vec![val; (w * h * ch) as usize]
}

/// Creates a frame with a patch of noise (not just a single pixel)
/// so that NLMeans has matching noisy patches to work with.
#[allow(clippy::too_many_arguments)]
pub(super) fn make_frame_with_noisy_region(
    w: u32,
    h: u32,
    ch: u32,
    base: f32,
    cx: u32,
    cy: u32,
    radius: u32,
    noise_val: f32,
) -> Vec<f32> {
    let mut frame = vec![base; (w * h * ch) as usize];

    for dy in 0..=radius * 2 {
        for dx in 0..=radius * 2 {
            let x = cx + dx - radius;
            let y = cy + dy - radius;

            if x < w && y < h {
                for c in 0..ch {
                    frame[((y * w + x) * ch + c) as usize] = noise_val;
                }
            }
        }
    }

    frame
}

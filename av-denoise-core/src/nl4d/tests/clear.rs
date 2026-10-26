use super::helpers::{make_client, noisy_frames, static_clip_params};
use crate::bench_api::HostIo;
use crate::nl4d::Nl4dDenoiser;

const SIZE: u32 = 64;

fn denoiser() -> Nl4dDenoiser {
    let client = make_client();
    let params = static_clip_params(1);

    Nl4dDenoiser::new(&client, params, SIZE, SIZE).expect("valid params")
}

fn run(denoiser: &mut Nl4dDenoiser, count: usize) {
    let frames = noisy_frames(SIZE, SIZE, 1, count);

    for frame in &frames {
        denoiser.push_frame(frame);
        denoiser.denoise().expect("denoise");
    }
}

#[test]
fn steady_passes_reuse_the_region_normalise_cleared() {
    let mut denoiser = denoiser();
    run(&mut denoiser, 12);

    // Only the scene's first pass clears, one region per ring slot.
    assert_eq!(denoiser.region_clears_for_test(), 3);
}

#[test]
fn a_continuation_clears_the_regions_it_has_not_read_out() {
    let mut denoiser = denoiser();
    denoiser.mark_continuation();
    run(&mut denoiser, 8);

    // The whole ring once, then the one newest region whose frame was never read out.
    assert_eq!(denoiser.region_clears_for_test(), 4);
}

use std::time::Duration;

use super::*;

const SIZE: u32 = 64;

fn layout() -> FrameLayout {
    FrameLayout {
        width: SIZE,
        height: SIZE,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Eight,
    }
}

fn nl4d_options() -> PlaneOptions {
    let nl4d = Nl4dOptions::default();

    PlaneOptions {
        accelerators: vec![crate::testing::test_accelerator()],
        device: crate::testing::test_device(),
        intent: ChannelIntent::LumaChroma,
        mode: DenoisingMode::Temporal { radius: 1 },
        algorithm: Algorithm::Nl4d(nl4d),
        luma_strength: None,
        chroma_strength: None,
        luma_lambda_ht: None,
        chroma_lambda_ht: None,
        tuning: TuningMode::Off,
        deterministic: false,
    }
}

/// A plane of hashed 8-bit values that differ per frame and per channel.
fn hashed_plane(width: u32, height: u32, frame: usize, channel: u32) -> Vec<u8> {
    let frame_salt = frame as u32 * 40_503;
    let channel_salt = channel * 9_973;

    (0..width * height)
        .map(|index| {
            let mixed = index.wrapping_mul(2_654_435_761) ^ frame_salt ^ channel_salt;
            let value = 64 + (mixed >> 16) % 128;
            value as u8
        })
        .collect()
}

fn noisy_frames(count: usize) -> Vec<Planes> {
    let (chroma_width, chroma_height) = layout().chroma_dims();

    (0..count)
        .map(|frame| {
            let y_plane = hashed_plane(SIZE, SIZE, frame, 0);
            let u_plane = hashed_plane(chroma_width, chroma_height, frame, 1);
            let v_plane = hashed_plane(chroma_width, chroma_height, frame, 2);

            Planes {
                y: y_plane,
                u: u_plane,
                v: v_plane,
            }
        })
        .collect()
}

fn denoise_all(denoiser: &mut PlanarDenoiser, frames: &[Planes]) -> Vec<Planes> {
    let mut outputs = Vec::new();

    for planes in frames {
        loop {
            let pushed = denoiser.push(planes);
            let retry = push_needs_retry(pushed).expect("push");
            if !retry {
                break;
            }

            let received = denoiser.recv().expect("recv");
            outputs.extend(received);
        }
    }

    denoiser.flush(|planes| outputs.push(planes)).expect("flush");
    outputs
}

#[test]
fn both_planes_report_their_frames_and_blocked_time() {
    let frames = noisy_frames(10);
    let mut denoiser = PlanarDenoiser::create(&nl4d_options(), layout()).expect("create");
    denoise_all(&mut denoiser, &frames);

    let timing = denoiser.timing();
    let luma = timing.luma.expect("luma is denoised");
    let chroma = timing.chroma.expect("chroma is denoised");

    assert_eq!(luma.frames, 10);
    assert_eq!(chroma.frames, 10);
    assert!(luma.blocked > Duration::ZERO);
    assert!(luma.submit > Duration::ZERO);
    assert_eq!(luma.device_busy, None, "device time is off by default");
}

#[test]
fn sampling_device_time_estimates_busy_time_without_changing_output() {
    let frames = noisy_frames(10);
    let mut plain = PlanarDenoiser::create(&nl4d_options(), layout()).expect("create");
    let expected = denoise_all(&mut plain, &frames);

    let mut sampled = PlanarDenoiser::create(&nl4d_options(), layout()).expect("create");
    sampled.sample_device_time(2);
    let actual = denoise_all(&mut sampled, &frames);

    assert_eq!(actual, expected);

    let timing = sampled.timing();
    let luma = timing.luma.expect("luma is denoised");
    assert!(luma.device_busy.is_some_and(|busy| busy > Duration::ZERO));
}

#[test]
fn planes_are_listed_in_luma_chroma_yuv_order() {
    let plane = PlaneTiming::default();
    let timing = PlanarTiming {
        luma: Some(plane),
        chroma: Some(plane),
        yuv: Some(plane),
    };

    let planes = timing.planes();
    let names: Vec<&str> = planes.iter().map(|(name, _)| *name).collect();
    assert_eq!(names, ["luma", "chroma", "yuv"]);
}

#[test]
fn unset_planes_are_left_out_of_the_listing() {
    let plane = PlaneTiming::default();
    let timing = PlanarTiming {
        luma: None,
        chroma: None,
        yuv: Some(plane),
    };

    let planes = timing.planes();
    let names: Vec<&str> = planes.iter().map(|(name, _)| *name).collect();
    assert_eq!(names, ["yuv"]);
}

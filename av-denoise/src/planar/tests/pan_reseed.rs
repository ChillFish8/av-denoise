use super::reseed::{reseed_shifted, shifted_output_count, test_plane_options, window_of_span};
use super::*;
use crate::{MotionCompensationMode, MotionSearch, Nl4dOptions, NlmeansHqOptions, NlmeansOptions};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 192;
/// Past the 12 px an unseeded search reaches, so the camera model decides the vectors.
const PAN: f32 = 30.0;
const FRAMES: usize = 30;

fn pan_layout() -> FrameLayout {
    FrameLayout {
        width: WIDTH,
        height: HEIGHT,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Eight,
    }
}

/// A hashed value between 0 and 1 at lattice point (`cell_x`, `cell_y`).
fn lattice(cell_x: u32, cell_y: u32, seed: u32) -> f32 {
    let mut hash = cell_x.wrapping_mul(0x8DA6_B343) ^ cell_y.wrapping_mul(0xD816_3841);
    hash ^= seed.wrapping_mul(0x9E37_79B9);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x2C1B_3C6D);
    hash ^= hash >> 12;

    hash as f32 / u32::MAX as f32
}

/// Smooth value noise at spacing 6 pixels, so every block has one clear match.
fn texture(x: f32, y: f32, seed: u32) -> f32 {
    let scaled_x = x / 6.0;
    let scaled_y = y / 6.0;
    let cell_x = scaled_x.floor();
    let cell_y = scaled_y.floor();
    let blend_x = scaled_x - cell_x;
    let blend_y = scaled_y - cell_y;
    let left = cell_x as u32;
    let top_row = cell_y as u32;
    let top_left = lattice(left, top_row, seed);
    let top_right = lattice(left + 1, top_row, seed);
    let bottom_left = lattice(left, top_row + 1, seed);
    let bottom_right = lattice(left + 1, top_row + 1, seed);
    let top = top_left + (top_right - top_left) * blend_x;
    let bottom = bottom_left + (bottom_right - bottom_left) * blend_x;

    top + (bottom - top) * blend_y
}

/// A periodic texture with a few cycles across the frame, which block matches can alias on.
fn sine_texture(x: f32, y: f32) -> f32 {
    let phase_x = x / WIDTH as f32 * std::f32::consts::TAU * 3.0;
    let phase_y = y / HEIGHT as f32 * std::f32::consts::TAU * 2.0;

    0.5 + 0.2 * phase_x.sin() * phase_y.cos()
}

/// The content a pan clip slides.
#[derive(Debug, Clone, Copy)]
enum Content {
    ValueNoise,
    Sine,
}

/// One plane of frame `index`, panned `travel` luma pixels and sampling the texture `scale` luma
/// pixels apart, with grain.
fn pan_plane(
    content: Content,
    size: (u32, u32),
    scale: u32,
    travel: f32,
    index: usize,
    seed: u32,
) -> Vec<u8> {
    let (width, height) = size;
    let mut samples = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let luma_x = (x * scale) as f32 + travel;
            let luma_y = (y * scale) as f32;
            let clean = match content {
                Content::ValueNoise => texture(luma_x, luma_y, seed),
                Content::Sine => sine_texture(luma_x, luma_y),
            };
            let grain_seed = seed ^ (index as u32).wrapping_mul(0x27D4_EB2F);
            let grain = lattice(x, y, grain_seed) - 0.5;
            let value = 40.0 + 170.0 * clean + 12.0 * grain;
            let code = value.round() as u8;
            samples.push(code);
        }
    }

    samples
}

/// A grainy clip of `content` panned `travel(index)` luma pixels left at frame `index`.
fn pan_clip(content: Content, travel: impl Fn(usize) -> f32) -> Vec<Planes> {
    let chroma_size = pan_layout().chroma_dims();
    let luma_size = (WIDTH, HEIGHT);

    (0..FRAMES)
        .map(|index| {
            let offset = travel(index);
            Planes {
                y: pan_plane(content, luma_size, 1, offset, index, 1),
                u: pan_plane(content, chroma_size, 2, offset, index, 2),
                v: pan_plane(content, chroma_size, 2, offset, index, 3),
            }
        })
        .collect()
}

/// How far a pan speeding up by 10 px a frame to 70 px a frame has travelled at frame `index`.
///
/// Its top speed lies past the unseeded reach, so only a model built from earlier pairs could
/// follow it, and a reseed starting at top speed would then differ from the stream.
fn ramp_travel(index: usize) -> f32 {
    (1..=index).map(|step| (10 * step).min(70) as f32).sum()
}

/// The clips every test runs, steady fast pans the camera model decides, a slow periodic pan and a
/// pan ramping past the model's reach.
fn clips() -> [(&'static str, Vec<Planes>); 4] {
    [
        (
            "value noise fast pan",
            pan_clip(Content::ValueNoise, |index| PAN * index as f32),
        ),
        (
            "sine fast pan",
            pan_clip(Content::Sine, |index| PAN * index as f32),
        ),
        (
            "sine slow pan",
            pan_clip(Content::Sine, |index| 1.5 * index as f32),
        ),
        ("value noise ramp", pan_clip(Content::ValueNoise, ramp_travel)),
    ]
}

/// Streams every frame through `denoiser` and returns what it puts out, the flushed tail included.
fn stream(denoiser: &mut PlanarDenoiser, frames: &[Planes]) -> Vec<Planes> {
    let mut outputs = Vec::new();
    for frame in frames {
        denoiser.push(frame).unwrap();
        if let Some(planes) = denoiser.recv().unwrap() {
            outputs.push(planes);
        }
    }

    denoiser.flush(|planes| outputs.push(planes)).unwrap();
    outputs
}

fn assert_same_frame(got: &Planes, expected: &Planes, label: &str) {
    assert!(got.y == expected.y, "{label}: luma differs from streaming");
    assert!(got.u == expected.u, "{label}: u differs from streaming");
    assert!(got.v == expected.v, "{label}: v differs from streaming");
}

#[test]
fn nl4d_reseed_window_matches_streaming_on_a_fast_pan() {
    let options = PlaneOptions {
        algorithm: Algorithm::Nl4d(Nl4dOptions::default()),
        ..test_plane_options(2)
    };

    for (name, frames) in clips() {
        let mut denoiser = PlanarDenoiser::create(&options, pan_layout()).unwrap();
        let streamed = stream(&mut denoiser, &frames);
        assert_eq!(streamed.len(), FRAMES, "{name}");

        for target in [FRAMES / 2, 0, FRAMES - 1] {
            let got = reseed_shifted(&mut denoiser, &frames, target);
            let expected_len = shifted_output_count(&denoiser, FRAMES, target);
            assert_eq!(got.len(), expected_len, "{name} k={target}");

            for (offset, planes) in got.iter().enumerate() {
                let index = target + offset;
                let label = format!("{name} k={target} frame {index}");
                assert_same_frame(planes, &streamed[index], &label);
            }
        }
    }
}

/// Mid-clip only, since a repeated-edge window at the clip's ends reads the noise of its copies.
#[test]
fn nlmeans_reseed_matches_streaming_on_a_fast_pan() {
    let compensated = NlmeansOptions {
        motion_compensation: MotionCompensationMode::from(MotionSearch::default()),
        ..NlmeansOptions::default()
    };
    let hq = NlmeansHqOptions {
        nlm: compensated,
        ..NlmeansHqOptions::default()
    };
    let options = PlaneOptions {
        algorithm: Algorithm::NlmeansHq(hq),
        ..test_plane_options(2)
    };

    for (name, frames) in clips() {
        let mut denoiser = PlanarDenoiser::create(&options, pan_layout()).unwrap();
        let streamed = stream(&mut denoiser, &frames);
        assert_eq!(streamed.len(), FRAMES, "{name}");

        for target in [FRAMES / 2, FRAMES / 3, 2 * FRAMES / 3] {
            let span = denoiser.window_span();
            let window = window_of_span(&frames, target, span);
            let got = denoiser.reseed(&window).unwrap();

            let label = format!("{name} k={target}");
            assert_same_frame(&got, &streamed[target], &label);
        }
    }
}

/// One plane of the parity harness's moving clip, a 1.5 px/frame sine pan plus hashed noise.
fn parity_plane(width: u32, height: u32, frame: usize, channel: u32) -> Vec<u8> {
    let shift = frame as f32 * 1.5;
    let mut samples = Vec::with_capacity((width * height) as usize);
    for row in 0..height {
        for column in 0..width {
            let phase_x = (column as f32 + shift) / width as f32 * std::f32::consts::TAU * 3.0;
            let phase_y = row as f32 / height as f32 * std::f32::consts::TAU * 2.0;
            let texture = 0.5 + 0.2 * phase_x.sin() * phase_y.cos();

            let mut hash = column.wrapping_mul(0x9E37_79B9) ^ row.wrapping_mul(0x85EB_CA6B);
            hash ^= (frame as u32).wrapping_mul(0xC2B2_AE35) ^ channel.wrapping_mul(0x27D4_EB2F);
            hash ^= hash >> 15;
            hash = hash.wrapping_mul(0x2C1B_3C6D);
            hash ^= hash >> 12;
            let noise = (hash as f32 / u32::MAX as f32 - 0.5) * 0.03;

            let value = (texture + noise).clamp(0.0, 1.0);
            let code = (value * 255.0).round() as u8;
            samples.push(code);
        }
    }

    samples
}

/// The parity harness's `nl4d_reseed_8` stream and mid-clip reseed, where the reseed must reproduce
/// the streamed frame.
#[test]
fn nl4d_mid_clip_reseed_matches_streaming_on_the_parity_clip() {
    let layout = FrameLayout {
        width: 160,
        height: 128,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Eight,
    };
    let options = PlaneOptions {
        algorithm: Algorithm::Nl4d(Nl4dOptions::default()),
        ..test_plane_options(2)
    };
    let mut denoiser = PlanarDenoiser::create(&options, layout).unwrap();
    let span = denoiser.window_span();
    let frame_count = span.frame_count();
    let clip_len = 12.max(frame_count + 2);
    let (chroma_width, chroma_height) = layout.chroma_dims();

    let clip: Vec<Planes> = (0..clip_len)
        .map(|frame| Planes {
            y: parity_plane(layout.width, layout.height, frame, 0),
            u: parity_plane(chroma_width, chroma_height, frame, 1),
            v: parity_plane(chroma_width, chroma_height, frame, 2),
        })
        .collect();
    let streamed = stream(&mut denoiser, &clip);
    assert_eq!(streamed.len(), clip_len);

    let window = &clip[1..1 + frame_count];
    let got = denoiser.reseed(window).unwrap();
    let target = 1 + span.behind;

    assert_same_frame(&got, &streamed[target], "parity clip mid-clip reseed");
}

mod convert;
mod coordinator;
mod decode;
mod dispatch;
mod grain_table;
mod run;
mod scenes;
mod source;
mod stage;
mod worker;

#[cfg(feature = "vulkan")]
use std::sync::{Arc, Mutex};

use av_denoise::frame::fill_plane;
use av_denoise::{Depth, FrameLayout, Planes, Subsampling};

/// A writer the test can read back after the coordinator thread drops it.
#[cfg(feature = "vulkan")]
#[derive(Clone, Default)]
pub(super) struct SharedBuffer(pub(super) Arc<Mutex<Vec<u8>>>);

#[cfg(feature = "vulkan")]
impl std::io::Write for SharedBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("buffer lock").extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub fn tiny_layout() -> FrameLayout {
    // 4:2:0 chroma at this size is 4x4, clearing the denoiser's 3x3
    // minimum frame dimension.
    FrameLayout {
        width: 8,
        height: 8,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Eight,
    }
}

pub fn tiny_planes(layout: FrameLayout) -> Planes {
    let luma_pixels = layout.luma_pixels();
    let neutral = layout.depth.neutral_chroma();

    Planes {
        y: fill_plane(luma_pixels, neutral, layout.depth),
        u: layout.neutral_chroma_plane(),
        v: layout.neutral_chroma_plane(),
    }
}

/// A 4x4 8-bit 4:2:0 y4m clip holding `frames` flat frames.
pub fn y4m_clip(frames: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut encoder = y4m::encode(4, 4, y4m::Ratio::new(25, 1))
        .with_colorspace(y4m::Colorspace::C420)
        .write_header(&mut bytes)
        .expect("header should write");
    let luma = vec![16u8; 16];
    let chroma = vec![128u8; 4];

    for _ in 0..frames {
        let frame = y4m::Frame::new([&luma, &chroma, &chroma], None);
        encoder.write_frame(&frame).expect("frame should write");
    }

    bytes
}

/// Frames per scene in [multi_scene_clip].
pub const SCENE_LENGTH: usize = 10;

/// Width and height of [multi_scene_clip].
pub const SCENE_CLIP_SIZE: usize = 64;

/// A tiny xorshift so the clip is the same on every run.
pub fn pattern(seed: u32, len: usize) -> Vec<u8> {
    let mut state = seed.max(1);
    let mut out = Vec::with_capacity(len);

    for _ in 0..len {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        out.push((state & 0xff) as u8);
    }

    out
}

/// A 64x64 8-bit 4:2:0 clip of `frames` frames, with `XCOLORRANGE=LIMITED` in the header.
///
/// The luma switches to a new textured pattern every [SCENE_LENGTH] frames, so each switch is a
/// hard cut.
pub fn multi_scene_clip(frames: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    let range =
        y4m::VendorExtensionString::new(b"COLORRANGE=LIMITED".to_vec()).expect("the extension has no spaces");
    let mut encoder = y4m::encode(SCENE_CLIP_SIZE, SCENE_CLIP_SIZE, y4m::Ratio::new(25, 1))
        .with_colorspace(y4m::Colorspace::C420)
        .append_vendor_extension(range)
        .write_header(&mut bytes)
        .expect("header should write");
    let luma_len = SCENE_CLIP_SIZE * SCENE_CLIP_SIZE;
    let chroma = vec![128u8; luma_len / 4];

    for index in 0..frames {
        let scene = index / SCENE_LENGTH;
        let offset = index % SCENE_LENGTH;
        let base = pattern(scene as u32 + 1, luma_len);
        let luma: Vec<u8> = base
            .iter()
            .map(|&sample| sample.saturating_add(offset as u8))
            .collect();
        let frame = y4m::Frame::new([&luma, &chroma, &chroma], None);

        encoder.write_frame(&frame).expect("frame should write");
    }

    bytes
}

/// Frames per scene in [grainy_clip].
#[cfg(feature = "vulkan")]
pub const GRAINY_SCENE_LENGTH: usize = 20;

/// A 192x128 8-bit 4:2:0 clip of `frames` frames, made of three flat luma bands under fresh grain
/// every frame.
///
/// The bands swap sides every [GRAINY_SCENE_LENGTH] frames, so each swap is a hard cut.
#[cfg(feature = "vulkan")]
pub fn grainy_clip(frames: usize) -> Vec<u8> {
    let width = 192;
    let height = 128;
    let mut bytes = Vec::new();
    let mut encoder = y4m::encode(width, height, y4m::Ratio::new(25, 1))
        .with_colorspace(y4m::Colorspace::C420)
        .write_header(&mut bytes)
        .expect("header should write");
    let chroma = vec![128u8; width * height / 4];

    for index in 0..frames {
        let scene = index / GRAINY_SCENE_LENGTH;
        let levels: [u8; 3] = if scene.is_multiple_of(2) {
            [60, 120, 180]
        } else {
            [180, 120, 60]
        };
        let seed = index as u32 * 7919 + 12345;
        let grain = pattern(seed, width * height);

        let luma: Vec<u8> = grain
            .iter()
            .enumerate()
            .map(|(pixel, &sample)| {
                let band = (pixel % width) * 3 / width;
                let offset = (sample % 9) as i16 - 4;
                (levels[band] as i16 + offset) as u8
            })
            .collect();
        let frame = y4m::Frame::new([&luma, &chroma, &chroma], None);

        encoder.write_frame(&frame).expect("frame should write");
    }

    bytes
}

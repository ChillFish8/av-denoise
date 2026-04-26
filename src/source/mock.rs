use anyhow::{Context, ensure};

use crate::source::{BitDepth, FrameSource, InputSource, VideoFrameBuffer};

pub(crate) struct MockInput {
    width: usize,
    height: usize,
    bit_depth: BitDepth,
    frames: Vec<Vec<u8>>,
}

impl MockInput {
    pub(crate) fn new(
        width: usize,
        height: usize,
        bit_depth: BitDepth,
        frames: Vec<Vec<u8>>,
    ) -> anyhow::Result<Self> {
        let expected_frame_bytes = expected_frame_bytes(width, height, bit_depth)?;

        for (index, frame) in frames.iter().enumerate() {
            ensure!(
                frame.len() == expected_frame_bytes,
                "mock frame {index} expected {expected_frame_bytes} bytes, got {}",
                frame.len()
            );
        }

        Ok(Self {
            width,
            height,
            bit_depth,
            frames,
        })
    }
}

impl InputSource for MockInput {
    type Source = MockFrameSource;

    fn into_frame_source(self) -> anyhow::Result<Self::Source> {
        Ok(MockFrameSource {
            frames: self.frames.into_iter(),
        })
    }

    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn bit_depth(&self) -> BitDepth {
        self.bit_depth
    }
}

pub(crate) struct MockFrameSource {
    frames: std::vec::IntoIter<Vec<u8>>,
}

impl FrameSource for MockFrameSource {
    fn step_next_frame(
        &mut self,
        mut frame: VideoFrameBuffer<'_>,
    ) -> anyhow::Result<bool> {
        let Some(next) = self.frames.next() else {
            return Ok(false);
        };

        ensure!(
            frame.len() == next.len(),
            "mock frame buffer expected {} bytes, got {}",
            next.len(),
            frame.len()
        );
        frame.copy_from_yuv(&next);

        Ok(true)
    }
}

fn expected_frame_bytes(
    width: usize,
    height: usize,
    bit_depth: BitDepth,
) -> anyhow::Result<usize> {
    let bytes_per_sample = bit_depth.bytes_per_sample();
    let luma_bytes = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(bytes_per_sample))
        .context("calculate mock luma bytes")?;
    let chroma_bytes = (width / 2)
        .checked_mul(height / 2)
        .and_then(|n| n.checked_mul(bytes_per_sample))
        .context("calculate mock chroma bytes")?;

    chroma_bytes
        .checked_mul(2)
        .and_then(|n| luma_bytes.checked_add(n))
        .context("calculate mock frame bytes")
}

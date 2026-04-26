use std::io::{BufReader, Read, StdinLock};

use anyhow::{Context, bail};
#[cfg(target_os = "linux")]
use rustix::pipe::{fcntl_getpipe_size, fcntl_setpipe_size};

use crate::source::{BitDepth, VideoFrameBuffer};

const STDIN_BUFFER_CAPACITY: usize = 60 << 20;
#[cfg(target_os = "linux")]
const PIPE_SIZE_ATTEMPTS: &[usize] = &[60 << 20, 16 << 20, 4 << 20, 1 << 20];

/// Get the video frame data provided via STDIN as sequential YUV420 frames.
///
/// # WARNING!
/// This source locks the STDIN source for the _entirety_ of its lifetime!
pub struct StdInInput {
    width: usize,
    height: usize,
    bit_depth: BitDepth,
}

impl StdInInput {
    pub fn new(width: usize, height: usize, hdr: bool) -> Self {
        let bit_depth = if hdr { BitDepth::Ten } else { BitDepth::Eight };
        Self {
            width,
            height,
            bit_depth,
        }
    }
}

impl super::InputSource for StdInInput {
    type Source = StdInFrameSource;

    fn into_frame_source(self) -> anyhow::Result<Self::Source> {
        let stdin = std::io::stdin().lock();
        best_effort_increase_stdin_pipe_size(&stdin);

        let source = BufferedSource::new(stdin, self.width, self.height, self.bit_depth)?;

        Ok(StdInFrameSource { source })
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

pub struct StdInFrameSource {
    source: BufferedSource<StdinLock<'static>>,
}

impl super::FrameSource for StdInFrameSource {
    fn step_next_frame(&mut self, frame: VideoFrameBuffer<'_>) -> anyhow::Result<bool> {
        self.source.step_next_frame(frame)
    }
}

struct BufferedSource<R: Read> {
    luma_bytes: usize,
    chroma_bytes: usize,
    reader: BufReader<R>,
}

impl<R: Read> BufferedSource<R> {
    fn new(
        reader: R,
        width: usize,
        height: usize,
        bit_depth: BitDepth,
    ) -> anyhow::Result<Self> {
        if width == 0 || height == 0 {
            bail!("frame width and height must both be non-zero")
        }
        if width % 2 != 0 || height % 2 != 0 {
            bail!("YUV420 requires even frame width and height")
        }

        let bps = bit_depth.bytes_per_sample();
        let luma_bytes = width
            .checked_mul(height)
            .and_then(|s| s.checked_mul(bps))
            .context("frame dimensions overflowed the luma plane byte size")?;
        let chroma_bytes = (width / 2)
            .checked_mul(height / 2)
            .and_then(|s| s.checked_mul(bps))
            .context("frame dimensions overflowed the chroma plane byte size")?;

        Ok(Self {
            luma_bytes,
            chroma_bytes,
            reader: BufReader::with_capacity(STDIN_BUFFER_CAPACITY, reader),
        })
    }

    fn step_next_frame(
        &mut self,
        mut frame: VideoFrameBuffer<'_>,
    ) -> anyhow::Result<bool> {
        self.validate_frame_buffer(&frame)?;

        let ok = read_exact_or_eof(
            &mut self.reader,
            frame_plane_mut(&mut frame, 0),
            true,
            "Y",
        )?;
        if !ok {
            return Ok(false);
        }
        read_exact_or_eof(&mut self.reader, frame_plane_mut(&mut frame, 1), false, "U")?;
        read_exact_or_eof(&mut self.reader, frame_plane_mut(&mut frame, 2), false, "V")?;

        Ok(true)
    }

    fn validate_frame_buffer(&self, frame: &VideoFrameBuffer<'_>) -> anyhow::Result<()> {
        let frame_bytes = self
            .chroma_bytes
            .checked_mul(2)
            .and_then(|c| self.luma_bytes.checked_add(c))
            .context("frame dimensions overflowed the YUV frame size")?;

        if frame.luma_stride != self.luma_bytes {
            bail!(
                "stdin source expected luma stride of {} bytes, got {}",
                self.luma_bytes,
                frame.luma_stride
            )
        }

        if frame.chroma_stride != self.chroma_bytes {
            bail!(
                "stdin source expected chroma stride of {} bytes, got {}",
                self.chroma_bytes,
                frame.chroma_stride
            )
        }

        if frame.inner.len() != frame_bytes {
            bail!(
                "stdin source expected frame buffer of {frame_bytes} bytes, got {}",
                frame.inner.len()
            )
        }

        Ok(())
    }
}

fn frame_plane_mut<'a>(
    frame: &'a mut VideoFrameBuffer<'_>,
    plane_idx: usize,
) -> &'a mut [u8] {
    let (start, len) = match plane_idx {
        0 => (0, frame.luma_stride),
        1 => (frame.luma_stride, frame.chroma_stride),
        2 => (frame.luma_stride + frame.chroma_stride, frame.chroma_stride),
        _ => unreachable!("invalid plane index"),
    };
    &mut frame.inner[start..start + len]
}

fn read_exact_or_eof<R: Read>(
    reader: &mut R,
    buf: &mut [u8],
    allow_clean_eof: bool,
    plane: &'static str,
) -> anyhow::Result<bool> {
    let mut read_len = 0;

    while read_len < buf.len() {
        match reader.read(&mut buf[read_len..]) {
            Ok(0) if read_len == 0 && allow_clean_eof => return Ok(false),
            Ok(0) => {
                bail!(
                    "truncated input while reading {plane} plane: read {read_len} of {} bytes",
                    buf.len()
                )
            },
            Ok(bytes) => read_len += bytes,
            Err(err) => {
                return Err(err)
                    .with_context(|| format!("failed to read {plane} plane from stdin"));
            },
        }
    }

    Ok(true)
}

#[cfg(target_os = "linux")]
fn best_effort_increase_stdin_pipe_size(stdin: &StdinLock<'_>) {
    let Ok(current_size) = fcntl_getpipe_size(stdin) else {
        tracing::debug!("stdin is not a pipe; skipping pipe buffer resize");
        return;
    };

    for &target_size in PIPE_SIZE_ATTEMPTS {
        if target_size <= current_size {
            tracing::debug!(current_size, "stdin pipe buffer already meets target size");
            return;
        }

        match fcntl_setpipe_size(stdin, target_size) {
            Ok(updated_size) => {
                tracing::debug!(
                    current_size,
                    target_size,
                    updated_size,
                    "resized stdin pipe buffer"
                );
                return;
            },
            Err(err) => {
                tracing::debug!(current_size, target_size, error = %err, "failed to resize stdin pipe buffer");
            },
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn best_effort_increase_stdin_pipe_size(_stdin: &StdinLock<'_>) {}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn make_frame_buffer(
        buffer: &mut [u8],
        luma_stride: usize,
        chroma_stride: usize,
    ) -> VideoFrameBuffer<'_> {
        VideoFrameBuffer {
            inner: buffer,
            luma_stride,
            chroma_stride,
        }
    }

    fn split_frame(
        buffer: &[u8],
        luma_bytes: usize,
        chroma_bytes: usize,
    ) -> (&[u8], &[u8], &[u8]) {
        let (y, uv) = buffer.split_at(luma_bytes);
        let (u, v) = uv.split_at(chroma_bytes);
        (y, u, v)
    }

    fn source_from_bytes(
        width: usize,
        height: usize,
        bit_depth: BitDepth,
        bytes: Vec<u8>,
    ) -> BufferedSource<Cursor<Vec<u8>>> {
        BufferedSource::new(Cursor::new(bytes), width, height, bit_depth)
            .expect("source should build")
    }

    // width=2, height=2, 8-bit YUV420: luma=4 bytes, chroma=1 byte each
    // Frame layout: Y(4), U(1), V(1)
    #[test]
    fn reads_frame() {
        let mut source =
            source_from_bytes(2, 2, BitDepth::Eight, vec![1, 2, 3, 4, 9, 11]);

        let luma_bytes = 4;
        let chroma_bytes = 1;
        let mut frame = vec![0u8; luma_bytes + chroma_bytes * 2];

        assert!(
            source
                .step_next_frame(make_frame_buffer(&mut frame, luma_bytes, chroma_bytes))
                .expect("frame should parse")
        );

        let (y, u, v) = split_frame(&frame, luma_bytes, chroma_bytes);
        assert_eq!(y, &[1, 2, 3, 4]);
        assert_eq!(u, &[9]);
        assert_eq!(v, &[11]);
    }

    #[test]
    fn returns_false_for_clean_end_of_stream() {
        let mut source =
            source_from_bytes(2, 2, BitDepth::Eight, vec![1, 2, 3, 4, 9, 11]);

        let luma_bytes = 4;
        let chroma_bytes = 1;
        let mut frame = vec![0u8; luma_bytes + chroma_bytes * 2];

        assert!(
            source
                .step_next_frame(make_frame_buffer(&mut frame, luma_bytes, chroma_bytes))
                .expect("first frame should parse")
        );
        assert!(
            !source
                .step_next_frame(make_frame_buffer(&mut frame, luma_bytes, chroma_bytes))
                .expect("second step should return eof")
        );
    }

    #[test]
    fn errors_on_partial_frame_input() {
        // 6 bytes needed for one 2x2 8-bit YUV420 frame; provide 5
        let mut source = source_from_bytes(2, 2, BitDepth::Eight, vec![1, 2, 3, 4, 9]);

        let luma_bytes = 4;
        let chroma_bytes = 1;
        let err = source
            .step_next_frame(make_frame_buffer(
                &mut vec![0u8; luma_bytes + chroma_bytes * 2],
                luma_bytes,
                chroma_bytes,
            ))
            .expect_err("truncated frame should error");
        assert!(err.to_string().contains("truncated input"));
    }

    #[test]
    fn reuses_allocated_frame_buffers_across_frames() {
        let mut source = source_from_bytes(
            2,
            2,
            BitDepth::Eight,
            vec![
                1, 2, 3, 4, 9, 11, // frame 1
                5, 6, 7, 8, 10, 12, // frame 2
            ],
        );

        let luma_bytes = 4;
        let chroma_bytes = 1;
        let mut frame = vec![0u8; luma_bytes + chroma_bytes * 2];

        assert!(
            source
                .step_next_frame(make_frame_buffer(&mut frame, luma_bytes, chroma_bytes))
                .expect("first frame should parse")
        );
        let first_y_ptr = frame.as_ptr();
        assert_eq!(&frame[..luma_bytes], &[1, 2, 3, 4]);

        assert!(
            source
                .step_next_frame(make_frame_buffer(&mut frame, luma_bytes, chroma_bytes))
                .expect("second frame should parse")
        );
        let (y, u, v) = split_frame(&frame, luma_bytes, chroma_bytes);
        assert_eq!(y.as_ptr(), first_y_ptr);
        assert_eq!(y, &[5, 6, 7, 8]);
        assert_eq!(u, &[10]);
        assert_eq!(v, &[12]);
    }

    // width=2, height=2, 10-bit YUV420: luma=8 bytes, chroma=2 bytes each
    #[test]
    fn reads_10bit_frame() {
        let mut source = source_from_bytes(
            2,
            2,
            BitDepth::Ten,
            vec![1, 0, 2, 0, 3, 0, 4, 0, 9, 0, 11, 0],
        );

        let luma_bytes = 8;
        let chroma_bytes = 2;
        let mut frame = vec![0u8; luma_bytes + chroma_bytes * 2];

        assert!(
            source
                .step_next_frame(make_frame_buffer(&mut frame, luma_bytes, chroma_bytes))
                .expect("frame should parse")
        );

        let (y, u, v) = split_frame(&frame, luma_bytes, chroma_bytes);
        assert_eq!(y, &[1, 0, 2, 0, 3, 0, 4, 0]);
        assert_eq!(u, &[9, 0]);
        assert_eq!(v, &[11, 0]);
    }
}

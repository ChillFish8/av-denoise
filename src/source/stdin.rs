use std::io::{BufReader, Read, StdinLock};

use anyhow::{Context, bail};
#[cfg(target_os = "linux")]
use rustix::pipe::{fcntl_getpipe_size, fcntl_setpipe_size};

use crate::source::{BitDepth, VideoFrameBuffer};

const STDIN_BUFFER_CAPACITY: usize = 60 << 20;
#[cfg(target_os = "linux")]
const PIPE_SIZE_ATTEMPTS: &[usize] = &[60 << 20, 16 << 20, 4 << 20, 1 << 20];

/// Get the video frame data provided via STDIN as sequential packed RGB frames.
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
    frame_bytes: usize,
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

        let frame_bytes = frame_bytes(width, height, bit_depth)?;

        Ok(Self {
            frame_bytes,
            reader: BufReader::with_capacity(STDIN_BUFFER_CAPACITY, reader),
        })
    }

    fn step_next_frame(
        &mut self,
        mut frame: VideoFrameBuffer<'_>,
    ) -> anyhow::Result<bool> {
        self.validate_frame_buffer(&frame)?;

        read_exact_or_eof(&mut self.reader, frame.as_rgb(), true, "packed RGB frame")
    }

    fn validate_frame_buffer(&self, frame: &VideoFrameBuffer<'_>) -> anyhow::Result<()> {
        if frame.len() != self.frame_bytes {
            bail!(
                "stdin source expected frame buffer of {} bytes, got {}",
                self.frame_bytes,
                frame.len()
            )
        }

        Ok(())
    }
}

fn frame_bytes(
    width: usize,
    height: usize,
    bit_depth: BitDepth,
) -> anyhow::Result<usize> {
    width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .and_then(|n| n.checked_mul(bit_depth.bytes_per_sample()))
        .context("frame dimensions overflowed the RGB frame byte size")
}

fn read_exact_or_eof<R: Read>(
    reader: &mut R,
    buf: &mut [u8],
    allow_clean_eof: bool,
    section: &'static str,
) -> anyhow::Result<bool> {
    let mut read_len = 0;

    while read_len < buf.len() {
        match reader.read(&mut buf[read_len..]) {
            Ok(0) if read_len == 0 && allow_clean_eof => return Ok(false),
            Ok(0) => {
                bail!(
                    "truncated input while reading {section}: read {read_len} of {} bytes",
                    buf.len()
                )
            },
            Ok(bytes) => read_len += bytes,
            Err(err) => {
                return Err(err)
                    .with_context(|| format!("failed to read {section} from stdin"));
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

    fn make_frame_buffer(buffer: &mut [u8]) -> VideoFrameBuffer<'_> {
        VideoFrameBuffer::new(buffer)
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

    #[test]
    fn reads_rgb24_frame() {
        let mut source = source_from_bytes(2, 1, BitDepth::Eight, vec![1, 2, 3, 4, 5, 6]);
        let mut frame = vec![0u8; 6];

        assert!(
            source
                .step_next_frame(make_frame_buffer(&mut frame))
                .expect("frame should parse")
        );

        assert_eq!(frame, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn reads_rgb48_frame() {
        let mut source = source_from_bytes(
            2,
            1,
            BitDepth::Ten,
            vec![1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0],
        );
        let mut frame = vec![0u8; 12];

        assert!(
            source
                .step_next_frame(make_frame_buffer(&mut frame))
                .expect("frame should parse")
        );

        assert_eq!(frame, vec![1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]);
    }

    #[test]
    fn returns_false_for_clean_end_of_stream() {
        let mut source = source_from_bytes(2, 1, BitDepth::Eight, vec![1, 2, 3, 4, 5, 6]);
        let mut frame = vec![0u8; 6];

        assert!(
            source
                .step_next_frame(make_frame_buffer(&mut frame))
                .expect("first frame should parse")
        );
        assert!(
            !source
                .step_next_frame(make_frame_buffer(&mut frame))
                .expect("second step should return eof")
        );
    }

    #[test]
    fn errors_on_partial_frame_input() {
        let mut source = source_from_bytes(2, 1, BitDepth::Eight, vec![1, 2, 3, 4, 5]);
        let err = source
            .step_next_frame(make_frame_buffer(&mut vec![0u8; 6]))
            .expect_err("truncated frame should error");
        assert!(err.to_string().contains("truncated input"));
    }

    #[test]
    fn rejects_wrong_sized_frame_buffer() {
        let mut source = source_from_bytes(2, 1, BitDepth::Eight, vec![1, 2, 3, 4, 5, 6]);
        let err = source
            .step_next_frame(make_frame_buffer(&mut vec![0u8; 5]))
            .expect_err("wrong-sized frame buffer should error");
        assert!(err.to_string().contains("expected frame buffer"));
    }
}

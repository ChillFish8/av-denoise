#[cfg(test)]
pub(crate) mod mock;
pub mod stdin;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
/// The bit depth of the video frame.
pub enum BitDepth {
    /// Packed RGB24.
    Eight,
    /// Packed RGB48LE with 10-bit samples stored in `u16` lanes.
    Ten,
}

impl BitDepth {
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::Eight => 1,
            Self::Ten => 2,
        }
    }
}

/// An input source provides [FrameSource]s.
pub trait InputSource: Send + 'static {
    /// The frame source being produced by the input source.
    type Source: FrameSource;

    /// Consume the input source producing a frame source.
    fn into_frame_source(self) -> anyhow::Result<Self::Source>;

    /// The width of the input source in pixels.
    fn width(&self) -> usize;

    /// The height of the input source in pixels.
    fn height(&self) -> usize;

    /// The bit depth of the video.
    fn bit_depth(&self) -> BitDepth;
}

/// An input source providing video frames one at a time.
///
/// Input frames are expected to be provided in packed **RGB24** or **RGB48LE**
/// formats, no other format will be handled correctly.
pub trait FrameSource {
    /// Read and parse the next video frame.
    ///
    /// Returns `true` if a new frame is ready or `false` for end-of-stream.
    fn step_next_frame(&mut self, frame: VideoFrameBuffer<'_>) -> anyhow::Result<bool>;
}

/// A slice of memory which contains a packed RGB video frame.
pub struct VideoFrameBuffer<'a> {
    inner: &'a mut [u8],
}

impl<'a> VideoFrameBuffer<'a> {
    /// Create a new [VideoFrameBuffer].
    pub(crate) fn new(inner: &'a mut [u8]) -> Self {
        Self { inner }
    }

    /// Return a reference to the inner packed RGB slice buffer.
    pub fn as_rgb(&mut self) -> &mut [u8] {
        self.inner
    }

    /// Return the total size of the packed RGB frame in bytes.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Copy a packed RGB frame into this buffer.
    pub fn copy_from_rgb(&mut self, src: &[u8]) {
        self.inner.copy_from_slice(src);
    }
}

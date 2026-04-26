#[cfg(test)]
pub(crate) mod mock;
pub mod stdin;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
/// The bit depth of the video frame.
pub enum BitDepth {
    /// SDR 8-bit
    Eight,
    /// HDR 10-bit
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
/// Input frames are expected to be provided in the **YUV420p** or
/// **YUV420p10le** formats, no other format will be handled correctly.
/// Any pixel format conversion should be performed before passing it in.
pub trait FrameSource {
    /// Read and parse the next video frame.
    ///
    /// Returns `true` if a new frame is ready or `false` for end-of-stream.
    fn step_next_frame(&mut self, frame: VideoFrameBuffer<'_>) -> anyhow::Result<bool>;
}

/// A slice of memory which contains a YUV420 video frame.
pub struct VideoFrameBuffer<'a> {
    inner: &'a mut [u8],
    luma_stride: usize,
    chroma_stride: usize,
}

impl<'a> VideoFrameBuffer<'a> {
    /// Create a new [VideoFrameBuffer].
    pub(crate) fn new(
        inner: &'a mut [u8],
        luma_stride: usize,
        chroma_stride: usize,
    ) -> Self {
        Self {
            inner,
            luma_stride,
            chroma_stride,
        }
    }

    /// Return a reference to the inner YUV slice buffer.
    pub fn as_yuv(&'a mut self) -> &'a mut [u8] {
        self.inner
    }

    /// Return the total size of the packed YUV frame in bytes.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Copy a packed YUV frame into this buffer.
    pub fn copy_from_yuv(&mut self, src: &[u8]) {
        self.inner.copy_from_slice(src);
    }

    /// Return a reference to the inner Y slice buffer.
    pub fn as_y(&'a mut self) -> &'a mut [u8] {
        &mut self.inner[0..][..self.luma_stride]
    }

    /// Return a reference to the inner U slice buffer.
    pub fn as_u(&'a mut self) -> &'a mut [u8] {
        &mut self.inner[self.luma_stride..][..self.chroma_stride]
    }

    /// Return a reference to the inner V slice buffer.
    pub fn as_v(&'a mut self) -> &'a mut [u8] {
        &mut self.inner[self.luma_stride + self.chroma_stride..][..self.chroma_stride]
    }
}

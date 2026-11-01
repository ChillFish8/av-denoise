use super::{DevicePlane, SampleFormat};
use crate::error::Error;

/// How a frame stores its colour planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChromaLayout {
    /// Luma alone, with no colour planes.
    Gray,
    /// Chroma at half width and half height.
    Yuv420,
    /// Chroma at half width and full height.
    Yuv422,
    /// Chroma at full size.
    Yuv444,
}

impl ChromaLayout {
    /// The planes a frame holds, 1 for [ChromaLayout::Gray] and 3 otherwise.
    pub fn plane_count(self) -> usize {
        match self {
            ChromaLayout::Gray => 1,
            ChromaLayout::Yuv420 | ChromaLayout::Yuv422 | ChromaLayout::Yuv444 => 3,
        }
    }

    /// The size of each chroma plane in a `width` by `height` frame, or `None` without chroma.
    ///
    /// Halved axes round up, so an odd dimension keeps the extra sample.
    pub fn chroma_dims(self, width: u32, height: u32) -> Option<(u32, u32)> {
        match self {
            ChromaLayout::Gray => None,
            ChromaLayout::Yuv420 => Some((width.div_ceil(2), height.div_ceil(2))),
            ChromaLayout::Yuv422 => Some((width.div_ceil(2), height)),
            ChromaLayout::Yuv444 => Some((width, height)),
        }
    }
}

/// The frame an engine is built for.
///
/// `width` and `height` are the luma plane's dimensions. Each chroma plane's size follows from
/// `chroma`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    pub width: u32,
    pub height: u32,
    pub chroma: ChromaLayout,
    pub input: SampleFormat,
    pub output: SampleFormat,
}

impl Geometry {
    /// Rejects sample formats the engines cannot read or write.
    pub fn validate(&self) -> Result<(), Error> {
        self.input.validate()?;
        self.output.validate()?;
        Ok(())
    }

    /// The width and height of plane `index`, where plane 0 is luma.
    pub(crate) fn plane_dims(&self, index: usize) -> (u32, u32) {
        if index == 0 {
            return (self.width, self.height);
        }

        let chroma = self.chroma.chroma_dims(self.width, self.height);
        chroma.unwrap_or((self.width, self.height))
    }

    /// Checks there is one plane per frame plane, each at its own size and covering its whole words.
    pub fn check_planes(&self, planes: &[DevicePlane<'_>], format: SampleFormat) -> Result<(), Error> {
        let expected = self.chroma.plane_count();
        if planes.len() != expected {
            let message = format!("expected {expected} planes, got {}", planes.len());
            return Err(Error::PlaneMismatch(message));
        }

        for (index, plane) in planes.iter().enumerate() {
            let (width, height) = self.plane_dims(index);
            let matches_size = plane.width() == width && plane.height() == height;
            if !matches_size {
                let message = format!(
                    "plane {index} is {}x{}, expected {width}x{height}",
                    plane.width(),
                    plane.height(),
                );
                return Err(Error::PlaneMismatch(message));
            }

            let pixels = u64::from(width) * u64::from(height);
            let needed = format.plane_bytes(pixels);
            let available = plane.handle().size_in_used();
            if available < needed {
                let message = format!("plane {index} handle holds {available} bytes, needs {needed}");
                return Err(Error::PlaneMismatch(message));
            }
        }

        Ok(())
    }
}

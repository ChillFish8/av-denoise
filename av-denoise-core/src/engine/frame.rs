use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{DevicePlane, SampleFormat};
use crate::error::Error;
use crate::motion::Subsampling;

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

    /// The chroma subsampling as a power of two per axis.
    pub(crate) fn subsampling(self) -> Subsampling {
        let (shift_x, shift_y) = match self {
            ChromaLayout::Yuv420 => (1, 1),
            ChromaLayout::Yuv422 => (1, 0),
            ChromaLayout::Gray | ChromaLayout::Yuv444 => (0, 0),
        };

        Subsampling { shift_x, shift_y }
    }
}

/// Where the chroma samples of a 4:2:0 frame sit relative to its luma samples.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ChromaSiting {
    /// Level with the left luma column, halfway between two rows.
    Left,
    /// Halfway between two luma columns and two luma rows.
    Center,
    /// Level with the top left luma sample.
    TopLeft,
    /// The source did not say.
    #[default]
    Unknown,
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
    /// Where the chroma samples sit. It is carried for callers and never changes the output.
    pub chroma_siting: ChromaSiting,
}

impl Geometry {
    /// Rejects sample formats the engines cannot read or write.
    pub fn validate(&self) -> Result<(), Error> {
        self.input.validate()?;
        self.output.validate()?;
        Ok(())
    }

    /// The width and height of plane `index`, where plane 0 is luma.
    pub fn plane_dims(&self, index: usize) -> (u32, u32) {
        if index == 0 {
            return (self.width, self.height);
        }

        let chroma = self.chroma.chroma_dims(self.width, self.height);
        chroma.unwrap_or((self.width, self.height))
    }

    /// The bytes plane `index` takes in `format`, rounded up to whole words.
    pub(crate) fn plane_bytes(&self, index: usize, format: SampleFormat) -> usize {
        let (width, height) = self.plane_dims(index);
        let pixels = u64::from(width) * u64::from(height);

        format.plane_bytes(pixels) as usize
    }

    /// New handles for every plane of one frame in `format`, whole words each.
    pub(crate) fn empty_planes(&self, client: &Client, format: SampleFormat) -> Vec<Handle> {
        (0..self.chroma.plane_count())
            .map(|index| {
                let bytes = self.plane_bytes(index, format);
                client.empty(bytes)
            })
            .collect()
    }

    /// `handles` as planes `first` onwards, each at its own size.
    pub(crate) fn device_planes<'a>(&self, first: usize, handles: &'a [Handle]) -> Vec<DevicePlane<'a>> {
        handles
            .iter()
            .enumerate()
            .map(|(offset, handle)| {
                let (width, height) = self.plane_dims(first + offset);
                DevicePlane::new(handle, width, height)
            })
            .collect()
    }

    /// Checks there is one plane per frame plane, each at its own size and covering its whole words.
    pub fn check_planes(&self, planes: &[DevicePlane<'_>], format: SampleFormat) -> Result<(), Error> {
        let expected = self.chroma.plane_count();
        if planes.len() != expected {
            let message = format!("expected {expected} planes, got {}", planes.len());
            return Err(Error::PlaneMismatch(message));
        }

        self.check_plane_sizes(planes, 0, format)
    }

    /// Checks each of `planes`, taken as planes `first` onwards, is at its size and covers its whole words.
    pub(crate) fn check_plane_sizes(
        &self,
        planes: &[DevicePlane<'_>],
        first: usize,
        format: SampleFormat,
    ) -> Result<(), Error> {
        for (offset, plane) in planes.iter().enumerate() {
            let index = first + offset;
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

            let needed = self.plane_bytes(index, format) as u64;
            let available = plane.handle().size_in_used();
            if available < needed {
                let message = format!("plane {index} handle holds {available} bytes, needs {needed}");
                return Err(Error::PlaneMismatch(message));
            }
        }

        Ok(())
    }
}

use cubecl::server::Handle;

use crate::error::Error;
use crate::nlmeans::ChannelMode;

/// How samples are stored in a plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleFormat {
    U8,
    /// 16-bit words holding `depth` significant bits, between 9 and 16.
    U16 {
        depth: u8,
    },
    /// Normalised samples between 0 and 1.
    F32,
}

impl SampleFormat {
    pub(crate) fn validate(self) -> Result<(), Error> {
        match self {
            SampleFormat::U16 { depth } if !(9..=16).contains(&depth) => {
                let message = format!("U16 depth must be between 9 and 16, got {depth}");
                Err(Error::InvalidGeometry(message))
            },
            _ => Ok(()),
        }
    }

    /// The largest sample value, which normalisation divides by.
    #[cfg_attr(not(test), expect(dead_code, reason = "no engine calls this yet"))]
    pub(crate) fn max_value(self) -> f32 {
        match self {
            SampleFormat::U8 => 255.0,
            SampleFormat::U16 { depth } => ((1u32 << depth) - 1) as f32,
            SampleFormat::F32 => 1.0,
        }
    }

    pub(crate) fn samples_per_word(self) -> u32 {
        match self {
            SampleFormat::U8 => 4,
            SampleFormat::U16 { .. } => 2,
            SampleFormat::F32 => 1,
        }
    }

    /// Bytes a plane of `pixels` samples occupies, rounded up to whole words.
    pub(crate) fn plane_bytes(self, pixels: u64) -> u64 {
        let samples_per_word = self.samples_per_word() as u64;
        let words = pixels.div_ceil(samples_per_word);
        words * 4
    }
}

/// One plane of samples on the GPU, tightly packed with a stride equal to its width.
#[derive(Debug, Clone, Copy)]
pub struct DevicePlane<'a> {
    handle: &'a Handle,
    width: u32,
    height: u32,
}

impl<'a> DevicePlane<'a> {
    pub fn new(handle: &'a Handle, width: u32, height: u32) -> Self {
        Self {
            handle,
            width,
            height,
        }
    }

    pub fn handle(&self) -> &'a Handle {
        self.handle
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}

/// The shape and sample formats an engine is built for.
///
/// `width` and `height` are the dimensions of the planes the engine sees, so a chroma engine is built at
/// chroma size. Luma takes one plane, Chroma takes U and V, and Yuv takes Y, U and V at one size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    pub width: u32,
    pub height: u32,
    pub channels: ChannelMode,
    pub input: SampleFormat,
    pub output: SampleFormat,
}

impl Geometry {
    #[expect(dead_code, reason = "no engine calls this yet")]
    pub(crate) fn validate(&self) -> Result<(), Error> {
        self.input.validate()?;
        self.output.validate()?;
        Ok(())
    }

    #[cfg_attr(not(test), expect(dead_code, reason = "no engine calls this yet"))]
    pub(crate) fn pixels(&self) -> u32 {
        self.width * self.height
    }

    /// Checks plane count, dimensions, and that every handle covers its whole words.
    #[cfg_attr(not(test), expect(dead_code, reason = "no engine calls this yet"))]
    pub(crate) fn check_planes(&self, planes: &[DevicePlane<'_>], format: SampleFormat) -> Result<(), Error> {
        let expected = self.channels.count() as usize;
        if planes.len() != expected {
            let message = format!("expected {expected} planes, got {}", planes.len());
            return Err(Error::PlaneMismatch(message));
        }

        let pixels = self.pixels() as u64;
        let needed = format.plane_bytes(pixels);

        for (index, plane) in planes.iter().enumerate() {
            let matches_size = plane.width == self.width && plane.height == self.height;
            if !matches_size {
                let message = format!(
                    "plane {index} is {}x{}, expected {}x{}",
                    plane.width, plane.height, self.width, self.height
                );
                return Err(Error::PlaneMismatch(message));
            }

            let available = plane.handle.size_in_used();
            if available < needed {
                let message = format!("plane {index} handle holds {available} bytes, needs {needed}");
                return Err(Error::PlaneMismatch(message));
            }
        }

        Ok(())
    }
}

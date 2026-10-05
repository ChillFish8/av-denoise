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
    pub fn plane_bytes(self, pixels: u64) -> u64 {
        let samples_per_word = self.samples_per_word() as u64;
        let words = pixels.div_ceil(samples_per_word);
        words * 4
    }
}

/// One plane of samples on the GPU, tightly packed with a stride equal to its width.
///
/// The handle must hold whole 4-byte words, which is `SampleFormat::plane_bytes(width * height)` bytes,
/// and [Engine::emit_into](crate::engine::Engine::emit_into) writes zeros into the padding past the last
/// sample.
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
    pub(crate) fn validate(&self) -> Result<(), Error> {
        self.input.validate()?;
        self.output.validate()?;
        Ok(())
    }

    /// The samples in one plane, in `u64` so no `u32` dimensions overflow it.
    pub(crate) fn pixels(&self) -> u64 {
        let width = u64::from(self.width);
        let height = u64::from(self.height);

        width * height
    }

    /// Rejects a geometry whose ring of `slots` frames stores more than `u32::MAX` elements.
    ///
    /// The kernels index the frame ring and its accumulators with `u32`, so a larger ring would wrap.
    pub(crate) fn check_ring_fits(&self, slots: u64) -> Result<(), Error> {
        let stored_ch = u64::from(self.channels.storage_count());
        let elements = self
            .pixels()
            .checked_mul(stored_ch)
            .and_then(|frame_len| frame_len.checked_mul(slots));
        let fits = elements.is_some_and(|elements| elements <= u64::from(u32::MAX));
        if !fits {
            let message = format!(
                "a {slots} frame ring of {}x{} planes stores more than u32::MAX elements",
                self.width, self.height
            );
            return Err(Error::InvalidGeometry(message));
        }

        Ok(())
    }

    /// Checks plane count, dimensions, and that every handle covers its whole words.
    pub(crate) fn check_planes(&self, planes: &[DevicePlane<'_>], format: SampleFormat) -> Result<(), Error> {
        let expected = self.channels.count() as usize;
        if planes.len() != expected {
            let message = format!("expected {expected} planes, got {}", planes.len());
            return Err(Error::PlaneMismatch(message));
        }

        let pixels = self.pixels();
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

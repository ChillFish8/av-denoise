use av_denoise_core::SampleFormat;

/// Bit depth of a source's samples.
///
/// Normalisation divides by [`Depth::max_value`], so a value in
/// normalised units means the same thing at every depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Depth {
    Eight,
    Ten,
    Twelve,
}

/// Returned when a source declares a bit depth the denoiser does not handle.
#[derive(Debug, thiserror::Error)]
#[error("unsupported bit depth {0}, av-denoise supports 8, 10, and 12-bit")]
pub struct UnsupportedDepthError(pub usize);

impl Depth {
    /// Maps a declared bit depth onto a [`Depth`].
    pub fn from_bits(bits: usize) -> Result<Self, UnsupportedDepthError> {
        match bits {
            8 => Ok(Depth::Eight),
            10 => Ok(Depth::Ten),
            12 => Ok(Depth::Twelve),
            other => Err(UnsupportedDepthError(other)),
        }
    }

    /// Bits per sample.
    pub fn bits(self) -> usize {
        match self {
            Depth::Eight => 8,
            Depth::Ten => 10,
            Depth::Twelve => 12,
        }
    }

    /// Bytes each sample takes up on the wire.
    ///
    /// Depths above 8 use a little-endian 16-bit word.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Depth::Eight => 1,
            Depth::Ten | Depth::Twelve => 2,
        }
    }

    /// The largest sample value this depth can hold, which is also the
    /// normalisation divisor.
    pub fn max_value(self) -> f32 {
        ((1u32 << self.bits()) - 1) as f32
    }

    /// The sample value that means neutral chroma at this depth.
    pub fn neutral_chroma(self) -> u16 {
        1 << (self.bits() - 1)
    }

    /// The plane format an engine reads and writes at this depth.
    pub(crate) fn sample_format(self) -> SampleFormat {
        match self {
            Depth::Eight => SampleFormat::U8,
            Depth::Ten => SampleFormat::U16 { depth: 10 },
            Depth::Twelve => SampleFormat::U16 { depth: 12 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_bits_accepts_supported_depths() {
        assert_eq!(Depth::from_bits(8).unwrap(), Depth::Eight);
        assert_eq!(Depth::from_bits(10).unwrap(), Depth::Ten);
        assert_eq!(Depth::from_bits(12).unwrap(), Depth::Twelve);
    }

    #[test]
    fn from_bits_rejects_unsupported_depths() {
        for bits in [0, 7, 9, 11, 16] {
            let result = Depth::from_bits(bits);
            assert!(result.is_err(), "{bits} bits should be rejected");
        }
    }

    #[test]
    fn depth_properties_match_the_format() {
        assert_eq!(Depth::Eight.bytes_per_sample(), 1);
        assert_eq!(Depth::Ten.bytes_per_sample(), 2);
        assert_eq!(Depth::Twelve.bytes_per_sample(), 2);

        assert_eq!(Depth::Eight.max_value(), 255.0);
        assert_eq!(Depth::Ten.max_value(), 1023.0);
        assert_eq!(Depth::Twelve.max_value(), 4095.0);

        assert_eq!(Depth::Eight.neutral_chroma(), 128);
        assert_eq!(Depth::Ten.neutral_chroma(), 512);
        assert_eq!(Depth::Twelve.neutral_chroma(), 2048);
    }

    #[test]
    fn sample_format_matches_the_depth() {
        assert_eq!(Depth::Eight.sample_format(), SampleFormat::U8);
        assert_eq!(Depth::Ten.sample_format(), SampleFormat::U16 { depth: 10 });
        assert_eq!(Depth::Twelve.sample_format(), SampleFormat::U16 { depth: 12 });
    }
}

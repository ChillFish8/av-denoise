use cubecl::prelude::*;

/// Byte alignment every buffer binding must start on, taken from the
/// runtime the denoiser is running against.
///
/// A GPU rejects a bind group whose buffer offset is not a multiple of
/// its `min_storage_buffer_offset_alignment`. Every buffer this crate
/// slices into per-slot regions therefore pads its slot stride up to
/// this value.
///
/// Each backend reports its own figure, 32 bytes on the Vulkan adapters
/// we test against and up to 256 elsewhere, which is why the value is
/// read from the runtime rather than assumed.
///
/// It is carried as its own type rather than a bare `u64` so it cannot
/// be swapped by mistake with the width, height, or frame-count
/// arguments it travels alongside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StorageAlign(u64);

impl StorageAlign {
    /// The alignment `client`'s runtime requires.
    ///
    /// cubecl aligns every allocation it hands out to this same value,
    /// so a slot offset that is a multiple of it always lands on a
    /// boundary the backend accepts.
    pub(crate) fn from_client<R: Runtime>(client: &ComputeClient<R>) -> Self {
        Self::new(client.properties().memory.alignment)
    }

    /// A fixed alignment, for tests that have no runtime to ask.
    pub(crate) fn new(bytes: u64) -> Self {
        debug_assert!(
            bytes.is_power_of_two(),
            "storage alignment {bytes} is not a power of two"
        );
        Self(bytes.max(1))
    }

    /// `bytes` rounded up to the next aligned boundary.
    pub(crate) fn pad_bytes(self, bytes: u64) -> u64 {
        bytes.next_multiple_of(self.0)
    }

    /// A count of `T` rounded up so that many elements cover a whole
    /// number of alignment boundaries.
    ///
    /// Alignments are powers of two, so for any `T` whose size divides
    /// the alignment this lands exactly on a boundary. For a larger `T`
    /// the elements are already aligned, so the count comes back
    /// unchanged.
    pub(crate) fn pad_elems<T>(self, elems: usize) -> usize {
        let per_boundary = (self.0 as usize).div_ceil(size_of::<T>()).max(1);
        elems.next_multiple_of(per_boundary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_bytes_rounds_up_to_the_boundary() {
        let align = StorageAlign::new(32);
        assert_eq!(align.pad_bytes(0), 0);
        assert_eq!(align.pad_bytes(1), 32);
        assert_eq!(align.pad_bytes(32), 32);
        assert_eq!(align.pad_bytes(33), 64);
    }

    #[test]
    fn pad_elems_spans_whole_boundaries() {
        // 32-byte boundaries hold 8 f32s.
        let align = StorageAlign::new(32);
        assert_eq!(align.pad_elems::<f32>(0), 0);
        assert_eq!(align.pad_elems::<f32>(1), 8);
        assert_eq!(align.pad_elems::<f32>(8), 8);
        assert_eq!(align.pad_elems::<f32>(9), 16);
    }

    #[test]
    fn pad_elems_tracks_a_larger_alignment() {
        // A 256-byte boundary holds 64 f32s, so the same element count
        // pads eight times further than it does at 32 bytes.
        let align = StorageAlign::new(256);
        assert_eq!(align.pad_elems::<f32>(1), 64);
        assert_eq!(align.pad_elems::<f32>(64), 64);
        assert_eq!(align.pad_elems::<f32>(65), 128);
    }

    #[test]
    fn padded_element_counts_are_byte_aligned() {
        for bytes in [4u64, 16, 32, 64, 256] {
            let align = StorageAlign::new(bytes);
            for elems in [1usize, 3, 7, 137, 24_660] {
                let padded = align.pad_elems::<f32>(elems) as u64 * size_of::<f32>() as u64;
                assert_eq!(padded % bytes, 0, "at align {bytes} with {elems} elements");
            }
        }
    }

    #[test]
    fn an_alignment_below_the_element_size_leaves_counts_alone() {
        let align = StorageAlign::new(4);
        assert_eq!(align.pad_elems::<f32>(3), 3);
    }
}

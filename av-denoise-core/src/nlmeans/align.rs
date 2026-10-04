use cubecl::prelude::*;

/// The byte alignment every per-slot buffer binding must start on, read from the runtime.
///
/// A GPU rejects a bind group whose offset is not a multiple of its
/// `min_storage_buffer_offset_alignment`, so per-slot strides pad up to this value. Backends differ,
/// from 32 bytes on the tested Vulkan adapters up to 256 elsewhere, so it is read rather than
/// assumed. It is its own type so it cannot be swapped with the width, height or frame-count
/// arguments it travels with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StorageAlign(u64);

impl StorageAlign {
    /// The alignment `client`'s runtime requires.
    ///
    /// cubecl aligns every allocation to the same value, so a slot offset that is a multiple of it
    /// always lands on a boundary the backend accepts.
    pub(crate) fn from_client<R: Runtime>(client: &ComputeClient<R>) -> Self {
        let alignment = client.properties().memory.alignment;
        Self::new(alignment)
    }

    /// A fixed alignment, for tests that have no runtime to ask.
    pub(crate) fn new(bytes: u64) -> Self {
        debug_assert!(
            bytes.is_power_of_two(),
            "storage alignment {bytes} is not a power of two"
        );
        Self(bytes.max(1))
    }

    pub(crate) fn pad_bytes(self, bytes: u64) -> u64 {
        bytes.next_multiple_of(self.0)
    }

    /// A count of `T` rounded up so the elements cover whole alignment boundaries.
    ///
    /// A `T` larger than the alignment is already aligned, so its count comes back unchanged.
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
        // A 256-byte boundary holds 64 f32s.
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

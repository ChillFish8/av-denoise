//! Converts between VapourSynth's strided plane buffers and the tightly
//! packed rows core's converters accept, plus the frame window core
//! windowed algorithms request around each output frame.

/// Copies `src`, a plane with row stride `stride` bytes, into a new
/// tightly packed buffer of `width_bytes * height` bytes.
///
/// `width_bytes` is `width * bytes_per_sample`, not a pixel count, so
/// this works the same at any bit depth. Passing a pixel count here
/// packs the wrong number of bytes per row.
pub fn pack_plane(src: &[u8], stride: usize, width_bytes: usize, height: usize) -> Vec<u8> {
    let mut packed = Vec::with_capacity(width_bytes * height);
    for row in src.chunks(stride).take(height) {
        packed.extend_from_slice(&row[..width_bytes]);
    }
    packed
}

/// Writes a tightly packed plane, `src`, back into `dst`, a strided
/// buffer with row stride `stride` bytes. The reverse of [`pack_plane`].
///
/// `width_bytes` is `width * bytes_per_sample`, not a pixel count, so
/// this works the same at any bit depth. `dst`'s padding bytes, if any,
/// are left untouched.
pub fn unpack_plane_into(dst: &mut [u8], stride: usize, width_bytes: usize, height: usize, src: &[u8]) {
    for (y, row) in dst.chunks_mut(stride).take(height).enumerate() {
        let packed_row = &src[y * width_bytes..(y + 1) * width_bytes];
        row[..width_bytes].copy_from_slice(packed_row);
    }
}

/// The `behind + 1 + ahead` source frame indices for the window around
/// output frame `n`, `behind` older and `ahead` newer, clamped so
/// nothing runs off either end of a clip whose last valid index is
/// `last_frame`.
///
/// `behind` and `ahead` come from the denoiser's own
/// [`av_denoise_core::PlanarDenoiser::window_span`], so this stays
/// correct for whichever algorithm the denoiser is running rather than
/// assuming every algorithm needs the same symmetric window.
///
/// Frame requests and window builds both call this, so the two always
/// agree on which frames a window at `n` pulls in.
pub fn window_indices(n: usize, behind: usize, ahead: usize, last_frame: usize) -> Vec<usize> {
    (0..=behind + ahead)
        .map(|i| (n + i).saturating_sub(behind).min(last_frame))
        .collect()
}

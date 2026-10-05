use std::ops::RangeInclusive;

use av_denoise::Planes;

/// Copies a plane with a row stride of `stride` bytes into a tightly packed buffer.
///
/// `width_bytes` is `width * bytes_per_sample`, not a pixel count, so this works the same at any bit
/// depth. Passing a pixel count here packs the wrong number of bytes per row.
pub fn pack_plane(src: &[u8], stride: usize, width_bytes: usize, height: usize) -> Vec<u8> {
    let mut packed = Vec::with_capacity(width_bytes * height);
    for row in src.chunks(stride).take(height) {
        packed.extend_from_slice(&row[..width_bytes]);
    }

    packed
}

/// Writes the tightly packed plane `src` back into `dst`, which has a row stride of `stride` bytes.
///
/// The reverse of [pack_plane]. `width_bytes` is `width * bytes_per_sample`, not a pixel count.
/// Padding bytes in `dst` are left untouched.
pub fn unpack_plane_into(dst: &mut [u8], stride: usize, width_bytes: usize, height: usize, src: &[u8]) {
    for (y, row) in dst.chunks_mut(stride).take(height).enumerate() {
        let packed_row = &src[y * width_bytes..(y + 1) * width_bytes];
        row[..width_bytes].copy_from_slice(packed_row);
    }
}

/// The `behind + 1 + ahead` source indices around an output frame, clamped to `0..=last_frame`.
///
/// Under repeated edges, frame requests and window builds both call this, so the two always agree on
/// which frames a window pulls in.
pub fn window_indices(output_index: usize, behind: usize, ahead: usize, last_frame: usize) -> Vec<usize> {
    (0..=behind + ahead)
        .map(|i| (output_index + i).saturating_sub(behind).min(last_frame))
        .collect()
}

/// The source frame range around an output frame under shifted edges.
///
/// The range is cut short at either end of the clip rather than repeating a boundary frame.
pub fn shifted_window_range(
    output_index: usize,
    behind: usize,
    ahead: usize,
    last_frame: usize,
) -> RangeInclusive<usize> {
    let first = output_index.saturating_sub(behind);
    let last = (output_index + ahead).min(last_frame);
    first..=last
}

/// Denoised frames from a clip's final flush, held for the requests that follow.
///
/// Each frame is handed out at most once, and a request for any other index misses.
pub struct TailCache {
    first: usize,
    frames: Vec<Option<Planes>>,
}

impl TailCache {
    pub fn new(first: usize, frames: Vec<Planes>) -> Self {
        let frames = frames.into_iter().map(Some).collect();
        Self { first, frames }
    }

    pub fn take(&mut self, output_index: usize) -> Option<Planes> {
        let index = output_index.checked_sub(self.first)?;
        self.frames.get_mut(index)?.take()
    }
}

mod block_match;
mod blocked_equivalence;
mod merged;
mod paths;
mod reference;

use cubecl::prelude::*;
use cubecl::server::Handle;

/// Clamps `value - delta` into `0..limit`, matching the kernel's clamp-to-edge reads.
pub(crate) fn shift_clamped(value: i32, delta: i32, limit: i32) -> i32 {
    (value - delta).clamp(0, limit - 1)
}

/// Builds a frame that reads `world` shifted by `(shift_x, shift_y)` pixels, clamped to its edges.
pub(crate) fn frame_shifted_by(
    world: &[f32],
    width: u32,
    height: u32,
    shift_x: i32,
    shift_y: i32,
) -> Vec<f32> {
    let mut frame = vec![0.0f32; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let source_x = shift_clamped(x as i32, shift_x, width as i32) as u32;
            let source_y = shift_clamped(y as i32, shift_y, height as i32) as u32;
            frame[(y * width + x) as usize] = world[(source_y * width + source_x) as usize];
        }
    }

    frame
}

/// Uploads `centre` and `neighbour` as frame slots 0 and 1 of one buffer, with a table naming slot 1.
///
/// The block-match kernels read every frame from one buffer, so this is how a test matches two planes.
pub(crate) fn pair_pyramid(client: &Client, centre: &[f32], neighbour: &[f32]) -> (Handle, Handle) {
    let mut frames = centre.to_vec();
    frames.extend_from_slice(neighbour);
    let frame_bytes = f32::as_bytes(&frames);
    let pyramid = client.create_from_slice(frame_bytes);

    let table = [1u32];
    let table_bytes = u32::as_bytes(&table);
    let neighbour_slots = client.create_from_slice(table_bytes);

    (pyramid, neighbour_slots)
}

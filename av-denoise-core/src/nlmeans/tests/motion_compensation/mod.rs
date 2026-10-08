mod block_match;
mod blocked_equivalence;
mod chain;
mod end_to_end;
mod reference;
mod seeding;

/// Clamps `value - delta` into `0..limit`, matching the kernel's clamp-to-edge reads.
fn shift_clamped(value: i32, delta: i32, limit: i32) -> i32 {
    (value - delta).clamp(0, limit - 1)
}

/// Builds a frame that reads `world` shifted by `(shift_x, shift_y)` pixels, clamped to its edges.
fn frame_shifted_by(world: &[f32], width: u32, height: u32, shift_x: i32, shift_y: i32) -> Vec<f32> {
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

use cubecl::prelude::*;

/// Normalises word-packed planes into one interleaved ring slot.
///
/// Each thread handles one pixel and writes all `stored_ch` lanes, with padding lanes set to zero.
/// A plane is only read when `channels` covers it, so an unused plane can be a placeholder.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding or a comptime shape"
)]
pub fn ingest_words(
    plane_0: &Array<u32>,
    plane_1: &Array<u32>,
    plane_2: &Array<u32>,
    ring: &mut Array<f32>,
    max: f32,
    offset: u32,
    #[comptime] pixels: u32,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
    #[comptime] samples_per_word: u32,
    #[comptime] total_threads: u32,
) {
    let bits = comptime![32u32 / samples_per_word];
    let mask = comptime![(1u32 << (32u32 / samples_per_word)) - 1];

    let mut pixel = ABSOLUTE_POS_X;
    while pixel < pixels {
        let word = (pixel / samples_per_word) as usize;
        let shift = (pixel % samples_per_word) * bits;
        let base = offset + pixel * stored_ch;

        let code_0 = (plane_0[word] >> shift) & mask;
        ring[base as usize] = f32::cast_from(code_0) / max;

        if channels > 1 {
            let code_1 = (plane_1[word] >> shift) & mask;
            ring[(base + 1) as usize] = f32::cast_from(code_1) / max;
        }

        if channels > 2 {
            let code_2 = (plane_2[word] >> shift) & mask;
            ring[(base + 2) as usize] = f32::cast_from(code_2) / max;
        }

        #[unroll]
        for lane in channels..stored_ch {
            ring[(base + lane) as usize] = 0.0f32;
        }

        pixel += total_threads;
    }
}

/// Copies f32 planes into one interleaved ring slot, with padding lanes set to zero.
///
/// A plane is only read when `channels` covers it, so an unused plane can be a placeholder.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding or a comptime shape"
)]
pub fn ingest_f32(
    plane_0: &Array<f32>,
    plane_1: &Array<f32>,
    plane_2: &Array<f32>,
    ring: &mut Array<f32>,
    offset: u32,
    #[comptime] pixels: u32,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
    #[comptime] total_threads: u32,
) {
    let mut pixel = ABSOLUTE_POS_X;
    while pixel < pixels {
        let base = offset + pixel * stored_ch;
        ring[base as usize] = plane_0[pixel as usize];

        if channels > 1 {
            ring[(base + 1) as usize] = plane_1[pixel as usize];
        }

        if channels > 2 {
            ring[(base + 2) as usize] = plane_2[pixel as usize];
        }

        #[unroll]
        for lane in channels..stored_ch {
            ring[(base + lane) as usize] = 0.0f32;
        }

        pixel += total_threads;
    }
}

/// Quantises an interleaved f32 frame into word-packed planes.
///
/// Each thread packs and writes one word of every plane, so no two threads share a word. Lanes past the
/// last pixel are written as zero. A plane is only written when `channels` covers it, so an unused plane
/// can be a placeholder.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding or a comptime shape"
)]
pub fn egress_words(
    frame: &Array<f32>,
    plane_0: &mut Array<u32>,
    plane_1: &mut Array<u32>,
    plane_2: &mut Array<u32>,
    max: f32,
    #[comptime] pixels: u32,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
    #[comptime] samples_per_word: u32,
    #[comptime] words: u32,
    #[comptime] total_threads: u32,
) {
    let bits = comptime![32u32 / samples_per_word];

    let mut word = ABSOLUTE_POS_X;
    while word < words {
        let mut packed_0 = 0u32;
        let mut packed_1 = 0u32;
        let mut packed_2 = 0u32;

        #[unroll]
        for lane in 0..samples_per_word {
            let pixel = word * samples_per_word + lane;
            let clamped = u32::min(pixel, pixels - 1);
            let in_range = pixel < pixels;
            let base = clamped * stored_ch;
            let shift = lane * bits;

            let value_0 = f32::clamp(frame[base as usize], 0.0, 1.0);
            let code_0 = u32::cast_from(value_0 * max + 0.5);
            packed_0 |= select(in_range, code_0, 0u32) << shift;

            if channels > 1 {
                let value_1 = f32::clamp(frame[(base + 1) as usize], 0.0, 1.0);
                let code_1 = u32::cast_from(value_1 * max + 0.5);
                packed_1 |= select(in_range, code_1, 0u32) << shift;
            }

            if channels > 2 {
                let value_2 = f32::clamp(frame[(base + 2) as usize], 0.0, 1.0);
                let code_2 = u32::cast_from(value_2 * max + 0.5);
                packed_2 |= select(in_range, code_2, 0u32) << shift;
            }
        }

        plane_0[word as usize] = packed_0;

        if channels > 1 {
            plane_1[word as usize] = packed_1;
        }

        if channels > 2 {
            plane_2[word as usize] = packed_2;
        }

        word += total_threads;
    }
}

/// Copies an interleaved f32 frame into f32 planes, skipping padding lanes.
///
/// A plane is only written when `channels` covers it, so an unused plane can be a placeholder.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a binding or a comptime shape"
)]
pub fn egress_f32(
    frame: &Array<f32>,
    plane_0: &mut Array<f32>,
    plane_1: &mut Array<f32>,
    plane_2: &mut Array<f32>,
    #[comptime] pixels: u32,
    #[comptime] channels: u32,
    #[comptime] stored_ch: u32,
    #[comptime] total_threads: u32,
) {
    let mut pixel = ABSOLUTE_POS_X;
    while pixel < pixels {
        let base = pixel * stored_ch;
        plane_0[pixel as usize] = frame[base as usize];

        if channels > 1 {
            plane_1[pixel as usize] = frame[(base + 1) as usize];
        }

        if channels > 2 {
            plane_2[pixel as usize] = frame[(base + 2) as usize];
        }

        pixel += total_threads;
    }
}

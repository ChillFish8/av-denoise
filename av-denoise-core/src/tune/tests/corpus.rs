use super::fixtures::CollabSpec;
use crate::nlmeans::ChannelMode;
use crate::tune::IntegerCodes;

/// One entry of the integer search corpus, a collab geometry and the ring frames it searches.
pub(crate) struct IntegerCase {
    pub(crate) label: &'static str,
    pub(crate) spec: CollabSpec,
    pub(crate) ring: Vec<f32>,
}

/// `value` rounded to the nearest code of `codes` and scaled back, clamped between 0 and 1.
fn quantise(value: f32, codes: IntegerCodes) -> f32 {
    let max_value = codes.max_code() as f32;
    let clamped = value.clamp(0.0, 1.0);
    let code = (clamped * max_value).round();

    code / max_value
}

/// Adds independent pseudo-Gaussian noise to a flat `base` field.
///
/// Each sample sums four hash-derived uniforms between -0.5 and 0.5 (Irwin-Hall) and rescales to
/// the requested standard deviation. `seed` picks the noise, so each frame can carry its own.
fn noisy_flat_field(width: u32, height: u32, base: f32, sigma: f32, seed: u32) -> Vec<f32> {
    let unit_std = (1.0f32 / 3.0f32).sqrt();
    let seed_offset = seed.wrapping_mul(0x85EB_CA6B);
    let mut frame = vec![0.0f32; (width * height) as usize];
    for idx in 0..(width * height) {
        let mut sum = 0.0f32;
        for k in 0..4u32 {
            let sample = (idx * 4 + k).wrapping_add(seed_offset);
            let mut hash = sample.wrapping_mul(2654435761).wrapping_add(0x9E3779B9);
            hash ^= hash >> 15;
            hash = hash.wrapping_mul(0x85EBCA6B);
            hash ^= hash >> 13;
            sum += (hash as f32 / u32::MAX as f32) - 0.5;
        }

        frame[idx as usize] = base + (sum / unit_std) * sigma;
    }

    frame
}

/// A horizontal ramp plus a per-pixel hash offset, so no 8x8 window repeats anywhere in the frame.
fn make_unique_frame(width: u32, height: u32) -> Vec<f32> {
    let mut frame = vec![0.0f32; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let mut hash = idx.wrapping_mul(2654435761).wrapping_add(0x9E3779B9);
            hash ^= hash >> 15;
            hash = hash.wrapping_mul(0x85EBCA6B);
            hash ^= hash >> 13;
            let offset = hash as f32 / u32::MAX as f32;
            frame[idx as usize] = x as f32 * 10.0 + offset * 10.0;
        }
    }

    frame
}

/// [make_unique_frame] rescaled to between 0 and 1.
fn unit_unique_frame(width: u32, height: u32) -> Vec<f32> {
    let raw = make_unique_frame(width, height);
    let peak = raw.iter().copied().fold(0.0f32, f32::max);

    raw.iter().map(|value| value / peak).collect()
}

/// A frame where every pixel holds `value`.
fn constant_frame(width: u32, height: u32, value: f32) -> Vec<f32> {
    vec![value; (width * height) as usize]
}

/// [make_unique_frame] content, rescaled to between 0 and 1, between black bars of `bar_rows` rows at
/// the top and bottom.
///
/// Every candidate inside a bar ties at distance zero, which is where search order decides members.
fn letterboxed_frame(width: u32, height: u32, bar_rows: u32) -> Vec<f32> {
    let raw = make_unique_frame(width, height);
    let peak = raw.iter().copied().fold(0.0f32, f32::max);
    let mut frame = vec![0.0f32; (width * height) as usize];

    for y in bar_rows..height - bar_rows {
        for x in 0..width {
            let index = (y * width + x) as usize;
            frame[index] = raw[index] / peak;
        }
    }

    frame
}

/// Flat cells of `cell` pixels in four grey levels, like flat-shaded animation.
fn cel_frame(width: u32, height: u32, cell: u32) -> Vec<f32> {
    let levels = [0.2f32, 0.45, 0.7, 0.9];
    let mut frame = vec![0.0f32; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let cell_index = (y / cell) * 7 + (x / cell) * 3;
            let level = levels[(cell_index % 4) as usize];
            frame[(y * width + x) as usize] = level;
        }
    }

    frame
}

/// `frame` moved by `(shift_x, shift_y)`, with the uncovered edge repeating the nearest pixel.
fn shifted_frame(frame: &[f32], width: u32, height: u32, shift_x: i32, shift_y: i32) -> Vec<f32> {
    let pixels = width * height;

    (0..pixels)
        .map(|index| {
            let x = (index % width) as i32;
            let y = (index / width) as i32;
            let source_x = (x - shift_x).clamp(0, width as i32 - 1) as u32;
            let source_y = (y - shift_y).clamp(0, height as i32 - 1) as u32;
            frame[(source_y * width + source_x) as usize]
        })
        .collect()
}

/// Channel `channel`'s copy of a channel-0 plane, remapped so the channels differ.
fn channel_plane(plane: &[f32], channel: u32) -> Vec<f32> {
    if channel == 0 {
        return plane.to_vec();
    }

    let stretch = 1.0 + 0.5 * channel as f32;
    let offset = 0.21 * channel as f32;
    plane
        .iter()
        .map(|value| (value * stretch + offset).fract())
        .collect()
}

/// What an integer corpus entry's frames hold.
#[derive(Debug, Clone, Copy)]
enum CaseContent {
    /// The same unique frame in every slot.
    Unique,
    /// Flat noise, drawn afresh for every slot.
    Noise,
    /// A unique frame that moves by `(2, 1)` pixels per frame away from the centre.
    Moving,
    Constant,
    Letterbox,
    Cel,
}

/// The ring of `spec` filled with `content`, every value rounded to `codes`.
///
/// Slot `slot` lies `slot - spec.radius` frames from the centre. Lanes past the logical channels
/// stay zero.
fn case_ring(spec: &CollabSpec, content: CaseContent, codes: IntegerCodes) -> Vec<f32> {
    let width = spec.width;
    let height = spec.height;
    let channels = spec.channels.count();
    let stored_ch = spec.channels.storage_count();
    let frames = 2 * spec.radius + 1;
    let pixels = (width * height) as usize;
    let unique = unit_unique_frame(width, height);
    let mut ring = vec![0.0f32; pixels * stored_ch as usize * frames as usize];

    for slot in 0..frames {
        let offset = slot as i32 - spec.radius as i32;
        let plane = match content {
            CaseContent::Unique => unique.clone(),
            CaseContent::Noise => noisy_flat_field(width, height, 0.5, 0.05, slot),
            CaseContent::Moving => shifted_frame(&unique, width, height, 2 * offset, offset),
            CaseContent::Constant => constant_frame(width, height, 0.5),
            CaseContent::Letterbox => letterboxed_frame(width, height, 8),
            CaseContent::Cel => cel_frame(width, height, 8),
        };

        for channel in 0..channels {
            let channel_values = channel_plane(&plane, channel);
            let slot_start = slot as usize * pixels;

            for (pixel, &value) in channel_values.iter().enumerate() {
                let lane = (slot_start + pixel) * stored_ch as usize + channel as usize;
                ring[lane] = quantise(value, codes);
            }
        }
    }

    ring
}

/// Whether a corpus entry's content ties candidates exactly.
pub(crate) fn is_flat_case(label: &str) -> bool {
    let flat_prefixes = ["constant", "letterbox", "cel"];
    flat_prefixes.iter().any(|prefix| label.starts_with(prefix))
}

/// The integer search corpus at `depth` bits.
///
/// It covers unique, ragged and noisy frames, cross-frame motion at radius 1 and 2, a pass whose
/// every motion block is gated off, chroma and yuv motion, and the flat contents where candidates
/// tie at radius 1 and 2. Every value is a code of `depth` bits scaled to between 0 and 1.
pub(crate) fn integer_corpus(depth: u32) -> Vec<IntegerCase> {
    let codes = IntegerCodes { depth };
    let luma_r0 = CollabSpec::new(64, 48, ChannelMode::Luma, 0);
    let luma_r1 = CollabSpec::new(64, 48, ChannelMode::Luma, 1);
    let luma_r2 = CollabSpec::new(64, 48, ChannelMode::Luma, 2);
    let ragged = CollabSpec::new(104, 96, ChannelMode::Luma, 1);
    let chroma_r1 = CollabSpec::new(64, 48, ChannelMode::Chroma, 1);
    let yuv_r1 = CollabSpec::new(64, 48, ChannelMode::Yuv, 1);
    let gated = CollabSpec {
        c_min: 2.0,
        ..luma_r1
    };
    let entries = [
        ("unique", luma_r0, CaseContent::Unique),
        ("ragged", ragged, CaseContent::Moving),
        ("noise", luma_r1, CaseContent::Noise),
        ("cross_frame_r1", luma_r1, CaseContent::Moving),
        ("cross_frame_r2", luma_r2, CaseContent::Moving),
        ("gated", gated, CaseContent::Moving),
        ("chroma_cross_frame", chroma_r1, CaseContent::Moving),
        ("yuv_cross_frame", yuv_r1, CaseContent::Moving),
        ("constant", luma_r1, CaseContent::Constant),
        ("constant_r2", luma_r2, CaseContent::Constant),
        ("letterbox", luma_r1, CaseContent::Letterbox),
        ("letterbox_r2", luma_r2, CaseContent::Letterbox),
        ("cel", luma_r1, CaseContent::Cel),
    ];

    entries
        .into_iter()
        .map(|(label, spec, content)| {
            let ring = case_ring(&spec, content, codes);
            IntegerCase { label, spec, ring }
        })
        .collect()
}

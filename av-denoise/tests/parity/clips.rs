use av_denoise::{Depth, FrameLayout, Planes};

/// The content a parity clip carries.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ClipKind {
    /// A moving sine texture plus hashed noise, so motion search and the noise estimator both have
    /// work.
    Moving,
    /// Still flat bands and a dark still texture band plus hashed noise.
    Still,
}

pub fn clip(kind: ClipKind, layout: FrameLayout, frame_count: usize) -> Vec<Planes> {
    let (chroma_width, chroma_height) = layout.chroma_dims();
    let mut frames = Vec::with_capacity(frame_count);

    for index in 0..frame_count {
        let y_plane = plane(kind, layout.width, layout.height, layout.depth, index, 0);
        let u_plane = plane(kind, chroma_width, chroma_height, layout.depth, index, 1);
        let v_plane = plane(kind, chroma_width, chroma_height, layout.depth, index, 2);
        frames.push(Planes {
            y: y_plane,
            u: u_plane,
            v: v_plane,
        });
    }

    frames
}

fn plane(kind: ClipKind, width: u32, height: u32, depth: Depth, frame: usize, channel: u32) -> Vec<u8> {
    let max_code = depth.max_value();
    let mut samples = Vec::with_capacity((width * height) as usize);

    for row in 0..height {
        for column in 0..width {
            let texture = match kind {
                ClipKind::Moving => moving_texture(width, height, column, row, frame),
                ClipKind::Still => still_texture(width, height, column, row),
            };
            let noise = hashed_noise(column, row, frame as u32, channel) * 0.03;
            let value = (texture + noise).clamp(0.0, 1.0);
            let code = (value * max_code).round() as u16;
            samples.push(code);
        }
    }

    encode(&samples, depth)
}

fn moving_texture(width: u32, height: u32, column: u32, row: u32, frame: usize) -> f32 {
    let shift = frame as f32 * 1.5;
    let phase_x = (column as f32 + shift) / width as f32 * std::f32::consts::TAU * 3.0;
    let phase_y = row as f32 / height as f32 * std::f32::consts::TAU * 2.0;
    0.5 + 0.2 * phase_x.sin() * phase_y.cos()
}

/// Three flat bands at different lumas over a dark texture band, each a quarter of the height.
fn still_texture(width: u32, height: u32, column: u32, row: u32) -> f32 {
    let band = row * 4 / height;

    match band {
        0 => 0.22,
        1 => 0.35,
        2 => 0.6,
        _ => {
            let phase_x = column as f32 / width as f32 * std::f32::consts::TAU * 8.0;
            0.4 + 0.1 * phase_x.sin()
        },
    }
}

fn hashed_noise(column: u32, row: u32, frame: u32, channel: u32) -> f32 {
    let mut hash = column.wrapping_mul(0x9E37_79B9) ^ row.wrapping_mul(0x85EB_CA6B);
    hash ^= frame.wrapping_mul(0xC2B2_AE35) ^ channel.wrapping_mul(0x27D4_EB2F);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x2C1B_3C6D);
    hash ^= hash >> 12;
    hash as f32 / u32::MAX as f32 - 0.5
}

fn encode(samples: &[u16], depth: Depth) -> Vec<u8> {
    match depth.bytes_per_sample() {
        1 => samples.iter().map(|&sample| sample as u8).collect(),
        _ => samples.iter().flat_map(|sample| sample.to_le_bytes()).collect(),
    }
}

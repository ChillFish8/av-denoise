use crate::nlmeans::motion::neighbour_idx_for_k;

/// A clean luma plane with values between 0 and 1.
#[derive(Debug, Clone)]
pub struct Still {
    pub width: u32,
    pub height: u32,
    pub luma: Vec<f32>,
}

impl Still {
    /// Parses a binary PGM (`P5`) at 8 or 16 bits per sample.
    pub fn from_pgm(bytes: &[u8]) -> Result<Still, String> {
        let mut pos = 0usize;
        let mut fields: Vec<u32> = Vec::new();
        if bytes.len() < 2 || &bytes[..2] != b"P5" {
            return Err("not a P5 pgm".to_string());
        }

        pos += 2;
        while fields.len() < 3 {
            while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }

            if pos < bytes.len() && bytes[pos] == b'#' {
                while pos < bytes.len() && bytes[pos] != b'\n' {
                    pos += 1;
                }
                continue;
            }

            let start = pos;
            while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                pos += 1;
            }

            if start == pos {
                return Err("malformed pgm header".to_string());
            }

            let text = std::str::from_utf8(&bytes[start..pos]).map_err(|error| error.to_string())?;
            let field = text.parse::<u32>().map_err(|error| error.to_string())?;
            fields.push(field);
        }

        // Exactly one whitespace byte separates maxval from the data.
        pos += 1;
        let (width, height, maxval) = (fields[0], fields[1], fields[2]);
        // Widened to 64 bits so huge header dimensions cannot wrap into a small `usize` on a 32-bit
        // target and slip past the bounds check.
        let sample_count_u64 = width as u64 * height as u64;
        let bytes_per_sample: u64 = if maxval > 255 { 2 } else { 1 };
        let available = (bytes.len() - pos.min(bytes.len())) as u64;
        if sample_count_u64 * bytes_per_sample > available {
            return Err(format!(
                "pgm data truncated: header claims {width}x{height} at {bytes_per_sample} bytes/sample, \
                 only {available} bytes remain"
            ));
        }

        let sample_count = sample_count_u64 as usize;
        let luma = if maxval > 255 {
            let data = bytes
                .get(pos..pos + 2 * sample_count)
                .ok_or("pgm data truncated")?;
            data.as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| u16::from_be_bytes(*chunk) as f32 / maxval as f32)
                .collect()
        } else {
            let data = bytes.get(pos..pos + sample_count).ok_or("pgm data truncated")?;
            data.iter().map(|&value| value as f32 / maxval as f32).collect()
        };

        Ok(Still { width, height, luma })
    }

    /// A textured plane for runs with no real still to hand.
    pub fn synthetic(width: u32, height: u32) -> Still {
        let mut luma = vec![0.0f32; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let phase_x = x as f32 * 0.31;
                let phase_y = y as f32 * 0.23;
                let value = 0.5
                    + 0.2 * (phase_x.sin() * phase_y.cos())
                    + 0.1 * ((phase_x * 2.7).cos() + (phase_y * 3.1).sin());
                luma[(y * width + x) as usize] = value.clamp(0.0, 1.0);
            }
        }

        Still { width, height, luma }
    }

    /// Samples the still at a fractional position with a Lanczos-3 kernel, clamping to the edge.
    fn sample(&self, sample_x: f32, sample_y: f32) -> f32 {
        const A: i32 = 3;

        let lanczos = |t: f32| -> f32 {
            if t == 0.0 {
                1.0
            } else if t.abs() >= A as f32 {
                0.0
            } else {
                let pi_t = std::f32::consts::PI * t;
                (A as f32 * pi_t.sin() * (pi_t / A as f32).sin()) / (pi_t * pi_t)
            }
        };

        let x0 = sample_x.floor() as i32;
        let y0 = sample_y.floor() as i32;
        let mut weighted_sum = 0.0f32;
        let mut weight_sum = 0.0f32;
        for j in (y0 - A + 1)..=(y0 + A) {
            let weight_y = lanczos(sample_y - j as f32);
            let clamped_y = j.clamp(0, self.height as i32 - 1) as u32;
            for i in (x0 - A + 1)..=(x0 + A) {
                let weight_x = lanczos(sample_x - i as f32);
                let weight = weight_y * weight_x;
                let clamped_x = i.clamp(0, self.width as i32 - 1) as u32;
                weighted_sum += weight * self.luma[(clamped_y * self.width + clamped_x) as usize];
                weight_sum += weight;
            }
        }

        (weighted_sum / weight_sum).clamp(0.0, 1.0)
    }
}

/// The motion each synthetic clip carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionClass {
    IntegerPan,
    HalfPelPan,
    Zoom,
    CutOut,
}

/// Scale per frame of the zoom class.
const ZOOM_PER_FRAME: f32 = 1.01;

impl MotionClass {
    pub const ALL: [MotionClass; 4] = [
        MotionClass::IntegerPan,
        MotionClass::HalfPelPan,
        MotionClass::Zoom,
        MotionClass::CutOut,
    ];

    pub fn label(self) -> &'static str {
        match self {
            MotionClass::IntegerPan => "pan_int",
            MotionClass::HalfPelPan => "pan_half",
            MotionClass::Zoom => "zoom",
            MotionClass::CutOut => "cutout",
        }
    }

    /// Per-frame velocity of the moving content, in pixels.
    pub fn velocity(self) -> [f32; 2] {
        match self {
            MotionClass::IntegerPan => [3.0, 1.0],
            MotionClass::HalfPelPan => [2.5, 0.5],
            MotionClass::Zoom => [0.0, 0.0],
            MotionClass::CutOut => [4.0, 2.0],
        }
    }

    /// Top-left corner and side of the cut-out square in the centre frame.
    ///
    /// The square is a third of the shorter side, left of centre so its rightward motion stays
    /// inside the frame.
    pub fn cut_out_rect(width: u32, height: u32) -> (u32, u32, u32) {
        let side = (width.min(height) / 3).max(8);
        let rect_x = width / 4;
        let rect_y = (height - side) / 2;

        (rect_x, rect_y, side)
    }
}

/// A synthetic window of frames with its per-pixel ground truth.
#[derive(Debug, Clone)]
pub struct Clip {
    pub width: u32,
    pub height: u32,
    pub radius: u32,
    /// `frames[i]` is the frame at offset `k = i - radius`.
    pub frames: Vec<Vec<f32>>,
    /// `truth[t][pixel]` is where the centre frame's pixel lies in neighbour `t`, as a
    /// displacement in pixels.
    pub truth: Vec<Vec<[f32; 2]>>,
    /// `occluded[t][pixel]` is true when that pixel has no true match in neighbour `t`.
    pub occluded: Vec<Vec<bool>>,
}

/// Where the centre frame's background pixel `(x, y)` sits in the frame at offset `k`.
fn background_displacement(class: MotionClass, k: i32, x: u32, y: u32, width: u32, height: u32) -> [f32; 2] {
    match class {
        MotionClass::IntegerPan | MotionClass::HalfPelPan => {
            let velocity = class.velocity();
            [velocity[0] * k as f32, velocity[1] * k as f32]
        },
        MotionClass::Zoom => {
            let scale = ZOOM_PER_FRAME.powi(k);
            let centre_x = width as f32 / 2.0;
            let centre_y = height as f32 / 2.0;
            [
                (x as f32 - centre_x) * (scale - 1.0),
                (y as f32 - centre_y) * (scale - 1.0),
            ]
        },
        MotionClass::CutOut => [0.0, 0.0],
    }
}

/// Deterministic Gaussian grain from a hashed uniform pair.
fn grain(idx: u32, seed: u32) -> f32 {
    let hash = |i: u32| -> f32 {
        let mut state = i
            .wrapping_mul(2654435761)
            .wrapping_add(seed.wrapping_mul(0x9E37_79B9));
        state ^= state >> 15;
        state = state.wrapping_mul(0x85EB_CA6B);
        state ^= state >> 13;
        (state as f32 + 1.0) / (u32::MAX as f32 + 2.0)
    };

    let u1 = hash(idx * 2);
    let u2 = hash(idx * 2 + 1);

    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Builds the window of `2 * radius + 1` frames for `class` and the ground truth toward every
/// neighbour.
///
/// Every frame carries its own Gaussian grain of `sigma`.
pub fn synthesise(still: &Still, class: MotionClass, radius: u32, sigma: f32, seed: u32) -> Clip {
    let (width, height) = (still.width, still.height);
    let pixels = (width * height) as usize;
    let (rect_x, rect_y, side) = MotionClass::cut_out_rect(width, height);
    let velocity = class.velocity();

    let in_rect_at = |x: f32, y: f32, k: i32| -> bool {
        let origin_x = rect_x as f32 + velocity[0] * k as f32;
        let origin_y = rect_y as f32 + velocity[1] * k as f32;
        x >= origin_x && x < origin_x + side as f32 && y >= origin_y && y < origin_y + side as f32
    };

    let mut frames = Vec::with_capacity((2 * radius + 1) as usize);
    for i in 0..(2 * radius + 1) as i32 {
        let k = i - radius as i32;
        let mut frame = vec![0.0f32; pixels];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let value = if class == MotionClass::CutOut && in_rect_at(x as f32, y as f32, k) {
                    // The rectangle's content, read from where it sat in the centre.
                    still.sample(
                        x as f32 - velocity[0] * k as f32,
                        y as f32 - velocity[1] * k as f32,
                    )
                } else {
                    let displacement = background_displacement(class, k, x, y, width, height);
                    // The frame at k shows the centre's pixel p at p + d, so pixel (x, y) here
                    // comes from the centre's (x, y) - d. For a pan and a zoom the inverse is exact.
                    match class {
                        MotionClass::Zoom => {
                            let scale = ZOOM_PER_FRAME.powi(k);
                            let source_x = width as f32 / 2.0 + (x as f32 - width as f32 / 2.0) / scale;
                            let source_y = height as f32 / 2.0 + (y as f32 - height as f32 / 2.0) / scale;
                            still.sample(source_x, source_y)
                        },
                        _ => still.sample(x as f32 - displacement[0], y as f32 - displacement[1]),
                    }
                };
                let noise = if sigma > 0.0 {
                    let frame_seed = seed.wrapping_add(1000 * (i as u32 + 1));
                    sigma * grain(idx as u32, frame_seed)
                } else {
                    0.0
                };
                frame[idx] = (value + noise).clamp(0.0, 1.0);
            }
        }

        frames.push(frame);
    }

    let neighbours = (2 * radius) as usize;
    let mut truth = vec![vec![[0.0f32; 2]; pixels]; neighbours];
    let mut occluded = vec![vec![false; pixels]; neighbours];
    for k in -(radius as i32)..=(radius as i32) {
        if k == 0 {
            continue;
        }

        let t = neighbour_idx_for_k(radius, k) as usize;
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let foreground = class == MotionClass::CutOut && in_rect_at(x as f32, y as f32, 0);
                let displacement = if foreground {
                    [velocity[0] * k as f32, velocity[1] * k as f32]
                } else {
                    background_displacement(class, k, x, y, width, height)
                };
                truth[t][idx] = displacement;

                if class == MotionClass::CutOut && !foreground {
                    let moved_x = x as f32 + displacement[0];
                    let moved_y = y as f32 + displacement[1];
                    occluded[t][idx] = in_rect_at(moved_x, moved_y, k);
                }
            }
        }
    }

    Clip {
        width,
        height,
        radius,
        frames,
        truth,
        occluded,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_still() -> Still {
        // Distinct values everywhere, so a shift is visible in any pixel.
        let (width, height) = (64u32, 48u32);
        let luma = (0..width * height)
            .map(|i| {
                let ramp_x = (i % width) as f32 * 0.9 / width as f32;
                let ramp_y = (i / width) as f32 * 0.1 / height as f32;
                (ramp_x + ramp_y).clamp(0.0, 1.0)
            })
            .collect();

        Still { width, height, luma }
    }

    #[test]
    fn integer_pan_frames_are_exact_shifts_and_truth_is_the_velocity() {
        let still = ramp_still();
        let clip = synthesise(&still, MotionClass::IntegerPan, 1, 0.0, 1);
        assert_eq!(clip.frames.len(), 3);

        let (width, height) = (clip.width, clip.height);
        let velocity = MotionClass::IntegerPan.velocity();
        // Frame k = +1 holds the still moved by +velocity. Check an interior pixel.
        let (x, y) = (20u32, 20u32);
        let moved = clip.frames[2][(y * width + x) as usize];
        let source_y = (y as i32 - velocity[1] as i32) as u32;
        let source_x = (x as i32 - velocity[0] as i32) as u32;
        let source = still.luma[(source_y * width + source_x) as usize];
        assert!(
            (moved - source).abs() < 1e-6,
            "an integer pan must copy pixels exactly"
        );

        // Truth toward k = +1 (t = 1 at radius 1) is +velocity everywhere.
        for idx in 0..(width * height) as usize {
            assert_eq!(clip.truth[1][idx], velocity);
            assert_eq!(clip.truth[0][idx], [-velocity[0], -velocity[1]]);
            assert!(!clip.occluded[1][idx]);
        }
    }

    #[test]
    fn half_pel_pan_truth_has_a_half_pixel_component() {
        let still = ramp_still();
        let clip = synthesise(&still, MotionClass::HalfPelPan, 1, 0.0, 1);
        let velocity = MotionClass::HalfPelPan.velocity();
        let half_x = (velocity[0].fract().abs() - 0.5).abs() < 1e-6;
        let half_y = (velocity[1].fract().abs() - 0.5).abs() < 1e-6;
        assert!(half_x || half_y);
        assert_eq!(clip.truth[1][100], velocity);
    }

    #[test]
    fn zoom_truth_grows_with_distance_from_the_centre() {
        let still = ramp_still();
        let clip = synthesise(&still, MotionClass::Zoom, 1, 0.0, 1);
        let (width, height) = (clip.width, clip.height);
        let centre = ((height / 2) * width + width / 2) as usize;
        let corner = 0usize;
        let centre_shift = clip.truth[1][centre];
        let corner_shift = clip.truth[1][corner];
        assert!(
            centre_shift[0].abs() < 0.01 && centre_shift[1].abs() < 0.01,
            "the centre does not move under a zoom"
        );
        assert!(
            corner_shift[0] < -0.1 && corner_shift[1] < -0.1,
            "the top-left corner moves outward, got {corner_shift:?}"
        );
    }

    #[test]
    fn cut_out_marks_background_hidden_under_the_moved_rectangle() {
        let still = ramp_still();
        let clip = synthesise(&still, MotionClass::CutOut, 1, 0.0, 1);
        let width = clip.width;
        let (rect_x, rect_y, side) = MotionClass::cut_out_rect(clip.width, clip.height);
        let velocity = MotionClass::CutOut.velocity();

        // A pixel inside the rectangle in the centre frame moves with it.
        let inside = ((rect_y + side / 2) * width + rect_x + side / 2) as usize;
        assert_eq!(clip.truth[1][inside], velocity);
        assert!(!clip.occluded[1][inside]);

        // A background pixel just right of the rectangle is covered once the rectangle moves
        // right, so it is occluded toward k = +1 and not toward k = -1.
        let just_right = ((rect_y + side / 2) * width + rect_x + side + 1) as usize;
        assert_eq!(clip.truth[1][just_right], [0.0, 0.0]);
        assert!(clip.occluded[1][just_right]);
        assert!(!clip.occluded[0][just_right]);
    }

    #[test]
    fn grain_has_the_requested_sigma_and_differs_between_frames() {
        let still = Still::synthetic(128, 128);
        let clip = synthesise(&still, MotionClass::IntegerPan, 1, 0.0, 1);
        let noisy = synthesise(&still, MotionClass::IntegerPan, 1, 6.0 / 255.0, 1);
        let pixels = clip.frames[1].len() as f32;
        let variance: f32 = clip.frames[1]
            .iter()
            .zip(&noisy.frames[1])
            .map(|(clean, grainy)| (clean - grainy) * (clean - grainy))
            .sum::<f32>()
            / pixels;
        let sigma = variance.sqrt();
        assert!(
            (sigma - 6.0 / 255.0).abs() < 0.1 * 6.0 / 255.0,
            "measured sigma {sigma}"
        );
        assert_ne!(
            noisy.frames[0], noisy.frames[1],
            "each frame carries its own grain"
        );
    }

    #[test]
    fn pgm_parses_8_and_16_bit_planes() {
        let mut pgm_8bit = b"P5\n# comment\n2 2\n255\n".to_vec();
        pgm_8bit.extend_from_slice(&[0, 128, 255, 64]);
        let parsed = Still::from_pgm(&pgm_8bit).expect("8-bit parse");
        assert_eq!((parsed.width, parsed.height), (2, 2));
        assert!((parsed.luma[1] - 128.0 / 255.0).abs() < 1e-6);

        let mut pgm_16bit = b"P5 2 1 65535\n".to_vec();
        pgm_16bit.extend_from_slice(&[0xFF, 0xFF, 0x00, 0x00]);
        let parsed = Still::from_pgm(&pgm_16bit).expect("16-bit parse");
        assert_eq!(parsed.luma, vec![1.0, 0.0]);
    }

    #[test]
    fn a_header_claiming_more_data_than_is_present_is_rejected() {
        let mut pgm = b"P5\n100 100\n255\n".to_vec();
        pgm.extend_from_slice(&[0, 128, 255, 64]);
        let err = Still::from_pgm(&pgm).expect_err("truncated data must be rejected");
        assert!(err.contains("truncated"), "got {err}");
    }

    #[test]
    fn a_header_with_dimensions_that_overflow_u32_is_rejected_not_wrapped() {
        let pgm = b"P5\n70000 70000\n255\n".to_vec();
        let err = Still::from_pgm(&pgm).expect_err("an overflowing header must be rejected");
        assert!(err.contains("truncated"), "got {err}");
    }
}

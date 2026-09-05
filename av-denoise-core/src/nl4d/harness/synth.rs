use crate::nlmeans::motion::neighbour_idx_for_k;

/// A clean luma plane, values in `[0, 1]`.
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
            let text = std::str::from_utf8(&bytes[start..pos]).map_err(|e| e.to_string())?;
            fields.push(text.parse::<u32>().map_err(|e| e.to_string())?);
        }
        // Exactly one whitespace byte separates maxval from the data.
        pos += 1;
        let (width, height, maxval) = (fields[0], fields[1], fields[2]);
        let n = (width * height) as usize;
        let luma = if maxval > 255 {
            let data = bytes.get(pos..pos + 2 * n).ok_or("pgm data truncated")?;
            data.as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_be_bytes(*c) as f32 / maxval as f32)
                .collect()
        } else {
            let data = bytes.get(pos..pos + n).ok_or("pgm data truncated")?;
            data.iter().map(|&v| v as f32 / maxval as f32).collect()
        };
        Ok(Still { width, height, luma })
    }

    /// A textured plane for runs with no real still to hand.
    pub fn synthetic(width: u32, height: u32) -> Still {
        let mut luma = vec![0.0f32; (width * height) as usize];
        for y in 0..height {
            for x in 0..width {
                let fx = x as f32 * 0.31;
                let fy = y as f32 * 0.23;
                let v = 0.5 + 0.2 * (fx.sin() * fy.cos()) + 0.1 * ((fx * 2.7).cos() + (fy * 3.1).sin());
                luma[(y * width + x) as usize] = v.clamp(0.0, 1.0);
            }
        }
        Still { width, height, luma }
    }

    /// Samples the still at a fractional position with a Lanczos-3
    /// kernel, clamping to the edge.
    fn sample(&self, sx: f32, sy: f32) -> f32 {
        const A: i32 = 3;
        let lanczos = |t: f32| -> f32 {
            if t == 0.0 {
                1.0
            } else if t.abs() >= A as f32 {
                0.0
            } else {
                let pt = std::f32::consts::PI * t;
                (A as f32 * pt.sin() * (pt / A as f32).sin()) / (pt * pt)
            }
        };
        let x0 = sx.floor() as i32;
        let y0 = sy.floor() as i32;
        let mut acc = 0.0f32;
        let mut wsum = 0.0f32;
        for j in (y0 - A + 1)..=(y0 + A) {
            let wy = lanczos(sy - j as f32);
            let yy = j.clamp(0, self.height as i32 - 1) as u32;
            for i in (x0 - A + 1)..=(x0 + A) {
                let w = wy * lanczos(sx - i as f32);
                let xx = i.clamp(0, self.width as i32 - 1) as u32;
                acc += w * self.luma[(yy * self.width + xx) as usize];
                wsum += w;
            }
        }
        (acc / wsum).clamp(0.0, 1.0)
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

    /// Top-left corner and side of the cut-out rectangle in the centre
    /// frame, a square a third of the shorter side, left of centre so
    /// its rightward motion stays inside the frame.
    pub fn cut_out_rect(width: u32, height: u32) -> (u32, u32, u32) {
        let side = (width.min(height) / 3).max(8);
        let x0 = width / 4;
        let y0 = (height - side) / 2;
        (x0, y0, side)
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
    /// `truth[t][pixel]` is where the centre frame's pixel lies in
    /// neighbour `t`, as a displacement in pixels.
    pub truth: Vec<Vec<[f32; 2]>>,
    /// `occluded[t][pixel]` is true when that pixel has no true match
    /// in neighbour `t`.
    pub occluded: Vec<Vec<bool>>,
}

/// Where the centre frame's pixel `(x, y)` sits in the frame at offset
/// `k`, for the background of `class`.
fn background_displacement(class: MotionClass, k: i32, x: u32, y: u32, width: u32, height: u32) -> [f32; 2] {
    match class {
        MotionClass::IntegerPan | MotionClass::HalfPelPan => {
            let v = class.velocity();
            [v[0] * k as f32, v[1] * k as f32]
        },
        MotionClass::Zoom => {
            let s = ZOOM_PER_FRAME.powi(k);
            let cx = width as f32 / 2.0;
            let cy = height as f32 / 2.0;
            [(x as f32 - cx) * (s - 1.0), (y as f32 - cy) * (s - 1.0)]
        },
        MotionClass::CutOut => [0.0, 0.0],
    }
}

/// Deterministic Gaussian grain from a hashed uniform pair.
fn grain(idx: u32, seed: u32) -> f32 {
    let hash = |i: u32| -> f32 {
        let mut h = i
            .wrapping_mul(2654435761)
            .wrapping_add(seed.wrapping_mul(0x9E37_79B9));
        h ^= h >> 15;
        h = h.wrapping_mul(0x85EB_CA6B);
        h ^= h >> 13;
        (h as f32 + 1.0) / (u32::MAX as f32 + 2.0)
    };
    let u1 = hash(idx * 2);
    let u2 = hash(idx * 2 + 1);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Builds the window of `2 * radius + 1` frames for `class`, with
/// Gaussian grain of `sigma` on every frame, and the ground truth toward
/// every neighbour.
pub fn synthesise(still: &Still, class: MotionClass, radius: u32, sigma: f32, seed: u32) -> Clip {
    let (w, h) = (still.width, still.height);
    let n = (w * h) as usize;
    let (cx0, cy0, side) = MotionClass::cut_out_rect(w, h);
    let v = class.velocity();

    let in_rect_at = |x: f32, y: f32, k: i32| -> bool {
        let ox = cx0 as f32 + v[0] * k as f32;
        let oy = cy0 as f32 + v[1] * k as f32;
        x >= ox && x < ox + side as f32 && y >= oy && y < oy + side as f32
    };

    let mut frames = Vec::with_capacity((2 * radius + 1) as usize);
    for i in 0..(2 * radius + 1) as i32 {
        let k = i - radius as i32;
        let mut frame = vec![0.0f32; n];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let value = if class == MotionClass::CutOut && in_rect_at(x as f32, y as f32, k) {
                    // The rectangle's content, read from where it sat in the centre.
                    still.sample(x as f32 - v[0] * k as f32, y as f32 - v[1] * k as f32)
                } else {
                    let d = background_displacement(class, k, x, y, w, h);
                    // The frame at k shows the centre's pixel p at p + d, so
                    // pixel (x, y) here comes from the centre's (x, y) - d.
                    // For a pan and a zoom the inverse map is exact.
                    match class {
                        MotionClass::Zoom => {
                            let s = ZOOM_PER_FRAME.powi(k);
                            let fx = w as f32 / 2.0 + (x as f32 - w as f32 / 2.0) / s;
                            let fy = h as f32 / 2.0 + (y as f32 - h as f32 / 2.0) / s;
                            still.sample(fx, fy)
                        },
                        _ => still.sample(x as f32 - d[0], y as f32 - d[1]),
                    }
                };
                let noise = if sigma > 0.0 {
                    sigma * grain(idx as u32, seed.wrapping_add(1000 * (i as u32 + 1)))
                } else {
                    0.0
                };
                frame[idx] = (value + noise).clamp(0.0, 1.0);
            }
        }
        frames.push(frame);
    }

    let neighbours = (2 * radius) as usize;
    let mut truth = vec![vec![[0.0f32; 2]; n]; neighbours];
    let mut occluded = vec![vec![false; n]; neighbours];
    for k in -(radius as i32)..=(radius as i32) {
        if k == 0 {
            continue;
        }
        let t = neighbour_idx_for_k(radius, k) as usize;
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let foreground = class == MotionClass::CutOut && in_rect_at(x as f32, y as f32, 0);
                let d = if foreground {
                    [v[0] * k as f32, v[1] * k as f32]
                } else {
                    background_displacement(class, k, x, y, w, h)
                };
                truth[t][idx] = d;
                if class == MotionClass::CutOut && !foreground {
                    occluded[t][idx] = in_rect_at(x as f32 + d[0], y as f32 + d[1], k);
                }
            }
        }
    }

    Clip {
        width: w,
        height: h,
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
        let (w, h) = (64u32, 48u32);
        let luma = (0..w * h)
            .map(|i| ((i % w) as f32 * 0.9 / w as f32 + (i / w) as f32 * 0.1 / h as f32).clamp(0.0, 1.0))
            .collect();
        Still {
            width: w,
            height: h,
            luma,
        }
    }

    #[test]
    fn integer_pan_frames_are_exact_shifts_and_truth_is_the_velocity() {
        let still = ramp_still();
        let clip = synthesise(&still, MotionClass::IntegerPan, 1, 0.0, 1);
        assert_eq!(clip.frames.len(), 3);
        let (w, h) = (clip.width, clip.height);
        let v = MotionClass::IntegerPan.velocity();
        // Frame k = +1 holds the still moved by +v. Check an interior pixel.
        let (x, y) = (20u32, 20u32);
        let moved = clip.frames[2][(y * w + x) as usize];
        let source =
            still.luma[((y as i32 - v[1] as i32) as u32 * w + (x as i32 - v[0] as i32) as u32) as usize];
        assert!(
            (moved - source).abs() < 1e-6,
            "an integer pan must copy pixels exactly"
        );
        // Truth toward k = +1 (t = 1 at radius 1) is +v everywhere.
        for idx in 0..(w * h) as usize {
            assert_eq!(clip.truth[1][idx], v);
            assert_eq!(clip.truth[0][idx], [-v[0], -v[1]]);
            assert!(!clip.occluded[1][idx]);
        }
    }

    #[test]
    fn half_pel_pan_truth_has_a_half_pixel_component() {
        let still = ramp_still();
        let clip = synthesise(&still, MotionClass::HalfPelPan, 1, 0.0, 1);
        let v = MotionClass::HalfPelPan.velocity();
        assert!((v[0].fract().abs() - 0.5).abs() < 1e-6 || (v[1].fract().abs() - 0.5).abs() < 1e-6);
        assert_eq!(clip.truth[1][100], v);
    }

    #[test]
    fn zoom_truth_grows_with_distance_from_the_centre() {
        let still = ramp_still();
        let clip = synthesise(&still, MotionClass::Zoom, 1, 0.0, 1);
        let (w, h) = (clip.width, clip.height);
        let centre = ((h / 2) * w + w / 2) as usize;
        let corner = 0usize;
        let dc = clip.truth[1][centre];
        let dk = clip.truth[1][corner];
        assert!(
            dc[0].abs() < 0.01 && dc[1].abs() < 0.01,
            "the centre does not move under a zoom"
        );
        assert!(
            dk[0] < -0.1 && dk[1] < -0.1,
            "the top-left corner moves outward, got {dk:?}"
        );
    }

    #[test]
    fn cut_out_marks_background_hidden_under_the_moved_rectangle() {
        let still = ramp_still();
        let clip = synthesise(&still, MotionClass::CutOut, 1, 0.0, 1);
        let w = clip.width;
        let (x0, y0, side) = MotionClass::cut_out_rect(clip.width, clip.height);
        let v = MotionClass::CutOut.velocity();
        // A pixel inside the rectangle in the centre frame moves with it.
        let inside = ((y0 + side / 2) * w + x0 + side / 2) as usize;
        assert_eq!(clip.truth[1][inside], v);
        assert!(!clip.occluded[1][inside]);
        // A background pixel just to the right of the rectangle is
        // covered once it moves right by v[0] pixels, so toward k = +1
        // it is occluded, and toward k = -1 it is not.
        let just_right = ((y0 + side / 2) * w + x0 + side + 1) as usize;
        assert_eq!(clip.truth[1][just_right], [0.0, 0.0]);
        assert!(clip.occluded[1][just_right]);
        assert!(!clip.occluded[0][just_right]);
    }

    #[test]
    fn grain_has_the_requested_sigma_and_differs_between_frames() {
        let still = Still::synthetic(128, 128);
        let clip = synthesise(&still, MotionClass::IntegerPan, 1, 0.0, 1);
        let noisy = synthesise(&still, MotionClass::IntegerPan, 1, 6.0 / 255.0, 1);
        let n = clip.frames[1].len() as f32;
        let var: f32 = clip.frames[1]
            .iter()
            .zip(&noisy.frames[1])
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / n;
        let sigma = var.sqrt();
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
        let mut p8 = b"P5\n# comment\n2 2\n255\n".to_vec();
        p8.extend_from_slice(&[0, 128, 255, 64]);
        let s = Still::from_pgm(&p8).expect("8-bit parse");
        assert_eq!((s.width, s.height), (2, 2));
        assert!((s.luma[1] - 128.0 / 255.0).abs() < 1e-6);
        let mut p16 = b"P5 2 1 65535\n".to_vec();
        p16.extend_from_slice(&[0xFF, 0xFF, 0x00, 0x00]);
        let s = Still::from_pgm(&p16).expect("16-bit parse");
        assert_eq!(s.luma, vec![1.0, 0.0]);
    }
}

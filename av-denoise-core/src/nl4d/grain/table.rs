use std::fmt::Write;

use super::consts::{AR_COEFFS, CHROMA_PLANES};
use super::segment::{FittedEntry, PlaneEntry, SceneGrain, fit_scenes};

/// Grain table timestamps count 100 ns ticks.
const TICKS_PER_SECOND: u128 = 10_000_000;
const SEED_BASE: u64 = 7391;
const SEED_STEP: u64 = 1237;
/// The scaling line values of a chroma plane without grain.
const ZEROED_SCALING: &str = "2 0 0 255 0";

/// The tick half a frame before `frame`, clamped at 0, for a rate of `numerator / denominator` fps.
pub(crate) fn boundary_ticks(frame: u64, frame_rate: (u64, u64)) -> u64 {
    if frame == 0 {
        return 0;
    }

    let (numerator, denominator) = frame_rate;
    let half_frames = 2 * frame as u128 - 1;
    let scaled = half_frames * denominator as u128 * TICKS_PER_SECOND;
    let divisor = 2 * numerator as u128;

    ((scaled + divisor / 2) / divisor) as u64
}

/// The seed for the entry at `index` in the written table.
fn seed_for(index: usize) -> u16 {
    let mixed = SEED_BASE + SEED_STEP * index as u64;

    (mixed & 0xFFFF) as u16
}

fn join_values<T: ToString>(values: &[T]) -> String {
    let words: Vec<String> = values.iter().map(ToString::to_string).collect();

    words.join(" ")
}

fn join_points(points: &[(u8, u8)]) -> String {
    let words: Vec<String> = points.iter().map(|(x, y)| format!("{x} {y}")).collect();

    words.join(" ")
}

/// A chroma plane's scaling line values, the point count then the points.
fn plane_scaling(plane: &PlaneEntry) -> String {
    let count = plane.points.len();
    let points = join_points(&plane.points);

    format!("{count} {points}")
}

/// A chroma plane's coefficient line values, the 24 weights then a luma coefficient of 0.
fn plane_coeffs(plane: &PlaneEntry) -> String {
    let mut values = plane.ar_coeffs.to_vec();
    values.push(0);

    join_values(&values)
}

/// The Cb and Cr values `line` writes for each plane, or `zeroed` for both without chroma.
fn chroma_values(
    chroma: Option<&[PlaneEntry; CHROMA_PLANES]>,
    line: fn(&PlaneEntry) -> String,
    zeroed: &str,
) -> [String; CHROMA_PLANES] {
    match chroma {
        Some(planes) => planes.each_ref().map(line),
        None => [zeroed.to_owned(), zeroed.to_owned()],
    }
}

pub(crate) fn format_table(entries: &[FittedEntry], frame_rate: (u64, u64)) -> String {
    let mut zeroed_values = vec![0i32; AR_COEFFS];
    zeroed_values.push(-128);
    let zeroed_coeffs = join_values(&zeroed_values);

    let mut text = String::from("filmgrn1\n");

    for (index, entry) in entries.iter().enumerate() {
        let start = boundary_ticks(entry.first_frame, frame_rate);
        let own_end = boundary_ticks(entry.last_frame + 1, frame_rate);
        let touches_next = entries
            .get(index + 1)
            .is_some_and(|next| next.first_frame == entry.last_frame + 1);
        let end = if touches_next { own_end - 1 } else { own_end };

        let seed = seed_for(index);
        let points = join_points(&entry.points);
        let point_count = entry.points.len();
        let coeffs = join_values(&entry.ar_coeffs);
        // Without chroma the planes scale to zero everywhere, so only luma gets grain.
        let chroma = entry.chroma.as_ref();
        let [cb_scaling, cr_scaling] = chroma_values(chroma, plane_scaling, ZEROED_SCALING);
        let [cb_coeffs, cr_coeffs] = chroma_values(chroma, plane_coeffs, &zeroed_coeffs);

        writeln!(text, "E {start} {end} 1 {seed} 1").expect("writing to a String");
        // Lag 3, the AR and scaling shifts, and overlap on.
        writeln!(
            text,
            "\tp 3 {} 0 {} 0 1 128 192 256 128 192 256",
            entry.ar_shift, entry.scaling_shift
        )
        .expect("writing to a String");
        writeln!(text, "\tsY {point_count}  {points}").expect("writing to a String");
        writeln!(text, "\tsCb {cb_scaling}").expect("writing to a String");
        writeln!(text, "\tsCr {cr_scaling}").expect("writing to a String");
        writeln!(text, "\tcY {coeffs}").expect("writing to a String");
        writeln!(text, "\tcCb {cb_coeffs}").expect("writing to a String");
        writeln!(text, "\tcCr {cr_coeffs}").expect("writing to a String");
    }

    text
}

/// The recommended share of the fine, fresh scan noise moved out of a grain table's texture.
pub const DEFAULT_FINE_NOISE: f64 = 0.5;

/// How a grain table is fitted.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct FitOptions {
    /// The share of the fine, fresh scan noise moved out of the grain's texture, between 0 and 1.
    ///
    /// The grain keeps its measured strength and only the texture shifts towards the textured
    /// grain. The default of 0 leaves the texture as measured. [DEFAULT_FINE_NOISE] is the
    /// recommended share.
    pub fine_noise: f64,
    /// Whether entries carry Cb and Cr grain as well as luma, off by default.
    ///
    /// With chroma on, one AR shift and one scaling shift fit all three planes, so luma's values
    /// can differ from a luma-only table by rounding.
    pub chroma: bool,
    /// The bit depth the measured video was rounded to, `None` by default.
    ///
    /// Each strength reading has that rounding taken off before the fit. `None` takes no rounding
    /// off.
    pub bit_depth: Option<u32>,
}

impl FitOptions {
    /// The options with `share` of the fine noise moved out of the texture.
    pub fn with_fine_noise(share: f64) -> Self {
        Self {
            fine_noise: share,
            ..Self::default()
        }
    }

    /// The options with chroma grain on or off.
    pub fn with_chroma(self, chroma: bool) -> Self {
        Self { chroma, ..self }
    }

    /// The options with the rounding of `bit_depth`-bit video taken off each strength reading.
    pub fn with_bit_depth(self, bit_depth: u32) -> Self {
        Self {
            bit_depth: Some(bit_depth),
            ..self
        }
    }

    /// The options with the fine noise share clamped to 0..=1, NaN read as 0.
    fn clamped(&self) -> Self {
        let share = self.fine_noise;
        let fine_noise = if share.is_nan() {
            0.0
        } else {
            share.clamp(0.0, 1.0)
        };

        Self {
            fine_noise,
            ..self.clone()
        }
    }
}

/// Fits every scene and returns the `filmgrn1` table text.
///
/// `frame_rate` is `(numerator, denominator)` frames per second. Entries follow the output's frame
/// timeline from 0. The fine noise share is clamped to 0..=1, and NaN reads as 0.
pub fn build_table(scenes: &[SceneGrain], frame_rate: (u64, u64), options: &FitOptions) -> String {
    let mut sorted: Vec<&SceneGrain> = scenes.iter().collect();
    sorted.sort_by_key(|scene| scene.first_frame);

    let clamped = options.clamped();
    let entries = fit_scenes(&sorted, &clamped);

    format_table(&entries, frame_rate)
}

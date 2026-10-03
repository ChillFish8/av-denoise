use std::fmt::Write;

use super::consts::AR_COEFFS;
use super::segment::{FittedEntry, SceneGrain, fit_scenes};

const TICKS_PER_SECOND: u128 = 10_000_000;
const SEED_BASE: u64 = 7391;
const SEED_STEP: u64 = 1237;

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

pub(crate) fn format_table(entries: &[FittedEntry], frame_rate: (u64, u64)) -> String {
    let mut chroma_values = vec![0i32; AR_COEFFS];
    chroma_values.push(-128);
    let chroma_coeffs = join_values(&chroma_values);

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

        writeln!(text, "E {start} {end} 1 {seed} 1").expect("writing to a String");
        writeln!(
            text,
            "\tp 3 {} 0 {} 0 1 128 192 256 128 192 256",
            entry.ar_shift, entry.scaling_shift
        )
        .expect("writing to a String");
        writeln!(text, "\tsY {point_count}  {points}").expect("writing to a String");
        text.push_str("\tsCb 2 0 0 255 0\n");
        text.push_str("\tsCr 2 0 0 255 0\n");
        writeln!(text, "\tcY {coeffs}").expect("writing to a String");
        writeln!(text, "\tcCb {chroma_coeffs}").expect("writing to a String");
        writeln!(text, "\tcCr {chroma_coeffs}").expect("writing to a String");
    }

    text
}

/// Fits every scene and returns the `filmgrn1` table text.
///
/// `frame_rate` is `(numerator, denominator)` frames per second. Entries follow the output's frame
/// timeline from 0.
pub fn build_table(scenes: &[SceneGrain], frame_rate: (u64, u64)) -> String {
    let mut sorted = scenes.to_vec();
    sorted.sort_by_key(|scene| scene.first_frame);

    let entries = fit_scenes(&sorted);

    format_table(&entries, frame_rate)
}

use super::synthetic::{chunk_at, fit_measured};
use crate::nl4d::grain::consts::{AR_COEFFS, CHUNK_FRAMES};
use crate::nl4d::grain::seeds::{SeedSigmas, fresh_seed};
use crate::nl4d::grain::segment::{FittedEntry, SceneGrain, fit_scenes_with};
use crate::nl4d::grain::table::{FitOptions, boundary_ticks, format_table};
use crate::nl4d::grain::template::{TemplatePlane, seed_sigma};

const NTSC: (u64, u64) = (24_000, 1_001);

/// A scene of `chunk_count` steady chunks starting at `first_frame`.
fn steady_scene(first_frame: u64, chunk_count: usize) -> SceneGrain {
    let chunks = (0..chunk_count).map(|_| chunk_at(2.0, 300, None)).collect();

    SceneGrain {
        first_frame,
        chunks,
        chroma_chunks: Vec::new(),
    }
}

/// A one-frame entry at `frame` with `seed`.
fn one_frame_entry(frame: u64, seed: u16) -> FittedEntry {
    let mut ar_coeffs = [0i32; AR_COEFFS];
    ar_coeffs[AR_COEFFS - 1] = 71;

    FittedEntry {
        first_frame: frame,
        last_frame: frame,
        seed,
        ar_coeffs,
        ar_shift: 7,
        scaling_shift: 11,
        points: vec![(24, 35), (120, 110)],
        chroma: None,
    }
}

/// The start and end ticks of each `E` line.
fn entry_spans(text: &str) -> Vec<(u64, u64)> {
    text.lines()
        .filter(|line| line.starts_with('E'))
        .map(|line| {
            let words: Vec<u64> = line
                .split_whitespace()
                .skip(1)
                .take(2)
                .map(|word| word.parse().expect("a whole number"))
                .collect();

            (words[0], words[1])
        })
        .collect()
}

#[test]
fn a_fresh_scene_gets_one_entry_per_frame() {
    let scene = steady_scene(0, 2);

    let entries = fit_measured(&[scene]);

    assert_eq!(entries.len(), 2 * CHUNK_FRAMES as usize);
    for (index, entry) in entries.iter().enumerate() {
        assert_eq!(entry.first_frame, index as u64);
        assert_eq!(entry.last_frame, index as u64);
    }
}

#[test]
fn fresh_entries_take_the_listed_seed_of_their_frame() {
    let first = steady_scene(0, 1);
    let second = steady_scene(200, 1);

    let entries = fit_measured(&[first, second]);

    assert_eq!(entries.len(), 2 * CHUNK_FRAMES as usize);
    assert_eq!(entries[CHUNK_FRAMES as usize].first_frame, 200);
    for entry in &entries {
        let expected = fresh_seed(entry.first_frame);
        assert_eq!(entry.seed, expected, "frame {}", entry.first_frame);
    }
}

#[test]
fn frames_of_one_segment_share_weights_and_differ_only_by_seed_scaling() {
    let scene = steady_scene(0, 2);

    let entries = fit_measured(&[scene]);
    let first = &entries[0];
    let first_lumas: Vec<u8> = first.points.iter().map(|point| point.0).collect();

    assert_eq!(entries.len(), 2 * CHUNK_FRAMES as usize);
    for entry in &entries {
        let lumas: Vec<u8> = entry.points.iter().map(|point| point.0).collect();
        assert_eq!(entry.ar_coeffs, first.ar_coeffs);
        assert_eq!(entry.ar_shift, first.ar_shift);
        assert_eq!(lumas, first_lumas);
    }
}

#[test]
fn scaling_follows_the_seeds_template_strength() {
    let scene = steady_scene(0, 2);

    let entries = fit_measured(&[scene]);
    let strength =
        |entry: &FittedEntry| seed_sigma(&entry.ar_coeffs, entry.ar_shift, TemplatePlane::Luma, entry.seed);

    // The weak seed's values scale down onto the strong seed's, so both roundings fit in one code.
    let first = &entries[0];
    let second = entries
        .iter()
        .find(|entry| entry.scaling_shift == first.scaling_shift && strength(entry) != strength(first))
        .expect("a second entry at the same scaling shift");
    let (strong, weak) = if strength(first) > strength(second) {
        (first, second)
    } else {
        (second, first)
    };
    let ratio = strength(weak) / strength(strong);

    for (&(_, strong_value), &(_, weak_value)) in strong.points.iter().zip(&weak.points) {
        let expected = weak_value as f64 * ratio;
        let gap = (strong_value as f64 - expected).abs();
        assert!(gap <= 1.0, "{strong_value} against {expected}");
    }
}

#[test]
fn a_long_fresh_stretch_reads_at_most_135_strengths_per_plane() {
    let chunk_count = 1000usize.div_ceil(CHUNK_FRAMES as usize);
    let scene = steady_scene(0, chunk_count);
    let options = FitOptions::default();
    let mut sigmas = SeedSigmas::new();

    let entries = fit_scenes_with(&[&scene], &options, &mut sigmas);
    let first = &entries[0];

    assert!(entries.len() >= 1000);
    for entry in &entries {
        assert_eq!(entry.ar_coeffs, first.ar_coeffs);
        assert_eq!(entry.ar_shift, first.ar_shift);
    }

    assert!(sigmas.len() <= 135, "{}", sigmas.len());
}

#[test]
fn one_frame_entries_do_not_overlap_in_the_table() {
    let entries = [
        one_frame_entry(0, 4),
        one_frame_entry(1, 63504),
        one_frame_entry(2, 64435),
    ];

    let text = format_table(&entries, NTSC);
    let spans = entry_spans(&text);

    assert_eq!(spans.len(), 3);
    for pair in spans.windows(2) {
        assert_eq!(pair[0].1 + 1, pair[1].0);
    }

    let last_end = boundary_ticks(3, NTSC);
    assert_eq!(spans[2].1, last_end);
    assert!(text.contains("E 0 208541 1 4 1\n"), "{text}");
}

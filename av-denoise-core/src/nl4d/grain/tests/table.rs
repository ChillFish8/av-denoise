use super::synthetic::chunk_at;
use crate::nl4d::grain::consts::AR_COEFFS;
use crate::nl4d::grain::segment::{FittedEntry, SceneGrain};
use crate::nl4d::grain::table::{boundary_ticks, build_table, format_table};

const NTSC: (u64, u64) = (24_000, 1_001);

fn entry(first_frame: u64, last_frame: u64) -> FittedEntry {
    let mut ar_coeffs = [0i32; AR_COEFFS];
    ar_coeffs[23] = 71;

    FittedEntry {
        first_frame,
        last_frame,
        ar_coeffs,
        ar_shift: 7,
        scaling_shift: 11,
        points: vec![(24, 35), (120, 110)],
    }
}

fn entry_lines(text: &str) -> Vec<&str> {
    text.lines().filter(|line| line.starts_with('E')).collect()
}

#[test]
fn boundaries_sit_half_a_frame_early() {
    assert_eq!(boundary_ticks(0, NTSC), 0);
    assert_eq!(boundary_ticks(24, NTSC), 9_801_458);
    assert_eq!(boundary_ticks(1, NTSC), 208_542);
}

#[test]
fn entries_end_one_tick_before_the_next_start() {
    let text = format_table(&[entry(0, 23), entry(24, 47)], NTSC);
    let starts = entry_lines(&text);

    assert_eq!(starts[0], "E 0 9801457 1 7391 1");
    assert!(starts[1].starts_with("E 9801458 "));
    assert!(starts[1].ends_with(" 1 8628 1"));
}

#[test]
fn the_last_entry_ends_half_a_frame_after_its_last_frame() {
    let text = format_table(&[entry(0, 23)], NTSC);
    let first = text.lines().nth(1).expect("an entry line");

    assert_eq!(first, "E 0 9801458 1 7391 1");
}

#[test]
fn an_entry_has_the_exact_format() {
    let text = format_table(&[entry(0, 23)], NTSC);
    let expected = "filmgrn1\n\
        E 0 9801458 1 7391 1\n\
        \tp 3 7 0 11 0 1 128 192 256 128 192 256\n\
        \tsY 2  24 35 120 110\n\
        \tsCb 2 0 0 255 0\n\
        \tsCr 2 0 0 255 0\n\
        \tcY 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 71\n\
        \tcCb 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n\
        \tcCr 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n";

    assert_eq!(text, expected);
}

#[test]
fn a_gap_between_entries_is_left_empty() {
    let text = format_table(&[entry(0, 23), entry(48, 71)], NTSC);
    let starts = entry_lines(&text);

    assert_eq!(starts[0], "E 0 9801458 1 7391 1");
    assert!(starts[1].starts_with("E 19811458 "));
}

#[test]
fn the_54th_entry_seed_wraps_at_16_bits() {
    let entries: Vec<FittedEntry> = (0..54).map(|index| entry(index * 2, index * 2)).collect();
    let text = format_table(&entries, NTSC);
    let starts = entry_lines(&text);

    assert!(starts[53].ends_with(" 1 7416 1"));
}

#[test]
fn no_entries_writes_header_only() {
    let text = build_table(&[], NTSC);

    assert_eq!(text, "filmgrn1\n");
}

#[test]
fn reruns_are_byte_identical() {
    let first = format_table(&[entry(0, 23), entry(24, 47)], NTSC);
    let second = format_table(&[entry(0, 23), entry(24, 47)], NTSC);

    assert_eq!(first, second);
}

#[test]
fn scenes_are_sorted_before_fitting() {
    let scenes = vec![
        SceneGrain {
            first_frame: 24,
            chunks: vec![chunk_at(2.0, 300, None)],
        },
        SceneGrain {
            first_frame: 0,
            chunks: vec![chunk_at(2.0, 300, None)],
        },
    ];

    let text = build_table(&scenes, NTSC);
    let starts = entry_lines(&text);

    assert_eq!(starts.len(), 2);
    assert!(starts[0].starts_with("E 0 "));
    assert!(starts[1].starts_with("E 9801458 "));
}

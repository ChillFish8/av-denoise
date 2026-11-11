use super::synthetic::{chroma_chunk, chroma_colour_frame, chunk_at, table_chunk};
use crate::nl4d::grain::chunk::GrainChunk;
use crate::nl4d::grain::consts::AR_COEFFS;
use crate::nl4d::grain::segment::{FittedEntry, PlaneEntry, SceneGrain};
use crate::nl4d::grain::table::{FitOptions, boundary_ticks, build_table, format_table};

const NTSC: (u64, u64) = (24_000, 1_001);

/// A scene whose chroma chunks repeat its luma tables for both planes, with `colour_frame` in every
/// frame.
fn scene_with_chroma(first_frame: u64, chunks: Vec<GrainChunk>, colour_frame: &[f32]) -> SceneGrain {
    let chroma_chunks = chunks
        .iter()
        .map(|chunk| chroma_chunk(chunk, chunk, colour_frame))
        .collect();

    SceneGrain {
        first_frame,
        chunks,
        chroma_chunks,
    }
}

/// Three scenes with drift, kept grain, a thin chunk and a sparse scene, each with chroma.
fn measured_scenes() -> Vec<SceneGrain> {
    let colour = chroma_colour_frame(1.5, 1.0);

    let steady = chunk_at(2.0, 300, None);
    let drifted = chunk_at(2.9, 300, Some(1.2));
    let thin = table_chunk();
    let first_chunks = vec![steady, drifted, thin];
    let first = scene_with_chroma(0, first_chunks, &colour);

    let partly_kept = chunk_at(1.5, 300, Some(0.5));
    let unkept = chunk_at(1.6, 300, None);
    let second_chunks = vec![partly_kept, unkept];
    let second = scene_with_chroma(72, second_chunks, &colour);

    let sparse = table_chunk();
    let third = scene_with_chroma(120, vec![sparse], &colour);

    vec![first, second, third]
}

/// The table of [measured_scenes] built at 35c1c8b with the default options.
const HEAD_TABLE: &str = "filmgrn1\n\
    E 0 9801457 1 7391 1\n\
    \tp 3 8 0 11 0 1 128 192 256 128 192 256\n\
    \tsY 8  72 119 88 119 104 119 120 119 136 119 152 119 168 119 184 119\n\
    \tsCb 2 0 0 255 0\n\
    \tsCr 2 0 0 255 0\n\
    \tcY 2 1 1 3 1 1 3 2 2 3 2 2 4 3 1 2 3 2 2 2 4 1 3 106\n\
    \tcCb 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n\
    \tcCr 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n\
    E 9801458 29821457 1 8628 1\n\
    \tp 3 8 0 11 0 1 128 192 256 128 192 256\n\
    \tsY 8  72 163 88 163 104 163 120 163 136 163 152 163 168 163 184 163\n\
    \tsCb 2 0 0 255 0\n\
    \tsCr 2 0 0 255 0\n\
    \tcY 2 1 1 3 1 1 3 2 2 3 2 2 4 3 1 2 3 2 2 2 4 1 3 106\n\
    \tcCb 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n\
    \tcCr 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n\
    E 29821458 49841457 1 9865 1\n\
    \tp 3 8 0 11 0 1 128 192 256 128 192 256\n\
    \tsY 8  72 88 88 88 104 88 120 88 136 88 152 88 168 88 184 88\n\
    \tsCb 2 0 0 255 0\n\
    \tsCr 2 0 0 255 0\n\
    \tcY 2 1 1 3 1 1 3 2 2 3 2 2 4 3 1 2 3 2 2 2 4 1 3 106\n\
    \tcCb 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n\
    \tcCr 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n\
    E 49841458 59851458 1 11102 1\n\
    \tp 3 8 0 11 0 1 128 192 256 128 192 256\n\
    \tsY 8  72 88 88 88 104 88 120 88 136 88 152 88 168 88 184 88\n\
    \tsCb 2 0 0 255 0\n\
    \tsCr 2 0 0 255 0\n\
    \tcY 2 1 1 3 1 1 3 2 2 3 2 2 4 3 1 2 3 2 2 2 4 1 3 106\n\
    \tcCb 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n\
    \tcCr 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n";

/// The table of [measured_scenes] built at 35c1c8b with half the fine noise taken out.
const HEAD_FINE_NOISE_TABLE: &str = "filmgrn1\n\
    E 0 9801457 1 7391 1\n\
    \tp 3 7 0 11 0 1 128 192 256 128 192 256\n\
    \tsY 8  72 100 88 100 104 100 120 100 136 100 152 100 168 100 184 100\n\
    \tsCb 2 0 0 255 0\n\
    \tsCr 2 0 0 255 0\n\
    \tcY 1 0 0 2 0 0 2 1 1 2 1 1 2 2 1 1 2 1 1 0 3 1 -8 71\n\
    \tcCb 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n\
    \tcCr 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n\
    E 9801458 29821457 1 8628 1\n\
    \tp 3 7 0 11 0 1 128 192 256 128 192 256\n\
    \tsY 8  72 137 88 137 104 137 120 137 136 137 152 137 168 137 184 137\n\
    \tsCb 2 0 0 255 0\n\
    \tsCr 2 0 0 255 0\n\
    \tcY 1 0 0 2 0 0 2 1 1 2 1 1 2 2 1 1 2 1 1 0 3 1 -8 71\n\
    \tcCb 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n\
    \tcCr 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n\
    E 29821458 49841457 1 9865 1\n\
    \tp 3 7 0 11 0 1 128 192 256 128 192 256\n\
    \tsY 8  72 74 88 74 104 74 120 74 136 74 152 74 168 74 184 74\n\
    \tsCb 2 0 0 255 0\n\
    \tsCr 2 0 0 255 0\n\
    \tcY 1 0 0 2 0 0 2 1 1 2 1 1 2 2 1 1 2 1 1 0 3 1 -8 71\n\
    \tcCb 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n\
    \tcCr 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n\
    E 49841458 59851458 1 11102 1\n\
    \tp 3 7 0 11 0 1 128 192 256 128 192 256\n\
    \tsY 8  72 74 88 74 104 74 120 74 136 74 152 74 168 74 184 74\n\
    \tsCb 2 0 0 255 0\n\
    \tsCr 2 0 0 255 0\n\
    \tcY 1 0 0 2 0 0 2 1 1 2 1 1 2 2 1 1 2 1 1 0 3 1 -8 71\n\
    \tcCb 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n\
    \tcCr 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -128\n";

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
        chroma: None,
    }
}

fn entry_lines(text: &str) -> Vec<&str> {
    text.lines().filter(|line| line.starts_with('E')).collect()
}

/// The lines of `text` that start with the tab and `prefix`, without the tab.
fn lines_named<'a>(text: &'a str, prefix: &str) -> Vec<&'a str> {
    text.lines()
        .filter_map(|line| line.strip_prefix('\t'))
        .filter(|line| line.starts_with(prefix))
        .collect()
}

/// The point count of a scaling line such as `sCb 2 0 0 255 0`.
fn point_count(line: &str) -> usize {
    let count = line.split_whitespace().nth(1).expect("a point count");

    count.parse().expect("a whole number")
}

fn chroma_options() -> FitOptions {
    let options = FitOptions::default();

    options.with_chroma(true)
}

#[test]
fn chroma_off_tables_are_byte_identical() {
    let scenes = measured_scenes();
    let default_options = FitOptions::default();
    let fine_noise_options = FitOptions::with_fine_noise(0.5);

    let table = build_table(&scenes, NTSC, &default_options);
    let fine_noise_table = build_table(&scenes, NTSC, &fine_noise_options);

    assert_eq!(table, HEAD_TABLE);
    assert_eq!(fine_noise_table, HEAD_FINE_NOISE_TABLE);
}

#[test]
fn chroma_on_entries_carry_both_planes() {
    let scenes = measured_scenes();
    let options = chroma_options();

    let table = build_table(&scenes, NTSC, &options);
    let entries = entry_lines(&table);

    assert_eq!(entries.len(), 4);
    for prefix in ["sCb ", "sCr "] {
        let lines = lines_named(&table, prefix);
        assert_eq!(lines.len(), entries.len());
        for line in lines {
            let count = point_count(line);
            assert!((3..=10).contains(&count), "{line}");
        }
    }

    for prefix in ["cCb ", "cCr "] {
        let lines = lines_named(&table, prefix);
        assert_eq!(lines.len(), entries.len());
        for line in lines {
            assert!(line.ends_with(" 0"), "{line}");
            assert_eq!(line.split_whitespace().count(), 1 + AR_COEFFS + 1);
        }
    }
}

#[test]
fn chroma_entries_follow_luma_segments() {
    let scenes = measured_scenes();
    let off_options = FitOptions::default();
    let on_options = chroma_options();

    let off_table = build_table(&scenes, NTSC, &off_options);
    let on_table = build_table(&scenes, NTSC, &on_options);

    assert_eq!(entry_lines(&on_table), entry_lines(&off_table));
}

#[test]
fn a_scene_without_any_chroma_data_zeros_both_planes() {
    let mut scenes = measured_scenes();
    for scene in &mut scenes {
        scene.chroma_chunks.clear();
    }

    let on_options = chroma_options();
    let off_options = FitOptions::default();
    let on_table = build_table(&scenes, NTSC, &on_options);
    let off_table = build_table(&scenes, NTSC, &off_options);

    for prefix in ["sCb ", "sCr "] {
        let lines = lines_named(&on_table, prefix);
        assert_eq!(lines.len(), 4);
        assert!(
            lines.iter().all(|line| line.ends_with(" 2 0 0 255 0")),
            "{lines:?}"
        );
    }

    assert_eq!(on_table, off_table);
}

#[test]
fn the_luma_coefficient_of_chroma_is_zero() {
    let mut chroma_entry = entry(0, 23);
    let mut cb_coeffs = [0i32; AR_COEFFS];
    cb_coeffs[23] = 40;
    let cb = PlaneEntry {
        ar_coeffs: cb_coeffs,
        points: vec![(40, 20), (120, 60), (200, 30)],
    };
    let cr = PlaneEntry {
        ar_coeffs: [-3; AR_COEFFS],
        points: vec![(40, 10), (120, 30), (200, 15)],
    };
    chroma_entry.chroma = Some([cb, cr]);

    let text = format_table(&[chroma_entry], NTSC);
    let cb_lines = lines_named(&text, "sCb ");
    let cr_lines = lines_named(&text, "sCr ");
    let cb_coeff_lines = lines_named(&text, "cCb ");
    let cr_coeff_lines = lines_named(&text, "cCr ");

    assert_eq!(cb_lines, ["sCb 3 40 20 120 60 200 30"]);
    assert_eq!(cr_lines, ["sCr 3 40 10 120 30 200 15"]);
    assert_eq!(
        cb_coeff_lines,
        ["cCb 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 40 0"]
    );
    assert_eq!(
        cr_coeff_lines,
        ["cCr -3 -3 -3 -3 -3 -3 -3 -3 -3 -3 -3 -3 -3 -3 -3 -3 -3 -3 -3 -3 -3 -3 -3 -3 0"]
    );
}

#[test]
fn boundaries_sit_half_a_frame_early() {
    let frame_0_ticks = boundary_ticks(0, NTSC);
    let frame_24_ticks = boundary_ticks(24, NTSC);
    let frame_1_ticks = boundary_ticks(1, NTSC);

    assert_eq!(frame_0_ticks, 0);
    assert_eq!(frame_24_ticks, 9_801_458);
    assert_eq!(frame_1_ticks, 208_542);
}

#[test]
fn entries_end_one_tick_before_the_next_start() {
    let entries = [entry(0, 23), entry(24, 47)];
    let text = format_table(&entries, NTSC);
    let starts = entry_lines(&text);

    assert_eq!(starts[0], "E 0 9801457 1 7391 1");
    assert!(starts[1].starts_with("E 9801458 "));
    assert!(starts[1].ends_with(" 1 8628 1"));
}

#[test]
fn the_last_entry_ends_half_a_frame_after_its_last_frame() {
    let entries = [entry(0, 23)];
    let text = format_table(&entries, NTSC);
    let first = text.lines().nth(1).expect("an entry line");

    assert_eq!(first, "E 0 9801458 1 7391 1");
}

#[test]
fn an_entry_has_the_exact_format() {
    let entries = [entry(0, 23)];
    let text = format_table(&entries, NTSC);
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
    let entries = [entry(0, 23), entry(48, 71)];
    let text = format_table(&entries, NTSC);
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
    let options = FitOptions::default();
    let text = build_table(&[], NTSC, &options);

    assert_eq!(text, "filmgrn1\n");
}

#[test]
fn reruns_are_byte_identical() {
    let first_entries = [entry(0, 23), entry(24, 47)];
    let second_entries = [entry(0, 23), entry(24, 47)];
    let first = format_table(&first_entries, NTSC);
    let second = format_table(&second_entries, NTSC);

    assert_eq!(first, second);
}

#[test]
fn scenes_are_sorted_before_fitting() {
    let later_chunk = chunk_at(2.0, 300, None);
    let earlier_chunk = chunk_at(2.0, 300, None);
    let later = SceneGrain {
        first_frame: 24,
        chunks: vec![later_chunk],
        chroma_chunks: Vec::new(),
    };
    let earlier = SceneGrain {
        first_frame: 0,
        chunks: vec![earlier_chunk],
        chroma_chunks: Vec::new(),
    };
    let scenes = vec![later, earlier];
    let options = FitOptions::default();

    let text = build_table(&scenes, NTSC, &options);
    let starts = entry_lines(&text);

    assert_eq!(starts.len(), 2);
    assert!(starts[0].starts_with("E 0 "));
    assert!(starts[1].starts_with("E 9801458 "));
}

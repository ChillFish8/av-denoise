use super::synthetic::{
    FRESH_GRAIN,
    FROZEN_GRAIN,
    ar_field,
    autocov_of,
    chroma_chunk,
    chroma_options,
    chunk_at,
    chunk_with_record,
    clear_pixels,
    entry_count,
    fit_measured,
    fresh_runs,
    record,
};
use crate::nl4d::grain::chunk::GrainChunk;
use crate::nl4d::grain::consts::{AR_COEFFS, AR_OFFSETS, CHUNK_FRAMES};
use crate::nl4d::grain::modes::{GrainMode, GrainRun, StaticGrain, grain_runs};
use crate::nl4d::grain::repeat::SeedScores;
use crate::nl4d::grain::seeds::{FGSE_SEEDS, SeedSigmas, fresh_seed};
use crate::nl4d::grain::segment::{FittedEntry, SceneGrain, fit_scenes, fit_scenes_with};
use crate::nl4d::grain::table::{FitOptions, boundary_ticks, build_table, format_table};
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

/// `chunks` with every frame's grain persisting as `grain` does.
fn persisting(chunks: Vec<GrainChunk>, grain: (f64, f64)) -> Vec<GrainChunk> {
    let (lag_one, lag_eight) = grain;
    let single = record(lag_one, lag_eight);

    chunks
        .into_iter()
        .map(|mut chunk| {
            chunk.frame_records = single.repeat(chunk.frames as usize);
            chunk
        })
        .collect()
}

/// `count` chunks of grain at `std_codes`.
fn chunks_at(std_codes: f32, count: usize) -> Vec<GrainChunk> {
    (0..count).map(|_| chunk_at(std_codes, 300, None)).collect()
}

/// `count` chunks with frame counts and no table statistics.
fn unmeasured_chunks(count: usize) -> Vec<GrainChunk> {
    (0..count)
        .map(|_| {
            let mut chunk = GrainChunk::empty();
            chunk.frames = CHUNK_FRAMES;
            chunk
        })
        .collect()
}

/// A chunk of grain at 2 codes whose texture leans on the pixel above rather than the left.
fn upward_chunk() -> GrainChunk {
    let mut weights = [0.0f64; AR_COEFFS];
    let up = AR_OFFSETS
        .iter()
        .position(|&offset| offset == (-1, 0))
        .expect("up offset");
    weights[up] = 0.4;

    let field = ar_field(&weights, 300, 300, 9);
    let record = autocov_of(&field, 300, 300);

    chunk_with_record(2.0, 300, None, &record)
}

/// Back-to-back scenes from frame 0, one per entry of `scene_chunks`.
fn back_to_back(scene_chunks: Vec<Vec<GrainChunk>>) -> Vec<SceneGrain> {
    let mut first_frame = 0;
    let mut scenes = Vec::new();
    for chunks in scene_chunks {
        let frames: u64 = chunks.iter().map(|chunk| chunk.frames as u64).sum();
        scenes.push(SceneGrain {
            first_frame,
            chunks,
            chroma_chunks: Vec::new(),
        });
        first_frame += frames;
    }

    scenes
}

/// Two frozen scenes, the first splitting into a segment at 2 codes and one at 3 codes.
fn two_strength_scenes() -> Vec<SceneGrain> {
    let mut first_chunks = chunks_at(2.0, 2);
    first_chunks.push(chunk_at(3.0, 300, None));
    let first = persisting(first_chunks, FROZEN_GRAIN);

    let second_chunks = chunks_at(2.0, 2);
    let second = persisting(second_chunks, FROZEN_GRAIN);

    back_to_back(vec![first, second])
}

/// Frozen, fresh and frozen scenes of 12 chunks each, built by `chunks`.
fn frozen_fresh_frozen(chunks: fn(usize) -> Vec<GrainChunk>) -> Vec<SceneGrain> {
    let first = persisting(chunks(12), FROZEN_GRAIN);
    let middle = persisting(chunks(12), FRESH_GRAIN);
    let last = persisting(chunks(12), FROZEN_GRAIN);

    back_to_back(vec![first, middle, last])
}

fn steady_chunks(count: usize) -> Vec<GrainChunk> {
    chunks_at(2.0, count)
}

fn run(first_frame: u64, last_frame: u64, mode: GrainMode) -> GrainRun {
    GrainRun {
        first_frame,
        last_frame,
        mode,
    }
}

/// A chroma chunk with `chunk`'s statistics on both planes.
fn chroma_like(chunk: &GrainChunk) -> GrainChunk {
    chroma_chunk(chunk, chunk)
}

/// A scene's luma chunks, persisting as frozen grain, and chroma chunks with the same statistics.
fn frozen_with_chroma(chunks: Vec<GrainChunk>) -> (Vec<GrainChunk>, Vec<GrainChunk>) {
    let chroma = chunks.iter().map(chroma_like).collect();
    let luma = persisting(chunks, FROZEN_GRAIN);

    (luma, chroma)
}

/// Fits the scenes with the runs `options` chooses for them.
fn fit_with_runs(scenes: &[SceneGrain], options: &FitOptions) -> Vec<FittedEntry> {
    let runs = grain_runs(scenes, options).expect("the declarations should be valid");
    let ordered: Vec<&SceneGrain> = scenes.iter().collect();

    fit_scenes(&ordered, options, &runs)
}

fn fit_detected(scenes: &[SceneGrain]) -> Vec<FittedEntry> {
    let options = FitOptions::default();

    fit_with_runs(scenes, &options)
}

fn frames_of(entries: &[FittedEntry]) -> Vec<(u64, u64)> {
    entries
        .iter()
        .map(|entry| (entry.first_frame, entry.last_frame))
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
    let runs = fresh_runs(std::slice::from_ref(&scene));
    let mut sigmas = SeedSigmas::new();

    let entries = fit_scenes_with(&[&scene], &options, &runs, &mut sigmas);
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

#[test]
fn a_frozen_run_holds_one_seed_across_scenes_and_segments() {
    let scenes = two_strength_scenes();

    let entries = fit_detected(&scenes);

    assert_eq!(frames_of(&entries), [(0, 47), (48, 71), (72, 119)]);
    let seed = entries[0].seed;
    assert!(FGSE_SEEDS.contains(&seed), "{seed}");
    for entry in &entries {
        assert_eq!(entry.seed, seed);
    }
}

#[test]
fn a_frozen_run_holds_one_texture() {
    let first = persisting(steady_chunks(2), FROZEN_GRAIN);
    let upward = vec![upward_chunk(), upward_chunk()];
    let second = persisting(upward, FROZEN_GRAIN);
    let scenes = back_to_back(vec![first, second]);
    let off_options = FitOptions::default().with_static_grain(StaticGrain::Off);

    let entries = fit_detected(&scenes);
    let fresh_entries = fit_with_runs(&scenes, &off_options);

    let fresh_first = &fresh_entries[0];
    let fresh_last = fresh_entries.last().expect("fresh entries");
    assert_ne!(fresh_first.ar_coeffs, fresh_last.ar_coeffs);

    assert_eq!(entries.len(), 2);
    for entry in &entries {
        assert_eq!(entry.ar_coeffs, entries[0].ar_coeffs);
        assert_eq!(entry.ar_shift, entries[0].ar_shift);
    }
}

#[test]
fn strength_still_follows_each_segment_in_a_frozen_run() {
    let scenes = two_strength_scenes();

    let entries = fit_detected(&scenes);

    assert_eq!(entries.len(), 3);
    assert_ne!(entries[0].points, entries[1].points);
}

#[test]
fn a_frozen_piece_is_one_entry_per_segment() {
    let chunks = persisting(steady_chunks(3), FROZEN_GRAIN);
    let scenes = back_to_back(vec![chunks]);

    let entries = fit_detected(&scenes);

    assert_eq!(frames_of(&entries), [(0, 71)]);
}

#[test]
fn the_frozen_seed_is_the_calmest_for_the_runs_weights() {
    let chunks = persisting(steady_chunks(1), FROZEN_GRAIN);
    let scenes = back_to_back(vec![chunks]);

    let entries = fit_detected(&scenes);
    let entry = &entries[0];
    let mut scores = SeedScores::new();
    let calmest = scores.calmest(&entry.ar_coeffs, entry.ar_shift, &[]);

    assert_eq!(entries.len(), 1);
    assert_eq!(Some(entry.seed), calmest);
}

#[test]
fn two_frozen_runs_take_different_seeds() {
    let scenes = frozen_fresh_frozen(steady_chunks);

    let entries = fit_detected(&scenes);
    let first = &entries[0];
    let last = entries.last().expect("entries");

    assert_eq!((first.first_frame, first.last_frame), (0, 287));
    assert_eq!((last.first_frame, last.last_frame), (576, 863));
    assert_eq!(first.ar_coeffs, last.ar_coeffs);
    assert_ne!(first.seed, last.seed);
}

#[test]
fn a_segment_straddling_a_run_edge_is_cut_there() {
    let scenes = back_to_back(vec![steady_chunks(3)]);
    let range = 24..=47;
    let declared = vec![range];
    let options = FitOptions::default()
        .with_static_grain(StaticGrain::Off)
        .with_static_frames(declared);

    let entries = fit_with_runs(&scenes, &options);

    let mut expected: Vec<(u64, u64)> = (0..24).map(|frame| (frame, frame)).collect();
    expected.push((24, 47));
    expected.extend((48..72).map(|frame| (frame, frame)));
    assert_eq!(frames_of(&entries), expected);
}

#[test]
fn a_frozen_run_without_its_own_texture_borrows_one() {
    let measured = steady_chunks(1);
    let mut cleared = steady_chunks(2);
    cleared.iter_mut().for_each(clear_pixels);
    let scenes = back_to_back(vec![measured, cleared]);
    let range = 1..=1;
    let declared = vec![range];
    let options = FitOptions::default()
        .with_static_grain(StaticGrain::Off)
        .with_static_scenes(declared);

    let entries = fit_with_runs(&scenes, &options);
    let neighbour = &entries[0];
    let frozen = entries.last().expect("entries");

    assert_eq!(entries.len(), 25);
    assert_eq!((frozen.first_frame, frozen.last_frame), (24, 71));
    assert_eq!(frozen.ar_coeffs, neighbour.ar_coeffs);
    assert_eq!(frozen.ar_shift, neighbour.ar_shift);
}

#[test]
fn a_frozen_run_with_nothing_to_fit_writes_no_entries() {
    let chunks = persisting(unmeasured_chunks(3), FROZEN_GRAIN);
    let scenes = back_to_back(vec![chunks]);
    let options = FitOptions::default().with_static_grain(StaticGrain::All);

    let table = build_table(&scenes, NTSC, &options).expect("a table");

    assert_eq!(table.text, "filmgrn1\n");
    assert_eq!(table.runs, [run(0, 71, GrainMode::Frozen)]);
}

#[test]
fn build_table_reports_the_runs() {
    let scenes = frozen_fresh_frozen(unmeasured_chunks);
    let options = FitOptions::default();

    let table = build_table(&scenes, NTSC, &options).expect("a table");
    let modes: Vec<GrainMode> = table.runs.iter().map(|run| run.mode).collect();

    assert_eq!(modes, [GrainMode::Frozen, GrainMode::Fresh, GrainMode::Frozen]);
}

#[test]
fn build_table_refuses_a_range_past_the_end() {
    let scenes = back_to_back(vec![unmeasured_chunks(3)]);
    let range = 60..=72;
    let declared = vec![range];
    let options = FitOptions::default().with_static_frames(declared);

    let table = build_table(&scenes, NTSC, &options);

    assert!(table.is_err());
}

#[test]
fn reruns_with_frozen_grain_are_byte_identical() {
    let scenes = two_strength_scenes();
    let options = FitOptions::default();

    let first = build_table(&scenes, NTSC, &options).expect("a table");
    let second = build_table(&scenes, NTSC, &options).expect("a table");

    assert!(first.text.contains("\nE "), "{}", first.text);
    assert_eq!(first, second);
}

#[test]
fn a_frozen_run_without_textures_borrows_one_for_all_its_entries() {
    let mut cleared = steady_chunks(2);
    cleared.iter_mut().for_each(clear_pixels);
    let [first_cleared, second_cleared] = <[GrainChunk; 2]>::try_from(cleared).expect("two chunks");
    let scene_chunks = vec![
        steady_chunks(1),
        vec![first_cleared],
        vec![second_cleared],
        vec![upward_chunk()],
    ];
    let scenes = back_to_back(scene_chunks);
    let range = 1..=2;
    let declared = vec![range];
    let options = FitOptions::default()
        .with_static_grain(StaticGrain::Off)
        .with_static_scenes(declared);

    let entries = fit_with_runs(&scenes, &options);
    let left_neighbour = &entries[0];
    let right_neighbour = entries.last().expect("entries");
    let frozen = &entries[24..26];

    assert_eq!(entries.len(), 50);
    assert_ne!(left_neighbour.ar_coeffs, right_neighbour.ar_coeffs);
    assert_eq!((frozen[0].first_frame, frozen[0].last_frame), (24, 47));
    assert_eq!((frozen[1].first_frame, frozen[1].last_frame), (48, 71));
    for entry in frozen {
        assert_eq!(entry.ar_coeffs, left_neighbour.ar_coeffs);
        assert_eq!(entry.ar_shift, left_neighbour.ar_shift);
    }
}

#[test]
fn a_frozen_run_holds_one_chroma_texture() {
    let (first_luma, first_chroma) = frozen_with_chroma(steady_chunks(1));
    let (second_luma, second_chroma) = frozen_with_chroma(vec![upward_chunk()]);
    let mut scenes = back_to_back(vec![first_luma, second_luma]);
    scenes[0].chroma_chunks = first_chroma;
    scenes[1].chroma_chunks = second_chroma;
    let frozen_options = chroma_options();
    let fresh_options = chroma_options().with_static_grain(StaticGrain::Off);

    let entries = fit_with_runs(&scenes, &frozen_options);
    let fresh_entries = fit_with_runs(&scenes, &fresh_options);

    let planes_of = |entry: &FittedEntry| entry.chroma.clone().expect("chroma grain");
    let fresh_first = planes_of(&fresh_entries[0]);
    let fresh_last = planes_of(fresh_entries.last().expect("fresh entries"));
    assert_ne!(fresh_first[0].ar_coeffs, fresh_last[0].ar_coeffs);
    assert_ne!(fresh_first[1].ar_coeffs, fresh_last[1].ar_coeffs);

    assert_eq!(entries.len(), 2);
    let first_planes = planes_of(&entries[0]);
    let second_planes = planes_of(&entries[1]);
    assert_eq!(first_planes[0].ar_coeffs, second_planes[0].ar_coeffs);
    assert_eq!(first_planes[1].ar_coeffs, second_planes[1].ar_coeffs);
}

#[test]
fn a_table_without_persistence_is_one_fresh_run_over_every_frame() {
    let scenes = back_to_back(vec![steady_chunks(2)]);
    let options = FitOptions::default();

    let table = build_table(&scenes, NTSC, &options).expect("a table");

    assert_eq!(table.runs, [run(0, 47, GrainMode::Fresh)]);
    assert_eq!(entry_count(&table.text), 48);
}

#[test]
fn static_grain_without_persistence_freezes_every_frame() {
    let scenes = back_to_back(vec![steady_chunks(2)]);
    let options = FitOptions::default().with_static_grain(StaticGrain::All);

    let table = build_table(&scenes, NTSC, &options).expect("a table");

    assert_eq!(table.runs, [run(0, 47, GrainMode::Frozen)]);
    assert_eq!(entry_count(&table.text), 1);
}

#[test]
fn declared_frames_without_persistence_reach_the_last_chunk_frame() {
    let scenes = back_to_back(vec![steady_chunks(2)]);
    let range = 40..=47;
    let declared = vec![range];
    let options = FitOptions::default().with_static_frames(declared);

    let table = build_table(&scenes, NTSC, &options).expect("a table");

    let expected = [run(0, 39, GrainMode::Fresh), run(40, 47, GrainMode::Frozen)];
    assert_eq!(table.runs, expected);
    assert_eq!(entry_count(&table.text), 41);
}

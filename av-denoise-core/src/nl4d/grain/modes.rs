use std::ops::RangeInclusive;

use anyhow::bail;

use super::report::{FrameGrain, frame_rows};
use super::segment::SceneGrain;
use super::table::FitOptions;

/// Frames in the median that smooths the 1-frame persistence.
const SMOOTHING_WINDOW: usize = 48;
/// The smoothed 1-frame persistence below which a frame's grain is fresh.
const FRESH_BELOW: f64 = 0.30;
/// Fewest frames for a run between two runs of the other kind to stay.
const SHORTEST_RUN: usize = 24;
/// The 8-frame over 1-frame persistence at or above which a held run is frozen.
const FROZEN_RATIO: f64 = 0.47;
/// The furthest a change of mode moves to reach a scene cut, in frames.
const CUT_REACH: u64 = 24;

/// How the grain's frozen frames are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StaticGrain {
    /// Static grain is detected from the measurements.
    #[default]
    Detect,
    /// No grain is static unless declared.
    Off,
    /// The whole clip is static.
    All,
}

/// Whether a frame's grain pattern changes every frame or holds still.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrainMode {
    /// A new pattern every frame.
    Fresh,
    /// One pattern held over the whole run.
    Frozen,
}

/// A stretch of neighbouring frames with one grain mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrainRun {
    pub first_frame: u64,
    /// The run's last frame, included in the run.
    pub last_frame: u64,
    pub mode: GrainMode,
}

/// Neighbouring rows of one kind, `start..end` in row positions.
#[derive(Debug, Clone, Copy)]
struct Span {
    start: usize,
    end: usize,
    marked: bool,
}

/// The runs of fresh and frozen frames over every frame of every scene, in frame order.
///
/// Frames are counted from each chunk's frame count, so frames without persistence readings are
/// covered too and declared ranges are checked against them. Declared frames and scenes are frozen
/// on top of what `options.static_grain` chooses.
pub(crate) fn grain_runs(
    scenes: &[SceneGrain],
    options: &FitOptions,
) -> Result<Vec<GrainRun>, anyhow::Error> {
    let rows = frame_rows(scenes);
    check_declarations(&rows, scenes.len(), options)?;

    if rows.is_empty() {
        return Ok(Vec::new());
    }

    let mut frozen = match options.static_grain {
        StaticGrain::All => vec![true; rows.len()],
        StaticGrain::Off => vec![false; rows.len()],
        StaticGrain::Detect => detected_frozen(&rows),
    };

    for (row, is_frozen) in rows.iter().zip(frozen.iter_mut()) {
        if is_declared(row, options) {
            *is_frozen = true;
        }
    }

    let spans = spans_of(&frozen);
    let runs = spans.iter().map(|span| run_of(&rows, span)).collect();

    Ok(runs)
}

fn check_declarations(
    rows: &[FrameGrain],
    scene_count: usize,
    options: &FitOptions,
) -> Result<(), anyhow::Error> {
    let last_frame = rows.last().map(|row| row.frame);
    let last_scene = scene_count.checked_sub(1).map(|last| last as u64);

    check_ranges("--static-frames", "frame", &options.static_frames, last_frame)?;
    check_ranges("--static-scenes", "scene", &options.static_scenes, last_scene)
}

fn check_ranges(
    flag: &str,
    unit: &str,
    ranges: &[RangeInclusive<u64>],
    last: Option<u64>,
) -> Result<(), anyhow::Error> {
    for range in ranges {
        let start = *range.start();
        let end = *range.end();
        if end < start {
            bail!("{flag} {start}:{end} ends before it starts");
        }

        let Some(last) = last else {
            bail!("{flag} {start}:{end} is past the end of a clip with no {unit}s");
        };

        if end > last {
            bail!("{flag} {start}:{end} is past the last {unit} {last}");
        }
    }

    Ok(())
}

fn is_declared(row: &FrameGrain, options: &FitOptions) -> bool {
    let frame_declared = options
        .static_frames
        .iter()
        .any(|range| range.contains(&row.frame));
    let scene_declared = options
        .static_scenes
        .iter()
        .any(|range| range.contains(&row.scene));

    frame_declared || scene_declared
}

/// Whether each frame's grain is frozen, read from the persistence.
fn detected_frozen(rows: &[FrameGrain]) -> Vec<bool> {
    let lag_one: Vec<Option<f64>> = rows.iter().map(|row| row.persistence[0]).collect();
    let smoothed = smoothed_lag_one(&lag_one);
    let mut held = held_frames(&smoothed);
    absorb_short_runs(&mut held);

    let mut frozen = vec![false; rows.len()];
    for span in spans_of(&held) {
        if span.marked && is_frozen_run(&rows[span.start..span.end]) {
            frozen[span.start..span.end].fill(true);
        }
    }

    snap_to_cuts(rows, &mut frozen);

    frozen
}

/// Each frame's median of the readable 1-frame persistence within half a window either side.
fn smoothed_lag_one(lag_one: &[Option<f64>]) -> Vec<Option<f64>> {
    let reach = SMOOTHING_WINDOW / 2;

    let mut smoothed = Vec::with_capacity(lag_one.len());
    for index in 0..lag_one.len() {
        let start = index.saturating_sub(reach);
        let end = (index + reach).min(lag_one.len());
        let mut readable: Vec<f64> = lag_one[start..end].iter().flatten().copied().collect();
        let median = median_of(&mut readable);
        smoothed.push(median);
    }

    smoothed
}

/// Whether each frame's grain is held, with unread frames taking the nearest answer.
///
/// The earlier frame wins a tie, and with no answer anywhere every frame is fresh.
pub(super) fn held_frames(smoothed: &[Option<f64>]) -> Vec<bool> {
    let answers: Vec<Option<bool>> = smoothed
        .iter()
        .map(|value| value.map(|persistence| persistence >= FRESH_BELOW))
        .collect();
    let behind = nearest_behind(&answers);
    let ahead = nearest_ahead(&answers);

    let mut held = Vec::with_capacity(answers.len());
    for index in 0..answers.len() {
        let nearest = match (behind[index], ahead[index]) {
            (Some(earlier), Some(later)) if later - index < index - earlier => Some(later),
            (Some(earlier), _) => Some(earlier),
            (None, later) => later,
        };

        let answer = nearest.and_then(|position| answers[position]);
        held.push(answer.unwrap_or(false));
    }

    held
}

/// The position of the nearest answer at or before each frame.
fn nearest_behind(answers: &[Option<bool>]) -> Vec<Option<usize>> {
    let mut nearest = Vec::with_capacity(answers.len());
    let mut last_seen = None;
    for (index, answer) in answers.iter().enumerate() {
        if answer.is_some() {
            last_seen = Some(index);
        }

        nearest.push(last_seen);
    }

    nearest
}

/// The position of the nearest answer at or after each frame.
fn nearest_ahead(answers: &[Option<bool>]) -> Vec<Option<usize>> {
    let mut nearest = vec![None; answers.len()];
    let mut next_seen = None;
    for (index, answer) in answers.iter().enumerate().rev() {
        if answer.is_some() {
            next_seen = Some(index);
        }

        nearest[index] = next_seen;
    }

    nearest
}

/// Flips short unheld runs between held runs, then short held runs between unheld runs.
pub(super) fn absorb_short_runs(held: &mut [bool]) {
    flip_short_runs(held, false);
    flip_short_runs(held, true);
}

/// Flips runs of `kind` shorter than [SHORTEST_RUN] that have a run on both sides.
fn flip_short_runs(held: &mut [bool], kind: bool) {
    let spans = spans_of(held);
    let inner_count = spans.len().saturating_sub(1);

    for span in spans.iter().take(inner_count).skip(1) {
        let length = span.end - span.start;
        if span.marked == kind && length < SHORTEST_RUN {
            held[span.start..span.end].fill(!kind);
        }
    }
}

/// Whether a held run's 8-frame persistence keeps enough of its 1-frame persistence to be frozen.
///
/// A run without a readable median on either lag, or with a 1-frame median of 0, is frozen.
fn is_frozen_run(rows: &[FrameGrain]) -> bool {
    let mut lag_one: Vec<f64> = rows.iter().filter_map(|row| row.persistence[0]).collect();
    let mut lag_eight: Vec<f64> = rows.iter().filter_map(|row| row.persistence[7]).collect();
    let lag_one_median = median_of(&mut lag_one);
    let lag_eight_median = median_of(&mut lag_eight);

    let (Some(near), Some(far)) = (lag_one_median, lag_eight_median) else {
        return true;
    };

    if near <= 0.0 {
        return true;
    }

    far / near >= FROZEN_RATIO
}

/// Moves each change of mode to the nearest scene cut within [CUT_REACH] frames.
///
/// A move that would leave a run with no frames is skipped.
pub(super) fn snap_to_cuts(rows: &[FrameGrain], frozen: &mut [bool]) {
    let cuts: Vec<usize> = (1..rows.len())
        .filter(|&index| rows[index].scene != rows[index - 1].scene)
        .collect();
    let mut spans = spans_of(frozen);

    for index in 1..spans.len() {
        let change = spans[index].start;
        let Some(cut) = nearest_cut(rows, &cuts, change) else {
            continue;
        };

        let empties_a_run = cut <= spans[index - 1].start || cut >= spans[index].end;
        if empties_a_run {
            continue;
        }

        spans[index - 1].end = cut;
        spans[index].start = cut;
    }

    for span in spans {
        frozen[span.start..span.end].fill(span.marked);
    }
}

/// The cut nearest to the row `change` within [CUT_REACH] frames, the earlier winning a tie.
fn nearest_cut(rows: &[FrameGrain], cuts: &[usize], change: usize) -> Option<usize> {
    let change_frame = rows[change].frame;

    cuts.iter()
        .copied()
        .filter(|&cut| rows[cut].frame.abs_diff(change_frame) <= CUT_REACH)
        .min_by_key(|&cut| rows[cut].frame.abs_diff(change_frame))
}

fn spans_of(marks: &[bool]) -> Vec<Span> {
    let mut spans: Vec<Span> = Vec::new();
    for (index, &marked) in marks.iter().enumerate() {
        match spans.last_mut() {
            Some(last) if last.marked == marked => last.end = index + 1,
            _ => spans.push(Span {
                start: index,
                end: index + 1,
                marked,
            }),
        }
    }

    spans
}

fn run_of(rows: &[FrameGrain], span: &Span) -> GrainRun {
    let mode = if span.marked {
        GrainMode::Frozen
    } else {
        GrainMode::Fresh
    };

    GrainRun {
        first_frame: rows[span.start].frame,
        last_frame: rows[span.end - 1].frame,
        mode,
    }
}

/// The median, the mean of the two middle values for an even count, `None` when empty.
fn median_of(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }

    values.sort_by(f64::total_cmp);

    let middle = values.len() / 2;
    if values.len() % 2 == 1 {
        return Some(values[middle]);
    }

    let lower = values[middle - 1];
    let upper = values[middle];

    Some((lower + upper) / 2.0)
}

use super::fold_reference::{COEFFICIENTS, CONSUMER, HostRecord, SnapshotGap, curve_gap, fold_window};
use super::frames::{
    Agreement,
    MAX_RELATIVE,
    Plane,
    RADIUS,
    ReadingHarness,
    TEXTURE,
    chroma_settings,
    luma_settings,
    synthetic_stream,
};
use super::y4m::read_y4m;
use crate::noise::NOISE_CHANNELS;
use crate::noise::fold::{FoldSettings, NoiseSettings};
use crate::noise::history::WindowBounds;
use crate::noise::reading::ReadingRecord;
use crate::noise::snapshot::NoiseSnapshot;
use crate::noise::spatial::sigma_from_abs_sum;
use crate::noise::state::ReadingSettings;
use crate::tune::gpu::sync;

/// What one window fold showed on both sides.
struct FoldStep {
    gpu: NoiseSnapshot,
    host: NoiseSnapshot,
    /// Whether every frame of the window read with matching decisions on both sides.
    decisions_match: bool,
}

/// A reading harness that records every push and folds the window, with the host's readings
/// folded beside it.
///
/// Frame 0 folds alone, and every push from a full ring folds the window around the ring centre.
struct FoldHarness {
    readings: ReadingHarness,
    settings: NoiseSettings,
    records: Vec<HostRecord>,
    decisions: Vec<bool>,
    /// The push whose host reading is forced onto the no-sample path, to flip one decision.
    forced_flip: Option<u32>,
}

impl FoldHarness {
    fn new(plane: Plane, settings: NoiseSettings) -> Self {
        let readings = ReadingHarness::with_settings(plane, settings);

        Self {
            readings,
            settings,
            records: Vec::new(),
            decisions: Vec::new(),
            forced_flip: None,
        }
    }

    /// Pushes `frame`, records it and folds, returning the fold, if any.
    fn push(&mut self, frame: &[f32]) -> Option<(WindowBounds, FoldStep)> {
        let (gpu, host) = self.readings.push(frame);
        let agreement = Agreement::measure(&gpu, &host);
        let push = self.readings.pushed - 1;
        let slot = push % (2 * RADIUS + 1);

        let mut sample = host.reading.sample;
        let mut curve = host.reading.curve;
        let mut decisions_match = agreement.decisions_match();
        if self.forced_flip == Some(push) {
            sample = None;
            curve = None;
            decisions_match = false;
        }

        let plane = self.readings.plane;
        let mut immerkaer = [0.0f32; NOISE_CHANNELS as usize];
        for (channel, sigma) in immerkaer.iter_mut().enumerate().take(plane.channels as usize) {
            *sigma = sigma_from_abs_sum(host.sums[channel], plane.width, plane.height);
        }

        let reading = ReadingRecord {
            immerkaer_low: host.immerkaer_low,
            sample,
            accepted: 0,
            curve,
            populated_bins: 0,
        };
        let record = HostRecord { reading, immerkaer };

        self.readings.state.record_reading(slot, push);
        self.records.push(record);
        self.decisions.push(decisions_match);

        self.fold(push)
    }

    fn fold(&mut self, newest: u32) -> Option<(WindowBounds, FoldStep)> {
        let frames = 2 * RADIUS + 1;
        let pushed = newest + 1;
        let centre = match pushed {
            1 => 0,
            _ if pushed >= frames => newest - RADIUS,
            _ => return None,
        };

        let bounds = WindowBounds::around(centre, newest);
        self.readings
            .state
            .queue_window_fold(bounds)
            .expect("the window fold queues");
        let gpu = self.readings.state.snapshot().expect("the snapshot reads back");

        let channels = self.readings.plane.channels;
        let sigma_scale = self.settings.fold.sigma_scale;
        let host_window = fold_window(&self.records, bounds, sigma_scale);
        let host = host_window.snapshot(self.settings, channels);

        let first = bounds.first() as usize;
        let last = bounds.last() as usize;
        let decisions_match = self.decisions[first..=last].iter().all(|&matched| matched);
        let step = FoldStep {
            gpu,
            host,
            decisions_match,
        };

        Some((bounds, step))
    }
}

/// How many folds of a stream each rule judged.
#[derive(Debug, Default)]
struct FoldSummary {
    folds: usize,
    /// Folds whose window holds a frame with a decision flip.
    flipped: usize,
    largest_state_gap: f32,
}

/// Folds `frames` through a harness and checks every fold against the host window.
///
/// A fold whose window holds only frames with matching decisions holds the same chains, rho,
/// and flags as the host, every value within 1e-5 relative, and the same curve.
fn check_fold_stream<'a>(
    plane: Plane,
    settings: NoiseSettings,
    frames: impl IntoIterator<Item = &'a [f32]>,
    forced_flip: Option<u32>,
    what: &str,
) -> FoldSummary {
    let mut harness = FoldHarness::new(plane, settings);
    harness.forced_flip = forced_flip;
    let channels = plane.channels as usize;
    let mut summary = FoldSummary::default();

    for frame in frames {
        let Some((bounds, step)) = harness.push(frame) else {
            continue;
        };

        let fold = summary.folds;
        summary.folds += 1;

        if !step.decisions_match {
            summary.flipped += 1;
            continue;
        }

        let gap = SnapshotGap::measure(&step.gpu, &step.host, channels);
        let state_gap = gap.state;
        let derived_gap = gap.derived;
        let curve_gap = curve_gap(&step.gpu, &step.host);
        let window = (bounds.first(), bounds.last());

        assert!(
            gap.structure_matches,
            "{what}: fold {fold} over {window:?} holds other chains or flags"
        );
        assert!(
            state_gap <= MAX_RELATIVE,
            "{what}: fold {fold} over {window:?} state is {state_gap:e} apart"
        );
        assert!(
            derived_gap <= MAX_RELATIVE,
            "{what}: fold {fold} over {window:?} derived values are {derived_gap:e} apart"
        );
        assert!(
            curve_gap <= MAX_RELATIVE,
            "{what}: fold {fold} over {window:?} curve is {curve_gap:e} apart"
        );

        summary.largest_state_gap = summary.largest_state_gap.max(state_gap);
    }

    sync(&harness.readings.client).expect("the device syncs");
    summary
}

/// The fold settings of a plane with every derived value in use.
fn fold_settings(reading: ReadingSettings) -> NoiseSettings {
    let fold = FoldSettings {
        sigma_scale: 1.0,
        consumer: CONSUMER,
        coefficients: COEFFICIENTS,
        offset_tables: None,
    };

    NoiseSettings { reading, fold }
}

#[test]
fn synthetic_folds_track_the_host_window() {
    let plane = Plane::luma(256, 256);
    let frames = synthetic_stream(plane, 11);
    let frame_slices = frames.iter().map(Vec::as_slice);
    let settings = fold_settings(luma_settings());
    let summary = check_fold_stream(plane, settings, frame_slices, None, "synthetic luma");

    let folds = summary.folds;
    let flipped = summary.flipped;
    let expected = frames.len() - 2 * RADIUS as usize + 1;
    assert_eq!(
        folds, expected,
        "one fold of frame 0 and one per push from a full ring"
    );
    assert_eq!(flipped, 0, "no window holds a flip");
}

#[test]
fn a_one_sided_flip_only_reaches_the_windows_that_hold_it() {
    // Push 6 takes the no-sample path on the host alone.
    const FLIPPED_PUSH: u32 = 6;
    let plane = Plane::luma(256, 256);
    let frames = synthetic_stream(plane, 11);
    let frame_slices = frames.iter().map(Vec::as_slice);
    let settings = fold_settings(luma_settings());
    let summary = check_fold_stream(plane, settings, frame_slices, Some(FLIPPED_PUSH), "forced flip");

    // Every fold from a full ring centres on `push - RADIUS` and reads up to `push`.
    let pushes = frames.len() as u32;
    let holding = (2 * RADIUS..pushes)
        .filter(|&push| {
            let bounds = WindowBounds::around(push - RADIUS, push);
            (bounds.first()..=bounds.last()).contains(&FLIPPED_PUSH)
        })
        .count();

    assert!(holding > 0);
    assert_eq!(
        summary.flipped, holding,
        "the flip reaches exactly the windows that hold it"
    );
}

#[test]
fn texture_folds_track_the_host_window() {
    let clip = read_y4m(TEXTURE).expect("the clip reads");
    let luma = Plane::luma(clip.width, clip.height);
    let (chroma_width, chroma_height) = clip.chroma_dims();
    let chroma = Plane::chroma(chroma_width, chroma_height);
    let cases = [
        (luma, luma_settings(), "texture luma"),
        (chroma, chroma_settings(), "texture chroma"),
    ];

    for (plane, reading, what) in cases {
        let is_luma = plane.channels == 1;
        let frames = clip.frames.iter().map(|frame| {
            let samples = if is_luma { &frame.luma } else { &frame.chroma };
            samples.as_slice()
        });
        let settings = fold_settings(reading);
        let summary = check_fold_stream(plane, settings, frames, None, what);

        // Decision agreement of 0.999 over 21 folds allows no flip.
        let folds = summary.folds;
        let flipped = summary.flipped;
        assert!(folds > 0, "{what}: nothing folded");
        assert_eq!(flipped, 0, "{what}: a window holds a flip");
    }
}

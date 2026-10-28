use super::fold_reference::{COEFFICIENTS, CONSUMER, SnapshotGap, curve_gap, host_snapshot, largest_gap};
use super::frames::{
    Agreement,
    MAX_RELATIVE,
    Plane,
    RADIUS,
    ReadingHarness,
    TEXTURE,
    chroma_map,
    flat_boost,
    luma_map,
    synthetic_stream,
};
use super::y4m::read_y4m;
use crate::noise::correlation::correlation_factor;
use crate::noise::fold::{Fold, FoldSettings, NoiseSettings};
use crate::noise::host::HostNoise;
use crate::noise::reading::ReadingRecord;
use crate::noise::snapshot::NoiseSnapshot;
use crate::noise::spatial::sigma_from_abs_sum;
use crate::noise::state::ReadingSettings;
use crate::noise::temporal::{TemporalNoiseReading, TemporalNoiseSample};
use crate::tune::gpu::sync;

/// One slot's readings on both sides, as the fold reads them.
struct FoldSlot {
    gpu: ReadingRecord,
    host: TemporalNoiseReading,
    host_low: [f32; 3],
    sums: [f32; 4],
    decisions_match: bool,
}

/// What one fold showed on both sides.
struct FoldStep {
    gpu: NoiseSnapshot,
    host: NoiseSnapshot,
    lender_matches: bool,
    /// Whether the lender choice and the decisions of every slot the fold read match.
    decisions_match: bool,
    /// The largest relative gap between the two sides' raw chain inputs, at most 1.0, and 1.0
    /// where only one side has a sample.
    raw_gap: f32,
    /// Whether the held curve came from a fold whose decisions matched.
    curve_clean: bool,
}

/// A reading harness that folds every push into the GPU state and folds the host's readings
/// beside it.
///
/// Frame 0 seeds the stream, and every push from a full ring folds the window centre.
struct FoldHarness {
    readings: ReadingHarness,
    settings: NoiseSettings,
    host: HostNoise,
    slots: Vec<Option<FoldSlot>>,
    curve_clean: bool,
    /// The push whose host reading is forced onto the no-sample path, to flip one decision.
    forced_flip: Option<u32>,
}

impl FoldHarness {
    fn new(plane: Plane, settings: NoiseSettings) -> Self {
        let readings = ReadingHarness::with_settings(plane, settings);
        let frames = 2 * RADIUS + 1;
        let slots = (0..frames).map(|_| None).collect();

        Self {
            readings,
            settings,
            host: HostNoise::default(),
            slots,
            curve_clean: true,
            forced_flip: None,
        }
    }

    /// Pushes `frame` and folds, returning the slot reading's agreement and the fold, if any.
    fn push(&mut self, frame: &[f32]) -> (Agreement, Option<FoldStep>) {
        let boost = flat_boost(self.settings.reading.map);
        let (gpu, host) = self.readings.push(frame);
        let agreement = Agreement::measure(&gpu, &host, boost);
        let push = self.readings.pushed - 1;
        let slot = push % (2 * RADIUS + 1);

        let mut host_reading = host.reading;
        let mut decisions_match = agreement.decisions_match();
        if self.forced_flip == Some(push) {
            host_reading = TemporalNoiseReading::empty();
            decisions_match = false;
        }

        self.slots[slot as usize] = Some(FoldSlot {
            gpu: gpu.reading,
            host: host_reading,
            host_low: host.immerkaer_low,
            sums: host.sums,
            decisions_match,
        });

        let step = self.fold(slot);
        (agreement, step)
    }

    fn fold(&mut self, slot: u32) -> Option<FoldStep> {
        let frames = 2 * RADIUS + 1;
        let pushed = self.readings.pushed;
        let fold = match pushed {
            1 => Fold::Seed { slot },
            _ if pushed >= frames => Fold::Centre {
                ring_start: pushed % frames,
                centre: RADIUS,
            },
            _ => return None,
        };

        self.readings.state.queue_fold(fold).expect("the fold queues");
        let gpu = self.readings.state.snapshot().expect("the snapshot reads back");

        let (ring_start, centre, seed) = match fold {
            Fold::Seed { slot } => (slot, 0, true),
            Fold::Centre { ring_start, centre } => (ring_start, centre, false),
        };
        let slot_at = |logical: u32| (ring_start + logical) % frames;
        let borrow = self.settings.fold.borrow_ahead && !seed;
        let gpu_lender = lender(centre, frames, borrow, |logical| {
            let reading_slot = slot_at(logical) as usize;
            let entry = self.slots[reading_slot].as_ref();
            entry.is_some_and(|entry| entry.gpu.sample.is_some())
        });
        let host_lender = lender(centre, frames, borrow, |logical| {
            let reading_slot = slot_at(logical) as usize;
            let entry = self.slots[reading_slot].as_ref();
            entry.is_some_and(|entry| entry.host.sample.is_some())
        });

        let centre_slot = self.slot(slot_at(centre));
        let gpu_lent = self.slot(slot_at(gpu_lender));
        let host_lent = self.slot(slot_at(host_lender));
        let lender_matches = gpu_lender == host_lender;
        let decisions_match = lender_matches
            && centre_slot.decisions_match
            && gpu_lent.decisions_match
            && host_lent.decisions_match;

        let channels = self.readings.plane.channels as usize;
        let sigma_scale = self.settings.fold.sigma_scale;
        let mut immerkaer = [0.0f32; 3];
        for (channel, sigma) in immerkaer.iter_mut().enumerate().take(channels) {
            let plane = self.readings.plane;
            *sigma = sigma_from_abs_sum(centre_slot.sums[channel], plane.width, plane.height);
        }

        let gpu_sample = gpu_lent.gpu.sample.filter(|_| !seed);
        let host_sample = host_lent.host.sample.filter(|_| !seed);
        let gpu_raw = raw_inputs(immerkaer, centre_slot.gpu.immerkaer_low, gpu_sample, channels);
        let host_raw = raw_inputs(immerkaer, centre_slot.host_low, host_sample, channels);
        let samples_agree = gpu_sample.is_some() == host_sample.is_some();
        let measured_gap = largest_gap(&gpu_raw, &host_raw);
        let raw_gap = if samples_agree { measured_gap.min(1.0) } else { 1.0 };

        let host_low = centre_slot.host_low;
        // Either side replacing its curve on a fold whose decisions differ leaves the curves apart.
        let adopts = host_sample.is_some() || gpu_sample.is_some();
        let lent_reading = host_lent.host.clone();
        if !seed {
            self.host.adopt(&lent_reading);
        }

        self.host
            .fold(immerkaer, host_sample, host_low, channels, sigma_scale);

        if adopts {
            self.curve_clean = decisions_match;
        }

        let host = host_snapshot(&self.host, self.settings, channels as u32);
        let step = FoldStep {
            gpu,
            host,
            lender_matches,
            decisions_match,
            raw_gap,
            curve_clean: self.curve_clean,
        };

        Some(step)
    }

    fn slot(&self, slot: u32) -> &FoldSlot {
        self.slots[slot as usize]
            .as_ref()
            .expect("every folded slot holds a reading")
    }
}

/// The logical position a fold of `centre` takes its sample from, by the borrow rule.
fn lender(centre: u32, frames: u32, borrow: bool, has_sample: impl Fn(u32) -> bool) -> u32 {
    if !borrow || has_sample(centre) {
        return centre;
    }

    let ahead = (centre + 1..frames).find(|&logical| has_sample(logical));
    ahead.unwrap_or(centre)
}

/// The values a fold combines into its chains.
///
/// Each channel's median, low and unboosted low inputs come first. With a sample, its sigmas and
/// its correlation follow.
fn raw_inputs(
    immerkaer: [f32; 3],
    immerkaer_low: [f32; 3],
    sample: Option<TemporalNoiseSample>,
    channels: usize,
) -> Vec<f32> {
    let mut values = Vec::with_capacity(4 * channels + 1);

    let Some(sample) = sample else {
        for channel in 0..channels {
            values.extend([immerkaer[channel], immerkaer_low[channel], immerkaer_low[channel]]);
        }

        return values;
    };

    let factor = correlation_factor(sample.rho);
    for channel in 0..channels {
        let boosted = immerkaer[channel].max(sample.sigma[channel] * factor);
        let boosted_low = immerkaer_low[channel].max(sample.sigma_low[channel] * factor);
        let unboosted_low = immerkaer_low[channel].max(sample.sigma_low[channel]);
        values.extend([boosted, boosted_low, unboosted_low, sample.sigma[channel]]);
    }

    values.push(sample.rho);
    values
}

/// How many folds of a stream each rule judged.
#[derive(Debug, Default)]
struct FoldSummary {
    folds: usize,
    /// Folds at least 15 folds after the last flip, or with no flip before them.
    settled: usize,
    /// Folds within 15 folds of a flip.
    recovering: usize,
    flips: usize,
    curve_compared: usize,
    largest_state_gap: f32,
}

/// The folds after a flip a fold counts as recovering.
const RECOVERY_FOLDS: usize = 15;

/// Folds `frames` through a harness and checks every fold against the host fold.
///
/// A fold flips when its decisions differ or the two sides hold different chains, rho, flags or
/// map modes. A one-sided flip has a raw gap `d` of 1.0, and any other flip its measured raw gap,
/// at most 1.0. Fold `k` is within `1e-5 + sum(0.2 * d * 0.8^(k - f))` relative of the host,
/// summed over every flip `f` up to `k`, on every value both sides hold. With no flip that is
/// 1e-5. Derived values square sigma, so they get twice the sum. Presence, flags and map modes
/// follow the decisions, so a fold more than 15 folds after the last decision flip must hold the
/// same chains, rho, flags and map mode on both sides. The curve is compared where the lender
/// choice matches and the held curve came from a matching fold.
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
    let mut flips: Vec<(usize, f32)> = Vec::new();
    let mut since_flip = usize::MAX;
    let mut since_decision_flip = usize::MAX;

    for frame in frames {
        let (_, step) = harness.push(frame);
        let Some(step) = step else {
            continue;
        };

        let fold = summary.folds;
        let gap = SnapshotGap::measure(&step.gpu, &step.host, channels);
        if !step.decisions_match {
            since_decision_flip = 0;
        }

        let decided_long_ago = since_decision_flip > RECOVERY_FOLDS;
        let structure_matches = gap.structure_matches;
        assert!(
            structure_matches || !decided_long_ago,
            "{what}: fold {fold} holds other chains, flags or map modes long after any decision flip"
        );

        let flipped = !step.decisions_match || !structure_matches;
        if flipped {
            let raw_gap = if structure_matches { step.raw_gap } else { 1.0 };
            flips.push((fold, raw_gap));
            summary.flips += 1;
            since_flip = 0;
        }

        let excursion: f32 = flips
            .iter()
            .map(|&(flip, gap)| {
                let folds_since = (fold - flip) as i32;
                0.2 * gap * 0.8f32.powi(folds_since)
            })
            .sum();
        let state_bound = excursion + MAX_RELATIVE;
        let derived_bound = 2.0 * excursion + MAX_RELATIVE;
        let state_gap = gap.state;
        let derived_gap = gap.derived;

        assert!(
            state_gap <= state_bound,
            "{what}: fold {fold} state is {state_gap:e} apart, bound {state_bound:e}"
        );
        assert!(
            derived_gap <= derived_bound,
            "{what}: fold {fold} derived values are {derived_gap:e} apart, bound {derived_bound:e}"
        );

        if step.lender_matches && step.curve_clean {
            let curve_gap = curve_gap(&step.gpu, &step.host);
            let classes_match = step.gpu.classes_present == step.host.classes_present;
            assert!(
                curve_gap <= MAX_RELATIVE,
                "{what}: fold {fold} curve is {curve_gap:e} apart"
            );
            assert!(classes_match, "{what}: fold {fold} classes presence");
            summary.curve_compared += 1;
        }

        let settled = since_flip >= RECOVERY_FOLDS;
        if settled {
            summary.settled += 1;
        } else {
            summary.recovering += 1;
        }

        summary.folds += 1;
        summary.largest_state_gap = summary.largest_state_gap.max(state_gap);
        since_flip = since_flip.saturating_add(1);
        since_decision_flip = since_decision_flip.saturating_add(1);
    }

    sync(&harness.readings.client).expect("the device syncs");
    summary
}

/// The fold settings of a plane read with shifted edges, with every derived value in use.
fn fold_settings(reading: ReadingSettings) -> NoiseSettings {
    let fold = FoldSettings {
        sigma_scale: 1.0,
        borrow_ahead: true,
        consumer: CONSUMER,
        coefficients: COEFFICIENTS,
        offset_tables: None,
    };

    NoiseSettings { reading, fold }
}

#[test]
fn synthetic_folds_track_the_host_fold() {
    let plane = Plane::luma(256, 256);
    let frames = synthetic_stream(plane, 11);
    let frame_slices = frames.iter().map(Vec::as_slice);
    let settings = fold_settings(luma_map());
    let summary = check_fold_stream(plane, settings, frame_slices, None, "synthetic luma");
    println!("synthetic luma: {summary:?}");

    let folds = summary.folds;
    let flips = summary.flips;
    let expected = frames.len() - 2 * RADIUS as usize + 1;
    assert_eq!(folds, expected, "one seed and one fold per push from a full ring");
    assert_eq!(flips, 0, "no fold flips");
}

#[test]
fn a_one_sided_flip_recovers_under_the_flip_bound() {
    // Push 6 takes the no-sample path on the host alone, so the host borrows the next sample
    // when the fold centres on it.
    let plane = Plane::luma(256, 256);
    let frames = synthetic_stream(plane, 11);
    let frame_slices = frames.iter().map(Vec::as_slice);
    let settings = fold_settings(luma_map());
    let summary = check_fold_stream(plane, settings, frame_slices, Some(6), "forced flip");
    println!("forced flip: {summary:?}");

    let flips = summary.flips;
    let largest_gap = summary.largest_state_gap;
    assert_eq!(flips, 1, "the forced reading flips one fold");
    assert!(
        largest_gap > MAX_RELATIVE,
        "the flip shows only {largest_gap:e}, so the bound let no real gap through"
    );
}

#[test]
fn texture_folds_track_the_host_fold() {
    let clip = read_y4m(TEXTURE).expect("the clip reads");
    let luma = Plane::luma(clip.width, clip.height);
    let (chroma_width, chroma_height) = clip.chroma_dims();
    let chroma = Plane::chroma(chroma_width, chroma_height);
    let cases = [
        (luma, luma_map(), "texture luma"),
        (chroma, chroma_map(), "texture chroma"),
    ];

    for (plane, reading, what) in cases {
        let is_luma = plane.channels == 1;
        let frames = clip.frames.iter().map(|frame| {
            let samples = if is_luma { &frame.luma } else { &frame.chroma };
            samples.as_slice()
        });
        let settings = fold_settings(reading);
        let summary = check_fold_stream(plane, settings, frames, None, what);
        println!("{what}: {summary:?}");

        // Decision agreement of 0.999 over 21 folds allows no flip.
        let folds = summary.folds;
        let flips = summary.flips;
        assert!(folds > 0, "{what}: nothing folded");
        assert_eq!(flips, 0, "{what}: folds flipped");
    }
}

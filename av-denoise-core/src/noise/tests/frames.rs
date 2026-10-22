use std::collections::BTreeMap;

use cubecl::prelude::*;

use super::records::{hashed_unit, ulp_gap};
use super::y4m::read_y4m;
use crate::collab::geometry::strength_map_dims;
use crate::engine::align::StorageAlign;
use crate::nl4d::PsyParams;
use crate::noise::curve::{MIN_QUARTERS_PER_BIN, NOISE_CURVE_BINS, curve_bin_counts};
use crate::noise::derived::DerivedValues;
use crate::noise::fold::NoiseSettings;
use crate::noise::reading::{CURVE_TAGS, ReadingRecord};
use crate::noise::spatial::sigma_block_p25_from_partials;
use crate::noise::state::{BLOCK_ACCEPTED, NoiseShape, NoiseState, ReadingSettings, StrengthMapKind};
use crate::noise::strength_map::{QuarterSettings, StrengthMapParams};
use crate::noise::temporal::{
    TEMPORAL_QUARTER_SIZE,
    TemporalNoiseReading,
    accepted_static_blocks,
    temporal_noise_reading,
    temporal_stats_blocks,
};
use crate::tune::gpu::sync;
use crate::tune::tests::fixtures::upload;
use crate::tune::tests::runtime::{TestRuntime, test_client};

/// The lowest share of blocks, bins or quarters whose decisions must match on any frame.
const MIN_AGREEMENT: f64 = 0.999;
/// The largest relative gap any value may show on a frame whose decisions all match.
pub(super) const MAX_RELATIVE: f32 = 1.0e-5;
pub(super) const RADIUS: u32 = 2;
const SYNTHETIC_FRAMES: u32 = 12;
pub(super) const BRICK: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../data/host_stalls_2026_10/brick48.y4m"
);
const EDGES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../data/host_stalls_2026_10/edges63.y4m"
);

/// The geometry of one plane a stream runs on.
#[derive(Debug, Clone, Copy)]
pub(super) struct Plane {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) channels: u32,
    pub(super) stored_ch: u32,
}

impl Plane {
    pub(super) fn luma(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            channels: 1,
            stored_ch: 1,
        }
    }

    pub(super) fn chroma(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            channels: 2,
            stored_ch: 2,
        }
    }

    fn frame_len(self) -> usize {
        (self.width * self.height * self.stored_ch) as usize
    }

    fn blocks(self) -> usize {
        let (blocks_x, blocks_y) = temporal_stats_blocks(self.width, self.height);
        (blocks_x * blocks_y) as usize
    }
}

/// The luma plane's settings with the curve and no map, as nl4d reads luma without psy.
fn luma_off() -> ReadingSettings {
    ReadingSettings {
        luma_fields: true,
        ..ReadingSettings::default()
    }
}

/// The luma plane's settings with the map, the texture cut and the line ring at the psy defaults.
pub(super) fn luma_map() -> ReadingSettings {
    let psy = PsyParams::default();
    let params = StrengthMapParams {
        flat_boost: psy.flat_boost,
        shadow_soften: psy.shadow_soften,
    };
    let ring = psy.line_ring.div_ceil(TEMPORAL_QUARTER_SIZE) as usize;
    let quarter = QuarterSettings {
        texture_cut: Some(psy.flat_texture_cut),
        line_ring: Some(ring),
    };

    ReadingSettings {
        luma_fields: true,
        quarter,
        map: StrengthMapKind::Luma(params),
    }
}

/// The chroma plane's settings without a map, as nl4d reads chroma without psy.
fn chroma_off() -> ReadingSettings {
    ReadingSettings::default()
}

/// The chroma plane's settings with the chroma map at the psy default boost.
pub(super) fn chroma_map() -> ReadingSettings {
    let psy = PsyParams::default();
    let map = StrengthMapKind::Chroma {
        flat_boost: psy.chroma_flat_boost,
    };

    ReadingSettings {
        luma_fields: true,
        map,
        ..ReadingSettings::default()
    }
}

/// The multiplier a flat quarter gets under `map`, `None` without a map.
pub(super) fn flat_boost(map: StrengthMapKind) -> Option<f32> {
    match map {
        StrengthMapKind::Off => None,
        StrengthMapKind::Luma(params) => Some(params.flat_boost),
        StrengthMapKind::Chroma { flat_boost } => Some(flat_boost),
    }
}

/// What the GPU wrote for one slot.
pub(super) struct GpuSlot {
    pub(super) reading: ReadingRecord,
    block_flags: Vec<u32>,
    bin_counts: [usize; NOISE_CURVE_BINS],
    /// Empty without a map.
    map: Vec<f32>,
}

/// What the host reference reads from the same slot's stats.
pub(super) struct HostSlot {
    /// The Immerkær abs sums both sides fold.
    pub(super) sums: [f32; 4],
    pub(super) immerkaer_low: [f32; 3],
    pub(super) reading: TemporalNoiseReading,
    accepted: Vec<bool>,
    /// The bin counts, present when the curve is built and the slot has a sample.
    bin_counts: Option<[usize; NOISE_CURVE_BINS]>,
    /// Empty without a map.
    map: Vec<f32>,
}

/// A noise state over a ring of `2 * radius + 1` slots that reads every pushed frame.
pub(super) struct ReadingHarness {
    pub(super) client: ComputeClient<TestRuntime>,
    pub(super) plane: Plane,
    settings: ReadingSettings,
    pub(super) state: NoiseState<TestRuntime>,
    /// The host copy of the input ring, uploaded whole on every push.
    ring: Vec<f32>,
    pub(super) pushed: u32,
}

impl ReadingHarness {
    fn new(plane: Plane, settings: ReadingSettings) -> Self {
        let noise_settings = NoiseSettings {
            reading: settings,
            ..NoiseSettings::default()
        };

        Self::with_settings(plane, noise_settings)
    }

    pub(super) fn with_settings(plane: Plane, noise_settings: NoiseSettings) -> Self {
        let client = test_client();
        let frames = 2 * RADIUS + 1;
        let shape = NoiseShape {
            width: plane.width,
            height: plane.height,
            channels: plane.channels,
            stored_ch: plane.stored_ch,
            frames,
            temporal: true,
            align: StorageAlign::from_client(&client),
        };
        let state = NoiseState::new(&client, shape, noise_settings, DerivedValues::default());
        let ring = vec![0.0f32; plane.frame_len() * frames as usize];

        Self {
            client,
            plane,
            settings: noise_settings.reading,
            state,
            ring,
            pushed: 0,
        }
    }

    /// Pushes `frame` into the next slot, queues its stats and reading, and returns the slot.
    fn queue_frame(&mut self, frame: &[f32]) -> u32 {
        let frames = self.state.shape().frames;
        let frame_len = self.plane.frame_len();
        let slot = self.pushed % frames;
        let previous = (self.pushed > 0).then_some((slot + frames - 1) % frames);

        let frame_size = frame.len();
        assert_eq!(frame_size, frame_len, "one frame's samples");

        let start = slot as usize * frame_len;
        self.ring[start..start + frame_len].copy_from_slice(frame);

        let input = upload(&self.client, &self.ring);
        self.state
            .queue_stats(&input, slot, previous)
            .expect("the stats queue");
        self.state.queue_reading(slot).expect("the reading queues");
        self.pushed += 1;

        slot
    }

    fn push_gpu(&mut self, frame: &[f32]) -> GpuSlot {
        let slot = self.queue_frame(frame);
        self.read_gpu(slot)
    }

    pub(super) fn push(&mut self, frame: &[f32]) -> (GpuSlot, HostSlot) {
        let slot = self.queue_frame(frame);
        let gpu = self.read_gpu(slot);
        let host = self.read_host(slot);

        (gpu, host)
    }

    fn read_gpu(&self, slot: u32) -> GpuSlot {
        let reading = self.state.read_reading(slot);
        let block_flags = self.state.read_block_flags();
        let tags = self.state.read_curve_tags();
        let map = self.state.read_slot_map(slot);

        let mut bin_counts = [0usize; NOISE_CURVE_BINS];
        for tag in tags {
            if tag < CURVE_TAGS {
                bin_counts[tag as usize] += 1;
            }
        }

        GpuSlot {
            reading,
            block_flags,
            bin_counts,
            map,
        }
    }

    fn read_host(&self, slot: u32) -> HostSlot {
        let plane = self.plane;
        let settings = self.settings;
        let (partials, sums, records) = self.state.read_slot_stats(slot);
        let immerkaer_low =
            sigma_block_p25_from_partials(&partials, plane.channels, plane.width, plane.height);
        let reading = temporal_noise_reading(
            &records,
            plane.channels,
            plane.stored_ch,
            plane.width,
            plane.height,
            settings.luma_fields,
            settings.quarter,
        );
        let accepted_blocks = accepted_static_blocks(
            &records,
            plane.channels,
            plane.stored_ch,
            plane.width,
            plane.height,
        );

        let mut accepted = vec![false; plane.blocks()];
        for block in accepted_blocks.iter().flatten() {
            accepted[block.index] = true;
        }

        let bin_counts = match (reading.sample, accepted_blocks.as_deref()) {
            (Some(sample), Some(blocks)) if settings.luma_fields => {
                let counts = curve_bin_counts(&records, plane.stored_ch, blocks, sample.sigma[0]);
                Some(counts)
            },
            _ => None,
        };

        let map = host_map(&reading, settings.map, plane);

        HostSlot {
            sums,
            immerkaer_low,
            reading,
            accepted,
            bin_counts,
            map,
        }
    }
}

/// The host's multipliers for the reading's classes, ones without classes, empty without a map.
fn host_map(reading: &TemporalNoiseReading, map: StrengthMapKind, plane: Plane) -> Vec<f32> {
    let (cols, rows) = strength_map_dims(plane.width, plane.height);
    let quarters = (cols * rows) as usize;
    let classes = reading.classes.as_ref();

    match (map, classes) {
        (StrengthMapKind::Off, _) => Vec::new(),
        (_, None) => vec![1.0; quarters],
        (StrengthMapKind::Luma(params), Some(classes)) => classes.luma_multipliers(params),
        (StrengthMapKind::Chroma { flat_boost }, Some(classes)) => classes.chroma_multipliers(flat_boost),
    }
}

/// How many decisions of one frame the GPU and the host make alike.
#[derive(Debug, Clone, Copy)]
pub(super) struct Agreement {
    blocks: usize,
    block_flips: usize,
    bins: usize,
    bin_flips: usize,
    quarters: usize,
    quarter_flips: usize,
    none_paths_agree: bool,
}

impl Agreement {
    pub(super) fn measure(gpu: &GpuSlot, host: &HostSlot, boost: Option<f32>) -> Self {
        let blocks = host.accepted.len();
        let gpu_blocks = gpu.block_flags.len();
        assert_eq!(gpu_blocks, blocks, "block flag count");

        let block_flips = gpu
            .block_flags
            .iter()
            .zip(&host.accepted)
            .filter(|(flags, accepted)| (**flags & BLOCK_ACCEPTED != 0) != **accepted)
            .count();

        let gpu_sampled = gpu.reading.sample.is_some();
        let host_sampled = host.reading.sample.is_some();
        let gpu_curved = gpu.reading.curve.is_some();
        let host_curved = host.reading.curve.is_some();
        let none_paths_agree = gpu_sampled == host_sampled && gpu_curved == host_curved;

        // Bins are compared only where both sides have a sample, since only then do they tag quarters
        // against the same flat gate.
        let host_bins = host.bin_counts.filter(|_| gpu_sampled);
        let (bins, bin_flips) = match host_bins {
            Some(host_counts) => {
                let flips = bin_flips(&gpu.bin_counts, &host_counts);
                (NOISE_CURVE_BINS, flips)
            },
            None => (0, 0),
        };

        let (quarters, quarter_flips) = match boost {
            Some(boost) => quarter_flips(&gpu.map, &host.map, boost),
            None => (0, 0),
        };

        Self {
            blocks,
            block_flips,
            bins,
            bin_flips,
            quarters,
            quarter_flips,
            none_paths_agree,
        }
    }

    pub(super) fn decisions_match(&self) -> bool {
        let no_flips = self.block_flips == 0 && self.bin_flips == 0 && self.quarter_flips == 0;
        no_flips && self.none_paths_agree
    }

    fn lowest(&self) -> f64 {
        let block_share = share(self.block_flips, self.blocks);
        let bin_share = share(self.bin_flips, self.bins);
        let quarter_share = share(self.quarter_flips, self.quarters);

        block_share.min(bin_share).min(quarter_share)
    }
}

fn share(flips: usize, total: usize) -> f64 {
    if total == 0 {
        return 1.0;
    }

    1.0 - flips as f64 / total as f64
}

/// Bins whose populated state differs, or which are populated on both sides with other counts.
fn bin_flips(gpu: &[usize; NOISE_CURVE_BINS], host: &[usize; NOISE_CURVE_BINS]) -> usize {
    gpu.iter()
        .zip(host)
        .filter(|(gpu_count, host_count)| {
            let gpu_populated = **gpu_count >= MIN_QUARTERS_PER_BIN;
            let host_populated = **host_count >= MIN_QUARTERS_PER_BIN;
            let populated = gpu_populated || host_populated;

            populated && gpu_count != host_count
        })
        .count()
}

/// The quarter count and the quarters that only one side boosts as flat.
fn quarter_flips(gpu: &[f32], host: &[f32], boost: f32) -> (usize, usize) {
    let quarters = host.len();
    let gpu_quarters = gpu.len();
    assert_eq!(gpu_quarters, quarters, "map length");

    let flips = gpu
        .iter()
        .zip(host)
        .filter(|(gpu_value, host_value)| (**gpu_value == boost) != (**host_value == boost))
        .count();

    (quarters, flips)
}

/// The largest gaps one quantity showed.
#[derive(Debug, Clone, Copy, Default)]
struct Gap {
    ulps: u32,
    relative: f32,
}

/// The largest gap of every quantity over the frames whose decisions all match.
#[derive(Debug, Default)]
struct Gaps {
    quantities: BTreeMap<&'static str, Gap>,
}

impl Gaps {
    fn record(&mut self, name: &'static str, gpu: f32, host: f32) {
        let ulps = ulp_gap(gpu, host);
        let measured = relative_gap(gpu, host);

        // `f32::max` drops a NaN, so a gap that is not finite is kept as infinite.
        let relative = if measured.is_finite() {
            measured
        } else {
            f32::INFINITY
        };

        let gap = self.quantities.entry(name).or_default();

        gap.ulps = gap.ulps.max(ulps);
        gap.relative = gap.relative.max(relative);
    }

    fn record_slot(&mut self, gpu: &GpuSlot, host: &HostSlot, channels: usize) {
        if let (Some(gpu_sample), Some(host_sample)) = (gpu.reading.sample, host.reading.sample) {
            for channel in 0..channels {
                self.record("sigma", gpu_sample.sigma[channel], host_sample.sigma[channel]);
                self.record(
                    "sigma_low",
                    gpu_sample.sigma_low[channel],
                    host_sample.sigma_low[channel],
                );
            }

            self.record("rho", gpu_sample.rho, host_sample.rho);
            self.record(
                "static_fraction",
                gpu_sample.static_fraction,
                host_sample.static_fraction,
            );
        }

        if let (Some(gpu_curve), Some(host_curve)) = (gpu.reading.curve, host.reading.curve) {
            self.record(
                "quarter_median",
                gpu_curve.sigma_quarter_median,
                host_curve.sigma_quarter_median,
            );

            for bin in 0..NOISE_CURVE_BINS {
                self.record("ratio", gpu_curve.ratios[bin], host_curve.ratios[bin]);
            }
        }

        for (&gpu_value, &host_value) in gpu.map.iter().zip(&host.map) {
            self.record("multiplier", gpu_value, host_value);
        }
    }
}

fn relative_gap(gpu: f32, host: f32) -> f32 {
    if gpu == host {
        return 0.0;
    }

    let difference = (gpu - host).abs();
    difference / host.abs()
}

/// What one stream showed over all its frames.
#[derive(Debug, Default)]
struct StreamSummary {
    frames: usize,
    sampled: usize,
    curved: usize,
    flat_quarters: usize,
    /// Each frame with a flip, with its agreement.
    flipped: Vec<(usize, Agreement)>,
    /// The immerkaer quartile's gaps over every frame.
    immerkaer: Gaps,
    /// The gaps over the frames whose decisions all match.
    matched: Gaps,
}

impl StreamSummary {
    fn print(&self, what: &str) {
        let frames = self.frames;
        let sampled = self.sampled;
        let curved = self.curved;
        let flipped = self.flipped.len();
        let flat = self.flat_quarters;
        println!(
            "{what}: {frames} frames, {sampled} sampled, {curved} curved, {flat} flat quarters, \
             {flipped} with a flip"
        );

        for (frame, agreement) in &self.flipped {
            let Agreement {
                blocks,
                block_flips,
                bins,
                bin_flips,
                quarters,
                quarter_flips,
                none_paths_agree,
            } = agreement;
            println!(
                "{what}: frame {frame}: {block_flips}/{blocks} blocks, {bin_flips}/{bins} bins, \
                 {quarter_flips}/{quarters} quarters, none paths agree {none_paths_agree}"
            );
        }

        let gap_sets = [
            ("every frame", &self.immerkaer),
            ("matched frames", &self.matched),
        ];
        for (label, gaps) in gap_sets {
            for (name, gap) in &gaps.quantities {
                let Gap { ulps, relative } = gap;
                println!("{what}: {label}: {name} largest gap {ulps} ulp, {relative:e} relative");
            }
        }
    }

    fn assert_agrees(&self, what: &str) {
        for (frame, agreement) in &self.flipped {
            let none_paths_agree = agreement.none_paths_agree;
            let lowest = agreement.lowest();
            assert!(
                none_paths_agree,
                "{what}: frame {frame} takes a None path on one side"
            );
            assert!(
                lowest >= MIN_AGREEMENT,
                "{what}: frame {frame} agrees on {lowest} of its decisions, {agreement:?}"
            );
        }

        for (name, gap) in &self.immerkaer.quantities {
            let relative = gap.relative;
            assert!(relative <= MAX_RELATIVE, "{what}: {name} is {relative:e} apart");
        }

        for (name, gap) in &self.matched.quantities {
            let relative = gap.relative;
            assert!(relative <= MAX_RELATIVE, "{what}: {name} is {relative:e} apart");
        }
    }
}

/// Runs `frames` through a harness and compares every slot's reading with the host's.
fn check_stream<'a>(
    plane: Plane,
    settings: ReadingSettings,
    frames: impl IntoIterator<Item = &'a [f32]>,
) -> StreamSummary {
    let mut harness = ReadingHarness::new(plane, settings);
    let boost = flat_boost(settings.map);
    let channels = plane.channels as usize;
    let mut summary = StreamSummary::default();

    for (index, frame) in frames.into_iter().enumerate() {
        let (gpu, host) = harness.push(frame);
        let agreement = Agreement::measure(&gpu, &host, boost);

        for channel in 0..channels {
            let gpu_low = gpu.reading.immerkaer_low[channel];
            let host_low = host.immerkaer_low[channel];
            summary.immerkaer.record("immerkaer_low", gpu_low, host_low);
        }

        if agreement.decisions_match() {
            summary.matched.record_slot(&gpu, &host, channels);
        } else {
            summary.flipped.push((index, agreement));
        }

        let flat_quarters = match boost {
            Some(boost) => host.map.iter().filter(|&&value| value == boost).count(),
            None => 0,
        };

        summary.frames += 1;
        summary.sampled += usize::from(host.reading.sample.is_some());
        summary.curved += usize::from(host.reading.curve.is_some());
        summary.flat_quarters += flat_quarters;
    }

    sync(&harness.client).expect("the device syncs");
    summary
}

/// A smooth gradient with grain that grows with brightness, a frozen corner and a band that moves
/// five pixels per frame.
fn grain_frame(width: u32, height: u32, seed: u32, frame: u32) -> Vec<f32> {
    let frozen_width = width / 8;
    let frozen_height = height / 8;
    let band_width = width / 8;
    let band_start = width / 4 + 5 * frame;
    let band_end = band_start + band_width;
    let spread = 12.0f32.sqrt();
    let mut samples = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let across = x as f32 / width as f32;
            let down = y as f32 / height as f32;
            let base = 0.3 + 0.3 * across + 0.05 * down;
            let in_band = (band_start..band_end).contains(&x);
            let luma = if in_band { base + 0.15 } else { base };

            let frozen = x < frozen_width && y < frozen_height;
            let grain_frame = if frozen { 0 } else { frame };
            let grain_seed = seed.wrapping_add(grain_frame.wrapping_mul(7919));
            let unit = hashed_unit(grain_seed, y * width + x);
            let sigma = (3.0 + 1.5 * luma) / 255.0;
            let grain = (unit - 0.5) * spread * sigma;

            samples.push(luma + grain);
        }
    }

    samples
}

/// [grain_frame] for each channel of `plane`, interleaved per pixel.
fn synthetic_frame(plane: Plane, seed: u32, frame: u32) -> Vec<f32> {
    let mut channels = Vec::with_capacity(plane.stored_ch as usize);
    for channel in 0..plane.stored_ch {
        let channel_seed = seed + 1000 * channel;
        let samples = grain_frame(plane.width, plane.height, channel_seed, frame);
        channels.push(samples);
    }

    let pixels = (plane.width * plane.height) as usize;
    let mut interleaved = Vec::with_capacity(plane.frame_len());
    for pixel in 0..pixels {
        for samples in &channels {
            interleaved.push(samples[pixel]);
        }
    }

    interleaved
}

pub(super) fn synthetic_stream(plane: Plane, seed: u32) -> Vec<Vec<f32>> {
    (0..SYNTHETIC_FRAMES)
        .map(|frame| synthetic_frame(plane, seed, frame))
        .collect()
}

#[test]
fn synthetic_streams_agree_with_the_host_reading() {
    let luma = Plane::luma(256, 256);
    let chroma = Plane::chroma(128, 128);
    let cases = [
        (luma, luma_map(), 3, "synthetic luma"),
        (chroma, chroma_map(), 5, "synthetic chroma"),
    ];

    for (plane, settings, seed, what) in cases {
        let frames = synthetic_stream(plane, seed);
        let frame_slices = frames.iter().map(Vec::as_slice);
        let summary = check_stream(plane, settings, frame_slices);
        summary.print(what);
        summary.assert_agrees(what);

        // Frame 0 has no previous frame, so only the frames after it carry a sample.
        let sampled = summary.sampled;
        let curved = summary.curved;
        let flat_quarters = summary.flat_quarters;
        let expected = frames.len() - 1;
        assert_eq!(sampled, expected, "{what}: sampled frames");
        assert!(curved > 0, "{what}: no frame has a curve");
        assert!(flat_quarters > 0, "{what}: no quarter is flat");
    }
}

/// Checks the luma and chroma planes of the clip at `path`, with and without a map.
fn check_clip(path: &str, name: &str) {
    let clip = read_y4m(path).expect("the clip reads");
    let luma = Plane::luma(clip.width, clip.height);
    let (chroma_width, chroma_height) = clip.chroma_dims();
    let chroma = Plane::chroma(chroma_width, chroma_height);

    let cases = [
        (luma, luma_off(), "luma, map off"),
        (luma, luma_map(), "luma, luma map"),
        (chroma, chroma_off(), "chroma, map off"),
        (chroma, chroma_map(), "chroma, chroma map"),
    ];

    let mut summaries = Vec::with_capacity(cases.len());
    for (plane, settings, label) in cases {
        let is_luma = plane.channels == 1;
        let frames = clip.frames.iter().map(|frame| {
            let samples = if is_luma { &frame.luma } else { &frame.chroma };
            samples.as_slice()
        });
        let summary = check_stream(plane, settings, frames);
        let what = format!("{name} {label}");
        summary.print(&what);
        summaries.push((what, summary));
    }

    for (what, summary) in &summaries {
        summary.assert_agrees(what);
    }
}

#[test]
#[ignore = "reads data/host_stalls_2026_10/brick48.y4m, which is not in the repository"]
fn brick_agrees_with_the_host_reading() {
    check_clip(BRICK, "brick");
}

#[test]
#[ignore = "reads data/host_stalls_2026_10/edges63.y4m, which is not in the repository"]
fn edges_agrees_with_the_host_reading() {
    check_clip(EDGES, "edges");
}

#[test]
fn the_per_slot_path_is_byte_identical_across_runs() {
    let plane = Plane::luma(256, 256);
    let frames = synthetic_stream(plane, 9);
    let mut first = ReadingHarness::new(plane, luma_map());
    let mut second = ReadingHarness::new(plane, luma_map());

    for (index, frame) in frames.iter().enumerate() {
        let first_slot = first.push_gpu(frame);
        let second_slot = second.push_gpu(frame);
        let first_words = first_slot.reading.to_words().map(f32::to_bits);
        let second_words = second_slot.reading.to_words().map(f32::to_bits);
        let first_map: Vec<u32> = first_slot.map.iter().map(|value| value.to_bits()).collect();
        let second_map: Vec<u32> = second_slot.map.iter().map(|value| value.to_bits()).collect();

        assert_eq!(first_words, second_words, "frame {index} reading");
        assert_eq!(
            first_slot.block_flags, second_slot.block_flags,
            "frame {index} block flags"
        );
        assert_eq!(
            first_slot.bin_counts, second_slot.bin_counts,
            "frame {index} bins"
        );
        assert_eq!(first_map, second_map, "frame {index} map");
    }

    sync(&first.client).expect("the device syncs");
}

#[test]
#[should_panic(expected = "sigma is inf apart")]
fn a_nan_gpu_value_fails_the_value_check() {
    let mut summary = StreamSummary::default();
    summary.matched.record("sigma", 0.01, 0.01);
    summary.matched.record("sigma", f32::NAN, 0.01);

    summary.assert_agrees("nan");
}

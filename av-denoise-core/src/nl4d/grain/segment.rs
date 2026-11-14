use std::ops::Range;

use super::chroma::{
    ChromaFit,
    ChromaParts,
    ChromaScaling,
    ChromaTexture,
    chroma_parts,
    chroma_scaling,
    chroma_texture_of,
};
use super::chunk::{GrainChunk, TableSums};
use super::consts::{
    AR_COEFFS,
    BUCKETS_PER_GROUP,
    CHROMA_PLANES,
    DRIFT,
    LAG_COUNT,
    LUMA_BINS,
    MAX_POINTS,
    MIN_AR_PIXELS,
    MIN_BLOCKS_PER_BIN,
    MIN_CHUNK_BLOCKS,
    MIN_POPULATED_BINS,
    STD_BUCKETS,
    STRENGTH_GROUPS,
};
use super::fit::{
    bucket_edges,
    code_step,
    finest_ar_shift,
    finest_scaling_shift,
    floor_fraction,
    hist_median,
    quantise_ar,
    quantise_at,
    scaling_peak,
    scaling_points,
    scaling_points_at,
    thinned,
    undo_mean_removal,
    yule_walker,
};
use super::modes::{GrainMode, GrainRun};
use super::repeat::SeedScores;
use super::seeds::{SeedSigmas, fresh_seed};
use super::table::FitOptions;
use super::template::{ChromaPlane, CurveCache, TemplatePlane};

/// The grain chunks of one scene, in frame order.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneGrain {
    /// The output index of the scene's first frame.
    pub first_frame: u64,
    /// The luma grain chunks in frame order.
    pub chunks: Vec<GrainChunk>,
    /// The chroma planes' grain chunks in frame order, empty when chroma is not measured.
    pub chroma_chunks: Vec<GrainChunk>,
}

/// A run of chunks in one scene with steady grain.
pub(crate) struct Segment {
    pub first_frame: u64,
    pub last_frame: u64,
    /// The indices of the scene's chunks the segment covers.
    pub chunks: Range<usize>,
    /// The table statistics of every chunk in the segment, summed.
    pub table: TableSums,
}

/// One table entry's parameters.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FittedEntry {
    pub first_frame: u64,
    pub last_frame: u64,
    /// The seed the decoder draws every plane's grain template from.
    pub seed: u16,
    pub ar_coeffs: [i32; AR_COEFFS],
    pub ar_shift: u32,
    pub scaling_shift: u32,
    pub points: Vec<(u8, u8)>,
    /// The Cb and then the Cr plane, `None` when the entry has no chroma grain.
    pub chroma: Option<[PlaneEntry; CHROMA_PLANES]>,
}

/// One chroma plane's table values.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PlaneEntry {
    pub ar_coeffs: [i32; AR_COEFFS],
    pub points: Vec<(u8, u8)>,
}

/// The one-frame high-pass std of one populated luma bin.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BinStrength {
    pub bin: usize,
    /// The bin's median one-frame high-pass std, in normalised units.
    pub median: f64,
}

/// Quantised AR weights and their shift.
type Weights = ([i32; AR_COEFFS], u32);

/// A segment's fitted texture.
#[derive(Debug, Clone, Copy)]
struct Texture {
    /// The weights the table carries, with the chosen share of the fine noise taken out.
    table: Weights,
    /// The solved weights the table's weights are rounded from.
    table_weights: [f64; AR_COEFFS],
    /// The weights of all the measured grain, whose high-pass curve reads its strength.
    measured: Weights,
}

/// The texture band's lower edge, as a multiple of the median source std.
const BAND_LOW: f64 = 0.5;
/// The texture band's upper edge, as a multiple of the median source std.
const BAND_HIGH: f64 = 2.0;
/// The largest share of the floor taken out of a texture's zero lag.
pub(crate) const MAX_FLOOR_TAKEN: f64 = 0.95;

/// A segment's own measurements, either of which may be missing.
struct Parts {
    first_frame: u64,
    last_frame: u64,
    scene: usize,
    /// The segment's summed luma table statistics.
    table: TableSums,
    strength: Option<Vec<BinStrength>>,
    texture: Option<Texture>,
    /// The Cb and then the Cr plane's own measurements, all missing when chroma is off.
    chroma: [ChromaParts; CHROMA_PLANES],
}

/// A frozen run's textures, pooled over every segment with a frame in the run.
///
/// A plane is `None` only when no segment has a texture for it.
struct RunTexture {
    luma: Option<Texture>,
    /// The Cb and then the Cr plane's pooled texture.
    chroma: [Option<ChromaTexture>; CHROMA_PLANES],
}

/// The frames of one part that lie in one run.
struct Piece {
    first_frame: u64,
    last_frame: u64,
    /// The position of the frozen run the piece lies in, `None` for fresh frames.
    frozen_run: Option<usize>,
}

/// The table's rounded weights and the solved weights behind them.
#[derive(Debug, Clone, Copy)]
struct TableWeights {
    quantised: Weights,
    weights: [f64; AR_COEFFS],
}

/// One part's table values that hold for every seed.
pub(crate) struct EntryPlan {
    ar_coeffs: [i32; AR_COEFFS],
    ar_shift: u32,
    /// Luma's whole-field strength targets, as `(luma, sigma)` with sigma in normalised units.
    targets: Vec<(u8, f64)>,
    /// The Cb and then the Cr plane, `None` when the entry has no chroma grain.
    chroma: Option<[ChromaScaling; CHROMA_PLANES]>,
}

/// The std median over every bin of a histogram.
pub(crate) fn overall_median(hist: &[u32], edges: &[f32]) -> Option<f64> {
    let mut merged = vec![0u32; STD_BUCKETS];
    for bin in 0..LUMA_BINS {
        let row = &hist[bin * STD_BUCKETS..(bin + 1) * STD_BUCKETS];
        for (total, &count) in merged.iter_mut().zip(row.iter()) {
            *total += count;
        }
    }

    hist_median(&merged, edges)
}

/// Splits a scene into segments, closing one when a chunk's grain drifts past [DRIFT].
pub(crate) fn segment_scene(scene: &SceneGrain) -> Vec<Segment> {
    let edges = bucket_edges();
    let mut segments: Vec<Segment> = Vec::new();
    let mut next_frame = scene.first_frame;

    for (index, chunk) in scene.chunks.iter().enumerate() {
        let chunk_first = next_frame;
        let chunk_last = chunk_first + chunk.frames as u64 - 1;
        next_frame = chunk_last + 1;

        let table = chunk.summed_table();
        let starts_new = match segments.last() {
            None => true,
            Some(open) => drifted(&open.table, &table, &edges),
        };
        if starts_new {
            segments.push(Segment {
                first_frame: chunk_first,
                last_frame: chunk_last,
                chunks: index..index + 1,
                table,
            });
            continue;
        }

        let open = segments.last_mut().expect("a segment is open");
        open.last_frame = chunk_last;
        open.chunks.end = index + 1;
        open.table.add(&table);
    }

    segments
}

fn drifted(open: &TableSums, chunk: &TableSums, edges: &[f32]) -> bool {
    if chunk.source_blocks() < MIN_CHUNK_BLOCKS {
        return false;
    }

    let open_median = overall_median(&open.source_hist, edges);
    let chunk_median = overall_median(&chunk.source_hist, edges);
    let (Some(open_median), Some(chunk_median)) = (open_median, chunk_median) else {
        return false;
    };

    let ratio = chunk_median / open_median;
    !(1.0 / DRIFT..=DRIFT).contains(&ratio)
}

/// Per-bin one-frame high-pass medians, `None` when too few bins are populated.
pub(crate) fn strength_of(stats: &TableSums, edges: &[f32]) -> Option<Vec<BinStrength>> {
    let mut bins = Vec::new();
    for bin in 0..LUMA_BINS {
        let counts = &stats.frame_hist[bin * STD_BUCKETS..(bin + 1) * STD_BUCKETS];
        let blocks: u64 = counts.iter().map(|&count| count as u64).sum();
        if blocks < MIN_BLOCKS_PER_BIN {
            continue;
        }

        let median = hist_median(counts, edges).expect("bin is populated");
        bins.push(BinStrength { bin, median });
    }

    (bins.len() >= MIN_POPULATED_BINS).then_some(bins)
}

/// The 8-bit luma at the centre of luma bin `bin`.
pub(crate) fn bin_luma(bin: usize) -> u8 {
    let centre = (bin as f64 + 0.5) * 256.0 / LUMA_BINS as f64;

    centre.round() as u8
}

/// The texture solved from the strength groups near the median source std.
///
/// The pooled record has its cell-mean bias undone before the solve. The table's weights are solved
/// again after `fine_noise` times the floor's variance comes off the zero lag.
fn texture_of(stats: &TableSums, edges: &[f32], fine_noise: f64) -> Option<Texture> {
    let record = pooled_record(stats, edges)?;
    let mut corrected = undo_mean_removal(&record);
    let measured_weights = yule_walker(&corrected)?;
    let floor = floor_fraction(&measured_weights);
    let measured_quantised = quantise_ar(&measured_weights);

    take_out_fine_noise(&mut corrected, floor, fine_noise);
    let measured = TableWeights {
        quantised: measured_quantised,
        weights: measured_weights,
    };
    let reduced = reduced_weights(&corrected, measured);

    Some(Texture {
        table: reduced.quantised,
        table_weights: reduced.weights,
        measured: measured_quantised,
    })
}

/// The summed lag record of the strength groups near the median source std.
///
/// A group joins when its std range overlaps the band between [BAND_LOW] and [BAND_HIGH] times the
/// median, so blocks far from the typical grain strength never shape the texture. `None` when the
/// pooled groups cover fewer than [MIN_AR_PIXELS] pixels.
pub(crate) fn pooled_record(stats: &TableSums, edges: &[f32]) -> Option<Vec<f64>> {
    let median = overall_median(&stats.source_hist, edges)?;
    let band_low = BAND_LOW * median;
    let band_high = BAND_HIGH * median;
    let mut record = vec![0.0f64; LAG_COUNT + 1];

    for group in 0..STRENGTH_GROUPS {
        let group_low = edges[group * BUCKETS_PER_GROUP] as f64;
        let group_high = edges[(group + 1) * BUCKETS_PER_GROUP] as f64;
        if group_high < band_low || group_low > band_high {
            continue;
        }

        for (sum, &extra) in record.iter_mut().zip(stats.group_autocov(group)) {
            *sum += extra;
        }

        record[LAG_COUNT] += stats.pixels[group];
    }

    if record[LAG_COUNT] < MIN_AR_PIXELS {
        return None;
    }

    Some(record)
}

/// Takes `fine_noise` times the floor's share of the zero lag off a corrected record.
pub(crate) fn take_out_fine_noise(corrected: &mut [f64], floor: f64, fine_noise: f64) {
    // At most [MAX_FLOOR_TAKEN] of the floor comes out, so the reduced system stays well conditioned.
    let share = fine_noise.min(MAX_FLOOR_TAKEN) * floor;
    corrected[0] -= share * corrected[0];
}

/// The weights solved from the reduced record.
///
/// When the reduced record has no system to solve, or rounds to the measured weights, the measured
/// weights stay.
fn reduced_weights(corrected: &[f64], measured: TableWeights) -> TableWeights {
    let Some(solved) = yule_walker(corrected) else {
        return measured;
    };

    let reduced = quantise_ar(&solved);
    if reduced == measured.quantised {
        return measured;
    }

    TableWeights {
        quantised: reduced,
        weights: solved,
    }
}

/// Fits every scene's segments, borrowing missing strength or texture, and drops what has no donor.
///
/// Every plane reads its strength from one frame's grain through a high-pass curve rounded at
/// [FitOptions::bit_depth]. With chroma on, each chroma plane borrows its texture and target the
/// same way. An entry gets chroma only when both planes have both.
///
/// Each segment is cut at the edges of `runs`, which cover every frame of every scene's chunks. A
/// fresh frame gets its own entry and seed. A frozen run holds one seed and one texture per plane,
/// with one entry per segment.
pub(crate) fn fit_scenes(
    scenes: &[&SceneGrain],
    options: &FitOptions,
    runs: &[GrainRun],
) -> Vec<FittedEntry> {
    let mut sigmas = SeedSigmas::new();

    fit_scenes_with(scenes, options, runs, &mut sigmas)
}

/// Fits as [fit_scenes] does, reading each seed's template strength through `sigmas`.
pub(crate) fn fit_scenes_with(
    scenes: &[&SceneGrain],
    options: &FitOptions,
    runs: &[GrainRun],
    sigmas: &mut SeedSigmas,
) -> Vec<FittedEntry> {
    let edges = bucket_edges();
    let rounding = options.bit_depth.map(code_step);
    let mut curves = CurveCache::new(rounding);
    let parts = parts_of(scenes, options, &edges);
    let run_textures = run_textures(&parts, runs, options, &edges);

    let mut scores = SeedScores::new();
    let mut run_seeds = vec![None; runs.len()];
    let mut taken = Vec::new();
    let mut entries = Vec::new();
    for index in 0..parts.len() {
        let Some(strength) = borrow(&parts, index, |part| part.strength.clone()) else {
            continue;
        };

        let part = &parts[index];
        let pieces = pieces_of(part.first_frame, part.last_frame, runs);
        for piece in pieces {
            let run_texture = piece.frozen_run.and_then(|run| run_textures[run].as_ref());
            let plan = piece_plan(&parts, index, &strength, run_texture, options, &mut curves);
            let Some(plan) = plan else {
                continue;
            };

            match piece.frozen_run {
                Some(run) => {
                    let seed = run_seed(&mut run_seeds[run], &plan, &mut taken, &mut scores);
                    let entry = seeded_entry(&plan, piece.first_frame, piece.last_frame, seed, sigmas);
                    entries.push(entry);
                },
                None => {
                    for frame in piece.first_frame..=piece.last_frame {
                        let seed = fresh_seed(frame);
                        let entry = seeded_entry(&plan, frame, frame, seed, sigmas);
                        entries.push(entry);
                    }
                },
            }
        }
    }

    entries
}

/// Every scene's segments with their own measurements, in frame order.
fn parts_of(scenes: &[&SceneGrain], options: &FitOptions, edges: &[f32]) -> Vec<Parts> {
    let mut parts = Vec::new();
    for (scene_index, scene) in scenes.iter().enumerate() {
        for segment in segment_scene(scene) {
            let strength = strength_of(&segment.table, edges);
            let texture = texture_of(&segment.table, edges, options.fine_noise);
            let chroma = if options.chroma {
                segment_chroma(scene, &segment, edges)
            } else {
                Default::default()
            };

            parts.push(Parts {
                first_frame: segment.first_frame,
                last_frame: segment.last_frame,
                scene: scene_index,
                table: segment.table,
                strength,
                texture,
                chroma,
            });
        }
    }

    parts
}

/// Each run's pooled textures, `None` for fresh runs.
fn run_textures(
    parts: &[Parts],
    runs: &[GrainRun],
    options: &FitOptions,
    edges: &[f32],
) -> Vec<Option<RunTexture>> {
    runs.iter()
        .map(|run| {
            let frozen = run.mode == GrainMode::Frozen;
            frozen.then(|| run_texture(parts, run, options, edges))
        })
        .collect()
}

/// The textures of every frame in `run`, solved from the summed tables of every part with a frame
/// in it.
///
/// A plane whose pooled solve gives nothing borrows once for the whole run, as the run's first part
/// would. It is `None` only when no part has a texture to give.
fn run_texture(parts: &[Parts], run: &GrainRun, options: &FitOptions, edges: &[f32]) -> RunTexture {
    let in_run = |part: &Parts| part.first_frame <= run.last_frame && part.last_frame >= run.first_frame;
    let mut luma = TableSums::zeroed();
    let mut chroma = [TableSums::zeroed(), TableSums::zeroed()];
    let covered = parts.iter().filter(|part| in_run(part));
    for part in covered {
        luma.add(&part.table);
        for (sums, plane) in chroma.iter_mut().zip(&part.chroma) {
            sums.add(&plane.table);
        }
    }

    let first_part = parts.iter().position(in_run);
    let pooled_luma = texture_of(&luma, edges, options.fine_noise);
    let luma_texture = match (pooled_luma, first_part) {
        (Some(pooled), _) => Some(pooled),
        (None, Some(first)) => borrow(parts, first, |part| part.texture),
        (None, None) => None,
    };

    let [cb_sums, cr_sums] = &chroma;
    let cb = run_plane_texture(parts, first_part, ChromaPlane::Cb, cb_sums, edges);
    let cr = run_plane_texture(parts, first_part, ChromaPlane::Cr, cr_sums, edges);

    RunTexture {
        luma: luma_texture,
        chroma: [cb, cr],
    }
}

/// One chroma plane's texture for a run, pooled from `sums` or borrowed as the run's first part
/// would.
fn run_plane_texture(
    parts: &[Parts],
    first_part: Option<usize>,
    plane: ChromaPlane,
    sums: &TableSums,
    edges: &[f32],
) -> Option<ChromaTexture> {
    let slot = plane.index();
    let pooled = chroma_texture_of(sums, edges);

    match (pooled, first_part) {
        (Some(pooled), _) => Some(pooled),
        (None, Some(first)) => borrow(parts, first, |part| part.chroma[slot].texture),
        (None, None) => None,
    }
}

/// The frames `first_frame..=last_frame` cut at every run edge, in frame order.
fn pieces_of(first_frame: u64, last_frame: u64, runs: &[GrainRun]) -> Vec<Piece> {
    let overlapping = runs
        .iter()
        .enumerate()
        .filter(|(_, run)| run.first_frame <= last_frame && run.last_frame >= first_frame);

    let mut pieces = Vec::new();
    for (run_index, run) in overlapping {
        let frozen = run.mode == GrainMode::Frozen;
        pieces.push(Piece {
            first_frame: run.first_frame.max(first_frame),
            last_frame: run.last_frame.min(last_frame),
            frozen_run: frozen.then_some(run_index),
        });
    }

    let covered: u64 = pieces
        .iter()
        .map(|piece| piece.last_frame - piece.first_frame + 1)
        .sum();
    debug_assert_eq!(
        covered,
        last_frame - first_frame + 1,
        "the runs cover every frame"
    );

    pieces
}

/// The plan of one piece of part `index`, `None` when no texture can be found.
///
/// A frozen piece takes its run's textures in place of the part's own or borrowed ones.
fn piece_plan(
    parts: &[Parts],
    index: usize,
    strength: &[BinStrength],
    run_texture: Option<&RunTexture>,
    options: &FitOptions,
    curves: &mut CurveCache,
) -> Option<EntryPlan> {
    let pooled_luma = run_texture.and_then(|pooled| pooled.luma);
    let texture = match pooled_luma {
        Some(pooled) => pooled,
        None => borrow(parts, index, |part| part.texture)?,
    };

    let chroma = if options.chroma {
        let pooled_chroma = run_texture.map(|pooled| pooled.chroma);
        borrowed_chroma(parts, index, pooled_chroma)
    } else {
        None
    };

    let plan = entry_plan(strength, texture, chroma, options.strength, curves);

    Some(plan)
}

/// The seed a frozen run holds, picked on its first entry.
///
/// The run takes the calmest seed for the plan's weights that no earlier run holds. Once every
/// listed seed is held, it takes the calmest of all of them.
fn run_seed(held: &mut Option<u16>, plan: &EntryPlan, taken: &mut Vec<u16>, scores: &mut SeedScores) -> u16 {
    if let Some(seed) = *held {
        return seed;
    }

    let untaken = scores.calmest(&plan.ar_coeffs, plan.ar_shift, taken);
    let seed = match untaken {
        Some(seed) => seed,
        None => scores
            .calmest(&plan.ar_coeffs, plan.ar_shift, &[])
            .expect("the seed list has seeds"),
    };

    taken.push(seed);
    *held = Some(seed);

    seed
}

/// The chroma measurements of the scene's chroma chunks the segment covers.
///
/// A scene without chroma chunks for the segment gives all of them missing.
fn segment_chroma(scene: &SceneGrain, segment: &Segment, edges: &[f32]) -> [ChromaParts; CHROMA_PLANES] {
    let covered = scene.chroma_chunks.get(segment.chunks.clone());
    let chunks = covered.unwrap_or_default();

    chroma_parts(chunks, edges)
}

/// Both chroma planes' texture and target, each borrowed on its own, `None` when any is missing.
///
/// A plane with a `pooled` texture takes it in place of its own or borrowed one.
fn borrowed_chroma(
    parts: &[Parts],
    index: usize,
    pooled: Option<[Option<ChromaTexture>; CHROMA_PLANES]>,
) -> Option<[ChromaFit; CHROMA_PLANES]> {
    let [pooled_cb, pooled_cr] = pooled.unwrap_or_default();
    let cb = borrowed_plane(parts, index, ChromaPlane::Cb, pooled_cb)?;
    let cr = borrowed_plane(parts, index, ChromaPlane::Cr, pooled_cr)?;

    Some([cb, cr])
}

fn borrowed_plane(
    parts: &[Parts],
    index: usize,
    plane: ChromaPlane,
    pooled: Option<ChromaTexture>,
) -> Option<ChromaFit> {
    let slot = plane.index();
    let texture = match pooled {
        Some(texture) => texture,
        None => borrow(parts, index, |part| part.chroma[slot].texture)?,
    };
    let target = borrow(parts, index, |part| part.chroma[slot].target.clone())?;

    Some(ChromaFit { texture, target })
}

/// The part's own value, else the nearest donor in its scene, else the nearest in any scene.
///
/// Searches outwards in both directions, and the earlier one wins a tie.
fn borrow<T>(parts: &[Parts], index: usize, value: impl Fn(&Parts) -> Option<T>) -> Option<T> {
    if let Some(own) = value(&parts[index]) {
        return Some(own);
    }

    let scene = parts[index].scene;
    for same_scene_only in [true, false] {
        for distance in 1..parts.len() {
            let candidates = [index.checked_sub(distance), index.checked_add(distance)];
            for candidate in candidates.into_iter().flatten() {
                let Some(part) = parts.get(candidate) else {
                    continue;
                };

                if same_scene_only && part.scene != scene {
                    continue;
                }

                if let Some(found) = value(part) {
                    return Some(found);
                }
            }
        }
    }

    None
}

/// The seed-independent values of one part's entries.
///
/// Without chroma, luma's weights are the table's own. With chroma, the AR shift is the finest that
/// fits every plane's weights. Every plane's targets are `dial` times the measured ones.
fn entry_plan(
    strength: &[BinStrength],
    texture: Texture,
    chroma: Option<[ChromaFit; CHROMA_PLANES]>,
    dial: f64,
    curves: &mut CurveCache,
) -> EntryPlan {
    let targets = luma_targets(strength, texture, dial, curves);
    let (luma_coeffs, luma_shift) = texture.table;
    let Some([cb_fit, cr_fit]) = chroma else {
        return EntryPlan {
            ar_coeffs: luma_coeffs,
            ar_shift: luma_shift,
            targets,
            chroma: None,
        };
    };

    let cb_shift = finest_ar_shift(&cb_fit.texture.weights);
    let cr_shift = finest_ar_shift(&cr_fit.texture.weights);
    let ar_shift = luma_shift.min(cb_shift).min(cr_shift);
    let ar_coeffs = if ar_shift == luma_shift {
        luma_coeffs
    } else {
        quantise_at(&texture.table_weights, ar_shift)
    };

    let cb = chroma_scaling(&cb_fit, ChromaPlane::Cb, ar_shift, dial, curves);
    let cr = chroma_scaling(&cr_fit, ChromaPlane::Cr, ar_shift, dial, curves);

    EntryPlan {
        ar_coeffs,
        ar_shift,
        targets,
        chroma: Some([cb, cr]),
    }
}

/// The entry over `first_frame..=last_frame` whose grain the decoder draws from `seed`.
///
/// Each plane's points scale its targets by the std of that seed's template. Without chroma, luma
/// alone picks the scaling shift. With chroma, the scaling shift is the finest that fits every
/// plane's strongest point.
fn seeded_entry(
    plan: &EntryPlan,
    first_frame: u64,
    last_frame: u64,
    seed: u16,
    sigmas: &mut SeedSigmas,
) -> FittedEntry {
    let luma_sigma = sigmas.sigma(&plan.ar_coeffs, plan.ar_shift, TemplatePlane::Luma, seed);
    let Some([cb, cr]) = &plan.chroma else {
        let (points, scaling_shift) = scaling_points(&plan.targets, luma_sigma);

        return FittedEntry {
            first_frame,
            last_frame,
            seed,
            ar_coeffs: plan.ar_coeffs,
            ar_shift: plan.ar_shift,
            scaling_shift,
            points,
            chroma: None,
        };
    };

    let cb_sigma = cb.seed_sigma(seed, sigmas);
    let cr_sigma = cr.seed_sigma(seed, sigmas);
    let luma_peak = scaling_peak(&plan.targets, luma_sigma);
    let cb_peak = cb.peak(cb_sigma);
    let cr_peak = cr.peak(cr_sigma);
    let peak = luma_peak.max(cb_peak).max(cr_peak);
    let scaling_shift = finest_scaling_shift(peak);
    let points = scaling_points_at(&plan.targets, luma_sigma, scaling_shift);
    let cb_entry = cb.entry_at(cb_sigma, scaling_shift);
    let cr_entry = cr.entry_at(cr_sigma, scaling_shift);

    FittedEntry {
        first_frame,
        last_frame,
        seed,
        ar_coeffs: plan.ar_coeffs,
        ar_shift: plan.ar_shift,
        scaling_shift,
        points,
        chroma: Some([cb_entry, cr_entry]),
    }
}

/// Luma's strength targets, as `(luma, sigma)` with sigma in normalised units.
///
/// Each bin's target is the whole-field std whose grain reads the bin's high-pass median, times
/// `dial`. The curve is the measured grain's, before any fine noise comes out.
fn luma_targets(
    strength: &[BinStrength],
    texture: Texture,
    dial: f64,
    curves: &mut CurveCache,
) -> Vec<(u8, f64)> {
    let (coeffs, ar_shift) = texture.measured;
    let curve = curves.curve(&coeffs, ar_shift, TemplatePlane::Luma);

    let chosen = thinned(strength, MAX_POINTS);

    chosen
        .iter()
        .map(|entry| {
            let luma = bin_luma(entry.bin);
            let field_std = curve.field_std(entry.median);
            (luma, field_std * dial)
        })
        .collect()
}

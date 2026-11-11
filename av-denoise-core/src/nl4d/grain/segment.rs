use std::ops::Range;

use super::chroma::{ChromaFit, ChromaParts, chroma_parts, chroma_scaling};
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
use super::table::FitOptions;
use super::template::{ChromaPlane, template_stats};

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

/// The source and kept std medians of one populated luma bin.
#[derive(Debug, Clone, Copy)]
struct BinStrength {
    bin: usize,
    source: f64,
    kept: f64,
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
    /// The template std of the table's weights, in 8-bit units.
    table_sigma: f64,
    /// The measured template's ratio of its whole-field std to its 8x8 block median std.
    ///
    /// The block stds were measured on all the grain, so this converts them to whole-field stds.
    block_to_field: f64,
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
    strength: Option<Vec<BinStrength>>,
    texture: Option<Texture>,
    /// The Cb and then the Cr plane's own measurements, all missing when chroma is off.
    chroma: [ChromaParts; CHROMA_PLANES],
}

/// The table's rounded weights, the solved weights behind them and their template std.
#[derive(Debug, Clone, Copy)]
struct TableWeights {
    quantised: Weights,
    weights: [f64; AR_COEFFS],
    sigma: f64,
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

/// Per-bin strength from the medians, `None` when too few bins are populated.
fn strength_of(stats: &TableSums, edges: &[f32]) -> Option<Vec<BinStrength>> {
    let mut bins = Vec::new();
    for bin in 0..LUMA_BINS {
        let source = &stats.source_hist[bin * STD_BUCKETS..(bin + 1) * STD_BUCKETS];
        let kept = &stats.kept_hist[bin * STD_BUCKETS..(bin + 1) * STD_BUCKETS];
        let source_blocks: u64 = source.iter().map(|&count| count as u64).sum();
        if source_blocks < MIN_BLOCKS_PER_BIN {
            continue;
        }

        let kept_blocks: u64 = kept.iter().map(|&count| count as u64).sum();
        let source_median = hist_median(source, edges).expect("bin is populated");
        let kept_median = if kept_blocks >= MIN_BLOCKS_PER_BIN {
            hist_median(kept, edges).expect("bin is populated")
        } else {
            0.0
        };
        bins.push(BinStrength {
            bin,
            source: source_median,
            kept: kept_median,
        });
    }

    (bins.len() >= MIN_POPULATED_BINS).then_some(bins)
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
    let (measured_coeffs, measured_shift) = quantise_ar(&measured_weights);
    let (measured_sigma, measured_median) = template_stats(&measured_coeffs, measured_shift);
    let block_to_field = measured_sigma / measured_median;

    take_out_fine_noise(&mut corrected, floor, fine_noise);
    let measured = TableWeights {
        quantised: (measured_coeffs, measured_shift),
        weights: measured_weights,
        sigma: measured_sigma,
    };
    let reduced = reduced_weights(&corrected, measured);

    Some(Texture {
        table: reduced.quantised,
        table_weights: reduced.weights,
        table_sigma: reduced.sigma,
        block_to_field,
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

/// The weights solved from the reduced record and their template std.
///
/// When the reduced record has no system to solve, or rounds to the measured weights, the measured
/// weights and std stay.
fn reduced_weights(corrected: &[f64], measured: TableWeights) -> TableWeights {
    let Some(solved) = yule_walker(corrected) else {
        return measured;
    };

    let reduced = quantise_ar(&solved);
    if reduced == measured.quantised {
        return measured;
    }

    let (coeffs, shift) = reduced;
    let (sigma, _) = template_stats(&coeffs, shift);

    TableWeights {
        quantised: reduced,
        weights: solved,
        sigma,
    }
}

/// Fits every scene's segments, borrowing missing strength or texture, and drops what has no donor.
///
/// With chroma on, each chroma plane borrows its texture and target the same way. An entry gets
/// chroma only when both planes have both.
pub(crate) fn fit_scenes(scenes: &[&SceneGrain], options: &FitOptions) -> Vec<FittedEntry> {
    let edges = bucket_edges();
    let mut parts = Vec::new();
    for (scene_index, scene) in scenes.iter().enumerate() {
        for segment in segment_scene(scene) {
            let strength = strength_of(&segment.table, &edges);
            let texture = texture_of(&segment.table, &edges, options.fine_noise);
            let chroma = if options.chroma {
                segment_chroma(scene, &segment, &edges)
            } else {
                Default::default()
            };

            parts.push(Parts {
                first_frame: segment.first_frame,
                last_frame: segment.last_frame,
                scene: scene_index,
                strength,
                texture,
                chroma,
            });
        }
    }

    let mut entries = Vec::new();
    for index in 0..parts.len() {
        let strength = borrow(&parts, index, |part| part.strength.clone());
        let texture = borrow(&parts, index, |part| part.texture);
        let (Some(strength), Some(texture)) = (strength, texture) else {
            continue;
        };

        let chroma = if options.chroma {
            borrowed_chroma(&parts, index)
        } else {
            None
        };

        let entry = build_entry(&parts[index], &strength, texture, chroma);
        entries.push(entry);
    }

    entries
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
fn borrowed_chroma(parts: &[Parts], index: usize) -> Option<[ChromaFit; CHROMA_PLANES]> {
    let cb = borrowed_plane(parts, index, ChromaPlane::Cb)?;
    let cr = borrowed_plane(parts, index, ChromaPlane::Cr)?;

    Some([cb, cr])
}

fn borrowed_plane(parts: &[Parts], index: usize, plane: ChromaPlane) -> Option<ChromaFit> {
    let slot = plane.index();
    let texture = borrow(parts, index, |part| part.chroma[slot].texture)?;
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

/// The entry for one part.
///
/// Without chroma, luma alone picks the shifts. With chroma, the AR shift is the finest that fits
/// every plane's weights and the scaling shift the finest that fits every plane's strongest point.
fn build_entry(
    part: &Parts,
    strength: &[BinStrength],
    texture: Texture,
    chroma: Option<[ChromaFit; CHROMA_PLANES]>,
) -> FittedEntry {
    let targets = luma_targets(strength, texture);
    let Some([cb_fit, cr_fit]) = chroma else {
        let (coeffs, ar_shift) = texture.table;
        let (points, scaling_shift) = scaling_points(&targets, texture.table_sigma);

        return FittedEntry {
            first_frame: part.first_frame,
            last_frame: part.last_frame,
            ar_coeffs: coeffs,
            ar_shift,
            scaling_shift,
            points,
            chroma: None,
        };
    };

    let (luma_coeffs, luma_shift) = texture.table;
    let cb_shift = finest_ar_shift(&cb_fit.texture.weights);
    let cr_shift = finest_ar_shift(&cr_fit.texture.weights);
    let ar_shift = luma_shift.min(cb_shift).min(cr_shift);

    let (coeffs, sigma_template) = if ar_shift == luma_shift {
        (luma_coeffs, texture.table_sigma)
    } else {
        let coeffs = quantise_at(&texture.table_weights, ar_shift);
        let (sigma, _) = template_stats(&coeffs, ar_shift);
        (coeffs, sigma)
    };

    let cb = chroma_scaling(&cb_fit, ChromaPlane::Cb, ar_shift);
    let cr = chroma_scaling(&cr_fit, ChromaPlane::Cr, ar_shift);
    let luma_peak = scaling_peak(&targets, sigma_template);
    let peak = luma_peak.max(cb.peak()).max(cr.peak());
    let scaling_shift = finest_scaling_shift(peak);
    let points = scaling_points_at(&targets, sigma_template, scaling_shift);
    let cb_entry = cb.entry_at(scaling_shift);
    let cr_entry = cr.entry_at(scaling_shift);

    FittedEntry {
        first_frame: part.first_frame,
        last_frame: part.last_frame,
        ar_coeffs: coeffs,
        ar_shift,
        scaling_shift,
        points,
        chroma: Some([cb_entry, cr_entry]),
    }
}

/// Luma's strength targets, as `(luma, sigma)` with sigma in normalised units.
///
/// Each bin keeps the variance the denoiser removed.
fn luma_targets(strength: &[BinStrength], texture: Texture) -> Vec<(u8, f64)> {
    let factor = texture.block_to_field;

    let chosen = thinned(strength, MAX_POINTS);

    chosen
        .iter()
        .map(|entry| {
            let sigma_source = entry.source * factor;
            let sigma_kept = entry.kept * factor;
            let source_variance = sigma_source * sigma_source;
            let kept_variance = sigma_kept * sigma_kept;
            // Only the grain the denoiser removed is synthesised, so the kept grain comes off.
            let removed_variance = (source_variance - kept_variance).max(0.0);
            let centre = (entry.bin as f64 + 0.5) * 256.0 / LUMA_BINS as f64;
            (centre.round() as u8, removed_variance.sqrt())
        })
        .collect()
}

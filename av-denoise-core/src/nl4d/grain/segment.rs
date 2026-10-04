use super::chunk::GrainChunk;
use super::consts::{
    AR_COEFFS,
    BUCKETS_PER_GROUP,
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
use super::fit::{bucket_edges, hist_median, quantise_ar, scaling_points, undo_mean_removal, yule_walker};
use super::template::template_stats;

/// The grain chunks of one scene, in frame order.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneGrain {
    /// The output index of the scene's first frame.
    pub first_frame: u64,
    pub chunks: Vec<GrainChunk>,
}

/// A run of chunks in one scene with steady grain.
pub(crate) struct Segment {
    pub first_frame: u64,
    pub last_frame: u64,
    pub stats: GrainChunk,
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
}

/// The source and kept std medians of one populated luma bin.
#[derive(Debug, Clone, Copy)]
struct BinStrength {
    bin: usize,
    source: f64,
    kept: f64,
}

/// Quantised AR weights and their shift.
type Texture = ([i32; AR_COEFFS], u32);

/// The texture band's lower edge, as a multiple of the median source std.
const BAND_LOW: f64 = 0.5;
/// The texture band's upper edge, as a multiple of the median source std.
const BAND_HIGH: f64 = 2.0;

/// A segment's own measurements, either of which may be missing.
struct Parts {
    first_frame: u64,
    last_frame: u64,
    scene: usize,
    strength: Option<Vec<BinStrength>>,
    texture: Option<Texture>,
}

/// The source std median over every luma bin of a histogram.
fn overall_median(hist: &[u32], edges: &[f32]) -> Option<f64> {
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

    for chunk in &scene.chunks {
        let chunk_first = next_frame;
        let chunk_last = chunk_first + chunk.frames as u64 - 1;
        next_frame = chunk_last + 1;

        let starts_new = match segments.last() {
            None => true,
            Some(open) => drifted(&open.stats, chunk, &edges),
        };
        if starts_new {
            segments.push(Segment {
                first_frame: chunk_first,
                last_frame: chunk_last,
                stats: chunk.clone(),
            });
            continue;
        }

        let open = segments.last_mut().expect("a segment is open");
        open.last_frame = chunk_last;
        open.stats.merge(chunk);
    }

    segments
}

fn drifted(open: &GrainChunk, chunk: &GrainChunk, edges: &[f32]) -> bool {
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
fn strength_of(stats: &GrainChunk, edges: &[f32]) -> Option<Vec<BinStrength>> {
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
/// A group joins when its std range overlaps the band between [BAND_LOW] and [BAND_HIGH] times the
/// median, so blocks far from the typical grain strength never shape the texture. The summed record
/// has its cell-mean bias undone before the solve.
fn texture_of(stats: &GrainChunk, edges: &[f32]) -> Option<Texture> {
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

    let corrected = undo_mean_removal(&record);
    let weights = yule_walker(&corrected)?;
    Some(quantise_ar(&weights))
}

/// Fits every scene's segments, borrowing missing strength or texture, and drops what has no donor.
pub(crate) fn fit_scenes(scenes: &[SceneGrain]) -> Vec<FittedEntry> {
    let edges = bucket_edges();
    let mut parts = Vec::new();
    for (scene_index, scene) in scenes.iter().enumerate() {
        for segment in segment_scene(scene) {
            let strength = strength_of(&segment.stats, &edges);
            let texture = texture_of(&segment.stats, &edges);
            parts.push(Parts {
                first_frame: segment.first_frame,
                last_frame: segment.last_frame,
                scene: scene_index,
                strength,
                texture,
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

        let entry = build_entry(&parts[index], &strength, texture);
        entries.push(entry);
    }

    entries
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

fn build_entry(part: &Parts, strength: &[BinStrength], texture: Texture) -> FittedEntry {
    let (coeffs, ar_shift) = texture;
    let (sigma_template, template_median) = template_stats(&coeffs, ar_shift);
    // The measured stds are medians of mean-removed 8x8 blocks. The template's ratio of its
    // whole-field std to its block median converts them to a whole-field std.
    let factor = sigma_template / template_median;

    let mut chosen: Vec<BinStrength> = strength.to_vec();
    if chosen.len() > MAX_POINTS {
        let last = (chosen.len() - 1) as f64;
        let steps = (MAX_POINTS - 1) as f64;
        chosen = (0..MAX_POINTS)
            .map(|point| (point as f64 * last / steps).round() as usize)
            .map(|position| strength[position])
            .collect();
    }

    let targets: Vec<(u8, f64)> = chosen
        .iter()
        .map(|entry| {
            let sigma_source = entry.source * factor;
            let sigma_kept = entry.kept * factor;
            // Only the grain the denoiser removed is synthesised, so the kept grain comes off.
            let variance = (sigma_source * sigma_source - sigma_kept * sigma_kept).max(0.0);
            let centre = (entry.bin as f64 + 0.5) * 256.0 / LUMA_BINS as f64;
            (centre.round() as u8, variance.sqrt())
        })
        .collect();

    let (points, scaling_shift) = scaling_points(&targets, sigma_template);

    FittedEntry {
        first_frame: part.first_frame,
        last_frame: part.last_frame,
        ar_coeffs: coeffs,
        ar_shift,
        scaling_shift,
        points,
    }
}

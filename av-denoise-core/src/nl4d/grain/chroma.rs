use super::chunk::{GrainChunk, TableSums};
use super::consts::{AR_COEFFS, CHROMA_PLANES, MAX_CHROMA_POINTS, MIN_BLOCKS_PER_BIN, MIN_POPULATED_BINS};
use super::fit::{
    finest_ar_shift,
    quantise_at,
    scaling_peak,
    scaling_points_at,
    undo_mean_removal,
    yule_walker,
};
use super::segment::{PlaneEntry, overall_median, pooled_record};
use super::template::{ChromaPlane, chroma_template_stats};
use crate::noise::colour::{
    COLOUR_BRIGHTNESS_BINS,
    COLOUR_BRIGHTNESS_STEP,
    COLOUR_CELLS,
    COLOUR_CELLS_PER_BRIGHTNESS,
    COLOUR_FIT_PLANE_LUMA_HIGH,
    COLOUR_FIT_PLANE_LUMA_LOW,
    COLOUR_FIT_PLANE_POPULATED,
    COLOUR_FIT_PLANE_WORDS,
};

/// A chroma plane's fitted texture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ChromaTexture {
    /// The solved AR weights, before rounding to AV1 integers.
    pub weights: [f64; AR_COEFFS],
    /// The chroma template's ratio of its whole-field std to its median block std.
    ///
    /// It is read with the weights rounded at their finest AR shift.
    pub block_to_field: f64,
}

/// A segment's own measurements of one chroma plane, either of which may be missing.
#[derive(Debug, Clone, Default)]
pub(crate) struct ChromaParts {
    pub texture: Option<ChromaTexture>,
    /// The removed grain's block std at up to [MAX_CHROMA_POINTS] lumas, in 8-bit codes.
    pub target: Option<Vec<(u8, f64)>>,
}

/// One chroma plane's texture and strength target for a table entry.
#[derive(Debug, Clone)]
pub(crate) struct ChromaFit {
    pub texture: ChromaTexture,
    /// The removed grain's block std at each luma, in 8-bit codes.
    pub target: Vec<(u8, f64)>,
}

/// A chroma plane's weights and whole-field strength targets at an entry's AR shift.
pub(crate) struct ChromaScaling {
    coeffs: [i32; AR_COEFFS],
    /// The whole-field grain std at each luma, in normalised units.
    targets: Vec<(u8, f64)>,
    /// The template std of the plane's weights, in 8-bit units.
    sigma_template: f64,
}

/// Each chroma plane's own measurements over `chunks`, Cb then Cr.
pub(crate) fn chroma_parts(chunks: &[GrainChunk], edges: &[f32]) -> [ChromaParts; CHROMA_PLANES] {
    let mut colour = Vec::new();
    let mut frames = 0usize;
    for chunk in chunks {
        colour.extend_from_slice(&chunk.frame_colour);
        frames += chunk.frames as usize;
    }

    let cb = plane_parts(chunks, &colour, frames, ChromaPlane::Cb, edges);
    let cr = plane_parts(chunks, &colour, frames, ChromaPlane::Cr, edges);

    [cb, cr]
}

/// One plane's texture from its summed table and its target from the colour fit.
///
/// The target needs both the colour fit's levels and the removed share of the table.
fn plane_parts(
    chunks: &[GrainChunk],
    colour: &[f32],
    frames: usize,
    plane: ChromaPlane,
    edges: &[f32],
) -> ChromaParts {
    let mut stats = TableSums::zeroed();
    for chunk in chunks {
        let sums = chunk.summed_chroma_table(plane.index());
        stats.add(&sums);
    }

    let texture = chroma_texture_of(&stats, edges);
    let levels = chroma_levels(colour, frames, plane);
    let share = removed_share(&stats, edges);
    let target = match (levels, share) {
        // The entry's texture, which may be borrowed, turns these block stds into field stds.
        (Some(levels), Some(share)) => Some(chroma_targets(&levels, share, 1.0)),
        _ => None,
    };

    ChromaParts { texture, target }
}

/// The plane's weights rounded at `ar_shift`, its template std and its whole-field targets.
pub(crate) fn chroma_scaling(fit: &ChromaFit, plane: ChromaPlane, ar_shift: u32) -> ChromaScaling {
    let coeffs = quantise_at(&fit.texture.weights, ar_shift);
    let (sigma_template, _) = chroma_template_stats(&coeffs, ar_shift, plane);
    // The targets leave 8-bit codes for the normalised units the luma targets use.
    let factor = fit.texture.block_to_field / 255.0;
    let targets = fit
        .target
        .iter()
        .map(|&(luma, block_std)| (luma, block_std * factor))
        .collect();

    ChromaScaling {
        coeffs,
        targets,
        sigma_template,
    }
}

impl ChromaScaling {
    /// The largest scaling value the plane needs before the scaling shift.
    pub(crate) fn peak(&self) -> f64 {
        scaling_peak(&self.targets, self.sigma_template)
    }

    /// The plane's table values at `scaling_shift`.
    pub(crate) fn entry_at(&self, scaling_shift: u32) -> PlaneEntry {
        let points = scaling_points_at(&self.targets, self.sigma_template, scaling_shift);

        PlaneEntry {
            ar_coeffs: self.coeffs,
            points,
        }
    }
}

/// The texture solved from the strength groups near the median source std, as luma's is.
///
/// The pooled record has its cell-mean bias undone before the solve.
pub(crate) fn chroma_texture_of(stats: &TableSums, edges: &[f32]) -> Option<ChromaTexture> {
    let record = pooled_record(stats, edges)?;
    let corrected = undo_mean_removal(&record);
    let weights = yule_walker(&corrected)?;

    let shift = finest_ar_shift(&weights);
    let coeffs = quantise_at(&weights, shift);
    let (field_std, block_std) = chroma_template_stats(&coeffs, shift, ChromaPlane::Cb);
    let block_to_field = field_std / block_std;

    Some(ChromaTexture {
        weights,
        block_to_field,
    })
}

/// The share of the source grain variance the denoiser removed, between 0 and 1.
///
/// It compares the median kept std with the median source std over every luma bin. `None` with
/// fewer than [MIN_BLOCKS_PER_BIN] source blocks. Fewer kept blocks than that count as no kept
/// grain.
pub(crate) fn removed_share(stats: &TableSums, edges: &[f32]) -> Option<f64> {
    if stats.source_blocks() < MIN_BLOCKS_PER_BIN {
        return None;
    }

    let source = overall_median(&stats.source_hist, edges)?;
    let kept_blocks: u64 = stats.kept_hist.iter().map(|&count| count as u64).sum();
    if kept_blocks < MIN_BLOCKS_PER_BIN {
        return Some(1.0);
    }

    let kept = overall_median(&stats.kept_hist, edges)?;
    let ratio = kept / source;
    let share = 1.0 - ratio * ratio;

    Some(share.max(0.0))
}

/// The grain std of each brightness bin of one chroma plane, read from `frames` frames of colour
/// fit.
///
/// The colour model's log grain is the natural log of a 16x16 block's grain std in 8-bit codes, so
/// each level is a block std on that scale. A bin reads its neutral cell, and its level is the root
/// mean square over the frames. Frames whose plane has no populated cell are skipped. Only the bins
/// between the lowest and highest populated luma of the read frames are kept, each at its centre
/// luma. `None` with fewer than [MIN_POPULATED_BINS] bins.
pub(crate) fn chroma_levels(colour: &[f32], frames: usize, plane: ChromaPlane) -> Option<Vec<(u8, f64)>> {
    let cells = COLOUR_CELLS as usize;
    let bins = COLOUR_BRIGHTNESS_BINS as usize;
    let cells_per_bin = COLOUR_CELLS_PER_BRIGHTNESS as usize;
    let plane_len = 2 * cells + COLOUR_FIT_PLANE_WORDS as usize;
    let frame_len = CHROMA_PLANES * plane_len;
    let plane_start = plane.index() * plane_len;

    let mut energy = vec![0.0f64; bins];
    let mut read_frames = 0usize;
    let mut luma_low = f64::INFINITY;
    let mut luma_high = f64::NEG_INFINITY;

    let frame_values = colour.chunks_exact(frame_len).take(frames);
    for frame in frame_values {
        let plane_values = &frame[plane_start..plane_start + plane_len];
        let (log_grain, rest) = plane_values.split_at(cells);
        let header = &rest[cells..];
        let populated = header[COLOUR_FIT_PLANE_POPULATED as usize];
        if populated <= 0.0 {
            continue;
        }

        let frame_low = header[COLOUR_FIT_PLANE_LUMA_LOW as usize] as f64;
        let frame_high = header[COLOUR_FIT_PLANE_LUMA_HIGH as usize] as f64;
        luma_low = luma_low.min(frame_low);
        luma_high = luma_high.max(frame_high);

        for (bin, total) in energy.iter_mut().enumerate() {
            let neutral = log_grain[bin * cells_per_bin] as f64;
            *total += (2.0 * neutral).exp();
        }

        read_frames += 1;
    }

    if read_frames == 0 {
        return None;
    }

    let step = COLOUR_BRIGHTNESS_STEP as f64;
    let first_bin = (luma_low / step).floor() as usize;
    let highest_bin = (luma_high / step).floor() as usize;
    let last_bin = highest_bin.min(bins - 1);

    let mut levels = Vec::new();
    let kept_bins = energy.iter().enumerate().take(last_bin + 1).skip(first_bin);
    for (bin, &total) in kept_bins {
        let mean_energy = total / read_frames as f64;
        let centre = (bin as f64 + 0.5) * step;
        let luma = centre.round() as u8;
        let level = mean_energy.sqrt();
        levels.push((luma, level));
    }

    (levels.len() >= MIN_POPULATED_BINS).then_some(levels)
}

/// The strength targets of one chroma plane, as whole-field grain stds in 8-bit codes.
///
/// Each block std level becomes a field std through `block_to_field`, and only the removed `share`
/// of its variance is kept. The levels thin evenly to [MAX_CHROMA_POINTS], keeping the first and
/// last.
pub(crate) fn chroma_targets(levels: &[(u8, f64)], share: f64, block_to_field: f64) -> Vec<(u8, f64)> {
    let factor = block_to_field * share.sqrt();
    let chosen = thinned(levels, MAX_CHROMA_POINTS);

    chosen
        .iter()
        .map(|&(luma, level)| (luma, level * factor))
        .collect()
}

/// At most `limit` of the levels, evenly spaced and keeping the first and last.
fn thinned(levels: &[(u8, f64)], limit: usize) -> Vec<(u8, f64)> {
    if levels.len() <= limit {
        return levels.to_vec();
    }

    let last = (levels.len() - 1) as f64;
    let steps = (limit - 1) as f64;

    (0..limit)
        .map(|point| {
            let position = (point as f64 * last / steps).round() as usize;
            levels[position]
        })
        .collect()
}

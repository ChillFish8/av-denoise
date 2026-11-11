use super::chunk::{GrainChunk, TableSums};
use super::consts::{AR_COEFFS, CHROMA_PLANES, MAX_CHROMA_POINTS};
use super::fit::{
    finest_ar_shift,
    quantise_at,
    scaling_peak,
    scaling_points_at,
    thinned,
    undo_mean_removal,
    yule_walker,
};
use super::segment::{BinStrength, PlaneEntry, bin_luma, pooled_record, strength_of};
use super::template::{ChromaPlane, chroma_template_stats};

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
    /// The grain's one-frame block std at up to [MAX_CHROMA_POINTS] lumas, in normalised units.
    pub target: Option<Vec<(u8, f64)>>,
}

/// One chroma plane's texture and strength target for a table entry.
#[derive(Debug, Clone)]
pub(crate) struct ChromaFit {
    pub texture: ChromaTexture,
    /// The grain's one-frame block std at each luma, in normalised units.
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
///
/// With `rounding` holding a code step, each strength reading has that rounding taken off.
pub(crate) fn chroma_parts(
    chunks: &[GrainChunk],
    edges: &[f32],
    rounding: Option<f64>,
) -> [ChromaParts; CHROMA_PLANES] {
    let cb = plane_parts(chunks, ChromaPlane::Cb, edges, rounding);
    let cr = plane_parts(chunks, ChromaPlane::Cr, edges, rounding);

    [cb, cr]
}

/// One plane's texture and target, both from its summed table.
///
/// The target reads the one-frame histogram, binned by luma, and thins evenly to
/// [MAX_CHROMA_POINTS] bins, keeping the first and last.
fn plane_parts(
    chunks: &[GrainChunk],
    plane: ChromaPlane,
    edges: &[f32],
    rounding: Option<f64>,
) -> ChromaParts {
    let mut stats = TableSums::zeroed();
    for chunk in chunks {
        let sums = chunk.summed_chroma_table(plane.index());
        stats.add(&sums);
    }

    let texture = chroma_texture_of(&stats, edges, plane);
    let strength = strength_of(&stats, edges, rounding);
    let target = strength.map(|bins| thinned_targets(&bins));

    ChromaParts { texture, target }
}

/// The `(luma, std)` targets of a plane's bins, thinned to [MAX_CHROMA_POINTS].
fn thinned_targets(bins: &[BinStrength]) -> Vec<(u8, f64)> {
    let chosen = thinned(bins, MAX_CHROMA_POINTS);

    chosen
        .iter()
        .map(|entry| {
            let luma = bin_luma(entry.bin);
            (luma, entry.std)
        })
        .collect()
}

/// The plane's weights rounded at `ar_shift`, its template std and its whole-field targets.
pub(crate) fn chroma_scaling(fit: &ChromaFit, plane: ChromaPlane, ar_shift: u32) -> ChromaScaling {
    let coeffs = quantise_at(&fit.texture.weights, ar_shift);
    let (sigma_template, _) = chroma_template_stats(&coeffs, ar_shift, plane);
    // The entry's texture, which may be borrowed, turns the block stds into field stds.
    let factor = fit.texture.block_to_field;
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
pub(crate) fn chroma_texture_of(
    stats: &TableSums,
    edges: &[f32],
    plane: ChromaPlane,
) -> Option<ChromaTexture> {
    let record = pooled_record(stats, edges)?;
    let corrected = undo_mean_removal(&record);
    let weights = yule_walker(&corrected)?;

    let shift = finest_ar_shift(&weights);
    let coeffs = quantise_at(&weights, shift);
    let (field_std, block_std) = chroma_template_stats(&coeffs, shift, plane);
    let block_to_field = field_std / block_std;

    Some(ChromaTexture {
        weights,
        block_to_field,
    })
}

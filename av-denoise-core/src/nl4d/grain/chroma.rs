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
use super::seeds::SeedSigmas;
use super::segment::{BinStrength, PlaneEntry, bin_luma, pooled_record, strength_of};
use super::template::{ChromaPlane, CurveCache, TemplatePlane};

/// A chroma plane's fitted texture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ChromaTexture {
    /// The solved AR weights, before rounding to AV1 integers.
    pub weights: [f64; AR_COEFFS],
}

/// A segment's own measurements of one chroma plane, either of which may be missing.
#[derive(Debug, Clone, Default)]
pub(crate) struct ChromaParts {
    /// The plane's summed table statistics.
    pub table: TableSums,
    pub texture: Option<ChromaTexture>,
    /// The grain's median one-frame high-pass std at up to [MAX_CHROMA_POINTS] lumas, in
    /// normalised units.
    pub target: Option<Vec<(u8, f64)>>,
}

/// One chroma plane's texture and strength target for a table entry.
#[derive(Debug, Clone)]
pub(crate) struct ChromaFit {
    pub texture: ChromaTexture,
    /// The grain's median one-frame high-pass std at each luma, in normalised units.
    pub target: Vec<(u8, f64)>,
}

/// A chroma plane's weights and whole-field strength targets at an entry's AR shift.
pub(crate) struct ChromaScaling {
    plane: ChromaPlane,
    coeffs: [i32; AR_COEFFS],
    ar_shift: u32,
    /// The whole-field grain std at each luma, in normalised units.
    targets: Vec<(u8, f64)>,
}

/// Each chroma plane's own measurements over `chunks`, Cb then Cr.
pub(crate) fn chroma_parts(chunks: &[GrainChunk], edges: &[f32]) -> [ChromaParts; CHROMA_PLANES] {
    let cb = plane_parts(chunks, ChromaPlane::Cb, edges);
    let cr = plane_parts(chunks, ChromaPlane::Cr, edges);

    [cb, cr]
}

/// One plane's texture and target, both from its summed table.
///
/// The target reads the one-frame histogram, binned by luma, and thins evenly to
/// [MAX_CHROMA_POINTS] bins, keeping the first and last.
fn plane_parts(chunks: &[GrainChunk], plane: ChromaPlane, edges: &[f32]) -> ChromaParts {
    let mut stats = TableSums::zeroed();
    for chunk in chunks {
        let sums = chunk.summed_chroma_table(plane.index());
        stats.add(&sums);
    }

    let texture = chroma_texture_of(&stats, edges);
    let strength = strength_of(&stats, edges);
    let target = strength.map(|bins| thinned_targets(&bins));

    ChromaParts {
        table: stats,
        texture,
        target,
    }
}

/// The `(luma, median)` targets of a plane's bins, thinned to [MAX_CHROMA_POINTS].
fn thinned_targets(bins: &[BinStrength]) -> Vec<(u8, f64)> {
    let chosen = thinned(bins, MAX_CHROMA_POINTS);

    chosen
        .iter()
        .map(|entry| {
            let luma = bin_luma(entry.bin);
            (luma, entry.median)
        })
        .collect()
}

/// The plane's weights rounded at `ar_shift` and its whole-field targets, times `dial`.
///
/// Each target is the field std whose grain reads the target's high-pass median on the plane's
/// template, with the weights rounded at their finest AR shift.
pub(crate) fn chroma_scaling(
    fit: &ChromaFit,
    plane: ChromaPlane,
    ar_shift: u32,
    dial: f64,
    curves: &mut CurveCache,
) -> ChromaScaling {
    let template_plane = TemplatePlane::Chroma(plane);
    let coeffs = quantise_at(&fit.texture.weights, ar_shift);

    // The entry's texture, which may be borrowed, turns the medians into field stds.
    let finest_shift = finest_ar_shift(&fit.texture.weights);
    let finest_coeffs = quantise_at(&fit.texture.weights, finest_shift);
    let curve = curves.curve(&finest_coeffs, finest_shift, template_plane);
    let targets = fit
        .target
        .iter()
        .map(|&(luma, median)| {
            let field_std = curve.field_std(median);
            (luma, field_std * dial)
        })
        .collect();

    ChromaScaling {
        plane,
        coeffs,
        ar_shift,
        targets,
    }
}

impl ChromaScaling {
    /// The template std, in 8-bit units, of the plane's weights drawn from `seed`.
    pub(crate) fn seed_sigma(&self, seed: u16, sigmas: &mut SeedSigmas) -> f64 {
        let template_plane = TemplatePlane::Chroma(self.plane);

        sigmas.sigma(&self.coeffs, self.ar_shift, template_plane, seed)
    }

    /// The largest scaling value the plane needs before the scaling shift, for a template std of
    /// `sigma_template`.
    pub(crate) fn peak(&self, sigma_template: f64) -> f64 {
        scaling_peak(&self.targets, sigma_template)
    }

    /// The plane's table values at `scaling_shift`, for a template std of `sigma_template`.
    pub(crate) fn entry_at(&self, sigma_template: f64, scaling_shift: u32) -> PlaneEntry {
        let points = scaling_points_at(&self.targets, sigma_template, scaling_shift);

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

    Some(ChromaTexture { weights })
}

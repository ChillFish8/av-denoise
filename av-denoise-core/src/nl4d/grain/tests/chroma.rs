use super::synthetic::chunk_at;
use crate::nl4d::grain::chroma::{chroma_levels, chroma_targets, chroma_texture_of};
use crate::nl4d::grain::consts::{BUCKETS_PER_GROUP, LUMA_BINS};
use crate::nl4d::grain::fit::{
    bucket_edges,
    bucket_of,
    finest_ar_shift,
    quantise_at,
    undo_mean_removal,
    yule_walker,
};
use crate::nl4d::grain::template::{ChromaPlane, chroma_template_stats};
use crate::noise::colour::{
    COLOUR_CELLS,
    COLOUR_CELLS_PER_BRIGHTNESS,
    COLOUR_FIT_PLANE_BLOCKS,
    COLOUR_FIT_PLANE_LUMA_HIGH,
    COLOUR_FIT_PLANE_LUMA_LOW,
    COLOUR_FIT_PLANE_POPULATED,
    COLOUR_FIT_PLANE_WORDS,
};

const CELLS: usize = COLOUR_CELLS as usize;
const PLANE_LEN: usize = 2 * CELLS + COLOUR_FIT_PLANE_WORDS as usize;
const FRAME_LEN: usize = 2 * PLANE_LEN;

/// One frame's chroma colour fit with `neutral` log grain in the neutral cells of `bins`.
///
/// Every other cell of both planes holds `ln(5)`. The `plane` header holds `blocks` kept blocks,
/// whether a cell is populated and the populated luma span `luma_span`.
fn colour_frame(
    plane: ChromaPlane,
    neutral: f32,
    bins: std::ops::RangeInclusive<usize>,
    luma_span: (f32, f32),
    header: (f32, bool),
) -> Vec<f32> {
    let other = 5.0f32.ln();
    let mut frame = vec![other; FRAME_LEN];
    for plane_start in [0, PLANE_LEN] {
        let header_start = plane_start + 2 * CELLS;
        frame[header_start + COLOUR_FIT_PLANE_POPULATED as usize] = 1.0;
        frame[header_start + COLOUR_FIT_PLANE_BLOCKS as usize] = 500.0;
        frame[header_start + COLOUR_FIT_PLANE_LUMA_LOW as usize] = 0.0;
        frame[header_start + COLOUR_FIT_PLANE_LUMA_HIGH as usize] = 255.0;
    }

    let plane_start = plane.index() * PLANE_LEN;

    for bin in bins {
        let cell = bin * COLOUR_CELLS_PER_BRIGHTNESS as usize;
        frame[plane_start + cell] = neutral;
    }

    let header_start = plane_start + 2 * CELLS;
    let (blocks, populated) = header;
    let (luma_low, luma_high) = luma_span;
    let populated_word = if populated { 1.0 } else { 0.0 };
    frame[header_start + COLOUR_FIT_PLANE_POPULATED as usize] = populated_word;
    frame[header_start + COLOUR_FIT_PLANE_BLOCKS as usize] = blocks;
    frame[header_start + COLOUR_FIT_PLANE_LUMA_LOW as usize] = luma_low;
    frame[header_start + COLOUR_FIT_PLANE_LUMA_HIGH as usize] = luma_high;

    frame
}

/// A frame whose plane has no populated cell, as the colour fit writes it, with `blocks` kept blocks.
fn unpopulated_frame(plane: ChromaPlane, blocks: f32) -> Vec<f32> {
    let mut frame = colour_frame(plane, 0.0, 0..=LUMA_BINS - 1, (0.0, 255.0), (blocks, false));
    let plane_start = plane.index() * PLANE_LEN;

    for value in &mut frame[plane_start..plane_start + 2 * CELLS] {
        *value = 0.0;
    }

    frame
}

/// A populated frame of `plane` with `neutral` log grain in bins 2 to 12.
fn populated_frame(plane: ChromaPlane, neutral: f32) -> Vec<f32> {
    colour_frame(plane, neutral, 2..=12, (40.0, 200.0), (600.0, true))
}

fn level_lumas(levels: &[(u8, f64)]) -> Vec<u8> {
    levels.iter().map(|&(luma, _)| luma).collect()
}

#[test]
fn chroma_texture_matches_the_luma_fit_of_the_same_sums() {
    let edges = bucket_edges();
    let chunk = chunk_at(2.0, 300, None);
    let stats = chunk.summed_table();
    let group = bucket_of(2.0 / 255.0, &edges) / BUCKETS_PER_GROUP;
    let mut record = stats.group_autocov(group).to_vec();
    record.push(stats.pixels[group]);

    let corrected = undo_mean_removal(&record);
    let expected = yule_walker(&corrected).expect("a well-conditioned system");
    let texture = chroma_texture_of(&stats, &edges, ChromaPlane::Cb).expect("the chunk has a texture");

    assert_eq!(texture.weights, expected);
    assert!(texture.block_to_field > 1.0, "{}", texture.block_to_field);
}

#[test]
fn chroma_texture_reads_its_own_plane_template() {
    let edges = bucket_edges();
    let chunk = chunk_at(2.0, 300, None);
    let stats = chunk.summed_table();

    for plane in [ChromaPlane::Cb, ChromaPlane::Cr] {
        let texture = chroma_texture_of(&stats, &edges, plane).expect("the chunk has a texture");
        let shift = finest_ar_shift(&texture.weights);
        let coeffs = quantise_at(&texture.weights, shift);
        let (field_std, block_std) = chroma_template_stats(&coeffs, shift, plane);

        assert_eq!(texture.block_to_field, field_std / block_std);
    }
}

#[test]
fn chroma_levels_read_the_neutral_cells() {
    let neutral = 0.8f32.ln();
    let frame = populated_frame(ChromaPlane::Cr, neutral);
    let colour = frame.repeat(3);

    let levels = chroma_levels(&colour, 3, ChromaPlane::Cr).expect("enough bins");
    let expected_lumas: Vec<u8> = (2..=12).map(|bin| bin * 16 + 8).collect();

    assert_eq!(level_lumas(&levels), expected_lumas);
    for &(luma, level) in &levels {
        assert!((level - 0.8).abs() < 1e-5, "luma {luma}: {level}");
    }
}

#[test]
fn chroma_levels_are_the_root_mean_square_over_frames() {
    let low = populated_frame(ChromaPlane::Cb, 0.6f32.ln());
    let high = populated_frame(ChromaPlane::Cb, 0.8f32.ln());
    let colour = [low, high].concat();

    let levels = chroma_levels(&colour, 2, ChromaPlane::Cb).expect("enough bins");
    let expected = 0.5f64.sqrt();

    for &(luma, level) in &levels {
        assert!((level - expected).abs() < 1e-5, "luma {luma}: {level}");
    }
}

#[test]
fn chroma_levels_skip_empty_frames() {
    let neutral = 0.8f32.ln();
    let frame = populated_frame(ChromaPlane::Cb, neutral);
    let empty = unpopulated_frame(ChromaPlane::Cb, 0.0);
    let colour = frame.repeat(2);
    let mut with_empty = colour.clone();
    with_empty.extend_from_slice(&empty);

    let levels = chroma_levels(&colour, 2, ChromaPlane::Cb);
    let levels_with_empty = chroma_levels(&with_empty, 3, ChromaPlane::Cb);

    assert!(levels.is_some());
    assert_eq!(levels_with_empty, levels);
}

#[test]
fn chroma_levels_skip_frames_without_populated_cells() {
    let neutral = 0.8f32.ln();
    let frame = populated_frame(ChromaPlane::Cb, neutral);
    let sparse = unpopulated_frame(ChromaPlane::Cb, 40.0);
    let colour = frame.repeat(2);
    let mut with_sparse = colour.clone();
    with_sparse.extend_from_slice(&sparse);

    let levels = chroma_levels(&colour, 2, ChromaPlane::Cb);
    let levels_with_sparse = chroma_levels(&with_sparse, 3, ChromaPlane::Cb);

    assert!(levels.is_some());
    assert_eq!(levels_with_sparse, levels);
}

#[test]
fn chroma_levels_drop_bins_outside_the_populated_luma_range() {
    let neutral = 0.8f32.ln();
    let first = colour_frame(ChromaPlane::Cb, neutral, 0..=15, (72.0, 120.0), (600.0, true));
    let second = colour_frame(ChromaPlane::Cb, neutral, 0..=15, (100.0, 150.0), (600.0, true));
    let colour = [first, second].concat();

    let levels = chroma_levels(&colour, 2, ChromaPlane::Cb).expect("enough bins");

    assert_eq!(level_lumas(&levels), vec![72, 88, 104, 120, 136, 152]);
}

#[test]
fn chroma_levels_need_populated_bins() {
    let neutral = 0.8f32.ln();
    let frame = colour_frame(ChromaPlane::Cb, neutral, 0..=15, (40.0, 60.0), (600.0, true));

    let levels = chroma_levels(&frame, 1, ChromaPlane::Cb);

    assert_eq!(levels, None);
}

#[test]
fn chroma_targets_thin_to_ten_points() {
    let levels: Vec<(u8, f64)> = (0..13u8)
        .map(|bin| (bin * 16 + 8, f64::from(bin) + 1.0))
        .collect();

    let targets = chroma_targets(&levels);
    let kept_bins = [0u8, 1, 3, 4, 5, 7, 8, 9, 11, 12];

    assert_eq!(targets.len(), 10);
    for (&(luma, target), &bin) in targets.iter().zip(&kept_bins) {
        let expected = f64::from(bin) + 1.0;
        assert_eq!(luma, bin * 16 + 8);
        assert!((target - expected).abs() < 1e-12, "luma {luma}: {target}");
    }
}

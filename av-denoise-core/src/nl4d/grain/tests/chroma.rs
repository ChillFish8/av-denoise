use super::synthetic::{chroma_chunk, chunk_at, set_frame_count, table_chunk};
use crate::nl4d::grain::chroma::{chroma_parts, chroma_texture_of};
use crate::nl4d::grain::consts::{BUCKETS_PER_GROUP, STD_BUCKETS};
use crate::nl4d::grain::fit::{
    bucket_edges,
    bucket_of,
    finest_ar_shift,
    hist_median,
    quantise_at,
    undo_mean_removal,
    yule_walker,
};
use crate::nl4d::grain::template::{ChromaPlane, chroma_template_stats};

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
fn chroma_targets_read_the_luma_binned_frame_hist() {
    let edges = bucket_edges();
    let cb_bucket = bucket_of(0.8 / 255.0, &edges);
    let mut cb = table_chunk();
    for bin in 2..=12 {
        set_frame_count(&mut cb, bin * STD_BUCKETS + cb_bucket, 300);
    }

    let cr = table_chunk();
    let chunk = chroma_chunk(&cb, &cr);
    let chunks = [chunk];

    let [cb_parts, cr_parts] = chroma_parts(&chunks, &edges, None);
    let targets = cb_parts.target.expect("the Cb plane has a target");
    let mut counts = vec![0u32; STD_BUCKETS];
    counts[cb_bucket] = 300;
    let median = hist_median(&counts, &edges).expect("a populated bucket");
    let kept_bins = [2u8, 3, 4, 5, 6, 8, 9, 10, 11, 12];

    assert!(cr_parts.target.is_none());
    assert_eq!(targets.len(), 10);
    for (&(luma, target), &bin) in targets.iter().zip(&kept_bins) {
        assert_eq!(luma, bin * 16 + 8);
        assert!((target - median).abs() < 1e-12, "luma {luma}: {target}");
    }
}

use super::{Setup, chroma_cross_frame_setup, cross_frame_setup, unique_frame, yuv_cross_frame_setup};
use crate::collab::tests::helpers::{cel_frame, constant_frame, letterboxed_frame, noisy_flat_field};
use crate::nl4d::nl4d_pool_ratio;
use crate::nlmeans::ChannelMode;

/// A ring of `2 * radius + 1` copies of `frame`, with the cross-frame motion and full confidence.
fn repeated_ring(frame: &[f32], width: u32, height: u32, radius: u32) -> Setup {
    let mut setup = cross_frame_setup(width, height, radius);
    let frames = (2 * radius + 1) as usize;
    setup.ring = frame.repeat(frames);
    setup.confidence.fill(1.0);

    setup
}

/// Every configuration whose output the exact changes must keep.
///
/// The flat entries tie everywhere, so they decide whether a changed search breaks ties the same way.
pub(super) fn collab_corpus() -> Vec<(&'static str, Setup)> {
    let mut corpus = Vec::new();

    let unique = unique_frame(128, 96);
    let unique_setup = Setup::spatial_only(unique, 128, 96);
    corpus.push(("unique", unique_setup));

    let ragged = unique_frame(104, 96);
    let ragged_setup = Setup::spatial_only(ragged, 104, 96);
    corpus.push(("ragged", ragged_setup));

    let short = unique_frame(64, 64);
    let mut short_search = Setup::spatial_only(short, 64, 64);
    short_search.spatial_radius = 1;
    corpus.push(("short_search", short_search));

    let noise = noisy_flat_field(64, 64, 0.5, 0.05);
    let noise_setup = Setup::spatial_only(noise, 64, 64);
    corpus.push(("noise", noise_setup));

    let cross_frame_r1 = cross_frame_setup(64, 64, 1);
    corpus.push(("cross_frame_r1", cross_frame_r1));

    let cross_frame_r2 = cross_frame_setup(64, 64, 2);
    corpus.push(("cross_frame_r2", cross_frame_r2));

    let mut all_gated = cross_frame_setup(64, 64, 2);
    all_gated.c_min = 2.0;
    corpus.push(("all_gated", all_gated));

    let mut f16_luma = cross_frame_setup(64, 64, 2);
    f16_luma.f16_search = true;
    corpus.push(("f16_luma", f16_luma));

    let chroma = chroma_cross_frame_setup(64, 64, 2);
    corpus.push(("chroma", chroma));

    let mut f16_chroma = chroma_cross_frame_setup(64, 64, 2);
    f16_chroma.f16_search = true;
    corpus.push(("f16_chroma", f16_chroma));

    let yuv = yuv_cross_frame_setup(64, 64, 2);
    corpus.push(("yuv", yuv));

    let mut pooled_luma = cross_frame_setup(64, 64, 2);
    let luma_ratio = nl4d_pool_ratio(ChannelMode::Luma);
    pooled_luma.pooled = Some(luma_ratio);
    corpus.push(("pooled_luma", pooled_luma));

    let mut pooled_chroma = chroma_cross_frame_setup(64, 64, 2);
    let chroma_ratio = nl4d_pool_ratio(ChannelMode::Chroma);
    pooled_chroma.pooled = Some(chroma_ratio);
    corpus.push(("pooled_chroma", pooled_chroma));

    let constant = constant_frame(64, 64, 0.5);
    let constant_cross_frame = repeated_ring(&constant, 64, 64, 2);
    let constant_spatial = Setup::spatial_only(constant, 64, 64);
    corpus.push(("constant_spatial", constant_spatial));
    corpus.push(("constant_cross_frame", constant_cross_frame));

    let letterbox = letterboxed_frame(96, 64, 16);
    let letterbox_setup = repeated_ring(&letterbox, 96, 64, 2);
    corpus.push(("letterbox", letterbox_setup));

    let cel = cel_frame(96, 64, 12);
    let mut cel_setup = repeated_ring(&cel, 96, 64, 2);
    cel_setup.spatial_radius = 9;
    corpus.push(("cel", cel_setup));

    corpus
}

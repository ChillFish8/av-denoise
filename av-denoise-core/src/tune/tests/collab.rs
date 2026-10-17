use std::collections::HashSet;

use cubecl::prelude::*;

use super::engine::{nl4d_outputs, noisy_frames};
use super::equivalence::assert_listed_candidates_match_baseline;
use super::fixtures::{CollabSpec, collab_site, collab_site_f16, variant_chunk_tests};
use super::probe::small_device_limits;
use super::runtime::{TestRuntime, test_client};
use crate::Nl4dOptions;
use crate::collab::geometry::refs_along;
use crate::collab::{PATCH_SIZE, STEP, needs_warp_uniform_search, supports_f16_search};
use crate::nlmeans::ChannelMode;
use crate::tune::gpu::{read_f32, sync};
use crate::tune::session::timed_count;
use crate::tune::sites::{
    COLLAB_FUSED,
    Centre,
    CollabOutputs,
    CollabVariant,
    NeighbourWalk,
    Walk,
    tile_height,
    tile_width,
};
use crate::tune::{DeviceLimits, KernelPlan, SiteCandidate, VariantSite, fit_candidates};

fn luma_spec() -> CollabSpec {
    CollabSpec::new(64, 48, ChannelMode::Luma, 2)
}

#[test]
fn collab_variant_names_round_trip_and_are_unique() {
    let mut seen = HashSet::new();

    for &variant in CollabVariant::ALL {
        let name = variant.name();
        let parsed = CollabVariant::from_name(name);
        assert_eq!(parsed, Some(variant), "{name}");
        assert!(seen.insert(name), "{name} names two variants");
    }

    assert_eq!(CollabVariant::baseline(), CollabVariant::BASELINE);
    assert_eq!(CollabVariant::ALL[0], CollabVariant::BASELINE);
    assert_eq!(CollabVariant::BASELINE.name(), "groups8_clipped");
}

#[test]
fn the_listed_variants_cross_every_dimension() {
    assert_eq!(CollabVariant::ALL.len(), 60);

    for groups in [8u32, 4, 16] {
        for walk in [Walk::Clipped, Walk::Uniform] {
            for centre in [Centre::Global, Centre::Tiled, Centre::Sliding] {
                for batched in [false, true] {
                    for neighbour in [NeighbourWalk::Rows, NeighbourWalk::Sliding] {
                        let variant = CollabVariant {
                            groups,
                            walk,
                            centre,
                            batched,
                            neighbour,
                        };
                        let listed = CollabVariant::ALL.contains(&variant);
                        let builds = !(centre == Centre::Sliding && batched);
                        assert_eq!(listed, builds, "{variant:?}");
                    }
                }
            }
        }
    }

    let everything = CollabVariant {
        groups: 16,
        walk: Walk::Uniform,
        centre: Centre::Tiled,
        batched: true,
        neighbour: NeighbourWalk::Sliding,
    };
    assert_eq!(everything.name(), "groups16_uniform_tiled_batched_nslide");

    let sliding = CollabVariant {
        centre: Centre::Sliding,
        ..CollabVariant::BASELINE
    };
    assert_eq!(sliding.name(), "groups8_clipped_sliding");

    let sliding_neighbours = CollabVariant {
        neighbour: NeighbourWalk::Sliding,
        ..CollabVariant::BASELINE
    };
    assert_eq!(sliding_neighbours.name(), "groups8_clipped_nslide");
}

/// Before fit checks the uniform walk holds half the listed variants.
#[test]
fn a_uniform_only_runtime_is_offered_half_the_variants() {
    let uniform_only_count = CollabVariant::ALL
        .iter()
        .filter(|variant| variant.runs_where(true))
        .count();

    assert_eq!(uniform_only_count, 30);
}

/// The staged site times every fitted variant at 8 groups, then 2 more group counts for each of the
/// 3 fastest. That is 20 and then 6, 26 in all, or 10 and then 6 where only the uniform walk runs.
#[test]
fn the_staged_collab_site_times_the_first_stage_and_three_regroupings() {
    let client = test_client();
    let spec = luma_spec();
    let site = collab_site(&client, &spec);
    let limits = DeviceLimits::from_client(&client);
    let uniform_only = needs_warp_uniform_search(&client);

    let fitted = fit_candidates(&site, &limits).expect("the baseline fits");
    let at_eight_groups = fitted.iter().filter(|variant| variant.groups == 8).count();
    let timed = timed_count(&site, &fitted);
    let expected_first = if uniform_only { 10 } else { 20 };

    assert_eq!(at_eight_groups, expected_first);
    assert_eq!(timed, at_eight_groups + 3 * 2);
}

#[test]
fn the_baseline_walk_follows_the_runtime() {
    let uniform = CollabVariant::baseline_for(true);
    let mixed = CollabVariant::baseline_for(false);

    assert_eq!(mixed, CollabVariant::BASELINE);
    assert_eq!(uniform.walk, Walk::Uniform);
    assert_eq!(uniform.groups, CollabVariant::BASELINE.groups);
    assert_eq!(uniform.name(), "groups8_uniform");
}

#[test]
fn a_uniform_only_runtime_runs_only_uniform_walks() {
    for &variant in CollabVariant::ALL {
        let name = variant.name();
        let uniform = variant.walk == Walk::Uniform;
        let runs_anywhere = variant.runs_where(false);
        let runs_uniform_only = variant.runs_where(true);

        assert!(runs_anywhere, "{name}");
        assert_eq!(runs_uniform_only, uniform, "{name}");
    }
}

#[test]
fn this_runtime_is_offered_its_baseline_first_and_only_legal_walks() {
    let client = test_client();
    let spec = luma_spec();
    let site = collab_site(&client, &spec);
    let limits = DeviceLimits::from_client(&client);
    let uniform_only = needs_warp_uniform_search(&client);

    let fitted = fit_candidates(&site, &limits).expect("the baseline fits");
    let expected_baseline = CollabVariant::baseline_for(uniform_only);
    let legal: Vec<CollabVariant> = CollabVariant::ALL
        .iter()
        .copied()
        .filter(|variant| variant.runs_where(uniform_only))
        .collect();

    assert_eq!(fitted[0], expected_baseline);
    assert_eq!(fitted.len(), legal.len());

    for variant in &fitted {
        let name = variant.name();
        assert!(legal.contains(variant), "{name} is not legal here");
    }
}

/// Each 8-lane group owns a 65-float transpose region, and the global and sliding centres keep a
/// one-element placeholder tile.
#[test]
fn collab_shared_bytes_count_the_transpose_regions_and_the_tile() {
    let client = test_client();
    let spec = luma_spec();
    let site = collab_site(&client, &spec);
    let groups16 = CollabVariant {
        groups: 16,
        ..CollabVariant::BASELINE
    };
    let tiled = CollabVariant {
        centre: Centre::Tiled,
        ..CollabVariant::BASELINE
    };

    let baseline_bytes = site.shared_bytes(CollabVariant::BASELINE);
    let groups16_bytes = site.shared_bytes(groups16);
    let tiled_bytes = site.shared_bytes(tiled);
    let sliding = CollabVariant {
        centre: Centre::Sliding,
        ..CollabVariant::BASELINE
    };
    let sliding_bytes = site.shared_bytes(sliding);
    let baseline_units = site.shape(CollabVariant::BASELINE).units();

    // A luma tile at radius 9 is 54 by 26 pixels of one f32 lane.
    assert_eq!(baseline_bytes, 8 * 65 * 4 + 4);
    assert_eq!(groups16_bytes, 16 * 65 * 4 + 4);
    assert_eq!(tiled_bytes, 8 * 65 * 4 + 54 * 26 * 4);
    assert_eq!(sliding_bytes, baseline_bytes);
    assert_eq!(baseline_units, 64);
}

#[test]
fn the_tile_spans_every_reference_window_of_the_cube() {
    assert_eq!(tile_width(8, 9), 54);
    assert_eq!(tile_width(16, 9), 86);
    assert_eq!(tile_height(9), 26);

    for width in [40u32, 64, 70, 104, 129, 1920] {
        let refs = refs_along(width);
        let max_x = width - PATCH_SIZE;

        for groups in [4u32, 8, 16] {
            for spatial_radius in [1u32, 9, 16] {
                let span = tile_width(groups, spatial_radius);
                let cubes = refs.div_ceil(groups);

                for cube in 0..cubes {
                    let first_ref = (cube * groups * STEP).min(max_x);
                    let origin = clamp_window_edge(first_ref, -(spatial_radius as i32), max_x);

                    for group in 0..groups {
                        let ref_index = (cube * groups + group).min(refs - 1);
                        let ref_x = (ref_index * STEP).min(max_x);
                        let left = clamp_window_edge(ref_x, -(spatial_radius as i32), max_x);
                        let right = clamp_window_edge(ref_x, spatial_radius as i32, max_x);
                        let last_column = right + PATCH_SIZE - 1;
                        let label =
                            format!("width {width} groups {groups} radius {spatial_radius} cube {cube}");

                        assert!(left >= origin, "{label}: the window starts left of the tile");
                        assert!(
                            last_column < origin + span,
                            "{label}: the window ends right of the tile"
                        );
                    }
                }
            }
        }
    }
}

/// `position + offset` clamped to the top-left positions a patch can take, as the kernel clamps a
/// search window's edge.
fn clamp_window_edge(position: u32, offset: i32, max_pos: u32) -> u32 {
    let shifted = position as i32 + offset;
    shifted.clamp(0, max_pos as i32) as u32
}

/// The tile holds every stored lane, so a radius 9 yuv tile at 16 groups is 86 by 26 pixels of
/// four lanes. That is 35,776 bytes in f32 and half that in f16, plus 4,160 bytes of transpose.
#[test]
fn the_tile_holds_every_stored_lane_in_the_search_precision() {
    let client = test_client();
    let spec = CollabSpec::new(64, 48, ChannelMode::Yuv, 1);
    let tiled16 = CollabVariant {
        groups: 16,
        centre: Centre::Tiled,
        ..CollabVariant::BASELINE
    };

    let f32_site = collab_site(&client, &spec);
    let f32_bytes = f32_site.shared_bytes(tiled16);
    assert_eq!(f32_bytes, 39_936);

    let f16_site = collab_site_f16(&client, &spec);
    let f16_bytes = f16_site.shared_bytes(tiled16);
    assert_eq!(f16_bytes, 22_048);

    // The HIP runtime crashes at process exit while an upload is still queued.
    sync(&client).expect("the uploads finish");
}

#[test]
fn collab_keys_name_every_compiled_value() {
    let client = test_client();
    let spec = luma_spec();
    let site = collab_site(&client, &spec);

    let key = site.key();

    assert_eq!(
        key.as_str(),
        "collab_fused|w64|h64|ch=1|stored=1|radius=2|grid=4|refine=2|blksize=16|step=8|kmax=8|spatial=9|pooled=0|f16=0"
    );
}

/// The baseline launches on any plane width, so only the other variants drop.
#[test]
fn collab_baseline_fits_planes_narrower_than_eight_lanes() {
    let client = test_client();
    let spec = luma_spec();
    let site = collab_site(&client, &spec);
    let mut limits = small_device_limits();
    limits.plane_size_min = 4;

    let fitted = fit_candidates(&site, &limits).expect("the baseline skips the plane check");
    let baseline_plane = site.min_plane_size(CollabVariant::BASELINE);
    let baseline = site.baseline();

    assert_eq!(baseline_plane, 8);
    assert_eq!(fitted, vec![baseline]);
}

/// The sliding centre needs no tile, so it fits a 32 KB device at every group count.
#[test]
fn sliding_centres_fit_a_32kb_device() {
    let client = test_client();
    let spec = luma_spec();
    let site = collab_site(&client, &spec);
    let limits = small_device_limits();

    let fitted = fit_candidates(&site, &limits).expect("the baseline fits");

    for groups in [4u32, 8, 16] {
        let offered = fitted
            .iter()
            .any(|variant| variant.centre == Centre::Sliding && variant.groups == groups);
        assert!(offered, "no sliding centre with {groups} groups fits 32 KB");
    }
}

#[test]
fn sixteen_groups_still_fit_a_32kb_device() {
    let client = test_client();
    let spec = luma_spec();
    let site = collab_site(&client, &spec);
    let limits = small_device_limits();
    let groups16 = CollabVariant {
        groups: 16,
        ..site.baseline()
    };

    let fitted = fit_candidates(&site, &limits).expect("the baseline fits");

    assert!(fitted.contains(&groups16));
}

#[test]
fn a_32kb_device_drops_only_the_tiles_that_do_not_fit() {
    let client = test_client();
    let spec = CollabSpec::new(64, 48, ChannelMode::Yuv, 1);
    let site = collab_site(&client, &spec);
    let limits = small_device_limits();
    let uniform_only = needs_warp_uniform_search(&client);

    let fitted = fit_candidates(&site, &limits).expect("the baseline fits");

    for &variant in CollabVariant::ALL {
        let too_big = variant.groups == 16 && variant.centre == Centre::Tiled;
        let expected = !too_big && variant.runs_where(uniform_only);
        let offered = fitted.contains(&variant);
        let name = variant.name();
        assert_eq!(offered, expected, "{name}");
    }
}

/// At spatial radius 16 a yuv f32 tile at 16 groups needs 68,160 bytes, past a 64 KB device, while
/// the 8-group tile needs 45,600.
#[test]
fn radius_16_drops_the_widest_tile_on_a_64kb_device() {
    let client = test_client();
    let mut spec = CollabSpec::new(64, 48, ChannelMode::Yuv, 1);
    spec.spatial_radius = 16;
    let site = collab_site(&client, &spec);
    let limits = DeviceLimits {
        max_shared_bytes: 64 * 1024,
        ..small_device_limits()
    };
    let tiled8 = CollabVariant {
        centre: Centre::Tiled,
        ..site.baseline()
    };
    let tiled16 = CollabVariant { groups: 16, ..tiled8 };

    let fitted = fit_candidates(&site, &limits).expect("the baseline fits");
    let tiled16_bytes = site.shared_bytes(tiled16);

    assert_eq!(tiled16_bytes, 68_160);
    assert!(fitted.contains(&tiled8));
    assert!(!fitted.contains(&tiled16));
}

/// Stage 0 holds only 8-group variants, and stage 1 around a seed changes only its group count.
#[test]
fn collab_stages_vary_the_search_first_and_the_group_count_second() {
    let client = test_client();
    let spec = luma_spec();
    let site = collab_site(&client, &spec);
    let seed = CollabVariant::from_name("groups8_uniform").expect("a listed variant");

    let first_stage: Vec<&str> = CollabVariant::ALL
        .iter()
        .copied()
        .filter(|&variant| site.in_stage(0, CollabVariant::BASELINE, variant))
        .map(|variant| variant.name())
        .collect();
    let mut regrouped: Vec<&str> = CollabVariant::ALL
        .iter()
        .copied()
        .filter(|&variant| site.in_stage(1, seed, variant))
        .map(|variant| variant.name())
        .collect();
    regrouped.sort_unstable();

    assert!(
        first_stage.iter().all(|name| name.starts_with("groups8_")),
        "{first_stage:?}"
    );
    assert_eq!(first_stage.len(), CollabVariant::ALL.len() / 3);
    assert_eq!(
        regrouped,
        vec!["groups16_uniform", "groups4_uniform", "groups8_uniform"]
    );
}

fn assert_groups_retired(client: &ComputeClient<TestRuntime>, outputs: &CollabOutputs, label: &str) {
    let weights = read_f32(client, &outputs.group_weight, outputs.refs).expect("weights read back");
    let retired = weights.iter().any(|&weight| weight > 0.0);

    assert!(retired, "{label}: the baseline retired no group");
}

/// Checks each of `variants` that is legal and fits this device against the baseline, bit for bit.
///
/// A device without f16 support has no f16 case to check.
fn assert_variants_match_baseline(spec: CollabSpec, variants: &[CollabVariant]) {
    let client = test_client();
    let label = format!("{spec:?}");

    if spec.f16_search {
        let supported = supports_f16_search(&client);
        if !supported {
            return;
        }

        let site = collab_site_f16(&client, &spec);
        let baseline = assert_listed_candidates_match_baseline(&site, &label, variants);
        assert_groups_retired(&client, &baseline, &label);
        return;
    }

    let site = collab_site(&client, &spec);
    let baseline = assert_listed_candidates_match_baseline(&site, &label, variants);
    assert_groups_retired(&client, &baseline, &label);
}

fn odd_width_luma_spec() -> CollabSpec {
    CollabSpec::new(70, 45, ChannelMode::Luma, 2)
}

variant_chunk_tests! {
    assert_variants_match_baseline(odd_width_luma_spec()), 4 chunks,
    [
        luma_at_an_odd_width_across_frames_matches_the_baseline_in_the_first_variant_quarter => 0,
        luma_at_an_odd_width_across_frames_matches_the_baseline_in_the_second_variant_quarter => 1,
        luma_at_an_odd_width_across_frames_matches_the_baseline_in_the_third_variant_quarter => 2,
        luma_at_an_odd_width_across_frames_matches_the_baseline_in_the_fourth_variant_quarter => 3,
    ]
}

fn pooled_chroma_spec() -> CollabSpec {
    let mut spec = CollabSpec::new(64, 48, ChannelMode::Chroma, 1);
    spec.pooled = true;

    spec
}

variant_chunk_tests! {
    assert_variants_match_baseline(pooled_chroma_spec()), 10 chunks,
    [
        pooled_chroma_matches_the_baseline_in_the_first_variant_tenth => 0,
        pooled_chroma_matches_the_baseline_in_the_second_variant_tenth => 1,
        pooled_chroma_matches_the_baseline_in_the_third_variant_tenth => 2,
        pooled_chroma_matches_the_baseline_in_the_fourth_variant_tenth => 3,
        pooled_chroma_matches_the_baseline_in_the_fifth_variant_tenth => 4,
        pooled_chroma_matches_the_baseline_in_the_sixth_variant_tenth => 5,
        pooled_chroma_matches_the_baseline_in_the_seventh_variant_tenth => 6,
        pooled_chroma_matches_the_baseline_in_the_eighth_variant_tenth => 7,
        pooled_chroma_matches_the_baseline_in_the_ninth_variant_tenth => 8,
        pooled_chroma_matches_the_baseline_in_the_tenth_variant_tenth => 9,
    ]
}

/// 133 by 33 clamps the last reference onto the right and bottom frame edges, and its 33
/// references per row leave dead groups in the last cube of each row at every group count.
fn frame_edge_yuv_spec() -> CollabSpec {
    CollabSpec::new(133, 33, ChannelMode::Yuv, 2)
}

variant_chunk_tests! {
    assert_variants_match_baseline(frame_edge_yuv_spec()), 20 chunks,
    [
        yuv_at_the_frame_edges_matches_the_baseline_in_the_first_variant_twentieth => 0,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_second_variant_twentieth => 1,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_third_variant_twentieth => 2,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_fourth_variant_twentieth => 3,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_fifth_variant_twentieth => 4,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_sixth_variant_twentieth => 5,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_seventh_variant_twentieth => 6,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_eighth_variant_twentieth => 7,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_ninth_variant_twentieth => 8,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_tenth_variant_twentieth => 9,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_eleventh_variant_twentieth => 10,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_twelfth_variant_twentieth => 11,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_thirteenth_variant_twentieth => 12,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_fourteenth_variant_twentieth => 13,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_fifteenth_variant_twentieth => 14,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_sixteenth_variant_twentieth => 15,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_seventeenth_variant_twentieth => 16,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_eighteenth_variant_twentieth => 17,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_nineteenth_variant_twentieth => 18,
        yuv_at_the_frame_edges_matches_the_baseline_in_the_twentieth_variant_twentieth => 19,
    ]
}

/// 104 pixels gives 25 references per row, so the last cube of a row has dead groups at every
/// group count.
fn dead_groups_spec() -> CollabSpec {
    CollabSpec::new(104, 40, ChannelMode::Luma, 0)
}

variant_chunk_tests! {
    assert_variants_match_baseline(dead_groups_spec()), 4 chunks,
    [
        dead_groups_at_the_end_of_a_row_match_the_baseline_in_the_first_variant_quarter => 0,
        dead_groups_at_the_end_of_a_row_match_the_baseline_in_the_second_variant_quarter => 1,
        dead_groups_at_the_end_of_a_row_match_the_baseline_in_the_third_variant_quarter => 2,
        dead_groups_at_the_end_of_a_row_match_the_baseline_in_the_fourth_variant_quarter => 3,
    ]
}

fn smaller_than_the_search_window_spec() -> CollabSpec {
    CollabSpec::new(40, 24, ChannelMode::Yuv, 1)
}

variant_chunk_tests! {
    assert_variants_match_baseline(smaller_than_the_search_window_spec()), 20 chunks,
    [
        a_frame_inside_the_search_window_matches_the_baseline_in_the_first_variant_twentieth => 0,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_second_variant_twentieth => 1,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_third_variant_twentieth => 2,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_fourth_variant_twentieth => 3,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_fifth_variant_twentieth => 4,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_sixth_variant_twentieth => 5,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_seventh_variant_twentieth => 6,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_eighth_variant_twentieth => 7,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_ninth_variant_twentieth => 8,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_tenth_variant_twentieth => 9,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_eleventh_variant_twentieth => 10,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_twelfth_variant_twentieth => 11,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_thirteenth_variant_twentieth => 12,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_fourteenth_variant_twentieth => 13,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_fifteenth_variant_twentieth => 14,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_sixteenth_variant_twentieth => 15,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_seventeenth_variant_twentieth => 16,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_eighteenth_variant_twentieth => 17,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_nineteenth_variant_twentieth => 18,
        a_frame_inside_the_search_window_matches_the_baseline_in_the_twentieth_variant_twentieth => 19,
    ]
}

fn flat_spec(f16_search: bool) -> CollabSpec {
    let mut spec = CollabSpec::new(64, 48, ChannelMode::Luma, 1);
    spec.flat = true;
    spec.f16_search = f16_search;

    spec
}

variant_chunk_tests! {
    assert_variants_match_baseline(flat_spec(false)), 4 chunks,
    [
        flat_content_matches_the_baseline_in_the_first_variant_quarter => 0,
        flat_content_matches_the_baseline_in_the_second_variant_quarter => 1,
        flat_content_matches_the_baseline_in_the_third_variant_quarter => 2,
        flat_content_matches_the_baseline_in_the_fourth_variant_quarter => 3,
    ]
}

variant_chunk_tests! {
    assert_variants_match_baseline(flat_spec(true)), 4 chunks,
    [
        flat_content_matches_the_baseline_with_the_f16_search_in_the_first_variant_quarter => 0,
        flat_content_matches_the_baseline_with_the_f16_search_in_the_second_variant_quarter => 1,
        flat_content_matches_the_baseline_with_the_f16_search_in_the_third_variant_quarter => 2,
        flat_content_matches_the_baseline_with_the_f16_search_in_the_fourth_variant_quarter => 3,
    ]
}

fn f16_pooled_luma_spec() -> CollabSpec {
    let mut spec = CollabSpec::new(70, 45, ChannelMode::Luma, 2);
    spec.f16_search = true;
    spec.pooled = true;

    spec
}

variant_chunk_tests! {
    assert_variants_match_baseline(f16_pooled_luma_spec()), 4 chunks,
    [
        the_f16_search_on_pooled_luma_matches_the_baseline_in_the_first_variant_quarter => 0,
        the_f16_search_on_pooled_luma_matches_the_baseline_in_the_second_variant_quarter => 1,
        the_f16_search_on_pooled_luma_matches_the_baseline_in_the_third_variant_quarter => 2,
        the_f16_search_on_pooled_luma_matches_the_baseline_in_the_fourth_variant_quarter => 3,
    ]
}

fn f16_yuv_spec() -> CollabSpec {
    let mut spec = CollabSpec::new(64, 48, ChannelMode::Yuv, 1);
    spec.f16_search = true;

    spec
}

variant_chunk_tests! {
    assert_variants_match_baseline(f16_yuv_spec()), 20 chunks,
    [
        the_f16_search_on_yuv_matches_the_baseline_in_the_first_variant_twentieth => 0,
        the_f16_search_on_yuv_matches_the_baseline_in_the_second_variant_twentieth => 1,
        the_f16_search_on_yuv_matches_the_baseline_in_the_third_variant_twentieth => 2,
        the_f16_search_on_yuv_matches_the_baseline_in_the_fourth_variant_twentieth => 3,
        the_f16_search_on_yuv_matches_the_baseline_in_the_fifth_variant_twentieth => 4,
        the_f16_search_on_yuv_matches_the_baseline_in_the_sixth_variant_twentieth => 5,
        the_f16_search_on_yuv_matches_the_baseline_in_the_seventh_variant_twentieth => 6,
        the_f16_search_on_yuv_matches_the_baseline_in_the_eighth_variant_twentieth => 7,
        the_f16_search_on_yuv_matches_the_baseline_in_the_ninth_variant_twentieth => 8,
        the_f16_search_on_yuv_matches_the_baseline_in_the_tenth_variant_twentieth => 9,
        the_f16_search_on_yuv_matches_the_baseline_in_the_eleventh_variant_twentieth => 10,
        the_f16_search_on_yuv_matches_the_baseline_in_the_twelfth_variant_twentieth => 11,
        the_f16_search_on_yuv_matches_the_baseline_in_the_thirteenth_variant_twentieth => 12,
        the_f16_search_on_yuv_matches_the_baseline_in_the_fourteenth_variant_twentieth => 13,
        the_f16_search_on_yuv_matches_the_baseline_in_the_fifteenth_variant_twentieth => 14,
        the_f16_search_on_yuv_matches_the_baseline_in_the_sixteenth_variant_twentieth => 15,
        the_f16_search_on_yuv_matches_the_baseline_in_the_seventeenth_variant_twentieth => 16,
        the_f16_search_on_yuv_matches_the_baseline_in_the_eighteenth_variant_twentieth => 17,
        the_f16_search_on_yuv_matches_the_baseline_in_the_nineteenth_variant_twentieth => 18,
        the_f16_search_on_yuv_matches_the_baseline_in_the_twentieth_variant_twentieth => 19,
    ]
}

fn f16_frame_edge_chroma_spec() -> CollabSpec {
    let mut spec = CollabSpec::new(133, 33, ChannelMode::Chroma, 2);
    spec.f16_search = true;

    spec
}

variant_chunk_tests! {
    assert_variants_match_baseline(f16_frame_edge_chroma_spec()), 10 chunks,
    [
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_first_variant_tenth => 0,
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_second_variant_tenth => 1,
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_third_variant_tenth => 2,
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_fourth_variant_tenth => 3,
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_fifth_variant_tenth => 4,
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_sixth_variant_tenth => 5,
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_seventh_variant_tenth => 6,
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_eighth_variant_tenth => 7,
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_ninth_variant_tenth => 8,
        the_f16_search_on_chroma_at_the_frame_edges_matches_the_baseline_in_the_tenth_variant_tenth => 9,
    ]
}

/// [assert_variants_match_baseline] with the f16 search and the pooled threshold each off and on.
fn assert_switches_match_baseline(spec: CollabSpec, variants: &[CollabVariant]) {
    for f16_search in [false, true] {
        for pooled in [false, true] {
            let switched = CollabSpec {
                f16_search,
                pooled,
                ..spec
            };
            assert_variants_match_baseline(switched, variants);
        }
    }
}

fn site_chroma_spec() -> CollabSpec {
    CollabSpec::new(64, 48, ChannelMode::Chroma, 2)
}

/// Yuv takes one switch setting per test, since each yuv variant compiles several times slower.
fn site_yuv_spec(f16_search: bool, pooled: bool) -> CollabSpec {
    let spec = CollabSpec::new(72, 40, ChannelMode::Yuv, 1);

    CollabSpec {
        f16_search,
        pooled,
        ..spec
    }
}

variant_chunk_tests! {
    assert_switches_match_baseline(luma_spec()), 20 chunks,
    [
        every_luma_switch_matches_the_baseline_through_the_site_in_the_first_variant_twentieth => 0,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_second_variant_twentieth => 1,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_third_variant_twentieth => 2,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_fourth_variant_twentieth => 3,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_fifth_variant_twentieth => 4,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_sixth_variant_twentieth => 5,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_seventh_variant_twentieth => 6,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_eighth_variant_twentieth => 7,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_ninth_variant_twentieth => 8,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_tenth_variant_twentieth => 9,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_eleventh_variant_twentieth => 10,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_twelfth_variant_twentieth => 11,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_thirteenth_variant_twentieth => 12,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_fourteenth_variant_twentieth => 13,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_fifteenth_variant_twentieth => 14,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_sixteenth_variant_twentieth => 15,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_seventeenth_variant_twentieth => 16,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_eighteenth_variant_twentieth => 17,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_nineteenth_variant_twentieth => 18,
        every_luma_switch_matches_the_baseline_through_the_site_in_the_twentieth_variant_twentieth => 19,
    ]
}

variant_chunk_tests! {
    assert_switches_match_baseline(site_chroma_spec()), 20 chunks,
    [
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_first_variant_twentieth => 0,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_second_variant_twentieth => 1,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_third_variant_twentieth => 2,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_fourth_variant_twentieth => 3,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_fifth_variant_twentieth => 4,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_sixth_variant_twentieth => 5,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_seventh_variant_twentieth => 6,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_eighth_variant_twentieth => 7,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_ninth_variant_twentieth => 8,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_tenth_variant_twentieth => 9,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_eleventh_variant_twentieth => 10,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_twelfth_variant_twentieth => 11,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_thirteenth_variant_twentieth => 12,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_fourteenth_variant_twentieth => 13,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_fifteenth_variant_twentieth => 14,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_sixteenth_variant_twentieth => 15,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_seventeenth_variant_twentieth => 16,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_eighteenth_variant_twentieth => 17,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_nineteenth_variant_twentieth => 18,
        every_chroma_switch_matches_the_baseline_through_the_site_in_the_twentieth_variant_twentieth => 19,
    ]
}

variant_chunk_tests! {
    assert_variants_match_baseline(site_yuv_spec(false, false)), 20 chunks,
    [
        yuv_matches_the_baseline_through_the_site_in_the_first_variant_twentieth => 0,
        yuv_matches_the_baseline_through_the_site_in_the_second_variant_twentieth => 1,
        yuv_matches_the_baseline_through_the_site_in_the_third_variant_twentieth => 2,
        yuv_matches_the_baseline_through_the_site_in_the_fourth_variant_twentieth => 3,
        yuv_matches_the_baseline_through_the_site_in_the_fifth_variant_twentieth => 4,
        yuv_matches_the_baseline_through_the_site_in_the_sixth_variant_twentieth => 5,
        yuv_matches_the_baseline_through_the_site_in_the_seventh_variant_twentieth => 6,
        yuv_matches_the_baseline_through_the_site_in_the_eighth_variant_twentieth => 7,
        yuv_matches_the_baseline_through_the_site_in_the_ninth_variant_twentieth => 8,
        yuv_matches_the_baseline_through_the_site_in_the_tenth_variant_twentieth => 9,
        yuv_matches_the_baseline_through_the_site_in_the_eleventh_variant_twentieth => 10,
        yuv_matches_the_baseline_through_the_site_in_the_twelfth_variant_twentieth => 11,
        yuv_matches_the_baseline_through_the_site_in_the_thirteenth_variant_twentieth => 12,
        yuv_matches_the_baseline_through_the_site_in_the_fourteenth_variant_twentieth => 13,
        yuv_matches_the_baseline_through_the_site_in_the_fifteenth_variant_twentieth => 14,
        yuv_matches_the_baseline_through_the_site_in_the_sixteenth_variant_twentieth => 15,
        yuv_matches_the_baseline_through_the_site_in_the_seventeenth_variant_twentieth => 16,
        yuv_matches_the_baseline_through_the_site_in_the_eighteenth_variant_twentieth => 17,
        yuv_matches_the_baseline_through_the_site_in_the_nineteenth_variant_twentieth => 18,
        yuv_matches_the_baseline_through_the_site_in_the_twentieth_variant_twentieth => 19,
    ]
}

variant_chunk_tests! {
    assert_variants_match_baseline(site_yuv_spec(false, true)), 20 chunks,
    [
        pooled_yuv_matches_the_baseline_through_the_site_in_the_first_variant_twentieth => 0,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_second_variant_twentieth => 1,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_third_variant_twentieth => 2,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_fourth_variant_twentieth => 3,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_fifth_variant_twentieth => 4,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_sixth_variant_twentieth => 5,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_seventh_variant_twentieth => 6,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_eighth_variant_twentieth => 7,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_ninth_variant_twentieth => 8,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_tenth_variant_twentieth => 9,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_eleventh_variant_twentieth => 10,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_twelfth_variant_twentieth => 11,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_thirteenth_variant_twentieth => 12,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_fourteenth_variant_twentieth => 13,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_fifteenth_variant_twentieth => 14,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_sixteenth_variant_twentieth => 15,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_seventeenth_variant_twentieth => 16,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_eighteenth_variant_twentieth => 17,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_nineteenth_variant_twentieth => 18,
        pooled_yuv_matches_the_baseline_through_the_site_in_the_twentieth_variant_twentieth => 19,
    ]
}

variant_chunk_tests! {
    assert_variants_match_baseline(site_yuv_spec(true, false)), 20 chunks,
    [
        f16_yuv_matches_the_baseline_through_the_site_in_the_first_variant_twentieth => 0,
        f16_yuv_matches_the_baseline_through_the_site_in_the_second_variant_twentieth => 1,
        f16_yuv_matches_the_baseline_through_the_site_in_the_third_variant_twentieth => 2,
        f16_yuv_matches_the_baseline_through_the_site_in_the_fourth_variant_twentieth => 3,
        f16_yuv_matches_the_baseline_through_the_site_in_the_fifth_variant_twentieth => 4,
        f16_yuv_matches_the_baseline_through_the_site_in_the_sixth_variant_twentieth => 5,
        f16_yuv_matches_the_baseline_through_the_site_in_the_seventh_variant_twentieth => 6,
        f16_yuv_matches_the_baseline_through_the_site_in_the_eighth_variant_twentieth => 7,
        f16_yuv_matches_the_baseline_through_the_site_in_the_ninth_variant_twentieth => 8,
        f16_yuv_matches_the_baseline_through_the_site_in_the_tenth_variant_twentieth => 9,
        f16_yuv_matches_the_baseline_through_the_site_in_the_eleventh_variant_twentieth => 10,
        f16_yuv_matches_the_baseline_through_the_site_in_the_twelfth_variant_twentieth => 11,
        f16_yuv_matches_the_baseline_through_the_site_in_the_thirteenth_variant_twentieth => 12,
        f16_yuv_matches_the_baseline_through_the_site_in_the_fourteenth_variant_twentieth => 13,
        f16_yuv_matches_the_baseline_through_the_site_in_the_fifteenth_variant_twentieth => 14,
        f16_yuv_matches_the_baseline_through_the_site_in_the_sixteenth_variant_twentieth => 15,
        f16_yuv_matches_the_baseline_through_the_site_in_the_seventeenth_variant_twentieth => 16,
        f16_yuv_matches_the_baseline_through_the_site_in_the_eighteenth_variant_twentieth => 17,
        f16_yuv_matches_the_baseline_through_the_site_in_the_nineteenth_variant_twentieth => 18,
        f16_yuv_matches_the_baseline_through_the_site_in_the_twentieth_variant_twentieth => 19,
    ]
}

variant_chunk_tests! {
    assert_variants_match_baseline(site_yuv_spec(true, true)), 20 chunks,
    [
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_first_variant_twentieth => 0,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_second_variant_twentieth => 1,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_third_variant_twentieth => 2,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_fourth_variant_twentieth => 3,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_fifth_variant_twentieth => 4,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_sixth_variant_twentieth => 5,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_seventh_variant_twentieth => 6,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_eighth_variant_twentieth => 7,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_ninth_variant_twentieth => 8,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_tenth_variant_twentieth => 9,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_eleventh_variant_twentieth => 10,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_twelfth_variant_twentieth => 11,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_thirteenth_variant_twentieth => 12,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_fourteenth_variant_twentieth => 13,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_fifteenth_variant_twentieth => 14,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_sixteenth_variant_twentieth => 15,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_seventeenth_variant_twentieth => 16,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_eighteenth_variant_twentieth => 17,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_nineteenth_variant_twentieth => 18,
        pooled_f16_yuv_matches_the_baseline_through_the_site_in_the_twentieth_variant_twentieth => 19,
    ]
}

/// Runs nl4d over `channels` with each of `variants` forced and checks it against the baseline.
fn assert_forced_variants_leave_nl4d_output_unchanged(channels: ChannelMode, variants: &[CollabVariant]) {
    let options = Nl4dOptions {
        temporal_radius: 1,
        ..Nl4dOptions::default()
    };
    let planes = channels.count() as usize;
    let (width, height) = (72u32, 48u32);
    let frames = noisy_frames(width, height, planes, 7);
    let baseline_plan = KernelPlan::baseline();
    let baseline = nl4d_outputs(options, channels, width, height, &frames, baseline_plan);

    for &variant in variants {
        let name = variant.name();
        let plan = KernelPlan::baseline().with_site(COLLAB_FUSED, name);
        let forced = nl4d_outputs(options, channels, width, height, &frames, plan);
        assert_eq!(forced, baseline, "{channels:?} {name}");
    }
}

variant_chunk_tests! {
    assert_forced_variants_leave_nl4d_output_unchanged(ChannelMode::Luma), 10 chunks,
    [
        forcing_the_first_variant_tenth_leaves_luma_nl4d_output_unchanged => 0,
        forcing_the_second_variant_tenth_leaves_luma_nl4d_output_unchanged => 1,
        forcing_the_third_variant_tenth_leaves_luma_nl4d_output_unchanged => 2,
        forcing_the_fourth_variant_tenth_leaves_luma_nl4d_output_unchanged => 3,
        forcing_the_fifth_variant_tenth_leaves_luma_nl4d_output_unchanged => 4,
        forcing_the_sixth_variant_tenth_leaves_luma_nl4d_output_unchanged => 5,
        forcing_the_seventh_variant_tenth_leaves_luma_nl4d_output_unchanged => 6,
        forcing_the_eighth_variant_tenth_leaves_luma_nl4d_output_unchanged => 7,
        forcing_the_ninth_variant_tenth_leaves_luma_nl4d_output_unchanged => 8,
        forcing_the_tenth_variant_tenth_leaves_luma_nl4d_output_unchanged => 9,
    ]
}

variant_chunk_tests! {
    assert_forced_variants_leave_nl4d_output_unchanged(ChannelMode::Chroma), 10 chunks,
    [
        forcing_the_first_variant_tenth_leaves_chroma_nl4d_output_unchanged => 0,
        forcing_the_second_variant_tenth_leaves_chroma_nl4d_output_unchanged => 1,
        forcing_the_third_variant_tenth_leaves_chroma_nl4d_output_unchanged => 2,
        forcing_the_fourth_variant_tenth_leaves_chroma_nl4d_output_unchanged => 3,
        forcing_the_fifth_variant_tenth_leaves_chroma_nl4d_output_unchanged => 4,
        forcing_the_sixth_variant_tenth_leaves_chroma_nl4d_output_unchanged => 5,
        forcing_the_seventh_variant_tenth_leaves_chroma_nl4d_output_unchanged => 6,
        forcing_the_eighth_variant_tenth_leaves_chroma_nl4d_output_unchanged => 7,
        forcing_the_ninth_variant_tenth_leaves_chroma_nl4d_output_unchanged => 8,
        forcing_the_tenth_variant_tenth_leaves_chroma_nl4d_output_unchanged => 9,
    ]
}

variant_chunk_tests! {
    assert_forced_variants_leave_nl4d_output_unchanged(ChannelMode::Yuv), 20 chunks,
    [
        forcing_the_first_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 0,
        forcing_the_second_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 1,
        forcing_the_third_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 2,
        forcing_the_fourth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 3,
        forcing_the_fifth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 4,
        forcing_the_sixth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 5,
        forcing_the_seventh_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 6,
        forcing_the_eighth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 7,
        forcing_the_ninth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 8,
        forcing_the_tenth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 9,
        forcing_the_eleventh_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 10,
        forcing_the_twelfth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 11,
        forcing_the_thirteenth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 12,
        forcing_the_fourteenth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 13,
        forcing_the_fifteenth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 14,
        forcing_the_sixteenth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 15,
        forcing_the_seventeenth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 16,
        forcing_the_eighteenth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 17,
        forcing_the_nineteenth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 18,
        forcing_the_twentieth_variant_twentieth_leaves_yuv_nl4d_output_unchanged => 19,
    ]
}

/// Launches the chroma collab baseline once, so a compilation log holds its generated source.
#[test]
#[ignore = "used to dump the compiled chroma kernel for register counts"]
fn dump_chroma_collab_kernel() {
    let client = test_client();
    let mut spec = CollabSpec::new(64, 48, ChannelMode::Chroma, 2);
    spec.pooled = true;
    let site = collab_site(&client, &spec);
    let outputs = site.scratch_outputs();
    let baseline = site.baseline();

    site.launch(baseline, &outputs);

    let weights = read_f32(&client, &outputs.group_weight, outputs.refs).expect("weights read back");
    assert!(weights.iter().any(|&weight| weight > 0.0));
}

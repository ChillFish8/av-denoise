mod chunks;

use std::collections::HashSet;

use super::engine::{nl4d_outputs, noisy_frames};
use super::fixtures::{CollabSpec, collab_site, collab_site_f16};
use super::probe::small_device_limits;
use super::runtime::{test_client, test_legality};
use crate::Nl4dOptions;
use crate::collab::geometry::refs_along;
use crate::collab::{PATCH_SIZE, STEP, needs_warp_uniform_search};
use crate::nlmeans::ChannelMode;
use crate::tune::gpu::{read_f32, sync};
use crate::tune::session::timed_count;
use crate::tune::sites::{
    COLLAB_FUSED,
    Centre,
    ChannelView,
    CollabVariant,
    NeighbourWalk,
    Walk,
    tile_height,
    tile_width,
};
use crate::tune::{DeviceLimits, KernelPlan, SiteCandidate, VariantSite, fit_candidates, legal_candidates};

pub(super) fn luma_spec() -> CollabSpec {
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
    assert_eq!(CollabVariant::ALL.len(), 120);

    for groups in [8u32, 4, 16] {
        for walk in [Walk::Clipped, Walk::Uniform] {
            for centre in [Centre::Global, Centre::Tiled, Centre::Sliding] {
                for batched in [false, true] {
                    for neighbour in [NeighbourWalk::Rows, NeighbourWalk::Sliding] {
                        for channel_view in [ChannelView::Lines, ChannelView::Scalar] {
                            let variant = CollabVariant {
                                groups,
                                walk,
                                centre,
                                batched,
                                neighbour,
                                channel_view,
                            };
                            let listed = CollabVariant::ALL.contains(&variant);
                            let builds = !(centre == Centre::Sliding && batched);
                            assert_eq!(listed, builds, "{variant:?}");
                        }
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
        channel_view: ChannelView::Scalar,
    };
    assert_eq!(everything.name(), "groups16_uniform_tiled_batched_nslide_scalar");

    let scalar = CollabVariant {
        channel_view: ChannelView::Scalar,
        ..CollabVariant::BASELINE
    };
    assert_eq!(scalar.name(), "groups8_clipped_scalar");

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

    assert_eq!(uniform_only_count, 60);
}

#[test]
fn only_multi_channel_planes_build_the_scalar_view() {
    for &variant in CollabVariant::ALL {
        let name = variant.name();
        let scalar = variant.channel_view == ChannelView::Scalar;

        assert_eq!(variant.distinct_for(1), !scalar, "{name}");
        assert!(variant.distinct_for(2), "{name}");
        assert!(variant.distinct_for(3), "{name}");
    }
}

/// Asserts the staged site over `spec` times `expected_first` fitted variants at 8 groups, or half
/// as many where only the uniform walk runs, then 2 more group counts for each of the 3 fastest.
fn assert_staged_counts(spec: CollabSpec, expected_first: usize) {
    let client = test_client();
    let site = collab_site(&client, &spec);
    let limits = DeviceLimits::from_client(&client);
    let uniform_only = needs_warp_uniform_search(&client);

    let fitted = fit_candidates(&site, &limits).expect("the baseline fits");
    let at_eight_groups = fitted.iter().filter(|variant| variant.groups == 8).count();
    let scalar_at_eight_groups = fitted
        .iter()
        .filter(|variant| variant.groups == 8 && variant.channel_view == ChannelView::Scalar)
        .count();
    let timed = timed_count(&site, &fitted);
    let expected = if uniform_only {
        expected_first / 2
    } else {
        expected_first
    };
    let expected_scalar = if spec.channels == ChannelMode::Luma {
        0
    } else {
        expected / 2
    };

    assert_eq!(at_eight_groups, expected);
    assert_eq!(scalar_at_eight_groups, expected_scalar);
    assert_eq!(timed, at_eight_groups + 3 * 2);

    // The HIP runtime crashes at process exit while an upload is still queued.
    sync(&client).expect("the uploads finish");
}

/// Luma has no scalar view, so it times 20 and then 6, 26 in all.
#[test]
fn the_staged_collab_site_times_the_first_stage_and_three_regroupings() {
    assert_staged_counts(luma_spec(), 20);
}

/// Chroma and yuv time each search variant with both channel views, 40 and then 6.
#[test]
fn the_staged_chroma_and_yuv_sites_time_both_channel_views() {
    let chroma = CollabSpec::new(64, 48, ChannelMode::Chroma, 2);
    let yuv = CollabSpec::new(64, 48, ChannelMode::Yuv, 1);

    assert_staged_counts(chroma, 40);
    assert_staged_counts(yuv, 40);
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
        .filter(|variant| variant.runs_where(uniform_only) && variant.distinct_for(1))
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

/// Luma has one channel, so the scalar view is never a legal candidate there.
///
/// Both views build the same kernel on one channel, so the output check only shows the forced plan still runs.
#[test]
fn forcing_the_scalar_view_on_luma_runs_the_baseline() {
    let client = test_client();
    let spec = luma_spec();
    let site = collab_site(&client, &spec);
    let limits = DeviceLimits::from_client(&client);
    let legality = test_legality(&client);
    let scalar = CollabVariant::from_name("groups8_clipped_scalar").expect("a listed variant");

    let legal = legal_candidates(&site, &limits, &legality).expect("the baseline fits");
    assert!(!legal.contains(&scalar));

    let options = Nl4dOptions {
        temporal_radius: 1,
        ..Nl4dOptions::default()
    };
    let (width, height) = (72u32, 48u32);
    let frames = noisy_frames(width, height, 1, 7);
    let baseline_plan = KernelPlan::baseline();
    let forced_plan = KernelPlan::baseline().with_site(COLLAB_FUSED, scalar.name());

    let baseline = nl4d_outputs(options, ChannelMode::Luma, width, height, &frames, baseline_plan);
    let forced = nl4d_outputs(options, ChannelMode::Luma, width, height, &frames, forced_plan);

    assert_eq!(forced, baseline);
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

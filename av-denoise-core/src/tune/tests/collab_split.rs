use cubecl::prelude::*;

use super::equivalence::{assert_candidates_match_baseline, assert_listed_candidates_match_baseline};
use super::fixtures::{CollabSpec, collab_site, collab_site_f16, split_sites};
use super::runtime::{TestRuntime, test_client};
use crate::collab::kernels::fused::{GROUP_SIZE_MASK, MEMBER_WORDS};
use crate::collab::supports_f16_search;
use crate::nlmeans::ChannelMode;
use crate::tune::gpu::read_i32;
use crate::tune::sites::{
    COLLAB_FILTER,
    COLLAB_SEARCH,
    CollabSite,
    FLOAT_SEARCH_COUNT,
    FilterCandidate,
    SearchVariant,
    candidate_known,
    compare_collab_outputs,
};
use crate::tune::{Fidelity, SiteCandidate, VariantSite};

fn assert_split_matches_fused<S: Float>(
    client: &ComputeClient<TestRuntime>,
    fused: &CollabSite<TestRuntime, S>,
    label: &str,
) {
    let (search, filter) = split_sites(fused);

    let want = fused.scratch_outputs();
    fused.launch(fused.baseline(), &want);

    let members = search.real_outputs();
    search.launch(search.baseline(), &members);

    let got = filter.scratch_outputs();
    filter.launch(filter.baseline(), &got);

    let verdict = compare_collab_outputs(client, &want, &got, Fidelity::Exact).expect("outputs read back");
    assert!(verdict.passed(), "{label}: {verdict:?}");
}

#[test]
fn the_split_matches_the_fused_kernel_at_every_grid_k_max_and_channel_mode() {
    let client = test_client();
    let channel_modes = [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv];

    for channels in channel_modes {
        for radius in [0u32, 1, 2] {
            for k_max in [2u32, 4, 8] {
                for pooled in [false, true] {
                    let mut spec = CollabSpec::new(72, 40, channels, radius);
                    spec.pooled = pooled;
                    let mut fused = collab_site(&client, &spec);
                    fused.shape.k_max = k_max;
                    let label = format!("{channels:?} radius={radius} k_max={k_max} pooled={pooled}");

                    assert_split_matches_fused(&client, &fused, &label);
                }
            }
        }
    }
}

#[test]
fn the_split_matches_the_fused_kernel_with_an_f16_search() {
    let client = test_client();
    if !supports_f16_search(&client) {
        return;
    }

    let channel_modes = [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv];

    for channels in channel_modes {
        for radius in [1u32, 2] {
            for pooled in [false, true] {
                let mut spec = CollabSpec::new(72, 40, channels, radius);
                spec.pooled = pooled;
                spec.f16_search = true;
                let mut fused = collab_site_f16(&client, &spec);
                fused.shape.k_max = 8;
                let label = format!("f16 {channels:?} radius={radius} k_max=8 pooled={pooled}");

                assert_split_matches_fused(&client, &fused, &label);
            }
        }
    }
}

fn assert_search_variants_match_baseline(variants: &[SearchVariant]) {
    let client = test_client();
    let spec = CollabSpec::new(64, 48, ChannelMode::Luma, 2);
    let fused = collab_site(&client, &spec);
    let (search, _) = split_sites(&fused);

    let members = search.real_outputs();
    search.launch(search.baseline(), &members);

    assert_listed_candidates_match_baseline(&search, "search", variants);
}

#[test]
fn the_first_half_of_the_search_variants_write_the_baselines_members() {
    assert_search_variants_match_baseline(&SearchVariant::ALL[..10]);
}

#[test]
fn the_second_half_of_the_search_variants_write_the_baselines_members() {
    assert_search_variants_match_baseline(&SearchVariant::ALL[10..FLOAT_SEARCH_COUNT]);
}

#[test]
fn every_filter_candidate_matches_the_baseline() {
    let client = test_client();
    let spec = CollabSpec::new(64, 48, ChannelMode::Luma, 2);
    let fused = collab_site(&client, &spec);
    let (search, filter) = split_sites(&fused);

    let members = search.real_outputs();
    search.launch(search.baseline(), &members);

    assert_candidates_match_baseline(&filter, "filter");
}

#[test]
fn split_site_names_round_trip_and_are_known() {
    assert_eq!(SearchVariant::ALL.len(), 25);

    for &variant in SearchVariant::ALL {
        let name = variant.name();
        assert_eq!(SearchVariant::from_name(name), Some(variant));
        assert!(candidate_known(COLLAB_SEARCH, name), "{name}");

        if let SearchVariant::Float(float_variant) = variant {
            assert_eq!(float_variant.groups, 8);
        }
    }

    let float_variants = &SearchVariant::ALL[..FLOAT_SEARCH_COUNT];
    let integer_variants = &SearchVariant::ALL[FLOAT_SEARCH_COUNT..];
    let floats_lead = float_variants
        .iter()
        .all(|variant| matches!(variant, SearchVariant::Float(_)));
    let integers_follow = integer_variants
        .iter()
        .all(|variant| matches!(variant, SearchVariant::Integer(_)));
    assert!(floats_lead && integers_follow);

    for &candidate in FilterCandidate::ALL {
        let name = candidate.name();
        assert_eq!(FilterCandidate::from_name(name), Some(candidate));
        assert!(candidate_known(COLLAB_FILTER, name), "{name}");
    }
}

#[test]
fn every_reference_gets_a_group_past_the_dead_groups() {
    let client = test_client();
    let spec = CollabSpec::new(104, 32, ChannelMode::Luma, 1);
    let fused = collab_site(&client, &spec);
    let (search, _) = split_sites(&fused);
    let outputs = search.real_outputs();

    search.launch(search.baseline(), &outputs);
    let words = read_i32(&client, &outputs.members, outputs.words()).expect("members read back");

    let group_words = words.chunks(MEMBER_WORDS as usize);
    let filled = group_words
        .clone()
        .all(|group| group[MEMBER_WORDS as usize - 1] as u32 & GROUP_SIZE_MASK != 0);
    assert_eq!(group_words.len(), outputs.refs);
    assert!(filled, "every live reference holds a group");
}

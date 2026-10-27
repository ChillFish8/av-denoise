use std::collections::HashSet;

use crate::tune::sites::{
    AccumulateCandidate,
    Centre,
    CollabVariant,
    FilterCandidate,
    RegulariseCandidate,
    SearchVariant,
    Walk,
    WindowCandidate,
};
use crate::tune::{
    BASELINE_FAMILY,
    DimensionFamilies,
    Families,
    FamilyId,
    FlatFamily,
    SiteCandidate,
    opened_members,
    representatives,
};

/// Every family id a site declares, dimension values included.
fn family_ids<C: SiteCandidate>() -> Vec<FamilyId> {
    match C::families() {
        Families::Flat(flat) => flat.iter().map(|family| family.id).collect(),
        Families::Dimensions(dimensions) => dimensions
            .dimensions
            .iter()
            .flat_map(|dimension| dimension.values.iter().map(|value| value.id))
            .collect(),
    }
}

fn flat_families<C: SiteCandidate>() -> &'static [FlatFamily<C>] {
    let Families::Flat(flat) = C::families() else {
        panic!("{} declares flat families", C::SITE);
    };

    flat
}

fn dimension_families<C: SiteCandidate>() -> DimensionFamilies<C> {
    let Families::Dimensions(dimensions) = C::families() else {
        panic!("{} declares dimension families", C::SITE);
    };

    dimensions
}

fn assert_reachable<C: SiteCandidate>() {
    let families = C::families();
    let ids = family_ids::<C>();
    let baseline = C::baseline();
    let members = opened_members(&families, baseline, &ids);

    assert_eq!(members.len(), C::ALL.len(), "{} reaches {members:?}", C::SITE);

    for candidate in C::ALL {
        let reached = members.contains(candidate);
        assert!(reached, "{} never reaches {candidate:?}", C::SITE);
    }
}

fn assert_flat_families_hold_their_members<C: SiteCandidate>() {
    let flat = flat_families::<C>();
    let mut seen = Vec::new();

    for family in flat {
        assert!(
            family.members.contains(&family.representative),
            "{} family {:?} holds its representative",
            C::SITE,
            family.id
        );

        for &member in family.members {
            assert!(
                !seen.contains(&member),
                "{} lists {member:?} in two families",
                C::SITE
            );
            seen.push(member);
        }
    }
}

fn assert_representatives_vary_one_value<C: SiteCandidate>(baseline: C) {
    let dimensions = dimension_families::<C>();
    let families = Families::Dimensions(dimensions);
    let representatives = representatives(&families, baseline);
    let base = values_of(&dimensions, baseline);

    assert_eq!(representatives[0], (BASELINE_FAMILY, baseline), "{}", C::SITE);

    for &(id, representative) in &representatives[1..] {
        let values = values_of(&dimensions, representative);
        let changed: Vec<usize> = (0..values.len())
            .filter(|&dimension| values[dimension] != base[dimension])
            .collect();
        assert_eq!(changed.len(), 1, "{} representative {representative:?}", C::SITE);

        let dimension = changed[0];
        let value = dimensions.dimensions[dimension].values[values[dimension]];
        assert_eq!(value.id, id, "{} representative {representative:?}", C::SITE);
    }

    let other_values: usize = dimensions
        .dimensions
        .iter()
        .map(|dimension| dimension.values.len() - 1)
        .sum();
    assert_eq!(representatives.len(), 1 + other_values, "{}", C::SITE);
}

fn values_of<C: SiteCandidate>(dimensions: &DimensionFamilies<C>, candidate: C) -> Vec<usize> {
    (0..dimensions.dimensions.len())
        .map(|dimension| (dimensions.value_of)(candidate, dimension))
        .collect()
}

fn assert_baseline_is_a_representative<C: SiteCandidate>() {
    let families = C::families();
    let baseline = C::baseline();
    let representatives = representatives(&families, baseline);

    let found = representatives
        .iter()
        .any(|&(_, representative)| representative == baseline);
    assert!(
        found,
        "{} has no representative for its baseline {baseline:?}",
        C::SITE
    );
}

fn assert_only_baselines_open_the_baseline<C: SiteCandidate>(baseline: C) {
    let dimensions = dimension_families::<C>();
    let base = values_of(&dimensions, baseline);
    let mut baseline_ids: Vec<FamilyId> = dimensions
        .dimensions
        .iter()
        .zip(&base)
        .map(|(dimension, &index)| dimension.values[index].id)
        .collect();
    baseline_ids.push(BASELINE_FAMILY);

    let families = Families::Dimensions(dimensions);
    let with_baselines = opened_members(&families, baseline, &baseline_ids);
    let with_nothing = opened_members(&families, baseline, &[]);

    assert_eq!(with_baselines, vec![baseline], "{}", C::SITE);
    assert_eq!(with_nothing, vec![baseline], "{}", C::SITE);
}

/// The family id each collab dimension gives `variant`, built from the option it takes.
fn expected_id(variant: CollabVariant, dimension: &str) -> String {
    match dimension {
        "centre" => format!("centre_{:?}", variant.centre).to_lowercase(),
        "batched" if variant.batched => "batched_on".to_string(),
        "batched" => "batched_off".to_string(),
        "groups" => format!("groups{}", variant.groups),
        "walk" => format!("walk_{}", variant.walk.name()),
        "neighbour" => format!("neighbour_{:?}", variant.neighbour).to_lowercase(),
        _ => panic!("no collab dimension is named {dimension}"),
    }
}

fn assert_ids_name_their_options<C: SiteCandidate>(variant_of: fn(C) -> CollabVariant) {
    let dimensions = dimension_families::<C>();

    for &candidate in C::ALL {
        let variant = variant_of(candidate);

        for (index, dimension) in dimensions.dimensions.iter().enumerate() {
            let value_index = (dimensions.value_of)(candidate, index);
            let id = dimension.values[value_index].id;
            let expected = expected_id(variant, dimension.name);
            assert_eq!(id.0, expected, "{} {variant:?}", C::SITE);
        }
    }
}

fn assert_ids_unique<C: SiteCandidate>() {
    let ids = family_ids::<C>();
    let unique: HashSet<FamilyId> = ids.iter().copied().collect();
    assert_eq!(unique.len(), ids.len(), "{} repeats a family id", C::SITE);
    assert!(!unique.contains(&BASELINE_FAMILY), "{}", C::SITE);
}

fn assert_value_of_inverts_compose<C: SiteCandidate>() {
    let dimensions = dimension_families::<C>();

    for &candidate in C::ALL {
        let values = values_of(&dimensions, candidate);
        let composed = (dimensions.compose)(&values);
        assert_eq!(composed, Some(candidate), "{}", C::SITE);
    }
}

#[test]
fn every_candidate_is_reachable_from_its_families() {
    assert_reachable::<CollabVariant>();
    assert_reachable::<SearchVariant>();
    assert_reachable::<FilterCandidate>();
    assert_reachable::<WindowCandidate>();
    assert_reachable::<AccumulateCandidate>();
    assert_reachable::<RegulariseCandidate>();
}

#[test]
fn family_ids_are_unique_within_each_site() {
    assert_ids_unique::<CollabVariant>();
    assert_ids_unique::<SearchVariant>();
    assert_ids_unique::<FilterCandidate>();
    assert_ids_unique::<WindowCandidate>();
    assert_ids_unique::<AccumulateCandidate>();
    assert_ids_unique::<RegulariseCandidate>();
}

#[test]
fn flat_families_are_disjoint_and_hold_their_representative() {
    assert_flat_families_hold_their_members::<FilterCandidate>();
    assert_flat_families_hold_their_members::<WindowCandidate>();
    assert_flat_families_hold_their_members::<AccumulateCandidate>();
    assert_flat_families_hold_their_members::<RegulariseCandidate>();
}

#[test]
fn dimension_representatives_vary_one_value_from_the_baseline() {
    assert_representatives_vary_one_value(CollabVariant::baseline());
    assert_representatives_vary_one_value(SearchVariant::baseline());
}

#[test]
fn value_of_inverts_compose_for_every_candidate() {
    assert_value_of_inverts_compose::<CollabVariant>();
    assert_value_of_inverts_compose::<SearchVariant>();
}

#[test]
fn the_baseline_is_a_representative_of_every_site() {
    assert_baseline_is_a_representative::<CollabVariant>();
    assert_baseline_is_a_representative::<SearchVariant>();
    assert_baseline_is_a_representative::<FilterCandidate>();
    assert_baseline_is_a_representative::<WindowCandidate>();
    assert_baseline_is_a_representative::<AccumulateCandidate>();
    assert_baseline_is_a_representative::<RegulariseCandidate>();
}

#[test]
fn opened_members_of_only_baseline_families_is_the_baseline() {
    assert_only_baselines_open_the_baseline(CollabVariant::baseline());
    assert_only_baselines_open_the_baseline(SearchVariant::baseline());
}

#[test]
fn an_unbuildable_combination_is_left_out_of_the_cross_product() {
    let families = CollabVariant::families();
    let opened = [FamilyId("centre_sliding"), FamilyId("batched_on")];

    let baseline = CollabVariant::baseline();
    let members = opened_members(&families, baseline, &opened);

    let sliding_batched = members
        .iter()
        .any(|variant| variant.centre == Centre::Sliding && variant.batched);
    let sliding = members.iter().any(|variant| variant.centre == Centre::Sliding);
    let batched = members.iter().any(|variant| variant.batched);
    assert_eq!(members.len(), 3, "{members:?}");
    assert!(sliding && batched);
    assert!(!sliding_batched);
}

#[test]
fn dimension_family_ids_name_the_option_at_their_index() {
    assert_ids_name_their_options::<CollabVariant>(|variant| variant);
    assert_ids_name_their_options::<SearchVariant>(|search| search.0);
}

#[test]
fn a_uniform_walk_baseline_anchors_the_representatives_and_the_opened_walk() {
    let uniform = CollabVariant::baseline_for(true);
    let families = CollabVariant::families();

    assert_representatives_vary_one_value(uniform);
    assert_only_baselines_open_the_baseline(uniform);

    let representatives = representatives(&families, uniform);
    let clipped: Vec<FamilyId> = representatives
        .iter()
        .filter(|(_, representative)| representative.walk == Walk::Clipped)
        .map(|&(id, _)| id)
        .collect();
    assert_eq!(clipped, vec![FamilyId("walk_clipped")]);

    let opened = [FamilyId("groups4")];
    let members = opened_members(&families, uniform, &opened);
    let groups4_uniform = CollabVariant { groups: 4, ..uniform };
    assert_eq!(members, vec![uniform, groups4_uniform]);
}

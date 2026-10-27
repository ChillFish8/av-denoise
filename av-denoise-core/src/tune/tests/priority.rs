use crate::tune::sites::{
    AccumulateCandidate,
    COLLAB_FUSED,
    COLLAB_SEARCH,
    CollabVariant,
    FilterCandidate,
    NL4D_MV_REGULARISE,
    NLM_WINDOW,
    RegulariseCandidate,
    SearchVariant,
    WindowCandidate,
};
use crate::tune::{
    BASELINE_FAMILY,
    Basis,
    DeviceClass,
    Families,
    FamilyId,
    PipelineProfile,
    PriorityTable,
    SiteCandidate,
    SitePriority,
    family_order,
    priority_for,
    skipped,
};

const BACKENDS: [&str; 4] = ["vulkan", "rocm", "cuda", "metal"];

const CLASSES: [DeviceClass; 7] = [
    DeviceClass::Amd,
    DeviceClass::Apple,
    DeviceClass::Intel,
    DeviceClass::NvidiaPreVolta { fast_f16: true },
    DeviceClass::NvidiaPreVolta { fast_f16: false },
    DeviceClass::Nvidia,
    DeviceClass::Unknown,
];

/// A site's family ids in declaration order, and the ids no table may skip.
struct SiteFamilies {
    site: &'static str,
    ids: Vec<FamilyId>,
    baseline_ids: Vec<FamilyId>,
}

/// Every family id a site's representatives can carry, in declaration order.
fn declared_ids<C: SiteCandidate>() -> Vec<FamilyId> {
    match C::families() {
        Families::Flat(flat) => flat.iter().map(|family| family.id).collect(),
        Families::Dimensions(dimensions) => {
            let values = dimensions
                .dimensions
                .iter()
                .flat_map(|dimension| dimension.values.iter().map(|value| value.id));
            let mut ids = vec![BASELINE_FAMILY];
            ids.extend(values);
            ids
        },
    }
}

/// The families each of `baselines` belongs to, which always count as opened.
fn baseline_ids<C: SiteCandidate>(baselines: &[C]) -> Vec<FamilyId> {
    let mut ids = Vec::new();

    for &baseline in baselines {
        match C::families() {
            Families::Flat(flat) => {
                let holding = flat
                    .iter()
                    .filter(|family| family.members.contains(&baseline))
                    .map(|family| family.id);
                ids.extend(holding);
            },
            Families::Dimensions(dimensions) => {
                ids.push(BASELINE_FAMILY);

                for (index, dimension) in dimensions.dimensions.iter().enumerate() {
                    let value = (dimensions.value_of)(baseline, index);
                    ids.push(dimension.values[value].id);
                }
            },
        }
    }

    ids
}

fn site_families<C: SiteCandidate>(baselines: &[C]) -> SiteFamilies {
    SiteFamilies {
        site: C::SITE,
        ids: declared_ids::<C>(),
        baseline_ids: baseline_ids(baselines),
    }
}

fn every_site() -> Vec<SiteFamilies> {
    let clipped = CollabVariant::BASELINE;
    let uniform = CollabVariant::baseline_for(true);

    vec![
        site_families(&[clipped, uniform]),
        site_families(&[SearchVariant(clipped), SearchVariant(uniform)]),
        site_families(&[FilterCandidate::baseline()]),
        site_families(&[WindowCandidate::baseline()]),
        site_families(&[AccumulateCandidate::baseline()]),
        site_families(&[RegulariseCandidate::baseline()]),
    ]
}

/// Every distinct table [priority_for] hands out.
fn every_table() -> Vec<&'static PriorityTable> {
    let mut tables: Vec<&'static PriorityTable> = Vec::new();

    for class in CLASSES {
        for backend in BACKENDS {
            let table = priority_for(class, backend);
            let seen = tables.iter().any(|known| std::ptr::eq(*known, table));
            if !seen {
                tables.push(table);
            }
        }
    }

    tables
}

fn entry<'a>(table: &'a PriorityTable, site: &str) -> Option<&'a SitePriority> {
    table.sites.iter().find(|priority| priority.site == site)
}

fn assert_profiles_are_a_permutation(table: &PriorityTable) {
    let mut profiles = table.profiles.to_vec();
    profiles.sort();

    let mut every = PipelineProfile::ALL.to_vec();
    every.sort();

    assert_eq!(profiles, every, "{table:?} orders every profile once");
}

#[test]
fn every_table_lists_only_families_that_exist() {
    let sites = every_site();

    for table in every_table() {
        for priority in table.sites {
            let Some(declared) = sites.iter().find(|site| site.site == priority.site) else {
                panic!("{} is a tunable site", priority.site);
            };

            let entries = table
                .sites
                .iter()
                .filter(|other| other.site == priority.site)
                .count();
            assert_eq!(entries, 1, "{} has one entry per table", priority.site);

            for (index, family) in priority.order.iter().enumerate() {
                assert!(
                    declared.ids.contains(family),
                    "{} declares {family:?}",
                    priority.site
                );

                let repeated = priority.order[..index].contains(family);
                assert!(!repeated, "{} lists {family:?} once", priority.site);
            }

            for family in priority.skip {
                assert!(
                    declared.ids.contains(family),
                    "{} declares {family:?}",
                    priority.site
                );
            }
        }

        for site in &sites {
            let mut ordered = family_order(table, site.site, &site.ids);
            ordered.sort_by_key(|family| family.0);

            let mut ids = site.ids.clone();
            ids.sort_by_key(|family| family.0);

            assert_eq!(ordered, ids, "{} keeps every family once", site.site);
        }
    }
}

#[test]
fn no_table_skips_a_baseline_family() {
    let sites = every_site();

    for table in every_table() {
        for site in &sites {
            let skip = skipped(table, site.site);

            for family in &site.baseline_ids {
                assert!(
                    !skip.contains(family),
                    "{} skips its baseline family {family:?}",
                    site.site
                );
            }
        }
    }
}

#[test]
fn no_expected_entry_skips_a_family() {
    for table in every_table() {
        let expected = table
            .sites
            .iter()
            .filter(|priority| priority.basis == Basis::Expected);
        for priority in expected {
            assert!(
                priority.skip.is_empty(),
                "{} skips on an expected order",
                priority.site
            );
        }
    }
}

#[test]
fn unknown_devices_keep_declaration_order_and_skip_nothing() {
    let sites = every_site();

    for backend in BACKENDS {
        let table = priority_for(DeviceClass::Unknown, backend);
        assert_eq!(table.profiles, PipelineProfile::ALL.as_slice(), "{backend}");

        for site in &sites {
            let ordered = family_order(table, site.site, &site.ids);
            assert_eq!(ordered, site.ids, "{backend} {}", site.site);

            let skip = skipped(table, site.site);
            assert!(skip.is_empty(), "{backend} {} skips nothing", site.site);
        }
    }
}

#[test]
fn nvidia_cuda_and_vulkan_try_different_search_families_first() {
    let sites = every_site();
    let search = sites
        .iter()
        .find(|site| site.site == COLLAB_SEARCH)
        .expect("the search site exists");
    let cuda = priority_for(DeviceClass::Nvidia, "cuda");
    let vulkan = priority_for(DeviceClass::Nvidia, "vulkan");

    let cuda_order = family_order(cuda, COLLAB_SEARCH, &search.ids);
    let vulkan_order = family_order(vulkan, COLLAB_SEARCH, &search.ids);

    assert_eq!(cuda_order[1], FamilyId("neighbour_sliding"));
    assert_eq!(vulkan_order[1], FamilyId("centre_sliding"));
}

#[test]
fn family_order_puts_unlisted_families_last_in_declaration_order() {
    let table = PriorityTable {
        sites: &[SitePriority {
            site: NLM_WINDOW,
            basis: Basis::Expected,
            order: &[FamilyId("c"), FamilyId("missing"), FamilyId("a")],
            skip: &[],
        }],
        profiles_basis: Basis::Expected,
        profiles: &PipelineProfile::ALL,
    };
    let families = [FamilyId("a"), FamilyId("b"), FamilyId("c"), FamilyId("d")];

    let ordered = family_order(&table, NLM_WINDOW, &families);
    let expected = vec![FamilyId("c"), FamilyId("a"), FamilyId("b"), FamilyId("d")];
    assert_eq!(ordered, expected);

    let other_site = family_order(&table, NL4D_MV_REGULARISE, &families);
    assert_eq!(other_site, families.to_vec());
}

#[test]
fn every_device_class_and_backend_has_a_table() {
    let sites = every_site();

    for class in CLASSES {
        for backend in BACKENDS {
            let table = priority_for(class, backend);
            assert_profiles_are_a_permutation(table);

            if class == DeviceClass::Unknown {
                continue;
            }

            for site in &sites {
                let listed = entry(table, site.site);
                assert!(listed.is_some(), "{class:?} on {backend} orders {}", site.site);
            }
        }
    }
}

#[test]
fn dimension_tables_time_the_baseline_first() {
    for table in every_table() {
        for site in [COLLAB_FUSED, COLLAB_SEARCH] {
            let Some(priority) = entry(table, site) else {
                continue;
            };

            assert_eq!(priority.order.first(), Some(&BASELINE_FAMILY), "{site}");
        }
    }
}

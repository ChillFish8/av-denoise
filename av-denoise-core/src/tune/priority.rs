use super::family::{BASELINE_FAMILY, FamilyId};
use super::legality::DeviceClass;
use super::plan::PipelineProfile;
use super::sites::{
    COLLAB_FILTER,
    COLLAB_FUSED,
    COLLAB_SEARCH,
    NL4D_MV_REGULARISE,
    NLM_ACCUMULATE,
    NLM_WINDOW,
};

/// Where a table entry's order comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basis {
    /// Timings taken on a device of the class.
    Measured,
    /// What the class's architecture suggests, with no timings behind it.
    Expected,
}

/// The order one site's families are timed in, and the families left out by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SitePriority {
    pub site: &'static str,
    pub basis: Basis,
    /// The most likely winner first.
    pub order: &'static [FamilyId],
    pub skip: &'static [FamilyId],
}

/// The family and pipeline profile order for one device class on one backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PriorityTable {
    pub sites: &'static [SitePriority],
    /// Where the profile order comes from.
    pub profiles_basis: Basis,
    /// The fastest profile first.
    pub profiles: &'static [PipelineProfile],
}

/// Unknown devices.
const NEUTRAL: PriorityTable = PriorityTable {
    sites: &[],
    profiles_basis: Basis::Expected,
    profiles: &PipelineProfile::ALL,
};

/// AMD on Vulkan, from R9700 timings.
const AMD_VULKAN: PriorityTable = PriorityTable {
    sites: &[
        // groups4 -3..+6%, centre_tiled -7..+22%, groups16 +1..+6%, batched_on -1..+10%,
        // neighbour_sliding +10..+13%, centre_sliding +9..+18%, walk_uniform +12..+24%
        SitePriority {
            site: COLLAB_FUSED,
            basis: Basis::Measured,
            order: &[
                BASELINE_FAMILY,
                FamilyId("groups4"),
                FamilyId("centre_tiled"),
                FamilyId("groups16"),
                FamilyId("batched_on"),
                FamilyId("neighbour_sliding"),
                FamilyId("centre_sliding"),
                FamilyId("walk_uniform"),
            ],
            skip: &[],
        },
        // centre_tiled -6..+17%, centre_sliding +8..+19%, walk_uniform +13..+17%, neighbour_sliding +18%,
        // batched_on +23%
        SitePriority {
            site: COLLAB_SEARCH,
            basis: Basis::Measured,
            order: &[
                BASELINE_FAMILY,
                FamilyId("centre_tiled"),
                FamilyId("centre_sliding"),
                FamilyId("walk_uniform"),
                FamilyId("neighbour_sliding"),
                FamilyId("batched_on"),
            ],
            skip: &[],
        },
        // groups8 0%, groups4 0..+10%, groups16 +6..+10%
        SitePriority {
            site: COLLAB_FILTER,
            basis: Basis::Measured,
            order: &[FamilyId("groups8"), FamilyId("groups4"), FamilyId("groups16")],
            skip: &[],
        },
        // separable -52..-44%, direct 0%
        SitePriority {
            site: NLM_WINDOW,
            basis: Basis::Measured,
            order: &[FamilyId("separable"), FamilyId("direct")],
            skip: &[],
        },
        // No timings, default order
        SitePriority {
            site: NLM_ACCUMULATE,
            basis: Basis::Expected,
            order: &[FamilyId("thin32x16"), FamilyId("thin32x8")],
            skip: &[],
        },
        // serial 0%, coop -2..+1%
        SitePriority {
            site: NL4D_MV_REGULARISE,
            basis: Basis::Measured,
            order: &[FamilyId("serial"), FamilyId("coop")],
            skip: &[],
        },
    ],
    // split -7%, packed -3%, baseline 0%, f32 +3%
    profiles_basis: Basis::Measured,
    profiles: &[
        PipelineProfile::SplitSearch,
        PipelineProfile::PackedAtomics,
        PipelineProfile::Baseline,
        PipelineProfile::F32Search,
    ],
};

/// AMD on ROCm, from R9700 timings.
const AMD_ROCM: PriorityTable = PriorityTable {
    sites: &[
        // centre_tiled -15..-1%, batched_on -7..-4%, neighbour_sliding -3..-2%, centre_sliding -4..-3%,
        // walk_uniform n/a, groups4 n/a, groups16 n/a
        SitePriority {
            site: COLLAB_FUSED,
            basis: Basis::Measured,
            order: &[
                BASELINE_FAMILY,
                FamilyId("centre_tiled"),
                FamilyId("batched_on"),
                FamilyId("neighbour_sliding"),
                FamilyId("centre_sliding"),
                FamilyId("walk_uniform"),
                FamilyId("groups4"),
                FamilyId("groups16"),
            ],
            skip: &[],
        },
        // centre_tiled -20..-15%, centre_sliding -16%, neighbour_sliding n/a, batched_on n/a,
        // walk_uniform n/a
        SitePriority {
            site: COLLAB_SEARCH,
            basis: Basis::Measured,
            order: &[
                BASELINE_FAMILY,
                FamilyId("centre_tiled"),
                FamilyId("centre_sliding"),
                FamilyId("neighbour_sliding"),
                FamilyId("batched_on"),
                FamilyId("walk_uniform"),
            ],
            skip: &[],
        },
        // groups4 -3..0%, groups8 0%, groups16 +1..+3%
        SitePriority {
            site: COLLAB_FILTER,
            basis: Basis::Measured,
            order: &[FamilyId("groups4"), FamilyId("groups8"), FamilyId("groups16")],
            skip: &[],
        },
        // separable -42..-40%, direct 0%
        SitePriority {
            site: NLM_WINDOW,
            basis: Basis::Measured,
            order: &[FamilyId("separable"), FamilyId("direct")],
            skip: &[],
        },
        // No timings, default order
        SitePriority {
            site: NLM_ACCUMULATE,
            basis: Basis::Expected,
            order: &[FamilyId("thin32x16"), FamilyId("thin32x8")],
            skip: &[],
        },
        // coop -72..-67%, serial 0%
        SitePriority {
            site: NL4D_MV_REGULARISE,
            basis: Basis::Measured,
            order: &[FamilyId("coop"), FamilyId("serial")],
            skip: &[],
        },
    ],
    // No timings, expected order
    profiles_basis: Basis::Expected,
    profiles: &[
        PipelineProfile::SplitSearch,
        PipelineProfile::PackedAtomics,
        PipelineProfile::Baseline,
        PipelineProfile::F32Search,
    ],
};

/// NVIDIA from Volta onwards, from RTX 3050 Ti timings.
const NVIDIA: PriorityTable = PriorityTable {
    sites: &[
        // centre_tiled -6..-3%, groups16 -5..0%, batched_on -1%, walk_uniform +3%, groups4 n/a,
        // centre_sliding n/a, neighbour_sliding n/a
        SitePriority {
            site: COLLAB_FUSED,
            basis: Basis::Measured,
            order: &[
                BASELINE_FAMILY,
                FamilyId("centre_tiled"),
                FamilyId("groups16"),
                FamilyId("batched_on"),
                FamilyId("walk_uniform"),
                FamilyId("groups4"),
                FamilyId("centre_sliding"),
                FamilyId("neighbour_sliding"),
            ],
            skip: &[],
        },
        // centre_tiled -6..-3%, batched_on -1%, walk_uniform +3%, centre_sliding n/a, neighbour_sliding n/a
        SitePriority {
            site: COLLAB_SEARCH,
            basis: Basis::Measured,
            order: &[
                BASELINE_FAMILY,
                FamilyId("centre_tiled"),
                FamilyId("batched_on"),
                FamilyId("walk_uniform"),
                FamilyId("centre_sliding"),
                FamilyId("neighbour_sliding"),
            ],
            skip: &[],
        },
        // No timings, expected order
        SitePriority {
            site: COLLAB_FILTER,
            basis: Basis::Expected,
            order: &[FamilyId("groups16"), FamilyId("groups8"), FamilyId("groups4")],
            skip: &[],
        },
        // separable -42..-39%, direct 0%
        SitePriority {
            site: NLM_WINDOW,
            basis: Basis::Measured,
            order: &[FamilyId("separable"), FamilyId("direct")],
            skip: &[],
        },
        // No timings, default order
        SitePriority {
            site: NLM_ACCUMULATE,
            basis: Basis::Expected,
            order: &[FamilyId("thin32x16"), FamilyId("thin32x8")],
            skip: &[],
        },
        // coop -56..-29%, serial 0%
        SitePriority {
            site: NL4D_MV_REGULARISE,
            basis: Basis::Measured,
            order: &[FamilyId("coop"), FamilyId("serial")],
            skip: &[],
        },
    ],
    // No timings, expected order
    profiles_basis: Basis::Expected,
    profiles: &[
        PipelineProfile::SplitSearch,
        PipelineProfile::PackedAtomics,
        PipelineProfile::Baseline,
        PipelineProfile::F32Search,
    ],
};

/// NVIDIA before Volta.
const NVIDIA_PRE_VOLTA: PriorityTable = PriorityTable {
    sites: &[
        // No timings, expected order
        SitePriority {
            site: COLLAB_FUSED,
            basis: Basis::Expected,
            order: &[
                BASELINE_FAMILY,
                FamilyId("centre_tiled"),
                FamilyId("batched_on"),
                FamilyId("groups16"),
                FamilyId("groups4"),
                FamilyId("centre_sliding"),
                FamilyId("neighbour_sliding"),
            ],
            skip: &[],
        },
        // No timings, expected order
        SitePriority {
            site: COLLAB_SEARCH,
            basis: Basis::Expected,
            order: &[
                BASELINE_FAMILY,
                FamilyId("centre_tiled"),
                FamilyId("batched_on"),
                FamilyId("centre_sliding"),
                FamilyId("neighbour_sliding"),
            ],
            skip: &[],
        },
        // No timings, expected order
        SitePriority {
            site: COLLAB_FILTER,
            basis: Basis::Expected,
            order: &[FamilyId("groups16"), FamilyId("groups8"), FamilyId("groups4")],
            skip: &[],
        },
        // No timings, expected order
        SitePriority {
            site: NLM_WINDOW,
            basis: Basis::Expected,
            order: &[FamilyId("separable"), FamilyId("direct")],
            skip: &[],
        },
        // No timings, default order
        SitePriority {
            site: NLM_ACCUMULATE,
            basis: Basis::Expected,
            order: &[FamilyId("thin32x16"), FamilyId("thin32x8")],
            skip: &[],
        },
        // No timings, expected order
        SitePriority {
            site: NL4D_MV_REGULARISE,
            basis: Basis::Expected,
            order: &[FamilyId("coop"), FamilyId("serial")],
            skip: &[],
        },
    ],
    // No timings, expected order
    profiles_basis: Basis::Expected,
    profiles: &[
        PipelineProfile::SplitSearch,
        PipelineProfile::PackedAtomics,
        PipelineProfile::Baseline,
        PipelineProfile::F32Search,
    ],
};

/// Apple GPUs.
const APPLE: PriorityTable = PriorityTable {
    sites: &[
        // No timings, expected order
        SitePriority {
            site: COLLAB_FUSED,
            basis: Basis::Expected,
            order: &[
                BASELINE_FAMILY,
                FamilyId("groups4"),
                FamilyId("centre_tiled"),
                FamilyId("batched_on"),
                FamilyId("walk_uniform"),
                FamilyId("groups16"),
                FamilyId("centre_sliding"),
                FamilyId("neighbour_sliding"),
            ],
            skip: &[],
        },
        // No timings, expected order
        SitePriority {
            site: COLLAB_SEARCH,
            basis: Basis::Expected,
            order: &[
                BASELINE_FAMILY,
                FamilyId("centre_tiled"),
                FamilyId("batched_on"),
                FamilyId("walk_uniform"),
                FamilyId("centre_sliding"),
                FamilyId("neighbour_sliding"),
            ],
            skip: &[],
        },
        // No timings, expected order
        SitePriority {
            site: COLLAB_FILTER,
            basis: Basis::Expected,
            order: &[FamilyId("groups4"), FamilyId("groups8"), FamilyId("groups16")],
            skip: &[],
        },
        // No timings, expected order
        SitePriority {
            site: NLM_WINDOW,
            basis: Basis::Expected,
            order: &[FamilyId("separable"), FamilyId("direct")],
            skip: &[],
        },
        // No timings, expected order
        SitePriority {
            site: NLM_ACCUMULATE,
            basis: Basis::Expected,
            order: &[FamilyId("thin32x8"), FamilyId("thin32x16")],
            skip: &[],
        },
        // No timings, expected order
        SitePriority {
            site: NL4D_MV_REGULARISE,
            basis: Basis::Expected,
            order: &[FamilyId("coop"), FamilyId("serial")],
            skip: &[],
        },
    ],
    // No timings, expected order
    profiles_basis: Basis::Expected,
    profiles: &[
        PipelineProfile::SplitSearch,
        PipelineProfile::Baseline,
        PipelineProfile::F32Search,
        PipelineProfile::PackedAtomics,
    ],
};

/// Intel GPUs.
const INTEL: PriorityTable = PriorityTable {
    sites: &[
        // No timings, expected order
        SitePriority {
            site: COLLAB_FUSED,
            basis: Basis::Expected,
            order: &[
                BASELINE_FAMILY,
                FamilyId("groups4"),
                FamilyId("centre_tiled"),
                FamilyId("walk_uniform"),
                FamilyId("batched_on"),
                FamilyId("groups16"),
                FamilyId("centre_sliding"),
                FamilyId("neighbour_sliding"),
            ],
            skip: &[],
        },
        // No timings, expected order
        SitePriority {
            site: COLLAB_SEARCH,
            basis: Basis::Expected,
            order: &[
                BASELINE_FAMILY,
                FamilyId("centre_tiled"),
                FamilyId("walk_uniform"),
                FamilyId("batched_on"),
                FamilyId("centre_sliding"),
                FamilyId("neighbour_sliding"),
            ],
            skip: &[],
        },
        // No timings, expected order
        SitePriority {
            site: COLLAB_FILTER,
            basis: Basis::Expected,
            order: &[FamilyId("groups4"), FamilyId("groups8"), FamilyId("groups16")],
            skip: &[],
        },
        // No timings, expected order
        SitePriority {
            site: NLM_WINDOW,
            basis: Basis::Expected,
            order: &[FamilyId("separable"), FamilyId("direct")],
            skip: &[],
        },
        // No timings, expected order
        SitePriority {
            site: NLM_ACCUMULATE,
            basis: Basis::Expected,
            order: &[FamilyId("thin32x8"), FamilyId("thin32x16")],
            skip: &[],
        },
        // No timings, expected order
        SitePriority {
            site: NL4D_MV_REGULARISE,
            basis: Basis::Expected,
            order: &[FamilyId("coop"), FamilyId("serial")],
            skip: &[],
        },
    ],
    // No timings, expected order
    profiles_basis: Basis::Expected,
    profiles: &[
        PipelineProfile::SplitSearch,
        PipelineProfile::Baseline,
        PipelineProfile::PackedAtomics,
        PipelineProfile::F32Search,
    ],
};

/// The table for a device of `class` on `backend`.
pub fn priority_for(class: DeviceClass, backend: &str) -> &'static PriorityTable {
    match (class, backend) {
        (DeviceClass::Amd, "rocm") => &AMD_ROCM,
        (DeviceClass::Amd, _) => &AMD_VULKAN,
        (DeviceClass::Nvidia, _) => &NVIDIA,
        (DeviceClass::NvidiaPreVolta { .. }, _) => &NVIDIA_PRE_VOLTA,
        (DeviceClass::Apple, _) => &APPLE,
        (DeviceClass::Intel, _) => &INTEL,
        (DeviceClass::Unknown, _) => &NEUTRAL,
    }
}

/// `families` in the order `table` gives `site`.
///
/// Families the table does not list follow in the order `families` holds them.
pub fn family_order(table: &PriorityTable, site: &str, families: &[FamilyId]) -> Vec<FamilyId> {
    let listed = site_priority(table, site).map_or(&[][..], |priority| priority.order);
    let mut ordered: Vec<FamilyId> = listed
        .iter()
        .copied()
        .filter(|family| families.contains(family))
        .collect();

    let unlisted = families.iter().copied().filter(|family| !listed.contains(family));
    ordered.extend(unlisted);

    ordered
}

/// The families `table` leaves out at `site` by default.
pub fn skipped(table: &PriorityTable, site: &str) -> &'static [FamilyId] {
    site_priority(table, site).map_or(&[], |priority| priority.skip)
}

fn site_priority(table: &PriorityTable, site: &str) -> Option<&'static SitePriority> {
    table.sites.iter().find(|priority| priority.site == site)
}

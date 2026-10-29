//! Picks the fastest correct kernel variant for each launch site on the device in use
//!
//! Each tunable launch point is a site, and each way of running it is a candidate. Every candidate
//! declares a [Fidelity] class that says how closely its output matches the baseline's. Results are
//! saved per device, and engines are built from them.
//!
//! - [SiteKey] names one site at one compiled geometry
//! - [KernelPlan] maps site keys to the candidate to run
//! - [PipelineProfile] names a set of pipeline switches compared on whole frames
//! - [TuneResult] holds every timed ranking, so a plan can be rebuilt under different rules
//! - [VariantSite] is one tunable launch point
//! - [DeviceProfile] says which device a result belongs to and what launches fit on it
//! - [Legality] says what a device may run beyond the fit check
//! - [Families] groups a site's candidates, each group with a representative member
//! - [PriorityTable] orders each site's families and the pipeline profiles for one device class
//! - [TuningMode] says whether a run tunes, uses stored results or runs every baseline
//! - [TuneStore] saves and loads results per device
//! - [build_engine] builds an engine from the stored results, keyed by [profile_key]
//! - [SearchMode] says how widely tuning searches a key, and [SiteSearch] records how it did
//! - [tune_frames] runs a tuning pass, and [is_tuned] says whether one is needed
//! - [TunedGroup] tunes on the first frames of a stream, then replays them through the tuned engines

mod budget;
#[cfg(feature = "vulkan")]
mod compile_helpers;
mod config;
mod consts;
mod decide;
mod device;
mod driver;
mod engine_set;
mod family;
mod fidelity;
mod frame;
pub(crate) mod gpu;
mod group;
mod helpers;
mod identity;
mod key;
mod ledger;
pub(crate) mod legality;
mod lock;
mod plan;
pub(crate) mod precompile;
mod priority;
mod profile_key;
mod report;
mod result;
mod search;
mod selector;
mod session;
mod setup;
mod site;
pub(crate) mod sites;
mod store;

#[cfg(test)]
pub(crate) mod tests;

pub use self::budget::{
    Budget,
    BudgetRequest,
    KeyValue,
    WarmUpState,
    WarmUpTracker,
    can_afford,
    key_values,
    pending_keys,
    profile_wins,
    site_min_cost,
    timing_spent,
    tuning_budget,
    tuning_order,
};
pub use self::config::{CacheHook, CachePhase, TuningConfig, TuningMode};
pub use self::consts::*;
pub use self::decide::{
    CandidateStats,
    DecideLimits,
    SampleTiming,
    Sampler,
    Samples,
    SiteTiming,
    median,
    prescreen,
    sample_until_settled,
};
pub use self::device::{DeviceIdentity, DeviceLimits, DeviceProfile};
pub use self::driver::{TuneOutcome, TuneReport, TuneRequest, is_tuned, tune_frames};
pub use self::family::{
    BASELINE_FAMILY,
    Dimension,
    DimensionFamilies,
    DimensionValue,
    Families,
    FamilyId,
    FlatFamily,
    opened_members,
    representatives,
};
pub use self::fidelity::{
    CompareResult,
    ErrorBound,
    Fidelity,
    FidelityClass,
    PlaneDiff,
    diff_integers,
    diff_plane,
};
pub use self::frame::TuningFrame;
pub use self::group::TunedGroup;
#[cfg_attr(
    not(any(test, feature = "vulkan")),
    expect(unused_imports, reason = "only the Vulkan helpers and the tests name it")
)]
pub(crate) use self::helpers::CompileHelpers;
pub use self::helpers::compilation_cache_enabled;
#[cfg(feature = "vulkan")]
pub(crate) use self::helpers::vulkan_compile_helpers;
#[cfg(feature = "cuda")]
pub use self::identity::cuda_details;
pub use self::identity::describe;
#[cfg(feature = "vulkan")]
pub use self::identity::find_adapter;
pub use self::key::{SiteKey, SiteKeyBuilder};#[cfg_attr(
    not(test),
    expect(unused_imports, reason = "only the tests name it through this module")
)]
pub(crate) use self::ledger::CensusEntry;
pub(crate) use self::ledger::{TuneLedger, TunePhase};
pub use self::legality::{
    AMD_VENDOR_ID,
    APPLE_VENDOR_ID,
    DeviceClass,
    INTEL_VENDOR_ID,
    Legality,
    MIN_PLANE_SIZE,
    NVIDIA_VENDOR_ID,
    PlaneSizeError,
    check_plane_size,
};
pub use self::lock::{POLL_INTERVAL, open_lock_file, wait_for_lock};
pub use self::plan::{KernelPlan, PipelineProfile};
pub use self::precompile::Precompiled;
pub use self::priority::{Basis, PriorityTable, SitePriority, family_order, priority_for, skipped};
pub use self::profile_key::{profile_key, tuning_frame_count};
pub use self::result::{
    CandidateRecord,
    CompileProgress,
    ProfileRanking,
    ProfileRecord,
    SiteRanking,
    TuneResult,
    beats_baseline_profile,
    choose,
    choose_profile,
};
pub use self::search::{
    SearchMode,
    SearchOutcome,
    SearchSpace,
    SearchStep,
    SearchedCandidate,
    SiteSearch,
    builds_up,
    first_timed,
    opens_family,
    search_candidates,
};
pub(crate) use self::selector::{EngineKernels, KernelSelector};
pub(crate) use self::session::{calibrate, rewarm};
pub use self::setup::build_engine;
pub use self::site::{FitError, LaunchShape, SiteCandidate, VariantSite, fit_candidates, legal_candidates};
#[cfg_attr(
    not(test),
    expect(
        unused_imports,
        reason = "only the tests read the remembered results directly"
    )
)]
pub(crate) use self::store::remembered;
pub use self::store::{StoreError, TuneStore};

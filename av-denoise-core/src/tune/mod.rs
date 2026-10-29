//! Picks the fastest correct kernel variant for each launch site on the device in use
//!
//! Each tunable launch point is a site, and each way of running it is a candidate. Every candidate
//! declares a [Fidelity] class that says how closely its output matches the baseline's.
//!
//! - [SiteKey] names one site at one compiled geometry
//! - [KernelPlan] maps site keys to the candidate to run
//! - [PipelineProfile] names a set of pipeline switches compared on whole frames
//! - [TuneResult] holds every timed ranking, so a plan can be rebuilt under different rules
//! - [VariantSite] is one tunable launch point, and [KernelSelector] launches the planned candidate at each
//! - [DeviceProfile] says which device a result belongs to and what launches fit on it
//! - [TuneLedger] is the shared state a tuning pass uses to drive engines in place
//! - [Legality] says what a device may run beyond the fit check
//! - [Families] groups a site's candidates, each group with a representative member
//! - [PriorityTable] orders each site's families and the pipeline profiles for one device class
//! - [TuningMode] says whether a run tunes, uses stored results or runs every baseline
//! - [TuneStore] saves and loads results per device
//! - [build_engine] builds an engine from the stored results, keyed by [profile_key]
//! - [SearchMode] says how widely tuning searches a key, and [SiteSearch] records how it did

mod budget;
#[cfg(feature = "vulkan")]
mod compile_helpers;
mod config;
mod consts;
mod decide;
mod device;
mod family;
mod fidelity;
pub(crate) mod gpu;
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
#[cfg(feature = "vulkan")]
pub use self::compile_helpers::install_compile_helpers;
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
#[cfg(feature = "vulkan")]
pub use self::helpers::vulkan_compile_helpers;
pub use self::helpers::{CompileHelpers, HelperFactory, compilation_cache_enabled};
#[cfg(feature = "cuda")]
pub use self::identity::cuda_details;
pub use self::identity::describe;
#[cfg(feature = "vulkan")]
pub use self::identity::find_adapter;
pub use self::key::{SiteKey, SiteKeyBuilder, bucket_dim};
pub use self::ledger::{CensusEntry, TuneLedger, TunePhase};
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
pub use self::lock::{POLL_INTERVAL, TuneLock, TuneLockOutcome, open_lock_file, wait_for_lock};
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
pub use self::selector::{EngineKernels, KernelSelector};
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
pub use self::store::{StoreError, TuneStore, reload, remember, resolve};

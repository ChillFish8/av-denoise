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

mod budget;
mod consts;
mod decide;
mod device;
mod fidelity;
pub(crate) mod gpu;
mod key;
mod ledger;
pub(crate) mod legality;
mod plan;
pub(crate) mod precompile;
mod result;
mod selector;
mod session;
mod site;
pub(crate) mod sites;

#[cfg(test)]
pub(crate) mod tests;

pub use self::budget::{
    Budget,
    BudgetRequest,
    KeyValue,
    WarmUpState,
    WarmUpTracker,
    can_afford,
    profile_wins,
    site_min_cost,
    timing_spent,
    tuning_budget,
    tuning_order,
};
pub use self::consts::*;
pub use self::decide::{
    CandidateStats,
    DecideLimits,
    SampleTiming,
    Sampler,
    Samples,
    SiteTiming,
    Staging,
    median,
    prescreen,
    sample_in_stages,
    sample_in_stages_with,
    sample_until_settled,
};
pub use self::device::{DeviceIdentity, DeviceLimits, DeviceProfile};
pub use self::fidelity::{
    CompareResult,
    ErrorBound,
    Fidelity,
    FidelityClass,
    PlaneDiff,
    diff_integers,
    diff_plane,
};
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
pub use self::plan::{KernelPlan, PipelineProfile};
pub use self::precompile::Precompiled;
pub use self::result::{
    CandidateRecord,
    ProfileRanking,
    ProfileRecord,
    SiteRanking,
    TuneResult,
    beats_baseline_profile,
    choose,
    choose_profile,
};
pub use self::selector::{EngineKernels, KernelSelector};
pub(crate) use self::session::{calibrate, rewarm};
pub use self::site::{FitError, LaunchShape, SiteCandidate, VariantSite, fit_candidates, legal_candidates};

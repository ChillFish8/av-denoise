//! Picks the fastest correct kernel variant for each launch site on the device in use
//!
//! Each tunable launch point is a site, and each way of running it is a candidate. Every candidate
//! declares a [Fidelity] class that says how closely its output matches the baseline's.
//!
//! - [SiteKey] names one site at one compiled geometry
//! - [KernelPlan] maps site keys to the candidate to run
//! - [PipelineProfile] names a set of pipeline switches compared on whole frames
//! - [TuneResult] holds every timed ranking, so a plan can be rebuilt under different rules
//! - [DeviceProfile] says which device a result belongs to and what launches fit on it

mod budget;
mod consts;
mod decide;
mod device;
mod fidelity;
mod key;
mod plan;
mod result;

#[cfg(test)]
mod tests;

pub use self::budget::{
    Budget,
    BudgetRequest,
    KeyValue,
    WarmUpState,
    WarmUpTracker,
    can_afford,
    profile_wins,
    site_min_cost,
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
    median,
    prescreen,
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
pub use self::plan::{KernelPlan, PipelineProfile};
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

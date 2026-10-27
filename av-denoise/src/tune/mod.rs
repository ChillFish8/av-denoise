//! Kernel tuning for the device in use
//!
//! av-denoise times several launch shapes and kernels for each launch site on real frames, keeps
//! the fastest correct one, and saves the result next to the compiled kernel cache so later runs
//! skip tuning.
//!
//! - [TuneStore] saves and loads results per device
//! - [TuningMode] says whether a run tunes, uses stored results or runs every baseline
//! - [tune_planes] runs a tuning pass, and [is_tuned] says whether one is needed

mod driver;
mod mode;
mod report;
mod setup;
mod store;

#[cfg(test)]
mod tests;

pub use av_denoise_core::tune::{
    BudgetRequest,
    DeviceIdentity,
    DeviceProfile,
    PipelineProfile,
    SearchMode,
    TuneResult,
};

pub(crate) use self::driver::tune_planar;
pub use self::driver::{TuneOutcome, TuneReport, TuneRequest, is_tuned, tune_planes};
pub(crate) use self::mode::profile_key;
#[cfg_attr(
    not(test),
    expect(unused_imports, reason = "only the tests call it from outside its module")
)]
pub(crate) use self::mode::resolve_mode;
pub use self::mode::{TUNE_ENV, TuningMode, tuning_frame_count};
pub(crate) use self::setup::{KernelRequest, KernelSetup, engine_kernels};
#[cfg_attr(
    not(test),
    expect(
        unused_imports,
        reason = "only the tests read the remembered results directly"
    )
)]
pub(crate) use self::store::remembered;
pub use self::store::{StoreError, TuneStore};
pub(crate) use self::store::{reload, remember, resolve};
pub use crate::backend::selected_device;

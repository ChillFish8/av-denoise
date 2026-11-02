//! Kernel tuning for the planes av-denoise denoises
//!
//! This module connects the tuner to the CLI and plugin settings. It reads the tuning mode override
//! from the environment, picks the store directory, keeps the compiled kernel cache in step, and
//! uploads planar frames for a tuning pass.
//!
//! - [default_store_dir] says where results are kept
//! - [TuningMode] says whether a run tunes, uses stored results or runs every baseline
//! - [tuning_config] gathers what a tuning pass is told
//! - [tune_planes] runs a tuning pass, and [is_tuned] says whether one is needed

mod mode;
mod store;

#[cfg(test)]
mod tests;

pub use av_denoise_core::tune::{
    BudgetRequest,
    DeviceIdentity,
    DeviceProfile,
    PipelineProfile,
    SearchMode,
    StoreError,
    TuneOutcome,
    TuneReport,
    TuneRequest,
    TuneResult,
    TuneStore,
    TuningConfig,
    TuningMode,
};

#[cfg(test)]
pub(crate) use self::mode::resolve_mode;
pub use self::mode::{TUNE_ENV, tuning_frame_count, tuning_mode_from_env};
pub use self::store::default_store_dir;
use crate::backend::selected_client;
pub use crate::backend::selected_device;
use crate::cache::kernel_cache_hook;
use crate::planar::{FrameLayout, PlanarDenoiser, PlaneOptions, Planes, plane_spec};

/// The tuning config for `options`, with the default budget.
///
/// [TUNE_ENV] overrides the mode, results go in the [default_store_dir], and the kernel cache is
/// kept in step with other processes.
pub fn tuning_config(options: &PlaneOptions) -> TuningConfig {
    let mode = tuning_mode_from_env(options.tuning);
    let store = default_store_dir();
    let cache_hook = kernel_cache_hook();

    TuningConfig {
        mode,
        store,
        budget: BudgetRequest::default(),
        deterministic: options.deterministic,
        cache_hook: Some(cache_hook),
    }
}

/// Tunes the planes `options` denoises at `layout` on `frames` within `budget`, and saves the result.
///
/// The frames are uploaded through a denoiser of its own, which is dropped before the pass runs.
/// [TuningMode::Off] tunes nothing, and any other mode tunes.
pub fn tune_planes(
    options: &PlaneOptions,
    layout: FrameLayout,
    frames: &[Planes],
    budget: BudgetRequest,
    request: &TuneRequest,
) -> TuneReport {
    let defaults = tuning_config(options);
    let config = TuningConfig { budget, ..defaults };

    if config.mode == TuningMode::Off {
        return TuneReport::skipped("tuning is off");
    }

    let untuned = PlaneOptions {
        tuning: TuningMode::Off,
        ..options.clone()
    };
    let created = PlanarDenoiser::create(&untuned, layout);
    let planar = match created {
        Ok(planar) => planar,
        Err(error) => {
            let reason = format!("{error:#}");
            return TuneReport::skipped(reason);
        },
    };

    planar.tune_and_drop(frames, &config, request)
}

/// Whether the store already holds a complete result for every plane `options` denoises at `layout`.
///
/// It opens the selected device without building any engines. Answers `true` when there is nothing
/// to tune, because tuning is not [TuningMode::Auto] or no device can be opened.
pub fn is_tuned(options: &PlaneOptions, layout: FrameLayout) -> bool {
    let config = tuning_config(options);
    if config.mode != TuningMode::Auto {
        return true;
    }

    let opened = selected_client(&options.accelerators, &options.device);
    let Ok((device, client)) = opened else {
        return true;
    };

    let spec = plane_spec(options, layout);
    let specs = [spec];
    av_denoise_core::tune::is_tuned(&device, &client, &specs, &config)
}

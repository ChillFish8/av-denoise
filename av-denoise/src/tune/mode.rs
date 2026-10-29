use av_denoise_core::tune::{TuningMode, tuning_frame_count as group_frame_count};

use crate::planar::{FrameLayout, PlaneOptions, plane_specs};

/// The environment variable that overrides the tuning mode a caller asked for.
pub const TUNE_ENV: &str = "AV_DENOISE_TUNE";

/// `flag`, unless [TUNE_ENV] names another mode, which wins.
pub fn tuning_mode_from_env(flag: TuningMode) -> TuningMode {
    let env = std::env::var(TUNE_ENV).ok();
    resolve_mode(flag, env.as_deref())
}

/// `flag`, unless `env` names a valid mode. An invalid `env` is logged and ignored.
pub(crate) fn resolve_mode(flag: TuningMode, env: Option<&str>) -> TuningMode {
    let Some(text) = env else {
        return flag;
    };

    match text.parse() {
        Ok(mode) => mode,
        Err(error) => {
            tracing::warn!(%error, "ignoring {TUNE_ENV}");
            flag
        },
    }
}

/// How many frames a tuning pass for the planes `options` denoises at `layout` reads.
pub fn tuning_frame_count(options: &PlaneOptions, layout: FrameLayout) -> usize {
    let specs = plane_specs(options, layout);
    group_frame_count(&specs)
}

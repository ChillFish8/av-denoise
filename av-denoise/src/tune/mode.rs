use av_denoise_core::DenoisingMode;
use av_denoise_core::tune::{TUNE_EXTRA_FRAMES, TuningMode, bucket_dim};

use crate::host::{Algorithm, DenoiserOptions};
use crate::planar::PlaneOptions;

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

/// The key a plane's profile ranking is stored under.
///
/// It names the algorithm with its settings, the channels, the temporal mode, the depth and the
/// plane size rounded up to powers of two. Tuning switches are left out, so they share rankings.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "only the legacy key tests render the old keys")
)]
pub(crate) fn profile_key(options: &DenoiserOptions, width: u32, height: u32) -> String {
    let width_bucket = bucket_dim(width);
    let height_bucket = bucket_dim(height);

    format!(
        "{:?}|{:?}|{:?}|{:?}|w{width_bucket}|h{height_bucket}",
        options.algorithm, options.channel_mode, options.mode, options.depth,
    )
}

/// The temporal span `options` reads, plus eight frames.
pub fn tuning_frame_count(options: &PlaneOptions) -> usize {
    let radius = match options.mode {
        DenoisingMode::Spacial => 0,
        DenoisingMode::Temporal { radius } => radius as usize,
    };

    let span = match options.algorithm {
        Algorithm::Nl4d(_) => 4 * radius + 1,
        _ => 2 * radius + 1,
    };

    span + TUNE_EXTRA_FRAMES
}

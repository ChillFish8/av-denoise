use std::fmt;
use std::str::FromStr;

use av_denoise_core::DenoisingMode;
use av_denoise_core::tune::{TUNE_EXTRA_FRAMES, bucket_dim};

use crate::host::{Algorithm, DenoiserOptions};
use crate::planar::PlaneOptions;

/// The environment variable that overrides the tuning mode a caller asked for.
pub const TUNE_ENV: &str = "AV_DENOISE_TUNE";

/// How a run picks the kernel candidate at each launch site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TuningMode {
    /// Uses stored results and lets a tuning pass tune the launch sites in place.
    ///
    /// The pass ranks the keys the stored results lack, most costly first, and saves the rankings.
    #[default]
    Auto,
    /// Uses stored results only. Keys with none run their baseline.
    Cached,
    /// Runs every baseline.
    Off,
}

impl FromStr for TuningMode {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let lowered = text.trim().to_ascii_lowercase();

        match lowered.as_str() {
            "auto" => Ok(TuningMode::Auto),
            "cached" => Ok(TuningMode::Cached),
            "off" => Ok(TuningMode::Off),
            _ => {
                let message = format!("unknown tuning mode '{text}', expected auto, cached or off");
                Err(message)
            },
        }
    }
}

impl fmt::Display for TuningMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            TuningMode::Auto => "auto",
            TuningMode::Cached => "cached",
            TuningMode::Off => "off",
        };

        formatter.write_str(name)
    }
}

impl TuningMode {
    /// This mode, unless [TUNE_ENV] names another, which wins.
    pub fn with_env_override(self) -> Self {
        let env = std::env::var(TUNE_ENV).ok();
        resolve_mode(self, env.as_deref())
    }

    /// The mode a denoiser runs with once tuning is done.
    ///
    /// [TuningMode::Auto] becomes [TuningMode::Cached], so the denoiser builds from stored results
    /// and carries no tuning ledger.
    pub fn after_tuning(self) -> Self {
        match self {
            TuningMode::Auto => TuningMode::Cached,
            other => other,
        }
    }
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

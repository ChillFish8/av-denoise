use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use super::budget::BudgetRequest;

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

/// A point in a tuning pass where the kernel cache may need attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePhase {
    /// The tuning lock was just taken.
    LockTaken,
    /// The kernels a pass needs were just compiled.
    KernelsCompiled,
}

/// Called at each [CachePhase] of a tuning pass.
pub type CacheHook = Arc<dyn Fn(CachePhase) + Send + Sync>;

/// Everything a tuning pass is told by its caller.
#[derive(Clone, Default)]
pub struct TuningConfig {
    pub mode: TuningMode,
    /// The directory results are stored in. With none, results last for the process only.
    pub store: Option<PathBuf>,
    pub budget: BudgetRequest,
    pub deterministic: bool,
    pub cache_hook: Option<CacheHook>,
}

impl fmt::Debug for TuningConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TuningConfig")
            .field("mode", &self.mode)
            .field("store", &self.store)
            .field("budget", &self.budget)
            .field("deterministic", &self.deterministic)
            .field("cache_hook", &self.cache_hook.is_some())
            .finish()
    }
}

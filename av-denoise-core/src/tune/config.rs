use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

use super::budget::BudgetRequest;

/// How a run picks the kernel candidate at each launch site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TuningMode {
    /// Uses stored results and lets a tuning pass rank the keys they lack.
    ///
    /// The pass runs on engines of its own, ranks the missing keys most costly first, and saves the
    /// rankings.
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
    /// [TuningMode::Auto] becomes [TuningMode::Cached], so the denoiser builds from the stored
    /// results its tuning pass just saved.
    pub fn after_tuning(self) -> Self {
        match self {
            TuningMode::Auto => TuningMode::Cached,
            other => other,
        }
    }
}

/// Everything a tuning pass is told by its caller.
#[derive(Debug, Clone, Default)]
pub struct TuningConfig {
    pub mode: TuningMode,
    /// The directory results are stored in. With none, results last for the process only.
    pub store: Option<PathBuf>,
    /// `Automatic { frames }` is the expected stream length, used to judge whether tuning pays back.
    pub budget: BudgetRequest,
    /// Picks only bit-identical candidates and the baseline profile.
    pub deterministic: bool,
}

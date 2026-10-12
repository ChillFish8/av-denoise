use std::cmp::Reverse;
use std::time::Duration;

use super::consts::{
    MIN_CANDIDATE_TIME,
    MIN_SITE_SHARE,
    REWARM,
    TUNE_AUTO_CAP,
    TUNE_FLOOR,
    TUNE_FULL,
    TUNE_SHARE,
    TUNE_UNKNOWN,
    WARM_MAX,
    WARM_MIN,
    WARM_TOLERANCE,
    WARM_WINDOW,
};
use super::decide::median;
use super::key::SiteKey;
use super::result::beats_baseline_profile;

/// How a run asks for its tuning budget.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BudgetRequest {
    /// Scales with the encode, `frames` long when the length is known.
    Automatic { frames: Option<u64> },
    /// Explicit tuning, which gets [TUNE_FULL](crate::tune::TUNE_FULL).
    Full,
    /// A budget the caller set, in seconds.
    Seconds(f64),
}

/// The time a tuning pass may spend, warm-up and profile comparison included.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    pub total: Duration,
    /// Whether this is the full budget, which also tunes sites under a small share of frame time.
    pub full: bool,
}

/// The budget for `request`, given the warm-up's steady per-frame time.
///
/// An automatic budget is [TUNE_SHARE](crate::tune::TUNE_SHARE) of the estimated encode, capped at
/// [TUNE_AUTO_CAP](crate::tune::TUNE_AUTO_CAP). One below [TUNE_FLOOR](crate::tune::TUNE_FLOOR)
/// returns `None`, and the run uses stored results only.
pub fn tuning_budget(request: BudgetRequest, frame_time: Duration) -> Option<Budget> {
    match request {
        BudgetRequest::Full => Some(Budget {
            total: TUNE_FULL,
            full: true,
        }),
        BudgetRequest::Seconds(seconds) => {
            let total = Duration::from_secs_f64(seconds.max(0.0));
            let full = total >= TUNE_FULL;

            Some(Budget { total, full })
        },
        BudgetRequest::Automatic { frames: None } => Some(Budget {
            total: TUNE_UNKNOWN,
            full: false,
        }),
        BudgetRequest::Automatic { frames: Some(frames) } => {
            let encode_seconds = frame_time.as_secs_f64() * frames as f64;
            let share_seconds = encode_seconds * TUNE_SHARE;
            let share = Duration::from_secs_f64(share_seconds);
            let total = share.min(TUNE_AUTO_CAP);

            if total < TUNE_FLOOR {
                return None;
            }

            Some(Budget { total, full: false })
        },
    }
}

/// The least time tuning one site with `candidates` candidates takes.
pub fn site_min_cost(candidates: usize) -> Duration {
    let count = candidates as u32;
    REWARM + MIN_CANDIDATE_TIME * count
}

/// Whether `needed` more time fits in `budget` after `spent`.
pub fn can_afford(budget: &Budget, spent: Duration, needed: Duration) -> bool {
    let total_needed = spent + needed;
    total_needed <= budget.total
}

/// A site key and what it costs per frame on the baseline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyValue {
    pub key: SiteKey,
    pub per_frame: Duration,
    /// How many candidates the key has on this device.
    pub candidates: usize,
}

/// Orders keys most valuable first.
///
/// Keys under [MIN_SITE_SHARE](crate::tune::MIN_SITE_SHARE) of `frame_time` are dropped unless the
/// budget is full.
pub fn tuning_order(mut values: Vec<KeyValue>, frame_time: Duration, full: bool) -> Vec<KeyValue> {
    let threshold = frame_time.as_secs_f64() * MIN_SITE_SHARE;

    values.retain(|value| full || value.per_frame.as_secs_f64() >= threshold);
    values.sort_by_key(|value| Reverse(value.per_frame));
    values
}

/// Where the warm-up stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarmUpState {
    Warming,
    /// Frame time held steady, at this median.
    Stable(Duration),
    /// The warm-up ran out of time, with this median over the last frames.
    Capped(Duration),
}

/// Watches per-frame times until they hold steady.
///
/// Times are steady when the last [WARM_WINDOW](crate::tune::WARM_WINDOW) frames are within
/// [WARM_TOLERANCE](crate::tune::WARM_TOLERANCE) of each other, at least
/// [WARM_MIN](crate::tune::WARM_MIN) into the warm-up. At [WARM_MAX](crate::tune::WARM_MAX) the
/// warm-up ends regardless.
#[derive(Debug, Clone, Default)]
pub struct WarmUpTracker {
    recent: Vec<Duration>,
}

impl WarmUpTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one frame's time and how long the warm-up has run.
    pub fn push(&mut self, frame_time: Duration, elapsed: Duration) -> WarmUpState {
        self.recent.push(frame_time);
        if self.recent.len() > WARM_WINDOW {
            self.recent.remove(0);
        }

        let recent_median = median(&self.recent);

        if elapsed >= WARM_MAX {
            return WarmUpState::Capped(recent_median);
        }

        let window_full = self.recent.len() == WARM_WINDOW;
        if window_full && elapsed >= WARM_MIN && self.is_steady() {
            return WarmUpState::Stable(recent_median);
        }

        WarmUpState::Warming
    }

    /// The median of the frames in the current window, zero before the first frame.
    pub fn recent_median(&self) -> Duration {
        median(&self.recent)
    }

    fn is_steady(&self) -> bool {
        let fastest = self.recent.iter().min().copied().unwrap_or_default();
        let slowest = self.recent.iter().max().copied().unwrap_or_default();
        let fastest_seconds = fastest.as_secs_f64();

        if fastest_seconds == 0.0 {
            return false;
        }

        let spread = slowest.as_secs_f64() / fastest_seconds - 1.0;
        spread <= WARM_TOLERANCE
    }
}

/// Whether `candidate` beats `baseline` by more than the profile margin.
pub fn profile_wins(baseline: Duration, candidate: Duration) -> bool {
    let baseline_ns = baseline.as_nanos() as u64;
    let candidate_ns = candidate.as_nanos() as u64;

    beats_baseline_profile(baseline_ns, candidate_ns)
}

use std::time::Duration;

/// The relative spread under which warm-up frame times count as stable.
pub const WARM_TOLERANCE: f64 = 0.03;

/// How many recent frames the warm-up stability check looks at.
pub const WARM_WINDOW: usize = 5;

/// The shortest warm-up, so laptop clocks have time to settle.
pub const WARM_MIN: Duration = Duration::from_secs(2);

/// The longest warm-up. A pipeline still unstable by then is tuned as it is.
pub const WARM_MAX: Duration = Duration::from_secs(15);

/// The index of the baseline in a site's allowed candidates.
pub const BASELINE_INDEX: usize = 0;

/// The compile time per candidate assumed before a pass has compiled anything.
///
/// It is close to what a cold chroma collab variant takes, the slowest candidate to compile.
pub const FIRST_COMPILE_ESTIMATE: Duration = Duration::from_secs(10);

/// How long the baseline runs just before a site is tuned, so clocks stay up.
pub const REWARM: Duration = Duration::from_millis(200);

/// The shortest time one timing sample may take.
///
/// Back-to-back launches are batched into a sample until it lasts this long, which also suits
/// backends that only timestamp whole passes.
pub const SAMPLE_TARGET: Duration = Duration::from_millis(2);

/// The most launches batched into one sample.
pub const MAX_LAUNCHES_PER_SAMPLE: u32 = 4096;

/// The fewest samples a candidate takes before the stop rule may end its site.
pub const MIN_SAMPLES: usize = 15;

/// The least sampled time a candidate collects before the stop rule may end its site.
pub const MIN_CANDIDATE_TIME: Duration = Duration::from_millis(250);

/// The time after which a site stops sampling once every minimum is met.
pub const SITE_CAP: Duration = Duration::from_secs(3);

/// The time after which a site that still lacks its minimum samples gives up and runs its baseline.
pub const HARD_SITE_CAP: Duration = Duration::from_secs(12);

/// How close to the fastest median a more faithful candidate must be to win instead.
pub const HYSTERESIS: f64 = 0.02;

/// Short samples each candidate takes in a pre-screen.
pub const PRESCREEN_SAMPLES: usize = 3;

/// How much slower than the fastest a candidate may be and survive the pre-screen.
pub const PRESCREEN_MARGIN: f64 = 0.25;

/// How many of the fastest candidates a staged site keeps between stages.
pub const STAGE_KEEP: usize = 3;

/// Frames in one interleaved block of the profile comparison.
pub const PROFILE_BLOCK_FRAMES: usize = 8;

/// The fewest blocks each profile is timed for.
pub const PROFILE_MIN_BLOCKS: usize = 3;

/// How much faster a profile has to be to replace the baseline profile.
pub const PROFILE_MARGIN: f64 = 0.03;

/// The share of the estimated encode time an automatic budget allows.
pub const TUNE_SHARE: f64 = 0.03;

/// The largest automatic budget.
pub const TUNE_AUTO_CAP: Duration = Duration::from_secs(120);

/// The smallest automatic budget worth tuning with. Below it a run uses stored results only.
pub const TUNE_FLOOR: Duration = Duration::from_secs(10);

/// The budget for explicit tuning.
pub const TUNE_FULL: Duration = Duration::from_secs(600);

/// The budget when the clip's length is unknown.
pub const TUNE_UNKNOWN: Duration = Duration::from_secs(30);

/// The share of frame time under which a site is only tuned with the full budget.
pub const MIN_SITE_SHARE: f64 = 0.02;

/// Frames the tuning pass decodes beyond the temporal span.
pub const TUNE_EXTRA_FRAMES: usize = 8;

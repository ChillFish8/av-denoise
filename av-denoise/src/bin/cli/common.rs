use av_denoise::TuningMode;

use super::InputSource;

/// Flags shared by every denoising subcommand.
#[derive(Debug, Clone, clap::Args)]
pub struct CommonArgs {
    /// Where to read frames from.
    ///
    /// A path opens the file with ffms2. Any container or codec
    /// supported by ffmpeg works.
    ///
    /// `-` or `pipe:0` reads a y4m stream from standard input.
    ///
    /// `pipe:N` for `N` of 3 or above reads a y4m stream from an
    /// inherited file descriptor.
    ///
    /// Every input is split by scene, piped or not.
    ///
    /// A file whose name would otherwise be read as a pipe is
    /// reachable by prefixing it, for example `./-`.
    ///
    /// The source's bit depth is detected automatically. 8, 10, and
    /// 12-bit sources are supported and the output keeps the source's
    /// depth. Other depths are rejected with an error.
    #[arg(short, long)]
    pub input: InputSource,

    /// How many scenes to clean in parallel.
    ///
    /// Each worker uses its own GPU memory for the frame ring
    /// buffer, so higher values trade GPU memory for throughput.
    ///
    /// `1` is valid and useful for debugging. Defaults to 2 when unset.
    #[arg(short = 'W', long)]
    pub workers: Option<usize>,

    /// How much memory frames in flight may occupy.
    ///
    /// A frame counts against this from the moment it is decoded until
    /// it has been written, so the budget covers the staging queues,
    /// each worker's GPU pipeline, and the reordering buffer together.
    ///
    /// Takes a size such as `8GB`, `4.5GiB`, or `512MB`, or a plain
    /// byte count. `GB` is 10^9 while `GiB` is 2^30, and the two differ
    /// by 7%.
    ///
    /// Lower it when several `av-denoise` processes share a machine, as
    /// Av1an chunking does. Raise it when one run has the machine to
    /// itself and the decoder is the bottleneck.
    ///
    /// Defaults to 1GiB when unset.
    #[arg(long, value_name = "SIZE", value_parser = parse_frame_budget)]
    pub frame_budget: Option<u64>,

    /// Whether to tune the kernels for this device before denoising.
    ///
    /// `auto` times several kernel variants on the first frames when this configuration has no
    /// stored results, saves the fastest correct ones next to the kernel cache, then denoises.
    /// Later runs on the same device load them and skip tuning. `cached` uses stored results only.
    /// `off` always runs the untuned kernels.
    ///
    /// Some tuned kernels add in a different order, so output can differ from `off` by rounding, and
    /// `--deterministic` keeps it bit identical.
    ///
    /// The `AV_DENOISE_TUNE` environment variable overrides this flag.
    #[arg(long, value_name = "MODE", default_value = "auto", value_parser = parse_tuning_mode)]
    pub tune: TuningMode,

    /// Runs only kernels whose output is bit identical to the untuned kernels.
    #[arg(long)]
    pub deterministic: bool,

    /// How many seconds tuning may take, instead of a share of the clip's estimated encode time.
    ///
    /// Takes a finite number between 0 and 86400.
    #[arg(long, value_name = "SECONDS", allow_negative_numbers = true, value_parser = parse_tune_budget)]
    pub tune_budget: Option<f64>,

    /// Prints where the time went at the end of the run.
    ///
    /// Shows, per plane and per frame, the host time blocked on GPU
    /// readbacks and the host time spent building and queuing work. It
    /// also shows how busy the GPU was where the device has timestamps,
    /// and the frames per second. The report goes to stderr.
    #[arg(long)]
    pub timing_report: bool,
}

/// The longest tuning budget accepted, one day in seconds.
const MAX_TUNE_BUDGET_SECONDS: f64 = 86_400.0;

fn parse_tuning_mode(raw: &str) -> Result<TuningMode, String> {
    raw.parse()
}

/// Reads a tuning budget in seconds, rejecting one that is negative, non-finite or over a day.
fn parse_tune_budget(raw: &str) -> Result<f64, String> {
    let seconds = raw.trim().parse::<f64>().map_err(|_| invalid_tune_budget(raw))?;

    if !is_valid_tune_budget(seconds) {
        let error = invalid_tune_budget(raw);
        return Err(error);
    }

    Ok(seconds)
}

fn invalid_tune_budget(shown: &str) -> String {
    format!("invalid tuning budget '{shown}', give between 0 and {MAX_TUNE_BUDGET_SECONDS} seconds")
}

fn is_valid_tune_budget(seconds: f64) -> bool {
    seconds.is_finite() && (0.0..=MAX_TUNE_BUDGET_SECONDS).contains(&seconds)
}

fn check_tune_budget(seconds: f64) -> Result<(), anyhow::Error> {
    if !is_valid_tune_budget(seconds) {
        let shown = seconds.to_string();
        let message = invalid_tune_budget(&shown);
        anyhow::bail!(message);
    }

    Ok(())
}

impl CommonArgs {
    /// The tuning mode, with `AV_DENOISE_TUNE` applied.
    pub fn tuning_mode(&self) -> TuningMode {
        self.tune.with_env_override()
    }

    /// The tuning budget, rejecting one that is negative, non-finite or over a day.
    pub fn checked_tune_budget(&self) -> Result<Option<f64>, anyhow::Error> {
        if let Some(seconds) = self.tune_budget {
            check_tune_budget(seconds)?;
        }

        Ok(self.tune_budget)
    }
}

/// Reads a size string such as `8GB` into a byte count.
fn parse_frame_budget(raw: &str) -> Result<u64, String> {
    raw.parse::<bytesize::ByteSize>()
        .map(|size| size.as_u64())
        .map_err(|_| {
            format!(
                "invalid frame budget '{raw}'. Give a size such as 8GB, 4.5GiB, or 512MB, or a plain \
                 byte count"
            )
        })
}

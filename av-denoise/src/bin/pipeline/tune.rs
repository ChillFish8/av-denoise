use av_denoise::tune::{
    BudgetRequest,
    SearchMode,
    TuneOutcome,
    TuneReport,
    TuneRequest,
    is_tuned,
    tune_planes,
    tuning_frame_count,
};
use av_denoise::{PlaneOptions, Planes};

use super::convert::decoded_planes;
use super::decode::{DecodeThread, FrameMsg};
use super::source::{self, SourceInfo};
use super::stage::frame_permit_channel;
use crate::cli::InputSource;

/// The budget a run asks for, `seconds` when the user set one.
pub fn budget_request(seconds: Option<f64>, fallback: BudgetRequest) -> BudgetRequest {
    match seconds {
        Some(seconds) => BudgetRequest::Seconds(seconds),
        None => fallback,
    }
}

/// The budget a denoising run asks for.
///
/// Without a budget from the user, it scales with the input's length, or uses the unknown-length
/// budget when the input does not know its length.
pub fn run_budget(tune_budget: Option<f64>, info: &SourceInfo) -> BudgetRequest {
    let frame_count = info.estimated_frames.map(|count| count as u64);
    let automatic = BudgetRequest::Automatic { frames: frame_count };

    budget_request(tune_budget, automatic)
}

/// The options workers build with, which never tune because tuning runs before they start.
pub fn worker_options(planes: &PlaneOptions) -> PlaneOptions {
    let tuning = planes.tuning.after_tuning();

    PlaneOptions {
        tuning,
        ..planes.clone()
    }
}

/// Takes up to `wanted` frames off `frames`, stopping early at the end of the input or at an error.
///
/// An error is kept in place, so it reaches the dispatcher in order like any other frame.
pub fn collect_prefix(frames: &crossbeam_channel::Receiver<FrameMsg>, wanted: usize) -> Vec<FrameMsg> {
    let mut prefix = Vec::with_capacity(wanted);

    while prefix.len() < wanted {
        let Ok(message) = frames.recv() else {
            break;
        };

        let failed = message.is_err();
        prefix.push(message);

        if failed {
            break;
        }
    }

    prefix
}

/// Copies the leading decodable frames of `prefix` into planes, stopping at the first error.
fn prefix_planes(prefix: &[FrameMsg], info: &SourceInfo) -> Vec<Planes> {
    let mut planes = Vec::with_capacity(prefix.len());

    for message in prefix {
        let Ok(decoded) = message else {
            break;
        };

        let converted = decoded_planes(decoded, info.layout);
        match converted {
            Ok(frame) => planes.push(frame),
            Err(error) => {
                tracing::warn!(%error, "cannot read a frame for tuning, tuning on fewer frames");
                break;
            },
        }
    }

    planes
}

fn log_report(report: &TuneReport) {
    match &report.outcome {
        TuneOutcome::Tuned => tracing::info!(
            keys = report.tuned_keys,
            partial_keys = report.partial_keys,
            seconds = report.elapsed.as_secs_f64(),
            compile_seconds = report.compile.as_secs_f64(),
            saved = report.saved,
            "tuned kernels for this device"
        ),
        TuneOutcome::AlreadyTuned => tracing::debug!("kernels are already tuned for this device"),
        TuneOutcome::Skipped(reason) => tracing::debug!(%reason, "kernel tuning skipped"),
    }
}

/// Tunes on the first decoded frames when this configuration needs it, and returns those frames.
///
/// The returned frames go to the dispatcher ahead of the rest, so nothing is decoded twice. Tuning
/// never fails the run, and a decode error among the first frames skips it. At most `permits` frames
/// are held, because each one holds a permit until it is written.
pub fn tune_before_workers(
    planes: &PlaneOptions,
    info: &SourceInfo,
    frames: &crossbeam_channel::Receiver<FrameMsg>,
    permits: usize,
    tune_budget: Option<f64>,
) -> Vec<FrameMsg> {
    if is_tuned(planes, info.layout) {
        return Vec::new();
    }

    let needed = tuning_frame_count(planes);
    let wanted = needed.min(permits);
    let prefix = collect_prefix(frames, wanted);
    let ends_in_error = prefix.last().is_some_and(|message| message.is_err());

    if ends_in_error {
        tracing::debug!("a frame failed to decode, kernel tuning skipped");
        return prefix;
    }

    let decoded = prefix_planes(&prefix, info);

    if decoded.is_empty() {
        return prefix;
    }

    let budget = run_budget(tune_budget, info);
    let request = TuneRequest {
        budget,
        force: false,
        mode: SearchMode::Families,
    };
    let report = tune_planes(planes, info.layout, &decoded, &request);
    log_report(&report);

    prefix
}

/// Decodes the frames tuning needs from `input` and tunes on them, without denoising anything.
pub fn tune_input(
    planes: &PlaneOptions,
    input: &InputSource,
    request: &TuneRequest,
) -> Result<TuneReport, anyhow::Error> {
    let owned_input = input.clone();
    let opener = move || source::open_source(&owned_input);
    let (decode_thread, info) = DecodeThread::spawn(opener)?;

    let wanted = tuning_frame_count(planes);
    let (give, take) = frame_permit_channel(wanted);
    let frames = decode_thread.start(take);
    let prefix = collect_prefix(&frames, wanted);

    // The decoder stops once its frame channel and permits are gone.
    drop(frames);
    drop(give);
    decode_thread.join()?;

    let mut decoded = Vec::with_capacity(prefix.len());

    for message in prefix {
        let frame = message?;
        let frame_planes = decoded_planes(&frame, info.layout)?;
        decoded.push(frame_planes);
    }

    if decoded.is_empty() {
        anyhow::bail!("the input holds no decodable frames");
    }

    let report = tune_planes(planes, info.layout, &decoded, request);
    log_report(&report);

    Ok(report)
}

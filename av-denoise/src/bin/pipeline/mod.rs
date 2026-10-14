mod convert;
mod coordinator;
mod decode;
mod grain_table;
pub mod report;
mod scenes;
mod source;
mod stage;
mod tune;
mod worker;

#[cfg(test)]
mod tests;

use std::io::{IsTerminal, stdout};
use std::path::PathBuf;
use std::time::Instant;

use av_denoise::{DenoisingMode, Depth, FrameLayout, PlaneOptions, SceneGrain};

use self::convert::SourcePixel;
use self::coordinator::spawn_coordinator;
use self::decode::{DecodeThread, FrameMsg};
use self::grain_table::TablePath;
use self::report::{RunSummary, WorkerTiming};
use self::scenes::{Decided, SceneSplitter};
use self::source::{OpenedSource, SourceInfo};
use self::stage::{SceneJob, Stager, checked_frame_permits, frame_permit_channel};
pub use self::tune::{budget_request, tune_input};
use self::worker::{WorkerJoin, WorkerOutput, spawn_workers};
use crate::cli::{InputSource, RunOptions};
use crate::progress::denoise_bar_visible;

/// How a pipeline run uses the machine and what it writes beside the video.
pub struct RunSettings {
    pub workers: usize,
    pub frame_budget_bytes: u64,
    /// Whether to draw the denoising progress bar.
    pub visible: bool,
    /// Where to write the AV1 film grain table, when one was asked for.
    pub grain_table: Option<PathBuf>,
    /// Seconds tuning may take, when the user set a budget.
    pub tune_budget: Option<f64>,
    /// Whether the workers sample device time for the timing report.
    pub timing_report: bool,
}

/// Denoises `input` scene by scene, writing y4m on stdout.
pub fn run(
    opts: &RunOptions,
    input: &InputSource,
    workers: usize,
    frame_budget_bytes: u64,
) -> Result<(), anyhow::Error> {
    let is_terminal = std::io::stderr().is_terminal();
    let visible = denoise_bar_visible(opts.progress, is_terminal);
    let owned_input = input.clone();
    let opener = move || source::open_source(&owned_input);
    let output = stdout();
    let grain_table = opts.grain_table.clone();
    let settings = RunSettings {
        workers,
        frame_budget_bytes,
        visible,
        grain_table,
        tune_budget: opts.tune_budget,
        timing_report: opts.timing_report,
    };

    let summary = run_with(&opts.planes, opener, settings, output)?;

    if opts.timing_report {
        let report = report::format_report(&summary);
        eprint!("{report}");
    }

    Ok(())
}

/// Denoises the source `opener` returns scene by scene, writing y4m into `output`.
///
/// Blocks until every thread in the pipeline has finished. The grain table is written only when
/// every stage succeeds. When the options ask for `auto` tuning and nothing is stored for this
/// configuration, the first frames are tuned on before the workers start, then denoised as usual.
/// Returns where the run's time went.
pub fn run_with<W, F>(
    planes: &PlaneOptions,
    opener: F,
    settings: RunSettings,
    output: W,
) -> Result<RunSummary, anyhow::Error>
where
    W: std::io::Write + Send + 'static,
    F: FnOnce() -> Result<OpenedSource, anyhow::Error> + Send + 'static,
{
    let RunSettings {
        workers,
        frame_budget_bytes,
        visible,
        grain_table,
        tune_budget,
        timing_report,
    } = settings;

    let started = Instant::now();

    if workers == 0 {
        anyhow::bail!("--workers must be at least 1");
    }

    let table = grain_table.as_deref().map(TablePath::create).transpose()?;

    let (decode_thread, info) = DecodeThread::spawn(opener)?;
    let layout = info.layout;

    tracing::info!(
        width = layout.width,
        height = layout.height,
        subsampling = ?layout.subsampling,
        depth = ?layout.depth,
        estimated_frames = ?info.estimated_frames,
        workers,
        "input opened",
    );

    let radius = match planes.mode {
        DenoisingMode::Temporal { radius } => radius,
        DenoisingMode::Spacial => 0,
    };
    let frame_bytes = layout.luma_bytes() + 2 * layout.chroma_bytes();
    let permits = checked_frame_permits(frame_budget_bytes, frame_bytes, workers, radius)?;
    let (give, take) = frame_permit_channel(permits);

    tracing::info!(
        permits,
        frame_bytes,
        ceiling_mib = (permits * frame_bytes) / (1 << 20),
        "frame buffer budget",
    );

    let frames = decode_thread.start(take);
    let prefix = tune::tune_before_workers(planes, &info, &frames, permits, tune_budget);

    let (staged_tx, staged_rx) = crossbeam_channel::bounded::<u64>(1);
    let worker_planes = tune::worker_options(planes);
    let (job_tx, worker_handles, output_rx) = spawn_workers(&worker_planes, layout, workers, timing_report);
    let coordinator_info = info.clone();
    let coordinator = spawn_coordinator(coordinator_info, output_rx, staged_rx, visible, give, output);

    let feed = prefix.into_iter().chain(frames.iter());
    let dispatched = match layout.depth {
        Depth::Eight => dispatch::<u8>(feed, &info, &job_tx),
        Depth::Ten | Depth::Twelve => dispatch::<u16>(feed, &info, &job_tx),
    };

    // Closing the queue tells the workers there are no more scenes, and dropping the frame
    // channel frees a decode thread blocked on a send.
    drop(job_tx);
    drop(frames);

    let staged = dispatched.as_ref().ok().copied();

    // A closed channel means the coordinator failed, and it reports its own error when joined.
    if let Some(staged) = staged {
        let _ = staged_tx.send(staged);
    }

    drop(staged_tx);

    let workers_joined = join_workers(worker_handles);
    let coordinator_joined = coordinator
        .join()
        .map_err(|panic| anyhow::anyhow!("coordinator panicked: {panic:?}"))
        .and_then(|result| result);

    // The decode thread can wait on a permit until the coordinator exits and drops its
    // giving end, so it joins last.
    let decoder_joined = decode_thread.join();

    // Root causes come first. A coordinator error is a failed write. A worker error only shows
    // elsewhere as the dispatcher's disconnect. Dispatch errors carry decode failures, which
    // never make a worker fail.
    //
    // Returning early drops `table`, which removes its temporary file.
    let (scenes, worker_timings) = coordinator_joined
        .and(workers_joined)
        .and_then(|joined| dispatched.map(|_| joined))
        .and_then(|joined| decoder_joined.map(|_| joined))?;

    if let Some(table) = table {
        let rate = info.details.frame_rate;
        let frame_rate = (*rate.numer() as u64, *rate.denom() as u64);
        table.write(&scenes, frame_rate)?;
    }

    let summary = RunSummary {
        wall: started.elapsed(),
        frames: staged.unwrap_or_default(),
        workers: worker_timings,
    };

    Ok(summary)
}

/// Feeds decoded frames through the scene splitter into scene jobs.
///
/// Returns how many frames were staged.
pub fn dispatch<T: SourcePixel>(
    frames: impl Iterator<Item = FrameMsg>,
    info: &SourceInfo,
    jobs: &crossbeam_channel::Sender<SceneJob>,
) -> Result<u64, anyhow::Error> {
    let mut splitter = SceneSplitter::<T>::new(&info.details);
    let mut stager = Stager::new(jobs);

    for message in frames {
        let decoded = message?;
        let frame = T::from_decoded(decoded)
            .ok_or_else(|| anyhow::anyhow!("the decoder changed sample width mid-stream"))?;

        let released = splitter.push(frame);
        stage_decided(&mut stager, released, info.layout)?;
    }

    let tail = splitter.finish();
    stage_decided(&mut stager, tail, info.layout)?;

    let staged = stager.finish();

    if staged == 0 {
        anyhow::bail!("the input holds no decodable frames");
    }

    tracing::info!(staged, "every frame staged");

    Ok(staged)
}

fn stage_decided<T: SourcePixel>(
    stager: &mut Stager,
    decided: Vec<Decided<T>>,
    layout: FrameLayout,
) -> Result<(), anyhow::Error> {
    for frame in decided {
        let planes = T::to_planes(&frame.frame, layout)?;
        stager.stage(planes, frame.starts_scene)?;
    }

    Ok(())
}

/// Joins every worker and gathers the grain and timing they measured.
///
/// Logs each error or panic and returns the first.
fn join_workers(handles: Vec<WorkerJoin>) -> Result<(Vec<SceneGrain>, Vec<WorkerTiming>), anyhow::Error> {
    let mut first_error = None;
    let mut scenes = Vec::new();
    let mut timings = Vec::new();

    for (worker_id, handle) in handles.into_iter().enumerate() {
        let result = handle
            .join()
            .map_err(|panic| anyhow::anyhow!("worker panicked: {panic:?}"))
            .and_then(|result| result);

        let err = match result {
            Ok(WorkerOutput {
                scenes: worker_scenes,
                timing,
            }) => {
                scenes.extend(worker_scenes);
                timings.push(timing);
                continue;
            },
            Err(err) => err,
        };

        tracing::error!(worker_id, error = %err, "worker failed");

        if first_error.is_none() {
            first_error = Some(err);
        }
    }

    first_error.map_or(Ok((scenes, timings)), Err)
}

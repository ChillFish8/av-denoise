mod convert;
mod coordinator;
mod decode;
mod grain_table;
mod scenes;
mod source;
mod stage;
mod worker;

#[cfg(test)]
mod tests;

use std::io::{IsTerminal, stdout};
use std::path::PathBuf;

use av_denoise::{DenoisingMode, Depth, FrameLayout, PlaneOptions, SceneGrain};

use self::convert::SourcePixel;
use self::coordinator::spawn_coordinator;
use self::decode::{DecodeThread, FrameMsg};
use self::grain_table::TablePath;
use self::scenes::{Decided, SceneSplitter};
use self::source::{OpenedSource, SourceInfo};
use self::stage::{SceneJob, Stager, checked_frame_permits, frame_permit_channel};
use self::worker::{WorkerJoin, spawn_workers};
use crate::cli::{InputSource, RunOptions};
use crate::progress::denoise_bar_visible;

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

    run_with(
        &opts.planes,
        opener,
        workers,
        frame_budget_bytes,
        visible,
        output,
        grain_table,
    )
}

/// Denoises the source `opener` returns scene by scene, writing y4m into `output`.
///
/// Blocks until every thread in the pipeline has finished. The grain table is written to
/// `grain_table` only when every stage succeeds.
pub fn run_with<W, F>(
    planes: &PlaneOptions,
    opener: F,
    workers: usize,
    frame_budget_bytes: u64,
    visible: bool,
    output: W,
    grain_table: Option<PathBuf>,
) -> Result<(), anyhow::Error>
where
    W: std::io::Write + Send + 'static,
    F: FnOnce() -> Result<OpenedSource, anyhow::Error> + Send + 'static,
{
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

    let (staged_tx, staged_rx) = crossbeam_channel::bounded::<u64>(1);
    let (job_tx, worker_handles, output_rx) = spawn_workers(planes, layout, workers);
    let coordinator_info = info.clone();
    let coordinator = spawn_coordinator(coordinator_info, output_rx, staged_rx, visible, give, output);
    let frames = decode_thread.start(take);

    let dispatched = match layout.depth {
        Depth::Eight => dispatch::<u8>(&frames, &info, &job_tx),
        Depth::Ten | Depth::Twelve => dispatch::<u16>(&frames, &info, &job_tx),
    };

    // Closing the queue tells the workers there are no more scenes, and dropping the frame
    // channel frees a decode thread blocked on a send.
    drop(job_tx);
    drop(frames);

    // A closed channel means the coordinator failed, and it reports its own error when joined.
    if let Ok(staged) = dispatched {
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
    let outcome = coordinator_joined
        .and(workers_joined)
        .and_then(|scenes| dispatched.map(|_| scenes))
        .and_then(|scenes| decoder_joined.map(|_| scenes));

    // Returning early drops `table`, which removes its temporary file.
    let scenes = outcome?;

    if let Some(table) = table {
        let rate = info.details.frame_rate;
        let frame_rate = (*rate.numer() as u64, *rate.denom() as u64);
        table.write(&scenes, frame_rate)?;
    }

    Ok(())
}

/// Feeds decoded frames through the scene splitter into scene jobs.
///
/// Returns how many frames were staged.
pub fn dispatch<T: SourcePixel>(
    frames: &crossbeam_channel::Receiver<FrameMsg>,
    info: &SourceInfo,
    jobs: &crossbeam_channel::Sender<SceneJob>,
) -> Result<u64, anyhow::Error> {
    let mut splitter = SceneSplitter::<T>::new(&info.details);
    let mut stager = Stager::new(jobs);

    for message in frames.iter() {
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

/// Joins every worker and gathers the grain they measured.
///
/// Logs each error or panic and returns the first.
fn join_workers(handles: Vec<WorkerJoin>) -> Result<Vec<SceneGrain>, anyhow::Error> {
    let mut first_error = None;
    let mut scenes = Vec::new();

    for (worker_id, handle) in handles.into_iter().enumerate() {
        let result = handle
            .join()
            .map_err(|panic| anyhow::anyhow!("worker panicked: {panic:?}"))
            .and_then(|result| result);

        let err = match result {
            Ok(worker_scenes) => {
                scenes.extend(worker_scenes);
                continue;
            },
            Err(err) => err,
        };

        tracing::error!(worker_id, error = %err, "worker failed");

        if first_error.is_none() {
            first_error = Some(err);
        }
    }

    first_error.map_or(Ok(scenes), Err)
}

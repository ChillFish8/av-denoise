use std::collections::VecDeque;
use std::thread;

use av_denoise::{FrameLayout, PlanarDenoiser, PlaneOptions, Planes, SceneGrain, WarmUp, push_needs_retry};

use super::coordinator::OutputMsg;
use super::stage::SceneJob;
use crate::warm_start::{create_denoiser, finish_warm_up};

pub type WorkerJoin = thread::JoinHandle<Result<Vec<SceneGrain>, anyhow::Error>>;

/// Spawns `workers` worker threads over one shared scene queue.
///
/// The queue is a rendezvous, so a scene is only offered when a worker is free and at most
/// `workers` scenes are ever in flight. Returns the queue's sender, the join handles and the
/// shared output channel the workers emit denoised frames on.
pub fn spawn_workers(
    opts: &PlaneOptions,
    layout: FrameLayout,
    workers: usize,
) -> (
    crossbeam_channel::Sender<SceneJob>,
    Vec<WorkerJoin>,
    crossbeam_channel::Receiver<OutputMsg>,
) {
    let (job_tx, job_rx) = crossbeam_channel::bounded::<SceneJob>(0);
    let (output_tx, output_rx) = crossbeam_channel::unbounded::<OutputMsg>();
    let mut worker_handles: Vec<WorkerJoin> = Vec::with_capacity(workers);

    for worker_id in 0..workers {
        let opts = opts.clone();
        let output_tx = output_tx.clone();
        let job_rx = job_rx.clone();

        let handle = thread::spawn(move || run_worker(worker_id, opts, layout, job_rx, output_tx));
        worker_handles.push(handle);
    }

    // Dropping the original sender lets the channel close once every worker's clone is gone.
    drop(output_tx);

    (job_tx, worker_handles, output_rx)
}

/// Denoises every scene this worker claims and returns the grain it measured.
pub fn run_worker(
    worker_id: usize,
    opts: PlaneOptions,
    layout: FrameLayout,
    jobs: crossbeam_channel::Receiver<SceneJob>,
    output_tx: crossbeam_channel::Sender<OutputMsg>,
) -> Result<Vec<SceneGrain>, anyhow::Error> {
    let mut scenes = Vec::new();
    let mut denoiser_slot: Option<PlanarDenoiser> = None;

    // Held until the denoiser's first output frame proves its kernels are compiled and cached.
    let mut warm_up: Option<WarmUp> = None;

    while let Ok(job) = jobs.recv() {
        // Built on the first claimed scene, so a worker that never claims one never compiles.
        if denoiser_slot.is_none() {
            let (denoiser, place) = create_denoiser(&opts, layout)?;
            denoiser_slot = Some(denoiser);
            warm_up = place;
        }

        let denoiser = denoiser_slot
            .as_mut()
            .expect("denoiser exists after the check above");

        tracing::debug!(worker_id, scene_idx = job.scene_idx, "worker started scene");

        // Indices of pushed but not yet emitted frames, in push order.
        let mut pending: VecDeque<u64> = VecDeque::new();
        let mut first_frame = None;

        // Nothing is received straight after a push, because that would clamp the 2-deep pending
        // pipeline to depth 1 and put the GPU readback in the next push's critical path.
        // `push_with_drain` drains on QueueFull and `flush_worker` drains the tail instead.
        for frame in job.frames {
            first_frame.get_or_insert(frame.global_idx);
            push_with_drain(
                denoiser,
                &mut warm_up,
                &mut pending,
                frame.global_idx,
                &frame.planes,
                &output_tx,
            )?;
        }

        // The denoiser is reused across scenes, so flushing here stops a temporal window
        // spanning two of them.
        flush_worker(denoiser, &mut warm_up, &mut pending, &output_tx)?;

        let chunks = denoiser.drain_grain_chunks()?;
        if let Some(first_frame) = first_frame
            && !chunks.is_empty()
        {
            scenes.push(SceneGrain { first_frame, chunks });
        }
    }

    Ok(scenes)
}

/// Pushes one frame, emitting the oldest pending output and retrying when the queue is full.
pub fn push_with_drain(
    denoiser: &mut PlanarDenoiser,
    warm_up: &mut Option<WarmUp>,
    pending: &mut VecDeque<u64>,
    global_idx: u64,
    planes: &Planes,
    output_tx: &crossbeam_channel::Sender<OutputMsg>,
) -> Result<(), anyhow::Error> {
    pending.push_back(global_idx);

    let push_result = denoiser.push(planes);

    if push_needs_retry(push_result)? {
        if let Some(denoised) = denoiser.recv()? {
            let oldest_idx = pending
                .pop_front()
                .expect("pending has at least one entry on QueueFull recv");
            send_output(output_tx, oldest_idx, denoised)?;
            finish_warm_up(warm_up);
        }

        denoiser.push(planes)?;
    }

    Ok(())
}

pub fn send_output(
    output_tx: &crossbeam_channel::Sender<OutputMsg>,
    global_idx: u64,
    planes: Planes,
) -> Result<(), anyhow::Error> {
    output_tx
        .send(OutputMsg { global_idx, planes })
        .map_err(|_| anyhow::anyhow!("coordinator disconnected"))
}

/// Flushes the denoiser, emitting every remaining frame against its pending index.
pub fn flush_worker(
    denoiser: &mut PlanarDenoiser,
    warm_up: &mut Option<WarmUp>,
    pending: &mut VecDeque<u64>,
    output_tx: &crossbeam_channel::Sender<OutputMsg>,
) -> Result<(), anyhow::Error> {
    let mut disconnected = false;

    denoiser.flush(|denoised| {
        if disconnected {
            return;
        }

        if let Some(global_idx) = pending.pop_front() {
            let message = OutputMsg {
                global_idx,
                planes: denoised,
            };
            let did_send = output_tx.send(message).is_ok();

            if did_send {
                finish_warm_up(warm_up);
            } else {
                disconnected = true;
            }
        } else {
            tracing::warn!("worker emitted flushed frame with no pending global index");
        }
    })?;

    if disconnected {
        anyhow::bail!("coordinator disconnected while flushing worker output");
    }

    Ok(())
}

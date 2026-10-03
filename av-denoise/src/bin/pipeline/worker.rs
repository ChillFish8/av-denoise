use std::thread;

use av_denoise::{FrameLayout, PlanarDenoiser, PlaneOptions, Planes, SceneGrain, WarmUp, push_needs_retry};

use super::coordinator::OutputMsg;
use super::stage::SceneJob;
use crate::warm_start::{create_denoiser, finish_warm_up};

pub type WorkerJoin = thread::JoinHandle<Result<Vec<SceneGrain>, anyhow::Error>>;

/// Spawns `workers` worker threads over one shared scene queue.
///
/// The queue is a rendezvous, so a scene is only offered when a worker is
/// free and at most `workers` scenes are ever in flight.
///
/// Returns the queue's sender, their join handles, and the shared output
/// channel they emit denoised frames on.
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
    let (out_tx, out_rx) = crossbeam_channel::unbounded::<OutputMsg>();
    let mut worker_handles: Vec<WorkerJoin> = Vec::with_capacity(workers);

    for worker_id in 0..workers {
        let opts = opts.clone();
        let out_tx = out_tx.clone();
        let job_rx = job_rx.clone();

        worker_handles.push(thread::spawn(move || {
            run_worker(worker_id, opts, layout, job_rx, out_tx)
        }));
    }

    // Drop the original sender so the channel closes once every worker
    // clone has terminated.
    drop(out_tx);

    (job_tx, worker_handles, out_rx)
}

pub fn run_worker(
    worker_id: usize,
    opts: PlaneOptions,
    layout: FrameLayout,
    jobs: crossbeam_channel::Receiver<SceneJob>,
    tx: crossbeam_channel::Sender<OutputMsg>,
) -> Result<Vec<SceneGrain>, anyhow::Error> {
    let mut scenes = Vec::new();
    let mut denoiser_slot: Option<PlanarDenoiser> = None;
    // The cold-cache queue place this worker's denoiser holds, until its
    // first output frame proves the kernels are compiled and cached.
    let mut warm_up: Option<WarmUp> = None;

    while let Ok(job) = jobs.recv() {
        // Built on the first claimed scene, so a worker that never claims
        // one never compiles.
        if denoiser_slot.is_none() {
            let (denoiser, place) = create_denoiser(&opts, layout)?;
            denoiser_slot = Some(denoiser);
            warm_up = place;
        }

        let denoiser = denoiser_slot
            .as_mut()
            .expect("denoiser exists after the check above");

        tracing::debug!(worker_id, scene_idx = job.scene_idx, "worker started scene");

        // Indices of pushed-but-not-yet-emitted frames, in push order.
        let mut pending: std::collections::VecDeque<u64> = Default::default();
        let mut first_frame = None;

        // Nothing is received straight after the push.
        // `push_with_drain` handles backpressure through QueueFull
        // when the 2-deep pending pipeline fills, and `flush_worker`
        // drains the tail below. Receiving after every push would clamp
        // the pipeline back to depth 1 and put the GPU readback in the
        // critical path of the next push.
        for frame in job.frames {
            first_frame.get_or_insert(frame.global_idx);
            push_with_drain(
                denoiser,
                &mut warm_up,
                &mut pending,
                frame.global_idx,
                &frame.planes,
                &tx,
            )?;
        }

        // Reuse the PlanarDenoiser across scenes. Flushing here ensures
        // no temporal window spans two of them.
        flush_worker(denoiser, &mut warm_up, &mut pending, &tx)?;

        let chunks = denoiser.drain_grain_chunks()?;
        if let Some(first_frame) = first_frame
            && !chunks.is_empty()
        {
            scenes.push(SceneGrain { first_frame, chunks });
        }
    }

    Ok(scenes)
}

/// Push one frame, draining any pending output first if the queue is full.
pub fn push_with_drain(
    denoiser: &mut PlanarDenoiser,
    warm_up: &mut Option<WarmUp>,
    pending: &mut std::collections::VecDeque<u64>,
    global_idx: u64,
    planes: &Planes,
    tx: &crossbeam_channel::Sender<OutputMsg>,
) -> Result<(), anyhow::Error> {
    pending.push_back(global_idx);

    if push_needs_retry(denoiser.push(planes))? {
        if let Some(out) = denoiser.recv()? {
            let oldest_idx = pending
                .pop_front()
                .expect("pending has at least one entry on QueueFull recv");
            send_output(tx, oldest_idx, out)?;
            finish_warm_up(warm_up);
        }

        denoiser.push(planes)?;
    }

    Ok(())
}

pub fn send_output(
    tx: &crossbeam_channel::Sender<OutputMsg>,
    global_idx: u64,
    planes: Planes,
) -> Result<(), anyhow::Error> {
    tx.send(OutputMsg { global_idx, planes })
        .map_err(|_| anyhow::anyhow!("coordinator disconnected"))
}

pub fn flush_worker(
    denoiser: &mut PlanarDenoiser,
    warm_up: &mut Option<WarmUp>,
    pending: &mut std::collections::VecDeque<u64>,
    tx: &crossbeam_channel::Sender<OutputMsg>,
) -> Result<(), anyhow::Error> {
    let mut disconnected = false;

    denoiser.flush(|out| {
        if disconnected {
            return;
        }

        if let Some(global_idx) = pending.pop_front() {
            let msg = OutputMsg {
                global_idx,
                planes: out,
            };
            let did_send = tx.send(msg).is_ok();
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

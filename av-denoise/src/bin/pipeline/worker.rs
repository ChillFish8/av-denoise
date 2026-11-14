use std::collections::VecDeque;
use std::thread;
use std::time::{Duration, Instant};

use av_denoise::{FrameLayout, PlanarDenoiser, PlaneOptions, Planes, SceneGrain, WarmUp, push_needs_retry};

use super::coordinator::OutputMsg;
use super::report::{DEVICE_SAMPLE_EVERY, WorkerTiming};
use super::stage::SceneJob;
use crate::warm_start::{create_denoiser, finish_warm_up};

pub type WorkerJoin = thread::JoinHandle<Result<WorkerTiming, anyhow::Error>>;

/// How each worker builds and measures its denoiser.
#[derive(Debug, Clone, Copy)]
pub struct WorkerSettings {
    /// Whether each worker samples its denoiser's device time.
    pub device_timing: bool,
    /// Whether each worker builds its denoiser on a wgpu device of its own.
    pub own_device: bool,
}

impl WorkerSettings {
    /// The settings for a run of `workers` workers.
    ///
    /// Workers open devices of their own only when there are several and `shared_device` is unset.
    pub fn for_run(workers: usize, timing_report: bool, shared_device: bool) -> Self {
        Self {
            device_timing: timing_report,
            own_device: workers > 1 && !shared_device,
        }
    }
}

/// Spawns `workers` worker threads over one shared scene queue.
///
/// The queue is a rendezvous, so a scene is only offered when a worker is free and at most
/// `workers` scenes are ever in flight. Each worker sends a scene's grain on `grain` once the
/// scene is done. Returns the queue's sender, the join handles and the shared output channel the
/// workers emit denoised frames on.
pub fn spawn_workers(
    opts: &PlaneOptions,
    layout: FrameLayout,
    workers: usize,
    settings: WorkerSettings,
    grain: Option<&crossbeam_channel::Sender<SceneGrain>>,
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
        let channels = WorkerChannels {
            jobs: job_rx,
            output: output_tx,
            grain: grain.cloned(),
        };

        let handle = thread::spawn(move || run_worker(worker_id, opts, layout, channels, settings));
        worker_handles.push(handle);
    }

    // Dropping the original sender lets the channel close once every worker's clone is gone.
    drop(output_tx);

    (job_tx, worker_handles, output_rx)
}

/// The channels one worker reads scenes from and writes results to.
pub struct WorkerChannels {
    /// The shared scene queue.
    pub jobs: crossbeam_channel::Receiver<SceneJob>,
    /// Where denoised frames go.
    pub output: crossbeam_channel::Sender<OutputMsg>,
    /// Where each scene's grain goes, `None` when the run keeps no grain.
    pub grain: Option<crossbeam_channel::Sender<SceneGrain>>,
}

/// Denoises every scene this worker claims, sending each scene's grain as the scene finishes.
///
/// Returns where the worker's time went.
pub fn run_worker(
    worker_id: usize,
    opts: PlaneOptions,
    layout: FrameLayout,
    channels: WorkerChannels,
    settings: WorkerSettings,
) -> Result<WorkerTiming, anyhow::Error> {
    let WorkerChannels {
        jobs,
        output: output_tx,
        grain,
    } = channels;

    let started = Instant::now();
    let mut setup = Duration::ZERO;
    let mut idle = Duration::ZERO;

    let mut denoiser_slot: Option<PlanarDenoiser> = None;

    // Held until the denoiser's first output frame proves its kernels are compiled and cached.
    let mut warm_up: Option<WarmUp> = None;

    loop {
        let waiting = Instant::now();
        let next = jobs.recv();
        idle += waiting.elapsed();

        let Ok(job) = next else {
            break;
        };

        // Built on the first claimed scene, so a worker that never claims one never compiles.
        if denoiser_slot.is_none() {
            let building = Instant::now();
            let (mut denoiser, place) = create_denoiser(&opts, layout, settings.own_device)?;
            if settings.device_timing {
                denoiser.sample_device_time(DEVICE_SAMPLE_EVERY);
            }

            setup += building.elapsed();
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

        let drained = denoiser.drain_grain_chunks()?;
        let measured = !drained.luma.is_empty() || !drained.chroma.is_empty();
        if let (Some(first_frame), Some(grain)) = (first_frame, grain.as_ref())
            && measured
        {
            let scene = SceneGrain {
                first_frame,
                chunks: drained.luma,
                chroma_chunks: drained.chroma,
            };
            grain
                .send(scene)
                .map_err(|_| anyhow::anyhow!("grain writer disconnected"))?;
        }
    }

    let planes = denoiser_slot
        .as_mut()
        .map(PlanarDenoiser::timing)
        .unwrap_or_default();
    let accelerator = denoiser_slot.as_ref().map(PlanarDenoiser::selected_accelerator);
    let timing = WorkerTiming {
        wall: started.elapsed(),
        setup,
        idle,
        planes,
        accelerator,
    };

    Ok(timing)
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

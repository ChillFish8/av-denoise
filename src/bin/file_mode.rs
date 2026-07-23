use std::collections::BTreeMap;
use std::io::{IsTerminal, stdout};
use std::path::Path;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread;

use av_decoders::{Decoder, Rational32};
use av_scenechange::{DetectionOptions, detect_scene_changes};
use y4m::Frame as Y4mFrame;

use crate::ingest::{CliOptions, FrameLayout, Planes, Subsampling, WorkerDenoiser, subsampling_to_y4m};
use crate::progress::{progress_visible, scene_progress_bar};

const FRAME_CHANNEL_DEPTH: usize = 8;
const OUTPUT_CHANNEL_DEPTH: usize = 32;

/// Scene boundaries + the video metadata needed to build the output y4m header.
struct SceneLayout {
    layout: FrameLayout,
    framerate: Rational32,
    total_frames: usize,
    /// `scene_starts[i]` is the inclusive start frame of scene `i`. The
    /// final entry is `total_frames` so scene `i` covers
    /// `scene_starts[i] .. scene_starts[i + 1]`.
    scene_starts: Vec<usize>,
}

impl SceneLayout {
    fn scene_count(&self) -> usize {
        self.scene_starts.len() - 1
    }
}

pub fn run_file(opts: &CliOptions, input: &Path, workers: usize) -> Result<(), anyhow::Error> {
    if workers == 0 {
        anyhow::bail!("--workers must be at least 1");
    }

    let scenes = detect_scenes(input, opts.no_progress)?;

    tracing::info!(
        scene_count = scenes.scene_count(),
        total_frames = scenes.total_frames,
        workers,
        "scene detection complete",
    );

    encode_scenes(opts, input, &scenes, workers)
}

/// Open the input, read its layout, and run `av_scenechange` to produce a
/// list of scene boundaries. Returns once the detector pass has finished
/// and the temporary decoder it used has been dropped.
fn detect_scenes(input: &Path, no_progress: bool) -> Result<SceneLayout, anyhow::Error> {
    let mut decoder = Decoder::from_file(input)?;
    let details = *decoder.get_video_details();

    if details.bit_depth != 8 {
        anyhow::bail!("only 8-bit sources are supported (got {}-bit)", details.bit_depth);
    }

    let layout = FrameLayout {
        width: details.width as u32,
        height: details.height as u32,
        subsampling: subsampling_from_av_decoders(details.chroma_sampling)?,
    };

    tracing::info!(
        width = layout.width,
        height = layout.height,
        subsampling = ?layout.subsampling,
        total_frames = details.total_frames,
        "running scene detection",
    );

    let visible = progress_visible(no_progress, std::io::stderr().is_terminal());
    let pb = scene_progress_bar(details.total_frames, visible);
    let on_progress = |frames_analyzed: usize, _keyframe_count: usize| {
        pb.set_position(frames_analyzed as u64);
    };

    let detect_opts = DetectionOptions::default();
    let detection = detect_scene_changes::<u8>(&mut decoder, detect_opts, None, Some(&on_progress))?;

    pb.finish_and_clear();

    drop(decoder);

    let mut scene_starts = detection.scene_changes;

    if scene_starts.is_empty() || scene_starts[0] != 0 {
        scene_starts.insert(0, 0);
    }

    let total_frames = detection.frame_count;
    scene_starts.push(total_frames);

    Ok(SceneLayout {
        layout,
        framerate: details.frame_rate,
        total_frames,
        scene_starts,
    })
}

/// Spin up the worker pool + coordinator, re-open the decoder for actual
/// frame reading, and drive the dispatch loop until EOF. Blocks until
/// every worker and the coordinator have completed.
fn encode_scenes(
    opts: &CliOptions,
    input: &Path,
    scenes: &SceneLayout,
    workers: usize,
) -> Result<(), anyhow::Error> {
    let (worker_txs, worker_handles, out_rx) = spawn_workers(opts, scenes.layout, workers);
    let coordinator = spawn_coordinator(scenes.layout, scenes.framerate, out_rx, scenes.total_frames);

    dispatch_frames(input, scenes, &worker_txs)?;

    for tx in &worker_txs {
        let _ = tx.send(WorkerMsg::Eof);
    }

    drop(worker_txs);

    for h in worker_handles {
        h.join()
            .map_err(|e| anyhow::anyhow!("worker panicked: {e:?}"))??;
    }

    coordinator
        .join()
        .map_err(|e| anyhow::anyhow!("coordinator panicked: {e:?}"))??;

    Ok(())
}

type WorkerJoin = thread::JoinHandle<Result<(), anyhow::Error>>;

/// Spawn `workers` worker threads, returning their input channels, join
/// handles, and the shared output channel they emit denoised frames on.
fn spawn_workers(
    opts: &CliOptions,
    layout: FrameLayout,
    workers: usize,
) -> (Vec<SyncSender<WorkerMsg>>, Vec<WorkerJoin>, Receiver<OutputMsg>) {
    let mut worker_txs: Vec<SyncSender<WorkerMsg>> = Vec::with_capacity(workers);
    let (out_tx, out_rx) = sync_channel::<OutputMsg>(OUTPUT_CHANNEL_DEPTH);
    let mut worker_handles: Vec<WorkerJoin> = Vec::with_capacity(workers);

    for worker_id in 0..workers {
        let (frame_tx, frame_rx) = sync_channel::<WorkerMsg>(FRAME_CHANNEL_DEPTH);
        let opts = opts.clone();
        let out_tx = out_tx.clone();

        worker_txs.push(frame_tx);
        worker_handles.push(thread::spawn(move || {
            run_worker(worker_id, opts, layout, frame_rx, out_tx)
        }));
    }

    // Drop the original sender so the channel closes once every worker
    // clone has terminated.
    drop(out_tx);

    (worker_txs, worker_handles, out_rx)
}

fn spawn_coordinator(
    layout: FrameLayout,
    framerate: Rational32,
    rx: Receiver<OutputMsg>,
    total_frames: usize,
) -> thread::JoinHandle<Result<(), anyhow::Error>> {
    thread::spawn(move || run_coordinator(layout, framerate, rx, total_frames))
}

/// Sequential decode loop: read every frame, route it to worker
/// `scene_idx % N`. Blocks on `send` when a worker's channel fills,
/// which provides natural backpressure.
fn dispatch_frames(
    input: &Path,
    scenes: &SceneLayout,
    worker_txs: &[SyncSender<WorkerMsg>],
) -> Result<(), anyhow::Error> {
    let mut decoder = Decoder::from_file(input)?;
    let workers = worker_txs.len();

    let mut scene_idx = 0usize;
    let mut next_boundary = scenes.scene_starts[1];

    for g in 0..scenes.total_frames {
        while g >= next_boundary && scene_idx + 1 < scenes.scene_count() {
            scene_idx += 1;
            next_boundary = scenes.scene_starts[scene_idx + 1];
        }

        let frame = decoder.read_video_frame::<u8>()?;
        let planes = planes_from_v_frame(&frame, scenes.layout);
        let target = scene_idx % workers;

        worker_txs[target]
            .send(WorkerMsg::Frame {
                global_idx: g as u64,
                scene_idx: scene_idx as u32,
                planes,
            })
            .map_err(|_| anyhow::anyhow!("worker {target} disconnected"))?;
    }

    Ok(())
}

enum WorkerMsg {
    Frame {
        global_idx: u64,
        scene_idx: u32,
        planes: Planes,
    },
    Eof,
}

struct OutputMsg {
    global_idx: u64,
    planes: Planes,
}

fn run_worker(
    worker_id: usize,
    opts: CliOptions,
    layout: FrameLayout,
    rx: Receiver<WorkerMsg>,
    tx: SyncSender<OutputMsg>,
) -> Result<(), anyhow::Error> {
    let mut current_scene: Option<u32> = None;
    let mut wd: Option<WorkerDenoiser> = None;

    // Indices of pushed-but-not-yet-emitted frames, in push order.
    let mut pending: std::collections::VecDeque<u64> = Default::default();

    loop {
        match rx.recv() {
            Ok(WorkerMsg::Frame {
                global_idx,
                scene_idx,
                planes,
            }) => {
                if current_scene != Some(scene_idx) {
                    // Reuse the existing WorkerDenoiser across scenes, flushing the worker
                    // will ensure there is no cross-scene blending during temporal workloads.
                    if let Some(prev) = wd.as_mut() {
                        flush_worker(prev, &mut pending, &tx)?;
                    } else {
                        wd = Some(WorkerDenoiser::create(&opts, layout)?);
                    }

                    current_scene = Some(scene_idx);
                    pending.clear();

                    tracing::debug!(worker_id, scene_idx, "worker started scene");
                }

                let denoiser = wd.as_mut().expect("denoiser exists after new-scene init");

                // No post-push recv: push_with_drain handles backpressure via
                // QueueFull when the 2-deep pending pipeline fills, and
                // flush_worker drains the tail at the scene boundary. Pulling
                // a recv after every push would clamp the pipeline back to
                // depth 1 and serialise the GPU readback into the critical
                // path of the next push.
                push_with_drain(denoiser, &mut pending, global_idx, &planes, &tx)?;
            },
            Ok(WorkerMsg::Eof) | Err(_) => {
                if let Some(mut prev) = wd.take() {
                    flush_worker(&mut prev, &mut pending, &tx)?;
                }

                break;
            },
        }
    }

    Ok(())
}

/// Push one frame, draining any pending output first if the queue is full.
fn push_with_drain(
    denoiser: &mut WorkerDenoiser,
    pending: &mut std::collections::VecDeque<u64>,
    global_idx: u64,
    planes: &Planes,
    tx: &SyncSender<OutputMsg>,
) -> Result<(), anyhow::Error> {
    pending.push_back(global_idx);

    if let Err(av_denoise::DenoiserError::QueueFull) = denoiser.push(planes) {
        if let Some(out) = denoiser.recv()? {
            let g = pending
                .pop_front()
                .expect("pending has at least one entry on QueueFull recv");
            send_output(tx, g, out)?;
        }

        denoiser.push(planes)?;
    }

    Ok(())
}

fn send_output(tx: &SyncSender<OutputMsg>, global_idx: u64, planes: Planes) -> Result<(), anyhow::Error> {
    tx.send(OutputMsg { global_idx, planes })
        .map_err(|_| anyhow::anyhow!("coordinator disconnected"))
}

fn flush_worker(
    wd: &mut WorkerDenoiser,
    pending: &mut std::collections::VecDeque<u64>,
    tx: &SyncSender<OutputMsg>,
) -> Result<(), anyhow::Error> {
    wd.flush(|out| {
        if let Some(g) = pending.pop_front() {
            let _ = tx.send(OutputMsg {
                global_idx: g,
                planes: out,
            });
        } else {
            tracing::warn!("worker emitted flushed frame with no pending global index");
        }
    })?;

    Ok(())
}

fn run_coordinator(
    layout: FrameLayout,
    framerate: Rational32,
    rx: Receiver<OutputMsg>,
    total_frames: usize,
) -> Result<(), anyhow::Error> {
    let stdout = stdout();
    let lock = stdout.lock();

    // No `XCOLORRANGE=` tag is emitted here. `av_decoders::VideoDetails`
    // doesn't surface the source's color range for any of its backends
    // (ffms2 included), so there's nothing to forward.
    let mut encoder = y4m::encode(
        layout.width as usize,
        layout.height as usize,
        y4m::Ratio::new((*framerate.numer()) as usize, (*framerate.denom()) as usize),
    )
    .with_colorspace(subsampling_to_y4m(layout.subsampling))
    .write_header(lock)?;

    let mut pending: BTreeMap<u64, Planes> = BTreeMap::new();
    let mut next_emit: u64 = 0;
    let total = total_frames as u64;

    while next_emit < total {
        let msg = match rx.recv() {
            Ok(m) => m,
            Err(_) => break,
        };

        pending.insert(msg.global_idx, msg.planes);

        while let Some(planes) = pending.remove(&next_emit) {
            let frame = Y4mFrame::new([&planes.y, &planes.u, &planes.v], None);
            encoder.write_frame(&frame)?;
            next_emit += 1;
        }
    }

    Ok(())
}

fn planes_from_v_frame(frame: &v_frame::frame::Frame<u8>, layout: FrameLayout) -> Planes {
    let chroma_pixels = layout.chroma_pixels();

    Planes {
        y: collect_plane(&frame.y_plane),
        u: frame
            .u_plane
            .as_ref()
            .map(collect_plane)
            .unwrap_or_else(|| vec![128u8; chroma_pixels]),
        v: frame
            .v_plane
            .as_ref()
            .map(collect_plane)
            .unwrap_or_else(|| vec![128u8; chroma_pixels]),
    }
}

fn collect_plane(plane: &v_frame::plane::Plane<u8>) -> Vec<u8> {
    let width = plane.width().get();
    let height = plane.height().get();
    let mut out = Vec::with_capacity(width * height);

    for y in 0..height {
        if let Some(row) = plane.row(y) {
            out.extend_from_slice(&row[..width]);
        }
    }

    out
}

fn subsampling_from_av_decoders(
    cs: v_frame::chroma::ChromaSubsampling,
) -> Result<Subsampling, anyhow::Error> {
    use v_frame::chroma::ChromaSubsampling;

    match cs {
        ChromaSubsampling::Yuv420 => Ok(Subsampling::Yuv420),
        ChromaSubsampling::Yuv422 => Ok(Subsampling::Yuv422),
        ChromaSubsampling::Yuv444 => Ok(Subsampling::Yuv444),
        other => anyhow::bail!("unsupported chroma subsampling {other:?}; need 4:2:0, 4:2:2, or 4:4:4"),
    }
}

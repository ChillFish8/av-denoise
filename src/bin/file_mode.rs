use std::collections::BTreeMap;
use std::io::{IsTerminal, stdout};
use std::path::Path;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread;
use std::time::Duration;

use av_decoders::{Decoder, Rational32};
use av_denoise::Depth;
use av_scenechange::{DetectionOptions, detect_scene_changes};
use indicatif::ProgressBar;
use y4m::Frame as Y4mFrame;

use crate::ingest::{
    CliOptions,
    FrameLayout,
    Planes,
    Subsampling,
    WorkerDenoiser,
    push_needs_retry,
    subsampling_to_y4m,
};
use crate::progress::{self, denoise_bar_visible, denoise_progress_bar, scene_progress_bar};

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

    // Scene detection finishes before a single frame is written, so its
    // bar has the terminal to itself and needs no opt-in. The denoising
    // bar shares the terminal with whatever consumes our output, so it
    // waits for --progress.
    let is_terminal = std::io::stderr().is_terminal();
    let scenes = detect_scenes(input, is_terminal)?;

    tracing::info!(
        scene_count = scenes.scene_count(),
        total_frames = scenes.total_frames,
        workers,
        "scene detection complete",
    );

    encode_scenes(
        opts,
        input,
        &scenes,
        workers,
        denoise_bar_visible(opts.progress, is_terminal),
    )
}

/// Open the input, read its layout, and run `av_scenechange` to produce a
/// list of scene boundaries. Returns once the detector pass has finished
/// and the temporary decoder it used has been dropped.
fn detect_scenes(input: &Path, visible: bool) -> Result<SceneLayout, anyhow::Error> {
    let mut decoder = Decoder::from_file(input)?;
    let details = *decoder.get_video_details();

    let depth = Depth::from_bits(details.bit_depth)?;

    let layout = FrameLayout {
        width: details.width as u32,
        height: details.height as u32,
        subsampling: subsampling_from_av_decoders(details.chroma_sampling)?,
        depth,
    };

    tracing::info!(
        width = layout.width,
        height = layout.height,
        subsampling = ?layout.subsampling,
        depth = ?layout.depth,
        total_frames = details.total_frames,
        "running scene detection",
    );

    let pb = scene_progress_bar(details.total_frames, visible);
    let on_progress = |frames_analyzed: usize, _keyframe_count: usize| {
        pb.set_position(frames_analyzed as u64);
    };

    let detect_opts = DetectionOptions::default();
    let detection = match depth {
        Depth::Eight => detect_scene_changes::<u8>(&mut decoder, detect_opts, None, Some(&on_progress))?,
        Depth::Ten | Depth::Twelve => {
            detect_scene_changes::<u16>(&mut decoder, detect_opts, None, Some(&on_progress))?
        },
    };

    progress::finish(&pb);

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
    visible: bool,
) -> Result<(), anyhow::Error> {
    let (worker_txs, worker_handles, out_rx) = spawn_workers(opts, scenes.layout, workers);
    let coordinator = spawn_coordinator(
        scenes.layout,
        scenes.framerate,
        out_rx,
        scenes.total_frames,
        visible,
    );

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
    visible: bool,
) -> thread::JoinHandle<Result<(), anyhow::Error>> {
    thread::spawn(move || run_coordinator(layout, framerate, rx, total_frames, visible))
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

        let planes = match scenes.layout.depth {
            Depth::Eight => {
                let frame = decoder.read_video_frame::<u8>()?;
                planes_from_v_frame_u8(&frame, scenes.layout)
            },
            Depth::Ten | Depth::Twelve => {
                let frame = decoder.read_video_frame::<u16>()?;
                planes_from_v_frame_u16(&frame, scenes.layout)
            },
        };
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

    if push_needs_retry(denoiser.push(planes))? {
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
    let mut disconnected = false;

    wd.flush(|out| {
        if disconnected {
            return;
        }

        if let Some(g) = pending.pop_front() {
            let msg = OutputMsg {
                global_idx: g,
                planes: out,
            };
            let did_send = tx.send(msg).is_ok();
            if !did_send {
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

fn run_coordinator(
    layout: FrameLayout,
    framerate: Rational32,
    rx: Receiver<OutputMsg>,
    total_frames: usize,
    visible: bool,
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
    .with_colorspace(subsampling_to_y4m(layout.subsampling, layout.depth))
    .write_header(lock)?;

    // Frames written to the output, which lags the frames read by the
    // depth of the worker pipelines. Emitted frames are the honest
    // measure of progress: the count stalls whenever whatever consumes
    // our stdout stops reading.
    let pb = denoise_progress_bar(total_frames, visible);

    // The first frame only lands once a worker has compiled its
    // kernels, which takes seconds. A steady tick draws the bar right
    // away and keeps its elapsed time moving until then.
    pb.enable_steady_tick(Duration::from_millis(250));

    let result = emit_frames(&mut encoder, &rx, total_frames as u64, &pb);

    progress::finish(&pb);

    result
}

/// Reorders worker output by frame index and writes it out, updating
/// `pb` as frames are emitted. Returns once every frame has been
/// written. Errors if the workers all disconnect before `total` frames
/// have landed, naming how many were written and how many were expected.
fn emit_frames<W: std::io::Write>(
    encoder: &mut y4m::Encoder<W>,
    rx: &Receiver<OutputMsg>,
    total: u64,
    pb: &ProgressBar,
) -> Result<(), anyhow::Error> {
    let mut pending: BTreeMap<u64, Planes> = BTreeMap::new();
    let mut next_emit: u64 = 0;

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

        pb.set_position(next_emit);
    }

    if next_emit != total {
        anyhow::bail!(
            "wrote {next_emit} frames but expected {total}; every worker disconnected \
             before the stream finished (a frame index was likely lost)"
        );
    }

    Ok(())
}

fn planes_from_v_frame_u8(frame: &v_frame::frame::Frame<u8>, layout: FrameLayout) -> Planes {
    Planes {
        y: collect_plane_u8(&frame.y_plane),
        u: frame
            .u_plane
            .as_ref()
            .map(collect_plane_u8)
            .unwrap_or_else(|| layout.neutral_chroma_plane()),
        v: frame
            .v_plane
            .as_ref()
            .map(collect_plane_u8)
            .unwrap_or_else(|| layout.neutral_chroma_plane()),
    }
}

fn planes_from_v_frame_u16(frame: &v_frame::frame::Frame<u16>, layout: FrameLayout) -> Planes {
    Planes {
        y: collect_plane_u16(&frame.y_plane),
        u: frame
            .u_plane
            .as_ref()
            .map(collect_plane_u16)
            .unwrap_or_else(|| layout.neutral_chroma_plane()),
        v: frame
            .v_plane
            .as_ref()
            .map(collect_plane_u16)
            .unwrap_or_else(|| layout.neutral_chroma_plane()),
    }
}

fn collect_plane_u8(plane: &v_frame::plane::Plane<u8>) -> Vec<u8> {
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

fn collect_plane_u16(plane: &v_frame::plane::Plane<u16>) -> Vec<u8> {
    let width = plane.width().get();
    let height = plane.height().get();
    let mut out = Vec::with_capacity(width * height * 2);

    for y in 0..height {
        if let Some(row) = plane.row(y) {
            for &s in &row[..width] {
                out.extend_from_slice(&s.to_le_bytes());
            }
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

#[cfg(test)]
mod tests {
    use av_denoise::accelerate::Accelerator;
    use av_denoise::{Algorithm, DenoisingMode, Device, MotionCompensationMode};
    use indicatif::ProgressBar;

    use super::*;
    use crate::ingest::{BinaryChannelIntent, fill_plane};

    fn tiny_layout() -> FrameLayout {
        // 4:2:0 chroma at this size is 4x4, clearing the denoiser's 3x3
        // minimum frame dimension.
        FrameLayout {
            width: 8,
            height: 8,
            subsampling: Subsampling::Yuv420,
            depth: Depth::Eight,
        }
    }

    fn tiny_planes(layout: FrameLayout) -> Planes {
        Planes {
            y: fill_plane(layout.luma_pixels(), layout.depth.neutral_chroma(), layout.depth),
            u: layout.neutral_chroma_plane(),
            v: layout.neutral_chroma_plane(),
        }
    }

    #[test]
    fn emit_frames_errors_when_a_frame_index_is_lost() {
        let layout = tiny_layout();
        let (tx, rx) = sync_channel::<OutputMsg>(4);
        let planes = tiny_planes(layout);

        // Frame 1 is never sent (its index was lost somewhere upstream)
        // and every worker then disconnects. This used to make
        // `emit_frames` fall through to `Ok(())` with a truncated y4m.
        tx.send(OutputMsg {
            global_idx: 0,
            planes: planes.clone(),
        })
        .unwrap();
        tx.send(OutputMsg {
            global_idx: 2,
            planes,
        })
        .unwrap();
        drop(tx);

        let mut buf: Vec<u8> = Vec::new();
        let mut encoder = y4m::encode(
            layout.width as usize,
            layout.height as usize,
            y4m::Ratio::new(30, 1),
        )
        .with_colorspace(subsampling_to_y4m(layout.subsampling, layout.depth))
        .write_header(&mut buf)
        .expect("header write failed");

        let pb = ProgressBar::hidden();
        let err = emit_frames(&mut encoder, &rx, 3, &pb).expect_err("expected a lost-frame error");

        let msg = err.to_string();
        assert!(
            msg.contains('1') && msg.contains('3'),
            "error should name frames written (1) vs expected (3): {msg}"
        );
    }

    fn temporal_opts() -> CliOptions {
        CliOptions {
            accelerators: vec![Accelerator::Vulkan],
            device: Device::Default,
            intent: BinaryChannelIntent::LumaChroma,
            mode: DenoisingMode::Temporal { radius: 1 },
            prefilter: None,
            motion_compensation: MotionCompensationMode::None,
            algorithm: Algorithm::Nlmeans,
            nlm_tuning: None,
            luma_strength: None,
            chroma_strength: None,
            progress: false,
        }
    }

    /// A 10-bit v_frame plane serialises to little-endian wire bytes at
    /// twice the sample count.
    #[test]
    fn collect_plane_u16_writes_little_endian_bytes() {
        use std::num::{NonZeroU8, NonZeroUsize};

        use v_frame::chroma::ChromaSubsampling;
        use v_frame::frame::{Frame, FrameBuilder};

        let mut frame: Frame<u16> = FrameBuilder::new(
            NonZeroUsize::new(2).expect("width is non-zero"),
            NonZeroUsize::new(2).expect("height is non-zero"),
            ChromaSubsampling::Yuv420,
            NonZeroU8::new(10).expect("depth is non-zero"),
        )
        .build()
        .expect("a 2x2 10-bit frame builds");

        frame
            .y_plane
            .copy_from_slice(&[0u16, 1, 512, 1023])
            .expect("four samples fill a 2x2 plane");

        let bytes = collect_plane_u16(&frame.y_plane);

        assert_eq!(bytes.len(), 8, "4 samples at 2 bytes each");
        assert_eq!(
            bytes,
            vec![0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0xFF, 0x03],
            "samples must be little-endian"
        );
    }

    #[test]
    fn flush_worker_errors_when_coordinator_has_disconnected() {
        let layout = tiny_layout();
        let mut wd = WorkerDenoiser::create(&temporal_opts(), layout).expect("denoiser construction failed");
        let planes = tiny_planes(layout);

        // One push into a temporal window leaves a trailing tail that
        // `flush` will pad and emit.
        wd.push(&planes).expect("push failed");

        let mut pending: std::collections::VecDeque<u64> = std::collections::VecDeque::new();
        pending.push_back(0);

        let (tx, rx) = sync_channel::<OutputMsg>(4);
        drop(rx);

        let err = flush_worker(&mut wd, &mut pending, &tx)
            .expect_err("expected the coordinator disconnect to surface as an error");

        assert!(
            err.to_string().contains("disconnect"),
            "error should mention the coordinator disconnect: {err}"
        );
    }
}

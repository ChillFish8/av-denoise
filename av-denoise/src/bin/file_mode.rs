use std::collections::{BTreeMap, BTreeSet};
use std::io::{IsTerminal, stdout};
use std::path::Path;
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread;
use std::time::Duration;

use av_decoders::{Decoder, Rational32};
use av_denoise::{Depth, FrameLayout, PlanarDenoiser, PlaneOptions, Planes, Subsampling, push_needs_retry};
use av_scenechange::{DetectionOptions, detect_scene_changes};
use indicatif::ProgressBar;
use y4m::Frame as Y4mFrame;

use crate::cli::RunOptions;
use crate::frame_index;
use crate::progress::{self, denoise_bar_visible, denoise_progress_bar, scene_progress_bar};
use crate::y4m_format::subsampling_to_y4m;

/// Target ceiling for CPU-side frame buffers held in flight. Channel
/// depths shrink to stay under this when frames are large.
const FRAME_MEMORY_BUDGET_BYTES: usize = 1 << 30;

/// Channel depths used when frames are small enough to afford them.
const FRAME_CHANNEL_DEPTH_MAX: usize = 8;
const OUTPUT_CHANNEL_DEPTH_MAX: usize = 32;

/// Floors that keep the pipeline from starving no matter the frame size.
const FRAME_CHANNEL_DEPTH_MIN: usize = 2;
const OUTPUT_CHANNEL_DEPTH_MIN: usize = 4;

/// Channel depths for one run, with the frame counts they imply.
#[derive(Debug, Clone, Copy)]
struct ChannelBudget {
    frame_depth: usize,
    output_depth: usize,
    /// Frames held in the channels. This is what the budget scales, and
    /// it stays under [`FRAME_MEMORY_BUDGET_BYTES`] whenever the depth
    /// floors allow it.
    ceiling_frames: usize,
    /// Frames alive anywhere, including the ones each worker holds
    /// inside its GPU pipeline. Larger than `ceiling_frames`, and the
    /// figure the reorder high-water mark is judged against.
    peak_frames: usize,
}

/// Picks channel depths so the frames held in flight stay near
/// [`FRAME_MEMORY_BUDGET_BYTES`].
///
/// Small frames keep the maximum depths. Large frames scale both depths
/// down together, never below their floors.
fn channel_budget(layout: FrameLayout, workers: usize) -> ChannelBudget {
    let frame_bytes = layout.luma_bytes() + 2 * layout.chroma_bytes();
    let max_frames = workers * FRAME_CHANNEL_DEPTH_MAX + OUTPUT_CHANNEL_DEPTH_MAX;

    let affordable = FRAME_MEMORY_BUDGET_BYTES / frame_bytes.max(1);

    let (frame_depth, output_depth) = if max_frames <= affordable {
        (FRAME_CHANNEL_DEPTH_MAX, OUTPUT_CHANNEL_DEPTH_MAX)
    } else {
        let scale = affordable as f64 / max_frames as f64;
        let frame_depth =
            ((FRAME_CHANNEL_DEPTH_MAX as f64 * scale).floor() as usize).max(FRAME_CHANNEL_DEPTH_MIN);
        let output_depth =
            ((OUTPUT_CHANNEL_DEPTH_MAX as f64 * scale).floor() as usize).max(OUTPUT_CHANNEL_DEPTH_MIN);
        (frame_depth, output_depth)
    };

    // Each worker also holds frames the channels never see, being the
    // readbacks in flight inside its denoiser plus the one it is
    // pushing.
    //
    // Those count toward real memory, and toward how far the reorder
    // map can legitimately run ahead, so they belong in the peak even
    // though the budget only scales channel capacity.
    let per_worker_pipeline = av_denoise::MAX_PENDING + 1;

    ChannelBudget {
        frame_depth,
        output_depth,
        ceiling_frames: workers * frame_depth + output_depth,
        peak_frames: workers * (frame_depth + per_worker_pipeline) + output_depth,
    }
}

/// Scene boundaries plus the video metadata needed to build the output
/// y4m header.
struct SceneLayout {
    layout: FrameLayout,
    framerate: Rational32,
    /// Frames this run emits, being `raw_frames` less the phantom entries.
    total_frames: usize,
    /// Frames the decoder hands over, phantom entries included. Every
    /// one has to be read to keep the sequential decoder in step, even
    /// though only `total_frames` of them are emitted.
    raw_frames: usize,
    /// Decoder frame numbers that carry no picture of their own. See
    /// [`crate::frame_index`].
    phantom: BTreeSet<usize>,
    /// `scene_starts[i]` is the inclusive start frame of scene `i`, in  emitted frame numbers.
    /// The final entry is `total_frames` so scene `i` covers `scene_starts[i]..scene_starts[i + 1]`.
    scene_starts: Vec<usize>,
}

impl SceneLayout {
    fn scene_count(&self) -> usize {
        self.scene_starts.len() - 1
    }
}

pub fn run_file(opts: &RunOptions, input: &Path, workers: usize) -> Result<(), anyhow::Error> {
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
        &opts.planes,
        input,
        &scenes,
        workers,
        denoise_bar_visible(opts.progress, is_terminal),
    )
}

/// Opens the input, reads its layout, and runs `av_scenechange` to
/// produce a list of scene boundaries.
///
/// Returns once the detector pass has finished and the temporary decoder
/// it used has been dropped.
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

    // Read the index before decoding anything. This only inspects the
    // metadata ffms2 already built, so it costs nothing.
    let phantom = frame_index::read_index(&mut decoder)
        .map(|index| frame_index::phantom_indices(&index))
        .unwrap_or_default();

    if !phantom.is_empty() {
        tracing::info!(
            dropped = phantom.len(),
            "the decoder reports frames that carry no picture of their own, dropping them",
        );
    }

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

    // Detection ran over every frame the decoder offers, so both the count and the boundaries
    // are in decoder frame numbers. Both move into emitted frame numbers together.
    let raw_frames = detection.frame_count;

    // The index lists every entry the container holds, while detection counts
    // what the decoder handed over. A decode that stops early leaves entries
    // above the last frame read, and those match no frame this run sees.
    let phantom: BTreeSet<usize> = phantom.into_iter().take_while(|&raw| raw < raw_frames).collect();

    scene_starts.push(raw_frames);

    let scene_starts = frame_index::remap_scene_starts(&scene_starts, &phantom);
    let total_frames = raw_frames - phantom.len();

    if total_frames == 0 {
        anyhow::bail!("{} holds no decodable frames", input.display());
    }

    Ok(SceneLayout {
        layout,
        framerate: details.frame_rate,
        total_frames,
        raw_frames,
        phantom,
        scene_starts,
    })
}

/// Starts the worker pool and the coordinator, reopens the decoder for
/// frame reading, then drives the dispatch loop until EOF.
///
/// Blocks until every worker and the coordinator have finished.
fn encode_scenes(
    opts: &PlaneOptions,
    input: &Path,
    scenes: &SceneLayout,
    workers: usize,
    visible: bool,
) -> Result<(), anyhow::Error> {
    let budget = channel_budget(scenes.layout, workers);
    let frame_bytes = scenes.layout.luma_bytes() + 2 * scenes.layout.chroma_bytes();

    tracing::info!(
        frame_depth = budget.frame_depth,
        output_depth = budget.output_depth,
        ceiling_frames = budget.ceiling_frames,
        peak_frames = budget.peak_frames,
        peak_mib = (budget.peak_frames * frame_bytes) / (1 << 20),
        "frame buffer budget",
    );

    let (worker_txs, worker_handles, out_rx) = spawn_workers(opts, scenes.layout, workers, budget);
    let coordinator = spawn_coordinator(
        scenes.layout,
        scenes.framerate,
        out_rx,
        scenes.total_frames,
        visible,
        budget.peak_frames,
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

/// Spawns `workers` worker threads.
///
/// Returns their input channels, their join handles, and the shared
/// output channel they emit denoised frames on.
fn spawn_workers(
    opts: &PlaneOptions,
    layout: FrameLayout,
    workers: usize,
    budget: ChannelBudget,
) -> (Vec<SyncSender<WorkerMsg>>, Vec<WorkerJoin>, Receiver<OutputMsg>) {
    let mut worker_txs: Vec<SyncSender<WorkerMsg>> = Vec::with_capacity(workers);
    let (out_tx, out_rx) = sync_channel::<OutputMsg>(budget.output_depth);
    let mut worker_handles: Vec<WorkerJoin> = Vec::with_capacity(workers);

    for worker_id in 0..workers {
        let (frame_tx, frame_rx) = sync_channel::<WorkerMsg>(budget.frame_depth);
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
    peak_frames: usize,
) -> thread::JoinHandle<Result<(), anyhow::Error>> {
    thread::spawn(move || run_coordinator(layout, framerate, rx, total_frames, visible, peak_frames))
}

/// Reads every frame in order and routes it to worker `scene_idx % N`.
///
/// The send blocks when a worker's channel fills, which gives the
/// pipeline its backpressure.
fn dispatch_frames(
    input: &Path,
    scenes: &SceneLayout,
    worker_txs: &[SyncSender<WorkerMsg>],
) -> Result<(), anyhow::Error> {
    let mut decoder = Decoder::from_file(input)?;
    let workers = worker_txs.len();

    let mut scene_idx = 0usize;
    let mut next_boundary = scenes.scene_starts[1];
    let mut g = 0u64;

    for raw in 0..scenes.raw_frames {
        // Every frame is read, phantom or not. The decoder walks the file in order and has
        // no way to be told to skip one.
        let planes = match scenes.layout.depth {
            Depth::Eight => {
                let frame = decoder.read_video_frame::<u8>()?;
                planes_from_v_frame_u8(&frame, scenes.layout)?
            },
            Depth::Ten | Depth::Twelve => {
                let frame = decoder.read_video_frame::<u16>()?;
                planes_from_v_frame_u16(&frame, scenes.layout)?
            },
        };

        // A phantom frame repeats one of its neighbours. Emitting it would lengthen the output
        // and shift everything after it, and feeding it to a worker would put a false
        // still frame into the temporal window.
        if scenes.phantom.contains(&raw) {
            continue;
        }

        while g >= next_boundary as u64 && scene_idx + 1 < scenes.scene_count() {
            scene_idx += 1;
            next_boundary = scenes.scene_starts[scene_idx + 1];
        }

        let target = scene_idx % workers;

        worker_txs[target]
            .send(WorkerMsg::Frame {
                global_idx: g,
                scene_idx: scene_idx as u32,
                planes,
            })
            .map_err(|_| anyhow::anyhow!("worker {target} disconnected"))?;

        g += 1;
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
    opts: PlaneOptions,
    layout: FrameLayout,
    rx: Receiver<WorkerMsg>,
    tx: SyncSender<OutputMsg>,
) -> Result<(), anyhow::Error> {
    let mut current_scene: Option<u32> = None;
    let mut wd: Option<PlanarDenoiser> = None;

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
                    // Reuse the existing PlanarDenoiser across scenes, flushing the worker
                    // will ensure there is no cross-scene blending during temporal workloads.
                    if let Some(prev) = wd.as_mut() {
                        flush_worker(prev, &mut pending, &tx)?;
                    } else {
                        wd = Some(PlanarDenoiser::create(&opts, layout)?);
                    }

                    current_scene = Some(scene_idx);
                    pending.clear();

                    tracing::debug!(worker_id, scene_idx, "worker started scene");
                }

                let denoiser = wd.as_mut().expect("denoiser exists after new-scene init");

                // Nothing is received straight after the push.
                // `push_with_drain` handles backpressure through QueueFull
                // when the 2-deep pending pipeline fills, and `flush_worker`
                // drains the tail at the scene boundary. Receiving after
                // every push would clamp the pipeline back to depth 1 and
                // put the GPU readback in the critical path of the next
                // push.
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
    denoiser: &mut PlanarDenoiser,
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
    wd: &mut PlanarDenoiser,
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
    peak_frames: usize,
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

    // Counts frames written to the output, which lags the frames read by
    // the depth of the worker pipelines. Emitted frames are the honest
    // measure of progress, because the count stalls whenever whatever
    // consumes our stdout stops reading.
    let pb = denoise_progress_bar(total_frames, visible);

    // The first frame only lands once a worker has compiled its
    // kernels, which takes seconds. A steady tick draws the bar right
    // away and keeps its elapsed time moving until then.
    pb.enable_steady_tick(Duration::from_millis(250));

    let result = emit_frames(&mut encoder, &rx, total_frames as u64, &pb, peak_frames);

    progress::finish(&pb);

    result
}

/// Reorders worker output by frame index and writes it out, updating
/// `pb` as frames are emitted.
///
/// Returns once every frame has been written. If the workers all
/// disconnect before `total` frames have landed this errors, naming how
/// many were written and how many were expected.
fn emit_frames<W: std::io::Write>(
    encoder: &mut y4m::Encoder<W>,
    rx: &Receiver<OutputMsg>,
    total: u64,
    pb: &ProgressBar,
    peak_frames: usize,
) -> Result<(), anyhow::Error> {
    let mut pending: BTreeMap<u64, Planes> = BTreeMap::new();
    let mut next_emit: u64 = 0;

    let mut high_water = 0usize;
    let mut warned = false;

    while next_emit < total {
        let msg = match rx.recv() {
            Ok(m) => m,
            Err(_) => break,
        };

        pending.insert(msg.global_idx, msg.planes);

        if pending.len() > high_water {
            high_water = pending.len();

            if high_water > peak_frames && !warned {
                warned = true;
                tracing::warn!(
                    high_water,
                    peak_frames,
                    "reorder buffer exceeded its predicted peak, frame memory may run high"
                );
            }
        }

        while let Some(planes) = pending.remove(&next_emit) {
            let frame = Y4mFrame::new([&planes.y, &planes.u, &planes.v], None);
            encoder.write_frame(&frame)?;
            next_emit += 1;
        }

        pb.set_position(next_emit);
    }

    tracing::debug!(high_water, peak_frames, "reorder buffer high-water mark");

    if next_emit != total {
        anyhow::bail!(
            "wrote {next_emit} frames but expected {total}. Every worker disconnected \
             before the stream finished, so a frame index was likely lost"
        );
    }

    Ok(())
}

/// Checks each plane's byte length against the layout, failing with an error naming which plane
/// is wrong, the length found and the length expected.
fn check_plane_lens(planes: &Planes, layout: FrameLayout) -> Result<(), anyhow::Error> {
    for (name, got, expected) in [
        ("y", planes.y.len(), layout.luma_bytes()),
        ("u", planes.u.len(), layout.chroma_bytes()),
        ("v", planes.v.len(), layout.chroma_bytes()),
    ] {
        if got != expected {
            anyhow::bail!("{name} plane is {got} bytes, expected {expected} from the frame layout");
        }
    }
    Ok(())
}

fn planes_from_v_frame_u8(
    frame: &v_frame::frame::Frame<u8>,
    layout: FrameLayout,
) -> Result<Planes, anyhow::Error> {
    let y = collect_plane_u8(&frame.y_plane);
    let u = frame
        .u_plane
        .as_ref()
        .map(collect_plane_u8)
        .unwrap_or_else(|| layout.neutral_chroma_plane());
    let v = frame
        .v_plane
        .as_ref()
        .map(collect_plane_u8)
        .unwrap_or_else(|| layout.neutral_chroma_plane());

    let planes = Planes { y, u, v };
    check_plane_lens(&planes, layout)?;

    Ok(planes)
}

fn planes_from_v_frame_u16(
    frame: &v_frame::frame::Frame<u16>,
    layout: FrameLayout,
) -> Result<Planes, anyhow::Error> {
    let y = collect_plane_u16(&frame.y_plane);
    let u = frame
        .u_plane
        .as_ref()
        .map(collect_plane_u16)
        .unwrap_or_else(|| layout.neutral_chroma_plane());
    let v = frame
        .v_plane
        .as_ref()
        .map(collect_plane_u16)
        .unwrap_or_else(|| layout.neutral_chroma_plane());

    let planes = Planes { y, u, v };
    check_plane_lens(&planes, layout)?;

    Ok(planes)
}

fn collect_plane_u8(plane: &v_frame::plane::Plane<u8>) -> Vec<u8> {
    let width = plane.width().get();
    let height = plane.height().get();
    let mut out = Vec::with_capacity(width * height);

    for row in plane.rows() {
        out.extend_from_slice(row);
    }

    out
}

fn collect_plane_u16(plane: &v_frame::plane::Plane<u16>) -> Vec<u8> {
    let width = plane.width().get();
    let height = plane.height().get();
    let mut out = Vec::with_capacity(width * height * 2);

    for row in plane.rows() {
        for &s in row {
            out.extend_from_slice(&s.to_le_bytes());
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
        other => {
            anyhow::bail!("unsupported chroma subsampling {other:?}, need 4:2:0, 4:2:2, or 4:4:4")
        },
    }
}

#[cfg(test)]
mod tests {
    // `temporal_opts` and the one test that uses it are the only things
    // naming `Accelerator::Vulkan`, `Algorithm`, `DenoisingMode`,
    // `Device`, `MotionCompensationMode`, and `ChannelIntent`.
    // Their imports are gated the same way to keep cpu-only builds free
    // of unused-import warnings.
    #[cfg(feature = "vulkan")]
    use av_denoise::accelerate::Accelerator;
    use av_denoise::frame::fill_plane;
    #[cfg(feature = "vulkan")]
    use av_denoise::{Algorithm, ChannelIntent, DenoisingMode, Device};
    use indicatif::ProgressBar;

    use super::*;

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

        // Frame 1 is never sent, as if its index was lost somewhere
        // upstream, and every worker then disconnects. This used to make
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
        let err = emit_frames(&mut encoder, &rx, 3, &pb, 4).expect_err("expected a lost-frame error");

        let msg = err.to_string();
        assert!(
            msg.contains('1') && msg.contains('3'),
            "error should name frames written (1) vs expected (3): {msg}"
        );
    }

    /// Gated because it names the `Vulkan` accelerator variant, which only
    /// exists when the `vulkan` feature is enabled.
    #[cfg(feature = "vulkan")]
    fn temporal_opts() -> PlaneOptions {
        PlaneOptions {
            accelerators: vec![Accelerator::Vulkan],
            device: Device::Default,
            intent: ChannelIntent::LumaChroma,
            mode: DenoisingMode::Temporal { radius: 1 },
            algorithm: Algorithm::default(),
            luma_strength: None,
            chroma_strength: None,
            luma_lambda_ht: None,
            chroma_lambda_ht: None,
            luma_mismatch_scale: None,
            chroma_mismatch_scale: None,
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
    fn planes_from_v_frame_u8_matching_layout_succeeds() {
        use std::num::{NonZeroU8, NonZeroUsize};

        use v_frame::chroma::ChromaSubsampling;
        use v_frame::frame::FrameBuilder;

        let layout = FrameLayout {
            width: 2,
            height: 2,
            subsampling: Subsampling::Yuv420,
            depth: Depth::Eight,
        };
        let frame: v_frame::frame::Frame<u8> = FrameBuilder::new(
            NonZeroUsize::new(2).expect("width is non-zero"),
            NonZeroUsize::new(2).expect("height is non-zero"),
            ChromaSubsampling::Yuv420,
            NonZeroU8::new(8).expect("depth is non-zero"),
        )
        .build()
        .expect("a 2x2 8-bit frame builds");

        let planes = planes_from_v_frame_u8(&frame, layout).expect("matching layout should not error");

        assert_eq!(planes.y.len(), layout.luma_bytes());
        assert_eq!(planes.u.len(), layout.chroma_bytes());
        assert_eq!(planes.v.len(), layout.chroma_bytes());
    }

    #[test]
    fn planes_from_v_frame_u16_matching_layout_succeeds() {
        use std::num::{NonZeroU8, NonZeroUsize};

        use v_frame::chroma::ChromaSubsampling;
        use v_frame::frame::FrameBuilder;

        let layout = FrameLayout {
            width: 2,
            height: 2,
            subsampling: Subsampling::Yuv420,
            depth: Depth::Ten,
        };
        let frame: v_frame::frame::Frame<u16> = FrameBuilder::new(
            NonZeroUsize::new(2).expect("width is non-zero"),
            NonZeroUsize::new(2).expect("height is non-zero"),
            ChromaSubsampling::Yuv420,
            NonZeroU8::new(10).expect("depth is non-zero"),
        )
        .build()
        .expect("a 2x2 10-bit frame builds");

        let planes = planes_from_v_frame_u16(&frame, layout).expect("matching layout should not error");

        assert_eq!(planes.y.len(), layout.luma_bytes());
        assert_eq!(planes.u.len(), layout.chroma_bytes());
        assert_eq!(planes.v.len(), layout.chroma_bytes());
    }

    #[test]
    fn planes_from_v_frame_u8_mismatched_layout_errors() {
        use std::num::{NonZeroU8, NonZeroUsize};

        use v_frame::chroma::ChromaSubsampling;
        use v_frame::frame::FrameBuilder;

        let frame: v_frame::frame::Frame<u8> = FrameBuilder::new(
            NonZeroUsize::new(2).expect("width is non-zero"),
            NonZeroUsize::new(2).expect("height is non-zero"),
            ChromaSubsampling::Yuv420,
            NonZeroU8::new(8).expect("depth is non-zero"),
        )
        .build()
        .expect("a 2x2 8-bit frame builds");

        let layout = FrameLayout {
            width: 4,
            height: 4,
            subsampling: Subsampling::Yuv420,
            depth: Depth::Eight,
        };

        let err = planes_from_v_frame_u8(&frame, layout).expect_err("a smaller frame should not pass");
        let msg = err.to_string();

        assert!(msg.contains('y'), "error should name the plane: {msg}");
        assert!(
            msg.contains('4'),
            "error should name the 2x2 plane's length (4): {msg}"
        );
        assert!(
            msg.contains("16"),
            "error should name the layout's expected length (16): {msg}"
        );
    }

    /// Gated because it depends on `temporal_opts`, which names the
    /// `Vulkan` accelerator variant and only builds when the `vulkan`
    /// feature is enabled.
    #[cfg(feature = "vulkan")]
    #[test]
    fn flush_worker_errors_when_coordinator_has_disconnected() {
        let layout = tiny_layout();
        let mut wd = PlanarDenoiser::create(&temporal_opts(), layout).expect("denoiser construction failed");
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

#[cfg(test)]
mod budget_tests {
    use super::*;

    fn layout(width: u32, height: u32, depth: Depth) -> FrameLayout {
        FrameLayout {
            width,
            height,
            subsampling: Subsampling::Yuv420,
            depth,
        }
    }

    /// The common case must keep today's depths exactly, so 8-bit 1080p
    /// behaviour does not change.
    #[test]
    fn small_frames_keep_the_maximum_depths() {
        let b = channel_budget(layout(1920, 1080, Depth::Eight), 2);

        assert_eq!(b.frame_depth, FRAME_CHANNEL_DEPTH_MAX);
        assert_eq!(b.output_depth, OUTPUT_CHANNEL_DEPTH_MAX);
    }

    /// Pins the exact output of the scaling branch.
    ///
    /// Relative assertions like "large <= small" pass even if the shrink
    /// path never runs, so this names the numbers instead.
    ///
    /// A 4K 10-bit 4:2:0 frame is 24,883,200 bytes, so the 1 GiB budget
    /// affords 43 frames against a 96-frame request. That gives a scale
    /// of 43/96.
    #[test]
    fn large_frames_shrink_the_depths() {
        let small = channel_budget(layout(1920, 1080, Depth::Eight), 8);
        let large = channel_budget(layout(3840, 2160, Depth::Ten), 8);

        assert_eq!(
            (small.frame_depth, small.output_depth),
            (FRAME_CHANNEL_DEPTH_MAX, OUTPUT_CHANNEL_DEPTH_MAX),
            "1080p 8-bit at 8 workers still fits the budget"
        );

        assert_eq!(large.frame_depth, 3, "floor(8 * 43/96)");
        assert_eq!(large.output_depth, 14, "floor(32 * 43/96)");
        assert_eq!(large.ceiling_frames, 38, "8 * 3 + 14");
        assert_eq!(large.peak_frames, 62, "8 * (3 + 3) + 14");
    }

    #[test]
    fn depths_never_fall_below_the_minimum() {
        // Deliberately absurd frame size, far past any budget.
        let b = channel_budget(layout(15360, 8640, Depth::Twelve), 16);

        assert!(b.frame_depth >= FRAME_CHANNEL_DEPTH_MIN);
        assert!(b.output_depth >= OUTPUT_CHANNEL_DEPTH_MIN);
    }

    #[test]
    fn budget_is_respected_where_the_minimums_allow_it() {
        let l = layout(3840, 2160, Depth::Ten);
        let b = channel_budget(l, 8);
        let frame_bytes = l.luma_bytes() + 2 * l.chroma_bytes();

        let floor_frames = 8 * FRAME_CHANNEL_DEPTH_MIN + OUTPUT_CHANNEL_DEPTH_MIN;
        if floor_frames * frame_bytes <= FRAME_MEMORY_BUDGET_BYTES {
            assert!(
                b.ceiling_frames * frame_bytes <= FRAME_MEMORY_BUDGET_BYTES,
                "ceiling {} frames x {frame_bytes} bytes exceeds the budget",
                b.ceiling_frames
            );
        }
    }
}

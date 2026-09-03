use std::collections::{BTreeMap, BTreeSet};
use std::io::{IsTerminal, stdout};
use std::path::Path;
use std::thread;
use std::time::Duration;

use av_decoders::{Decoder, Rational32};
use av_denoise::{
    DenoisingMode,
    Depth,
    FrameLayout,
    PlanarDenoiser,
    PlaneOptions,
    Planes,
    Subsampling,
    WarmUp,
    push_needs_retry,
};
use av_scenechange::{DetectionOptions, detect_scene_changes};
use indicatif::ProgressBar;
use y4m::Frame as Y4mFrame;

use crate::cli::RunOptions;
use crate::frame_index;
use crate::progress::{self, denoise_bar_visible, denoise_progress_bar, scene_progress_bar};
use crate::warm_start::{create_denoiser, finish_warm_up};
use crate::y4m_format::subsampling_to_y4m;

/// Frames in flight this run allows.
fn frame_permits(budget_bytes: u64, frame_bytes: usize, workers: usize, radius: u32) -> usize {
    // A worker emits nothing until `push` first returns QueueFull, which
    // takes `radius + MAX_PENDING + 1` pushes. Fewer permits than that
    // and its scene can never return one, so the dispatcher waits on a
    // permit the worker cannot release.
    let floor = workers * (radius as usize + av_denoise::MAX_PENDING + 2);

    frames_afforded(budget_bytes, frame_bytes).max(floor)
}

/// Frames the budget pays for at this frame size.
fn frames_afforded(budget_bytes: u64, frame_bytes: usize) -> usize {
    let per_frame = (frame_bytes as u64).max(1);

    usize::try_from(budget_bytes / per_frame).unwrap_or(usize::MAX)
}

/// Renders a byte count in the decimal units `--frame-budget` accepts.
///
/// The space `ByteSize` puts before the unit goes, so the result is one
/// shell argument a caller can paste straight back into the flag.
fn size_string(bytes: u64) -> String {
    bytesize::ByteSize::b(bytes)
        .display()
        .si()
        .to_string()
        .replace(' ', "")
}

/// Rounds a byte count up to the precision [`size_string`] prints at.
///
/// The rendering keeps one decimal place, so a raw minimum can round
/// down to a size that still fails the budget check.
fn suggested_budget(bytes: u64) -> String {
    let unit = [bytesize::GB, bytesize::MB, bytesize::KB]
        .into_iter()
        .find(|&unit| bytes >= unit)
        .unwrap_or(1);
    let step = (unit / 10).max(1);

    size_string(bytes.div_ceil(step) * step)
}

/// Frames in flight this run allows, refusing a budget below the floor.
///
/// A budget the floor has to raise serialises the pipeline, so it fails
/// here rather than running on with too few frames in flight.
fn checked_frame_permits(
    budget_bytes: u64,
    frame_bytes: usize,
    workers: usize,
    radius: u32,
) -> Result<usize, anyhow::Error> {
    let afforded = frames_afforded(budget_bytes, frame_bytes);
    let permits = frame_permits(budget_bytes, frame_bytes, workers, radius);

    if permits > afforded {
        let suggestion = suggested_budget(permits as u64 * frame_bytes as u64);

        anyhow::bail!(
            "--frame-budget {budget} affords {afforded} frames at {frame_bytes} bytes per frame, \
             but {workers} workers at temporal radius {radius} need at least {permits}. Pass at \
             least --frame-budget {suggestion}.",
            budget = size_string(budget_bytes),
        );
    }

    Ok(permits)
}

/// Builds a counting semaphore holding `count` permits.
///
/// Returns the giving end and the taking end. The coordinator holds the
/// giving end, so if it dies the dispatcher's next take fails instead of
/// blocking forever.
fn frame_permit_channel(count: usize) -> (crossbeam_channel::Sender<()>, crossbeam_channel::Receiver<()>) {
    let (give, take) = crossbeam_channel::bounded::<()>(count);

    for _ in 0..count {
        give.send(()).expect("the channel holds exactly `count` permits");
    }

    (give, take)
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

pub fn run_file(
    opts: &RunOptions,
    input: &Path,
    workers: usize,
    frame_budget_bytes: u64,
) -> Result<(), anyhow::Error> {
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
        frame_budget_bytes,
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
    frame_budget_bytes: u64,
) -> Result<(), anyhow::Error> {
    let radius = match opts.mode {
        DenoisingMode::Temporal { radius } => radius,
        DenoisingMode::Spacial => 0,
    };

    let frame_bytes = scenes.layout.luma_bytes() + 2 * scenes.layout.chroma_bytes();
    let permits = checked_frame_permits(frame_budget_bytes, frame_bytes, workers, radius)?;

    let (give, take) = frame_permit_channel(permits);

    tracing::info!(
        permits,
        frame_bytes,
        ceiling_mib = (permits * frame_bytes) / (1 << 20),
        "frame buffer budget",
    );

    let (job_tx, worker_handles, out_rx) = spawn_workers(opts, scenes.layout, workers);
    let coordinator = spawn_coordinator(
        scenes.layout,
        scenes.framerate,
        out_rx,
        scenes.total_frames,
        visible,
        give,
    );

    dispatch_frames(input, scenes, &job_tx, &take)?;

    // Closing the queue is what tells the workers there are no more scenes.
    drop(job_tx);

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

/// Spawns `workers` worker threads over one shared scene queue.
///
/// The queue is a rendezvous, so a scene is only offered when a worker is
/// free and at most `workers` scenes are ever in flight.
///
/// Returns the queue's sender, their join handles, and the shared output
/// channel they emit denoised frames on.
fn spawn_workers(
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

fn spawn_coordinator(
    layout: FrameLayout,
    framerate: Rational32,
    rx: crossbeam_channel::Receiver<OutputMsg>,
    total_frames: usize,
    visible: bool,
    permits: crossbeam_channel::Sender<()>,
) -> thread::JoinHandle<Result<(), anyhow::Error>> {
    thread::spawn(move || run_coordinator(layout, framerate, rx, total_frames, visible, permits))
}

/// Reads every frame in order and offers each scene to the worker pool.
///
/// A scene's frames go into a channel of their own. Dropping that
/// channel's sender is what tells the claiming worker the scene has
/// ended.
///
/// Each staged frame holds a permit from here until the coordinator has
/// written it, which is the only bound on frames in flight. The permit
/// is taken just before the send rather than before the decode, so a
/// phantom frame never takes one and at most one decoded frame is
/// transient outside the budget.
fn stage_frames<I>(
    frames: I,
    scenes: &SceneLayout,
    jobs: &crossbeam_channel::Sender<SceneJob>,
    permits: &crossbeam_channel::Receiver<()>,
) -> Result<(), anyhow::Error>
where
    I: Iterator<Item = Result<Planes, anyhow::Error>>,
{
    let mut scene_idx = 0usize;
    let mut next_boundary = scenes.scene_starts[1];
    let mut g = 0u64;
    let mut current: Option<(usize, crossbeam_channel::Sender<StagedFrame>)> = None;

    // The iterator yields frames in raw decoder order, so position is the
    // raw index. Every frame is read, phantom or not, because the decoder
    // walks the file in order and cannot be told to skip one.
    for (raw, planes) in frames.enumerate() {
        let planes = planes?;

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

        if !matches!(&current, Some((idx, _)) if *idx == scene_idx) {
            let (tx, rx) = crossbeam_channel::unbounded::<StagedFrame>();

            // Dropping the previous scene's sender ends that scene, which
            // frees the worker holding it to claim this one. The queue is a
            // rendezvous, so offering the job first would deadlock whenever
            // every worker is busy.
            drop(current.take());

            jobs.send(SceneJob {
                scene_idx: scene_idx as u32,
                frames: rx,
            })
            .map_err(|_| anyhow::anyhow!("worker pool disconnected"))?;

            current = Some((scene_idx, tx));
        }

        // Taken here rather than before the decode, so a phantom frame
        // never takes one. At most one decoded frame is transient outside
        // the budget.
        permits
            .recv()
            .map_err(|_| anyhow::anyhow!("the coordinator stopped before the stream finished"))?;

        let (_, tx) = current
            .as_ref()
            .expect("a scene sender exists after the check above");

        tx.send(StagedFrame {
            global_idx: g,
            planes,
        })
        .map_err(|_| anyhow::anyhow!("the worker holding scene {scene_idx} disconnected"))?;

        g += 1;
    }

    Ok(())
}

/// Opens the input and stages every frame it decodes.
fn dispatch_frames(
    input: &Path,
    scenes: &SceneLayout,
    jobs: &crossbeam_channel::Sender<SceneJob>,
    permits: &crossbeam_channel::Receiver<()>,
) -> Result<(), anyhow::Error> {
    let mut decoder = Decoder::from_file(input)?;
    let layout = scenes.layout;

    let frames = (0..scenes.raw_frames).map(move |_| -> Result<Planes, anyhow::Error> {
        match layout.depth {
            Depth::Eight => {
                let frame = decoder.read_video_frame::<u8>()?;
                planes_from_v_frame_u8(&frame, layout)
            },
            Depth::Ten | Depth::Twelve => {
                let frame = decoder.read_video_frame::<u16>()?;
                planes_from_v_frame_u16(&frame, layout)
            },
        }
    });

    stage_frames(frames, scenes, jobs, permits)
}

/// One decoded frame, staged for the worker that claims its scene.
struct StagedFrame {
    global_idx: u64,
    planes: Planes,
}

/// One scene, offered to whichever worker is free.
///
/// `frames` closes when the scene has no more frames, which is how a
/// worker knows to flush.
struct SceneJob {
    scene_idx: u32,
    frames: crossbeam_channel::Receiver<StagedFrame>,
}

struct OutputMsg {
    global_idx: u64,
    planes: Planes,
}

fn run_worker(
    worker_id: usize,
    opts: PlaneOptions,
    layout: FrameLayout,
    jobs: crossbeam_channel::Receiver<SceneJob>,
    tx: crossbeam_channel::Sender<OutputMsg>,
) -> Result<(), anyhow::Error> {
    let mut wd: Option<PlanarDenoiser> = None;
    // The cold-cache queue place this worker's denoiser holds, until its
    // first output frame proves the kernels are compiled and cached.
    let mut warm_up: Option<WarmUp> = None;

    while let Ok(job) = jobs.recv() {
        // Built on the first claimed scene, so a worker that never claims
        // one never compiles.
        if wd.is_none() {
            let (denoiser, place) = create_denoiser(&opts, layout)?;
            wd = Some(denoiser);
            warm_up = place;
        }

        let denoiser = wd.as_mut().expect("denoiser exists after the check above");

        tracing::debug!(worker_id, scene_idx = job.scene_idx, "worker started scene");

        // Indices of pushed-but-not-yet-emitted frames, in push order.
        let mut pending: std::collections::VecDeque<u64> = Default::default();

        // Nothing is received straight after the push.
        // `push_with_drain` handles backpressure through QueueFull
        // when the 2-deep pending pipeline fills, and `flush_worker`
        // drains the tail below. Receiving after every push would clamp
        // the pipeline back to depth 1 and put the GPU readback in the
        // critical path of the next push.
        for frame in job.frames {
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
    }

    Ok(())
}

/// Push one frame, draining any pending output first if the queue is full.
fn push_with_drain(
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
            let g = pending
                .pop_front()
                .expect("pending has at least one entry on QueueFull recv");
            send_output(tx, g, out)?;
            finish_warm_up(warm_up);
        }

        denoiser.push(planes)?;
    }

    Ok(())
}

fn send_output(
    tx: &crossbeam_channel::Sender<OutputMsg>,
    global_idx: u64,
    planes: Planes,
) -> Result<(), anyhow::Error> {
    tx.send(OutputMsg { global_idx, planes })
        .map_err(|_| anyhow::anyhow!("coordinator disconnected"))
}

fn flush_worker(
    wd: &mut PlanarDenoiser,
    warm_up: &mut Option<WarmUp>,
    pending: &mut std::collections::VecDeque<u64>,
    tx: &crossbeam_channel::Sender<OutputMsg>,
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

fn run_coordinator(
    layout: FrameLayout,
    framerate: Rational32,
    rx: crossbeam_channel::Receiver<OutputMsg>,
    total_frames: usize,
    visible: bool,
    permits: crossbeam_channel::Sender<()>,
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

    let result = emit_frames(&mut encoder, &rx, total_frames as u64, &pb, &permits);

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
    rx: &crossbeam_channel::Receiver<OutputMsg>,
    total: u64,
    pb: &ProgressBar,
    permits: &crossbeam_channel::Sender<()>,
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

            // Returning the permit is what lets the decoder run further
            // ahead. The send never blocks, because permits held plus
            // permits waiting is always the channel's capacity. The
            // result is discarded because it fails once the dispatcher
            // has already errored out and dropped its receiver.
            let _ = permits.send(());
        }

        pb.set_position(next_emit);
    }

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
    // naming `Accelerator::Vulkan`, `Algorithm`, `Device`, and
    // `ChannelIntent`. Their imports are gated the same way to keep
    // cpu-only builds free of unused-import warnings.
    #[cfg(feature = "vulkan")]
    use av_denoise::accelerate::Accelerator;
    use av_denoise::frame::fill_plane;
    #[cfg(feature = "vulkan")]
    use av_denoise::{Algorithm, ChannelIntent, Device};
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

    fn scene_layout(scene_starts: Vec<usize>, phantom: BTreeSet<usize>) -> SceneLayout {
        let total_frames = *scene_starts
            .last()
            .expect("scene_starts ends with the frame count");

        SceneLayout {
            layout: tiny_layout(),
            framerate: Rational32::new(30, 1),
            total_frames,
            raw_frames: total_frames + phantom.len(),
            phantom,
            scene_starts,
        }
    }

    /// Drains every job the stager offers, returning each scene index with
    /// the frame indices that scene carried.
    ///
    /// Runs on its own thread because the scene queue is a rendezvous, so the
    /// stager blocks until someone claims each job.
    fn collect_jobs(rx: crossbeam_channel::Receiver<SceneJob>) -> thread::JoinHandle<Vec<(u32, Vec<u64>)>> {
        thread::spawn(move || {
            let mut out = Vec::new();

            while let Ok(job) = rx.recv() {
                let idx = job.scene_idx;
                let frames = job.frames.iter().map(|f| f.global_idx).collect();
                out.push((idx, frames));
            }

            out
        })
    }

    #[test]
    fn stage_frames_offers_one_job_per_scene_in_order() {
        let scenes = scene_layout(vec![0, 2, 4, 6], BTreeSet::new());
        let planes = tiny_planes(scenes.layout);
        let frames = (0..6).map(move |_| Ok(planes.clone()));

        let (_give, take) = frame_permit_channel(8);
        let (job_tx, job_rx) = crossbeam_channel::bounded::<SceneJob>(0);
        let collector = collect_jobs(job_rx);

        stage_frames(frames, &scenes, &job_tx, &take).expect("staging should succeed");
        drop(job_tx);

        let jobs = collector.join().expect("collector panicked");

        assert_eq!(jobs, vec![(0, vec![0, 1]), (1, vec![2, 3]), (2, vec![4, 5])],);
    }

    #[test]
    fn stage_frames_skips_phantom_frames_without_advancing_the_index() {
        let scenes = scene_layout(vec![0, 4], BTreeSet::from([1, 3]));
        let planes = tiny_planes(scenes.layout);
        let frames = (0..6).map(move |_| Ok(planes.clone()));

        let (_give, take) = frame_permit_channel(8);
        let (job_tx, job_rx) = crossbeam_channel::bounded::<SceneJob>(0);
        let collector = collect_jobs(job_rx);

        stage_frames(frames, &scenes, &job_tx, &take).expect("staging should succeed");
        drop(job_tx);

        let jobs = collector.join().expect("collector panicked");

        assert_eq!(jobs, vec![(0, vec![0, 1, 2, 3])]);
    }

    #[test]
    fn a_scene_job_channel_closes_when_its_scene_ends() {
        let scenes = scene_layout(vec![0, 2], BTreeSet::new());
        let planes = tiny_planes(scenes.layout);
        let frames = (0..2).map(move |_| Ok(planes.clone()));

        let (_give, take) = frame_permit_channel(8);
        let (job_tx, job_rx) = crossbeam_channel::bounded::<SceneJob>(0);
        let claimed = thread::spawn(move || job_rx.recv().expect("one job is offered"));

        stage_frames(frames, &scenes, &job_tx, &take).expect("staging should succeed");
        drop(job_tx);

        let job = claimed.join().expect("claimant panicked");

        assert_eq!(job.frames.recv().map(|f| f.global_idx).ok(), Some(0));
        assert_eq!(job.frames.recv().map(|f| f.global_idx).ok(), Some(1));
        assert!(
            job.frames.recv().is_err(),
            "the scene's channel closes after its last frame"
        );
    }

    /// A worker that claims a scene and dies without draining it must surface
    /// as an error. Before the queue became a rendezvous, the dead worker's
    /// job could sit in the queue keeping the scene channel alive, and the
    /// stager blocked on it forever.
    #[test]
    fn staging_fails_rather_than_hanging_when_the_pool_dies() {
        let scenes = scene_layout(vec![0, 10], BTreeSet::new());
        let planes = tiny_planes(scenes.layout);
        let frames = (0..10).map(move |_| Ok(planes.clone()));

        let (_give, take) = frame_permit_channel(16);
        let (job_tx, job_rx) = crossbeam_channel::bounded::<SceneJob>(0);
        let pool = thread::spawn(move || drop(job_rx.recv()));

        let err = stage_frames(frames, &scenes, &job_tx, &take).expect_err("staging must not hang");

        pool.join().expect("pool panicked");

        assert!(
            err.to_string().contains("disconnect"),
            "error should name the disconnect: {err}"
        );
    }

    #[test]
    fn frame_permits_follows_the_budget_when_it_clears_the_floor() {
        // A 1080p 8-bit 4:2:0 frame is 3,110,400 bytes, so 1 GiB affords 345.
        assert_eq!(frame_permits(1 << 30, 3_110_400, 4, 0), 345);
    }

    #[test]
    fn frame_permits_applies_the_floor_when_the_budget_is_too_small() {
        let floor = 4 * (av_denoise::MAX_PENDING + 2);

        // A 4K 10-bit frame is 24,883,200 bytes, so 1 MiB affords none.
        assert_eq!(frame_permits(1 << 20, 24_883_200, 4, 0), floor);
    }

    #[test]
    fn a_budget_below_the_floor_is_rejected() {
        // A 4K 10-bit frame is 24,883,200 bytes, so 1 MB affords none.
        let err = checked_frame_permits(1_000_000, 24_883_200, 4, 8)
            .expect_err("1 MB cannot feed 4 workers at radius 8");
        let msg = err.to_string();

        let floor = 4 * (8 + av_denoise::MAX_PENDING + 2);

        assert!(msg.contains("affords 0 frames"), "got {msg}");
        assert!(msg.contains(&format!("at least {floor}")), "got {msg}");
        assert!(msg.contains("Pass at least --frame-budget 1.2GB"), "got {msg}");
    }

    #[test]
    fn a_budget_that_clears_the_floor_is_accepted() {
        let permits = checked_frame_permits(1 << 30, 3_110_400, 4, 0).expect("1 GiB clears the floor");

        assert_eq!(permits, 345);
    }

    #[test]
    fn the_floor_covers_a_workers_first_output_at_every_radius() {
        // A budget far too small for any real frame, so the floor decides.
        for radius in [0u32, 1, 4, 8] {
            let permits = frame_permits(1, 199_065_600, 1, radius);

            // Pushes a worker needs before `push` first returns QueueFull,
            // which is the first point it can emit and return a permit.
            let first_output = radius as usize + av_denoise::MAX_PENDING + 1;

            assert!(
                permits >= first_output,
                "radius {radius} needs {first_output} permits before a worker emits, got {permits}",
            );
        }
    }

    #[test]
    fn every_permit_is_accounted_for_once_staging_finishes() {
        let scenes = scene_layout(vec![0, 3, 6], BTreeSet::new());
        let planes = tiny_planes(scenes.layout);
        let frames = (0..6).map(move |_| Ok(planes.clone()));

        let (give, take) = frame_permit_channel(8);
        let (job_tx, job_rx) = crossbeam_channel::bounded::<SceneJob>(0);

        let drained = thread::spawn(move || {
            let mut n = 0usize;
            while let Ok(job) = job_rx.recv() {
                n += job.frames.iter().count();
            }
            n
        });

        stage_frames(frames, &scenes, &job_tx, &take).expect("staging should succeed");
        drop(job_tx);

        assert_eq!(drained.join().expect("drain panicked"), 6);
        assert_eq!(take.len(), 2, "6 of 8 permits are out, since nothing was written");

        for _ in 0..6 {
            give.send(()).expect("returning a permit never blocks");
        }

        assert_eq!(take.len(), 8);
    }

    /// A worker that claims a scene and stops reading it must not stop
    /// later scenes being offered to anyone else.
    ///
    /// Scene 0 holds ten frames that nobody drains. Only the permit budget
    /// bounds staging, and it counts the whole pipeline rather than one
    /// scene, so the stager runs past scene 0 and offers scene 1 to a free
    /// worker. Bounding each scene's channel instead would block the stager
    /// inside scene 0 and starve every idle worker behind it, which is the
    /// stall this pins.
    #[test]
    fn a_backlogged_scene_does_not_stop_later_scenes_being_offered() {
        let scenes = scene_layout(vec![0, 10, 12], BTreeSet::new());
        let planes = tiny_planes(scenes.layout);
        let frames = (0..12).map(move |_| Ok(planes.clone()));

        let (give, take) = frame_permit_channel(64);
        let (job_tx, job_rx) = crossbeam_channel::bounded::<SceneJob>(0);

        let consumer = thread::spawn(move || {
            let first = job_rx.recv().expect("scene 0 is offered");

            // Held rather than discarded. Dropping a job closes its frame
            // channel, and the stager is still filling scene 1's.
            let second = job_rx.recv_timeout(Duration::from_secs(5)).ok();
            let offered_while_backlogged = second.is_some();

            // Drain everything either way, so a failing run finishes and
            // reports instead of hanging. The second job outlives this, so
            // staging never sees its channel close early.
            for _ in first.frames.iter() {}
            while job_rx.recv().is_ok() {}
            drop(second);

            offered_while_backlogged
        });

        stage_frames(frames, &scenes, &job_tx, &take).expect("staging should not stall");
        drop(job_tx);

        assert!(
            consumer.join().expect("consumer panicked"),
            "scene 1 must be offered while scene 0 is still backlogged",
        );

        drop(give);
    }

    #[test]
    fn emit_frames_errors_when_a_frame_index_is_lost() {
        let layout = tiny_layout();
        let (tx, rx) = crossbeam_channel::unbounded::<OutputMsg>();
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
        // Two permits stand in for the two staged frames, so returning
        // one has somewhere to go. A full permit channel would block the
        // return, which cannot happen in a real run.
        let (give, take) = frame_permit_channel(4);
        take.recv().expect("a permit is available");
        take.recv().expect("a permit is available");

        let err = emit_frames(&mut encoder, &rx, 3, &pb, &give).expect_err("expected a lost-frame error");

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

        let (tx, rx) = crossbeam_channel::unbounded::<OutputMsg>();
        drop(rx);

        let mut warm_up = None;
        let err = flush_worker(&mut wd, &mut warm_up, &mut pending, &tx)
            .expect_err("expected the coordinator disconnect to surface as an error");

        assert!(
            err.to_string().contains("disconnect"),
            "error should mention the coordinator disconnect: {err}"
        );
    }
}

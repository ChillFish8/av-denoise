mod convert;
mod coordinator;
mod decode;
mod scenes;
mod source;
mod stage;
mod worker;

#[cfg(test)]
mod tests;

use std::collections::BTreeSet;
use std::io::IsTerminal;
use std::path::Path;

use av_decoders::{Decoder, VideoDetails};
use av_denoise::{DenoisingMode, Depth, FrameLayout, PlaneOptions, Planes};
use av_scenechange::{DetectionOptions, detect_scene_changes};

use self::convert::{planes_from_v_frame_u8, planes_from_v_frame_u16, subsampling_from_av_decoders};
use self::coordinator::spawn_coordinator;
use self::source::SourceInfo;
use self::stage::{SceneJob, Stager, checked_frame_permits, frame_permit_channel};
use self::worker::spawn_workers;
use crate::cli::RunOptions;
use crate::frame_index;
use crate::progress::{self, denoise_bar_visible, scene_progress_bar};

/// Scene boundaries plus the video metadata needed to build the output
/// y4m header.
pub struct SceneLayout {
    pub details: VideoDetails,
    pub layout: FrameLayout,
    /// Frames this run emits, being `raw_frames` less the phantom entries.
    pub total_frames: usize,
    /// Frames the decoder hands over, phantom entries included. Every
    /// one has to be read to keep the sequential decoder in step, even
    /// though only `total_frames` of them are emitted.
    pub raw_frames: usize,
    /// Decoder frame numbers that carry no picture of their own. See
    /// [`crate::frame_index`].
    pub phantom: BTreeSet<usize>,
    /// `scene_starts[i]` is the inclusive start frame of scene `i`, in  emitted frame numbers.
    /// The final entry is `total_frames` so scene `i` covers `scene_starts[i]..scene_starts[i + 1]`.
    pub scene_starts: Vec<usize>,
}

impl SceneLayout {
    pub fn scene_count(&self) -> usize {
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
pub fn detect_scenes(input: &Path, visible: bool) -> Result<SceneLayout, anyhow::Error> {
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
        details,
        layout,
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
pub fn encode_scenes(
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
    let (staged_tx, staged_rx) = crossbeam_channel::bounded::<u64>(1);
    let info = SourceInfo {
        details: scenes.details,
        layout: scenes.layout,
        pixel_aspect: None,
        vendor_extensions: Vec::new(),
        estimated_frames: Some(scenes.total_frames),
    };
    let coordinator = spawn_coordinator(info, out_rx, staged_rx, visible, give, std::io::stdout());

    let staged = dispatch_frames(input, scenes, &job_tx, &take)?;

    // A closed channel means the coordinator failed, and it reports its own error when joined.
    let _ = staged_tx.send(staged);

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

/// Opens the input and stages every frame it decodes, returning how many were staged.
pub fn dispatch_frames(
    input: &Path,
    scenes: &SceneLayout,
    jobs: &crossbeam_channel::Sender<SceneJob>,
    permits: &crossbeam_channel::Receiver<()>,
) -> Result<u64, anyhow::Error> {
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

/// Reads every frame in order and stages the ones the scene layout keeps.
///
/// The permit is taken just before each frame is staged, so a phantom frame never takes one.
fn stage_frames<I>(
    frames: I,
    scenes: &SceneLayout,
    jobs: &crossbeam_channel::Sender<SceneJob>,
    permits: &crossbeam_channel::Receiver<()>,
) -> Result<u64, anyhow::Error>
where
    I: Iterator<Item = Result<Planes, anyhow::Error>>,
{
    let mut stager = Stager::new(jobs);
    let mut emitted = 0usize;
    let mut next_scene = 0usize;

    // The iterator yields frames in raw decoder order, so position is the raw index. Every
    // frame is read, phantom or not, because the decoder cannot be told to skip one.
    for (raw, planes) in frames.enumerate() {
        let planes = planes?;

        // A phantom frame repeats one of its neighbours. Emitting it would lengthen the output
        // and shift everything after it.
        if scenes.phantom.contains(&raw) {
            continue;
        }

        let starts_scene = scenes.scene_starts.get(next_scene) == Some(&emitted);

        if starts_scene {
            next_scene += 1;
        }

        permits
            .recv()
            .map_err(|_| anyhow::anyhow!("the coordinator stopped before the stream finished"))?;
        stager.stage(planes, starts_scene)?;
        emitted += 1;
    }

    Ok(stager.finish())
}

use av_denoise::Planes;

use super::decode::PREFETCH_FRAMES;
use super::scenes::LOOKAHEAD_DISTANCE;

/// Frames holding a permit that no worker has received yet.
///
/// These sit in the prefetch channel, the scene splitter's look-ahead window, or in the decode
/// thread's hand while it waits to send.
pub const IN_TRANSIT_FRAMES: usize = LOOKAHEAD_DISTANCE + 2 + PREFETCH_FRAMES + 1;

/// Frames in flight this run allows.
pub fn frame_permits(budget_bytes: u64, frame_bytes: usize, workers: usize, radius: u32) -> usize {
    // A worker emits nothing until `push` first returns QueueFull, which
    // takes `radius + MAX_PENDING + 1` pushes, and the frames upstream of
    // the workers hold permits too. Fewer permits than that and the
    // dispatcher waits on a permit nothing can release.
    let per_worker = radius as usize + av_denoise::MAX_PENDING + 2;
    let floor = workers * per_worker + IN_TRANSIT_FRAMES;

    frames_afforded(budget_bytes, frame_bytes).max(floor)
}

/// Frames the budget pays for at this frame size.
pub fn frames_afforded(budget_bytes: u64, frame_bytes: usize) -> usize {
    let per_frame = (frame_bytes as u64).max(1);

    usize::try_from(budget_bytes / per_frame).unwrap_or(usize::MAX)
}

/// Renders a byte count in the decimal units `--frame-budget` accepts.
///
/// The space `ByteSize` puts before the unit goes, so the result is one
/// shell argument a caller can paste straight back into the flag.
pub fn size_string(bytes: u64) -> String {
    bytesize::ByteSize::b(bytes)
        .display()
        .si()
        .to_string()
        .replace(' ', "")
}

/// Rounds a byte count up to the precision [size_string] prints at.
///
/// The rendering keeps one decimal place, so a raw minimum can round
/// down to a size that still fails the budget check.
pub fn suggested_budget(bytes: u64) -> String {
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
pub fn checked_frame_permits(
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
pub fn frame_permit_channel(
    count: usize,
) -> (crossbeam_channel::Sender<()>, crossbeam_channel::Receiver<()>) {
    let (give, take) = crossbeam_channel::bounded::<()>(count);

    for _ in 0..count {
        give.send(()).expect("the channel holds exactly `count` permits");
    }

    (give, take)
}

/// Splits decided frames into scene jobs for the worker pool.
///
/// A scene's frames go into a channel of their own. Dropping that channel's sender is what tells
/// the claiming worker the scene has ended.
pub struct Stager<'a> {
    jobs: &'a crossbeam_channel::Sender<SceneJob>,
    current: Option<crossbeam_channel::Sender<StagedFrame>>,
    scene_count: u32,
    staged: u64,
}

impl<'a> Stager<'a> {
    pub fn new(jobs: &'a crossbeam_channel::Sender<SceneJob>) -> Self {
        Self {
            jobs,
            current: None,
            scene_count: 0,
            staged: 0,
        }
    }

    pub fn stage(&mut self, planes: Planes, starts_scene: bool) -> Result<(), anyhow::Error> {
        if starts_scene || self.current.is_none() {
            self.open_scene()?;
        }

        let scene_idx = self.scene_count - 1;
        let sender = self
            .current
            .as_ref()
            .expect("a scene is open after the check above");
        let frame = StagedFrame {
            global_idx: self.staged,
            planes,
        };

        sender
            .send(frame)
            .map_err(|_| anyhow::anyhow!("the worker holding scene {scene_idx} disconnected"))?;

        self.staged += 1;

        Ok(())
    }

    /// Closes the last scene and returns how many frames were staged.
    pub fn finish(self) -> u64 {
        self.staged
    }

    fn open_scene(&mut self) -> Result<(), anyhow::Error> {
        let (sender, receiver) = crossbeam_channel::unbounded::<StagedFrame>();

        // Dropping the previous scene's sender frees the worker holding it to claim this one.
        // The queue is a rendezvous, so offering the job first would deadlock whenever every
        // worker is busy.
        drop(self.current.take());

        let job = SceneJob {
            scene_idx: self.scene_count,
            frames: receiver,
        };

        self.jobs
            .send(job)
            .map_err(|_| anyhow::anyhow!("worker pool disconnected"))?;

        self.current = Some(sender);
        self.scene_count += 1;

        Ok(())
    }
}

/// One decoded frame, staged for the worker that claims its scene.
pub struct StagedFrame {
    pub global_idx: u64,
    pub planes: Planes,
}

/// One scene, offered to whichever worker is free.
///
/// `frames` closes when the scene has no more frames, which is how a
/// worker knows to flush.
pub struct SceneJob {
    pub scene_idx: u32,
    pub frames: crossbeam_channel::Receiver<StagedFrame>,
}

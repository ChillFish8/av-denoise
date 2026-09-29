use av_denoise::Planes;

use super::SceneLayout;

/// Frames in flight this run allows.
pub fn frame_permits(budget_bytes: u64, frame_bytes: usize, workers: usize, radius: u32) -> usize {
    // A worker emits nothing until `push` first returns QueueFull, which
    // takes `radius + MAX_PENDING + 1` pushes. Fewer permits than that
    // and its scene can never return one, so the dispatcher waits on a
    // permit the worker cannot release.
    let floor = workers * (radius as usize + av_denoise::MAX_PENDING + 2);

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

/// Rounds a byte count up to the precision [`size_string`] prints at.
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
pub fn stage_frames<I>(
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

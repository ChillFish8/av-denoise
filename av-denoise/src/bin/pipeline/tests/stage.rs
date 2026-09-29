use std::thread;
use std::time::Duration;

use super::{tiny_layout, tiny_planes};
use crate::pipeline::stage::{IN_TRANSIT_FRAMES, SceneJob, Stager, checked_frame_permits, frame_permits};

/// Stages one frame per flag, starting a new scene wherever a flag is set.
fn stage_all(flags: &[bool], jobs: &crossbeam_channel::Sender<SceneJob>) -> Result<u64, anyhow::Error> {
    let layout = tiny_layout();
    let planes = tiny_planes(layout);
    let mut stager = Stager::new(jobs);

    for &starts_scene in flags {
        stager.stage(planes.clone(), starts_scene)?;
    }

    Ok(stager.finish())
}

fn flags(scene_lengths: &[usize]) -> Vec<bool> {
    scene_lengths
        .iter()
        .flat_map(|&length| (0..length).map(|offset| offset == 0))
        .collect()
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
fn the_stager_offers_one_job_per_scene_in_order() {
    let (job_tx, job_rx) = crossbeam_channel::bounded::<SceneJob>(0);
    let collector = collect_jobs(job_rx);

    let scene_starts = flags(&[2, 2, 2]);

    let staged = stage_all(&scene_starts, &job_tx).expect("staging should succeed");
    drop(job_tx);

    let jobs = collector.join().expect("collector panicked");

    assert_eq!(staged, 6);
    assert_eq!(jobs, vec![(0, vec![0, 1]), (1, vec![2, 3]), (2, vec![4, 5])]);
}

#[test]
fn a_scene_job_channel_closes_when_its_scene_ends() {
    let (job_tx, job_rx) = crossbeam_channel::bounded::<SceneJob>(0);
    let claimed = thread::spawn(move || job_rx.recv().expect("one job is offered"));

    let scene_starts = flags(&[2]);

    stage_all(&scene_starts, &job_tx).expect("staging should succeed");
    drop(job_tx);

    let job = claimed.join().expect("claimant panicked");

    assert_eq!(job.frames.recv().map(|frame| frame.global_idx).ok(), Some(0));
    assert_eq!(job.frames.recv().map(|frame| frame.global_idx).ok(), Some(1));
    assert!(
        job.frames.recv().is_err(),
        "the scene's channel closes after its last frame"
    );
}

/// A worker that claims a scene and dies without draining it must surface
/// as an error rather than hanging the stager.
#[test]
fn staging_fails_rather_than_hanging_when_the_pool_dies() {
    let (job_tx, job_rx) = crossbeam_channel::bounded::<SceneJob>(0);
    let pool = thread::spawn(move || drop(job_rx.recv()));

    let scene_starts = flags(&[10]);

    let err = stage_all(&scene_starts, &job_tx).expect_err("staging must not hang");

    pool.join().expect("pool panicked");

    assert!(
        err.to_string().contains("disconnect"),
        "error should name the disconnect: {err}"
    );
}

#[test]
fn a_leading_frame_without_a_scene_flag_still_opens_a_scene() {
    let (job_tx, job_rx) = crossbeam_channel::bounded::<SceneJob>(0);
    let collector = collect_jobs(job_rx);

    stage_all(&[false, false], &job_tx).expect("staging should succeed");
    drop(job_tx);

    let jobs = collector.join().expect("collector panicked");

    assert_eq!(jobs, vec![(0, vec![0, 1])]);
}

#[test]
fn frame_permits_follows_the_budget_when_it_clears_the_floor() {
    // A 1080p 8-bit 4:2:0 frame is 3,110,400 bytes, so 1 GiB affords 345.
    assert_eq!(frame_permits(1 << 30, 3_110_400, 4, 0), 345);
}

#[test]
fn frame_permits_applies_the_floor_when_the_budget_is_too_small() {
    let floor = 4 * (av_denoise::MAX_PENDING + 2) + IN_TRANSIT_FRAMES;

    // A 4K 10-bit frame is 24,883,200 bytes, so 1 MiB affords none.
    assert_eq!(frame_permits(1 << 20, 24_883_200, 4, 0), floor);
}

#[test]
fn a_budget_below_the_floor_is_rejected() {
    // A 4K 10-bit frame is 24,883,200 bytes, so 1 MB affords none.
    let err = checked_frame_permits(1_000_000, 24_883_200, 4, 8)
        .expect_err("1 MB cannot feed 4 workers at radius 8");
    let msg = err.to_string();

    let floor = 4 * (8 + av_denoise::MAX_PENDING + 2) + IN_TRANSIT_FRAMES;

    assert!(msg.contains("affords 0 frames"), "got {msg}");
    assert!(msg.contains(&format!("at least {floor}")), "got {msg}");
    // 60 frames at 24,883,200 bytes is 1,492,992,000, rounded up to 1.5GB.
    assert!(msg.contains("Pass at least --frame-budget 1.5GB"), "got {msg}");
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
fn the_floor_covers_the_look_ahead_and_prefetch() {
    // A budget far too small for any real frame, so the floor decides.
    let permits = frame_permits(1, 199_065_600, 1, 0);
    let first_output = av_denoise::MAX_PENDING + 1;
    let held_upstream =
        crate::pipeline::scenes::LOOKAHEAD_DISTANCE + 2 + crate::pipeline::decode::PREFETCH_FRAMES + 1;

    assert!(permits >= first_output + held_upstream, "got {permits}");
}

/// A worker that claims a scene and stops reading it must not stop
/// later scenes being offered to anyone else.
///
/// Scene 0 holds ten frames that nobody drains. Bounding each scene's
/// channel would block the stager inside scene 0 and starve every idle
/// worker behind it, which is the stall this pins.
#[test]
fn a_backlogged_scene_does_not_stop_later_scenes_being_offered() {
    let (job_tx, job_rx) = crossbeam_channel::bounded::<SceneJob>(0);

    let consumer = thread::spawn(move || {
        let first = job_rx.recv().expect("scene 0 is offered");

        // Held rather than discarded. Dropping a job closes its frame
        // channel, and the stager is still filling scene 1's.
        let second = job_rx.recv_timeout(Duration::from_secs(5)).ok();
        let offered_while_backlogged = second.is_some();

        // Drain everything either way, so a failing run finishes and
        // reports instead of hanging.
        for _ in first.frames.iter() {}
        while job_rx.recv().is_ok() {}
        drop(second);

        offered_while_backlogged
    });

    let scene_starts = flags(&[10, 2]);

    stage_all(&scene_starts, &job_tx).expect("staging should not stall");
    drop(job_tx);

    assert!(
        consumer.join().expect("consumer panicked"),
        "scene 1 must be offered while scene 0 is still backlogged",
    );
}

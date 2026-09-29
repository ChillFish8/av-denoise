use std::collections::BTreeSet;
use std::thread;
use std::time::Duration;

use av_decoders::Rational32;

use super::{tiny_layout, tiny_planes};
use crate::pipeline::SceneLayout;
use crate::pipeline::stage::{
    SceneJob,
    checked_frame_permits,
    frame_permit_channel,
    frame_permits,
    stage_frames,
};

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

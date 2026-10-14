use std::time::Duration;

use av_denoise::accelerate::Accelerator;
use av_denoise::{PlanarTiming, PlaneTiming};

/// One frame in this many has its device time sampled when the report is on.
pub const DEVICE_SAMPLE_EVERY: u32 = 8;

/// Where one worker's time went.
#[derive(Debug, Clone, Default)]
pub struct WorkerTiming {
    /// From the worker starting to it finishing its last scene.
    pub wall: Duration,
    /// Building the denoiser.
    pub setup: Duration,
    /// Waiting for the next scene.
    pub idle: Duration,
    pub planes: PlanarTiming,
    /// The backend the worker's denoiser ran on, unset when it never claimed a scene.
    pub accelerator: Option<Accelerator>,
}

impl WorkerTiming {
    /// Host time spent inside the denoiser, blocked and building work together.
    pub fn denoising(&self) -> Duration {
        let mut total = Duration::ZERO;

        for (_, plane) in self.planes.planes() {
            total += plane.blocked + plane.submit;
        }

        total
    }

    /// Wall time that none of the measured parts covers.
    pub fn unaccounted(&self) -> Duration {
        let denoising = self.denoising();
        let covered = self.setup + self.idle + denoising;

        self.wall.saturating_sub(covered)
    }
}

/// What one run measured.
#[derive(Debug, Clone, Default)]
pub struct RunSummary {
    pub wall: Duration,
    /// Frames written.
    pub frames: u64,
    pub workers: Vec<WorkerTiming>,
}

/// One plane's timing summed over every worker.
#[derive(Debug, Clone, Copy, Default)]
struct PlaneTotals {
    blocked: Duration,
    submit: Duration,
    device_busy: Option<Duration>,
    frames: u64,
}

impl PlaneTotals {
    fn add(&mut self, plane: PlaneTiming) {
        self.blocked += plane.blocked;
        self.submit += plane.submit;
        self.frames += plane.frames;

        if let Some(busy) = plane.device_busy {
            let total = self.device_busy.unwrap_or_default() + busy;
            self.device_busy = Some(total);
        }
    }
}

/// Every plane's totals over all workers, in luma, chroma, yuv order.
fn plane_totals(workers: &[WorkerTiming]) -> Vec<(&'static str, PlaneTotals)> {
    let mut totals: Vec<(&'static str, PlaneTotals)> = Vec::new();

    for worker in workers {
        for (name, plane) in worker.planes.planes() {
            let position = totals.iter().position(|(existing, _)| *existing == name);
            match position {
                Some(index) => totals[index].1.add(plane),
                None => {
                    let mut fresh = PlaneTotals::default();
                    fresh.add(plane);
                    totals.push((name, fresh));
                },
            }
        }
    }

    let order = ["luma", "chroma", "yuv"];
    totals.sort_by_key(|(name, _)| order.iter().position(|known| known == name));
    totals
}

fn per_frame_ms(total: Duration, frames: u64) -> f64 {
    let frames = frames.max(1) as f64;
    total.as_secs_f64() * 1000.0 / frames
}

/// Formats device time as a share of the time `streams` streams had over `wall`.
pub fn busy_share(busy: Option<Duration>, wall: Duration, streams: usize) -> String {
    let Some(busy) = busy else {
        return "n/a".to_string();
    };

    let stream_count = streams.max(1) as f64;
    let seconds = (wall.as_secs_f64() * stream_count).max(f64::EPSILON);
    let share = 100.0 * busy.as_secs_f64() / seconds;
    format!("{share:.1}%")
}

/// How many streams the device time is shared over.
///
/// Workers on a shared queue take turns on one stream. Otherwise each worker has its own.
fn busy_streams(accelerator: Option<Accelerator>, worker_count: usize) -> usize {
    let separate = accelerator.is_some_and(|accelerator| !accelerator.shares_device_queue());

    if separate { worker_count } else { 1 }
}

fn push_line(report: &mut String, text: &str) {
    report.push_str(text);
    report.push('\n');
}

/// Formats the report the CLI prints after a run with `--timing-report`.
pub fn format_report(summary: &RunSummary) -> String {
    let mut report = String::new();
    let seconds = summary.wall.as_secs_f64();
    let fps = summary.frames as f64 / seconds.max(f64::EPSILON);
    let worker_count = summary.workers.len();
    let worker_noun = if worker_count == 1 { "worker" } else { "workers" };
    let accelerator = summary.workers.iter().find_map(|worker| worker.accelerator);
    let streams = busy_streams(accelerator, worker_count);

    let heading = format!(
        "Timing report, {worker_count} {worker_noun}, {} frames in {seconds:.2} s ({fps:.2} fps)",
        summary.frames,
    );
    push_line(&mut report, &heading);

    let columns = format!(
        "  {:<8}{:>16}{:>16}{:>10}",
        "plane", "blocked/frame", "submit/frame", "gpu busy"
    );
    push_line(&mut report, &columns);

    let totals = plane_totals(&summary.workers);
    for (name, plane) in totals {
        let blocked = per_frame_ms(plane.blocked, plane.frames);
        let submit = per_frame_ms(plane.submit, plane.frames);
        let busy = busy_share(plane.device_busy, summary.wall, streams);
        let row = format!("  {name:<8}{blocked:>13.2} ms{submit:>13.2} ms{busy:>10}");
        push_line(&mut report, &row);
    }

    for (index, worker) in summary.workers.iter().enumerate() {
        let denoising = worker.denoising().as_secs_f64();
        let idle = worker.idle.as_secs_f64();
        let setup = worker.setup.as_secs_f64();
        let other = worker.unaccounted().as_secs_f64();
        let wall = worker.wall.as_secs_f64();
        let row = format!(
            "  worker {index}: denoising {denoising:.2} s, waiting for scenes {idle:.2} s, setup {setup:.2} s, \
             other {other:.2} s, wall {wall:.2} s",
        );
        push_line(&mut report, &row);
    }

    let legend = format!(
        "Blocked and submit are host time per frame. GPU busy is device time over wall time, from \
         timestamps on 1 in {DEVICE_SAMPLE_EVERY} frames.",
    );
    push_line(&mut report, &legend);

    if let Some(accelerator) = accelerator {
        let queue = if accelerator.shares_device_queue() {
            format!("On {accelerator} every worker submits to the device's one queue.")
        } else {
            format!(
                "On {accelerator} each worker submits on its own stream, so GPU busy is the mean over \
                 the workers' streams."
            )
        };
        push_line(&mut report, &queue);
    }

    report
}

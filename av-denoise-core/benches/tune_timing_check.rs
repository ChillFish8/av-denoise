#[expect(
    dead_code,
    reason = "shares the kernel bench fixtures, of which this binary uses a few"
)]
mod kernels;

use std::cmp::Ordering;
use std::time::{Duration, Instant};

use av_denoise_core::bench_api::Device;
use av_denoise_core::bench_api::tune_sites::{
    CollabSite,
    CollabVariant,
    RegulariseCandidate,
    WindowCandidate,
};
use av_denoise_core::bench_api::tune_timing::{calibrate, gpu_sampler, rewarm, sync};
use av_denoise_core::tune::{
    DeviceLimits,
    HYSTERESIS,
    MIN_CANDIDATE_TIME,
    MIN_SAMPLES,
    Sampler,
    Samples,
    SiteCandidate,
    VariantSite,
    fit_candidates,
    median,
};
use clap::Parser;
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use kernels::collab_fused::CollabFusedBench;
use kernels::fused_window::{WindowBench, WindowKernel};
use kernels::mv_regularise::MvRegulariseBench;

/// How long the device is kept busy before wall-clock batches start.
const HEAT: Duration = Duration::from_secs(1);
/// The wall-clock length one batch is calibrated to.
const BATCH_TARGET: Duration = Duration::from_millis(150);
/// Batches timed per candidate in the wall-clock ranking.
const BATCHES: usize = 7;
/// Launches queued between syncs while heating and calibrating.
const PROBE_LAUNCHES: u32 = 8;
/// Probes whose median sizes a wall-clock batch.
const PROBE_RUNS: usize = 3;

#[derive(clap::Parser, Debug)]
#[command(
    about = "Ranks tuner candidates with the tuner's device timing and with wall-clock batches",
    long_about = None
)]
struct Cli {
    /// GPU device to bind to, one of `default`, `discrete[:N]`, `integrated[:N]`, `virtual[:N]` or `cpu`.
    #[arg(long, default_value = "default")]
    device: Device,

    /// Swallowed, since cargo passes this when invoking the bench binary.
    #[arg(long, hide = true)]
    bench: bool,

    /// Times the whole comparison this many times.
    #[arg(long, default_value_t = 3)]
    repeats: usize,
}

/// One candidate's medians under both methods, in nanoseconds per launch.
struct Row {
    name: &'static str,
    tuner_ns: f64,
    wall_ns: f64,
}

fn main() {
    let cli = Cli::parse();

    let device = cli.device.to_wgpu().expect("wgpu device conversion failed");
    println!("device: {device:?}");

    run_repeats::<cubecl::wgpu::WgpuRuntime>(&device, cli.repeats);
}

fn run_repeats<R: Runtime>(device: &R::Device, repeats: usize) {
    let client = R::client(device);
    let limits = DeviceLimits::from_client(&client);
    let mut disagreements = 0;

    for repeat in 1..=repeats {
        println!();
        println!("=== repeat {repeat} ===");

        let window = window_site(&client);
        disagreements += compare_site(&window, &limits, "nlm_window pair luma 1080p");

        let collab = collab_site(&client);
        disagreements += compare_site(&collab, &limits, "collab_fused luma 1080p");

        let regularise = regularise_site(&client);
        disagreements += compare_site(&regularise, &limits, "nl4d_mv_regularise 1080p");
    }

    println!();
    println!("site comparisons that disagree: {disagreements}");
}

fn window_site<R: Runtime>(client: &ComputeClient<R>) -> impl VariantSite<R, Candidate = WindowCandidate> {
    let bench = WindowBench {
        client: client.clone(),
        channels: 1,
        channel_name: "luma",
        kernel: WindowKernel::Pair,
        candidate: WindowCandidate::baseline(),
    };
    let (site, _) = bench.prepare();

    site
}

fn regularise_site<R: Runtime>(
    client: &ComputeClient<R>,
) -> impl VariantSite<R, Candidate = RegulariseCandidate> {
    let bench = MvRegulariseBench {
        client: client.clone(),
        candidate: RegulariseCandidate::baseline(),
        neighbours: 4,
        static_field: false,
    };
    let (site, _) = bench.prepare();

    site
}

/// The collab site over a 1080p luma ring, built from the kernel bench's buffers.
fn collab_site<R: Runtime>(client: &ComputeClient<R>) -> CollabSite<R, f32> {
    let bench = CollabFusedBench {
        client: client.clone(),
        channels: 1,
        channel_name: "luma",
        split_mv: false,
        noise_curve: false,
        strength_map: false,
        pooled: false,
        candidate: CollabVariant::BASELINE,
    };
    let args = bench.prepare();

    bench.site(&args)
}

/// Ranks every fitting candidate of `site` both ways, prints the table and returns 1 on disagreement.
fn compare_site<R: Runtime, S: VariantSite<R>>(site: &S, limits: &DeviceLimits, label: &str) -> usize {
    let allowed = fit_candidates(site, limits).expect("the baseline must fit");
    let outputs = site.scratch_outputs();

    let tuner_ns = rank_with_tuner(site, &allowed, &outputs);
    let wall_ns = rank_with_wall_clock(site, &allowed, &outputs);

    let rows: Vec<Row> = allowed
        .iter()
        .enumerate()
        .map(|(index, candidate)| Row {
            name: candidate.name(),
            tuner_ns: tuner_ns[index],
            wall_ns: wall_ns[index],
        })
        .collect();

    print_table(label, &rows)
}

/// Median ns per launch of each candidate, sampled the way the tuner does.
fn rank_with_tuner<R: Runtime, S: VariantSite<R>>(
    site: &S,
    allowed: &[S::Candidate],
    outputs: &S::Outputs,
) -> Vec<f64> {
    rewarm(site, outputs).expect("re-warm failed");
    let launches = calibrate(site, outputs).expect("calibration failed");
    println!("  tuner sample: {launches} launches");

    let mut sampler = gpu_sampler(site, allowed, outputs, launches);
    let mut samples: Vec<Samples> = vec![Samples::default(); allowed.len()];

    loop {
        let settled = samples
            .iter()
            .all(|taken| taken.count() >= MIN_SAMPLES && taken.spent() >= MIN_CANDIDATE_TIME);
        if settled {
            break;
        }

        for (index, taken) in samples.iter_mut().enumerate() {
            let timing = sampler.sample(index).expect("tuner sample failed");
            taken.push(timing);
        }
    }

    samples
        .iter()
        .map(|taken| taken.median().as_nanos() as f64)
        .collect()
}

/// Median ns per launch of each candidate, from interleaved wall-clock batches after heating.
fn rank_with_wall_clock<R: Runtime, S: VariantSite<R>>(
    site: &S,
    allowed: &[S::Candidate],
    outputs: &S::Outputs,
) -> Vec<f64> {
    let client = site.client();
    let baseline = site.baseline();
    let heat_started = Instant::now();

    while heat_started.elapsed() < HEAT {
        for _ in 0..PROBE_LAUNCHES {
            site.launch(baseline, outputs);
        }

        sync(client).expect("sync failed");
    }

    let batch_sizes: Vec<u32> = allowed
        .iter()
        .map(|&candidate| calibrate_batch_launches(site, candidate, outputs))
        .collect();
    println!("  wall batch launches: {batch_sizes:?}");

    let mut samples: Vec<Vec<Duration>> = vec![Vec::new(); allowed.len()];

    for _ in 0..BATCHES {
        for (index, &candidate) in allowed.iter().enumerate() {
            let launches = batch_sizes[index];

            sync(client).expect("sync failed");
            let started = Instant::now();

            for _ in 0..launches {
                site.launch(candidate, outputs);
            }

            sync(client).expect("sync failed");

            let elapsed = started.elapsed();
            samples[index].push(elapsed / launches);
        }
    }

    samples
        .iter()
        .map(|taken| median(taken).as_nanos() as f64)
        .collect()
}

/// The launches that make one wall-clock batch of `candidate` last about [BATCH_TARGET].
///
/// The size comes from the median of [PROBE_RUNS] probes of [PROBE_LAUNCHES] launches.
fn calibrate_batch_launches<R: Runtime, S: VariantSite<R>>(
    site: &S,
    candidate: S::Candidate,
    outputs: &S::Outputs,
) -> u32 {
    let client = site.client();
    let mut probes = Vec::with_capacity(PROBE_RUNS);

    for _ in 0..PROBE_RUNS {
        sync(client).expect("sync failed");
        let started = Instant::now();

        for _ in 0..PROBE_LAUNCHES {
            site.launch(candidate, outputs);
        }

        sync(client).expect("sync failed");

        probes.push(started.elapsed());
    }

    let probe = median(&probes);
    let per_launch = probe.as_secs_f64() / f64::from(PROBE_LAUNCHES);
    let needed = BATCH_TARGET.as_secs_f64() / per_launch.max(1e-9);
    let launches = needed.ceil() as u32;

    launches.max(1)
}

/// Prints one site's table and returns 1 when the two methods pick different winners beyond
/// [HYSTERESIS].
fn print_table(label: &str, rows: &[Row]) -> usize {
    let tuner_best = fastest(rows, |row| row.tuner_ns);
    let wall_best = fastest(rows, |row| row.wall_ns);

    println!();
    println!("{label}");
    println!(
        "  {:<12} {:>12} {:>12} {:>8} {:>7} {:>7}",
        "candidate", "A tuner ns", "B wall ns", "A/B", "rank A", "rank B"
    );

    for (index, row) in rows.iter().enumerate() {
        let ratio = row.tuner_ns / row.wall_ns;
        let rank_a = rank_of(rows, index, |row| row.tuner_ns);
        let rank_b = rank_of(rows, index, |row| row.wall_ns);

        println!(
            "  {:<12} {:>12.0} {:>12.0} {:>8.3} {:>7} {:>7}",
            row.name, row.tuner_ns, row.wall_ns, ratio, rank_a, rank_b
        );
    }

    if tuner_best == wall_best {
        println!("  winner: {} (both methods)", rows[tuner_best].name);
        return 0;
    }

    let gap_a = rows[wall_best].tuner_ns / rows[tuner_best].tuner_ns - 1.0;
    let gap_b = rows[tuner_best].wall_ns / rows[wall_best].wall_ns - 1.0;
    let within = gap_a < HYSTERESIS && gap_b < HYSTERESIS;

    let verdict = if within { "within hysteresis" } else { "DISAGREE" };

    println!(
        "  winner A {}, winner B {}, gap under A {:.2}%, gap under B {:.2}% -> {}",
        rows[tuner_best].name,
        rows[wall_best].name,
        gap_a * 100.0,
        gap_b * 100.0,
        verdict,
    );

    usize::from(!within)
}

fn fastest(rows: &[Row], key: impl Fn(&Row) -> f64) -> usize {
    (0..rows.len())
        .min_by(|&left, &right| {
            let left_key = key(&rows[left]);
            let right_key = key(&rows[right]);

            left_key.partial_cmp(&right_key).unwrap_or(Ordering::Equal)
        })
        .expect("a site has at least one candidate")
}

/// The 1-based rank of the row at `index` when sorted fastest first.
fn rank_of(rows: &[Row], index: usize, key: impl Fn(&Row) -> f64) -> usize {
    let own = key(&rows[index]);
    let faster = rows.iter().filter(|row| key(row) < own).count();

    faster + 1
}

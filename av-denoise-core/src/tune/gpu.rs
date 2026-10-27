use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use cubecl::prelude::*;
use cubecl::profile::ProfileDuration;
use cubecl::server::{Handle, ProfileError};

use crate::engine::{BLOCK_1D, MAX_GRID_1D};
use crate::nlmeans::kernels::gpu_zero_buffers;
use crate::tune::decide::{SampleTiming, Sampler};
use crate::tune::fidelity::FidelityClass;
use crate::tune::site::{SiteCandidate, VariantSite};

/// Binds the first `len` elements of `handle` as a kernel array argument.
pub(crate) fn bind(handle: &Handle, len: usize) -> BufferArg {
    let owned = handle.clone();

    // SAFETY: every caller passes at most the element count the handle was allocated for.
    unsafe { BufferArg::from_raw_parts(owned, len) }
}

/// A new buffer of `len` 32-bit words on `client`, zeroed on the GPU.
pub(crate) fn zeroed_words(client: &Client, len: usize) -> Handle {
    let words = len.max(1);
    let handle = client.empty(words * size_of::<u32>());
    zero_words(client, &handle, words);

    handle
}

/// Zeroes the first `len` 32-bit words of `handle` with the zero kernel.
///
/// A zero bit pattern is zero for both `f32` and `i32`, so this clears either kind of buffer.
pub(crate) fn zero_words(client: &Client, handle: &Handle, len: usize) {
    // The kernel also clears two per-pixel buffers, and a zero length leaves these untouched.
    let unused_sum = client.empty(size_of::<f32>());
    let unused_max = client.empty(size_of::<f32>());

    zero_accumulators(client, handle, &unused_sum, &unused_max, len, 0);
}

/// Zeroes `accum_len` words of `accum` and `pixels` words of each per-pixel buffer in one launch.
///
/// `accum_len` must be at least `pixels`.
pub(crate) fn zero_accumulators(
    client: &Client,
    accum: &Handle,
    weight_sum: &Handle,
    max_weight: &Handle,
    accum_len: usize,
    pixels: usize,
) {
    let word_count = accum_len as u32;
    let grid = word_count.div_ceil(BLOCK_1D).clamp(1, MAX_GRID_1D);
    let total_threads = grid * BLOCK_1D;
    let cube_count = CubeCount::new_1d(grid);
    let cube_dim = CubeDim::new_1d(BLOCK_1D);

    let accum_arg = bind(accum, accum_len);
    let weight_sum_arg = bind(weight_sum, pixels.max(1));
    let max_weight_arg = bind(max_weight, pixels.max(1));

    // SAFETY: each handle holds at least the element count bound above, and the kernel only
    // writes below `accum_len` and `pixels`.
    unsafe {
        gpu_zero_buffers::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            accum_arg,
            weight_sum_arg,
            max_weight_arg,
            word_count,
            pixels as u32,
            total_threads,
        );
    }
}

/// Reads the first `len` values of `handle` back to the host.
fn read_values<T: CubePrimitive + CubeElement>(
    client: &Client,
    handle: &Handle,
    len: usize,
) -> Result<Vec<T>, anyhow::Error> {
    let owned = handle.clone();
    let bytes = client
        .read_one(owned)
        .map_err(|error| anyhow::anyhow!("readback failed: {error:?}"))?;
    let values = T::from_bytes(&bytes);
    let prefix = values
        .get(..len)
        .ok_or_else(|| anyhow::anyhow!("readback returned {} values, expected {len}", values.len()))?;

    Ok(prefix.to_vec())
}

/// Reads the first `len` `f32` values of `handle` back to the host.
pub(crate) fn read_f32(client: &Client, handle: &Handle, len: usize) -> Result<Vec<f32>, anyhow::Error> {
    read_values(client, handle, len)
}

/// Reads the first `len` `i32` values of `handle` back to the host.
pub(crate) fn read_i32(client: &Client, handle: &Handle, len: usize) -> Result<Vec<i32>, anyhow::Error> {
    read_values(client, handle, len)
}

/// A profiled batch whose timing failed.
#[derive(Debug, thiserror::Error)]
#[error("profiling {site} failed: {reason}")]
pub(crate) struct ProfileFailure {
    pub(crate) site: &'static str,
    pub(crate) reason: String,
    /// Whether the launches were submitted before the failure.
    pub(crate) work_ran: bool,
}

/// The wgpu runtime's default for the dispatches one compute pass holds.
const WGPU_DEFAULT_MAX_TASKS: u32 = 32;

/// The most launches one profile holds.
///
/// The wgpu runtime timestamps only the first compute pass of a profile, and it ends a pass after
/// `CUBECL_WGPU_MAX_TASKS` dispatches, 32 by default. A profile therefore holds the lower of 32 and
/// that variable, read once.
pub(crate) fn profile_chunk_limit() -> u32 {
    static CHUNK: OnceLock<u32> = OnceLock::new();

    *CHUNK.get_or_init(|| {
        let configured = std::env::var("CUBECL_WGPU_MAX_TASKS");
        let parsed = configured.ok().and_then(|value| value.parse::<u32>().ok());

        match parsed {
            Some(max_tasks) => max_tasks.clamp(1, WGPU_DEFAULT_MAX_TASKS),
            None => WGPU_DEFAULT_MAX_TASKS,
        }
    })
}

/// Times `launches` back-to-back launches of `candidate` into `outputs` with device timestamps.
///
/// The launches run in profiles of a few launches each, whose device times are summed. Each site
/// launch must be a single dispatch, since a launch of several dispatches is under-timed. Each
/// profile holds the device for its launches. Timing can fail before any work is submitted or after
/// it, and the failure says which.
pub(crate) fn profile_launches<S: VariantSite>(
    site: &S,
    candidate: S::Candidate,
    outputs: &S::Outputs,
    launches: u32,
) -> Result<Duration, ProfileFailure> {
    let queued = submit_launches(site, candidate, outputs, launches)?;
    resolve_launches(queued)
}

/// The profiles of one sample's launches, submitted and not yet resolved.
struct QueuedLaunches {
    site: &'static str,
    profiles: Vec<ProfileDuration>,
}

/// Submits `launches` launches of `candidate` into `outputs` in profiles of a few launches each.
///
/// A failure says whether any of these launches were submitted before it.
fn submit_launches<S: VariantSite>(
    site: &S,
    candidate: S::Candidate,
    outputs: &S::Outputs,
    launches: u32,
) -> Result<QueuedLaunches, ProfileFailure> {
    let chunk_limit = profile_chunk_limit();
    let work_ran = AtomicBool::new(false);
    let mut profiles = Vec::new();
    let mut remaining = launches;

    while remaining > 0 {
        let chunk = remaining.min(chunk_limit);
        remaining -= chunk;

        let profiled = profile_chunk(site, candidate, outputs, chunk, &work_ran);
        let duration = profiled.map_err(|error| ProfileFailure {
            site: S::Candidate::SITE,
            reason: format!("{error:?}"),
            work_ran: work_ran.load(Ordering::Relaxed),
        })?;
        profiles.push(duration);
    }

    Ok(QueuedLaunches {
        site: S::Candidate::SITE,
        profiles,
    })
}

/// Waits for the device times of `queued` and sums them.
fn resolve_launches(queued: QueuedLaunches) -> Result<Duration, ProfileFailure> {
    let mut elapsed = Duration::ZERO;

    for duration in queued.profiles {
        let resolving = duration.resolve();
        let resolved = cubecl::future::block_on(resolving);
        let Some(ticks) = resolved else {
            return Err(ProfileFailure {
                site: queued.site,
                reason: "the device timer returned no measurement".to_string(),
                work_ran: true,
            });
        };

        elapsed += ticks.duration();
    }

    Ok(elapsed)
}

/// Submits `launches` launches inside one profile, setting `work_ran` before the first.
fn profile_chunk<S: VariantSite>(
    site: &S,
    candidate: S::Candidate,
    outputs: &S::Outputs,
    launches: u32,
    work_ran: &AtomicBool,
) -> Result<ProfileDuration, ProfileError> {
    let client = site.client();
    let owned_site = site.clone();
    let owned_outputs = outputs.clone();
    let work = move || {
        work_ran.store(true, Ordering::Relaxed);

        for _ in 0..launches {
            owned_site.launch(candidate, &owned_outputs);
        }
    };

    let profiled = client.profile(work, S::Candidate::SITE);
    let (_, duration) = profiled?;

    Ok(duration)
}

/// Waits for every launch queued on `client`.
pub(crate) fn sync(client: &Client) -> Result<(), anyhow::Error> {
    let pending = client.sync();
    let synced = cubecl::future::block_on(pending);

    synced.map_err(|error| anyhow::anyhow!("device sync failed: {error:?}"))
}

/// Samples candidates of one site with device timestamps, `launches` launches per sample.
///
/// A sample the device timer reports as taking no time is an error, since a calibrated sample
/// always takes some. Time spent compiling is left out of its elapsed time.
pub(crate) struct GpuSampler<'a, S: VariantSite> {
    site: &'a S,
    candidates: &'a [S::Candidate],
    outputs: &'a S::Outputs,
    launches: u32,
    started: Instant,
    excluded: Duration,
}

impl<'a, S: VariantSite> GpuSampler<'a, S> {
    pub(crate) fn new(
        site: &'a S,
        candidates: &'a [S::Candidate],
        outputs: &'a S::Outputs,
        launches: u32,
    ) -> Self {
        Self {
            site,
            candidates,
            outputs,
            launches,
            started: Instant::now(),
            excluded: Duration::ZERO,
        }
    }

    /// The timing of one sample that took `total`, failing when the device timer reports no time.
    fn timing(&self, candidate: S::Candidate, total: Duration) -> Result<SampleTiming, anyhow::Error> {
        if total.is_zero() {
            anyhow::bail!("the device timer reported no time for {}", candidate.name());
        }

        Ok(SampleTiming {
            total,
            launches: self.launches,
        })
    }
}

impl<S: VariantSite> Sampler for GpuSampler<'_, S> {
    fn sample(&mut self, index: usize) -> Result<SampleTiming, anyhow::Error> {
        let candidate = self.candidates[index];
        let total = profile_launches(self.site, candidate, self.outputs, self.launches)?;

        self.timing(candidate, total)
    }

    /// Queues the profiles of every sample before it resolves any, so the device never idles between them.
    fn sample_batch(&mut self, index: usize, count: usize) -> Result<Vec<SampleTiming>, anyhow::Error> {
        let candidate = self.candidates[index];

        let mut queued = Vec::with_capacity(count);
        for _ in 0..count {
            let submitted = submit_launches(self.site, candidate, self.outputs, self.launches)?;
            queued.push(submitted);
        }

        let mut timings = Vec::with_capacity(count);
        for submitted in queued {
            let total = resolve_launches(submitted)?;
            let timing = self.timing(candidate, total)?;
            timings.push(timing);
        }

        Ok(timings)
    }

    fn elapsed(&self) -> Duration {
        let wall = self.started.elapsed();
        wall.saturating_sub(self.excluded)
    }

    fn exclude(&mut self, spent: Duration) {
        self.excluded += spent;
    }

    fn candidate_name(&self, index: usize) -> String {
        let candidate = self.candidates[index];
        candidate.name().to_string()
    }

    fn fidelity(&self, index: usize) -> FidelityClass {
        let candidate = self.candidates[index];
        let fidelity = candidate.fidelity();
        fidelity.class()
    }
}

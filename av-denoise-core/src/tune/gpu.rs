use std::marker::PhantomData;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use cubecl::prelude::*;
use cubecl::profile::ProfileDuration;
use cubecl::server::{Handle, ProfileError};

use crate::nlmeans::kernels::gpu_zero_buffers;
use crate::nlmeans::{BLOCK_1D, MAX_GRID_1D};
use crate::tune::decide::{SampleTiming, Sampler};
use crate::tune::site::{SiteCandidate, VariantSite};

/// Binds the first `len` elements of `handle` as a kernel array argument.
pub(crate) fn bind<R: Runtime>(handle: &Handle, len: usize) -> ArrayArg<R> {
    let owned = handle.clone();

    // SAFETY: every caller passes at most the element count the handle was allocated for.
    unsafe { ArrayArg::from_raw_parts(owned, len) }
}

/// Zeroes the first `len` 32-bit words of `handle` with the zero kernel.
///
/// A zero bit pattern is zero for both `f32` and `i32`, so this clears either kind of buffer.
pub(crate) fn zero_words<R: Runtime>(client: &ComputeClient<R>, handle: &Handle, len: usize) {
    // The kernel also clears two per-pixel buffers, and a zero length leaves these untouched.
    let unused_sum = client.empty(size_of::<f32>());
    let unused_max = client.empty(size_of::<f32>());

    zero_accumulators(client, handle, &unused_sum, &unused_max, len, 0);
}

/// Zeroes `accum_len` words of `accum` and `pixels` words of each per-pixel buffer in one launch.
///
/// `accum_len` must be at least `pixels`.
pub(crate) fn zero_accumulators<R: Runtime>(
    client: &ComputeClient<R>,
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

    let accum_arg = bind::<R>(accum, accum_len);
    let weight_sum_arg = bind::<R>(weight_sum, pixels.max(1));
    let max_weight_arg = bind::<R>(max_weight, pixels.max(1));

    // SAFETY: each handle holds at least the element count bound above, and the kernel only
    // writes below `accum_len` and `pixels`.
    unsafe {
        gpu_zero_buffers::launch_unchecked::<R>(
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
fn read_values<R: Runtime, T: CubePrimitive + CubeElement>(
    client: &ComputeClient<R>,
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
pub(crate) fn read_f32<R: Runtime>(
    client: &ComputeClient<R>,
    handle: &Handle,
    len: usize,
) -> Result<Vec<f32>, anyhow::Error> {
    read_values(client, handle, len)
}

/// Reads the first `len` `i32` values of `handle` back to the host.
pub(crate) fn read_i32<R: Runtime>(
    client: &ComputeClient<R>,
    handle: &Handle,
    len: usize,
) -> Result<Vec<i32>, anyhow::Error> {
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
fn profile_chunk_limit() -> u32 {
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
pub(crate) fn profile_launches<R: Runtime, S: VariantSite<R>>(
    site: &S,
    candidate: S::Candidate,
    outputs: &S::Outputs,
    launches: u32,
) -> Result<Duration, ProfileFailure> {
    let chunk_limit = profile_chunk_limit();
    let work_ran = AtomicBool::new(false);
    let mut pending = Vec::new();
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
        pending.push(duration);
    }

    let mut elapsed = Duration::ZERO;

    for duration in pending {
        let resolving = duration.resolve();
        let ticks = cubecl::future::block_on(resolving);
        elapsed += ticks.duration();
    }

    Ok(elapsed)
}

/// Submits `launches` launches inside one profile, setting `work_ran` before the first.
fn profile_chunk<R: Runtime, S: VariantSite<R>>(
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
pub(crate) fn sync<R: Runtime>(client: &ComputeClient<R>) -> Result<(), anyhow::Error> {
    let pending = client.sync();
    let synced = cubecl::future::block_on(pending);

    synced.map_err(|error| anyhow::anyhow!("device sync failed: {error:?}"))
}

/// Samples candidates of one site with device timestamps, `launches` launches per sample.
///
/// A sample the device timer reports as taking no time is an error, since a calibrated sample
/// always takes some.
pub(crate) struct GpuSampler<'a, R: Runtime, S: VariantSite<R>> {
    site: &'a S,
    candidates: &'a [S::Candidate],
    outputs: &'a S::Outputs,
    launches: u32,
    started: Instant,
    runtime: PhantomData<R>,
}

impl<'a, R: Runtime, S: VariantSite<R>> GpuSampler<'a, R, S> {
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
            runtime: PhantomData,
        }
    }
}

impl<R: Runtime, S: VariantSite<R>> Sampler for GpuSampler<'_, R, S> {
    fn sample(&mut self, index: usize) -> Result<SampleTiming, anyhow::Error> {
        let candidate = self.candidates[index];
        let total = profile_launches(self.site, candidate, self.outputs, self.launches)?;

        if total.is_zero() {
            anyhow::bail!("the device timer reported no time for {}", candidate.name());
        }

        Ok(SampleTiming {
            total,
            launches: self.launches,
        })
    }

    fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

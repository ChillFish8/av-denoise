use cubecl::prelude::*;
use cubecl::server::Handle;

use super::align::StorageAlign;
use super::kernels::{
    nlm_noise_partial,
    nlm_noise_reduce,
    nlm_temporal_noise_stats,
    nlm_temporal_stats_zero,
};
use super::{BLOCK_1D, BLOCK_X, BLOCK_Y, MAX_GRID_1D};

/// Inputs for a single-slot Immerkær noise estimate dispatch. Lives
/// only for the duration of one estimate call, so borrows on the
/// denoiser's buffers are sound.
pub(super) struct NoiseCtx<'a> {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub stored_ch: u32,
    pub frame_count: u32,
    pub frame: u32,
    pub slot: u32,
    pub input_buf: &'a Handle,
    pub partials_buf: &'a Handle,
    pub results_buf: &'a Handle,
}

/// Number of `f32` elements the stage-1 partials buffer needs for a
/// `width × height` frame. One cube covers a `BLOCK_X × BLOCK_Y` tile
/// and contributes four lanes.
pub(super) fn partials_len(width: u32, height: u32) -> usize {
    (width.div_ceil(BLOCK_X) * height.div_ceil(BLOCK_Y) * 4) as usize
}

/// Byte stride between ring slots in the stage-1 partials buffer,
/// padded up to the runtime's buffer-binding alignment. Small frames
/// can produce a [`partials_len`] short of one boundary, and `wgpu`
/// rejects a bind-group offset that isn't a multiple of its
/// `min_storage_buffer_offset_alignment` — mirrors
/// [`temporal_stats_slot_stride_bytes`].
pub(super) fn noise_partials_slot_stride_bytes(width: u32, height: u32, align: StorageAlign) -> u64 {
    align.pad_bytes(partials_len(width, height) as u64 * size_of::<f32>() as u64)
}

/// Dispatch both stages of the Immerkær noise estimate for `ctx.frame`,
/// writing the per-channel absolute-mask-response totals into
/// `ctx.results_buf` at `ctx.slot`. The results buffer is sized for
/// `ctx.frame_count` ring slots (four `f32`s each), mirroring the input
/// ring's frame capacity.
pub(super) fn run_noise_estimate<R: Runtime>(
    client: &ComputeClient<R>,
    ctx: &NoiseCtx<'_>,
) -> Result<(), anyhow::Error> {
    let total_input = (ctx.frame_count * ctx.height * ctx.width * ctx.stored_ch) as usize;
    let n_partials = partials_len(ctx.width, ctx.height);
    let total_results = (ctx.frame_count * 4) as usize;
    let stored_ch = ctx.stored_ch as usize;

    unsafe {
        nlm_noise_partial::launch_unchecked::<R>(
            client,
            CubeCount::new_2d(ctx.width.div_ceil(BLOCK_X), ctx.height.div_ceil(BLOCK_Y)),
            CubeDim::new_2d(BLOCK_X, BLOCK_Y),
            stored_ch,
            ArrayArg::from_raw_parts(ctx.input_buf.clone(), total_input),
            ArrayArg::from_raw_parts(ctx.partials_buf.clone(), n_partials),
            ctx.frame,
            ctx.width,
            ctx.height,
            ctx.channels,
            BLOCK_X,
            BLOCK_Y,
        );
    }

    let num_partials = (n_partials / 4) as u32;
    unsafe {
        nlm_noise_reduce::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(BLOCK_1D),
            ArrayArg::from_raw_parts(ctx.partials_buf.clone(), n_partials),
            ArrayArg::from_raw_parts(ctx.results_buf.clone(), total_results),
            ctx.slot,
            num_partials,
            BLOCK_1D,
        );
    }

    Ok(())
}

/// Immerkær estimate from the summed absolute mask responses. The
/// interior area excludes the one-pixel border the mask cannot reach.
pub(super) fn sigma_from_abs_sum(abs_sum: f32, width: u32, height: u32) -> f32 {
    let interior = ((width - 2) as f32) * ((height - 2) as f32);
    (std::f32::consts::FRAC_PI_2).sqrt() * abs_sum / (6.0 * interior)
}

/// Per-channel lower quartile of per-cube Immerkær sigma estimates,
/// read straight from one slot's stage-1 partials, `partials[cube_index
/// * 4 + ch]` as the stage-1 kernel lays them out. A cube's own sigma
/// is its interior-pixel overlap's share of `sigma_from_abs_sum`'s
/// formula, the overlap of the cube's `BLOCK_X × BLOCK_Y` tile with
/// the frame's interior rect (`1 <= x <= width - 2`, `1 <= y <= height
/// - 2`). Cubes with no interior overlap are skipped rather than
/// diluting the quartile with a spurious zero. The block-level
/// quartile reads lower than [`sigma_from_abs_sum`]'s frame-wide mean
/// whenever noise is spatially uneven, the more conservative estimate
/// the low chain wants. Channels past `channels` stay 0.
pub(super) fn sigma_block_p25_from_partials(
    partials: &[f32],
    channels: u32,
    width: u32,
    height: u32,
) -> [f32; 3] {
    let cubes_x = width.div_ceil(BLOCK_X);
    let cubes_y = height.div_ceil(BLOCK_Y);
    let channels = channels as usize;

    let mut cube_sigmas: Vec<Vec<f32>> = vec![Vec::new(); channels];

    for cy in 0..cubes_y {
        let tile_y0 = cy * BLOCK_Y;
        let tile_y1 = ((cy + 1) * BLOCK_Y).min(height);
        let overlap_y0 = tile_y0.max(1);
        let overlap_y1 = tile_y1.min(height - 1);

        for cx in 0..cubes_x {
            let tile_x0 = cx * BLOCK_X;
            let tile_x1 = ((cx + 1) * BLOCK_X).min(width);
            let overlap_x0 = tile_x0.max(1);
            let overlap_x1 = tile_x1.min(width - 1);

            if overlap_x1 <= overlap_x0 || overlap_y1 <= overlap_y0 {
                continue;
            }

            let area = ((overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)) as f32;
            let cube_index = (cy * cubes_x + cx) as usize;
            let base = cube_index * 4;

            for (c, sigmas) in cube_sigmas.iter_mut().enumerate() {
                let sum = partials[base + c];
                sigmas.push(std::f32::consts::FRAC_PI_2.sqrt() * sum / (6.0 * area));
            }
        }
    }

    let mut sigma_low = [0.0f32; 3];
    for (c, sigmas) in cube_sigmas.iter_mut().enumerate() {
        if sigmas.is_empty() {
            continue;
        }
        sort_ascending(sigmas);
        sigma_low[c] = lower_quartile(sigmas);
    }
    sigma_low
}

/// Spatial block size for the temporal residual noise-stats kernel.
/// One cube per `TEMPORAL_NOISE_BLOCK × TEMPORAL_NOISE_BLOCK` block.
pub(super) const TEMPORAL_NOISE_BLOCK: u32 = 16;

/// Number of `f32`s in one block's stats record: `sum_d` and `sum_d2`
/// per stored channel, plus one channel-0 lag-1 product.
pub(super) fn temporal_stats_record_len(stored_ch: u32) -> u32 {
    2 * stored_ch + 1
}

/// Block grid covering a `width × height` frame at
/// `TEMPORAL_NOISE_BLOCK` resolution, row-major. Ragged edges are
/// truncated rather than padded, mirroring how the block matcher
/// handles its own ragged last block.
pub(super) fn temporal_stats_blocks(width: u32, height: u32) -> (u32, u32) {
    (
        width.div_ceil(TEMPORAL_NOISE_BLOCK),
        height.div_ceil(TEMPORAL_NOISE_BLOCK),
    )
}

/// Number of `f32`s in one ring slot's stats region.
pub(super) fn temporal_stats_slot_len(width: u32, height: u32, stored_ch: u32) -> usize {
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
    (blocks_x * blocks_y * temporal_stats_record_len(stored_ch)) as usize
}

/// Byte stride between ring slots in the temporal-stats buffer, padded
/// up to the runtime's buffer-binding alignment. Small frames (or a
/// single-channel mode) can produce a [`temporal_stats_slot_len`]
/// short of one boundary, and `wgpu` rejects a bind-group offset that
/// isn't a multiple of its `min_storage_buffer_offset_alignment` —
/// mirrors `MotionCtx::confidence_bytes_per_neighbour`.
pub(super) fn temporal_stats_slot_stride_bytes(
    width: u32,
    height: u32,
    stored_ch: u32,
    align: StorageAlign,
) -> u64 {
    align.pad_bytes(temporal_stats_slot_len(width, height, stored_ch) as u64 * size_of::<f32>() as u64)
}

/// Total byte size of a `frame_count`-slot temporal-stats ring.
pub(super) fn temporal_stats_buf_bytes(
    width: u32,
    height: u32,
    stored_ch: u32,
    frame_count: u32,
    align: StorageAlign,
) -> usize {
    (temporal_stats_slot_stride_bytes(width, height, stored_ch, align) * frame_count as u64) as usize
}

/// Inputs for a single temporal-residual noise-stats dispatch, diffing
/// `slot_new` against `slot_prev` on the input ring.
pub(super) struct TemporalStatsCtx<'a> {
    pub width: u32,
    pub height: u32,
    pub stored_ch: u32,
    pub frame_count: u32,
    pub slot_new: u32,
    pub slot_prev: u32,
    pub input_buf: &'a Handle,
    pub stats_buf: &'a Handle,
    pub align: StorageAlign,
}

/// Dispatch the temporal residual stats kernel for `ctx.slot_new`,
/// writing one block record per `TEMPORAL_NOISE_BLOCK` block into
/// `ctx.stats_buf` at `ctx.slot_new`'s (padded) region. The kernel
/// itself only ever addresses within that one slot's slice, so it
/// needs no awareness of the ring's other slots or the stride padding
/// between them.
pub(super) fn run_temporal_noise_stats<R: Runtime>(
    client: &ComputeClient<R>,
    ctx: &TemporalStatsCtx<'_>,
) -> Result<(), anyhow::Error> {
    let total_input = (ctx.frame_count * ctx.height * ctx.width * ctx.stored_ch) as usize;
    let (blocks_x, blocks_y) = temporal_stats_blocks(ctx.width, ctx.height);
    let slot_len = temporal_stats_slot_len(ctx.width, ctx.height, ctx.stored_ch);
    let stride = temporal_stats_slot_stride_bytes(ctx.width, ctx.height, ctx.stored_ch, ctx.align);
    let stats_slot = ctx.stats_buf.clone().offset_start((ctx.slot_new as u64) * stride);

    unsafe {
        nlm_temporal_noise_stats::launch_unchecked::<R>(
            client,
            CubeCount::new_2d(blocks_x, blocks_y),
            CubeDim::new_2d(TEMPORAL_NOISE_BLOCK, TEMPORAL_NOISE_BLOCK),
            ctx.stored_ch as usize,
            ArrayArg::from_raw_parts(ctx.input_buf.clone(), total_input),
            ArrayArg::from_raw_parts(stats_slot, slot_len),
            ctx.slot_new,
            ctx.slot_prev,
            ctx.width,
            ctx.height,
            ctx.stored_ch,
            TEMPORAL_NOISE_BLOCK,
        );
    }

    Ok(())
}

/// Zero-fill one ring slot's temporal-stats region. Used when a ring
/// slot is a duplicate of its predecessor (stream priming or
/// end-of-stream flush), so aggregation sees it as having no static
/// blocks with measurable noise instead of a fabricated zero-noise
/// reading.
pub(super) fn zero_temporal_stats_slot<R: Runtime>(
    client: &ComputeClient<R>,
    stats_buf: &Handle,
    width: u32,
    height: u32,
    stored_ch: u32,
    slot: u32,
    align: StorageAlign,
) {
    let slot_len = temporal_stats_slot_len(width, height, stored_ch) as u32;
    let stride = temporal_stats_slot_stride_bytes(width, height, stored_ch, align);
    let dst = stats_buf.clone().offset_start((slot as u64) * stride);

    let grid = slot_len.div_ceil(BLOCK_1D).min(MAX_GRID_1D);
    let total_threads = grid * BLOCK_1D;

    unsafe {
        nlm_temporal_stats_zero::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(grid),
            CubeDim::new_1d(BLOCK_1D),
            ArrayArg::from_raw_parts(dst, slot_len as usize),
            slot_len,
            total_threads,
        );
    }
}

/// Reads back exactly one ring slot's temporal-stats region as owned
/// `f32`s. Slices the shared ring handle by byte offset so the
/// transfer is proportional to one slot instead of the whole ring.
#[allow(clippy::too_many_arguments)]
pub(super) fn read_temporal_stats_slot<R: Runtime>(
    client: &ComputeClient<R>,
    stats_buf: &Handle,
    width: u32,
    height: u32,
    stored_ch: u32,
    frame_count: u32,
    slot: u32,
    align: StorageAlign,
) -> Result<Vec<f32>, anyhow::Error> {
    let slot_len_bytes = temporal_stats_slot_len(width, height, stored_ch) as u64 * size_of::<f32>() as u64;
    let stride = temporal_stats_slot_stride_bytes(width, height, stored_ch, align);
    let total_bytes = frame_count as u64 * stride;
    let start = (slot as u64) * stride;
    let end_trim = total_bytes - start - slot_len_bytes;

    let sliced = stats_buf.clone().offset_start(start).offset_end(end_trim);
    let bytes = client
        .read_one(sliced)
        .map_err(|e| anyhow::anyhow!("temporal noise stats readback failed: {e}"))?;
    Ok(f32::from_bytes(&bytes).to_vec())
}

/// One centre slot's aggregated temporal-residual noise measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct TemporalNoiseSample {
    /// Per-channel temporal sigma (median over static blocks), `[0,
    /// 1]` units. Unused entries past the active channel count are 0.
    pub sigma: [f32; 3],
    /// Per-channel temporal sigma at the lower quartile over the same
    /// static-block sigma sets `sigma` medians, `[0, 1]` units. A more
    /// conservative read than `sigma`, meant for consumers where an
    /// over-read is more harmful than an under-read. Unused entries
    /// past the active channel count are 0.
    pub sigma_low: [f32; 3],
    /// Grain autocorrelation: the median, over static blocks with
    /// measurable channel-0 noise, of the channel-0 residual's
    /// horizontal lag-1 correlation.
    pub rho: f32,
    /// Fraction of blocks that passed the static gate.
    pub static_fraction: f32,
}

/// Static gate on a block's mean channel-0 residual, `[0, 1]` units.
/// A block whose average residual exceeds this is treated as moving
/// content rather than measurement noise.
const STATIC_GATE: f32 = 1.5 / 255.0;
/// Minimum channel-0 block sigma for a static block to contribute to
/// the rho median. Below this the block carries too little signal for
/// its lag-1 correlation to mean anything.
const RHO_SIGMA_GATE: f32 = 0.3 / 255.0;
/// Minimum fraction of static blocks for a sample to be trusted.
/// Below this, motion or a scene change dominates the frame and the
/// Immerkær estimate is the only usable signal.
const STATIC_FRACTION_MIN: f32 = 0.05;
/// How far above the mean-gate-passing population's own lower
/// quartile a block's channel-0 sigma may sit before it is treated as
/// displaced texture rather than measurement noise. A block panning
/// over texture can average to zero over a `TEMPORAL_NOISE_BLOCK`
/// window (clearing [`STATIC_GATE`]) while its variance is entirely
/// the shifted texture, not repeatable noise, and that variance runs
/// several times higher than the rest of the frame's blocks. Real
/// measurement noise stays within a much narrower spread across a
/// frame even where its magnitude is genuinely uneven (a dark region
/// reading noisier than a bright one), so this leaves headroom past
/// that legitimate spread while still catching texture outliers. The
/// lower quartile (not the median) is the reference so the filter
/// still works once texture is the majority of gate-passing blocks,
/// as long as a genuinely static minority remains to anchor it. The
/// quartile is computed only over blocks whose own sigma clears
/// [`RHO_SIGMA_GATE`], since letterbox bars and other exactly-static
/// regions read a channel-0 sigma of exactly `0`. Left in, they can
/// dominate the mean-gate-passing population enough to drag the
/// quartile itself to `0`, which would reject every block carrying
/// real noise instead of just the texture outliers. That exclusion
/// only holds up when a genuinely low population still exists to
/// anchor the quartile against. See
/// [`aggregate_temporal_noise_stats`]'s doc for what happens when it
/// doesn't.
const SIGMA_OUTLIER_FACTOR: f32 = 5.0;

/// One mean-gate-passing block's per-channel stats, held only long
/// enough to compute the population's own outlier reference before
/// deciding which blocks are actually static.
struct StaticGateCandidate {
    sigmas: [f32; 3],
    sigma_ch0: f32,
    var_ch0: f32,
    mean0: f32,
    mean_lag: f32,
    n_pairs: f32,
}

/// Aggregates one centre slot's per-block temporal-residual records
/// into a [`TemporalNoiseSample`]. `records` holds exactly one slot's
/// region, laid out block row-major as documented on
/// [`nlm_temporal_noise_stats`]. Returns `None` when static blocks
/// fall below [`STATIC_FRACTION_MIN`] (too much motion to trust), when
/// no static block clears [`RHO_SIGMA_GATE`] (no measurable noise to
/// correlate — the median rho would be undefined — which is also what
/// a zero-filled duplicate slot's records produce), or when the
/// outlier gate below has no way to validate its own ceiling.
///
/// Classifying a block as static takes two passes. The first is the
/// signed mean-residual gate ([`STATIC_GATE`]). It clears a block
/// whose average residual is near zero, but a block panning over
/// texture can clear it too, since a displaced texture pattern
/// averages toward zero over a block just as noise does. The second
/// pass catches what the first lets through. It rejects any
/// mean-gate-passing block whose channel-0 sigma sits far above the
/// passing population's own lower quartile ([`SIGMA_OUTLIER_FACTOR`]),
/// since a panning block's variance comes from the texture it slid
/// across, not from noise shared with the rest of the frame. That
/// quartile itself only looks at blocks clearing [`RHO_SIGMA_GATE`],
/// so an exactly-static population (letterbox bars, for example)
/// cannot drag the ceiling down to `0` and reject every noisy block
/// along with it.
///
/// Excluding those low-sigma blocks has its own failure mode. If
/// every block that clears `RHO_SIGMA_GATE` turns out to be
/// texture, that population sets its own ceiling and lets every one
/// of its own members through, since a value never exceeds a
/// multiple of itself. Nothing in a single block's stats says
/// "texture" versus "noise", so the only corroboration available is
/// whether the surviving population shows any internal spread at
/// all. Real per-block sigma, measured over a finite sample, always
/// carries some variance across blocks, even for physically uniform
/// noise. When low-sigma blocks were excluded and what's left is
/// perfectly uniform, that is the specific signature of a
/// self-selected outlier population with no genuine anchor to check
/// it against, and this function reports `None` rather than a
/// confident, possibly texture-inflated sigma.
pub(super) fn aggregate_temporal_noise_stats(
    records: &[f32],
    channels: u32,
    stored_ch: u32,
    width: u32,
    height: u32,
) -> Option<TemporalNoiseSample> {
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
    let total_blocks = (blocks_x * blocks_y) as usize;
    if total_blocks == 0 {
        return None;
    }

    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let channels = channels as usize;
    let stored_ch = stored_ch as usize;

    let mut candidates = Vec::new();

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let block_index = (by * blocks_x + bx) as usize;
            let rec = &records[block_index * record_len..(block_index + 1) * record_len];

            let block_origin_x = bx * TEMPORAL_NOISE_BLOCK;
            let block_origin_y = by * TEMPORAL_NOISE_BLOCK;
            let block_w = TEMPORAL_NOISE_BLOCK.min(width - block_origin_x);
            let block_h = TEMPORAL_NOISE_BLOCK.min(height - block_origin_y);
            let n = (block_w * block_h) as f32;
            let n_pairs = (block_h * block_w.saturating_sub(1)) as f32;

            let mean0 = rec[0] / n;
            if mean0.abs() >= STATIC_GATE {
                continue;
            }

            let mut sigmas = [0.0f32; 3];
            let mut sigma_ch0 = 0.0f32;
            let mut var_ch0 = 0.0f32;
            for c in 0..channels {
                let mean = rec[c] / n;
                let var = (rec[stored_ch + c] / n - mean * mean).max(0.0);
                let sigma_block = var.sqrt() / std::f32::consts::SQRT_2;
                sigmas[c] = sigma_block;
                if c == 0 {
                    sigma_ch0 = sigma_block;
                    var_ch0 = var;
                }
            }

            let mean_lag = if n_pairs > 0.0 {
                rec[2 * stored_ch] / n_pairs
            } else {
                0.0
            };

            candidates.push(StaticGateCandidate {
                sigmas,
                sigma_ch0,
                var_ch0,
                mean0,
                mean_lag,
                n_pairs,
            });
        }
    }

    // Exactly-static blocks (letterbox bars, a duplicate frame) read
    // sigma_ch0 == 0 and clear STATIC_GATE trivially, so they can
    // dominate the mean-gate-passing population without carrying any
    // measurable noise. Left in this reference set, they drag the
    // quartile itself toward 0, which then rejects every block with
    // real noise instead of just the texture outliers the gate exists
    // to catch. Restricting the reference to blocks that themselves
    // clear RHO_SIGMA_GATE keeps the ceiling anchored to blocks that
    // could plausibly be noise.
    let mut reference_sigma_ch0: Vec<f32> = candidates
        .iter()
        .map(|c| c.sigma_ch0)
        .filter(|&sigma| sigma > RHO_SIGMA_GATE)
        .collect();
    sort_ascending(&mut reference_sigma_ch0);

    // Excluding low-sigma blocks from the reference set removes the
    // information they carried (or lacked), and that has a cost. `x <=
    // x * SIGMA_OUTLIER_FACTOR` holds for any `x >= 0`, so if every
    // mean-gate-passing block above RHO_SIGMA_GATE turns out to be a
    // texture-panning block, that population sets its own ceiling and
    // lets every one of them through, exactly the failure the ceiling
    // exists to prevent. There's no feature available here (sigma
    // magnitude, rho) that tells texture and noise apart on its own,
    // so this function instead requires corroboration. Some candidates
    // must have been excluded by RHO_SIGMA_GATE, and the reference
    // population that's left must show genuine internal spread (its
    // lowest and highest readings differ). Real per-block sigma is
    // measured over a finite (`TEMPORAL_NOISE_BLOCK`)² sample, so even
    // physically uniform noise carries some sampling variance across
    // blocks. A reference population with zero spread at all, sitting
    // alongside excluded low-sigma blocks, is the specific signature
    // this function can't resolve, so it reports `None` rather than a
    // confident, potentially texture-inflated sigma. Populations where
    // nothing was excluded (no low-sigma blocks to begin with) skip
    // this check entirely and are trusted directly, unaffected by
    // whether a low-sigma population happens to exist elsewhere in the
    // frame.
    let candidates_were_excluded = candidates.len() > reference_sigma_ch0.len();
    let reference_has_spread = match (reference_sigma_ch0.first(), reference_sigma_ch0.last()) {
        (Some(&lo), Some(&hi)) => hi > lo,
        (None, _) | (_, None) => false,
    };
    if candidates_were_excluded && !reference_has_spread {
        return None;
    }

    let sigma_ceiling = if reference_sigma_ch0.is_empty() {
        0.0
    } else {
        lower_quartile(&reference_sigma_ch0) * SIGMA_OUTLIER_FACTOR
    };

    let mut static_sigmas: Vec<Vec<f32>> = vec![Vec::new(); channels];
    let mut rho_samples = Vec::new();
    let mut static_count = 0usize;

    for candidate in &candidates {
        if candidate.sigma_ch0 > sigma_ceiling {
            continue;
        }
        static_count += 1;

        for (c, sigmas) in static_sigmas.iter_mut().enumerate().take(channels) {
            sigmas.push(candidate.sigmas[c]);
        }

        if candidate.sigma_ch0 > RHO_SIGMA_GATE && candidate.n_pairs > 0.0 {
            let rho = (candidate.mean_lag - candidate.mean0 * candidate.mean0) / candidate.var_ch0;
            rho_samples.push(rho.clamp(0.0, 1.0));
        }
    }

    let static_fraction = static_count as f32 / total_blocks as f32;
    if static_fraction < STATIC_FRACTION_MIN || rho_samples.is_empty() {
        return None;
    }

    let mut sigma = [0.0f32; 3];
    let mut sigma_low = [0.0f32; 3];
    for (c, sigmas) in static_sigmas.iter_mut().enumerate() {
        sort_ascending(sigmas);
        sigma[c] = median(sigmas);
        sigma_low[c] = lower_quartile(sigmas);
    }
    sort_ascending(&mut rho_samples);
    let rho = median(&rho_samples);

    Some(TemporalNoiseSample {
        sigma,
        sigma_low,
        rho,
        static_fraction,
    })
}

/// Sorts `values` ascending in place. A shared first step for
/// [`median`] and [`lower_quartile`], so a caller that needs both
/// statistics from the same set sorts once and passes the same slice
/// to each.
fn sort_ascending(values: &mut [f32]) {
    values.sort_by(|a, b| a.partial_cmp(b).expect("noise stats are never NaN"));
}

/// Median of an ascending-sorted `values`. The average of the two
/// middle elements on an even count. Callers only invoke this on a
/// non-empty slice.
fn median(values: &[f32]) -> f32 {
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// Lower quartile of an ascending-sorted `values`, index `0.25 * (n -
/// 1)` with linear interpolation between the two neighbouring
/// elements. A single-element slice returns that element. Callers
/// only invoke this on a non-empty slice.
fn lower_quartile(values: &[f32]) -> f32 {
    let n = values.len();
    if n == 1 {
        return values[0];
    }
    let idx = 0.25 * (n - 1) as f32;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let t = idx - lo as f32;
        values[lo] + t * (values[hi] - values[lo])
    }
}

/// Piecewise-linear correlation-correction table, `(rho, factor)`
/// points sorted by `rho`. Measured on synthetic correlated-grain
/// sweeps against the clean bench reference. Each factor is how far
/// the quality peak shifts above the true marginal sigma at that
/// grain autocorrelation, relative to the white-noise optimum at the
/// same sigma. At the heaviest measured correlation both XPSNR and
/// SSIM prefer the raised value. Clamped flat past the last measured
/// point.
const CORRELATION_FACTOR_TABLE: [(f32, f32); 4] = [(0.0, 1.0), (0.3, 1.05), (0.5, 1.25), (0.65, 1.45)];

/// Correction factor turning a temporal sigma into an effective sigma
/// that accounts for spatial grain correlation:
/// `sigma_eff = sigma * correlation_factor(rho)`.
pub(super) fn correlation_factor(rho: f32) -> f32 {
    interpolate_table(&CORRELATION_FACTOR_TABLE, rho)
}

/// Piecewise-linear interpolation over a small table of `(x, y)`
/// points sorted by `x`. Clamps `x` to the table's own range first, so
/// the result never extrapolates past the endpoints.
fn interpolate_table(table: &[(f32, f32)], x: f32) -> f32 {
    let x = x.clamp(table[0].0, table[table.len() - 1].0);

    for pair in table.windows(2) {
        let (x0, y0) = pair[0];
        let (x1, y1) = pair[1];
        if x <= x1 {
            if x1 == x0 {
                return y1;
            }
            let t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }

    table[table.len() - 1].1
}

/// Attenuation factor for the spatial noise-floor offset at grain
/// autocorrelation `rho` and integer candidate offset `(dx, dy)`.
/// Spatial candidates share correlated noise with their centre patch,
/// so only a `(1 - rho^d)` fraction of the white-noise floor is
/// independent noise, where `d = sqrt(dx^2 + dy^2)`. The self offset
/// (`dx == dy == 0`) always returns 0, its true distance is zero, so
/// none of the floor is independent noise. `rho <= 0` (no measured
/// correlation, or auto estimation inert) returns 1 for every other
/// offset, reproducing the flat white-noise floor exactly.
pub(super) fn spatial_offset_factor(dx: i32, dy: i32, rho: f32) -> f32 {
    if dx == 0 && dy == 0 {
        return 0.0;
    }
    if rho <= 0.0 {
        return 1.0;
    }
    let d = ((dx * dx + dy * dy) as f32).sqrt();
    1.0 - (d * rho.ln()).exp()
}

/// Number of `f32`s in a `search_radius`-sized spatial-offset LUT.
pub(super) fn spatial_offset_lut_len(search_radius: u32) -> usize {
    let side = (2 * search_radius + 1) as usize;
    side * side
}

/// Builds the per-candidate spatial noise-floor offset LUT for a
/// `search_radius`-sized search window, row-major
/// `(dy+search_radius)*(2*search_radius+1)+(dx+search_radius)`.
/// `lut[q] = noise_offset * spatial_offset_factor(dx, dy, rho)`. Cheap
/// enough to rebuild every `denoise_submit` (at most `(2*8+1)^2 = 289`
/// entries at the largest supported search radius).
pub(super) fn build_spatial_offset_lut(search_radius: u32, rho: f32, noise_offset: f32) -> Vec<f32> {
    let r = search_radius as i32;
    let side = (2 * search_radius + 1) as usize;
    let mut lut = vec![0.0f32; side * side];
    for dy in -r..=r {
        for dx in -r..=r {
            let idx = ((dy + r) as usize) * side + (dx + r) as usize;
            lut[idx] = noise_offset * spatial_offset_factor(dx, dy, rho);
        }
    }
    lut
}

/// EMA weight for the newest frame estimate. Shared by the sigma
/// estimator below and the denoiser's `rho_smoothed` EMA.
pub(super) const EMA_ALPHA: f32 = 0.2;
/// Lower bound on the smoothed sigma in `[0, 1]` units (0.1 in 8-bit
/// terms). Near-zero estimates would blow up the derived strength.
const SIGMA_FLOOR: f32 = 0.1 / 255.0;

/// Per-stream noise state. Smooths per-frame estimates with an EMA so
/// single busy frames cannot spike the strength, and floors the result
/// so near-clean content keeps a usable Welsch normalisation.
#[derive(Debug, Default)]
pub(super) struct NoiseEstimator {
    ema: Option<Vec<f32>>,
}

impl NoiseEstimator {
    /// Folds a new set of per-channel sigma samples into the running
    /// EMA and returns the smoothed result. The first call initialises
    /// the state directly from `sigmas` (no prior estimate to blend
    /// with). Every element is floored at [`SIGMA_FLOOR`] regardless of
    /// how it was produced.
    pub(super) fn update(&mut self, sigmas: &[f32]) -> &[f32] {
        match &mut self.ema {
            None => {
                self.ema = Some(sigmas.iter().map(|&s| s.max(SIGMA_FLOOR)).collect());
            },
            Some(ema) => {
                for (e, &s) in ema.iter_mut().zip(sigmas.iter()) {
                    *e = (EMA_ALPHA * s + (1.0 - EMA_ALPHA) * *e).max(SIGMA_FLOOR);
                }
            },
        }
        self.ema.as_deref().unwrap()
    }

    /// Clears the running estimate. The next [`Self::update`] call
    /// re-initialises from its sample instead of blending with stale
    /// state.
    pub(super) fn reset(&mut self) {
        self.ema = None;
    }

    /// Current smoothed per-channel sigma, or `None` before the first
    /// [`Self::update`] call.
    #[cfg(test)]
    pub(super) fn current(&self) -> Option<&[f32]> {
        self.ema.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_sample_initializes_state() {
        let mut est = NoiseEstimator::default();
        let out = est.update(&[0.05, 0.02]);
        assert_eq!(out, &[0.05, 0.02]);
    }

    #[test]
    fn update_converges_toward_changed_level() {
        let mut est = NoiseEstimator::default();
        est.update(&[0.02]);

        let mut last = 0.0;
        for _ in 0..200 {
            last = est.update(&[0.10])[0];
        }

        assert!(
            (last - 0.10).abs() < 1e-4,
            "expected convergence close to 0.10, got {last}"
        );
    }

    #[test]
    fn update_floors_near_zero_samples() {
        let mut est = NoiseEstimator::default();
        let out = est.update(&[0.0]);
        assert_eq!(out[0], SIGMA_FLOOR);

        let out = est.update(&[0.0]);
        assert_eq!(out[0], SIGMA_FLOOR);
    }

    #[test]
    fn reset_clears_state() {
        let mut est = NoiseEstimator::default();
        est.update(&[0.10]);
        est.reset();

        // After reset, the next update should re-initialise directly
        // from the sample rather than blending with the stale 0.10.
        let out = est.update(&[0.02]);
        assert_eq!(out, &[0.02]);
    }

    #[test]
    fn sigma_from_abs_sum_zero_for_zero_response() {
        // No mask response anywhere in the frame must estimate zero
        // noise regardless of frame size.
        assert_eq!(sigma_from_abs_sum(0.0, 64, 64), 0.0);
    }

    #[test]
    fn partials_len_matches_cube_grid() {
        assert_eq!(partials_len(32, 8), 4); // exactly one BLOCK_X x BLOCK_Y cube
        assert_eq!(partials_len(33, 9), 16); // spills into a 2x2 cube grid
        assert_eq!(partials_len(1920, 1080), 60 * 135 * 4);
    }

    #[test]
    fn noise_partials_slot_stride_bytes_pads_odd_cube_count() {
        // A single BLOCK_X x BLOCK_Y cube, partials_len(32, 8) = 4
        // f32s = 16 bytes, padded up to the next 32-byte multiple.
        assert_eq!(noise_partials_slot_stride_bytes(32, 8, StorageAlign::new(32)), 32);
    }

    #[test]
    fn noise_partials_slot_stride_bytes_aligned_count_unchanged() {
        // A 2x2 cube grid, partials_len(33, 9) = 16 f32s = 64 bytes,
        // already a multiple of 32, so no padding is added.
        assert_eq!(noise_partials_slot_stride_bytes(33, 9, StorageAlign::new(32)), 64);
    }

    /// A `70x20` frame grids into a ragged 3x3 cube layout (`BLOCK_X =
    /// 32`, `BLOCK_Y = 8`). Two full columns/rows plus one partial each,
    /// so every cube's interior-overlap area is distinct. Hand-computed
    /// per-cube interior areas, row-major `[cy][cx]`, summing exactly to
    /// the frame's own interior area `(70-2)*(20-2) = 1224`.
    const RAGGED_CUBE_AREAS: [[f32; 3]; 3] = [[217.0, 224.0, 35.0], [248.0, 256.0, 40.0], [93.0, 96.0, 15.0]];

    /// A uniform per-pixel mask response has the same sigma in every
    /// cube regardless of that cube's area, since `sum = r * area`
    /// cancels the area out of `sigma_from_abs_sum`'s formula. The
    /// block-level lower quartile of identical values is that same
    /// value, and it must equal the frame-wide estimate over the same
    /// total response.
    #[test]
    fn sigma_block_p25_from_partials_uniform_response_matches_frame_wide() {
        let width = 70;
        let height = 20;
        let channels = 1;
        let r = 0.02f32;

        let mut partials = vec![0.0f32; 3 * 3 * 4];
        let mut total_sum = 0.0f32;
        for (cy, row) in RAGGED_CUBE_AREAS.iter().enumerate() {
            for (cx, &area) in row.iter().enumerate() {
                let sum = r * area;
                partials[(cy * 3 + cx) * 4] = sum;
                total_sum += sum;
            }
        }

        let sigma_low = sigma_block_p25_from_partials(&partials, channels, width, height);
        let expected = sigma_from_abs_sum(total_sum, width, height);

        assert!(
            (sigma_low[0] - expected).abs() < expected * 1e-4,
            "uniform response should reproduce the frame-wide estimate {expected}, got {}",
            sigma_low[0]
        );
    }

    /// Nine cubes given a permutation of `1..9` (in `/255` sigma units,
    /// via each cube's own area) so sorting ascending reproduces exactly
    /// `[1, 2, ..., 9] / 255`. `n = 9` puts the lower-quartile index at
    /// `0.25 * 8 = 2.0` exactly, the third-smallest cube.
    #[test]
    fn sigma_block_p25_from_partials_distinct_sums_pick_expected_cube() {
        let width = 70;
        let height = 20;
        let channels = 1;
        let sigma_targets_255 = [5.0f32, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0, 4.0, 6.0];

        let mut partials = vec![0.0f32; 3 * 3 * 4];
        for (cy, row) in RAGGED_CUBE_AREAS.iter().enumerate() {
            for (cx, &area) in row.iter().enumerate() {
                let i = cy * 3 + cx;
                let sigma_target = sigma_targets_255[i] / 255.0;
                let sum = sigma_target * 6.0 * area / std::f32::consts::FRAC_PI_2.sqrt();
                partials[i * 4] = sum;
            }
        }

        let sigma_low = sigma_block_p25_from_partials(&partials, channels, width, height);
        let expected = 3.0 / 255.0;
        assert!(
            (sigma_low[0] - expected).abs() < 1e-4,
            "expected the lower quartile to land on the third-smallest cube sigma {expected}, got {}",
            sigma_low[0]
        );
    }

    #[test]
    fn temporal_stats_blocks_and_slot_len() {
        assert_eq!(temporal_stats_blocks(32, 16), (2, 1));
        assert_eq!(temporal_stats_blocks(33, 17), (3, 2)); // ragged on both axes
        assert_eq!(temporal_stats_record_len(1), 3);
        assert_eq!(temporal_stats_record_len(4), 9);
        assert_eq!(temporal_stats_slot_len(32, 16, 1), 6); // 2 blocks x record_len 3
    }

    #[test]
    fn lower_quartile_odd_count_exact_index() {
        // n = 5, idx = 0.25 * 4 = 1.0 exact, no interpolation needed.
        assert_eq!(lower_quartile(&[1.0, 2.0, 3.0, 4.0, 5.0]), 2.0);
    }

    #[test]
    fn lower_quartile_even_count() {
        // n = 4, idx = 0.25 * 3 = 0.75, between values[0] and values[1].
        let got = lower_quartile(&[1.0, 2.0, 3.0, 4.0]);
        assert!((got - 1.75).abs() < 1e-6, "expected 1.75, got {got}");
    }

    #[test]
    fn lower_quartile_interpolates_at_a_fractional_index() {
        // n = 3, idx = 0.25 * 2 = 0.5, halfway between values[0] and values[1].
        let got = lower_quartile(&[10.0, 20.0, 30.0]);
        assert!((got - 15.0).abs() < 1e-6, "expected 15.0, got {got}");
    }

    #[test]
    fn lower_quartile_single_element_returns_it() {
        assert_eq!(lower_quartile(&[42.0]), 42.0);
    }

    #[test]
    fn aggregate_only_static_block_contributes_to_sigma_and_rho() {
        let width = 32;
        let height = 16;
        let stored_ch = 1;
        let channels = 1;
        let n = 256.0f32;
        let n_pairs = 240.0f32;

        let sigma_target = 4.0 / 255.0;
        let var = 2.0 * sigma_target * sigma_target;
        let rho_target = 0.5f32;
        let sum_d2_static = n * var;
        let sum_lag_static = n_pairs * rho_target * var;

        let mean0_bad = 10.0 / 255.0;
        let sum_d_bad = n * mean0_bad;

        #[rustfmt::skip]
        let records = vec![
            0.0, sum_d2_static, sum_lag_static, // block 0: static
            sum_d_bad, 0.0, 0.0,                // block 1: fails the gate
        ];

        let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
            .expect("one of two blocks passes the static gate, above the 5% floor");

        assert!((sample.static_fraction - 0.5).abs() < 1e-6);
        assert!(
            (sample.sigma[0] - sigma_target).abs() < 1e-4,
            "sigma {} vs target {sigma_target}",
            sample.sigma[0]
        );
        assert!(
            (sample.rho - rho_target).abs() < 1e-4,
            "rho {} vs target {rho_target}",
            sample.rho
        );
    }

    /// Five static blocks with distinct sigmas and (for the four that
    /// clear the rho gate) distinct rhos. Exercises the odd-count sigma
    /// median, the even-count rho median, and the sigma lower quartile
    /// in one pass.
    #[test]
    fn aggregate_median_over_multiple_static_blocks() {
        let width = 16 * 5;
        let height = 16;
        let stored_ch = 1;
        let channels = 1;
        let n = 256.0f32;
        let n_pairs = 240.0f32;

        // Block 0's sigma sits below RHO_SIGMA_GATE, so its rho (0.0,
        // unused) never enters the rho set; the other four all clear it.
        let sigmas_255 = [0.1f32, 2.0, 3.0, 4.0, 5.0];
        let rhos = [0.0f32, 0.1, 0.3, 0.5, 0.7];

        let mut records = Vec::new();
        for (sigma_255, rho) in sigmas_255.iter().zip(rhos.iter()) {
            let sigma = sigma_255 / 255.0;
            let var = 2.0 * sigma * sigma;
            let sum_d2 = n * var;
            let sum_lag = n_pairs * rho * var;
            records.extend_from_slice(&[0.0, sum_d2, sum_lag]);
        }

        let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
            .expect("all five blocks are static");

        assert!((sample.static_fraction - 1.0).abs() < 1e-6);
        assert!(
            (sample.sigma[0] - 3.0 / 255.0).abs() < 1e-4,
            "expected median sigma 3/255 (middle of [0.1,2,3,4,5]), got {}",
            sample.sigma[0]
        );
        assert!(
            (sample.sigma_low[0] - 2.0 / 255.0).abs() < 1e-4,
            "expected lower-quartile sigma 2/255 (index 0.25*4=1.0 of [0.1,2,3,4,5]), got {}",
            sample.sigma_low[0]
        );
        assert!(
            (sample.rho - 0.4).abs() < 1e-4,
            "expected median rho 0.4 over the four blocks clearing the rho gate, got {}",
            sample.rho
        );
    }

    /// Only 1 of 25 blocks is static (4%), below `STATIC_FRACTION_MIN`.
    /// That single block otherwise carries perfectly valid noise, so
    /// this isolates the 5% floor from the empty-rho-set fallback.
    #[test]
    fn aggregate_below_static_floor_returns_none() {
        let width = 16 * 5;
        let height = 16 * 5;
        let stored_ch = 1;
        let channels = 1;
        let n = 256.0f32;

        let mut records = vec![0.0f32; 25 * 3];
        let mean0_bad = 10.0 / 255.0;
        for block in 1..25 {
            records[block * 3] = n * mean0_bad;
        }
        let sigma = 4.0 / 255.0;
        let var = 2.0 * sigma * sigma;
        records[1] = n * var;
        records[2] = 240.0 * 0.5 * var;

        assert!(
            aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height).is_none(),
            "1 of 25 static blocks (4%) should fall back below the 5% floor"
        );
    }

    /// A zero-filled duplicate slot's records must fall back to
    /// Immerkær, not report a fabricated sigma of zero. Every block
    /// passes the static gate trivially (mean0 = 0), so this exercises
    /// the empty-rho-set path rather than the 5% floor.
    #[test]
    fn aggregate_zeroed_slot_returns_none() {
        let width = 32;
        let height = 32;
        let stored_ch = 1;
        let channels = 1;
        let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
        let record_len = temporal_stats_record_len(stored_ch) as usize;
        let records = vec![0.0f32; (blocks_x * blocks_y) as usize * record_len];

        assert!(
            aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height).is_none(),
            "a zero-filled duplicate slot's stats must fall back to Immerkær, not report sigma=0"
        );
    }

    /// YUV storage (`stored_ch = 4`, `channels = 3`): per-channel sums
    /// must be read from the right stride offset, and the unused pad
    /// lane must never influence the result.
    #[test]
    fn aggregate_multi_channel_layout_reads_correct_offsets() {
        let width = 16;
        let height = 16;
        let stored_ch = 4;
        let channels = 3;
        let n = 256.0f32;
        let n_pairs = 240.0f32;

        let sigmas_255 = [2.0f32, 4.0, 6.0];
        let rho_target = 0.6f32;

        let mut record = vec![0.0f32; 9];
        let mut var0 = 0.0f32;
        for (c, sigma_255) in sigmas_255.iter().enumerate() {
            let sigma = sigma_255 / 255.0;
            let var = 2.0 * sigma * sigma;
            record[stored_ch as usize + c] = n * var;
            if c == 0 {
                var0 = var;
            }
        }
        record[2 * stored_ch as usize] = n_pairs * rho_target * var0;

        let sample = aggregate_temporal_noise_stats(&record, channels, stored_ch, width, height)
            .expect("the single block is static with measurable channel-0 noise");

        for (c, sigma_255) in sigmas_255.iter().enumerate() {
            let expected = sigma_255 / 255.0;
            assert!(
                (sample.sigma[c] - expected).abs() < 1e-4,
                "channel {c}: expected {expected}, got {}",
                sample.sigma[c]
            );
        }
        assert!((sample.rho - rho_target).abs() < 1e-4);
    }

    /// Builds one block's record from a closed-form channel-0 sigma and
    /// rho, `sum_d` always 0 so the block clears [`STATIC_GATE`]
    /// trivially. Shared by the outlier-gate tests below, which only
    /// vary each block's sigma.
    fn zero_mean_block_record(sigma_255: f32, rho: f32) -> [f32; 3] {
        let n = 256.0f32;
        let n_pairs = 240.0f32;
        let sigma = sigma_255 / 255.0;
        let var = 2.0 * sigma * sigma;
        [0.0, n * var, n_pairs * rho * var]
    }

    /// A `64x64` frame (16 `TEMPORAL_NOISE_BLOCK` blocks) where most
    /// blocks are a panning-texture stand-in. Each has zero mean
    /// residual (it clears [`STATIC_GATE`] the same way real
    /// measurement noise does) but a channel-0 sigma an order of
    /// magnitude above the minority of genuinely static blocks, and
    /// zero lag-1 correlation
    /// (`rho = 0`, ruling out a rho-based gate as the fix, since these
    /// blocks look exactly like white noise at lag 1). Without the
    /// outlier gate, the plain mean gate lets every block through and
    /// the majority-texture population dominates the median, reading
    /// the panning blocks' sigma instead of the real noise floor.
    #[test]
    fn aggregate_rejects_majority_zero_mean_texture_outliers() {
        let width = 64;
        let height = 64;
        let stored_ch = 1;
        let channels = 1;

        let background_sigma_255 = 2.0f32;
        let background_rho = 0.1f32;
        let texture_sigma_255 = 20.0f32;

        // 6 genuinely static blocks, 10 panning-texture blocks. The
        // texture population is the majority of the 16 blocks, so a
        // plain population median would read the texture level.
        let mut records = Vec::new();
        for _ in 0..6 {
            records.extend_from_slice(&zero_mean_block_record(background_sigma_255, background_rho));
        }
        for _ in 0..10 {
            records.extend_from_slice(&zero_mean_block_record(texture_sigma_255, 0.0));
        }

        let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
            .expect("the static minority clears STATIC_FRACTION_MIN on its own");

        let expected_sigma = background_sigma_255 / 255.0;
        assert!(
            (sample.sigma[0] - expected_sigma).abs() < 1e-4,
            "expected the outlier gate to isolate the real noise floor {expected_sigma}, got {}",
            sample.sigma[0]
        );
        assert!(
            (sample.static_fraction - 6.0 / 16.0).abs() < 1e-4,
            "expected only the 6 background blocks to survive both gates, got static_fraction={}",
            sample.static_fraction
        );
        assert!(
            (sample.rho - background_rho).abs() < 1e-4,
            "expected rho to come from the surviving background blocks only, got {}",
            sample.rho
        );
    }

    /// The opposite failure direction. Every block is genuinely static
    /// but the frame's real noise floor varies spatially by 3x (a dark
    /// region reading noisier than a bright one, same as real sensor
    /// noise commonly does). None of this spread is texture, so the
    /// outlier gate must not drop any of it.
    #[test]
    fn aggregate_keeps_genuinely_static_blocks_despite_spatial_sigma_spread() {
        let width = 64;
        let height = 64;
        let stored_ch = 1;
        let channels = 1;

        let low_sigma_255 = 2.0f32;
        let high_sigma_255 = 6.0f32; // 3x low_sigma_255, real spatial spread.
        let rho = 0.1f32;

        let mut records = Vec::new();
        for _ in 0..8 {
            records.extend_from_slice(&zero_mean_block_record(low_sigma_255, rho));
        }
        for _ in 0..8 {
            records.extend_from_slice(&zero_mean_block_record(high_sigma_255, rho));
        }

        let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
            .expect("every block is static");

        assert!(
            (sample.static_fraction - 1.0).abs() < 1e-6,
            "a real 3x spatial sigma spread must not trip the outlier gate, got static_fraction={}",
            sample.static_fraction
        );
    }

    /// 8 identical background blocks fix the mean-gate-passing
    /// population's own lower quartile at `background_sigma_255`
    /// regardless of a 9th block's sigma. With `n = 9` candidates the
    /// lower-quartile index (`0.25 * 8 = 2.0` exact) lands on the
    /// 3rd-smallest, still one of the 8 identical values as long as the
    /// 9th sorts above them. A `48x48` frame grids into exactly 9
    /// `TEMPORAL_NOISE_BLOCK` blocks, so the 9th block here is the only
    /// free variable.
    fn outlier_factor_boundary_records(
        background_sigma_255: f32,
        background_rho: f32,
        ninth_ratio: f32,
    ) -> Vec<f32> {
        let mut records = Vec::new();
        for _ in 0..8 {
            records.extend_from_slice(&zero_mean_block_record(background_sigma_255, background_rho));
        }
        records.extend_from_slice(&zero_mean_block_record(background_sigma_255 * ninth_ratio, 0.0));
        records
    }

    /// Calibrated boundary for the outlier gate, pinned as literals
    /// rather than derived from [`SIGMA_OUTLIER_FACTOR`] itself, so
    /// the two tests below assert the actual measured calibration
    /// instead of the gate's own arithmetic (`sigma <=
    /// quartile * SIGMA_OUTLIER_FACTOR` reduces to `ratio <=
    /// SIGMA_OUTLIER_FACTOR` for any constant value, which proves
    /// nothing about where that value should sit). `4.99` and `5.01`
    /// bracket the measured calibration from the original
    /// investigation (a scratchpad simulation, `sim3a.py`, not
    /// committed to the repo). A genuinely static frame with a real
    /// spatial noise-floor spread up to 4x survives, and a spread of
    /// 5x is where trimming starts, so `5.0` was chosen as
    /// [`SIGMA_OUTLIER_FACTOR`]'s value with headroom above the real
    /// spread and margin below where texture contamination is caught.
    /// A deliberate change to [`SIGMA_OUTLIER_FACTOR`] needs to update
    /// these two literals to match, or these tests will (correctly)
    /// fail.
    const OUTLIER_FACTOR_SURVIVES_RATIO: f32 = 4.99;
    const OUTLIER_FACTOR_REJECTS_RATIO: f32 = 5.01;

    #[test]
    fn aggregate_outlier_factor_survives_just_under_threshold() {
        let width = 48;
        let height = 48;
        let stored_ch = 1;
        let channels = 1;
        let background_sigma_255 = 2.0f32;
        let background_rho = 0.1f32;

        let records = outlier_factor_boundary_records(
            background_sigma_255,
            background_rho,
            OUTLIER_FACTOR_SURVIVES_RATIO,
        );

        let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
            .expect("all 9 blocks clear the static-fraction floor");

        assert!(
            (sample.static_fraction - 1.0).abs() < 1e-6,
            "a 9th block at {OUTLIER_FACTOR_SURVIVES_RATIO}x the reference must survive, \
             got static_fraction={}",
            sample.static_fraction
        );
    }

    #[test]
    fn aggregate_outlier_factor_rejects_just_over_threshold() {
        let width = 48;
        let height = 48;
        let stored_ch = 1;
        let channels = 1;
        let background_sigma_255 = 2.0f32;
        let background_rho = 0.1f32;

        let records = outlier_factor_boundary_records(
            background_sigma_255,
            background_rho,
            OUTLIER_FACTOR_REJECTS_RATIO,
        );

        let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
            .expect("the 8 background blocks alone still clear the static-fraction floor");

        assert!(
            (sample.static_fraction - 8.0 / 9.0).abs() < 1e-6,
            "a 9th block at {OUTLIER_FACTOR_REJECTS_RATIO}x the reference must be rejected, \
             got static_fraction={}",
            sample.static_fraction
        );
    }

    /// A `160x160` frame (100 `TEMPORAL_NOISE_BLOCK` blocks, a 10x10
    /// grid) where 26 blocks are exactly-zero-residual letterbox bars
    /// and the other 74 carry real static noise. Both populations
    /// trivially clear [`STATIC_GATE`] (mean residual is exactly zero
    /// for the bars, and centred on zero by construction for the real
    /// noise), so all 100 land in the mean-gate-passing population.
    ///
    /// The 74 real-noise blocks cycle through five close sigma levels
    /// (`3.8..=4.2` in `/255` units, `0.1` apart) rather than a single
    /// repeated value, standing in for the sampling variance any
    /// real per-block measurement carries even under a physically
    /// uniform noise process. That spread is what lets this population
    /// clear the anchor check the outlier gate now requires whenever it
    /// excludes low-sigma blocks (see
    /// [`aggregate_temporal_noise_stats`]'s doc) — a population with no
    /// internal spread at all, sitting next to an excluded low-sigma
    /// population, is indistinguishable from a self-captured outlier
    /// run and reports `None` instead (see
    /// [`aggregate_returns_none_when_the_only_above_gate_population_is_texture`]).
    ///
    /// 74 blocks split five ways gives counts `[15, 15, 15, 15, 14]`
    /// for levels `[3.8, 3.9, 4.0, 4.1, 4.2]` (`74 = 4*15 + 14`, the
    /// remainder falling to the earliest levels). Combined with the 26
    /// zero blocks sorted first, the `n = 100` population's median
    /// (`values[49]`/`values[50]`) lands at cumulative index 41..55,
    /// the `3.9` run, so the expected sigma is exactly `3.9 / 255`, not
    /// an approximation.
    #[test]
    fn aggregate_returns_correct_sigma_with_letterbox_zero_population() {
        let width = 160;
        let height = 160;
        let stored_ch = 1;
        let channels = 1;

        let background_rho = 0.2f32;
        let sigma_levels_255 = [3.8f32, 3.9, 4.0, 4.1, 4.2];

        let mut records = Vec::new();
        for _ in 0..26 {
            records.extend_from_slice(&zero_mean_block_record(0.0, 0.0));
        }
        for i in 0..74 {
            let sigma_255 = sigma_levels_255[i % sigma_levels_255.len()];
            records.extend_from_slice(&zero_mean_block_record(sigma_255, background_rho));
        }

        let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
            .expect("74 of 100 blocks carry real static noise, far above the 5% floor");

        let expected_sigma = 3.9 / 255.0;
        assert!(
            (sample.sigma[0] - expected_sigma).abs() < 1e-4,
            "expected the letterbox bars to leave the real noise floor near {expected_sigma} intact, got {}",
            sample.sigma[0]
        );
        assert!(
            (sample.rho - background_rho).abs() < 1e-4,
            "expected rho to come from the real-noise blocks, got {}",
            sample.rho
        );
    }

    /// The capture scenario the outlier-gate exclusion opens up. The
    /// same 26 exactly-zero letterbox blocks, but the other 74 are all
    /// panning-texture blocks at a single repeated `sigma_ch0 =
    /// 20/255`, with no legitimate above-gate noise anywhere in the
    /// frame. Every candidate above [`RHO_SIGMA_GATE`] is the same
    /// outlier value, so the reference population's own lower quartile
    /// equals that value, and `x <= x * SIGMA_OUTLIER_FACTOR` holds for
    /// any `x >= 0` — the outlier ceiling accepts every one of its own
    /// outliers. A population that homogeneous, sitting next to 26
    /// excluded zero-sigma blocks, is exactly the signature this
    /// function cannot resolve, so it must report `None` and let the
    /// caller fall back to the Immerkær read rather than a confidently
    /// wrong ~20/255.
    #[test]
    fn aggregate_returns_none_when_the_only_above_gate_population_is_texture() {
        let width = 160;
        let height = 160;
        let stored_ch = 1;
        let channels = 1;

        let texture_sigma_255 = 20.0f32;

        let mut records = Vec::new();
        for _ in 0..26 {
            records.extend_from_slice(&zero_mean_block_record(0.0, 0.0));
        }
        for _ in 0..74 {
            records.extend_from_slice(&zero_mean_block_record(texture_sigma_255, 0.0));
        }

        assert!(
            aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height).is_none(),
            "a homogeneous above-gate population with no genuine low anchor must fall back to \
             None rather than report the texture level as sigma"
        );
    }

    /// [`aggregate_rejects_majority_zero_mean_texture_outliers`] with a
    /// letterbox-style zero-sigma population mixed in. The outlier gate
    /// must still isolate the real noise floor from the texture
    /// majority, and the zero population must not disturb that result.
    #[test]
    fn aggregate_rejects_texture_outliers_with_zero_population_present() {
        let width = 64;
        let height = 80; // 4 x 5 TEMPORAL_NOISE_BLOCK grid, 20 blocks.
        let stored_ch = 1;
        let channels = 1;

        let background_sigma_255 = 2.0f32;
        let background_rho = 0.1f32;
        let texture_sigma_255 = 20.0f32;

        let mut records = Vec::new();
        for _ in 0..6 {
            records.extend_from_slice(&zero_mean_block_record(background_sigma_255, background_rho));
        }
        for _ in 0..10 {
            records.extend_from_slice(&zero_mean_block_record(texture_sigma_255, 0.0));
        }
        for _ in 0..4 {
            records.extend_from_slice(&zero_mean_block_record(0.0, 0.0));
        }

        let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
            .expect("the static minority clears STATIC_FRACTION_MIN on its own");

        let expected_sigma = background_sigma_255 / 255.0;
        assert!(
            (sample.sigma[0] - expected_sigma).abs() < 1e-4,
            "expected the outlier gate to isolate the real noise floor {expected_sigma} despite \
             the zero population, got {}",
            sample.sigma[0]
        );
        assert!(
            (sample.static_fraction - 10.0 / 20.0).abs() < 1e-4,
            "expected the 6 background blocks and 4 zero blocks to survive, got static_fraction={}",
            sample.static_fraction
        );
        assert!(
            (sample.rho - background_rho).abs() < 1e-4,
            "expected rho to come from the surviving background blocks only, got {}",
            sample.rho
        );
    }

    /// [`aggregate_keeps_genuinely_static_blocks_despite_spatial_sigma_spread`]
    /// with a letterbox-style zero-sigma population mixed in. A real
    /// spatial sigma spread must still survive the outlier gate, and
    /// the zero population must not trip it either.
    #[test]
    fn aggregate_keeps_static_spread_with_zero_population_present() {
        let width = 64;
        let height = 80; // 4 x 5 TEMPORAL_NOISE_BLOCK grid, 20 blocks.
        let stored_ch = 1;
        let channels = 1;

        let low_sigma_255 = 2.0f32;
        let high_sigma_255 = 6.0f32; // 3x low_sigma_255, real spatial spread.
        let rho = 0.1f32;

        let mut records = Vec::new();
        for _ in 0..8 {
            records.extend_from_slice(&zero_mean_block_record(low_sigma_255, rho));
        }
        for _ in 0..8 {
            records.extend_from_slice(&zero_mean_block_record(high_sigma_255, rho));
        }
        for _ in 0..4 {
            records.extend_from_slice(&zero_mean_block_record(0.0, 0.0));
        }

        let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
            .expect("every non-zero block is static");

        assert!(
            (sample.static_fraction - 1.0).abs() < 1e-6,
            "a real 3x spatial sigma spread plus a zero population must not trip the outlier \
             gate, got static_fraction={}",
            sample.static_fraction
        );
    }

    /// A block whose per-block correlation estimate is computed from
    /// [`STATIC_GATE`] and [`RHO_SIGMA_GATE`]'s own mismatched
    /// denominators (`mean_lag` over `n_pairs`, `mean0`/`var_ch0` over
    /// `n`) can read above 1. A full `TEMPORAL_NOISE_BLOCK` row here
    /// follows the path-graph eigenvector that maximises a row's
    /// lag-1-sum-to-total-variance ratio (`sin(i*pi/17)` for a
    /// 16-element row), replicated down every row of the block and
    /// scaled to clear both the mean gate and the rho-sample sigma
    /// gate. The resulting per-block estimate is a standalone
    /// computation, not the module's usual test-data shape, because
    /// it's checking the aggregation formula's own bound rather than a
    /// physically-motivated noise scenario.
    #[test]
    fn aggregate_rho_estimate_stays_within_unit_range() {
        let width = TEMPORAL_NOISE_BLOCK;
        let height = TEMPORAL_NOISE_BLOCK;
        let stored_ch = 1;
        let channels = 1;

        let scale = 0.007f32;
        let row: Vec<f32> = (1..=width)
            .map(|i| scale * (i as f32 * std::f32::consts::PI / (width + 1) as f32).sin())
            .collect();

        let n = (width * height) as f32;
        let sum_d: f32 = row.iter().sum::<f32>() * height as f32;
        let sum_d2: f32 = row.iter().map(|v| v * v).sum::<f32>() * height as f32;
        let sum_lag: f32 = row.windows(2).map(|w| w[0] * w[1]).sum::<f32>() * height as f32;

        // Confirms the construction actually clears both gates this
        // block needs to reach the rho computation at all, so the
        // assertion below is testing the clamp and not a gate miss.
        assert!(
            (sum_d / n).abs() < STATIC_GATE,
            "construction must clear the static gate"
        );
        let var = sum_d2 / n - (sum_d / n) * (sum_d / n);
        assert!(
            (var.sqrt() / std::f32::consts::SQRT_2) > RHO_SIGMA_GATE,
            "construction must clear the rho-sample sigma gate"
        );

        let records = [sum_d, sum_d2, sum_lag];
        let sample = aggregate_temporal_noise_stats(&records, channels, stored_ch, width, height)
            .expect("the single block clears both gates");

        assert!(
            (0.0..=1.0).contains(&sample.rho),
            "the mismatched-denominator estimate must stay within [0, 1], got {}",
            sample.rho
        );
    }

    #[test]
    fn interpolate_table_linear_between_points() {
        let table = [(0.0f32, 1.0f32), (0.5, 1.2), (1.0, 1.5)];
        assert!((interpolate_table(&table, 0.25) - 1.1).abs() < 1e-6);
        assert!((interpolate_table(&table, 0.75) - 1.35).abs() < 1e-6);
        assert_eq!(interpolate_table(&table, 0.0), 1.0);
        assert_eq!(interpolate_table(&table, 1.0), 1.5);
    }

    #[test]
    fn interpolate_table_clamps_outside_range() {
        let table = [(0.0f32, 1.0f32), (1.0, 2.0)];
        assert_eq!(interpolate_table(&table, -5.0), 1.0);
        assert_eq!(interpolate_table(&table, 5.0), 2.0);
    }

    #[test]
    fn correlation_factor_matches_measured_table() {
        assert_eq!(correlation_factor(0.0), 1.0);
        assert!((correlation_factor(0.65) - 1.45).abs() < 1e-6);
        // Clamped flat past the last measured point.
        assert!((correlation_factor(0.9) - 1.45).abs() < 1e-6);
        // White noise stays uncorrected.
        assert_eq!(correlation_factor(-0.2), 1.0);
        // Monotone non-decreasing across the measured range.
        let mut last = 0.0;
        for i in 0..=20 {
            let f = correlation_factor(i as f32 / 20.0);
            assert!(f >= last, "factor must not decrease, {f} < {last}");
            last = f;
        }
    }

    #[test]
    fn spatial_offset_factor_rho_zero_is_white_identity() {
        // rho = 0 (and negative, which the aggregator never produces
        // but the formula must still handle inertly): every non-self
        // candidate keeps the full white-noise factor of 1.
        for rho in [0.0f32, -0.2] {
            for dy in -3..=3 {
                for dx in -3..=3 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    assert_eq!(
                        spatial_offset_factor(dx, dy, rho),
                        1.0,
                        "dx={dx} dy={dy} rho={rho}"
                    );
                }
            }
        }
    }

    #[test]
    fn spatial_offset_factor_self_is_always_zero() {
        for rho in [-0.2f32, 0.0, 0.3, 0.65, 1.0] {
            assert_eq!(spatial_offset_factor(0, 0, rho), 0.0, "rho={rho}");
        }
    }

    #[test]
    fn spatial_offset_factor_monotone_nondecreasing_in_distance() {
        let rho = 0.65f32;
        let mut last = spatial_offset_factor(0, 0, rho);
        for d in 1..=8 {
            let f = spatial_offset_factor(d, 0, rho);
            assert!(f >= last, "factor decreased at d={d}: {f} < {last}");
            last = f;
        }
    }

    #[test]
    fn spatial_offset_factor_rho_0_65_shape() {
        let rho = 0.65f32;
        // d = 1: 1 - rho^1.
        assert!((spatial_offset_factor(1, 0, rho) - (1.0 - rho)).abs() < 1e-6);
        // d = 2 (axis-aligned): 1 - rho^2.
        assert!((spatial_offset_factor(2, 0, rho) - (1.0 - rho * rho)).abs() < 1e-6);
        // Diagonal d = sqrt(2): matches the sqrt(dx^2+dy^2) formula directly.
        let d = 2.0f32.sqrt();
        let expected = 1.0 - (d * rho.ln()).exp();
        assert!((spatial_offset_factor(1, 1, rho) - expected).abs() < 1e-6);
        // Every factor stays within [0, 1].
        for dy in -4..=4 {
            for dx in -4..=4 {
                let f = spatial_offset_factor(dx, dy, rho);
                assert!((0.0..=1.0).contains(&f), "dx={dx} dy={dy} factor={f}");
            }
        }
    }

    #[test]
    fn build_spatial_offset_lut_rho_zero_matches_flat_noise_offset() {
        let search_radius = 3;
        let noise_offset = 1.5f32;
        let lut = build_spatial_offset_lut(search_radius, 0.0, noise_offset);
        assert_eq!(lut.len(), spatial_offset_lut_len(search_radius));

        let side = (2 * search_radius + 1) as usize;
        let r = search_radius as i32;
        for dy in -r..=r {
            for dx in -r..=r {
                let idx = ((dy + r) as usize) * side + (dx + r) as usize;
                if dx == 0 && dy == 0 {
                    assert_eq!(lut[idx], 0.0, "self offset must be zero");
                } else {
                    assert_eq!(lut[idx], noise_offset, "dx={dx} dy={dy}");
                }
            }
        }
    }

    #[test]
    fn build_spatial_offset_lut_indexes_row_major_by_dy_then_dx() {
        let search_radius = 2;
        let lut = build_spatial_offset_lut(search_radius, 0.65, 10.0);
        let side = (2 * search_radius + 1) as usize;

        // (dx=1, dy=0) lands at row 2 (dy+2), column 3 (dx+2): index 2*5+3=13.
        let expected = 10.0 * spatial_offset_factor(1, 0, 0.65);
        assert_eq!(lut[2 * side + 3], expected);

        // (dx=0, dy=-2) lands at row 0, column 2: index 0*5+2=2.
        let expected = 10.0 * spatial_offset_factor(0, -2, 0.65);
        assert_eq!(lut[2], expected);

        // Centre (dx=0, dy=0) is always zero.
        assert_eq!(lut[2 * side + 2], 0.0);
    }
}

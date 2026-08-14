//! Measuring how noisy a source is.
//!
//! The HQ variant matches its strength to the noise level, so it needs a
//! number for that level. This module produces one per frame.
//!
//! # Two ways of looking
//!
//! The Immerkær estimate runs a small mask over each frame that cancels
//! smooth content and leaves mostly noise. It is cheap and needs only
//! one frame, but it reads grain that is correlated between neighbouring
//! pixels too low, because such grain looks partly like content to the
//! mask.
//!
//! The temporal estimate compares a frame against the one before it.
//! Where nothing moved, whatever is left over is noise, and correlated
//! grain shows up in full. It needs static content to work, so motion
//! and scene changes make it unreliable.
//!
//! The two are combined by taking whichever reads higher, which lets the
//! temporal estimate correct an Immerkær under-read without letting an
//! unreliable one drag the estimate down.
//!
//! # Two chains
//!
//! The result feeds two separate smoothed estimates.
//!
//! The median chain reads typical noise and drives the filter strength.
//!
//! The low chain reads cautiously, using lower-quartile statistics, and
//! drives the distance floor. Reading that too high scrubs fine texture,
//! so it is deliberately the more conservative of the two.

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

/// The inputs one Immerkær noise estimate needs.
///
/// This lives only for the length of a single estimate call, which is
/// what makes the borrows on the denoiser's buffers sound.
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

/// How many `f32` elements the first stage's partials buffer needs for
/// one frame.
///
/// Each block covers a tile of the frame and contributes four lanes.
pub(super) fn partials_len(width: u32, height: u32) -> usize {
    (width.div_ceil(BLOCK_X) * height.div_ceil(BLOCK_Y) * 4) as usize
}

/// The byte stride between ring slots in the partials buffer, padded up
/// to the runtime's buffer-binding alignment.
///
/// A small frame can leave [`partials_len`] short of a boundary, and
/// wgpu rejects a bind-group offset that is not a multiple of its
/// `min_storage_buffer_offset_alignment`.
///
/// This matches [`temporal_stats_slot_stride_bytes`].
pub(super) fn noise_partials_slot_stride_bytes(width: u32, height: u32, align: StorageAlign) -> u64 {
    align.pad_bytes(partials_len(width, height) as u64 * size_of::<f32>() as u64)
}

/// Runs both stages of the Immerkær noise estimate for one frame.
///
/// The per-channel totals go into the results buffer at this frame's
/// slot.
///
/// That buffer holds four values per ring slot, matching the input
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

/// Turns the summed absolute mask responses into an Immerkær sigma.
///
/// The interior area leaves out the one-pixel border the mask cannot
/// reach.
pub(super) fn sigma_from_abs_sum(abs_sum: f32, width: u32, height: u32) -> f32 {
    let interior = ((width - 2) as f32) * ((height - 2) as f32);
    (std::f32::consts::FRAC_PI_2).sqrt() * abs_sum / (6.0 * interior)
}

/// The per-channel lower quartile of the per-block Immerkær sigmas.
///
/// It reads one slot's partials directly, in the layout the first stage
/// wrote them.
///
/// Each block's own sigma comes from the same formula
/// [`sigma_from_abs_sum`] uses, applied to however much of that block's
/// tile overlaps the frame's interior.
///
/// A block with no interior overlap is skipped rather than diluting the
/// quartile with a spurious zero.
///
/// Wherever noise is uneven across a frame, this quartile reads lower
/// than the frame-wide mean, which is the cautious estimate the low
/// chain wants.
///
/// Channels past the active count stay at 0.
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

/// The spatial block size the temporal residual statistics use, with one
/// GPU block per square of this size.
pub(super) const TEMPORAL_NOISE_BLOCK: u32 = 16;

/// How many `f32`s one block's stats record holds, being a sum and a
/// sum of squares per stored channel plus one lag-1 total.
pub(super) fn temporal_stats_record_len(stored_ch: u32) -> u32 {
    2 * stored_ch + 1
}

/// The block grid covering a frame, laid out row-major.
///
/// Ragged edges are truncated rather than padded, the same way the block
/// matcher handles its own ragged last block.
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

/// The byte stride between ring slots in the temporal-stats buffer,
/// padded up to the runtime's buffer-binding alignment.
///
/// A small frame, or a single-channel mode, can leave
/// [`temporal_stats_slot_len`] short of a boundary, and wgpu rejects a
/// bind-group offset that is not a multiple of its
/// `min_storage_buffer_offset_alignment`.
///
/// This matches `MotionCtx::confidence_bytes_per_neighbour`.
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

/// The inputs one temporal residual statistics dispatch needs, comparing
/// the new slot against the previous one on the input ring.
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

/// Runs the temporal residual statistics kernel for the new slot,
/// writing one record per block into that slot's padded region.
///
/// The kernel only ever addresses within its own slice, so it needs to
/// know nothing about the ring's other slots or the padding between
/// them.
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

/// Fills one ring slot's temporal-stats region with zeroes.
///
/// This runs when a slot is a copy of the one before it, which happens
/// while priming a stream and during the end-of-stream flush.
///
/// The zeroes read as no static blocks with measurable noise, rather
/// than as a made-up reading of zero noise.
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

/// Reads exactly one ring slot's temporal-stats region back as owned
/// values.
///
/// The shared ring handle is sliced by byte offset, so the transfer only
/// covers one slot rather than the whole ring.
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
    /// The per-channel sigma, taken as the median over static blocks, in
    /// normalised units. Entries past the active channel count stay 0.
    pub sigma: [f32; 3],
    /// The same per-channel sigma taken at the lower quartile instead of
    /// the median, in normalised units.
    ///
    /// This reads more cautiously than `sigma`, for the consumers where
    /// reading too high does more harm than reading too low. Entries
    /// past the active channel count stay 0.
    pub sigma_low: [f32; 3],
    /// How correlated the grain is between neighbouring pixels.
    ///
    /// It is the median, over the static blocks with measurable noise,
    /// of how strongly each residual matches the one beside it.
    pub rho: f32,
    /// What fraction of blocks counted as static.
    pub static_fraction: f32,
}

/// How large a block's mean residual can be and still count as static,
/// in normalised units.
///
/// A block above this is treated as moving content rather than as noise.
const STATIC_GATE: f32 = 1.5 / 255.0;
/// The smallest block sigma that still counts toward the correlation
/// median.
///
/// Below this a block carries too little signal for its correlation
/// reading to mean anything.
const RHO_SIGMA_GATE: f32 = 0.3 / 255.0;
/// The smallest fraction of static blocks a sample needs to be trusted.
///
/// Below this, motion or a scene change dominates the frame and the
/// Immerkær estimate is the only usable reading.
const STATIC_FRACTION_MIN: f32 = 0.05;
/// How far above the surviving blocks' own lower quartile a block's
/// sigma may sit before it is treated as moving texture rather than
/// noise.
///
/// # Why this is needed
///
/// A block panning across texture can average out to nearly nothing
/// over its window, which clears [`STATIC_GATE`], while its variance is
/// entirely shifted texture rather than repeatable noise. That variance
/// runs several times higher than the rest of the frame's blocks.
///
/// Real noise keeps a much narrower spread across a frame, even where
/// its magnitude genuinely varies, such as a dark region reading noisier
/// than a bright one. This threshold leaves room for that spread while
/// still catching the texture outliers.
///
/// # Why the lower quartile
///
/// The reference is the lower quartile rather than the median, so the
/// filter still works once texture makes up most of the surviving
/// blocks, as long as a genuinely static minority remains to anchor it.
///
/// The quartile is computed only over blocks whose own sigma clears
/// [`RHO_SIGMA_GATE`]. Letterbox bars and other perfectly static regions
/// read a sigma of exactly 0, and leaving them in can drag the quartile
/// itself to 0, which would reject every block carrying real noise
/// rather than just the outliers.
///
/// That exclusion only holds up while a genuinely low population remains
/// to anchor the quartile. [`aggregate_temporal_noise_stats`] covers
/// what happens when none does.
const SIGMA_OUTLIER_FACTOR: f32 = 5.0;

/// One surviving block's per-channel stats, kept just long enough to
/// work out the reference the outlier check needs before deciding which
/// blocks are really static.
struct StaticGateCandidate {
    sigmas: [f32; 3],
    sigma_ch0: f32,
    var_ch0: f32,
    mean0: f32,
    mean_lag: f32,
    n_pairs: f32,
}

/// Combines one centre slot's per-block records into a single
/// [`TemporalNoiseSample`].
///
/// `records` holds exactly one slot's region, laid out block by block as
/// [`nlm_temporal_noise_stats`] documents.
///
/// # When no sample is produced
///
/// This returns `None` in three cases.
///
/// Too few blocks counted as static, below [`STATIC_FRACTION_MIN`], so
/// motion dominates the frame and nothing here can be trusted.
///
/// No static block carried measurable noise, so the correlation median
/// would be undefined. A zero-filled duplicate slot produces exactly
/// this.
///
/// The outlier check below had no way to validate its own ceiling.
///
/// # Deciding which blocks are static
///
/// This takes two passes.
///
/// The first checks each block's mean residual against [`STATIC_GATE`].
/// A block whose average residual is near zero passes, but a block
/// panning across texture passes too, because displaced texture averages
/// toward zero over a block just as noise does.
///
/// The second pass catches what the first let through. It rejects any
/// surviving block whose sigma sits far above the surviving population's
/// own lower quartile, by more than [`SIGMA_OUTLIER_FACTOR`]. A panning
/// block's variance comes from the texture it slid across, not from
/// noise shared with the rest of the frame.
///
/// That quartile looks only at blocks clearing [`RHO_SIGMA_GATE`], so a
/// perfectly static region such as a letterbox bar cannot drag the
/// ceiling to 0 and reject every noisy block with it.
///
/// # When the ceiling cannot be trusted
///
/// Excluding those low-sigma blocks has a cost of its own. If every
/// remaining block turns out to be texture, that population sets its own
/// ceiling and lets all of its members through, because a value never
/// exceeds a multiple of itself.
///
/// Nothing in a single block's stats distinguishes texture from noise,
/// so the only check left is whether the surviving population shows any
/// internal spread.
///
/// Real per-block sigma, measured over a finite sample, always varies a
/// little from block to block, even for physically uniform noise. So
/// once the low-sigma blocks are gone, a perfectly uniform remainder is
/// the signature of a self-selected outlier population with nothing to
/// anchor it.
///
/// In that case this reports `None` rather than a confident sigma that
/// may be inflated by texture.
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

    // A perfectly static block, such as a letterbox bar or a duplicate
    // frame, reads a sigma of 0 and clears STATIC_GATE without effort.
    // Blocks like that can dominate the surviving population while
    // carrying no measurable noise at all.
    //
    // Left in this reference set they drag the quartile toward 0, which
    // then rejects every block with real noise instead of just the
    // texture outliers the check exists for.
    //
    // Limiting the reference to blocks that themselves clear
    // RHO_SIGMA_GATE keeps the ceiling anchored to blocks that could
    // plausibly be noise.
    let mut reference_sigma_ch0: Vec<f32> = candidates
        .iter()
        .map(|c| c.sigma_ch0)
        .filter(|&sigma| sigma > RHO_SIGMA_GATE)
        .collect();
    sort_ascending(&mut reference_sigma_ch0);

    // Dropping the low-sigma blocks throws away what they told us, and
    // that has a cost. A value never exceeds a multiple of itself, so
    // if every remaining block turns out to be a texture-panning block,
    // that population sets its own ceiling and lets all of them
    // through, which is exactly the failure the ceiling exists to
    // prevent.
    //
    // Nothing available here tells texture and noise apart on its own,
    // so this asks for corroboration instead. Some blocks must have
    // been excluded, and the reference population left behind must show
    // real internal spread, meaning its lowest and highest readings
    // differ.
    //
    // Per-block sigma is measured over a finite sample, so even
    // physically uniform noise varies a little from block to block. A
    // reference population with no spread at all, sitting next to
    // excluded low-sigma blocks, is the case this cannot resolve, so it
    // reports None rather than a sigma that may be inflated by texture.
    //
    // A population where nothing was excluded, because there were no
    // low-sigma blocks to begin with, skips this check and is trusted
    // directly.
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

/// Sorts `values` into ascending order in place.
///
/// Both [`median`] and [`lower_quartile`] need this first, so a caller
/// wanting both sorts once and passes the same slice to each.
fn sort_ascending(values: &mut [f32]) {
    values.sort_by(|a, b| a.partial_cmp(b).expect("noise stats are never NaN"));
}

/// The median of an already-sorted slice, averaging the two middle
/// elements when the count is even.
///
/// Callers only ever pass a non-empty slice.
fn median(values: &[f32]) -> f32 {
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// The lower quartile of an already-sorted slice.
///
/// It reads the value a quarter of the way along, interpolating between
/// the two neighbouring elements when that lands between them.
///
/// A slice of one returns that element. Callers only ever pass a
/// non-empty slice.
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

/// The correlation-correction table, as points sorted by correlation.
///
/// It was measured on synthetic correlated-grain sweeps against the
/// clean bench reference.
///
/// Each factor is how far the quality peak sits above the true sigma at
/// that level of grain correlation, relative to the white-noise optimum
/// at the same sigma.
///
/// At the heaviest correlation measured, both quality metrics prefer the
/// raised value. Past the last measured point the table holds flat.
const CORRELATION_FACTOR_TABLE: [(f32, f32); 4] = [(0.0, 1.0), (0.3, 1.05), (0.5, 1.25), (0.65, 1.45)];

/// The factor that turns a measured temporal sigma into an effective one
/// allowing for grain correlation.
///
/// The effective sigma is the measured one multiplied by this.
pub(super) fn correlation_factor(rho: f32) -> f32 {
    interpolate_table(&CORRELATION_FACTOR_TABLE, rho)
}

/// Reads a value from a small table of points sorted by `x`,
/// interpolating between the two nearest entries.
///
/// `x` is clamped to the table's own range first, so the result never
/// runs past the endpoints.
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

/// How much of the noise floor still applies at a given candidate
/// offset, once grain correlation is taken into account.
///
/// A nearby candidate shares some of its grain with the centre patch, so
/// only part of the white-noise floor is genuinely independent noise.
/// The further away the candidate, the less they share, and the more of
/// the floor applies.
///
/// The centre itself always returns 0, because its true distance is zero
/// and none of the floor is independent noise there.
///
/// With no measured correlation, which is also what an inert estimator
/// gives, every other offset returns 1, reproducing the flat white-noise
/// floor exactly.
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

/// How many `f32`s a spatial-offset table needs at a given search
/// radius.
pub(super) fn spatial_offset_lut_len(search_radius: u32) -> usize {
    let side = (2 * search_radius + 1) as usize;
    side * side
}

/// Builds the per-candidate noise-floor table for a search window, laid
/// out row-major.
///
/// Each entry is the flat noise offset scaled by how much of it applies
/// at that candidate's distance.
///
/// It is cheap enough to rebuild on every submit, reaching at most 289
/// entries at the largest supported search radius.
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

/// How much weight the newest frame's estimate carries when smoothing.
///
/// The sigma estimator below and the denoiser's own correlation
/// smoothing both use it.
pub(super) const EMA_ALPHA: f32 = 0.2;
/// The smallest smoothed sigma allowed, in normalised units, which works
/// out at 0.1 in 8-bit terms.
///
/// An estimate near zero would send the derived strength to infinity.
const SIGMA_FLOOR: f32 = 0.1 / 255.0;

/// The noise state for one stream.
///
/// It smooths the per-frame estimates over time, so a single busy frame
/// cannot spike the strength, and applies a floor so near-clean content
/// keeps a usable normalisation factor.
#[derive(Debug, Default)]
pub(super) struct NoiseEstimator {
    ema: Option<Vec<f32>>,
}

impl NoiseEstimator {
    /// Folds a new set of per-channel sigmas into the running estimate
    /// and returns the smoothed result.
    ///
    /// The first call sets the state directly from the sample, because
    /// there is no earlier estimate to blend with.
    ///
    /// Every element is floored at [`SIGMA_FLOOR`] either way.
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

    /// Clears the running estimate.
    ///
    /// The next [`Self::update`] then starts from its own sample rather
    /// than blending with stale state.
    pub(super) fn reset(&mut self) {
        self.ema = None;
    }

    /// The current smoothed per-channel sigma, or `None` before the
    /// first [`Self::update`] call.
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

        // After a reset the next update should start from its own
        // sample rather than blending with the stale 0.10.
        let out = est.update(&[0.02]);
        assert_eq!(out, &[0.02]);
    }

    #[test]
    fn sigma_from_abs_sum_zero_for_zero_response() {
        // No mask response anywhere in the frame has to estimate zero
        // noise, whatever the frame size.
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
        // A single block, whose 4 values come to 16 bytes and pad up to
        // the next 32-byte multiple.
        assert_eq!(noise_partials_slot_stride_bytes(32, 8, StorageAlign::new(32)), 32);
    }

    #[test]
    fn noise_partials_slot_stride_bytes_aligned_count_unchanged() {
        // A 2x2 grid of blocks, whose 16 values come to 64 bytes.
        // That is already a multiple of 32, so nothing is padded.
        assert_eq!(noise_partials_slot_stride_bytes(33, 9, StorageAlign::new(32)), 64);
    }

    /// A 70x20 frame grids into a ragged 3x3 layout of blocks, two full
    /// rows and columns plus a partial one each way, so every block's
    /// interior overlap is a different size.
    ///
    /// These are the hand-computed interior areas, row by row, summing
    /// exactly to the frame's own interior area of 1224.
    const RAGGED_CUBE_AREAS: [[f32; 3]; 3] = [[217.0, 224.0, 35.0], [248.0, 256.0, 40.0], [93.0, 96.0, 15.0]];

    /// A uniform mask response gives every block the same sigma whatever
    /// its area, because the area cancels out of the formula.
    ///
    /// The lower quartile of identical values is that same value, and it
    /// has to match the frame-wide estimate over the same total
    /// response.
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

    /// Nine blocks are given a shuffled set of sigmas from 1 to 9 in
    /// 8-bit units, scaled through each block's own area, so sorting
    /// them reproduces that run exactly.
    ///
    /// With nine values the lower quartile lands exactly on the third
    /// smallest, with no interpolation.
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
        // With 5 values the quartile lands on index 1 exactly, so no
        // interpolation is needed.
        assert_eq!(lower_quartile(&[1.0, 2.0, 3.0, 4.0, 5.0]), 2.0);
    }

    #[test]
    fn lower_quartile_even_count() {
        // With 4 values the quartile lands at 0.75, between the first
        // two.
        let got = lower_quartile(&[1.0, 2.0, 3.0, 4.0]);
        assert!((got - 1.75).abs() < 1e-6, "expected 1.75, got {got}");
    }

    #[test]
    fn lower_quartile_interpolates_at_a_fractional_index() {
        // With 3 values the quartile lands halfway between the first
        // two.
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

    /// Five static blocks, each with its own sigma, and four of them
    /// with their own correlation reading too.
    ///
    /// That covers the odd-count median, the even-count median, and the
    /// lower quartile in one pass.
    #[test]
    fn aggregate_median_over_multiple_static_blocks() {
        let width = 16 * 5;
        let height = 16;
        let stored_ch = 1;
        let channels = 1;
        let n = 256.0f32;
        let n_pairs = 240.0f32;

        // Block 0's sigma sits below RHO_SIGMA_GATE, so its correlation
        // reading never enters the set. The other four all clear it.
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

    /// Only 1 of 25 blocks is static, which is below
    /// `STATIC_FRACTION_MIN`.
    ///
    /// That single block carries perfectly valid noise otherwise, which
    /// isolates the static-fraction floor from the other fallback.
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

    /// A zero-filled duplicate slot has to fall back to Immerkær rather
    /// than report a made-up sigma of zero.
    ///
    /// Every block passes the static check trivially, so this covers the
    /// no-measurable-noise path rather than the static-fraction floor.
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

    /// With YUV storage the three real channels sit in four lanes, so
    /// each channel's sums have to be read at the right stride and the
    /// unused padding lane must never affect the result.
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

    /// Builds one block's record from a chosen sigma and correlation,
    /// with a mean residual of 0 so the block clears [`STATIC_GATE`]
    /// without effort.
    ///
    /// The outlier tests below share this and only vary each block's
    /// sigma.
    fn zero_mean_block_record(sigma_255: f32, rho: f32) -> [f32; 3] {
        let n = 256.0f32;
        let n_pairs = 240.0f32;
        let sigma = sigma_255 / 255.0;
        let var = 2.0 * sigma * sigma;
        [0.0, n * var, n_pairs * rho * var]
    }

    /// A 64x64 frame of 16 blocks where most stand in for panning
    /// texture.
    ///
    /// Each has a mean residual of zero, clearing [`STATIC_GATE`] the
    /// same way real noise does, but a sigma an order of magnitude above
    /// the genuinely static minority.
    ///
    /// Their correlation reading is zero, so they look exactly like
    /// white noise on that measure. That rules out a correlation-based
    /// check as the fix.
    ///
    /// Without the outlier check the mean check lets every block
    /// through, and the texture majority dominates the median, reading
    /// their sigma instead of the real noise floor.
    #[test]
    fn aggregate_rejects_majority_zero_mean_texture_outliers() {
        let width = 64;
        let height = 64;
        let stored_ch = 1;
        let channels = 1;

        let background_sigma_255 = 2.0f32;
        let background_rho = 0.1f32;
        let texture_sigma_255 = 20.0f32;

        // 6 genuinely static blocks against 10 panning-texture ones.
        // The texture blocks are the majority of the 16, so a plain
        // median would read their level instead.
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

    /// The opposite failure. Every block here is genuinely static, but
    /// the frame's real noise varies threefold across it, the way a dark
    /// region often reads noisier than a bright one.
    ///
    /// None of that spread is texture, so the outlier check must not
    /// drop any of it.
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

    /// Eight identical background blocks pin the surviving population's
    /// lower quartile, whatever a ninth block's sigma turns out to be.
    ///
    /// With nine values the quartile lands exactly on the third
    /// smallest, which is still one of the eight identical ones as long
    /// as the ninth sorts above them.
    ///
    /// A 48x48 frame grids into exactly nine blocks, so that ninth block
    /// is the only thing that varies.
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

    /// The two ratios bracketing the outlier check's calibrated
    /// boundary.
    ///
    /// They are written as literals rather than derived from
    /// [`SIGMA_OUTLIER_FACTOR`], so the tests below assert the measured
    /// calibration rather than the check's own arithmetic. Deriving them
    /// would reduce to comparing the constant with itself, which proves
    /// nothing about where it should sit.
    ///
    /// The measurement came from a scratchpad simulation during the
    /// original investigation, which is not in this repo. A genuinely
    /// static frame with a real noise spread up to fourfold survives,
    /// and trimming starts at fivefold.
    ///
    /// So 5.0 was chosen with room above the real spread and margin
    /// below where texture contamination is caught.
    ///
    /// Changing [`SIGMA_OUTLIER_FACTOR`] on purpose means updating these
    /// two literals to match, or these tests will rightly fail.
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

    /// A 160x160 frame of 100 blocks, where 26 stand in for letterbox
    /// bars with no residual at all and the other 74 carry real static
    /// noise.
    ///
    /// Both groups clear [`STATIC_GATE`] without effort, the bars
    /// because their residual is exactly zero and the noisy blocks
    /// because theirs is centred on zero by construction, so all 100
    /// survive the first pass.
    ///
    /// # Why the noise levels vary
    ///
    /// The 74 noisy blocks cycle through five close sigma levels rather
    /// than one repeated value, standing in for the sampling variance
    /// any real measurement carries even under physically uniform noise.
    ///
    /// That spread is what lets them pass the anchor check the outlier
    /// gate applies whenever it excludes low-sigma blocks. See
    /// [`aggregate_temporal_noise_stats`].
    ///
    /// A population with no internal spread, sitting next to excluded
    /// low-sigma blocks, cannot be told apart from a self-selected
    /// outlier run, and reports `None` instead. See
    /// [`aggregate_returns_none_when_the_only_above_gate_population_is_texture`].
    ///
    /// # Why the expected value is exact
    ///
    /// Splitting 74 blocks five ways gives counts of 15, 15, 15, 15, and
    /// 14 across the five levels, with the remainder falling to the
    /// earliest ones.
    ///
    /// With the 26 zero blocks sorted first, the median of all 100 lands
    /// squarely inside the second level's run, so the expected sigma is
    /// exactly that level rather than an approximation.
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

    /// The case the outlier exclusion opens up.
    ///
    /// The same 26 letterbox blocks, but all 74 of the others are
    /// panning texture at one repeated sigma, with no genuine noise
    /// anywhere above the gate.
    ///
    /// Every surviving candidate holds the same value, so the reference
    /// population's lower quartile is that value too, and a value never
    /// exceeds a multiple of itself. The ceiling therefore accepts every
    /// one of its own outliers.
    ///
    /// A population that uniform, sitting next to 26 excluded zero-sigma
    /// blocks, is exactly the case this cannot resolve. It has to report
    /// `None` and let the caller fall back to the Immerkær reading,
    /// rather than confidently report the texture level.
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

    /// The same case as
    /// [`aggregate_rejects_majority_zero_mean_texture_outliers`], with a
    /// letterbox-style zero-sigma population mixed in.
    ///
    /// The outlier check must still pick out the real noise floor from
    /// the texture majority, and the zero blocks must not disturb it.
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

    /// The same case as
    /// [`aggregate_keeps_genuinely_static_blocks_despite_spatial_sigma_spread`],
    /// with a letterbox-style zero-sigma population mixed in.
    ///
    /// A real spread of noise across the frame must still survive the
    /// outlier check, and the zero blocks must not trip it either.
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

    /// A block's correlation estimate can read above 1, because the
    /// lag-1 total is averaged over adjacent pairs while the mean and
    /// variance are averaged over every pixel.
    ///
    /// Each row here follows the pattern that maximises the ratio
    /// between a row's lag-1 sum and its total variance, repeated down
    /// every row of the block and scaled to clear both checks.
    ///
    /// This is a standalone construction rather than the usual test data
    /// in this module, because it probes the formula's own bound instead
    /// of a realistic noise scenario.
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
        // With no correlation, every candidate but the centre keeps the
        // full white-noise factor of 1. A negative value, which the
        // aggregation never produces, has to behave the same way.
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
        // One pixel away.
        assert!((spatial_offset_factor(1, 0, rho) - (1.0 - rho)).abs() < 1e-6);
        // Two pixels away along an axis.
        assert!((spatial_offset_factor(2, 0, rho) - (1.0 - rho * rho)).abs() < 1e-6);
        // A diagonal candidate sits sqrt(2) away, straight from the
        // distance formula.
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

        // One step to the right lands at row 2, column 3 of the table,
        // which is index 13.
        let expected = 10.0 * spatial_offset_factor(1, 0, 0.65);
        assert_eq!(lut[2 * side + 3], expected);

        // (dx=0, dy=-2) lands at row 0, column 2: index 0*5+2=2.
        let expected = 10.0 * spatial_offset_factor(0, -2, 0.65);
        assert_eq!(lut[2], expected);

        // Centre (dx=0, dy=0) is always zero.
        assert_eq!(lut[2 * side + 2], 0.0);
    }
}

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::curve::{NoiseCurve, build_noise_curve};
use super::stats::{lower_quartile, median, sort_ascending};
use super::strength_map::{QuarterClasses, QuarterSettings, classify_quarters};
use crate::nlmeans::align::StorageAlign;
use crate::nlmeans::kernels::{nlm_temporal_noise_stats, nlm_temporal_stats_zero};
use crate::nlmeans::{BLOCK_1D, MAX_GRID_1D};

/// The side of one temporal-stats block in pixels, with one GPU block per square.
pub(in crate::nlmeans) const TEMPORAL_NOISE_BLOCK: u32 = 16;
/// How many 8x8 quarters one block record holds.
pub(in crate::nlmeans) const TEMPORAL_QUARTERS: u32 = 4;
pub(in crate::nlmeans) const TEMPORAL_QUARTER_SIZE: u32 = TEMPORAL_NOISE_BLOCK / 2;
/// The offset of the first quarter record past `2 * stored_ch`.
///
/// Quarter `q` starts at `2 * stored_ch + 1 + 9 * q`, in top-left, top-right, bottom-left,
/// bottom-right order.
pub(in crate::nlmeans) const TEMPORAL_QUARTER_BASE: u32 = 1;
/// How many `f32`s one quarter record holds.
pub(in crate::nlmeans) const TEMPORAL_QUARTER_FIELDS: u32 = 9;
/// The quarter-record offset of channel 0's summed residual.
pub(in crate::nlmeans) const QUARTER_SUM_D: u32 = 0;
/// The quarter-record offset of channel 0's summed squared residual.
pub(in crate::nlmeans) const QUARTER_SUM_D2: u32 = 1;
/// The quarter-record offset of the new frame's summed luma.
pub(in crate::nlmeans) const QUARTER_LUMA_SUM: u32 = 2;
/// The quarter-record offset of the smoothed temporal mean's gradient energy.
///
/// A quarter smaller than 8x8 reads `3.0e38`.
pub(in crate::nlmeans) const QUARTER_FLATNESS: u32 = 3;
/// The quarter-record offset of the new frame's minimum luma.
pub(in crate::nlmeans) const QUARTER_LUMA_MIN: u32 = 4;
/// The quarter-record offset of the new frame's maximum luma.
pub(in crate::nlmeans) const QUARTER_LUMA_MAX: u32 = 5;
/// The quarter-record offset of the temporal mean's summed squared horizontal gradient.
pub(in crate::nlmeans) const QUARTER_TENSOR_XX: u32 = 6;
/// The quarter-record offset of the temporal mean's summed squared vertical gradient.
pub(in crate::nlmeans) const QUARTER_TENSOR_YY: u32 = 7;
/// The quarter-record offset of the temporal mean's summed horizontal times vertical gradient.
pub(in crate::nlmeans) const QUARTER_TENSOR_XY: u32 = 8;

/// The largest mean block residual, in normalised units, that still counts as static.
///
/// A block above this is moving content rather than noise.
pub(super) const STATIC_GATE: f32 = 1.5 / 255.0;
/// The smallest block sigma that counts as measurable noise.
///
/// Below it a block carries too little signal for its correlation reading to mean anything. It
/// gates the correlation median, the outlier ceiling's reference set and the sigma population.
pub(super) const RHO_SIGMA_GATE: f32 = 0.3 / 255.0;
/// The smallest fraction of static blocks a sample needs to be trusted.
///
/// Below it, motion, a scene change or too little measurable noise dominates the frame, and the
/// Immerkær estimate is the only usable reading.
const STATIC_FRACTION_MIN: f32 = 0.05;
/// How far above the surviving blocks' lower quartile a block's sigma may sit before it counts as
/// moving texture.
///
/// A block panning across texture can average to nearly nothing and clear [STATIC_GATE], while
/// its variance is shifted texture running several times above the frame's other blocks. Real
/// noise keeps a much narrower spread, even where a dark region reads noisier than a bright one,
/// and this factor leaves room for that spread.
///
/// The reference is the lower quartile rather than the median, so the check still works when
/// texture makes up most of the surviving blocks, as long as a static minority remains to anchor
/// it. The quartile only counts blocks above [RHO_SIGMA_GATE], because letterbox bars and other
/// perfectly static regions read a sigma of exactly 0 and would drag it to 0, rejecting every
/// block with real noise.
const SIGMA_OUTLIER_FACTOR: f32 = 5.0;

/// How many `f32`s one block's stats record holds.
///
/// That is a sum and a sum of squares per stored channel, one lag-1 total, and the four quarter
/// records.
pub(in crate::nlmeans) fn temporal_stats_record_len(stored_ch: u32) -> u32 {
    2 * stored_ch + TEMPORAL_QUARTER_BASE + TEMPORAL_QUARTERS * TEMPORAL_QUARTER_FIELDS
}

/// The row-major block grid covering a frame, with ragged edges truncated rather than padded.
pub(in crate::nlmeans) fn temporal_stats_blocks(width: u32, height: u32) -> (u32, u32) {
    (
        width.div_ceil(TEMPORAL_NOISE_BLOCK),
        height.div_ceil(TEMPORAL_NOISE_BLOCK),
    )
}

pub(super) fn temporal_stats_slot_len(width: u32, height: u32, stored_ch: u32) -> usize {
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
    (blocks_x * blocks_y * temporal_stats_record_len(stored_ch)) as usize
}

/// The byte stride between temporal-stats ring slots, padded to the buffer-binding alignment.
///
/// wgpu rejects a bind-group offset that is not a multiple of its
/// `min_storage_buffer_offset_alignment`, and a small frame or a single-channel mode can leave
/// [temporal_stats_slot_len] short of one.
fn temporal_stats_slot_stride_bytes(width: u32, height: u32, stored_ch: u32, align: StorageAlign) -> u64 {
    let slot_bytes = temporal_stats_slot_len(width, height, stored_ch) as u64 * size_of::<f32>() as u64;
    align.pad_bytes(slot_bytes)
}

pub(in crate::nlmeans) fn temporal_stats_buf_bytes(
    width: u32,
    height: u32,
    stored_ch: u32,
    frame_count: u32,
    align: StorageAlign,
) -> usize {
    (temporal_stats_slot_stride_bytes(width, height, stored_ch, align) * frame_count as u64) as usize
}

/// The inputs one temporal residual statistics dispatch needs, comparing `slot_new` against
/// `slot_prev`.
pub(in crate::nlmeans) struct TemporalStatsCtx<'a> {
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

/// Runs the temporal residual statistics kernel, writing one record per block into the new slot.
///
/// Where nothing moved the residual is noise, so correlated grain shows in full, but motion and
/// scene changes make the reading unreliable. The kernel addresses only its own slot's slice, so
/// it needs nothing about the ring's other slots or the padding between them. With `luma_fields`
/// off, the kernel's four luma-only lanes read 0 at no extra cost.
pub(in crate::nlmeans) fn run_temporal_noise_stats<R: Runtime>(
    client: &ComputeClient<R>,
    ctx: &TemporalStatsCtx<'_>,
    luma_fields: bool,
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
            luma_fields,
        );
    }

    Ok(())
}

/// Fills one ring slot's temporal-stats region with zeroes.
///
/// It runs on a slot copied from the one before it. The zeroes read as no static blocks with
/// measurable noise rather than as a reading of zero noise.
pub(in crate::nlmeans) fn zero_temporal_stats_slot<R: Runtime>(
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
    let slot_region = stats_buf.clone().offset_start((slot as u64) * stride);

    let grid = slot_len.div_ceil(BLOCK_1D).min(MAX_GRID_1D);
    let total_threads = grid * BLOCK_1D;

    unsafe {
        nlm_temporal_stats_zero::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(grid),
            CubeDim::new_1d(BLOCK_1D),
            ArrayArg::from_raw_parts(slot_region, slot_len as usize),
            slot_len,
            total_threads,
        );
    }
}

/// Reads one ring slot's temporal-stats region back.
///
/// The ring handle is sliced to the slot, so the transfer skips the rest of the ring.
#[expect(
    clippy::too_many_arguments,
    reason = "the readback needs the ring handle plus every shape value that locates one slot in it"
)]
pub(in crate::nlmeans) fn read_temporal_stats_slot<R: Runtime>(
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
        .map_err(|error| anyhow::anyhow!("temporal noise stats readback failed: {error}"))?;
    let records = f32::from_bytes(&bytes).to_vec();

    Ok(records)
}

/// One centre slot's aggregated temporal-residual noise measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::nlmeans) struct TemporalNoiseSample {
    /// The per-channel median sigma over the static blocks with measurable noise, in normalised
    /// units.
    ///
    /// Entries past the active channel count stay 0.
    pub sigma: [f32; 3],
    /// The per-channel lower-quartile sigma in normalised units, for consumers where reading high
    /// does more harm than reading low.
    ///
    /// Entries past the active channel count stay 0.
    pub sigma_low: [f32; 3],
    /// The median lag-1 correlation of the residuals, measuring how correlated the grain is
    /// between neighbouring pixels.
    pub rho: f32,
    /// The fraction of blocks counted as static with measurable noise.
    pub static_fraction: f32,
}

/// A block that passed [STATIC_GATE], held until the outlier ceiling is known.
struct StaticGateCandidate {
    index: usize,
    width: u32,
    height: u32,
    sigmas: [f32; 3],
    sigma_ch0: f32,
    var_ch0: f32,
    mean0: f32,
    mean_lag: f32,
    n_pairs: f32,
}

/// One block [accepted_static_blocks] counted as static noise.
pub(in crate::nlmeans) struct AcceptedBlock {
    /// The block's position in `records`, in units of one record.
    pub(super) index: usize,
    /// The in-frame width, which a ragged edge truncates.
    pub(super) width: u32,
    /// The in-frame height, which a ragged edge truncates.
    pub(super) height: u32,
    /// The per-channel sigma, with entries past the active channel count at 0.
    pub(super) sigmas: [f32; 3],
    /// Channel 0's mean residual.
    pub(super) mean0: f32,
    /// Channel 0's mean lag-1 product over adjacent pixel pairs.
    pub(super) mean_lag: f32,
    /// Channel 0's residual variance.
    pub(super) var_ch0: f32,
    /// How many adjacent pixel pairs the block holds.
    pub(super) n_pairs: f32,
}

/// Picks out the blocks in one slot's records that count as static noise.
///
/// Three filters run in turn. [STATIC_GATE] keeps blocks whose mean residual is near zero, which a
/// block panning across texture also passes. [SIGMA_OUTLIER_FACTOR] then rejects blocks far above
/// the population's lower quartile, since a panning block's variance comes from texture rather
/// than shared noise. Last, blocks at or below [RHO_SIGMA_GATE] carry no measurable noise and are
/// dropped.
///
/// It returns `None` when the outlier ceiling cannot be trusted. Excluding low-sigma blocks lets an
/// all-texture remainder set its own ceiling and pass it, because a value never exceeds a multiple
/// of itself. Real per-block sigma always varies a little over a finite sample, so a remainder with
/// no spread beside excluded blocks is reported as `None` rather than as a sigma texture may have
/// inflated.
pub(in crate::nlmeans) fn accepted_static_blocks(
    records: &[f32],
    channels: u32,
    stored_ch: u32,
    width: u32,
    height: u32,
) -> Option<Vec<AcceptedBlock>> {
    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
    let total_blocks = (blocks_x * blocks_y) as usize;
    if total_blocks == 0 {
        return None;
    }

    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let channels = channels as usize;
    let stored_ch = stored_ch as usize;

    let mut candidates = Vec::new();

    for block_y in 0..blocks_y {
        for block_x in 0..blocks_x {
            let block_index = (block_y * blocks_x + block_x) as usize;
            let record = &records[block_index * record_len..(block_index + 1) * record_len];

            let block_origin_x = block_x * TEMPORAL_NOISE_BLOCK;
            let block_origin_y = block_y * TEMPORAL_NOISE_BLOCK;
            let block_width = TEMPORAL_NOISE_BLOCK.min(width - block_origin_x);
            let block_height = TEMPORAL_NOISE_BLOCK.min(height - block_origin_y);
            let pixel_count = (block_width * block_height) as f32;
            let n_pairs = (block_height * block_width.saturating_sub(1)) as f32;

            let mean0 = record[0] / pixel_count;
            if mean0.abs() >= STATIC_GATE {
                continue;
            }

            let mut sigmas = [0.0f32; 3];
            let mut sigma_ch0 = 0.0f32;
            let mut var_ch0 = 0.0f32;
            for channel in 0..channels {
                let mean = record[channel] / pixel_count;
                let variance = (record[stored_ch + channel] / pixel_count - mean * mean).max(0.0);
                let sigma_block = variance.sqrt() / std::f32::consts::SQRT_2;
                sigmas[channel] = sigma_block;
                if channel == 0 {
                    sigma_ch0 = sigma_block;
                    var_ch0 = variance;
                }
            }

            let mean_lag = if n_pairs > 0.0 {
                record[2 * stored_ch] / n_pairs
            } else {
                0.0
            };

            candidates.push(StaticGateCandidate {
                index: block_index,
                width: block_width,
                height: block_height,
                sigmas,
                sigma_ch0,
                var_ch0,
                mean0,
                mean_lag,
                n_pairs,
            });
        }
    }

    // Perfectly static blocks such as letterbox bars read a sigma of 0. Left in the reference they
    // drag the quartile to 0, which rejects every block with real noise.
    let mut reference_sigma_ch0: Vec<f32> = candidates
        .iter()
        .map(|candidate| candidate.sigma_ch0)
        .filter(|&sigma| sigma > RHO_SIGMA_GATE)
        .collect();
    sort_ascending(&mut reference_sigma_ch0);

    // A reference with no spread beside excluded blocks may be all texture setting its own
    // ceiling. With nothing excluded the population is trusted directly.
    let candidates_were_excluded = candidates.len() > reference_sigma_ch0.len();
    let reference_has_spread = match (reference_sigma_ch0.first(), reference_sigma_ch0.last()) {
        (Some(&lowest), Some(&highest)) => highest > lowest,
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

    let mut accepted = Vec::new();

    for candidate in candidates {
        if candidate.sigma_ch0 > sigma_ceiling {
            continue;
        }

        // A zero-sigma block carries no measurable noise, and a population of them would drag
        // the lower quartile to 0.
        if candidate.sigma_ch0 <= RHO_SIGMA_GATE {
            continue;
        }

        accepted.push(AcceptedBlock {
            index: candidate.index,
            width: candidate.width,
            height: candidate.height,
            sigmas: candidate.sigmas,
            mean0: candidate.mean0,
            mean_lag: candidate.mean_lag,
            var_ch0: candidate.var_ch0,
            n_pairs: candidate.n_pairs,
        });
    }

    Some(accepted)
}

/// The scalar sample [temporal_noise_reading] builds, without a curve.
#[cfg(test)]
pub(in crate::nlmeans) fn aggregate_temporal_noise_stats(
    records: &[f32],
    channels: u32,
    stored_ch: u32,
    width: u32,
    height: u32,
) -> Option<TemporalNoiseSample> {
    temporal_noise_reading(
        records,
        channels,
        stored_ch,
        width,
        height,
        false,
        QuarterSettings::default(),
    )
    .sample
}

/// One centre slot's temporal-noise reading.
pub(in crate::nlmeans) struct TemporalNoiseReading {
    pub(in crate::nlmeans) sample: Option<TemporalNoiseSample>,
    /// The per-frame luma noise curve, present only when requested and a sample exists.
    pub(in crate::nlmeans) curve: Option<NoiseCurve>,
    /// The frame's quarter classes, present exactly when `curve` is.
    pub(in crate::nlmeans) classes: Option<QuarterClasses>,
}

/// Builds one centre slot's [TemporalNoiseReading] from a single [accepted_static_blocks] call.
///
/// Sharing that selection keeps the scalar sample and the curve agreeing on which blocks are
/// static noise. The sample is `None` when the outlier ceiling cannot be trusted, when the static
/// fraction falls below [STATIC_FRACTION_MIN], or when no static block carries measurable noise,
/// as in a zero-filled duplicate slot.
///
/// The curve needs `with_curve` and a sample, since a frame too unreliable for a scalar sample is
/// too unreliable for a curve. `settings` control what [classify_quarters] does after classing the
/// quarters.
pub(in crate::nlmeans) fn temporal_noise_reading(
    records: &[f32],
    channels: u32,
    stored_ch: u32,
    width: u32,
    height: u32,
    with_curve: bool,
    settings: QuarterSettings,
) -> TemporalNoiseReading {
    let none = TemporalNoiseReading {
        sample: None,
        curve: None,
        classes: None,
    };

    let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
    let total_blocks = (blocks_x * blocks_y) as usize;

    let Some(accepted) = accepted_static_blocks(records, channels, stored_ch, width, height) else {
        return none;
    };

    let channels = channels as usize;

    let mut rho_samples = Vec::new();
    for block in &accepted {
        if block.n_pairs > 0.0 {
            let rho = (block.mean_lag - block.mean0 * block.mean0) / block.var_ch0;
            let clamped_rho = rho.clamp(0.0, 1.0);
            rho_samples.push(clamped_rho);
        }
    }

    let static_fraction = accepted.len() as f32 / total_blocks as f32;
    if static_fraction < STATIC_FRACTION_MIN || rho_samples.is_empty() {
        return none;
    }

    let mut static_sigmas: Vec<Vec<f32>> = vec![Vec::new(); channels];
    for block in &accepted {
        for (channel, sigmas) in static_sigmas.iter_mut().enumerate().take(channels) {
            sigmas.push(block.sigmas[channel]);
        }
    }

    let mut sigma = [0.0f32; 3];
    let mut sigma_low = [0.0f32; 3];
    for (channel, sigmas) in static_sigmas.iter_mut().enumerate() {
        sort_ascending(sigmas);
        sigma[channel] = median(sigmas);
        sigma_low[channel] = lower_quartile(sigmas);
    }

    sort_ascending(&mut rho_samples);
    let rho = median(&rho_samples);

    let sample = TemporalNoiseSample {
        sigma,
        sigma_low,
        rho,
        static_fraction,
    };

    let curve = if with_curve {
        build_noise_curve(records, stored_ch, &accepted, sample.sigma[0])
    } else {
        None
    };

    let classes = curve
        .as_ref()
        .map(|curve| classify_quarters(records, stored_ch, width, height, curve, settings));

    TemporalNoiseReading {
        sample: Some(sample),
        curve,
        classes,
    }
}

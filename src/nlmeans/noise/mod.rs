use cubecl::prelude::*;
use cubecl::server::Handle;

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
/// up to the GPU storage-buffer offset alignment (32 bytes). Small
/// frames (or a single-channel mode) can produce a
/// [`temporal_stats_slot_len`] under 8 `f32`s, and `wgpu` rejects a
/// bind-group offset that isn't a multiple of its
/// `min_storage_buffer_offset_alignment` — mirrors
/// `MotionCtx::confidence_bytes_per_neighbour`.
pub(super) fn temporal_stats_slot_stride_bytes(width: u32, height: u32, stored_ch: u32) -> u64 {
    let bytes = temporal_stats_slot_len(width, height, stored_ch) as u64 * size_of::<f32>() as u64;
    bytes.next_multiple_of(32)
}

/// Total byte size of a `frame_count`-slot temporal-stats ring.
pub(super) fn temporal_stats_buf_bytes(width: u32, height: u32, stored_ch: u32, frame_count: u32) -> usize {
    (temporal_stats_slot_stride_bytes(width, height, stored_ch) * frame_count as u64) as usize
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
    let stride = temporal_stats_slot_stride_bytes(ctx.width, ctx.height, ctx.stored_ch);
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
) {
    let slot_len = temporal_stats_slot_len(width, height, stored_ch) as u32;
    let stride = temporal_stats_slot_stride_bytes(width, height, stored_ch);
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
pub(super) fn read_temporal_stats_slot<R: Runtime>(
    client: &ComputeClient<R>,
    stats_buf: &Handle,
    width: u32,
    height: u32,
    stored_ch: u32,
    frame_count: u32,
    slot: u32,
) -> Result<Vec<f32>, anyhow::Error> {
    let slot_len_bytes = temporal_stats_slot_len(width, height, stored_ch) as u64 * size_of::<f32>() as u64;
    let stride = temporal_stats_slot_stride_bytes(width, height, stored_ch);
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

/// Aggregates one centre slot's per-block temporal-residual records
/// into a [`TemporalNoiseSample`]. `records` holds exactly one slot's
/// region, laid out block row-major as documented on
/// [`nlm_temporal_noise_stats`]. Returns `None` when static blocks
/// fall below [`STATIC_FRACTION_MIN`] (too much motion to trust) or
/// when no static block clears [`RHO_SIGMA_GATE`] (no measurable
/// noise to correlate — the median rho would be undefined — which is
/// also what a zero-filled duplicate slot's records produce).
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

    let mut static_sigmas: Vec<Vec<f32>> = vec![Vec::new(); channels];
    let mut rho_samples = Vec::new();
    let mut static_count = 0usize;

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
            static_count += 1;

            let mut sigma_ch0 = 0.0f32;
            let mut var_ch0 = 0.0f32;
            for (c, sigmas) in static_sigmas.iter_mut().enumerate() {
                let mean = rec[c] / n;
                let var = (rec[stored_ch + c] / n - mean * mean).max(0.0);
                let sigma_block = var.sqrt() / std::f32::consts::SQRT_2;
                sigmas.push(sigma_block);
                if c == 0 {
                    sigma_ch0 = sigma_block;
                    var_ch0 = var;
                }
            }

            if sigma_ch0 > RHO_SIGMA_GATE && n_pairs > 0.0 {
                let mean_lag = rec[2 * stored_ch] / n_pairs;
                rho_samples.push((mean_lag - mean0 * mean0) / var_ch0);
            }
        }
    }

    let static_fraction = static_count as f32 / total_blocks as f32;
    if static_fraction < STATIC_FRACTION_MIN || rho_samples.is_empty() {
        return None;
    }

    let mut sigma = [0.0f32; 3];
    for (c, sigmas) in static_sigmas.iter_mut().enumerate() {
        sigma[c] = median(sigmas);
    }
    let rho = median(&mut rho_samples);

    Some(TemporalNoiseSample {
        sigma,
        rho,
        static_fraction,
    })
}

/// Median of `values`, sorted in place. The average of the two middle
/// elements on an even count. Callers only invoke this on a
/// non-empty slice.
fn median(values: &mut [f32]) -> f32 {
    values.sort_by(|a, b| a.partial_cmp(b).expect("noise stats are never NaN"));
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// Piecewise-linear correlation-correction table, `(rho, factor)`
/// points sorted by `rho`. Currently an identity mapping with no
/// measured values landed.
const CORRELATION_FACTOR_TABLE: [(f32, f32); 2] = [(0.0, 1.0), (1.0, 1.0)];

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

/// Per-channel effective sigma from a temporal sample, correcting for
/// spatial grain correlation. Entries past `channels` stay 0.
pub(super) fn temporal_sigma_eff(sample: &TemporalNoiseSample, channels: usize) -> [f32; 3] {
    let factor = correlation_factor(sample.rho);
    let mut eff = [0.0f32; 3];
    for (c, e) in eff.iter_mut().enumerate().take(channels) {
        *e = sample.sigma[c] * factor;
    }
    eff
}

/// EMA weight for the newest frame estimate.
const EMA_ALPHA: f32 = 0.2;
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
    fn temporal_stats_blocks_and_slot_len() {
        assert_eq!(temporal_stats_blocks(32, 16), (2, 1));
        assert_eq!(temporal_stats_blocks(33, 17), (3, 2)); // ragged on both axes
        assert_eq!(temporal_stats_record_len(1), 3);
        assert_eq!(temporal_stats_record_len(4), 9);
        assert_eq!(temporal_stats_slot_len(32, 16, 1), 6); // 2 blocks x record_len 3
    }

    /// Two blocks, one static and one not. Only the static block
    /// should contribute to `sigma`/`rho`, and `static_fraction` must
    /// reflect the 1-of-2 split exactly.
    #[test]
    fn aggregate_gate_and_rho() {
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
    /// clear the rho gate) distinct rhos. Exercises both the odd-count
    /// sigma median and the even-count rho median in one pass.
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
    fn correlation_factor_is_identity_for_phase_a_placeholder_table() {
        for rho in [0.0, 0.2, 0.4, 0.65, 1.0] {
            assert_eq!(correlation_factor(rho), 1.0);
        }
    }

    #[test]
    fn temporal_sigma_eff_applies_correlation_factor_per_channel() {
        let sample = TemporalNoiseSample {
            sigma: [2.0 / 255.0, 4.0 / 255.0, 6.0 / 255.0],
            rho: 0.5,
            static_fraction: 1.0,
        };
        // Phase A's placeholder table is identity, so eff == sigma for
        // the active channels and 0 past them.
        let eff = temporal_sigma_eff(&sample, 2);
        assert!((eff[0] - sample.sigma[0]).abs() < 1e-6);
        assert!((eff[1] - sample.sigma[1]).abs() < 1e-6);
        assert_eq!(eff[2], 0.0);
    }
}

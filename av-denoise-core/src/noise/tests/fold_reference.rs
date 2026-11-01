use crate::collab::kernels::transforms::RECIPROCAL_FLOOR;
use crate::noise::NOISE_CHANNELS;
use crate::noise::correlation::correlation_factor;
use crate::noise::curve::{NOISE_CURVE_BINS, NoiseCurve};
use crate::noise::derived::{DerivedCoefficients, DerivedValues, NoiseConsumer};
use crate::noise::estimator::SIGMA_FLOOR;
use crate::noise::fold::NoiseSettings;
use crate::noise::history::WindowBounds;
use crate::noise::reading::ReadingRecord;
use crate::noise::snapshot::NoiseSnapshot;

/// One value per channel slot.
type ChannelValues = [f32; NOISE_CHANNELS as usize];

/// The largest gap in ulp the synthetic fold tests allow.
pub(super) const MAX_ULPS: u32 = 4;

/// Coefficients with every derived value in use, as an nl4d luma plane with motion compensation
/// and the NLM pilot sees them.
pub(super) const COEFFICIENTS: DerivedCoefficients = DerivedCoefficients {
    h2_per_inverse_variance: Some(3.5e-5),
    h2_fixed: 0.0,
    pilot_h2_factor: 0.64,
    offset_per_variance: 50.0,
    main_offset: true,
    sad_per_sigma: 120.0,
    confidence_sad_per_sigma: 180.0,
};

pub(super) const CONSUMER: NoiseConsumer = NoiseConsumer {
    curve_applies: true,
    weight_g_max: 1.3,
    weight_floor: RECIPROCAL_FLOOR,
};

/// The values of two snapshots' chains, rho and curve, after checking that the same ones exist.
pub(super) fn state_pairs(
    gpu: &NoiseSnapshot,
    host: &NoiseSnapshot,
    channels: u32,
) -> Vec<(&'static str, f32, f32)> {
    let active = channels as usize;
    let chains = [
        ("median", gpu.median, host.median),
        ("low", gpu.low, host.low),
        ("low_unboosted", gpu.low_unboosted, host.low_unboosted),
        ("temporal_only", gpu.temporal_only, host.temporal_only),
    ];
    let mut pairs = Vec::new();

    for (name, gpu_chain, host_chain) in chains {
        let gpu_present = gpu_chain.is_some();
        let host_present = host_chain.is_some();
        assert_eq!(gpu_present, host_present, "{name} chain presence");

        if let (Some(gpu_values), Some(host_values)) = (gpu_chain, host_chain) {
            for channel in 0..NOISE_CHANNELS as usize {
                if channel >= active {
                    assert_eq!(gpu_values[channel], 0.0, "{name} past the channel count");
                    continue;
                }

                pairs.push((name, gpu_values[channel], host_values[channel]));
            }
        }
    }

    let gpu_rho = gpu.rho.is_some();
    let host_rho = host.rho.is_some();
    assert_eq!(gpu_rho, host_rho, "rho presence");
    if let (Some(gpu_value), Some(host_value)) = (gpu.rho, host.rho) {
        pairs.push(("rho", gpu_value, host_value));
    }

    let gpu_curve = gpu.curve.is_some();
    let host_curve = host.curve.is_some();
    assert_eq!(gpu_curve, host_curve, "curve presence");

    // The curve is copied, never computed, so it matches bit for bit.
    let gpu_bits = gpu
        .curve
        .map(|(ratios, median)| (ratios.map(f32::to_bits), median.to_bits()));
    let host_bits = host
        .curve
        .map(|(ratios, median)| (ratios.map(f32::to_bits), median.to_bits()));
    assert_eq!(gpu_bits, host_bits, "curve");

    pairs
}

/// The values of two sets of derived values, after checking their curve flags match.
pub(super) fn derived_pairs(
    gpu: &DerivedValues,
    host: &DerivedValues,
    channels: u32,
) -> Vec<(&'static str, f32, f32)> {
    assert_eq!(gpu.curve_valid, host.curve_valid, "derived curve flag");

    let mut pairs = Vec::new();
    for channel in 0..NOISE_CHANNELS as usize {
        if channel >= channels as usize {
            assert_eq!(gpu.sigma[channel], 0.0, "derived sigma past the channel count");
            continue;
        }

        pairs.push(("sigma", gpu.sigma[channel], host.sigma[channel]));
    }

    let values = [
        ("weight_scale", gpu.weight_scale, host.weight_scale),
        ("sad_noise_floor", gpu.sad_noise_floor, host.sad_noise_floor),
        (
            "confidence_sad_noise_floor",
            gpu.confidence_sad_noise_floor,
            host.confidence_sad_noise_floor,
        ),
        ("h2_inv_norm", gpu.h2_inv_norm, host.h2_inv_norm),
        ("pilot_h2_inv_norm", gpu.pilot_h2_inv_norm, host.pilot_h2_inv_norm),
        ("noise_offset", gpu.noise_offset, host.noise_offset),
        (
            "input_noise_offset",
            gpu.input_noise_offset,
            host.input_noise_offset,
        ),
        ("rho", gpu.rho, host.rho),
    ];
    pairs.extend(values);

    pairs
}

pub(crate) fn relative_gap(gpu: f32, host: f32) -> f32 {
    if gpu == host {
        return 0.0;
    }

    let difference = (gpu - host).abs();
    let gap = difference / host.abs();

    if gap.is_finite() { gap } else { f32::INFINITY }
}

pub(super) fn largest_relative(pairs: &[(&'static str, f32, f32)]) -> f32 {
    pairs
        .iter()
        .map(|&(_, gpu, host)| relative_gap(gpu, host))
        .fold(0.0f32, f32::max)
}

pub(super) fn largest_gap(gpu: &[f32], host: &[f32]) -> f32 {
    gpu.iter()
        .zip(host)
        .map(|(&gpu_value, &host_value)| relative_gap(gpu_value, host_value))
        .fold(0.0f32, f32::max)
}

/// The largest relative gap of the values two snapshots both hold, and whether they hold the
/// same chains, rho and flags.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SnapshotGap {
    pub(crate) state: f32,
    pub(crate) derived: f32,
    pub(crate) structure_matches: bool,
}

impl SnapshotGap {
    pub(crate) fn measure(gpu: &NoiseSnapshot, host: &NoiseSnapshot, channels: usize) -> Self {
        let chains = [
            (gpu.median, host.median),
            (gpu.low, host.low),
            (gpu.low_unboosted, host.low_unboosted),
            (gpu.temporal_only, host.temporal_only),
        ];
        let mut state = 0.0f32;
        let mut structure_matches = true;

        for (gpu_chain, host_chain) in chains {
            structure_matches &= gpu_chain.is_some() == host_chain.is_some();

            if let (Some(gpu_values), Some(host_values)) = (gpu_chain, host_chain) {
                let chain_gap = largest_gap(&gpu_values[..channels], &host_values[..channels]);
                state = state.max(chain_gap);
            }
        }

        structure_matches &= gpu.rho.is_some() == host.rho.is_some();
        if let (Some(gpu_rho), Some(host_rho)) = (gpu.rho, host.rho) {
            let rho_gap = relative_gap(gpu_rho, host_rho);
            state = state.max(rho_gap);
        }

        let gpu_derived = gpu.derived;
        let host_derived = host.derived;
        structure_matches &= gpu_derived.curve_valid == host_derived.curve_valid;

        let derived_gap = derived_value_gaps(&gpu_derived, &host_derived);

        Self {
            state,
            derived: derived_gap,
            structure_matches,
        }
    }
}

/// The largest relative gap of two sets of derived values, leaving out the curve flag.
fn derived_value_gaps(gpu: &DerivedValues, host: &DerivedValues) -> f32 {
    let gpu_values = [
        gpu.sigma[0],
        gpu.sigma[1],
        gpu.weight_scale,
        gpu.sad_noise_floor,
        gpu.confidence_sad_noise_floor,
        gpu.h2_inv_norm,
        gpu.pilot_h2_inv_norm,
        gpu.noise_offset,
        gpu.input_noise_offset,
        gpu.rho,
    ];
    let host_values = [
        host.sigma[0],
        host.sigma[1],
        host.weight_scale,
        host.sad_noise_floor,
        host.confidence_sad_noise_floor,
        host.h2_inv_norm,
        host.pilot_h2_inv_norm,
        host.noise_offset,
        host.input_noise_offset,
        host.rho,
    ];

    largest_gap(&gpu_values, &host_values)
}

/// The relative gap of two curves, or infinite where only one side holds one.
pub(crate) fn curve_gap(gpu: &NoiseSnapshot, host: &NoiseSnapshot) -> f32 {
    match (gpu.curve, host.curve) {
        (Some((gpu_ratios, gpu_median)), Some((host_ratios, host_median))) => {
            let ratio_gap = largest_gap(&gpu_ratios, &host_ratios);
            let median_gap = relative_gap(gpu_median, host_median);
            ratio_gap.max(median_gap)
        },
        (None, None) => 0.0,
        _ => f32::INFINITY,
    }
}

/// One stream frame's reading as the host window model reads it.
#[derive(Debug, Clone)]
pub(crate) struct HostRecord {
    pub(crate) reading: ReadingRecord,
    /// The per-channel Immerkær sigma from the frame's abs sums.
    pub(crate) immerkaer: ChannelValues,
}

/// The chains, rho and curve one window fold gives.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HostDerived {
    pub(crate) median: ChannelValues,
    pub(crate) low: ChannelValues,
    pub(crate) low_unboosted: ChannelValues,
    /// `None` when no frame of the window has a sample.
    pub(crate) temporal_only: Option<ChannelValues>,
    pub(crate) rho: Option<f32>,
    /// The curve's ratios and quarter median.
    pub(crate) curve: Option<([f32; NOISE_CURVE_BINS], f32)>,
}

impl HostDerived {
    /// The window as a GPU snapshot holds it, with channels past `channels` at 0 and the derived
    /// values `derive_host` gives it.
    pub(crate) fn snapshot(&self, settings: NoiseSettings, channels: u32) -> NoiseSnapshot {
        let active = channels as usize;
        let bare = NoiseSnapshot {
            median: Some(active_channels(self.median, active)),
            low: Some(active_channels(self.low, active)),
            low_unboosted: Some(active_channels(self.low_unboosted, active)),
            temporal_only: self.temporal_only.map(|chain| active_channels(chain, active)),
            rho: self.rho,
            curve: self.curve,
            derived: DerivedValues::default(),
        };
        let fold = settings.fold;
        let derived = fold.coefficients.derive_host(&bare, active, fold.consumer);

        NoiseSnapshot { derived, ..bare }
    }
}

fn active_channels(values: ChannelValues, channels: usize) -> ChannelValues {
    let mut active = [0.0f32; NOISE_CHANNELS as usize];
    active[..channels].copy_from_slice(&values[..channels]);
    active
}

/// The median of `values`, the mean of the two middle values for an even count.
fn window_median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);

    let count = values.len();
    let lower = values[(count - 1) / 2];
    let upper = values[count / 2];

    (lower + upper) * 0.5
}

/// One frame's combined and scaled readings for the median, low and unboosted low chains.
fn frame_readings(record: &HostRecord, sigma_scale: f32) -> [ChannelValues; 3] {
    let reading = &record.reading;
    let mut raw = record.immerkaer;
    let mut raw_low = reading.immerkaer_low;
    let mut raw_unboosted = reading.immerkaer_low;

    if let Some(sample) = reading.sample {
        let factor = correlation_factor(sample.rho);
        for channel in 0..NOISE_CHANNELS as usize {
            raw[channel] = raw[channel].max(sample.sigma[channel] * factor);
            raw_low[channel] = raw_low[channel].max(sample.sigma_low[channel] * factor);
            raw_unboosted[channel] = raw_unboosted[channel].max(sample.sigma_low[channel]);
        }
    }

    let scale = |values: ChannelValues| values.map(|value| value * sigma_scale);

    [scale(raw), scale(raw_low), scale(raw_unboosted)]
}

/// The per-channel floored median of `readings`, `None` for no readings.
fn chain_median(readings: &[ChannelValues]) -> Option<ChannelValues> {
    if readings.is_empty() {
        return None;
    }

    let mut chain = [0.0f32; NOISE_CHANNELS as usize];
    for (channel, value) in chain.iter_mut().enumerate() {
        let mut values: Vec<f32> = readings.iter().map(|reading| reading[channel]).collect();
        let median = window_median(&mut values);
        *value = median.max(SIGMA_FLOOR);
    }

    Some(chain)
}

/// The host model of one window fold of `records`, indexed by stream frame, over `bounds`.
pub(crate) fn fold_window(records: &[HostRecord], bounds: WindowBounds, sigma_scale: f32) -> HostDerived {
    let first = bounds.first() as usize;
    let last = bounds.last() as usize;
    let window = &records[first..=last];
    let sampled: Vec<&HostRecord> = window
        .iter()
        .filter(|record| record.reading.sample.is_some())
        .collect();
    let curved: Vec<&HostRecord> = sampled
        .iter()
        .copied()
        .filter(|record| record.reading.curve.is_some())
        .collect();

    let readings: Vec<[ChannelValues; 3]> = window
        .iter()
        .map(|record| frame_readings(record, sigma_scale))
        .collect();
    let medians: Vec<ChannelValues> = readings.iter().map(|reading| reading[0]).collect();
    let lows: Vec<ChannelValues> = readings.iter().map(|reading| reading[1]).collect();
    let unboosted: Vec<ChannelValues> = readings.iter().map(|reading| reading[2]).collect();
    let temporal: Vec<ChannelValues> = sampled
        .iter()
        .map(|record| record.reading.sample.unwrap().sigma)
        .map(|sigma| sigma.map(|value| value * sigma_scale))
        .collect();

    let mut rhos: Vec<f32> = sampled
        .iter()
        .map(|record| record.reading.sample.unwrap().rho)
        .collect();
    let rho = (!rhos.is_empty()).then(|| window_median(&mut rhos));

    let curve_present = !sampled.is_empty() && 2 * curved.len() >= sampled.len();
    let curve = curve_present.then(|| window_curve(&curved));

    HostDerived {
        median: chain_median(&medians).expect("the window holds a frame"),
        low: chain_median(&lows).expect("the window holds a frame"),
        low_unboosted: chain_median(&unboosted).expect("the window holds a frame"),
        temporal_only: chain_median(&temporal),
        rho,
        curve,
    }
}

/// The elementwise median ratios and quarter median of `curved`, which all hold a curve.
fn window_curve(curved: &[&HostRecord]) -> ([f32; NOISE_CURVE_BINS], f32) {
    let curves: Vec<NoiseCurve> = curved
        .iter()
        .map(|record| record.reading.curve.unwrap())
        .collect();
    let mut ratios = [0.0f32; NOISE_CURVE_BINS];

    for (bin, ratio) in ratios.iter_mut().enumerate() {
        let mut values: Vec<f32> = curves.iter().map(|curve| curve.ratios[bin]).collect();
        *ratio = window_median(&mut values);
    }

    let mut quarter_medians: Vec<f32> = curves.iter().map(|curve| curve.sigma_quarter_median).collect();
    let quarter_median = window_median(&mut quarter_medians);

    (ratios, quarter_median)
}

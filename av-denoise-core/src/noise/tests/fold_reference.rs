use super::records::ulp_gap;
use crate::collab::kernels::transforms::RECIPROCAL_FLOOR;
use crate::noise::curve::NoiseCurve;
use crate::noise::derived::{DerivedCoefficients, DerivedValues, NoiseConsumer};
use crate::noise::estimator::NoiseEstimator;
use crate::noise::fold::NoiseSettings;
use crate::noise::host::HostNoise;
use crate::noise::reading::ReadingRecord;
use crate::noise::snapshot::NoiseSnapshot;
use crate::noise::strength_map::QuarterClasses;
use crate::noise::temporal::TemporalNoiseReading;

/// The largest gap in ulp the synthetic fold tests allow.
pub(super) const MAX_ULPS: u32 = 2;

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

/// The host reading a fold takes from `record`, with placeholder classes wherever it has a curve.
pub(super) fn host_reading(record: &ReadingRecord) -> TemporalNoiseReading {
    let classes = record
        .curve
        .map(|_| QuarterClasses::from_classes(1, 1, vec![None]));

    TemporalNoiseReading {
        sample: record.sample,
        curve: record.curve,
        classes,
    }
}

/// The host state as a snapshot, with the derived values `derive_host` gives it.
pub(super) fn host_snapshot(host: &HostNoise, settings: NoiseSettings, channels: u32) -> NoiseSnapshot {
    let active = channels as usize;
    let bare = host.snapshot(active, DerivedValues::default());
    let fold = settings.fold;
    let derived = fold
        .coefficients
        .derive_host(&bare, active, fold.consumer, settings.reading.map);

    NoiseSnapshot { derived, ..bare }
}

fn estimator(chain: Option<[f32; 3]>, channels: u32) -> NoiseEstimator {
    let active = channels as usize;
    let values = chain.map(|values| values[..active].to_vec());

    NoiseEstimator::from_current(values.as_deref())
}

/// A host state holding exactly what `snapshot` holds.
pub(super) fn host_from_snapshot(snapshot: &NoiseSnapshot, channels: u32) -> HostNoise {
    let curve = snapshot.curve.map(|(ratios, sigma_quarter_median)| NoiseCurve {
        ratios,
        sigma_quarter_median,
    });
    let classes = snapshot
        .classes_present
        .then(|| QuarterClasses::from_classes(1, 1, vec![None]));

    HostNoise {
        median: estimator(snapshot.median, channels),
        low: estimator(snapshot.low, channels),
        low_unboosted: estimator(snapshot.low_unboosted, channels),
        temporal_only: estimator(snapshot.temporal_only, channels),
        rho: snapshot.rho,
        curve,
        classes,
    }
}

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
            for channel in 0..3 {
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
    assert_eq!(gpu.classes_present, host.classes_present, "classes presence");

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

/// The values of two sets of derived values, after checking their flags and modes match.
pub(super) fn derived_pairs(
    gpu: &DerivedValues,
    host: &DerivedValues,
    channels: u32,
) -> Vec<(&'static str, f32, f32)> {
    assert_eq!(gpu.curve_valid, host.curve_valid, "derived curve flag");
    assert_eq!(gpu.map_mode, host.map_mode, "derived map mode");

    let mut pairs = Vec::new();
    for channel in 0..3 {
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

pub(super) fn relative_gap(gpu: f32, host: f32) -> f32 {
    if gpu == host {
        return 0.0;
    }

    let difference = (gpu - host).abs();
    let gap = difference / host.abs();

    if gap.is_finite() { gap } else { f32::INFINITY }
}

pub(super) fn largest_ulps(pairs: &[(&'static str, f32, f32)]) -> u32 {
    pairs
        .iter()
        .map(|&(_, gpu, host)| ulp_gap(gpu, host))
        .max()
        .unwrap_or(0)
}

pub(super) fn largest_relative(pairs: &[(&'static str, f32, f32)]) -> f32 {
    pairs
        .iter()
        .map(|&(_, gpu, host)| relative_gap(gpu, host))
        .fold(0.0f32, f32::max)
}

pub(super) fn assert_ulps(pairs: &[(&'static str, f32, f32)], what: &str) {
    for &(name, gpu, host) in pairs {
        let gap = ulp_gap(gpu, host);
        assert!(
            gap <= MAX_ULPS,
            "{what}: {name} is {gap} ulp apart, gpu {gpu:e} host {host:e}"
        );
    }
}

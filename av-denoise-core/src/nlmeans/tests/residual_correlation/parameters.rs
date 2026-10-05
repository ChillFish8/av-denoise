use super::{Sample, measure_flat};
use crate::nlmeans::tests::helpers::*;

/// The standard deviation an `a, 1 - 2a, a` horizontal blur leaves on white noise of `sigma_pre`.
fn tap_sigma(sigma_pre: f32, tap: f32) -> f32 {
    let centre_tap = 1.0 - 2.0 * tap;
    sigma_pre * (2.0 * tap * tap + centre_tap * centre_tap).sqrt()
}

/// Overlapping search windows make the residual more correlated than the input grain.
///
/// A search radius of 0 removes the overlap and isolates that mechanism. Every input is a
/// horizontal-only blur, so its vertical correlation is near 0.
#[test]
fn nlm_residual_correlation_exceeds_input_correlation_and_tracks_window_overlap() {
    let client = make_client();
    let width = 160;
    let height = 160;
    let base = 0.5f32;
    let sigma_pre = 0.06f32;

    struct Case {
        label: &'static str,
        rho_in_h: f64,
        rho_in_v: f64,
    }
    let cases = [
        Case {
            label: "rho_in=0.00",
            rho_in_h: 0.0,
            rho_in_v: 0.0,
        },
        Case {
            label: "rho_in=0.32",
            rho_in_h: 0.316,
            rho_in_v: 0.0,
        },
        Case {
            label: "rho_in=0.67",
            rho_in_h: 2.0 / 3.0,
            rho_in_v: 0.0,
        },
    ];

    let configs = [(2u32, 0u32), (2u32, 2u32), (0u32, 0u32), (0u32, 2u32)];

    eprintln!(
        "residual correlation sweep (w={width} h={height} sigma_pre={sigma_pre}):\n\
         {:<12} {:>6} {:>6} | {:>6} {:>6} | {:>8} {:>8} | {:>8}",
        "input", "R_s", "R_t", "rho_in_h", "rho_in_v", "rho_out_h", "rho_out_v", "sig_ratio"
    );

    struct Row {
        label: &'static str,
        search_radius: u32,
        temporal_radius: u32,
        rho_in_h: f64,
        measurement: Sample,
    }
    let mut rows = Vec::new();

    let clean = vec![base; (width * height) as usize];
    for case in &cases {
        for &(search_radius, temporal_radius) in &configs {
            let measurement = match case.label {
                "rho_in=0.00" => {
                    let sigma = tap_sigma(sigma_pre, 0.0);
                    measure_flat(
                        &client,
                        width,
                        height,
                        &clean,
                        sigma,
                        |clean, seed| noisy_field_over(clean, width, height, sigma_pre, seed),
                        search_radius,
                        temporal_radius,
                        2,
                    )
                },
                "rho_in=0.32" => {
                    let sigma = tap_sigma(sigma_pre, 0.125);
                    measure_flat(
                        &client,
                        width,
                        height,
                        &clean,
                        sigma,
                        |_clean, seed| {
                            correlated_noisy_frame_with_tap(width, height, base, sigma_pre, seed, 0.125)
                        },
                        search_radius,
                        temporal_radius,
                        2,
                    )
                },
                _ => {
                    let sigma = tap_sigma(sigma_pre, 0.25);
                    measure_flat(
                        &client,
                        width,
                        height,
                        &clean,
                        sigma,
                        |_clean, seed| correlated_noisy_frame(width, height, base, sigma_pre, seed),
                        search_radius,
                        temporal_radius,
                        2,
                    )
                },
            };

            eprintln!(
                "{:<12} {:>6} {:>6} | {:>8.4} {:>8.4} | {:>8.4} {:>8.4} | {:>8.4}",
                case.label,
                search_radius,
                temporal_radius,
                case.rho_in_h,
                case.rho_in_v,
                measurement.rho_out_h,
                measurement.rho_out_v,
                measurement.sigma_ratio
            );

            rows.push(Row {
                label: case.label,
                search_radius,
                temporal_radius,
                rho_in_h: case.rho_in_h,
                measurement,
            });
        }
    }

    // At search radius 2, window overlap alone raises the residual's horizontal correlation above
    // the input's.
    for row in &rows {
        if row.search_radius == 2 {
            assert!(
                row.measurement.rho_out_h > row.rho_in_h,
                "{} at search_radius=2 temporal_radius={}: residual rho_h={:.4} did not exceed \
                 input rho_h={:.4}, expected window overlap to raise it",
                row.label,
                row.temporal_radius,
                row.measurement.rho_out_h,
                row.rho_in_h
            );
        }
    }

    // At search radius 0 there is no window overlap, so the residual tracks the input. The 0.08
    // tolerance covers the real shift of up to about 0.025 that temporal radius 2 adds, not
    // measurement noise, since every generator is seeded.
    for row in &rows {
        if row.search_radius == 0 {
            assert!(
                (row.measurement.rho_out_h - row.rho_in_h).abs() < 0.08,
                "{} at search_radius=0 temporal_radius={}: residual rho_h={:.4} should track the \
                 input's rho_h={:.4} within 0.08 once spatial window overlap is removed",
                row.label,
                row.temporal_radius,
                row.measurement.rho_out_h,
                row.rho_in_h
            );
        }
    }

    // With no spatial window, nothing can create vertical correlation the input never had.
    for row in &rows {
        if row.search_radius == 0 {
            assert!(
                row.measurement.rho_out_v.abs() < 0.05,
                "{} at search_radius=0 temporal_radius={}: residual rho_v={:.4} should stay near \
                 zero, the input never carried vertical correlation and there is no spatial \
                 window to manufacture it",
                row.label,
                row.temporal_radius,
                row.measurement.rho_out_v
            );
        }
    }

    // The square search window mixes vertical neighbours too, so it creates strong vertical
    // correlation even from horizontal-only grain.
    for row in &rows {
        if row.search_radius == 2 {
            assert!(
                row.measurement.rho_out_v > 0.3,
                "{} at search_radius=2 temporal_radius={}: residual rho_v={:.4} did not rise \
                 well above the input's zero vertical correlation, expected the isotropic \
                 window to induce substantial vertical correlation regardless",
                row.label,
                row.temporal_radius,
                row.measurement.rho_out_v
            );
        }
    }
}

/// Varies patch radius, temporal radius and input correlation one at a time from a baseline of
/// search radius 2, patch radius 2, temporal radius 0 and uncorrelated flat input.
#[test]
fn nlm_residual_correlation_patch_radius_temporal_radius_and_input_correlation() {
    let client = make_client();
    let width = 160;
    let height = 160;
    let base = 0.5f32;
    let sigma_pre = 0.06f32;
    let flat_clean = vec![base; (width * height) as usize];
    let search_radius = 2;

    let baseline = measure_flat(
        &client,
        width,
        height,
        &flat_clean,
        sigma_pre,
        |clean, seed| noisy_field_over(clean, width, height, sigma_pre, seed),
        search_radius,
        0,
        2,
    );

    let patch_radius_4 = measure_flat(
        &client,
        width,
        height,
        &flat_clean,
        sigma_pre,
        |clean, seed| noisy_field_over(clean, width, height, sigma_pre, seed),
        search_radius,
        0,
        4,
    );

    let temporal_radius_2 = measure_flat(
        &client,
        width,
        height,
        &flat_clean,
        sigma_pre,
        |clean, seed| noisy_field_over(clean, width, height, sigma_pre, seed),
        search_radius,
        2,
        2,
    );

    let rho_in_h = 2.0 / 3.0;
    let correlated_input = measure_flat(
        &client,
        width,
        height,
        &flat_clean,
        sigma_pre,
        |_clean, seed| correlated_noisy_frame(width, height, base, sigma_pre, seed),
        search_radius,
        0,
        2,
    );

    eprintln!(
        "secondary checks at search_radius=2, flat content (w={width} h={height} sigma_pre={sigma_pre}):\n\
         {:<32} {:>8} {:>8} {:>8}",
        "config", "rho_h", "rho_v", "sig_ratio"
    );
    for (label, sample) in [
        ("baseline patch_radius=2 R_t=0 rho_in=0", &baseline),
        ("patch_radius=4", &patch_radius_4),
        ("temporal_radius=2", &temporal_radius_2),
        ("input rho_h=0.67", &correlated_input),
    ] {
        eprintln!(
            "{:<32} {:>8.4} {:>8.4} {:>8.4}",
            label, sample.rho_out_h, sample.rho_out_v, sample.sigma_ratio
        );
    }

    for (label, sample) in [
        ("baseline", &baseline),
        ("patch_radius=4", &patch_radius_4),
        ("temporal_radius=2", &temporal_radius_2),
        ("input rho_h=0.67", &correlated_input),
    ] {
        assert!(
            sample.sigma_ratio < 0.9,
            "{label}: sigma_ratio={:.4} too close to 1.0, no real smoothing happened, this \
             configuration's correlation number is not valid data",
            sample.sigma_ratio
        );
    }

    // Temporal averaging dilutes the window-induced correlation slightly, by about 0.02 to 0.05 at
    // search radius 2.
    assert!(
        temporal_radius_2.rho_out_h < baseline.rho_out_h,
        "temporal_radius=2 rho_out_h={:.4} should be below temporal_radius=0's {:.4}, temporal \
         averaging is expected to dilute the spatial window's contribution",
        temporal_radius_2.rho_out_h,
        baseline.rho_out_h
    );

    // The window sets a floor and correlated input lifts the residual further, within the headroom
    // below 1.
    assert!(
        correlated_input.rho_out_h > baseline.rho_out_h,
        "correlated input (rho_in_h={rho_in_h:.4}) rho_out_h={:.4} should exceed the uncorrelated \
         baseline's {:.4}",
        correlated_input.rho_out_h,
        baseline.rho_out_h
    );
    assert!(
        correlated_input.rho_out_h < 1.0,
        "correlated input rho_out_h={:.4} must stay below 1.0",
        correlated_input.rho_out_h
    );
}

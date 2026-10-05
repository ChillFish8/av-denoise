use super::{Sample, measure_diff, measure_flat};
use crate::nlmeans::tests::helpers::*;

/// On texture the Welsch weighting suppresses candidates whose patch does not match, which could
/// lower the induced correlation, so textured content is measured alongside flat.
#[test]
fn nlm_residual_correlation_search_radius_sweep_flat_vs_textured() {
    let client = make_client();
    let width = 160;
    let height = 160;
    let base = 0.5f32;
    let sigma_pre = 0.06f32;
    let patch_radius = 2;

    let flat_clean = vec![base; (width * height) as usize];
    let textured_clean = make_textured_frame(width, height);

    struct Row {
        search_radius: u32,
        flat: Sample,
        textured: Sample,
    }
    let mut rows = Vec::new();

    eprintln!(
        "search radius sweep, flat vs textured (w={width} h={height} sigma_pre={sigma_pre} \
         patch_radius={patch_radius}, uncorrelated input):\n\
         {:>3} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8}",
        "R_s", "flat_h", "flat_v", "flat_sig", "tex_h", "tex_v", "tex_sig"
    );

    for &search_radius in &[1u32, 2, 3, 4] {
        let flat = measure_flat(
            &client,
            width,
            height,
            &flat_clean,
            sigma_pre,
            |clean, seed| noisy_field_over(clean, width, height, sigma_pre, seed),
            search_radius,
            0,
            patch_radius,
        );
        let textured = measure_diff(
            &client,
            width,
            height,
            &textured_clean,
            sigma_pre,
            |clean, seed| noisy_field_over(clean, width, height, sigma_pre, seed),
            search_radius,
            0,
            patch_radius,
        );

        eprintln!(
            "{:>3} | {:>8.4} {:>8.4} {:>8.4} | {:>8.4} {:>8.4} {:>8.4}",
            search_radius,
            flat.rho_out_h,
            flat.rho_out_v,
            flat.sigma_ratio,
            textured.rho_out_h,
            textured.rho_out_v,
            textured.sigma_ratio
        );

        rows.push(Row {
            search_radius,
            flat,
            textured,
        });
    }

    // A sigma ratio near 1.0 means almost no smoothing happened, so the correlation is not valid
    // data.
    for row in &rows {
        assert!(
            row.flat.sigma_ratio < 0.9,
            "search_radius={}: flat sigma_ratio={:.4} too close to 1.0, no real smoothing \
             happened, this configuration's correlation number is not valid data",
            row.search_radius,
            row.flat.sigma_ratio
        );
        assert!(
            row.textured.sigma_ratio < 0.9,
            "search_radius={}: textured sigma_ratio={:.4} too close to 1.0, no real smoothing \
             happened, this configuration's correlation number is not valid data",
            row.search_radius,
            row.textured.sigma_ratio
        );
    }

    // Window overlap induces real correlation on flat content at every shipping search radius.
    for row in &rows {
        assert!(
            row.flat.rho_out_h > 0.3,
            "search_radius={}: flat rho_out_h={:.4} did not show substantial window-induced \
             correlation",
            row.search_radius,
            row.flat.rho_out_h
        );
    }

    // The sinusoidal texture tracks the flat field closely at every radius, with a measured
    // maximum gap of 0.0234 at search radius 4. The 0.08 tolerance leaves margin above that.
    for row in &rows {
        assert!(
            (row.flat.rho_out_h - row.textured.rho_out_h).abs() < 0.08,
            "search_radius={}: flat rho_out_h={:.4} and textured rho_out_h={:.4} diverge by more \
             than the measured tolerance, flat and textured no longer agree",
            row.search_radius,
            row.flat.rho_out_h,
            row.textured.rho_out_h
        );
    }

    // A wider window only adds overlapping candidates, so the induced correlation rises or
    // plateaus.
    for pair in rows.windows(2) {
        assert!(
            pair[1].flat.rho_out_h >= pair[0].flat.rho_out_h - 1e-6,
            "flat rho_out_h dropped from search_radius={} ({:.4}) to search_radius={} ({:.4})",
            pair[0].search_radius,
            pair[0].flat.rho_out_h,
            pair[1].search_radius,
            pair[1].flat.rho_out_h
        );
    }
}

/// The patch radius shifts the result noticeably, with 0.7770 at the default of 4 against 0.7006
/// at 2 for search radius 2. Search radius 0 is included as the case with no window overlap.
#[test]
fn nlm_residual_correlation_search_radius_sweep_at_shipped_patch_radius() {
    let client = make_client();
    let width = 160;
    let height = 160;
    let base = 0.5f32;
    let sigma_pre = 0.06f32;
    let patch_radius = 4;

    let flat_clean = vec![base; (width * height) as usize];
    let textured_clean = make_textured_frame(width, height);

    struct Row {
        search_radius: u32,
        flat: Sample,
        textured: Sample,
    }
    let mut rows = Vec::new();

    eprintln!(
        "search radius sweep at shipped patch_radius={patch_radius}, flat vs textured \
         (w={width} h={height} sigma_pre={sigma_pre}, uncorrelated input):\n\
         {:>3} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8}",
        "R_s", "flat_h", "flat_v", "flat_sig", "tex_h", "tex_v", "tex_sig"
    );

    for &search_radius in &[0u32, 1, 2, 3, 4] {
        let flat = measure_flat(
            &client,
            width,
            height,
            &flat_clean,
            sigma_pre,
            |clean, seed| noisy_field_over(clean, width, height, sigma_pre, seed),
            search_radius,
            0,
            patch_radius,
        );
        let textured = measure_diff(
            &client,
            width,
            height,
            &textured_clean,
            sigma_pre,
            |clean, seed| noisy_field_over(clean, width, height, sigma_pre, seed),
            search_radius,
            0,
            patch_radius,
        );

        eprintln!(
            "{:>3} | {:>8.4} {:>8.4} {:>8.4} | {:>8.4} {:>8.4} {:>8.4}",
            search_radius,
            flat.rho_out_h,
            flat.rho_out_v,
            flat.sigma_ratio,
            textured.rho_out_h,
            textured.rho_out_v,
            textured.sigma_ratio
        );

        rows.push(Row {
            search_radius,
            flat,
            textured,
        });
    }

    // Search radius 0 is exempt, because with no spatial window almost no smoothing happens and
    // its sigma ratio sits near 1.0 by construction.
    for row in &rows {
        if row.search_radius == 0 {
            continue;
        }

        assert!(
            row.flat.sigma_ratio < 0.9,
            "search_radius={}: flat sigma_ratio={:.4} too close to 1.0, no real smoothing \
             happened, this configuration's correlation number is not valid data",
            row.search_radius,
            row.flat.sigma_ratio
        );
        assert!(
            row.textured.sigma_ratio < 0.9,
            "search_radius={}: textured sigma_ratio={:.4} too close to 1.0, no real smoothing \
             happened, this configuration's correlation number is not valid data",
            row.search_radius,
            row.textured.sigma_ratio
        );
    }

    // With no window to overlap, the residual stays near the uncorrelated input's 0.
    for row in &rows {
        if row.search_radius == 0 {
            assert!(
                row.flat.rho_out_h.abs() < 0.1,
                "search_radius=0: flat rho_out_h={:.4} should stay near zero, there is no \
                 spatial window to manufacture correlation",
                row.flat.rho_out_h
            );
        }
    }

    // Flat content shows substantial window-induced correlation at every radius with a window.
    for row in &rows {
        if row.search_radius >= 1 {
            assert!(
                row.flat.rho_out_h > 0.3,
                "search_radius={}: flat rho_out_h={:.4} did not show substantial window-induced \
                 correlation",
                row.search_radius,
                row.flat.rho_out_h
            );
        }
    }

    // A larger patch makes each candidate's weight less noisy, which lets the window's
    // near-uniform weighting through and raises the correlation above the patch radius 2 values.
    for row in &rows {
        if row.search_radius == 0 {
            continue;
        }

        assert!(
            row.flat.rho_out_h > 0.6,
            "search_radius={}: patch_radius=4 flat rho_out_h={:.4} unexpectedly low, expected it \
             to sit above the patch_radius=2 sweep's own values at this radius",
            row.search_radius,
            row.flat.rho_out_h
        );
    }

    // Residual correlation should not fall as the window widens.
    for pair in rows.windows(2) {
        assert!(
            pair[1].flat.rho_out_h >= pair[0].flat.rho_out_h - 1e-6,
            "flat rho_out_h dropped from search_radius={} ({:.4}) to search_radius={} ({:.4})",
            pair[0].search_radius,
            pair[0].flat.rho_out_h,
            pair[1].search_radius,
            pair[1].flat.rho_out_h
        );
    }
}

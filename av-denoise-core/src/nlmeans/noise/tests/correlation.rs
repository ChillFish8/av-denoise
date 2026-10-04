use crate::nlmeans::noise::correlation::{
    build_spatial_offset_lut,
    correlation_factor,
    interpolate_table,
    spatial_offset_factor,
    spatial_offset_lut_len,
};

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
        let factor = correlation_factor(i as f32 / 20.0);
        assert!(factor >= last, "factor must not decrease, {factor} < {last}");
        last = factor;
    }
}

#[test]
fn spatial_offset_factor_rho_zero_is_white_identity() {
    // A negative rho, which the aggregation never produces, behaves like zero.
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
    for distance in 1..=8 {
        let factor = spatial_offset_factor(distance, 0, rho);
        assert!(
            factor >= last,
            "factor decreased at d={distance}: {factor} < {last}"
        );
        last = factor;
    }
}

#[test]
fn spatial_offset_factor_rho_0_65_shape() {
    let rho = 0.65f32;
    // One pixel away.
    assert!((spatial_offset_factor(1, 0, rho) - (1.0 - rho)).abs() < 1e-6);
    // Two pixels away along an axis.
    assert!((spatial_offset_factor(2, 0, rho) - (1.0 - rho * rho)).abs() < 1e-6);

    // A diagonal candidate sits sqrt(2) away.
    let distance = 2.0f32.sqrt();
    let log_rho = rho.ln();
    let expected = 1.0 - (distance * log_rho).exp();
    assert!((spatial_offset_factor(1, 1, rho) - expected).abs() < 1e-6);

    // Every factor stays within 0..=1.
    for dy in -4..=4 {
        for dx in -4..=4 {
            let factor = spatial_offset_factor(dx, dy, rho);
            assert!((0.0..=1.0).contains(&factor), "dx={dx} dy={dy} factor={factor}");
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
    let radius = search_radius as i32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let index = ((dy + radius) as usize) * side + (dx + radius) as usize;
            if dx == 0 && dy == 0 {
                assert_eq!(lut[index], 0.0, "self offset must be zero");
            } else {
                assert_eq!(lut[index], noise_offset, "dx={dx} dy={dy}");
            }
        }
    }
}

#[test]
fn build_spatial_offset_lut_indexes_row_major_by_dy_then_dx() {
    let search_radius = 2;
    let lut = build_spatial_offset_lut(search_radius, 0.65, 10.0);
    let side = (2 * search_radius + 1) as usize;

    // One step right lands at row 2, column 3, which is index 13.
    let expected = 10.0 * spatial_offset_factor(1, 0, 0.65);
    assert_eq!(lut[2 * side + 3], expected);

    // dx=0, dy=-2 lands at row 0, column 2, which is index 2.
    let expected = 10.0 * spatial_offset_factor(0, -2, 0.65);
    assert_eq!(lut[2], expected);

    // The centre is always zero.
    assert_eq!(lut[2 * side + 2], 0.0);
}

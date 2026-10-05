use crate::nlmeans::noise::stats::lower_quartile;

#[test]
fn lower_quartile_odd_count_exact_index() {
    // With 5 values the quartile lands exactly on index 1.
    let got = lower_quartile(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(got, 2.0);
}

#[test]
fn lower_quartile_even_count() {
    // With 4 values the quartile lands at 0.75, between the first two.
    let got = lower_quartile(&[1.0, 2.0, 3.0, 4.0]);
    assert!((got - 1.75).abs() < 1e-6, "expected 1.75, got {got}");
}

#[test]
fn lower_quartile_interpolates_at_a_fractional_index() {
    // With 3 values the quartile lands halfway between the first two.
    let got = lower_quartile(&[10.0, 20.0, 30.0]);
    assert!((got - 15.0).abs() < 1e-6, "expected 15.0, got {got}");
}

#[test]
fn lower_quartile_single_element_returns_it() {
    let got = lower_quartile(&[42.0]);
    assert_eq!(got, 42.0);
}

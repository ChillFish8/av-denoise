use crate::tune::{
    CompareResult,
    ErrorBound,
    Fidelity,
    FidelityClass,
    IntegerBound,
    PlaneDiff,
    diff_integers,
    diff_plane,
};

#[test]
fn identical_planes_are_identical_with_infinite_psnr() {
    let plane = [0.25f32, 0.5, 0.75];
    let diff = diff_plane(&plane, &plane);

    assert!(diff.identical);
    assert_eq!(diff.max_abs, 0.0);
    assert!(diff.psnr.is_infinite());
}

#[test]
fn a_single_ulp_breaks_identity_but_keeps_a_huge_psnr() {
    let baseline = [0.5f32, 0.5];
    let nudged_value = f32::from_bits(0.5f32.to_bits() + 1);
    let candidate = [0.5f32, nudged_value];
    let diff = diff_plane(&baseline, &candidate);

    assert!(!diff.identical);
    assert!(diff.max_abs > 0.0);
    assert!(diff.psnr > 120.0, "psnr {}", diff.psnr);
}

#[test]
fn a_nan_in_the_candidate_is_an_infinite_difference() {
    let baseline = [0.5f32, 0.5];
    let candidate = [0.5f32, f32::NAN];
    let diff = diff_plane(&baseline, &candidate);

    assert!(diff.max_abs.is_infinite());
    assert!(diff.psnr < 1.0);
}

#[test]
fn integer_planes_compare_exactly() {
    let baseline = [1_000_000_001i32, -7];
    let candidate = [1_000_000_000i32, -7];
    let same = diff_integers(&baseline, &baseline);
    let different = diff_integers(&baseline, &candidate);

    assert!(same.identical);
    assert!(!different.identical);
    assert_eq!(different.max_abs, 1.0);
}

#[test]
fn exact_rejects_any_bit_change() {
    let baseline = [0.5f32];
    let nudged_value = f32::from_bits(0.5f32.to_bits() + 1);
    let candidate = [nudged_value];
    let diff = diff_plane(&baseline, &candidate);
    let verdict = Fidelity::Exact.judge(&[diff]);

    assert!(!verdict.passed());
}

#[test]
fn rounding_accepts_a_drift_inside_both_bounds() {
    let baseline = [0.5f32, 0.25];
    let candidate = [0.500_01f32, 0.25];
    let diff = diff_plane(&baseline, &candidate);
    let fidelity = Fidelity::Rounding {
        min_psnr: 60.0,
        max_abs: 1.0e-4,
    };
    let verdict = fidelity.judge(&[diff]);

    assert_eq!(verdict, CompareResult::Pass);
}

#[test]
fn rounding_rejects_a_drift_past_max_abs() {
    let baseline = [0.5f32];
    let candidate = [0.6f32];
    let diff = diff_plane(&baseline, &candidate);
    let fidelity = Fidelity::Rounding {
        min_psnr: 0.0,
        max_abs: 1.0e-4,
    };
    let verdict = fidelity.judge(&[diff]);

    assert!(!verdict.passed());
}

#[test]
fn approximate_is_held_to_its_declared_bound() {
    let baseline = [0.5f32];
    let candidate = [0.51f32];
    let diff = diff_plane(&baseline, &candidate);
    let loose = Fidelity::Approximate {
        bound: ErrorBound {
            min_psnr: 20.0,
            max_abs: 0.02,
        },
        rationale: "test",
    };
    let tight = Fidelity::Approximate {
        bound: ErrorBound {
            min_psnr: 20.0,
            max_abs: 0.001,
        },
        rationale: "test",
    };

    assert!(loose.judge(&[diff]).passed());
    assert!(!tight.judge(&[diff]).passed());
}

#[test]
fn classes_order_from_most_to_least_faithful() {
    assert!(FidelityClass::Exact < FidelityClass::Rounding);
    assert!(FidelityClass::Rounding < FidelityClass::Approximate);
}

#[test]
fn exact_integer_names_round_trip_and_sit_between_exact_and_rounding() {
    let class = FidelityClass::ExactInteger;

    assert_eq!(class.name(), "exact_integer");
    assert_eq!(FidelityClass::from_name("exact_integer"), Some(class));
    assert!(FidelityClass::Exact < class && class < FidelityClass::Rounding);
    assert!(class.deterministic() && FidelityClass::Exact.deterministic());
    assert!(!FidelityClass::Rounding.deterministic());
    assert!(!FidelityClass::Approximate.deterministic());
}

#[test]
fn an_integer_window_bound_is_judged_like_rounding() {
    let bound = ErrorBound {
        min_psnr: 80.0,
        max_abs: 1.0e-3,
    };
    let fidelity = Fidelity::ExactInteger {
        reference: IntegerBound::Bound(bound),
    };
    let baseline = [0.5f32, 0.25];
    let within_values = [0.500_000_1f32, 0.25];
    let past_values = [0.51f32, 0.25];
    let within = diff_plane(&baseline, &within_values);
    let past = diff_plane(&baseline, &past_values);

    assert_eq!(fidelity.class(), FidelityClass::ExactInteger);
    assert!(fidelity.judge(&[within]).passed());
    assert!(!fidelity.judge(&[past]).passed());
}

#[test]
fn the_eighth_best_bound_is_never_judged_from_plane_diffs() {
    let fidelity = Fidelity::ExactInteger {
        reference: IntegerBound::EighthBest,
    };
    let identical = PlaneDiff {
        identical: true,
        max_abs: 0.0,
        psnr: f64::INFINITY,
    };

    assert!(!fidelity.judge(&[identical]).passed());
}

#[test]
fn class_names_round_trip() {
    let classes = [
        FidelityClass::Exact,
        FidelityClass::ExactInteger,
        FidelityClass::Rounding,
        FidelityClass::Approximate,
    ];

    for class in classes {
        let name = class.name();
        let parsed = FidelityClass::from_name(name);
        assert_eq!(parsed, Some(class), "{name}");
    }
}

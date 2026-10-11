/// How closely a candidate's output must match the baseline's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fidelity {
    /// Bit identical output.
    Exact,
    /// The same algorithm with its float operations reordered.
    Rounding { min_psnr: f64, max_abs: f32 },
    /// An algorithmic shortcut, held to the error bound it declares.
    Approximate {
        bound: ErrorBound,
        rationale: &'static str,
    },
}

/// The largest error an approximate candidate may show against the baseline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorBound {
    pub min_psnr: f64,
    pub max_abs: f32,
}

/// A fidelity class without its bounds, ordered from most to least faithful.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FidelityClass {
    Exact,
    Rounding,
    Approximate,
}

impl FidelityClass {
    /// The name the store writes.
    pub fn name(self) -> &'static str {
        match self {
            FidelityClass::Exact => "exact",
            FidelityClass::Rounding => "rounding",
            FidelityClass::Approximate => "approximate",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "exact" => Some(FidelityClass::Exact),
            "rounding" => Some(FidelityClass::Rounding),
            "approximate" => Some(FidelityClass::Approximate),
            _ => None,
        }
    }
}

/// The outcome of checking a candidate's output against the baseline's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompareResult {
    Pass,
    /// The check failed, with the reason the tuner logs.
    Fail(String),
}

impl CompareResult {
    pub fn passed(&self) -> bool {
        matches!(self, CompareResult::Pass)
    }
}

/// How far one output plane drifted from the baseline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaneDiff {
    /// Whether every element has the same bits.
    pub identical: bool,
    pub max_abs: f32,
    /// PSNR against the baseline's largest magnitude, infinite for identical planes.
    pub psnr: f64,
}

impl Fidelity {
    pub fn class(&self) -> FidelityClass {
        match self {
            Fidelity::Exact => FidelityClass::Exact,
            Fidelity::Rounding { .. } => FidelityClass::Rounding,
            Fidelity::Approximate { .. } => FidelityClass::Approximate,
        }
    }

    /// Checks every plane's drift against this class.
    pub fn judge(&self, planes: &[PlaneDiff]) -> CompareResult {
        match *self {
            Fidelity::Exact => judge_exact(planes),
            Fidelity::Rounding { min_psnr, max_abs } => judge_bound(planes, min_psnr, max_abs),
            Fidelity::Approximate { bound, .. } => judge_bound(planes, bound.min_psnr, bound.max_abs),
        }
    }
}

fn judge_exact(planes: &[PlaneDiff]) -> CompareResult {
    let changed = planes.iter().position(|plane| !plane.identical);

    match changed {
        None => CompareResult::Pass,
        Some(index) => {
            let reason = format!("plane {index} is not bit identical");
            CompareResult::Fail(reason)
        },
    }
}

fn judge_bound(planes: &[PlaneDiff], min_psnr: f64, max_abs: f32) -> CompareResult {
    for (index, plane) in planes.iter().enumerate() {
        if plane.max_abs > max_abs {
            let reason = format!(
                "plane {index} differs by {} where at most {max_abs} is allowed",
                plane.max_abs
            );
            return CompareResult::Fail(reason);
        }

        if plane.psnr < min_psnr {
            let reason = format!(
                "plane {index} has a PSNR of {:.2} dB where at least {min_psnr} dB is required",
                plane.psnr
            );
            return CompareResult::Fail(reason);
        }
    }

    CompareResult::Pass
}

/// Measures `candidate` against `baseline`, element by element.
///
/// Elements with the same bits count as equal, NaNs included. Any other NaN difference counts as
/// infinite. PSNR uses the baseline's largest magnitude as its peak, or 1 when every value is zero.
pub fn diff_plane(baseline: &[f32], candidate: &[f32]) -> PlaneDiff {
    assert_eq!(
        baseline.len(),
        candidate.len(),
        "compared planes must hold the same number of values"
    );

    let mut identical = true;
    let mut max_abs = 0.0f32;
    let mut squared_sum = 0.0f64;
    let mut peak = 0.0f32;

    for (&expected, &actual) in baseline.iter().zip(candidate) {
        let same_bits = expected.to_bits() == actual.to_bits();
        identical &= same_bits;

        let magnitude = expected.abs();
        peak = peak.max(magnitude);

        let raw_difference = (expected - actual).abs();
        let difference = match (same_bits, raw_difference.is_nan()) {
            (true, _) => 0.0,
            (false, true) => f32::INFINITY,
            (false, false) => raw_difference,
        };

        max_abs = max_abs.max(difference);
        let wide_difference = f64::from(difference);
        squared_sum += wide_difference * wide_difference;
    }

    let psnr = psnr_from(squared_sum, baseline.len(), peak);

    PlaneDiff {
        identical,
        max_abs,
        psnr,
    }
}

/// Measures an integer `candidate` against `baseline`, element by element.
pub fn diff_integers(baseline: &[i32], candidate: &[i32]) -> PlaneDiff {
    assert_eq!(
        baseline.len(),
        candidate.len(),
        "compared planes must hold the same number of values"
    );

    let mut identical = true;
    let mut max_abs = 0.0f32;
    let mut squared_sum = 0.0f64;
    let mut peak = 0.0f32;

    for (&expected, &actual) in baseline.iter().zip(candidate) {
        identical &= expected == actual;

        let wide_expected = i64::from(expected);
        let wide_actual = i64::from(actual);
        let difference = (wide_expected - wide_actual).abs() as f32;
        let magnitude = wide_expected.abs() as f32;

        peak = peak.max(magnitude);
        max_abs = max_abs.max(difference);
        let wide_difference = f64::from(difference);
        squared_sum += wide_difference * wide_difference;
    }

    let psnr = psnr_from(squared_sum, baseline.len(), peak);

    PlaneDiff {
        identical,
        max_abs,
        psnr,
    }
}

fn psnr_from(squared_sum: f64, count: usize, peak: f32) -> f64 {
    if squared_sum == 0.0 {
        return f64::INFINITY;
    }

    let peak = if peak > 0.0 { f64::from(peak) } else { 1.0 };
    let mean_squared = squared_sum / count.max(1) as f64;
    let ratio = peak * peak / mean_squared;

    10.0 * ratio.log10()
}

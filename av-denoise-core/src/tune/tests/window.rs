use cubecl::prelude::*;

use super::engine::{nlmeans_outputs, noisy_frames, temporal_fast};
use super::equivalence::assert_candidates_match_baseline;
use super::fixtures::{WindowSpec, upload, window_site};
use super::probe::small_device_limits;
use super::runtime::{TestRuntime, test_client};
use crate::nlmeans::{
    ChannelMode,
    DEFAULT_PILOT_STRENGTH_SCALE,
    NlmTuning,
    NlmeansAlgorithm,
    NlmeansHqOptions,
    NlmeansOptions,
    PrefilterMode,
};
use crate::tune::gpu::read_f32;
use crate::tune::sites::{
    AccumOutputs,
    NLM_WINDOW,
    SEPARABLE_MAX_ABS,
    SEPARABLE_MIN_PSNR,
    WindowCandidate,
    WindowSite,
    compare_accum,
    estimate_planes,
};
use crate::tune::{
    CandidateRecord,
    CompileProgress,
    ErrorBound,
    Fidelity,
    FidelityClass,
    IntegerBound,
    IntegerCodes,
    KernelPlan,
    PlaneDiff,
    SiteCandidate,
    SiteKey,
    SiteRanking,
    TuneResult,
    VariantSite,
    diff_plane,
    fit_candidates,
};

#[test]
fn window_candidate_names_round_trip() {
    for &candidate in WindowCandidate::ALL {
        let name = candidate.name();
        let parsed = WindowCandidate::from_name(name);
        assert_eq!(parsed, Some(candidate), "{name}");
    }

    let scalar_int = WindowCandidate::from_name("scalar_int");
    assert_eq!(scalar_int, Some(WindowCandidate::ScalarInt));
    assert_eq!(WindowCandidate::baseline(), WindowCandidate::Block32x8);
}

/// The pair kernel holds the expanded centre tile, a forward and a backward distance tile, and a
/// placeholder row-sum buffer for each.
#[test]
fn window_shared_bytes_count_the_centre_tile_and_every_distance_tile() {
    let client = test_client();
    let mut pair = WindowSpec::pair(64, 48, ChannelMode::Yuv);
    pair.patch_radius = 3;
    pair.search_radius = 7;
    let mut single = pair;
    single.pair = false;

    let pair_site = window_site(&client, &pair);
    let single_site = window_site(&client, &single);

    // Centre tile: (32 + 6 + 14) x (8 + 6 + 14) pixels of four f32 lanes = 23296 bytes.
    // One distance tile: (32 + 6) x (8 + 6) f32 = 2128 bytes.
    assert_eq!(
        pair_site.shared_bytes(WindowCandidate::Block32x8),
        23296 + 2 * 2128 + 2 * 4
    );
    assert_eq!(
        single_site.shared_bytes(WindowCandidate::Block32x8),
        23296 + 2128 + 4
    );
}

#[test]
fn window_keys_name_the_kernel_and_every_compiled_value() {
    let client = test_client();
    let mut spec = WindowSpec::pair(1920, 1080, ChannelMode::Chroma);
    spec.reference = true;
    spec.confidence = true;
    let site = window_site(&client, &spec);

    let key = site.key();

    assert_eq!(
        key.as_str(),
        "nlm_window|w2048|h2048|kernel=pair|ref=1|conf=1|step=8|ch=2|stored=2|pr=2|sr=3"
    );

    let mut integer_spec = WindowSpec::single(1920, 1080, ChannelMode::Luma);
    integer_spec.codes = Some(IntegerCodes { depth: 10 });
    let integer_site = window_site(&client, &integer_spec);

    let integer_key = integer_site.key();

    assert_eq!(
        integer_key.as_str(),
        "nlm_window|w2048|h2048|kernel=single|ref=0|conf=0|step=0|ch=1|stored=1|pr=2|sr=3|int=10"
    );
}

fn window_specs() -> Vec<WindowSpec> {
    let mut specs = Vec::new();

    for channels in [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv] {
        for (width, height) in [(70u32, 45u32), (33, 17), (96, 64)] {
            for pair in [false, true] {
                for reference in [false, true] {
                    let mut spec = WindowSpec::single(width, height, channels);
                    spec.pair = pair;
                    spec.reference = reference;
                    spec.confidence = pair && width == 70;
                    specs.push(spec);
                }
            }
        }
    }

    for channels in [ChannelMode::Luma, ChannelMode::Yuv] {
        for pair in [false, true] {
            let mut spec = WindowSpec::single(70, 45, channels);
            spec.patch_radius = 4;
            spec.search_radius = 2;
            spec.pair = pair;
            spec.confidence = pair;
            specs.push(spec);
        }
    }

    specs
}

#[test]
fn every_window_candidate_matches_the_baseline_within_its_fidelity() {
    let client = test_client();

    for spec in window_specs() {
        let site = window_site(&client, &spec);
        let label = format!("{spec:?}");
        let baseline = assert_candidates_match_baseline(&site, &label);

        let weights = read_f32(&client, &baseline.weight_sum, baseline.pixels).expect("weights read back");
        let has_signal = weights.iter().any(|&weight| weight > 0.0);
        assert!(has_signal, "{label}: the baseline accumulated nothing");
    }
}

#[test]
fn a_tall_cube_does_not_fit_a_32kb_device_at_wide_radii() {
    let client = test_client();
    let mut spec = WindowSpec::pair(64, 48, ChannelMode::Yuv);
    spec.patch_radius = 3;
    spec.search_radius = 7;
    let site = window_site(&client, &spec);
    let limits = small_device_limits();

    let fitted = fit_candidates(&site, &limits).expect("the baseline fits");

    // 32x16 needs 29952 + 2 * 3344 + 2 * 4 = 36648 bytes, and 42272 with separable sums.
    // 16x16 needs 20736 + 2 * 1936 + 2 * 4 = 24616, and 27424 with separable sums.
    // 32x8 needs 27560, and 31136 with separable sums.
    assert_eq!(site.shared_bytes(WindowCandidate::Block32x16), 36648);
    assert_eq!(site.shared_bytes(WindowCandidate::Block16x16), 24616);
    assert_eq!(
        fitted,
        vec![
            WindowCandidate::Block32x8,
            WindowCandidate::Block16x16,
            WindowCandidate::Separable32x8,
            WindowCandidate::Separable16x16,
        ]
    );
}

#[test]
fn a_baseline_past_the_cap_is_a_fit_error() {
    let client = test_client();
    let mut spec = WindowSpec::pair(64, 48, ChannelMode::Yuv);
    spec.patch_radius = 4;
    spec.search_radius = 8;
    let site = window_site(&client, &spec);
    let limits = small_device_limits();

    let result = fit_candidates(&site, &limits);

    assert!(result.is_err());
}

fn assert_forced_shapes_match(algorithm: NlmeansAlgorithm, label: &str) {
    for channels in [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv] {
        let planes = channels.count() as usize;
        let (width, height) = (70u32, 45u32);
        let frames = noisy_frames(width, height, planes, 6);
        let baseline_plan = KernelPlan::baseline();
        let baseline = nlmeans_outputs(algorithm, channels, width, height, &frames, baseline_plan);

        for &candidate in WindowCandidate::ALL {
            if candidate.fidelity() != Fidelity::Exact {
                continue;
            }

            let plan = KernelPlan::baseline().with_site(NLM_WINDOW, candidate.name());
            let forced = nlmeans_outputs(algorithm, channels, width, height, &frames, plan);
            assert_eq!(forced, baseline, "{label} {channels:?} {}", candidate.name());
        }
    }
}

#[test]
fn forcing_any_window_shape_leaves_temporal_nlmeans_output_unchanged() {
    let algorithm = temporal_fast();
    assert_forced_shapes_match(algorithm, "temporal fast");
}

/// The pilot runs the single-window kernel on the input, and the main pass the `_ref` kernels.
#[test]
fn forcing_any_window_shape_leaves_the_nlm_pilot_path_unchanged() {
    let NlmeansAlgorithm::Fast(fast) = temporal_fast() else {
        unreachable!("temporal_fast builds the fast variant");
    };
    let nlm = NlmeansOptions {
        prefilter: PrefilterMode::NlmSpatial {
            strength_scale: DEFAULT_PILOT_STRENGTH_SCALE,
        },
        ..fast
    };
    let hq = NlmeansHqOptions {
        nlm,
        ..NlmeansHqOptions::default()
    };
    let algorithm = NlmeansAlgorithm::Hq(hq);

    assert_forced_shapes_match(algorithm, "hq with the nlm pilot");
}

#[test]
fn separable_candidates_carry_the_rounding_bound() {
    let bound = Fidelity::Rounding {
        min_psnr: SEPARABLE_MIN_PSNR,
        max_abs: SEPARABLE_MAX_ABS,
    };
    let integer_bound = ErrorBound {
        min_psnr: SEPARABLE_MIN_PSNR,
        max_abs: SEPARABLE_MAX_ABS,
    };
    let integer = Fidelity::ExactInteger {
        reference: IntegerBound::Bound(integer_bound),
    };

    for &candidate in WindowCandidate::ALL {
        let expected = match (candidate.separable(), candidate.integer()) {
            (true, _) => bound,
            (false, true) => integer,
            (false, false) => Fidelity::Exact,
        };
        assert_eq!(candidate.fidelity(), expected, "{}", candidate.name());
    }

    assert_eq!(SEPARABLE_MIN_PSNR, 80.0);
    assert_eq!(SEPARABLE_MAX_ABS, 1.0e-3);
    assert_eq!(WindowCandidate::Separable32x8.name(), "32x8_separable");
    const { assert!(<WindowSite<TestRuntime> as VariantSite<TestRuntime>>::PRESCREEN) };
}

/// Reads `outputs` back, lets `change` edit all three buffers, and uploads the result.
fn edited_accum(
    client: &ComputeClient<TestRuntime>,
    outputs: &AccumOutputs,
    change: impl Fn(&mut [f32], &mut [f32], &mut [f32]),
) -> AccumOutputs {
    let mut accum = read_f32(client, &outputs.accum, outputs.accum_len).expect("accum reads back");
    let mut weight_sum = read_f32(client, &outputs.weight_sum, outputs.pixels).expect("weights read back");
    let mut max_weight = read_f32(client, &outputs.max_weight, outputs.pixels).expect("maxima read back");
    change(&mut accum, &mut weight_sum, &mut max_weight);

    let accum_handle = upload(client, &accum);
    let weight_sum_handle = upload(client, &weight_sum);
    let max_weight_handle = upload(client, &max_weight);

    AccumOutputs {
        accum: accum_handle,
        weight_sum: weight_sum_handle,
        max_weight: max_weight_handle,
        accum_len: outputs.accum_len,
        pixels: outputs.pixels,
    }
}

fn baseline_window_outputs(client: &ComputeClient<TestRuntime>) -> (WindowSite<TestRuntime>, AccumOutputs) {
    let spec = WindowSpec::pair(70, 45, ChannelMode::Yuv);
    let site = window_site(client, &spec);
    let outputs = site.scratch_outputs();
    site.launch(WindowCandidate::Block32x8, &outputs);

    (site, outputs)
}

#[test]
fn the_separable_check_compares_normalised_estimates_not_raw_sums() {
    let client = test_client();
    let (site, baseline) = baseline_window_outputs(&client);
    let bound = WindowCandidate::Separable32x8.fidelity();
    let scale_everything = |accum: &mut [f32], weight_sum: &mut [f32], max_weight: &mut [f32]| {
        let all_values = accum
            .iter_mut()
            .chain(weight_sum.iter_mut())
            .chain(max_weight.iter_mut());
        for value in all_values {
            *value *= 1.01;
        }
    };
    let scaled = edited_accum(&client, &baseline, scale_everything);

    let raw = compare_accum(&client, &baseline, &scaled, bound).expect("the outputs read back");
    let normalised = site
        .compare(&baseline, &scaled, bound)
        .expect("the outputs read back");

    assert!(!raw.passed(), "the raw sums should drift past the bound");
    assert!(normalised.passed(), "{normalised:?}");
}

#[test]
fn the_separable_check_rejects_an_estimate_past_the_bound() {
    let client = test_client();
    let (site, baseline) = baseline_window_outputs(&client);
    let bound = WindowCandidate::Separable32x8.fidelity();
    let shift_one_estimate = |accum: &mut [f32], weight_sum: &mut [f32], _max_weight: &mut [f32]| {
        accum[0] += 0.01 * weight_sum[0];
    };
    let shifted = edited_accum(&client, &baseline, shift_one_estimate);

    let verdict = site
        .compare(&baseline, &shifted, bound)
        .expect("the outputs read back");

    assert!(!verdict.passed());
}

#[test]
fn separable_sums_count_a_row_sum_buffer_per_distance_tile() {
    let client = test_client();
    let mut pair = WindowSpec::pair(64, 48, ChannelMode::Yuv);
    pair.patch_radius = 3;
    pair.search_radius = 7;
    let mut single = pair;
    single.pair = false;

    let pair_site = window_site(&client, &pair);
    let single_site = window_site(&client, &single);

    // A row-sum buffer at 32x8 and patch radius 3 is (8 + 6) x 32 f32 = 1792 bytes.
    assert_eq!(
        pair_site.shared_bytes(WindowCandidate::Separable32x8),
        23296 + 2 * 2128 + 2 * 1792
    );
    assert_eq!(
        single_site.shared_bytes(WindowCandidate::Separable32x8),
        23296 + 2128 + 1792
    );
}

/// Splits every emitted frame into its planes and checks each against the separable bound.
fn assert_within_separable_bound(forced: &[Vec<f32>], baseline: &[Vec<f32>], pixels: usize, label: &str) {
    let bound = Fidelity::Rounding {
        min_psnr: SEPARABLE_MIN_PSNR,
        max_abs: SEPARABLE_MAX_ABS,
    };

    assert_eq!(
        forced.len(),
        baseline.len(),
        "{label}: the runs emitted different frame counts"
    );

    for (frame, (forced_frame, baseline_frame)) in forced.iter().zip(baseline).enumerate() {
        let forced_planes = forced_frame.chunks(pixels);
        let baseline_planes = baseline_frame.chunks(pixels);
        let diffs: Vec<PlaneDiff> = baseline_planes
            .zip(forced_planes)
            .map(|(expected, actual)| diff_plane(expected, actual))
            .collect();
        let verdict = bound.judge(&diffs);

        assert!(verdict.passed(), "{label} frame {frame}: {verdict:?}");
    }
}

fn assert_forced_separable_within_bound(algorithm: NlmeansAlgorithm, label: &str) {
    for channels in [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv] {
        let planes = channels.count() as usize;
        let (width, height) = (70u32, 45u32);
        let pixels = (width * height) as usize;
        let frames = noisy_frames(width, height, planes, 6);
        let baseline_plan = KernelPlan::baseline();
        let baseline = nlmeans_outputs(algorithm, channels, width, height, &frames, baseline_plan);

        for &candidate in WindowCandidate::ALL {
            if !candidate.separable() {
                continue;
            }

            let plan = KernelPlan::baseline().with_site(NLM_WINDOW, candidate.name());
            let forced = nlmeans_outputs(algorithm, channels, width, height, &frames, plan);
            let case = format!("{label} {channels:?} {}", candidate.name());
            assert_within_separable_bound(&forced, &baseline, pixels, &case);
        }
    }
}

#[test]
fn forcing_a_separable_window_stays_within_the_bound_on_temporal_nlmeans() {
    let algorithm = temporal_fast();
    assert_forced_separable_within_bound(algorithm, "temporal fast");
}

#[test]
fn forcing_a_separable_window_stays_within_the_bound_at_patch_radius_4() {
    let NlmeansAlgorithm::Fast(fast) = temporal_fast() else {
        unreachable!("temporal_fast builds the fast variant");
    };
    let tuning = NlmTuning {
        search_radius: Some(2),
        patch_radius: Some(4),
        ..fast.tuning
    };
    let options = NlmeansOptions { tuning, ..fast };
    let algorithm = NlmeansAlgorithm::Fast(options);

    assert_forced_separable_within_bound(algorithm, "patch radius 4");
}

/// The pilot runs the single-window kernel on the input, and the main pass the `_ref` kernels.
#[test]
fn forcing_a_separable_window_stays_within_the_bound_on_the_nlm_pilot_path() {
    let NlmeansAlgorithm::Fast(fast) = temporal_fast() else {
        unreachable!("temporal_fast builds the fast variant");
    };
    let nlm = NlmeansOptions {
        prefilter: PrefilterMode::NlmSpatial {
            strength_scale: DEFAULT_PILOT_STRENGTH_SCALE,
        },
        ..fast
    };
    let hq = NlmeansHqOptions {
        nlm,
        ..NlmeansHqOptions::default()
    };
    let algorithm = NlmeansAlgorithm::Hq(hq);

    assert_forced_separable_within_bound(algorithm, "hq with the nlm pilot");
}

#[test]
fn a_deterministic_plan_never_picks_a_separable_window() {
    let key = SiteKey::builder(NLM_WINDOW, 1920, 1080)
        .param("kernel", "pair")
        .build();
    let separable = CandidateRecord {
        name: "32x8_separable".to_string(),
        fidelity: FidelityClass::Rounding,
        baseline: false,
        median_ns: 500,
        samples: 15,
    };
    let baseline = CandidateRecord {
        name: "32x8".to_string(),
        fidelity: FidelityClass::Exact,
        baseline: true,
        median_ns: 1000,
        samples: 15,
    };
    let ranking = SiteRanking {
        candidates: vec![separable, baseline],
        partial: false,
        progress: CompileProgress::default(),
        identity: String::new(),
    };
    let mut result = TuneResult::default();
    result.sites.insert(key.clone(), ranking);

    let fastest = result.kernel_plan(false);
    let exact = result.kernel_plan(true);

    assert_eq!(fastest.get(&key), Some("32x8_separable"));
    assert_eq!(exact.get(&key), Some("32x8"));
}

#[test]
fn a_pixel_without_weight_reads_as_zero_in_every_estimate_plane() {
    let accum = [2.0, 4.0, 3.0, 5.0];
    let weight_sum = [2.0, 0.0];
    let max_weight = [1.0, 0.5];

    let planes = estimate_planes(&accum, &weight_sum, &max_weight, 2);

    assert_eq!(planes, vec![vec![1.0, 0.0], vec![2.0, 0.0], vec![0.5, 0.0]]);
}

#[test]
fn a_nan_weight_sum_fails_the_separable_bound() {
    let accum = [2.0, 0.0];
    let max_weight = [1.0, 0.0];
    let baseline_weights = [2.0, 0.0];
    let candidate_weights = [2.0, f32::NAN];
    let bound = WindowCandidate::Separable32x8.fidelity();

    let expected = estimate_planes(&accum, &baseline_weights, &max_weight, 1);
    let actual = estimate_planes(&accum, &candidate_weights, &max_weight, 1);
    let diffs: Vec<PlaneDiff> = expected
        .iter()
        .zip(&actual)
        .map(|(expected_plane, actual_plane)| diff_plane(expected_plane, actual_plane))
        .collect();
    let verdict = bound.judge(&diffs);

    assert!(!verdict.passed(), "{verdict:?}");
}

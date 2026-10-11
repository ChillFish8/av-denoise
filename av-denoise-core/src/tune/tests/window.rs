use super::engine::{nlmeans_outputs, noisy_frames, temporal_fast};
use super::equivalence::assert_candidates_match_baseline;
use super::fixtures::{WindowSpec, window_site};
use super::probe::small_device_limits;
use super::runtime::test_client;
use crate::nlmeans::{
    ChannelMode,
    DEFAULT_PILOT_STRENGTH_SCALE,
    NlmeansAlgorithm,
    NlmeansHqOptions,
    NlmeansOptions,
    PrefilterMode,
};
use crate::tune::gpu::read_f32;
use crate::tune::sites::{NLM_WINDOW, WindowCandidate};
use crate::tune::{KernelPlan, SiteCandidate, VariantSite, fit_candidates};

#[test]
fn window_candidate_names_round_trip() {
    for &candidate in WindowCandidate::ALL {
        let name = candidate.name();
        let parsed = WindowCandidate::from_name(name);
        assert_eq!(parsed, Some(candidate), "{name}");
    }

    assert_eq!(WindowCandidate::baseline(), WindowCandidate::Block32x8);
}

/// The pair kernel holds the expanded centre tile plus a forward and a backward distance tile.
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
        23296 + 2 * 2128
    );
    assert_eq!(single_site.shared_bytes(WindowCandidate::Block32x8), 23296 + 2128);
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

    specs
}

#[test]
fn every_window_shape_matches_the_baseline_bit_for_bit() {
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

    // 32x16 needs 29952 + 2 * 3344 = 36640 bytes. 16x16 needs 20736 + 2 * 1936 = 24608.
    assert_eq!(site.shared_bytes(WindowCandidate::Block32x16), 36640);
    assert_eq!(site.shared_bytes(WindowCandidate::Block16x16), 24608);
    assert_eq!(
        fitted,
        vec![WindowCandidate::Block32x8, WindowCandidate::Block16x16]
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

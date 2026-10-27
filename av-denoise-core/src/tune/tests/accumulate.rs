use super::engine::{nlmeans_outputs, noisy_frames};
use super::equivalence::assert_candidates_match_baseline;
use super::fixtures::{AccumulateSpec, accumulate_site};
use super::runtime::test_client;
use crate::nlmeans::{ChannelMode, DenoisingMode, NlmTuning, NlmeansAlgorithm, NlmeansOptions};
use crate::tune::gpu::read_f32;
use crate::tune::sites::{AccumulateCandidate, NLM_ACCUMULATE};
use crate::tune::{KernelPlan, SiteCandidate, VariantSite};

#[test]
fn accumulate_candidate_names_round_trip() {
    for &candidate in AccumulateCandidate::ALL {
        let name = candidate.name();
        let parsed = AccumulateCandidate::from_name(name);
        assert_eq!(parsed, Some(candidate), "{name}");
    }

    assert_eq!(AccumulateCandidate::baseline(), AccumulateCandidate::Thin32x16);
}

#[test]
fn accumulate_uses_no_shared_memory_and_keys_on_its_lanes() {
    let client = test_client();
    let spec = AccumulateSpec {
        width: 640,
        height: 360,
        channels: ChannelMode::Yuv,
        q_x: 1,
        q_y: -1,
    };
    let site = accumulate_site(&client, &spec);

    let key = site.key();

    assert_eq!(site.shared_bytes(AccumulateCandidate::Thin32x16), 0);
    assert_eq!(key.as_str(), "nlm_accumulate|w1024|h512|stored=4");
}

#[test]
fn every_accumulate_shape_matches_the_baseline_bit_for_bit() {
    let client = test_client();

    for channels in [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv] {
        for (width, height) in [(70u32, 45u32), (33, 17)] {
            for (q_x, q_y) in [(1, 0), (-3, 2), (4, -4)] {
                let spec = AccumulateSpec {
                    width,
                    height,
                    channels,
                    q_x,
                    q_y,
                };
                let site = accumulate_site(&client, &spec);
                let label = format!("{spec:?}");
                let baseline = assert_candidates_match_baseline(&site, &label);

                let weights =
                    read_f32(&client, &baseline.weight_sum, baseline.pixels).expect("weights read back");
                let has_signal = weights.iter().any(|&weight| weight > 0.0);
                assert!(has_signal, "{label}: the baseline accumulated nothing");
            }
        }
    }
}

/// A patch radius past the separable threshold sends NLMeans down the path that accumulates.
#[test]
fn forcing_any_accumulate_shape_leaves_separable_nlmeans_output_unchanged() {
    let options = NlmeansOptions {
        mode: DenoisingMode::Spacial,
        tuning: NlmTuning {
            search_radius: Some(2),
            patch_radius: Some(9),
            ..NlmTuning::default()
        },
        ..NlmeansOptions::default()
    };
    let algorithm = NlmeansAlgorithm::Fast(options);

    for channels in [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv] {
        let planes = channels.count() as usize;
        let (width, height) = (70u32, 45u32);
        let frames = noisy_frames(width, height, planes, 2);
        let baseline_plan = KernelPlan::baseline();
        let baseline = nlmeans_outputs(algorithm, channels, width, height, &frames, baseline_plan);

        for &candidate in AccumulateCandidate::ALL {
            let plan = KernelPlan::baseline().with_site(NLM_ACCUMULATE, candidate.name());
            let forced = nlmeans_outputs(algorithm, channels, width, height, &frames, plan);
            assert_eq!(forced, baseline, "{channels:?} {}", candidate.name());
        }
    }
}

#[test]
fn the_narrow_accumulate_shape_is_a_candidate() {
    let parsed = AccumulateCandidate::from_name("32x8");

    assert_eq!(parsed, Some(AccumulateCandidate::Thin32x8));
}

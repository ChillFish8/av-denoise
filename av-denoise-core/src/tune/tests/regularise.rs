use super::engine::{nl4d_outputs, noisy_frames};
use super::equivalence::assert_candidates_match_baseline;
use super::fixtures::{RegulariseContent, RegulariseSpec, regularise_site};
use super::runtime::test_client;
use crate::Nl4dOptions;
use crate::nlmeans::ChannelMode;
use crate::tune::gpu::read_i32;
use crate::tune::sites::{NL4D_MV_REGULARISE, RegulariseCandidate};
use crate::tune::{DeviceLimits, FidelityClass, KernelPlan, SiteCandidate, VariantSite, fit_candidates};

fn textured_spec() -> RegulariseSpec {
    RegulariseSpec {
        width: 72,
        height: 40,
        content: RegulariseContent::Textured,
        field_lambda: 1.0,
    }
}

#[test]
fn regularise_candidate_names_round_trip() {
    for &candidate in RegulariseCandidate::ALL {
        let name = candidate.name();
        let parsed = RegulariseCandidate::from_name(name);
        assert_eq!(parsed, Some(candidate), "{name}");
    }

    assert_eq!(RegulariseCandidate::baseline(), RegulariseCandidate::Serial8x8);
}

/// The centre tile plus the candidate list, the median, and the SAD and cost of every candidate.
#[test]
fn serial_shared_bytes_count_the_centre_tile_and_every_scratch_array() {
    let client = test_client();
    let site = regularise_site(&client, &textured_spec());

    // 16 x 16 centre floats, then 14 + 2 + 7 + 7 words of candidates, median, SADs and costs.
    assert_eq!(
        site.shared_bytes(RegulariseCandidate::Serial8x8),
        16 * 16 * 4 + 30 * 4
    );
}

#[test]
fn regularise_keys_name_the_block_geometry() {
    let client = test_client();
    let site = regularise_site(&client, &textured_spec());

    let key = site.key();

    assert_eq!(key.as_str(), "nl4d_mv_regularise|w128|h64|blksize=16|step=8");
}

fn regularise_specs() -> Vec<RegulariseSpec> {
    let mut specs = Vec::new();

    for content in [
        RegulariseContent::Textured,
        RegulariseContent::Flat,
        RegulariseContent::Periodic,
    ] {
        for (width, height) in [(72u32, 40u32), (61, 37), (128, 64)] {
            for field_lambda in [0.0f32, 1.0, 4.0] {
                specs.push(RegulariseSpec {
                    width,
                    height,
                    content,
                    field_lambda,
                });
            }
        }
    }

    specs
}

#[test]
fn every_serial_shape_matches_the_baseline_bit_for_bit() {
    let client = test_client();
    let limits = DeviceLimits::from_client(&client);

    for spec in regularise_specs() {
        let site = regularise_site(&client, &spec);
        let label = format!("{spec:?}");

        let fitted = fit_candidates(&site, &limits).expect("the baseline fits");
        assert!(fitted.contains(&RegulariseCandidate::Serial8x4), "{label}");
        assert!(fitted.contains(&RegulariseCandidate::Serial4x4), "{label}");

        let baseline = assert_candidates_match_baseline(&site, &label);
        let vectors = read_i32(&client, &baseline.vectors, 2 * baseline.blocks).expect("vectors read back");
        if spec.content == RegulariseContent::Textured {
            let has_signal = vectors.iter().any(|&component| component != 0);
            assert!(has_signal, "{label}: the baseline wrote an all-zero field");
        }
    }
}

#[test]
fn forcing_any_serial_shape_leaves_nl4d_output_unchanged() {
    let options = Nl4dOptions {
        temporal_radius: 1,
        ..Nl4dOptions::default()
    };
    let channels = ChannelMode::Luma;
    let (width, height) = (72u32, 48u32);
    let frames = noisy_frames(width, height, 1, 7);
    let baseline_plan = KernelPlan::baseline();
    let baseline = nl4d_outputs(options, channels, width, height, &frames, baseline_plan);

    for &candidate in RegulariseCandidate::ALL {
        if candidate.fidelity().class() != FidelityClass::Exact {
            continue;
        }

        let plan = KernelPlan::baseline().with_site(NL4D_MV_REGULARISE, candidate.name());
        let forced = nl4d_outputs(options, channels, width, height, &frames, plan);
        assert_eq!(forced, baseline, "{}", candidate.name());
    }
}

#[test]
fn the_narrow_serial_shapes_are_candidates() {
    let eight_by_four = RegulariseCandidate::from_name("serial8x4");
    let four_by_four = RegulariseCandidate::from_name("serial4x4");

    assert_eq!(eight_by_four, Some(RegulariseCandidate::Serial8x4));
    assert_eq!(four_by_four, Some(RegulariseCandidate::Serial4x4));
}

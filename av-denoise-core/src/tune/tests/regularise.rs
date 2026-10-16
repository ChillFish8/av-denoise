use cubecl::prelude::*;

use super::engine::{nl4d_outputs, noisy_frames};
use super::equivalence::assert_candidates_match_baseline;
use super::fixtures::{RegulariseContent, RegulariseSpec, regularise_neighbour_slot, regularise_site};
use super::runtime::{TestRuntime, test_client};
use crate::Nl4dOptions;
use crate::nl4d::kernels::tests::reference::nl4d_mv_regularise_reference;
use crate::nlmeans::ChannelMode;
use crate::tune::gpu::{bind, read_f32, read_i32};
use crate::tune::sites::{
    COOP_MAX_ABS,
    COOP_MAX_FLIPPED_SHARE,
    NL4D_MV_REGULARISE,
    RegulariseCandidate,
    RegulariseLayout,
    RegulariseOutputs,
    RegulariseSite,
    compare_fields,
};
use crate::tune::{
    CandidateRecord,
    DeviceLimits,
    Fidelity,
    FidelityClass,
    KernelPlan,
    SiteCandidate,
    VariantSite,
    choose,
    diff_plane,
    fit_candidates,
};

fn textured_spec() -> RegulariseSpec {
    RegulariseSpec {
        width: 72,
        height: 40,
        content: RegulariseContent::Textured,
        field_lambda: 1.0,
        neighbours: 4,
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
fn regularise_keys_name_the_block_geometry_and_neighbours() {
    let client = test_client();
    let site = regularise_site(&client, &textured_spec());

    let key = site.key();

    assert_eq!(
        key.as_str(),
        "nl4d_mv_regularise|w128|h64|blksize=16|step=8|neighbours=4"
    );
}

pub(super) fn regularise_specs() -> Vec<RegulariseSpec> {
    let mut specs = Vec::new();

    for content in [
        RegulariseContent::Textured,
        RegulariseContent::Flat,
        RegulariseContent::Periodic,
    ] {
        for (width, height) in [(72u32, 40u32), (61, 37), (128, 64)] {
            for field_lambda in [0.0f32, 1.0, 4.0] {
                for neighbours in [1u32, 4] {
                    specs.push(RegulariseSpec {
                        width,
                        height,
                        content,
                        field_lambda,
                        neighbours,
                    });
                }
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
        let vectors =
            read_i32(&client, &baseline.vectors, baseline.vectors_len()).expect("vectors read back");
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

/// Runs the verbatim per-neighbour kernel once per neighbour and returns the field and confidence.
fn reference_outputs(site: &RegulariseSite<TestRuntime>) -> (Vec<i32>, Vec<f32>) {
    let client = &site.client;
    let outputs = site.scratch_outputs();
    let blocks = outputs.blocks;
    let level_len = (site.width * site.height) as usize;
    let slot_bytes = site.slot_pixels as u64 * size_of::<f32>() as u64;
    let mv_bytes = outputs.mv_stride as u64 * size_of::<i32>() as u64;
    let conf_bytes = outputs.conf_stride as u64 * size_of::<f32>() as u64;
    let centre_start = site.centre_slot as u64 * slot_bytes;
    let centre_frame = site.pyramid.clone().offset_start(centre_start);

    for neighbour in 0..outputs.neighbours as u64 {
        let slot = regularise_neighbour_slot(outputs.neighbours, neighbour as u32);
        let neighbour_start = slot as u64 * slot_bytes;
        let neighbour_frame = site.pyramid.clone().offset_start(neighbour_start);
        let mv_slice = site.mv_in.clone().offset_start(neighbour * mv_bytes);
        let vectors_slice = outputs.vectors.clone().offset_start(neighbour * mv_bytes);
        let confidence_slice = outputs.confidence.clone().offset_start(neighbour * conf_bytes);
        let cube_count = CubeCount::new_2d(site.blocks_x, site.blocks_y);
        let cube_dim = CubeDim::new_2d(8, 8);
        let centre = bind::<TestRuntime>(&centre_frame, level_len);
        let neighbour_plane = bind::<TestRuntime>(&neighbour_frame, level_len);
        let mv_in = bind::<TestRuntime>(&mv_slice, 2 * blocks);
        let vectors = bind::<TestRuntime>(&vectors_slice, 2 * blocks);
        let confidence = bind::<TestRuntime>(&confidence_slice, blocks);

        unsafe {
            nl4d_mv_regularise_reference::launch_unchecked::<TestRuntime>(
                client,
                cube_count,
                cube_dim,
                centre,
                neighbour_plane,
                mv_in,
                vectors,
                confidence,
                site.lambda_pixel,
                site.sad_noise_floor,
                site.thsad,
                site.width,
                site.height,
                site.blksize,
                site.step,
                site.blocks_x,
                site.blocks_y,
            );
        }
    }

    let field = read_i32(client, &outputs.vectors, outputs.vectors_len()).expect("vectors read back");
    let scores =
        read_f32(client, &outputs.confidence, outputs.confidence_len()).expect("confidence reads back");
    (field, scores)
}

#[test]
fn the_merged_serial_kernel_matches_one_reference_launch_per_neighbour() {
    let client = test_client();
    let mut textured_confidence_written = false;

    for spec in regularise_specs() {
        let site = regularise_site(&client, &spec);
        let outputs = site.scratch_outputs();
        site.launch(RegulariseCandidate::Serial8x8, &outputs);

        let field = read_i32(&client, &outputs.vectors, outputs.vectors_len()).expect("vectors read back");
        let scores =
            read_f32(&client, &outputs.confidence, outputs.confidence_len()).expect("confidence reads back");
        let (reference_field, reference_scores) = reference_outputs(&site);
        let score_bits: Vec<u32> = scores.iter().map(|value| value.to_bits()).collect();
        let reference_bits: Vec<u32> = reference_scores.iter().map(|value| value.to_bits()).collect();

        assert_eq!(field, reference_field, "{spec:?}: vectors");
        assert_eq!(score_bits, reference_bits, "{spec:?}: confidence");

        if spec.content == RegulariseContent::Textured {
            let wrote_vectors = reference_field.iter().any(|&component| component != 0);
            assert!(wrote_vectors, "{spec:?}: the reference wrote an all-zero field");

            let wrote_scores = reference_scores.iter().any(|&score| score != 0.0);
            textured_confidence_written |= wrote_scores;
        }
    }

    assert!(
        textured_confidence_written,
        "the reference wrote no confidence on textured content"
    );
}

#[test]
fn cooperative_thread_counts_are_powers_of_two_of_at_least_eight() {
    for &candidate in RegulariseCandidate::ALL {
        if let RegulariseLayout::Cooperative { threads } = candidate.layout() {
            assert!(threads.is_power_of_two(), "{}", candidate.name());
            assert!(threads >= 8, "{}", candidate.name());
        }
    }
}

#[test]
fn cooperative_shared_bytes_add_one_partial_per_candidate_per_thread() {
    let client = test_client();
    let site = regularise_site(&client, &textured_spec());
    let serial_bytes = 16 * 16 * 4 + 30 * 4;

    assert_eq!(
        site.shared_bytes(RegulariseCandidate::Coop32),
        serial_bytes + 7 * 32 * 4
    );
    assert_eq!(
        site.shared_bytes(RegulariseCandidate::Coop64),
        serial_bytes + 7 * 64 * 4
    );
}

#[test]
fn cooperative_candidates_are_rounding_and_prescreened() {
    let coop = RegulariseCandidate::Coop32.fidelity();

    assert_eq!(coop.class(), FidelityClass::Rounding);
    assert!(matches!(coop, Fidelity::Rounding { .. }));
    const { assert!(<RegulariseSite<TestRuntime> as VariantSite<TestRuntime>>::PRESCREEN) };
}

#[test]
fn deterministic_choice_never_picks_a_cooperative_candidate() {
    let records: Vec<CandidateRecord> = RegulariseCandidate::ALL
        .iter()
        .map(|&candidate| {
            let is_cooperative = matches!(candidate.layout(), RegulariseLayout::Cooperative { .. });
            let median_ns = if is_cooperative { 1_000 } else { 10_000 };

            CandidateRecord {
                name: candidate.name().to_string(),
                fidelity: candidate.fidelity().class(),
                baseline: candidate == RegulariseCandidate::baseline(),
                median_ns,
                samples: 15,
            }
        })
        .collect();

    let chosen = choose(&records, true).expect("a candidate is chosen");

    assert_eq!(chosen.fidelity, FidelityClass::Exact, "{}", chosen.name);
}

#[test]
fn the_cooperative_kernels_match_the_verbatim_reference() {
    let client = test_client();
    let mut textured_confidence_written = false;

    for spec in regularise_specs() {
        let site = regularise_site(&client, &spec);
        let (reference_field, reference_scores) = reference_outputs(&site);
        let reference_wrote_scores = reference_scores.iter().any(|&score| score != 0.0);
        let vectors = site.outputs.neighbours as usize * site.outputs.blocks;
        let allowed = (vectors as f64 * COOP_MAX_FLIPPED_SHARE).floor() as usize;

        for candidate in [RegulariseCandidate::Coop32, RegulariseCandidate::Coop64] {
            let label = format!("{spec:?} {}", candidate.name());
            let outputs = site.scratch_outputs();
            site.launch(candidate, &outputs);

            let vectors_len = outputs.vectors_len();
            let confidence_len = outputs.confidence_len();
            let field = read_i32(&client, &outputs.vectors, vectors_len).expect("vectors read back");
            let scores =
                read_f32(&client, &outputs.confidence, confidence_len).expect("confidence reads back");
            let agreement = compare_fields(&outputs, &reference_field, &field, &reference_scores, &scores);
            let flipped = agreement.flipped;
            let difference = agreement.confidence.max_abs;
            assert!(difference <= COOP_MAX_ABS, "{label}: confidence {difference:e}");

            assert!(flipped <= allowed, "{label}: {flipped} flipped vectors");

            if spec.content == RegulariseContent::Textured {
                let wrote_vectors = field.iter().any(|&component| component != 0);
                assert!(wrote_vectors, "{label}: wrote an all-zero field");

                let wrote_scores = scores.iter().any(|&score| score != 0.0);
                assert_eq!(
                    wrote_scores, reference_wrote_scores,
                    "{label}: confidence written"
                );
                textured_confidence_written |= wrote_scores;
            }
        }
    }

    assert!(
        textured_confidence_written,
        "the cooperative kernels wrote no confidence on textured content"
    );
}

#[test]
fn tie_heavy_content_picks_the_same_vectors_as_the_serial_kernel() {
    let client = test_client();

    for spec in regularise_specs() {
        if spec.content == RegulariseContent::Textured {
            continue;
        }

        let site = regularise_site(&client, &spec);
        let serial = site.scratch_outputs();
        site.launch(RegulariseCandidate::Serial8x8, &serial);
        let serial_field =
            read_i32(&client, &serial.vectors, serial.vectors_len()).expect("vectors read back");

        for candidate in [RegulariseCandidate::Coop32, RegulariseCandidate::Coop64] {
            let coop = site.scratch_outputs();
            site.launch(candidate, &coop);
            let coop_field = read_i32(&client, &coop.vectors, coop.vectors_len()).expect("vectors read back");
            assert_eq!(coop_field, serial_field, "{spec:?} {}", candidate.name());
        }
    }
}

#[test]
fn cooperative_confidence_stays_well_inside_its_declared_bound() {
    let client = test_client();
    let mut worst = 0.0f32;
    let mut flipped = 0usize;

    for spec in regularise_specs() {
        let site = regularise_site(&client, &spec);
        let serial = site.scratch_outputs();
        site.launch(RegulariseCandidate::Serial8x8, &serial);
        let serial_field =
            read_i32(&client, &serial.vectors, serial.vectors_len()).expect("vectors read back");
        let serial_scores =
            read_f32(&client, &serial.confidence, serial.confidence_len()).expect("confidence reads back");

        for candidate in [RegulariseCandidate::Coop32, RegulariseCandidate::Coop64] {
            let coop = site.scratch_outputs();
            site.launch(candidate, &coop);
            let coop_field = read_i32(&client, &coop.vectors, coop.vectors_len()).expect("vectors read back");
            let coop_scores =
                read_f32(&client, &coop.confidence, coop.confidence_len()).expect("confidence reads back");

            let agreement = compare_fields(&serial, &serial_field, &coop_field, &serial_scores, &coop_scores);
            worst = worst.max(agreement.confidence.max_abs);
            flipped += agreement.flipped;
        }
    }

    eprintln!("worst cooperative confidence difference {worst:e}, {flipped} flipped vectors");
    assert!(
        worst <= COOP_MAX_ABS / 10.0,
        "difference {worst:e} is too close to the bound"
    );
}

#[test]
fn cooperative_candidates_pass_the_output_check_against_the_serial_kernel() {
    let client = test_client();

    for spec in regularise_specs() {
        let site = regularise_site(&client, &spec);
        let label = format!("{spec:?}");
        assert_candidates_match_baseline(&site, &label);
    }
}

#[test]
fn forcing_the_cooperative_kernel_keeps_nl4d_output_within_rounding() {
    let options = Nl4dOptions {
        temporal_radius: 1,
        ..Nl4dOptions::default()
    };
    let channels = ChannelMode::Luma;
    let (width, height) = (72u32, 48u32);
    let frames = noisy_frames(width, height, 1, 7);
    let baseline_plan = KernelPlan::baseline();
    let baseline = nl4d_outputs(options, channels, width, height, &frames, baseline_plan);

    for candidate in [RegulariseCandidate::Coop32, RegulariseCandidate::Coop64] {
        let plan = KernelPlan::baseline().with_site(NL4D_MV_REGULARISE, candidate.name());
        let forced = nl4d_outputs(options, channels, width, height, &frames, plan);
        assert_eq!(forced.len(), baseline.len());

        for (forced_frame, baseline_frame) in forced.iter().zip(&baseline) {
            let diff = diff_plane(baseline_frame, forced_frame);
            assert!(diff.psnr >= 60.0, "{}: psnr {:.2}", candidate.name(), diff.psnr);
        }
    }
}

fn upload_field(
    site: &RegulariseSite<TestRuntime>,
    vectors: &[i32],
    confidence: &[f32],
) -> RegulariseOutputs {
    let vector_bytes = i32::as_bytes(vectors);
    let confidence_bytes = f32::as_bytes(confidence);
    let vectors = site.client.create_from_slice(vector_bytes);
    let confidence = site.client.create_from_slice(confidence_bytes);

    RegulariseOutputs {
        vectors,
        confidence,
        ..site.outputs.clone()
    }
}

#[test]
fn rounding_compare_allows_flipped_vectors_up_to_the_share() {
    let client = test_client();
    let spec = RegulariseSpec {
        width: 256,
        height: 256,
        content: RegulariseContent::Flat,
        field_lambda: 1.0,
        neighbours: 1,
    };
    let site = regularise_site(&client, &spec);
    let blocks = site.outputs.blocks;
    let allowed = (blocks as f64 * COOP_MAX_FLIPPED_SHARE).floor() as usize;
    assert!(allowed > 0, "{blocks} blocks allow no flipped vector");

    let fidelity = RegulariseCandidate::Coop32.fidelity();
    let expected_vectors = vec![1i32; site.outputs.vectors_len()];
    let confidence = vec![0.5f32; site.outputs.confidence_len()];
    let baseline = upload_field(&site, &expected_vectors, &confidence);

    let mut under_vectors = expected_vectors.clone();
    for block in 0..allowed {
        under_vectors[2 * block] = -1;
    }

    let under = upload_field(&site, &under_vectors, &confidence);
    let under_verdict = site
        .compare(&baseline, &under, fidelity)
        .expect("the outputs read back");
    assert!(under_verdict.passed(), "{under_verdict:?}");

    let mut over_vectors = under_vectors.clone();
    over_vectors[2 * allowed + 1] = -1;

    let over = upload_field(&site, &over_vectors, &confidence);
    let over_verdict = site
        .compare(&baseline, &over, fidelity)
        .expect("the outputs read back");
    assert!(!over_verdict.passed(), "{over_verdict:?}");
}

/// A regularise site whose motion field holds one vector over each 3x3 patch of blocks.
fn patched_field_site(
    client: &ComputeClient<TestRuntime>,
    spec: &RegulariseSpec,
) -> RegulariseSite<TestRuntime> {
    let mut site = regularise_site(client, spec);
    let mv_stride = site.outputs.mv_stride as usize;
    let neighbours = site.outputs.neighbours as usize;
    let mut field = vec![0i32; neighbours * mv_stride];

    for neighbour in 0..neighbours {
        for block_row in 0..site.blocks_y {
            for block_col in 0..site.blocks_x {
                let block = (block_row * site.blocks_x + block_col) as usize;
                let index = neighbour * mv_stride + 2 * block;
                field[index] = (block_col / 3 % 3) as i32 - 1 + neighbour as i32;
                field[index + 1] = (block_row / 3 % 2) as i32;
            }
        }
    }

    let field_bytes = i32::as_bytes(&field);
    site.mv_in = client.create_from_slice(field_bytes);
    site
}

/// At each patch's centre block, every adjacent block and the median repeat the block's own vector.
#[test]
fn repeated_candidates_pick_the_reference_kernel_vectors() {
    let client = test_client();
    let contents = [
        RegulariseContent::Flat,
        RegulariseContent::Periodic,
        RegulariseContent::Textured,
    ];

    for content in contents {
        let spec = RegulariseSpec {
            width: 72,
            height: 40,
            content,
            field_lambda: 1.0,
            neighbours: 4,
        };
        let site = patched_field_site(&client, &spec);
        let (expected_vectors, expected_confidence) = reference_outputs(&site);
        let vector_count = site.outputs.neighbours as usize * site.outputs.blocks;
        let allowed = (vector_count as f64 * COOP_MAX_FLIPPED_SHARE).floor() as usize;

        for &candidate in RegulariseCandidate::ALL {
            let label = format!("{content:?} {}", candidate.name());
            let outputs = site.scratch_outputs();
            site.launch(candidate, &outputs);

            let vectors_len = outputs.vectors_len();
            let confidence_len = outputs.confidence_len();
            let got_vectors = read_i32(&client, &outputs.vectors, vectors_len).expect("vectors read back");
            let got_confidence =
                read_f32(&client, &outputs.confidence, confidence_len).expect("confidence reads back");

            if candidate.fidelity().class() == FidelityClass::Exact {
                assert_eq!(got_vectors, expected_vectors, "{label}: vectors");
                assert_eq!(got_confidence, expected_confidence, "{label}: confidence");
                continue;
            }

            let agreement = compare_fields(
                &outputs,
                &expected_vectors,
                &got_vectors,
                &expected_confidence,
                &got_confidence,
            );
            let flipped = agreement.flipped;
            let difference = agreement.confidence.max_abs;
            assert!(flipped <= allowed, "{label}: {flipped} flipped vectors");
            assert!(difference <= COOP_MAX_ABS, "{label}: confidence {difference:e}");

            // Flat and periodic SADs tie exactly, so rounding cannot flip a vector.
            if content != RegulariseContent::Textured {
                assert_eq!(got_vectors, expected_vectors, "{label}: vectors");
            }
        }
    }
}

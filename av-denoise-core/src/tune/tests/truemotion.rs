use cubecl::prelude::*;

use super::engine::{nl4d_outputs, noisy_frames};
use super::equivalence::assert_candidates_match_baseline;
use super::fixtures::{
    TruemotionContent,
    TruemotionSpec,
    TruemotionStage,
    truemotion_neighbour_slot,
    truemotion_site,
};
use super::runtime::test_client;
use crate::Nl4dOptions;
use crate::motion::truemotion::kernels::tests::reference::nl4d_mv_regularise_reference;
use crate::nlmeans::ChannelMode;
use crate::noise::DERIVED_WORDS;
use crate::tune::gpu::{bind, read_f32, read_i32};
use crate::tune::sites::{
    COOP_MAX_ABS,
    COOP_MAX_FLIPPED_SHARE,
    MC_TRUEMOTION,
    TruemotionCandidate,
    TruemotionLayout,
    TruemotionOutputs,
    TruemotionSite,
    candidate_known,
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

/// The largest confidence difference the serial kernels may show against the verbatim reference.
///
/// The serial kernels sum each row before adding the rows, where the reference sums the block in
/// one run, so the two round differently.
const REFERENCE_MAX_ABS: f32 = 1.0e-5;

fn textured_spec(stage: TruemotionStage) -> TruemotionSpec {
    TruemotionSpec {
        width: 72,
        height: 40,
        content: TruemotionContent::Textured,
        field_lambda: 1.0,
        neighbours: 4,
        stage,
    }
}

#[test]
fn truemotion_candidate_names_round_trip() {
    for &candidate in TruemotionCandidate::ALL {
        let name = candidate.name();
        let parsed = TruemotionCandidate::from_name(name);
        assert_eq!(parsed, Some(candidate), "{name}");
    }

    assert_eq!(TruemotionCandidate::baseline(), TruemotionCandidate::Serial8x8);
}

/// The centre tile plus the vector list, its SADs, the state words and one partial sum.
#[test]
fn serial_shared_bytes_count_the_centre_tile_and_every_scratch_array() {
    let client = test_client();
    let spec = textured_spec(TruemotionStage::Shipped);
    let site = truemotion_site(&client, &spec);

    // 16 x 16 centre floats, then 26 + 13 + 9 + 3 words of vectors, SADs and state, 36 + 9 staged
    // neighbourhood and flag words, 13 x 16 row sums and one partial word.
    assert_eq!(
        site.shared_bytes(TruemotionCandidate::Serial8x8),
        (16 * 16 + 96 + 13 * 16 + 1) * 4
    );
}

#[test]
fn truemotion_keys_name_the_block_geometry_neighbours_and_candidate_sources() {
    let client = test_client();
    let legacy_spec = textured_spec(TruemotionStage::Legacy);
    let shipped_spec = textured_spec(TruemotionStage::Shipped);
    let legacy = truemotion_site(&client, &legacy_spec);
    let shipped = truemotion_site(&client, &shipped_spec);

    let legacy_key = legacy.key();
    let shipped_key = shipped.key();

    assert_eq!(
        legacy_key.as_str(),
        "mc_truemotion|w128|h64|blksize=16|step=8|neighbours=4|seeds=0|model=0|refine=0|changes=0|floor=0|\
         writes=2"
    );
    assert_eq!(
        shipped_key.as_str(),
        "mc_truemotion|w128|h64|blksize=16|step=8|neighbours=4|seeds=1|model=1|refine=1|changes=3|floor=1|\
         writes=3"
    );
}

pub(super) fn truemotion_specs() -> Vec<TruemotionSpec> {
    let mut specs = Vec::new();

    for content in [
        TruemotionContent::Textured,
        TruemotionContent::Flat,
        TruemotionContent::Periodic,
    ] {
        for (width, height) in [(72u32, 40u32), (61, 37), (128, 64)] {
            for field_lambda in [0.0f32, 1.0, 4.0] {
                for neighbours in [1u32, 4] {
                    for stage in [TruemotionStage::Legacy, TruemotionStage::Shipped] {
                        specs.push(TruemotionSpec {
                            width,
                            height,
                            content,
                            field_lambda,
                            neighbours,
                            stage,
                        });
                    }
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

    for spec in truemotion_specs() {
        let site = truemotion_site(&client, &spec);
        let label = format!("{spec:?}");

        let fitted = fit_candidates(&site, &limits).expect("the baseline fits");
        assert!(fitted.contains(&TruemotionCandidate::Serial8x4), "{label}");
        assert!(fitted.contains(&TruemotionCandidate::Serial4x4), "{label}");

        let baseline = assert_candidates_match_baseline(&site, &label);
        let vectors =
            read_i32(&client, &baseline.vectors, baseline.vectors_len()).expect("vectors read back");
        if spec.content == TruemotionContent::Textured {
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

    for &candidate in TruemotionCandidate::ALL {
        if candidate.fidelity().class() != FidelityClass::Exact {
            continue;
        }

        let plan = KernelPlan::baseline().with_site(MC_TRUEMOTION, candidate.name());
        let forced = nl4d_outputs(options, channels, width, height, &frames, plan);
        assert_eq!(forced, baseline, "{}", candidate.name());
    }
}

#[test]
fn the_narrow_serial_shapes_are_candidates() {
    let eight_by_four = TruemotionCandidate::from_name("serial8x4");
    let four_by_four = TruemotionCandidate::from_name("serial4x4");

    assert_eq!(eight_by_four, Some(TruemotionCandidate::Serial8x4));
    assert_eq!(four_by_four, Some(TruemotionCandidate::Serial4x4));
}

/// The specs the verbatim reference kernel runs, those of the legacy stage.
fn legacy_specs() -> Vec<TruemotionSpec> {
    let specs = truemotion_specs();
    specs
        .into_iter()
        .filter(|spec| spec.stage == TruemotionStage::Legacy)
        .collect()
}

/// Runs the verbatim per-neighbour kernel once per neighbour and returns the field and confidence.
fn reference_outputs(site: &TruemotionSite) -> (Vec<i32>, Vec<f32>) {
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
        let slot = truemotion_neighbour_slot(outputs.neighbours, neighbour as u32);
        let neighbour_start = slot as u64 * slot_bytes;
        let neighbour_frame = site.pyramid.clone().offset_start(neighbour_start);
        let mv_slice = site.mv_in.clone().offset_start(neighbour * mv_bytes);
        let vectors_slice = outputs.vectors.clone().offset_start(neighbour * mv_bytes);
        let confidence_slice = outputs.confidence.clone().offset_start(neighbour * conf_bytes);
        let cube_count = CubeCount::new_2d(site.blocks_x, site.blocks_y);
        let cube_dim = CubeDim::new_2d(8, 8);
        let centre = bind(&centre_frame, level_len);
        let neighbour_plane = bind(&neighbour_frame, level_len);
        let mv_in = bind(&mv_slice, 2 * blocks);
        let vectors = bind(&vectors_slice, 2 * blocks);
        let confidence = bind(&confidence_slice, blocks);
        let noise = bind(&site.noise, DERIVED_WORDS as usize);

        unsafe {
            nl4d_mv_regularise_reference::launch_unchecked(
                client,
                cube_count,
                cube_dim,
                centre,
                neighbour_plane,
                mv_in,
                vectors,
                confidence,
                site.values.lambda_base,
                noise,
                site.values.thsad,
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
fn the_merged_serial_kernel_picks_the_vectors_of_one_reference_launch_per_neighbour() {
    let client = test_client();
    let mut textured_confidence_written = false;

    for spec in legacy_specs() {
        let site = truemotion_site(&client, &spec);
        let outputs = site.scratch_outputs();
        site.launch(TruemotionCandidate::Serial8x8, &outputs);

        let field = read_i32(&client, &outputs.vectors, outputs.vectors_len()).expect("vectors read back");
        let scores =
            read_f32(&client, &outputs.confidence, outputs.confidence_len()).expect("confidence reads back");
        let (reference_field, reference_scores) = reference_outputs(&site);
        let difference = diff_plane(&reference_scores, &scores).max_abs;

        assert_eq!(field, reference_field, "{spec:?}: vectors");
        assert!(
            difference <= REFERENCE_MAX_ABS,
            "{spec:?}: confidence {difference:e}"
        );

        if spec.content == TruemotionContent::Textured {
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
    for &candidate in TruemotionCandidate::ALL {
        if let TruemotionLayout::Cooperative { threads } = candidate.layout() {
            assert!(threads.is_power_of_two(), "{}", candidate.name());
            assert!(threads >= 8, "{}", candidate.name());
        }
    }
}

#[test]
fn cooperative_shared_bytes_add_one_partial_per_candidate_per_thread() {
    let client = test_client();
    let spec = textured_spec(TruemotionStage::Shipped);
    let site = truemotion_site(&client, &spec);

    // The row sums shrink to one word and the partials grow to 13 per thread.
    assert_eq!(
        site.shared_bytes(TruemotionCandidate::Coop32),
        (16 * 16 + 96 + 13 * 32 + 1) * 4
    );
}

#[test]
fn cooperative_candidates_are_rounding_and_prescreened() {
    let coop = TruemotionCandidate::Coop32.fidelity();

    assert_eq!(coop.class(), FidelityClass::Rounding);
    assert!(matches!(coop, Fidelity::Rounding { .. }));
    const { assert!(<TruemotionSite as VariantSite>::PRESCREEN) };
}

#[test]
fn deterministic_choice_never_picks_a_cooperative_candidate() {
    let records: Vec<CandidateRecord> = TruemotionCandidate::ALL
        .iter()
        .map(|&candidate| {
            let is_cooperative = matches!(candidate.layout(), TruemotionLayout::Cooperative { .. });
            let median_ns = if is_cooperative { 1_000 } else { 10_000 };

            CandidateRecord {
                name: candidate.name().to_string(),
                fidelity: candidate.fidelity().class(),
                baseline: candidate == TruemotionCandidate::baseline(),
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

    for spec in legacy_specs() {
        let site = truemotion_site(&client, &spec);
        let (reference_field, reference_scores) = reference_outputs(&site);
        let reference_wrote_scores = reference_scores.iter().any(|&score| score != 0.0);
        let vectors = site.outputs.neighbours as usize * site.outputs.blocks;
        let allowed = (vectors as f64 * COOP_MAX_FLIPPED_SHARE).floor() as usize;

        let label = format!("{spec:?}");
        let outputs = site.scratch_outputs();
        site.launch(TruemotionCandidate::Coop32, &outputs);

        let vectors_len = outputs.vectors_len();
        let confidence_len = outputs.confidence_len();
        let field = read_i32(&client, &outputs.vectors, vectors_len).expect("vectors read back");
        let scores = read_f32(&client, &outputs.confidence, confidence_len).expect("confidence reads back");
        let agreement = compare_fields(&outputs, &reference_field, &field, &reference_scores, &scores);
        let flipped = agreement.flipped;
        let difference = agreement.confidence.max_abs;
        assert!(difference <= COOP_MAX_ABS, "{label}: confidence {difference:e}");

        assert!(flipped <= allowed, "{label}: {flipped} flipped vectors");

        if spec.content == TruemotionContent::Textured {
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

    assert!(
        textured_confidence_written,
        "the cooperative kernels wrote no confidence on textured content"
    );
}

#[test]
fn tie_heavy_content_picks_the_same_vectors_as_the_serial_kernel() {
    let client = test_client();

    for spec in truemotion_specs() {
        if spec.content == TruemotionContent::Textured {
            continue;
        }

        let site = truemotion_site(&client, &spec);
        let serial = site.scratch_outputs();
        site.launch(TruemotionCandidate::Serial8x8, &serial);
        let serial_field =
            read_i32(&client, &serial.vectors, serial.vectors_len()).expect("vectors read back");

        let coop = site.scratch_outputs();
        site.launch(TruemotionCandidate::Coop32, &coop);
        let coop_field = read_i32(&client, &coop.vectors, coop.vectors_len()).expect("vectors read back");
        assert_eq!(coop_field, serial_field, "{spec:?}");
    }
}

#[test]
fn cooperative_confidence_stays_well_inside_its_declared_bound() {
    let client = test_client();
    let mut worst = 0.0f32;
    let mut flipped = 0usize;

    for spec in truemotion_specs() {
        let site = truemotion_site(&client, &spec);
        let serial = site.scratch_outputs();
        site.launch(TruemotionCandidate::Serial8x8, &serial);
        let serial_field =
            read_i32(&client, &serial.vectors, serial.vectors_len()).expect("vectors read back");
        let serial_scores =
            read_f32(&client, &serial.confidence, serial.confidence_len()).expect("confidence reads back");

        let coop = site.scratch_outputs();
        site.launch(TruemotionCandidate::Coop32, &coop);
        let coop_field = read_i32(&client, &coop.vectors, coop.vectors_len()).expect("vectors read back");
        let coop_scores =
            read_f32(&client, &coop.confidence, coop.confidence_len()).expect("confidence reads back");

        let agreement = compare_fields(&serial, &serial_field, &coop_field, &serial_scores, &coop_scores);
        worst = worst.max(agreement.confidence.max_abs);
        flipped += agreement.flipped;
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

    for spec in truemotion_specs() {
        let site = truemotion_site(&client, &spec);
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

    let coop_name = TruemotionCandidate::Coop32.name();
    let plan = KernelPlan::baseline().with_site(MC_TRUEMOTION, coop_name);
    let forced = nl4d_outputs(options, channels, width, height, &frames, plan);
    assert_eq!(forced.len(), baseline.len());

    for (forced_frame, baseline_frame) in forced.iter().zip(&baseline) {
        let diff = diff_plane(baseline_frame, forced_frame);
        assert!(diff.psnr >= 60.0, "{coop_name}: psnr {:.2}", diff.psnr);
    }
}

#[test]
fn a_stored_pick_of_a_removed_candidate_runs_the_baseline() {
    let known = candidate_known(MC_TRUEMOTION, "coop64");
    assert!(!known, "coop64 is no longer a candidate");

    let options = Nl4dOptions {
        temporal_radius: 1,
        ..Nl4dOptions::default()
    };
    let channels = ChannelMode::Luma;
    let (width, height) = (72u32, 48u32);
    let frames = noisy_frames(width, height, 1, 7);
    let baseline_plan = KernelPlan::baseline();
    let baseline = nl4d_outputs(options, channels, width, height, &frames, baseline_plan);

    let stored_plan = KernelPlan::baseline().with_site(MC_TRUEMOTION, "coop64");
    let stored = nl4d_outputs(options, channels, width, height, &frames, stored_plan);

    assert_eq!(stored, baseline);
}

fn upload_field(site: &TruemotionSite, vectors: &[i32], confidence: &[f32]) -> TruemotionOutputs {
    let vector_bytes = i32::as_bytes(vectors);
    let confidence_bytes = f32::as_bytes(confidence);
    let vectors = site.client.create_from_slice(vector_bytes);
    let confidence = site.client.create_from_slice(confidence_bytes);

    TruemotionOutputs {
        vectors,
        confidence,
        ..site.outputs.clone()
    }
}

#[test]
fn rounding_compare_allows_flipped_vectors_up_to_the_share() {
    let client = test_client();
    let spec = TruemotionSpec {
        width: 256,
        height: 256,
        content: TruemotionContent::Flat,
        field_lambda: 1.0,
        neighbours: 1,
        stage: TruemotionStage::Legacy,
    };
    let site = truemotion_site(&client, &spec);
    let blocks = site.outputs.blocks;
    let allowed = (blocks as f64 * COOP_MAX_FLIPPED_SHARE).floor() as usize;
    assert!(allowed > 0, "{blocks} blocks allow no flipped vector");

    let fidelity = TruemotionCandidate::Coop32.fidelity();
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

/// A truemotion site whose motion field holds one vector over each 3x3 patch of blocks.
fn patched_field_site(client: &Client, spec: &TruemotionSpec) -> TruemotionSite {
    let mut site = truemotion_site(client, spec);
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
        TruemotionContent::Flat,
        TruemotionContent::Periodic,
        TruemotionContent::Textured,
    ];

    for content in contents {
        let spec = TruemotionSpec {
            width: 72,
            height: 40,
            content,
            field_lambda: 1.0,
            neighbours: 4,
            stage: TruemotionStage::Legacy,
        };
        let site = patched_field_site(&client, &spec);
        let (expected_vectors, expected_confidence) = reference_outputs(&site);
        let vector_count = site.outputs.neighbours as usize * site.outputs.blocks;
        let allowed = (vector_count as f64 * COOP_MAX_FLIPPED_SHARE).floor() as usize;

        for &candidate in TruemotionCandidate::ALL {
            let label = format!("{content:?} {}", candidate.name());
            let outputs = site.scratch_outputs();
            site.launch(candidate, &outputs);

            let vectors_len = outputs.vectors_len();
            let confidence_len = outputs.confidence_len();
            let got_vectors = read_i32(&client, &outputs.vectors, vectors_len).expect("vectors read back");
            let got_confidence =
                read_f32(&client, &outputs.confidence, confidence_len).expect("confidence reads back");

            if candidate.fidelity().class() == FidelityClass::Exact {
                let difference = diff_plane(&expected_confidence, &got_confidence).max_abs;
                assert_eq!(got_vectors, expected_vectors, "{label}: vectors");
                assert!(
                    difference <= REFERENCE_MAX_ABS,
                    "{label}: confidence {difference:e}"
                );
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
            if content != TruemotionContent::Textured {
                assert_eq!(got_vectors, expected_vectors, "{label}: vectors");
            }
        }
    }
}

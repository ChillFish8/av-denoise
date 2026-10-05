use super::{
    Setup,
    cross_frame_setup,
    flat_noise_setup,
    output_variance,
    patch_pool_variance,
    reference_cover_counts,
    run_fused,
    three_frame_ring_with_a_planted_match,
    unique_frame,
};
use crate::collab::STEP;
use crate::collab::geometry::refs_along;
use crate::collab::kernels::transforms::dct_noise_profile;
use crate::collab::tests::helpers::{deterministic_texture, plant_patch};

/// At `sigma = 0` nothing is discarded, so every contribution a pixel receives is its own input
/// value and the weighted mean must reproduce it.
///
/// `k_max = 1` covers `k_use = 1`, where the stack transform is a no-op and only the 2D DCT round
/// trip runs. `k_max = 8` forces a full stack over content where every position differs, so all
/// three Haar levels carry non-trivial detail.
#[test]
fn zero_sigma_hands_every_member_back_unchanged() {
    let (width, height) = (32u32, 32u32);
    let frame = unique_frame(width, height);

    for k_max in [1u32, 8] {
        let mut setup = Setup::spatial_only(frame.clone(), width, height);
        setup.k_max = k_max;
        setup.sigma = 0.0;
        setup.spatial_radius = 4;
        let got = run_fused(&setup);

        for (idx, &want) in frame.iter().enumerate() {
            assert!(
                got.wsum[idx] > 0,
                "k_max={k_max} idx={idx}: no member covered this pixel"
            );
            let have = got.pixel(idx);
            assert!(
                (want as f64 - have).abs() < 1e-3,
                "k_max={k_max} idx={idx}: want {want} got {have}"
            );
        }
    }
}

/// Every neighbour holds content unrelated to the centre. A group that reached the accumulators as
/// nothing would leave a covered pixel with an empty weight sum, which normalisation renders black.
#[test]
fn a_badly_matched_group_still_reaches_the_accumulators() {
    let (width, height) = (32u32, 32u32);
    let counts = reference_cover_counts(width, height);

    let mut setup = cross_frame_setup(width, height, 2);
    setup.spatial_radius = 9;
    setup.c_min = 0.0;

    let got = run_fused(&setup);
    let base = setup.centre_slot as usize * setup.pixels();

    for (idx, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }

        assert!(
            got.wsum[base + idx] > 0,
            "{count} references cover pixel {idx} and its weight sum is still {}",
            got.wsum[base + idx],
        );
    }
}

/// A patch corner is weighted by the square of the window's end tap, `0.193` at `beta = 2`, so the
/// smallest weight the fixed point has to resolve drops about fivefold against the uniform case.
#[test]
fn a_windowed_badly_matched_group_still_reaches_the_accumulators() {
    let (width, height) = (32u32, 32u32);
    let counts = reference_cover_counts(width, height);

    let mut setup = cross_frame_setup(width, height, 2);
    setup.spatial_radius = 9;
    setup.c_min = 0.0;
    setup.kaiser_beta = 2.0;

    let got = run_fused(&setup);
    let base = setup.centre_slot as usize * setup.pixels();

    for (idx, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }

        assert!(
            got.wsum[base + idx] > 0,
            "{count} references cover pixel {idx} and its weight sum is still {} with the \
             window on",
            got.wsum[base + idx],
        );
    }
}

/// At `k_max = 1` a group scatters only slot 0, and `sigma = 0` gives every group the same weight,
/// so a pixel's weight counts the patches that covered it.
///
/// That count matches the reference cover count only if every group scattered its own reference
/// position. A search result reaching slot 0 would write off the reference grid and leave the
/// counts uneven.
#[test]
fn the_reference_patch_is_always_the_first_member() {
    let (width, height) = (32u32, 32u32);
    let frame = unique_frame(width, height);
    let mut setup = Setup::spatial_only(frame, width, height);
    setup.k_max = 1;
    setup.sigma = 0.0;
    let got = run_fused(&setup);

    let counts = reference_cover_counts(width, height);
    let unit = got.wsum[0] as i64 / counts[0];
    assert!(unit > 0, "the per-patch weight increment must be positive");

    for (idx, &count) in counts.iter().enumerate() {
        assert_eq!(
            got.wsum[idx] as i64,
            unit * count,
            "pixel {idx} carries {} weight, expected {} reference patches at {unit} each",
            got.wsum[idx],
            count
        );
    }
}

/// The group size is the search space size rounded down to a power of two, capped at `k_max`.
///
/// At `spatial_radius = 1` the clipped rectangle holds 4 positions at a corner reference, 6 at an
/// edge one and 9 in the interior, which round to 4, 4 and 8. A `k_max = 4` run must therefore
/// match wherever rounding already reached 4 and differ wherever it reached 8. Clamping each offset
/// instead of clipping the rectangle would count nine positions at a corner, several of them the
/// same patch, and the corners would stop agreeing.
#[test]
fn group_size_rounds_down_to_a_power_of_two() {
    let (width, height) = (64u32, 64u32);
    let frame = unique_frame(width, height);

    let mut wide = Setup::spatial_only(frame.clone(), width, height);
    wide.spatial_radius = 1;
    let mut narrow = Setup::spatial_only(frame, width, height);
    narrow.spatial_radius = 1;
    narrow.k_max = 4;

    let wide = run_fused(&wide);
    let narrow = run_fused(&narrow);

    let refs_x = refs_along(width);
    let refs_y = refs_along(height);
    let mut interior_differed = 0usize;
    for ref_y in 0..refs_y {
        for ref_x in 0..refs_x {
            let idx = (ref_y * refs_x + ref_x) as usize;
            // A clipped axis contributes 2 positions instead of 3, so a reference is capped below 8
            // unless both of its axes are interior.
            let clipped = ref_x == 0 || ref_y == 0 || ref_x == refs_x - 1 || ref_y == refs_y - 1;
            if clipped {
                assert_eq!(
                    wide.group_weight[idx], narrow.group_weight[idx],
                    "reference ({ref_x}, {ref_y}) sees fewer than 8 positions, so both runs must \
                     round it to a group of 4"
                );
            } else if wide.group_weight[idx] != narrow.group_weight[idx] {
                interior_differed += 1;
            }
        }
    }

    let interior = ((refs_x - 2) * (refs_y - 2)) as usize;
    assert!(
        interior_differed * 2 > interior,
        "expected most of the {interior} interior references to reach a group of 8 and so \
         differ from the k_max = 4 run, only {interior_differed} did"
    );
}

/// One texture is planted at `(4, 4)` and `(16, 12)` over a flat background, and `k_max = 2` keeps
/// the self-match plus one member.
///
/// When that member is the twin, the Haar difference across the pair is zero and the threshold
/// keeps nothing from that level. When it is flat background, the difference level carries the
/// texture too, roughly twice as many coefficients survive and the weight halves. `lambda_ht = 1.0`
/// keeps nearly every coefficient offered, so the retained count tracks how many levels carry
/// content. The control plants the texture once, leaving nothing to match.
#[test]
fn a_planted_twin_is_found() {
    let (width, height) = (32u32, 32u32);
    let texture = deterministic_texture(7);

    let mut twinned = vec![0.2f32; (width * height) as usize];
    plant_patch(&mut twinned, width, 4, 4, &texture);
    plant_patch(&mut twinned, width, 16, 12, &texture);

    let mut alone = vec![0.2f32; (width * height) as usize];
    plant_patch(&mut alone, width, 4, 4, &texture);

    let run = |frame: Vec<f32>| {
        let mut setup = Setup::spatial_only(frame, width, height);
        setup.spatial_radius = 12;
        setup.k_max = 2;
        setup.lambda_ht = 1.0;
        run_fused(&setup)
    };

    let ref_idx = (4 / STEP + (4 / STEP) * refs_along(width)) as usize;
    let twinned_run = run(twinned);
    let alone_run = run(alone);
    let with_twin = twinned_run.group_weight[ref_idx];
    let without_twin = alone_run.group_weight[ref_idx];

    assert!(
        with_twin > without_twin * 1.5,
        "expected the group at (4, 4) to keep far more of its variance when its twin at \
         (16, 12) exists, got weight {with_twin} with the twin and {without_twin} without"
    );
}

/// Confidence is uniform per neighbour, so the skip is the same decision for every group. Both
/// neighbours hold an exact copy and neighbour 0 wins ties, so gating it moves every match onto
/// neighbour 1, and a slot that received even one member would show a non-zero weight sum.
#[test]
fn a_gated_neighbour_receives_no_scatter() {
    let (width, height) = (64u32, 64u32);
    let mut setup = three_frame_ring_with_a_planted_match(width, height);
    // Neighbour 0 is ring slot 0 and neighbour 1 is ring slot 2, so this gates the first of the two.
    let blocks = setup.conf_stride as usize;
    setup.confidence[..blocks].fill(0.0);
    setup.confidence[blocks..].fill(1.0);
    setup.c_min = 0.5;

    let got = run_fused(&setup);
    let gated_sum = got.frame_weight_sum(0);
    let centre_sum = got.frame_weight_sum(1);
    let ungated_sum = got.frame_weight_sum(2);

    assert_eq!(
        gated_sum, 0,
        "the gated neighbour's slot must receive no scatter at all"
    );
    assert!(centre_sum > 0, "the centre slot received nothing");
    assert!(ungated_sum > 0, "the ungated neighbour's slot received nothing");
}

#[test]
fn noise_is_suppressed_on_a_flat_field() {
    let (width, height) = (48u32, 48u32);
    let sigma = 0.04f32;
    let setup = flat_noise_setup(width, height, sigma);
    let input_var = patch_pool_variance(&setup.ring, width, height);
    let got = run_fused(&setup);

    // A run that wrote nothing would read a variance of zero and clear the bound below without
    // filtering, so the output must first be shown to keep the field's brightness.
    let output_sum: f64 = (0..got.accum.len()).map(|i| got.pixel(i)).sum();
    let output_mean = output_sum / got.accum.len() as f64;
    assert!(
        (output_mean - 0.5).abs() < 0.01,
        "expected the filtered field to keep its 0.5 mean, got {output_mean}"
    );

    let output_var = output_variance(&got);

    assert!(
        output_var <= input_var * 0.25,
        "expected filtered variance ({output_var}) to be at most a quarter of the input \
         variance ({input_var})"
    );
}

#[test]
fn group_weight_matches_uniform_theory() {
    let (width, height) = (48u32, 48u32);
    let sigma = 0.04f32;
    let setup = flat_noise_setup(width, height, sigma);
    let weights = run_fused(&setup).group_weight;

    // With every member's variance at `sigma^2` the ladder is a fixed point, so every coefficient
    // the threshold could keep carries variance `sigma^2`. `group_weight` is then exactly
    // `1 / (sigma^2 * n_ret)`, which backs out the mean retained count.
    //
    // That count must include the forced group DC. A hard threshold at 2.7 standard deviations lets
    // about 0.7% of pure-noise coefficients through, so out of the `k_max * PATCH_AREA - 1`
    // non-DC coefficients a full group offers, false positives stay far below that ceiling.
    let sigma2 = sigma * sigma;
    let weight_total: f64 = weights.iter().map(|&weight| weight as f64).sum();
    let mean_weight = weight_total / weights.len() as f64;
    let mean_n_ret = 1.0 / (mean_weight * sigma2 as f64);

    let false_positive_rate = 0.007; // ~P(|Z| >= 2.7) for a standard normal, two-tailed
    let ceiling = (8 * 64 - 1) as f64;
    let expected_n_ret = 1.0 + ceiling * false_positive_rate;

    // The kernel measures a mean retained count around 6 here, near `expected_n_ret` (about 4.5).
    // The lower bound rejects a forced-DC-only threshold (exactly 1), and the upper bound rejects
    // one that keeps everything (close to `ceiling + 1`).
    assert!(
        mean_n_ret > 2.0,
        "expected the mean retained count ({mean_n_ret}) to clearly exceed the forced-DC-\
         only value of 1, proving the threshold is admitting some noise-driven coefficients \
         through by chance, not just forcing the group DC"
    );
    assert!(
        mean_n_ret <= expected_n_ret * 2.0,
        "expected the mean retained count ({mean_n_ret}) to stay within 2x of the false-\
         positive-rate estimate ({expected_n_ret}), well short of the {ceiling} coefficient \
         ceiling"
    );
}

/// The production `dct_noise_profile(0.0)` path is compared against a hand-built `[1.0; 8]`
/// buffer, the exact identity multiplier, which stands in for no profile logic at all.
#[test]
fn dct_profile_rho_zero_matches_a_hand_built_all_ones_profile() {
    let (width, height) = (48u32, 48u32);
    let sigma = 0.04f32;

    let white_profile = dct_noise_profile(0.0);
    assert_eq!(
        white_profile, [1.0f32; 8],
        "dct_noise_profile(0.0) must be exactly [1.0; 8], the property this comparison relies on"
    );

    let produced = flat_noise_setup(width, height, sigma);
    let mut hand_built = flat_noise_setup(width, height, sigma);
    hand_built.profile_override = Some([1.0f32; 8]);

    let produced = run_fused(&produced);
    let hand_built = run_fused(&hand_built);

    let wrote_accum = produced.accum.iter().any(|&value| value != 0);
    let wrote_weight = produced.group_weight.iter().any(|&weight| weight != 0.0);
    assert!(
        wrote_accum || wrote_weight,
        "the kernel must actually have written output for this comparison to mean anything"
    );
    assert_eq!(
        produced.accum, hand_built.accum,
        "the accumulator at rho=0 must be identical to a hand-built all-ones profile, proving \
         correlation shaping off is exactly a no-op"
    );
    assert_eq!(
        produced.group_weight, hand_built.group_weight,
        "group_weight at rho=0 must be identical to a hand-built all-ones profile"
    );
}

/// A positive `rho` moves variance from the high frequencies into the low ones, so a fixed
/// `lambda_ht` reaches a smaller threshold on most non-DC coefficients and more pure noise survives.
///
/// This is the deliberate trade of correlation shaping. Where the true correlation is lower than
/// assumed it under-shrinks, leaving a little noise to preserve detail. A flat, noise-only field
/// isolates that trade.
#[test]
fn higher_rho_retains_more_noise_on_a_flat_field() {
    let (width, height) = (48u32, 48u32);
    let sigma = 0.04f32;

    let white = flat_noise_setup(width, height, sigma);
    let mut shaped = flat_noise_setup(width, height, sigma);
    shaped.rho = 0.86;

    let white_run = run_fused(&white);
    let shaped_run = run_fused(&shaped);
    let var_white = output_variance(&white_run);
    let var_shaped = output_variance(&shaped_run);

    assert!(
        var_shaped > var_white * 1.05,
        "expected rho=0.86 to leave meaningfully more residual variance than rho=0 at the same \
         lambda_ht, got rho=0 variance={var_white} rho=0.86 variance={var_shaped}"
    );
}

/// Both neighbours hold an exact copy of the centre, so every volume's two candidates tie at zero
/// and neighbour 0 takes every match. A radius-1 group falling back to a single frame would leave
/// slot 0 empty as well.
#[test]
fn radius_one_keeps_one_neighbour_per_volume() {
    let setup = three_frame_ring_with_a_planted_match(64, 64);
    let got = run_fused(&setup);
    let first_sum = got.frame_weight_sum(0);
    let centre_sum = got.frame_weight_sum(1);
    let second_sum = got.frame_weight_sum(2);

    assert!(first_sum > 0, "neighbour 0 must hold every volume's frame");
    assert!(centre_sum > 0, "the centre slot received nothing");
    assert_eq!(
        second_sum, 0,
        "a 2x4 volume keeps one neighbour, so the tied second neighbour must receive nothing"
    );
}

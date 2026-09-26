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

/// At `sigma = 0` every threshold is zero, so nothing is discarded and
/// the transform chain must hand every member's own pixels back
/// unchanged.
///
/// Every contribution any pixel receives is then that pixel's own input
/// value, whatever group carried it, and the weighted mean of a set of
/// identical values is that value. `k_max = 1` exercises the `k_use = 1`
/// case, where the stack transform is a no-op and only the 2D DCT round
/// trip runs. `k_max = 8` forces a full stack over content where every
/// position differs from every other, so all three Haar levels carry
/// non-trivial detail coefficients.
#[test]
fn zero_sigma_hands_every_member_back_unchanged() {
    let (w, h) = (32u32, 32u32);
    let frame = unique_frame(w, h);

    for k_max in [1u32, 8] {
        let mut s = Setup::spatial_only(frame.clone(), w, h);
        s.k_max = k_max;
        s.sigma = 0.0;
        s.spatial_radius = 4;
        let got = run_fused(&s);

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

/// A covered pixel never ends with an empty weight sum, even when every neighbour holds content
/// unrelated to the centre.
///
/// A group that reached the accumulators as nothing would leave such a pixel, which normalisation
/// can only render as black.
#[test]
fn a_badly_matched_group_still_reaches_the_accumulators() {
    let (w, h) = (32u32, 32u32);
    let counts = reference_cover_counts(w, h);

    let mut s = cross_frame_setup(w, h, 2);
    s.spatial_radius = 9;
    s.c_min = 0.0;

    let got = run_fused(&s);
    let base = s.centre_slot as usize * s.pixels();
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

/// The same invariant with the aggregation window on.
///
/// A patch corner is weighted by the square of the window's end tap, `0.193` at `beta = 2`, so the
/// smallest weight the fixed point has to resolve drops about fivefold against the uniform case.
#[test]
fn a_windowed_badly_matched_group_still_reaches_the_accumulators() {
    let (w, h) = (32u32, 32u32);
    let counts = reference_cover_counts(w, h);

    let mut s = cross_frame_setup(w, h, 2);
    s.spatial_radius = 9;
    s.c_min = 0.0;
    s.kaiser_beta = 2.0;

    let got = run_fused(&s);
    let base = s.centre_slot as usize * s.pixels();
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

/// The reference patch is always the group's first member.
///
/// At `k_max = 1` a group holds exactly one member, so the only patch it
/// scatters is whichever position slot 0 ended up holding. `sigma = 0`
/// makes every group's weight the same constant, so the weight one pixel
/// accumulates counts the patches that covered it. That count must be
/// exactly the number of reference patches covering it, which only holds
/// if every group scattered its own reference position and nothing else.
/// A group that let a search result reach slot 0 would write somewhere
/// off the reference grid and leave the counts uneven.
#[test]
fn the_reference_patch_is_always_the_first_member() {
    let (w, h) = (32u32, 32u32);
    let mut s = Setup::spatial_only(unique_frame(w, h), w, h);
    s.k_max = 1;
    s.sigma = 0.0;
    let got = run_fused(&s);

    let counts = reference_cover_counts(w, h);
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

/// The group size is the search space size rounded down to a power of
/// two, capped at `k_max`.
///
/// At `spatial_radius = 1` the clipped rectangle holds 4 positions at a
/// corner reference, 6 at an edge one, and 9 in the interior. Rounding
/// therefore takes the edge references from 6 down to 4, and leaves the
/// interior ones at 8. Running the same frame at `k_max = 4` caps every
/// group at 4, so the two runs must agree exactly wherever rounding
/// already reached 4 and differ wherever it reached 8.
///
/// Clipping the rectangle once is what makes those counts right. Were
/// each offset clamped in turn instead, a corner would count nine
/// positions rather than four, several of them the same physical patch,
/// and the corner references would stop agreeing across the two runs.
#[test]
fn group_size_rounds_down_to_a_power_of_two() {
    let (w, h) = (64u32, 64u32);
    let frame = unique_frame(w, h);

    let mut wide = Setup::spatial_only(frame.clone(), w, h);
    wide.spatial_radius = 1;
    let mut narrow = Setup::spatial_only(frame, w, h);
    narrow.spatial_radius = 1;
    narrow.k_max = 4;

    let wide = run_fused(&wide);
    let narrow = run_fused(&narrow);

    let refs_x = refs_along(w);
    let refs_y = refs_along(h);
    let mut interior_differed = 0usize;
    for ry in 0..refs_y {
        for rx in 0..refs_x {
            let idx = (ry * refs_x + rx) as usize;
            // A clipped axis contributes 2 positions instead of 3, so a
            // reference is capped below 8 unless both of its axes are
            // interior.
            let clipped = rx == 0 || ry == 0 || rx == refs_x - 1 || ry == refs_y - 1;
            if clipped {
                assert_eq!(
                    wide.group_weight[idx], narrow.group_weight[idx],
                    "reference ({rx}, {ry}) sees fewer than 8 positions, so both runs must \
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

/// A group that finds a genuine twin agrees with itself, and a group
/// that does not carries far more detail into the threshold.
///
/// One texture is planted twice over a flat background, at `(4, 4)` and
/// at `(16, 12)`. With `k_max = 2` the group at `(4, 4)` keeps the
/// self-match and exactly one other member, so the twin either is that
/// member or the matcher missed it. When it is, the two members are
/// pixel for pixel identical, the Haar difference across the pair is
/// exactly zero everywhere, and the threshold keeps nothing from that
/// level. When it is not, the second member is flat background against a
/// textured reference, the difference level carries the texture too, and
/// roughly twice as many coefficients survive, halving the weight.
///
/// `lambda_ht` sits at 1.0 so the threshold keeps nearly every
/// coefficient it is offered, which is what makes the retained count
/// track the number of levels carrying content rather than the size of
/// the coefficients in them.
///
/// The control run plants the same texture once, leaving nothing in the
/// window for the group to match.
#[test]
fn a_planted_twin_is_found() {
    let (w, h) = (32u32, 32u32);
    let texture = deterministic_texture(7);

    let mut twinned = vec![0.2f32; (w * h) as usize];
    plant_patch(&mut twinned, w, 4, 4, &texture);
    plant_patch(&mut twinned, w, 16, 12, &texture);

    let mut alone = vec![0.2f32; (w * h) as usize];
    plant_patch(&mut alone, w, 4, 4, &texture);

    let run = |frame: Vec<f32>| {
        let mut s = Setup::spatial_only(frame, w, h);
        s.spatial_radius = 12;
        s.k_max = 2;
        s.lambda_ht = 1.0;
        run_fused(&s)
    };

    let ref_idx = (4 / STEP + (4 / STEP) * refs_along(w)) as usize;
    let with_twin = run(twinned).group_weight[ref_idx];
    let without_twin = run(alone).group_weight[ref_idx];

    assert!(
        with_twin > without_twin * 1.5,
        "expected the group at (4, 4) to keep far more of its variance when its twin at \
         (16, 12) exists, got weight {with_twin} with the twin and {without_twin} without"
    );
}

/// A neighbour whose motion-block confidence sits below `c_min` is
/// skipped outright, so no member ever comes from it and its region of
/// the accumulator ring stays untouched.
///
/// The confidence field is uniform per neighbour here, so the skip is
/// the same decision for every group in the frame. Both neighbours hold
/// an exact copy, so neighbour 0 wins every tie. Gating it is what moves
/// every volume's match onto neighbour 1, and a slot that received even
/// one member would show a non-zero weight sum.
#[test]
fn a_gated_neighbour_receives_no_scatter() {
    let (w, h) = (64u32, 64u32);
    let mut s = three_frame_ring_with_a_planted_match(w, h);
    // Neighbour 0 is ring slot 0 and neighbour 1 is ring slot 2, so this
    // gates the first of the two.
    let blocks = s.conf_stride as usize;
    s.confidence[..blocks].fill(0.0);
    s.confidence[blocks..].fill(1.0);
    s.c_min = 0.5;

    let got = run_fused(&s);

    assert_eq!(
        got.frame_weight_sum(0),
        0,
        "the gated neighbour's slot must receive no scatter at all"
    );
    assert!(got.frame_weight_sum(1) > 0, "the centre slot received nothing");
    assert!(
        got.frame_weight_sum(2) > 0,
        "the ungated neighbour's slot received nothing"
    );
}

#[test]
fn noise_is_suppressed_on_a_flat_field() {
    let (w, h) = (48u32, 48u32);
    let sigma = 0.04f32;
    let s = flat_noise_setup(w, h, sigma);
    let input_var = patch_pool_variance(&s.ring, w, h);
    let got = run_fused(&s);

    // A run that wrote nothing would read a variance of zero and clear
    // the bound below without filtering anything, so the output has to
    // be shown to carry the field's own brightness first.
    let output_mean: f64 = (0..got.accum.len()).map(|i| got.pixel(i)).sum::<f64>() / got.accum.len() as f64;
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
    let (w, h) = (48u32, 48u32);
    let sigma = 0.04f32;
    let s = flat_noise_setup(w, h, sigma);
    let weights = run_fused(&s).group_weight;

    // With every member's variance equal to `sigma^2`, the ladder is a
    // fixed point, as
    // `transforms::tests::uniform_variance_is_unchanged_by_the_ladder`
    // shows. Every coefficient the threshold could keep therefore also
    // carries variance `sigma^2`, whatever level or spatial position it
    // came from.
    //
    // `group_weight` is then exactly `1 / (sigma^2 * n_ret)`, so this
    // backs out the mean retained count the run produced and checks two
    // things about it.
    //
    // It must include at least the forced group DC. And a hard threshold
    // at 2.7 standard deviations lets only about 0.7% of pure-noise
    // coefficients through by chance, so out of the up to
    // `k_max * PATCH_AREA - 1` coefficients besides the DC that a full
    // 8-member group offers, the mean false-positive count should be
    // small next to that ceiling rather than close to it.
    let sigma2 = sigma * sigma;
    let mean_weight: f64 = weights.iter().map(|&w| w as f64).sum::<f64>() / weights.len() as f64;
    let mean_n_ret = 1.0 / (mean_weight * sigma2 as f64);

    let false_positive_rate = 0.007; // ~P(|Z| >= 2.7) for a standard normal, two-tailed
    let ceiling = (8 * 64 - 1) as f64;
    let expected_n_ret = 1.0 + ceiling * false_positive_rate;

    // A run against the real kernel at this setup measures a mean
    // retained count around 6 (close to `expected_n_ret`, ~4.5, and
    // nowhere near a naive DC-only assumption of 1, which a 20% band
    // around would reject this correct result outright). The lower
    // bound below is what actually distinguishes a working threshold
    // from two ways it could be broken: forced-DC-only (would measure
    // exactly 1) and "threshold does nothing, keeps everything" (would
    // measure close to `ceiling + 1`, an order of magnitude past the
    // upper bound below).
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

/// `rho = 0` must leave the output bit for bit identical to what it
/// would be with no noise-shaping profile in the computation at all.
///
/// This is checked two ways from the same noisy group, once through the
/// real `dct_noise_profile(0.0)` production path, and once through a
/// profile buffer built entirely by hand, `[1.0; 8]`, which is
/// mathematically the exact identity multiplier and so stands in for "no
/// profile logic at all" without needing a second copy of the kernel to
/// prove it against.
#[test]
fn dct_profile_rho_zero_matches_a_hand_built_all_ones_profile() {
    let (w, h) = (48u32, 48u32);
    let sigma = 0.04f32;

    assert_eq!(
        dct_noise_profile(0.0),
        [1.0f32; 8],
        "dct_noise_profile(0.0) must be exactly [1.0; 8], the property this comparison relies on"
    );

    let produced = flat_noise_setup(w, h, sigma);
    let mut hand_built = flat_noise_setup(w, h, sigma);
    hand_built.profile_override = Some([1.0f32; 8]);

    let produced = run_fused(&produced);
    let hand_built = run_fused(&hand_built);

    assert!(
        produced.accum.iter().any(|&v| v != 0) || produced.group_weight.iter().any(|&w| w != 0.0),
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

/// Higher `rho` must retain more residual noise on a flat, noise-only
/// field than `rho = 0` does, at the same `lambda_ht`.
///
/// A positive `rho` moves variance out of the high frequencies and into
/// the low ones (`dct_noise_profile`'s own monotonic-decrease property),
/// so a fixed `lambda_ht` reaches a smaller threshold on most non-DC
/// coefficients than the white-noise assumption would, and more of the
/// pure noise sitting in those coefficients survives. This is the
/// documented, deliberate trade the shipped table's caveat describes. On
/// content where the true correlation is lower than the table assumes,
/// shaping under-shrinks rather than over-shrinks, trading a little
/// leftover noise for preserved detail. A flat, noise-only field
/// isolates that trade with nothing else going on.
#[test]
fn higher_rho_retains_more_noise_on_a_flat_field() {
    let (w, h) = (48u32, 48u32);
    let sigma = 0.04f32;

    let white = flat_noise_setup(w, h, sigma);
    let mut shaped = flat_noise_setup(w, h, sigma);
    shaped.rho = 0.86;

    let var_white = output_variance(&run_fused(&white));
    let var_shaped = output_variance(&run_fused(&shaped));

    assert!(
        var_shaped > var_white * 1.05,
        "expected rho=0.86 to leave meaningfully more residual variance than rho=0 at the same \
         lambda_ht, got rho=0 variance={var_white} rho=0.86 variance={var_shaped}"
    );
}

/// At radius 1 each volume keeps one neighbour frame, and the first-listed neighbour wins a tie.
///
/// Both neighbours hold an exact copy of the centre, so every volume's two candidates tie at zero
/// and slot 0, neighbour 0, takes every match. A radius-1 group falling back to a single frame
/// would leave slot 0 empty as well.
#[test]
fn radius_one_keeps_one_neighbour_per_volume() {
    let s = three_frame_ring_with_a_planted_match(64, 64);
    let got = run_fused(&s);

    assert!(
        got.frame_weight_sum(0) > 0,
        "neighbour 0 must hold every volume's frame"
    );
    assert!(got.frame_weight_sum(1) > 0, "the centre slot received nothing");
    assert_eq!(
        got.frame_weight_sum(2),
        0,
        "a 2x4 volume keeps one neighbour, so the tied second neighbour must receive nothing"
    );
}

use cubecl::prelude::*;

use super::grouping::{BLKSIZE, THSAD};
use super::helpers::{R, make_client, noisy_copy_of, textured_base};
use crate::collab::kernels::fused::mismatch_sigma2;
use crate::collab::kernels::transforms::haar_variance_ladder;
use crate::nl4d::{Nl4dDenoiser, Nl4dParams};
use crate::nlmeans::{ChannelMode, HqParams, MotionCompensationMode, MotionEstimation, NlmParams};

const REFINE: u32 = 2;

/// The formula [`mismatch_sigma2`] runs on the GPU, mirrored on the host
/// for these tests, with the same argument order and the same
/// operations, so floating-point rounding matches to well within the
/// tolerances below.
fn expected_mismatch_sigma2(confidence: f32, thsad: f32, blksize: u32) -> f32 {
    let blksize_area = (blksize * blksize) as f32;
    let ratio = (1.0 - confidence) / (1.0 + confidence);
    let e2 = thsad * thsad * ratio;
    let eps = e2.sqrt() / blksize_area;
    std::f32::consts::FRAC_PI_2 * eps * eps
}

/// Runs [`mismatch_sigma2`] on the GPU, one confidence per thread, so
/// the host mirror above is checked against the code the filter
/// actually calls rather than against itself.
#[cube(launch_unchecked)]
fn mismatch_sigma2_probe(
    confidence: &Array<f32>,
    thsad: f32,
    blksize_area: f32,
    out: &mut Array<f32>,
    #[comptime] n: u32,
) {
    let i = ABSOLUTE_POS_X;
    if i < n {
        out[i as usize] = mismatch_sigma2(confidence[i as usize], thsad, blksize_area);
    }
}

fn run_mismatch_sigma2(confidences: &[f32], thsad: f32, blksize: u32) -> Vec<f32> {
    let client = make_client();
    let n = confidences.len();
    let conf_buf = client.create_from_slice(f32::as_bytes(confidences));
    // One output slot per confidence. `size_of_val(confidences)` reaches
    // the same number but ties the output's size to the input's slice.
    #[expect(
        clippy::manual_slice_size_calculation,
        reason = "n is the element count this output holds, not the input's byte length"
    )]
    let out_buf = client.empty(n * size_of::<f32>());

    unsafe {
        mismatch_sigma2_probe::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(64),
            ArrayArg::from_raw_parts(conf_buf, n),
            thsad,
            (blksize * blksize) as f32,
            ArrayArg::from_raw_parts(out_buf.clone(), n),
            n as u32,
        );
    }

    let bytes = client.read_one(out_buf).expect("mismatch_sigma2 readback failed");
    f32::from_bytes(&bytes)[..n].to_vec()
}

/// `c = 1.0`, a perfect motion match, must give `sigma_m2 = 0.0`
/// exactly, whatever `thsad` is.
#[test]
fn confidence_one_gives_zero_mismatch_variance() {
    for thsad in [THSAD, 0.5, 12.0] {
        let out = run_mismatch_sigma2(&[1.0f32], thsad, BLKSIZE);
        assert_eq!(
            out[0], 0.0,
            "a perfect match must carry no mismatch variance at thsad={thsad}, got {}",
            out[0]
        );
    }
}

/// A known confidence must produce exactly the variance
/// [`expected_mismatch_sigma2`] derives, over the whole range the
/// confidence field can hold.
#[test]
fn low_confidence_produces_the_derived_mismatch_variance() {
    let confidences = [0.0f32, 0.05, 0.2, 0.5, 0.8, 0.95];
    let out = run_mismatch_sigma2(&confidences, THSAD, BLKSIZE);

    for (idx, &c) in confidences.iter().enumerate() {
        let expected = expected_mismatch_sigma2(c, THSAD, BLKSIZE);
        assert!(
            (out[idx] - expected).abs() < 1e-9,
            "expected sigma_m2 {expected} for confidence {c}, got {}",
            out[idx]
        );
    }

    // Sanity: a low confidence must actually derive a value far from
    // zero, or the assertions above would pass trivially against a
    // broken formula that always returns ~0.
    let low = expected_mismatch_sigma2(0.2, THSAD, BLKSIZE);
    assert!(low > 1e-4, "expected a non-trivial mismatch variance, got {low}");
}

// The mismatch variance only ever reaches a motion-predicted member.
// A centre-frame member is not motion-predicted, so there is no
// mismatch to model and it keeps the plain `sigma^2` whatever the
// confidence field holds.
// `collab::tests::fused::centre_frame_members_ignore_the_confidence_field`
// runs the filter with every neighbour gated out, leaving nothing but
// centre-frame members, and shows the `confidence_variance` flag then
// changes nothing at all even with the confidence buffer at 0.0, the
// worst value the formula above can see.

/// Inflating exactly one member's variance must raise exactly the stack
/// rows that member participates in and leave every other row
/// unchanged, checked against the host-mirror ladder
/// [`haar_variance_ladder`] (already pinned against the GPU
/// `variance_reg_level` it mirrors by
/// `collab::tests::transforms::gpu_variance_ladder_matches_the_host_mirror`).
///
/// For `k_use = 8`, member `m` feeds into exactly the rows the Haar
/// butterfly pairs it into: the DC rows `{0, 1}`, its quad row (`2` for
/// members `0..4`, `3` for `4..8`), and its own pair row (`4 + m / 2`).
/// Inflating member `3` therefore must move rows `{0, 1, 2, 5}` and
/// leave rows `{3, 4, 6, 7}` exactly where the uniform baseline put
/// them.
#[test]
fn inflated_member_variance_raises_exactly_the_rows_it_touches() {
    let base_sig2 = 0.0004f32; // a plausible sigma^2, e.g. sigma = 0.02
    let inflated_member = 3usize;
    let inflated_sig2 = base_sig2 + 0.05;

    let baseline = [base_sig2; 8];
    let mut mixed = [base_sig2; 8];
    mixed[inflated_member] = inflated_sig2;

    let baseline_ladder = haar_variance_ladder(&baseline, 8);
    let mixed_ladder = haar_variance_ladder(&mixed, 8);

    let touched: [usize; 4] = [0, 1, 2, 5];
    let untouched: [usize; 4] = [3, 4, 6, 7];

    for &row in &touched {
        assert!(
            mixed_ladder[row] > baseline_ladder[row] + 1e-9,
            "row {row} should rise when member {inflated_member} is inflated, baseline={} \
             mixed={}",
            baseline_ladder[row],
            mixed_ladder[row]
        );
    }
    for &row in &untouched {
        assert!(
            (mixed_ladder[row] - baseline_ladder[row]).abs() < 1e-9,
            "row {row} should hold steady when member {inflated_member} is inflated, \
             baseline={} mixed={}",
            baseline_ladder[row],
            mixed_ladder[row]
        );
    }
}

fn confidence_variance_test_params(temporal_radius: u32, confidence_variance: bool) -> Nl4dParams {
    mismatch_scale_test_params(temporal_radius, confidence_variance, 1.0)
}

fn mismatch_scale_test_params(
    temporal_radius: u32,
    confidence_variance: bool,
    mismatch_scale: f32,
) -> Nl4dParams {
    const SIGMA: f32 = 6.0 / 255.0;
    Nl4dParams {
        nlm: NlmParams {
            temporal_radius,
            search_radius: 2,
            patch_radius: 2,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: crate::nlmeans::PrefilterMode::None,
            motion_compensation: MotionCompensationMode::Mvtools {
                blksize: 16,
                overlap: 8,
                search_radius: 4,
                pyramid_levels: 2,
                estimation: MotionEstimation::Auto,
            },
            hq: Some(HqParams::with_sigma(SIGMA)),
        },
        temporal_radius,
        refine: REFINE,
        spatial_radius: 9,
        lambda_ht: 2.7,
        c_min: 0.05,
        mismatch_scale,
        confidence_variance,
    }
}

/// Runs [`Nl4dDenoiser`] over a short static-content clip and returns
/// every frame it emits.
fn run_denoiser(
    client: &ComputeClient<R>,
    params: Nl4dParams,
    w: u32,
    h: u32,
    frames: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let mut d = Nl4dDenoiser::<R>::new(client, params, w, h).expect("construction failed");
    let mut outputs = Vec::new();
    for frame in frames {
        d.push_frame(frame);
        if let Some(pending) = d.denoise_submit().expect("denoise_submit failed") {
            let frame = pending.wait().expect("readback failed");
            outputs.push(frame.into_f32().expect("f32 output"));
        }
    }
    outputs
}

/// The `confidence_variance` toggle actually reaches the output.
///
/// An ablation arm built on a flag that silently did nothing would
/// measure the same filter twice. The two arms have to differ on content
/// whose members carry imperfect confidence.
///
/// Note this does not assert a direction or a magnitude. It asserts the
/// switch is live, which is the part a refactor can break silently.
#[test]
fn confidence_variance_toggle_changes_the_output() {
    let client = make_client();
    let (w, h) = (64u32, 64u32);
    let radius = 2u32;
    let base = textured_base(w, h);
    let n = 9usize;
    let frames: Vec<Vec<f32>> = (0..n as u32)
        .map(|seed| noisy_copy_of(&base, w, h, 6.0 / 255.0, seed))
        .collect();

    let on = run_denoiser(
        &client,
        confidence_variance_test_params(radius, true),
        w,
        h,
        &frames,
    );
    let off = run_denoiser(
        &client,
        confidence_variance_test_params(radius, false),
        w,
        h,
        &frames,
    );

    assert_eq!(on.len(), off.len(), "both arms must emit the same frame count");
    assert!(
        !on.is_empty(),
        "expected at least one emitted frame from this clip length"
    );
    assert!(
        on.iter().zip(off.iter()).any(|(a, b)| a != b),
        "confidence_variance=true and false produced identical output on every frame, so the \
         toggle is not reaching the filter"
    );
}

/// A scale of zero leaves every member with the plain channel variance,
/// which is exactly what turning the mechanism off does.
///
/// The two arms have to agree bit for bit. They reach the same place by
/// different routes, one zeroing the variance and one skipping the
/// branch that computes it, so this pins the scale's plumbing against
/// the switch that is known to work.
#[test]
fn a_mismatch_scale_of_zero_reproduces_the_mechanism_off_arm() {
    let client = make_client();
    let (w, h) = (64u32, 64u32);
    let radius = 2u32;
    let base = textured_base(w, h);
    let frames: Vec<Vec<f32>> = (0..9u32)
        .map(|seed| noisy_copy_of(&base, w, h, 6.0 / 255.0, seed))
        .collect();

    let scaled_to_zero = run_denoiser(
        &client,
        mismatch_scale_test_params(radius, true, 0.0),
        w,
        h,
        &frames,
    );
    let mechanism_off = run_denoiser(
        &client,
        mismatch_scale_test_params(radius, false, 1.0),
        w,
        h,
        &frames,
    );

    assert_eq!(scaled_to_zero, mechanism_off);
}

/// Raising the scale moves the filter further from the mechanism-off
/// arm, monotonically.
///
/// This is what makes a ladder over the scale readable. A dial that
/// saturated early, or that moved the output without ordering it, would
/// leave every rung of such a sweep uninterpretable.
///
/// The rungs stop short of the saturation point
/// [`crate::nl4d::MAX_MISMATCH_SCALE`] documents, since the whole claim
/// there is that ordering stops past it.
#[test]
fn a_larger_mismatch_scale_moves_further_from_the_mechanism_off_arm() {
    let client = make_client();
    let (w, h) = (64u32, 64u32);
    let radius = 2u32;
    let base = textured_base(w, h);
    let frames: Vec<Vec<f32>> = (0..9u32)
        .map(|seed| noisy_copy_of(&base, w, h, 6.0 / 255.0, seed))
        .collect();

    let off = run_denoiser(
        &client,
        mismatch_scale_test_params(radius, false, 1.0),
        w,
        h,
        &frames,
    );

    let distance_from_off = |scale: f32| {
        let arm = run_denoiser(
            &client,
            mismatch_scale_test_params(radius, true, scale),
            w,
            h,
            &frames,
        );
        assert_eq!(arm.len(), off.len(), "both arms must emit the same frame count");
        let mut sum = 0.0f64;
        let mut n = 0usize;
        for (a, b) in arm.iter().zip(off.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                sum += (*x as f64 - *y as f64).powi(2);
                n += 1;
            }
        }
        (sum / n as f64).sqrt()
    };

    let mut previous = 0.0f64;
    for scale in [1.0f32, 2.0, 4.0] {
        let d = distance_from_off(scale);
        assert!(
            d > previous,
            "scale {scale} sits {d} from the mechanism-off arm, no further than the {previous} \
             the rung below reached"
        );
        previous = d;
    }
}

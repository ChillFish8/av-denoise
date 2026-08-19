use cubecl::prelude::*;
use cubecl::server::Handle;

use super::helpers::{R, deterministic_texture, make_client, noisy_field_over, plant_patch};
use crate::collab::geometry::{filtered_buf_len, member_buf_len, ref_count, refs_along};
use crate::collab::kernels::aggregate::weight_scale;
use crate::collab::kernels::filter_ht::collab_filter_ht;
use crate::collab::kernels::filter_wiener::{collab_filter_wiener, wiener_shrinkage_factor};
use crate::collab::kernels::group::collab_group_spatial;
use crate::collab::kernels::transforms::dct_noise_profile;

fn variance(data: &[f32]) -> f64 {
    let n = data.len() as f64;
    let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / n;
    data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n
}

struct Groups {
    member_pos: Handle,
    member_count: Handle,
    pos_len: usize,
    refs: usize,
}

/// Runs [`collab_group_spatial`] against `frame` and keeps the raw device
/// buffers, rather than reading them back, so a filter launch can
/// consume them directly.
#[allow(clippy::too_many_arguments)]
fn run_group_raw(
    client: &ComputeClient<R>,
    frame: &Handle,
    frame_len: usize,
    width: u32,
    height: u32,
    noise_floor: f32,
    tau_admit: f32,
    spatial_radius: u32,
    k_max: u32,
) -> Groups {
    let refs_x = refs_along(width);
    let refs_y = refs_along(height);
    let refs = ref_count(width, height);
    let pos_len = member_buf_len(width, height, k_max);

    let member_pos = client.empty(pos_len * size_of::<u32>());
    let member_count = client.empty(refs * size_of::<u32>());

    let grid = CubeCount::new_2d(refs_x, refs_y);
    let dim = CubeDim::new_2d(8, 8);

    unsafe {
        collab_group_spatial::launch_unchecked::<R>(
            client,
            grid,
            dim,
            1usize,
            ArrayArg::from_raw_parts(frame.clone(), frame_len),
            ArrayArg::from_raw_parts(member_pos.clone(), pos_len),
            ArrayArg::from_raw_parts(member_count.clone(), refs),
            0u32,
            noise_floor,
            tau_admit,
            width,
            height,
            1u32,
            k_max,
            spatial_radius,
            refs_x,
        );
    }

    Groups {
        member_pos,
        member_count,
        pos_len,
        refs,
    }
}

/// Runs [`collab_filter_wiener`] against an already-grouped pilot/noisy
/// pair and reads back both output buffers.
#[allow(clippy::too_many_arguments)]
fn run_wiener_raw(
    client: &ComputeClient<R>,
    noisy: &Handle,
    pilot: &Handle,
    frame_len: usize,
    groups: &Groups,
    width: u32,
    height: u32,
    k_max: u32,
    sigma: f32,
    rho: f32,
) -> (Vec<f32>, Vec<f32>) {
    let refs_x = refs_along(width);
    let refs_y = refs_along(height);
    let refs = groups.refs;
    let filt_len = filtered_buf_len(width, height, k_max);
    let pixels = (width * height) as usize;

    let filtered_buf = client.empty(filt_len * size_of::<f32>());
    let weight_buf = client.empty(refs * size_of::<f32>());
    let sigma_buf = client.create_from_slice(f32::as_bytes(&[sigma]));
    let profile = dct_noise_profile(rho);
    let dct_profile_buf = client.create_from_slice(f32::as_bytes(&profile));
    let dummy = client.empty(size_of::<f32>());
    // The filter scatters into these on every launch. These tests read
    // the filtered patches themselves rather than the aggregated frame,
    // so they are allocated only to give the scatter somewhere to land.
    let accum = client.create_from_slice(i32::as_bytes(&vec![0i32; pixels]));
    let wsum = client.create_from_slice(i32::as_bytes(&vec![0i32; pixels]));

    let grid = CubeCount::new_2d(refs_x, refs_y);
    let dim = CubeDim::new_2d(8, 8);

    unsafe {
        collab_filter_wiener::launch_unchecked::<R>(
            client,
            grid,
            dim,
            1usize,
            ArrayArg::from_raw_parts(noisy.clone(), frame_len),
            ArrayArg::from_raw_parts(pilot.clone(), frame_len),
            ArrayArg::from_raw_parts(groups.member_pos.clone(), groups.pos_len),
            ArrayArg::from_raw_parts(groups.member_count.clone(), refs),
            ArrayArg::from_raw_parts(dummy, 1),
            ArrayArg::from_raw_parts(accum, pixels),
            ArrayArg::from_raw_parts(wsum, pixels),
            ArrayArg::from_raw_parts(filtered_buf.clone(), filt_len),
            ArrayArg::from_raw_parts(weight_buf.clone(), refs),
            0u32,
            0u32,
            ArrayArg::from_raw_parts(sigma_buf, 1),
            ArrayArg::from_raw_parts(dct_profile_buf, 8),
            weight_scale(sigma, &profile),
            false,
            true,
            width,
            height,
            1u32,
            k_max,
            1u32,
            refs_x,
        );
    }

    let filtered_bytes = client.read_one(filtered_buf).expect("filtered readback failed");
    let weight_bytes = client.read_one(weight_buf).expect("group_weight readback failed");
    let all_k = f32::from_bytes(&filtered_bytes)[..filt_len].to_vec();

    // The kernel now writes every member of every group. These tests
    // assert against member 0, the reference patch itself, so that one
    // patch per reference is lifted out here and the callers keep their
    // original `ref_idx * PATCH_AREA + pos` indexing.
    let mut member_zero = vec![0.0f32; refs * 64];
    for r in 0..refs {
        let src = r * k_max as usize * 64;
        member_zero[r * 64..(r + 1) * 64].copy_from_slice(&all_k[src..src + 64]);
    }

    (member_zero, f32::from_bytes(&weight_bytes)[..refs].to_vec())
}

/// Runs [`collab_filter_ht`] against an already-grouped frame and reads
/// back the filtered buffer.
#[allow(clippy::too_many_arguments)]
fn run_ht_raw(
    client: &ComputeClient<R>,
    frame: &Handle,
    frame_len: usize,
    groups: &Groups,
    width: u32,
    height: u32,
    k_max: u32,
    sigma: f32,
    lambda_ht: f32,
) -> Vec<f32> {
    let refs_x = refs_along(width);
    let refs_y = refs_along(height);
    let refs = groups.refs;
    let filt_len = filtered_buf_len(width, height, k_max);
    let pixels = (width * height) as usize;

    let filtered_buf = client.empty(filt_len * size_of::<f32>());
    let weight_buf = client.empty(refs * size_of::<f32>());
    let sigma_buf = client.create_from_slice(f32::as_bytes(&[sigma]));
    let profile = dct_noise_profile(0.0);
    let dct_profile_buf = client.create_from_slice(f32::as_bytes(&profile));
    let dummy = client.empty(size_of::<f32>());
    let member_frame_dummy = client.empty(size_of::<u32>());
    let accum = client.create_from_slice(i32::as_bytes(&vec![0i32; pixels]));
    let wsum = client.create_from_slice(i32::as_bytes(&vec![0i32; pixels]));

    let grid = CubeCount::new_2d(refs_x, refs_y);
    let dim = CubeDim::new_2d(8, 8);

    unsafe {
        collab_filter_ht::launch_unchecked::<R>(
            client,
            grid,
            dim,
            1usize,
            ArrayArg::from_raw_parts(frame.clone(), frame_len),
            ArrayArg::from_raw_parts(groups.member_pos.clone(), groups.pos_len),
            ArrayArg::from_raw_parts(member_frame_dummy, 1),
            ArrayArg::from_raw_parts(groups.member_count.clone(), refs),
            ArrayArg::from_raw_parts(dummy, 1),
            ArrayArg::from_raw_parts(accum, pixels),
            ArrayArg::from_raw_parts(wsum, pixels),
            ArrayArg::from_raw_parts(filtered_buf.clone(), filt_len),
            ArrayArg::from_raw_parts(weight_buf, refs),
            0u32,
            ArrayArg::from_raw_parts(sigma_buf, 1),
            ArrayArg::from_raw_parts(dct_profile_buf, 8),
            lambda_ht,
            weight_scale(sigma, &profile),
            false,
            true,
            false,
            false,
            width,
            height,
            1u32,
            k_max,
            1u32,
            refs_x,
        );
    }

    let filtered_bytes = client.read_one(filtered_buf).expect("filtered readback failed");
    let all_k = f32::from_bytes(&filtered_bytes)[..filt_len].to_vec();
    let mut member_zero = vec![0.0f32; refs * 64];
    for r in 0..refs {
        let src = r * k_max as usize * 64;
        member_zero[r * 64..(r + 1) * 64].copy_from_slice(&all_k[src..src + 64]);
    }
    member_zero
}

fn extract_patch(frame: &[f32], w: u32, px: u32, py: u32) -> [f32; 64] {
    let mut out = [0.0f32; 64];
    for row in 0..8u32 {
        for col in 0..8u32 {
            out[(row * 8 + col) as usize] = frame[((py + row) * w + (px + col)) as usize];
        }
    }
    out
}

/// With `sigma = 0` every Wiener denominator degenerates to `p * p`,
/// making every shrinkage factor `1` for any coefficient the pilot
/// carries a nonzero value on, and `0 / 0 -> 0` (via the epsilon floor)
/// for a coefficient that is genuinely zero in both the pilot and the
/// noisy patch. Using the same content as both the pilot and the noisy
/// input, self-only grouped, checks the whole forward/shrink/inverse
/// round trip reproduces the input patch exactly.
#[test]
fn zero_sigma_is_identity() {
    let (w, h) = (32u32, 32u32);
    let mut frame = vec![0.3f32; (w * h) as usize];
    let texture = deterministic_texture(11);
    plant_patch(&mut frame, w, 12, 12, &texture);

    let client = make_client();
    let handle = client.create_from_slice(f32::as_bytes(&frame));
    let groups = run_group_raw(&client, &handle, frame.len(), w, h, 0.0, 1e-6, 6, 8);

    let (filtered, _weights) =
        run_wiener_raw(&client, &handle, &handle, frame.len(), &groups, w, h, 8, 0.0, 0.0);

    let refs_x = refs_along(w);
    let ref_idx = (3 * refs_x + 3) as usize; // ref_pos(3, 32) == 12 on both axes

    let expected = extract_patch(&frame, w, 12, 12);
    let got = &filtered[ref_idx * 64..ref_idx * 64 + 64];

    for (idx, (&want, &have)) in expected.iter().zip(got.iter()).enumerate() {
        assert!((want - have).abs() < 1e-4, "idx={idx}: want {want} got {have}");
    }
}

/// A non-finite sigma must never turn the shrinkage factor into an
/// amplifier.
///
/// `w = p * p / max(p * p + v_j, WIENER_EPSILON)` is a shrinkage factor by
/// definition, mathematically bounded to `[0, 1]` whenever `v_j` is a real
/// variance, since the denominator can only be as small as the numerator.
/// But `v_j` propagates from the caller-supplied sigma, and if that sigma
/// is `NaN` (which a poisoned upstream noise estimate can produce), the
/// sum `p * p + v_j` is `NaN` too. `f32::max(NaN, WIENER_EPSILON)`
/// silently discards the `NaN` and returns `WIENER_EPSILON` (`1e-20`)
/// instead, so the "shrinkage" factor becomes `p * p / 1e-20`, unbounded
/// and enormous for any ordinary coefficient. That is not a shrinkage
/// factor collapsing to a safe default, it is `max` treating an undefined
/// variance as if it were known to be essentially zero, the opposite of
/// the intended fail-safe.
///
/// This plants real (non-flat) content so the pilot carries genuine
/// nonzero coefficients for `w` to blow up, and checks the reconstructed
/// patch never leaves a sane multiple of the input's own range.
#[test]
fn non_finite_sigma_does_not_amplify_the_output() {
    let (w, h) = (32u32, 32u32);
    let mut frame = vec![0.3f32; (w * h) as usize];
    let texture = deterministic_texture(11);
    plant_patch(&mut frame, w, 12, 12, &texture);

    let client = make_client();
    let handle = client.create_from_slice(f32::as_bytes(&frame));
    let groups = run_group_raw(&client, &handle, frame.len(), w, h, 0.0, 1e-6, 6, 8);

    let (filtered, weights) = run_wiener_raw(
        &client,
        &handle,
        &handle,
        frame.len(),
        &groups,
        w,
        h,
        8,
        f32::NAN,
        0.0,
    );

    let in_min = frame.iter().cloned().fold(f32::INFINITY, f32::min);
    let in_max = frame.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let bound = (in_max - in_min).max(1.0) * 4.0;

    for (idx, &v) in filtered.iter().enumerate() {
        assert!(
            v.is_finite() && v.abs() <= in_max.abs().max(in_min.abs()) + bound,
            "filtered[{idx}]={v} escaped a sane multiple of the input range \
             [{in_min}, {in_max}] under a non-finite sigma"
        );
    }
    for (idx, &v) in weights.iter().enumerate() {
        assert!(
            v.is_finite(),
            "group_weight[{idx}]={v} is not finite under a non-finite sigma"
        );
    }
}

/// A flat, exactly noise-free pilot next to a genuinely noisy copy of
/// the same content, both grouped into a full 8-member stack (every
/// candidate the pilot offers is an exact match, since it is flat).
/// Filtering the noisy data through Wiener shrinkage steered by that
/// pilot is compared against filtering the same noisy data, with the
/// same group, through a hard threshold. The pilot's every non-DC
/// coefficient is exactly 0, so Wiener drives every non-DC shrinkage
/// factor to 0 as well, while a hard threshold only rejects
/// noise-driven coefficients probabilistically and lets some through by
/// chance, so Wiener is expected to suppress noise at least as hard.
#[test]
fn flat_field_noise_crushed_harder_than_ht() {
    let (w, h) = (48u32, 48u32);
    let sigma = 0.04f32;
    let lambda_ht = 2.7f32;

    let pilot_frame = vec![0.5f32; (w * h) as usize];
    let noisy_frame = noisy_field_over(w, h, 0.5, sigma);

    let client = make_client();
    let pilot_handle = client.create_from_slice(f32::as_bytes(&pilot_frame));
    let noisy_handle = client.create_from_slice(f32::as_bytes(&noisy_frame));

    // The pilot is exactly flat, so every candidate in the window has
    // distance 0 to the reference, filling every group to the full
    // `k_max = 8` regardless of the admission threshold.
    let groups = run_group_raw(&client, &pilot_handle, pilot_frame.len(), w, h, 0.0, 1e-6, 9, 8);

    let (wiener_filtered, _weights) = run_wiener_raw(
        &client,
        &noisy_handle,
        &pilot_handle,
        noisy_frame.len(),
        &groups,
        w,
        h,
        8,
        sigma,
        0.0,
    );
    let ht_filtered = run_ht_raw(
        &client,
        &noisy_handle,
        noisy_frame.len(),
        &groups,
        w,
        h,
        8,
        sigma,
        lambda_ht,
    );

    let wiener_var = variance(&wiener_filtered);
    let ht_var = variance(&ht_filtered);

    assert!(
        wiener_var <= ht_var + 1e-9,
        "expected Wiener-filtered variance ({wiener_var}) to be at most hard-threshold-filtered \
         variance ({ht_var}), both from the same full 8-member group"
    );
}

#[cube(launch_unchecked)]
fn wiener_shrinkage_factor_probe(p: &Array<f32>, vj: &Array<f32>, out: &mut Array<f32>, n: u32) {
    let tid = ABSOLUTE_POS_X;
    if tid < n {
        out[tid as usize] = wiener_shrinkage_factor(p[tid as usize], vj[tid as usize]);
    }
}

/// Pins the shrinkage factor itself to `[0, 1]` for a spread of pilot
/// and variance inputs, including a `NaN` variance, an infinite
/// variance, a zero pilot, and a huge pilot, rather than only checking
/// that a whole filtered patch stays in some generous range.
///
/// A regression that made the shrinkage factor 1.5 or 3 instead of
/// unbounded would still pass a test that only bounds a filtered patch
/// against several times the input's own range. Bounding the factor
/// directly closes that gap.
#[test]
fn shrinkage_factor_stays_in_zero_one_for_every_input() {
    let client = make_client();

    let p = vec![0.5f32, 0.0, 1.0e6, 0.5, 0.5, f32::NAN, 0.0, 1.0e6];
    let vj = vec![0.01f32, 0.01, 0.01, f32::NAN, f32::INFINITY, 0.01, 0.0, f32::NAN];
    let n = p.len();

    let p_buf = client.create_from_slice(f32::as_bytes(&p));
    let vj_buf = client.create_from_slice(f32::as_bytes(&vj));
    let out_buf = client.empty(n * size_of::<f32>());

    unsafe {
        wiener_shrinkage_factor_probe::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(n as u32),
            ArrayArg::from_raw_parts(p_buf, n),
            ArrayArg::from_raw_parts(vj_buf, n),
            ArrayArg::from_raw_parts(out_buf.clone(), n),
            n as u32,
        );
    }

    let bytes = client
        .read_one(out_buf)
        .expect("shrinkage factor readback failed");
    let w = f32::from_bytes(&bytes)[..n].to_vec();

    for (idx, &value) in w.iter().enumerate() {
        assert!(
            value.is_finite() && (0.0..=1.0).contains(&value),
            "w[{idx}] (p={}, vj={})={value} is outside [0, 1]",
            p[idx],
            vj[idx]
        );
    }

    // A NaN variance (indices 3 and 7) means the noise level for that
    // coefficient is unknown, neither large nor small, so it passes
    // through untouched at exactly 1 rather than being deleted at 0 on
    // a noise level nobody actually measured. Trust of exactly 1 can
    // never amplify anything, so this stays just as safe as shrinking
    // to 0 would have been.
    assert_eq!(w[3], 1.0, "a NaN variance must yield exactly 1, got {}", w[3]);
    assert_eq!(w[7], 1.0, "a NaN variance must yield exactly 1, got {}", w[7]);
    // An infinite variance (index 4) means the coefficient is pure
    // noise, and must also shrink to 0.
    assert_eq!(
        w[4], 0.0,
        "an infinite variance must yield exactly 0, got {}",
        w[4]
    );
    // A zero pilot (index 1) carries no signal, so its coefficient
    // shrinks to 0 regardless of the variance.
    assert_eq!(w[1], 0.0, "a zero pilot must yield exactly 0, got {}", w[1]);
    // A huge pilot next to an ordinary variance (index 2) is confidently
    // real signal, so it passes through close to unchanged.
    assert!(w[2] > 0.99, "a huge pilot should keep w near 1, got {}", w[2]);
}

/// `rho = 0` must leave `collab_filter_wiener`'s output bit for bit
/// identical to a hand-built `[1.0; 8]` profile, the same regression
/// [`super::filter_ht::dct_profile_rho_zero_matches_a_hand_built_all_ones_profile`]
/// runs for the hard-threshold kernel. `[1.0; 8]` is the exact identity
/// multiplier, so this stands in for "no profile logic in the kernel at
/// all" without a second copy of the kernel to compare against.
#[test]
fn dct_profile_rho_zero_matches_a_hand_built_all_ones_profile() {
    let (w, h) = (48u32, 48u32);
    let sigma = 0.04f32;
    let pilot_frame = vec![0.5f32; (w * h) as usize];
    let noisy_frame = noisy_field_over(w, h, 0.5, sigma);

    let client = make_client();
    let pilot_handle = client.create_from_slice(f32::as_bytes(&pilot_frame));
    let noisy_handle = client.create_from_slice(f32::as_bytes(&noisy_frame));
    let groups = run_group_raw(&client, &pilot_handle, pilot_frame.len(), w, h, 0.0, 1e-6, 9, 8);

    let (filtered_zero_rho, weight_zero_rho) = run_wiener_raw(
        &client,
        &noisy_handle,
        &pilot_handle,
        noisy_frame.len(),
        &groups,
        w,
        h,
        8,
        sigma,
        0.0,
    );

    let refs_x = refs_along(w);
    let refs = groups.refs;
    let filt_len = filtered_buf_len(w, h, 8);
    let pixels = (w * h) as usize;
    let filtered_buf = client.empty(filt_len * size_of::<f32>());
    let weight_buf = client.empty(refs * size_of::<f32>());
    let sigma_buf = client.create_from_slice(f32::as_bytes(&[sigma]));
    let ones_profile_buf = client.create_from_slice(f32::as_bytes(&[1.0f32; 8]));
    let dummy = client.empty(size_of::<f32>());
    let accum = client.create_from_slice(i32::as_bytes(&vec![0i32; pixels]));
    let wsum = client.create_from_slice(i32::as_bytes(&vec![0i32; pixels]));
    let grid = CubeCount::new_2d(refs_x, refs_along(h));
    let dim = CubeDim::new_2d(8, 8);
    unsafe {
        collab_filter_wiener::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            1usize,
            ArrayArg::from_raw_parts(noisy_handle.clone(), noisy_frame.len()),
            ArrayArg::from_raw_parts(pilot_handle.clone(), pilot_frame.len()),
            ArrayArg::from_raw_parts(groups.member_pos.clone(), groups.pos_len),
            ArrayArg::from_raw_parts(groups.member_count.clone(), refs),
            ArrayArg::from_raw_parts(dummy, 1),
            ArrayArg::from_raw_parts(accum, pixels),
            ArrayArg::from_raw_parts(wsum, pixels),
            ArrayArg::from_raw_parts(filtered_buf.clone(), filt_len),
            ArrayArg::from_raw_parts(weight_buf.clone(), refs),
            0u32,
            0u32,
            ArrayArg::from_raw_parts(sigma_buf, 1),
            ArrayArg::from_raw_parts(ones_profile_buf, 8),
            weight_scale(sigma, &[1.0f32; 8]),
            false,
            true,
            w,
            h,
            1u32,
            8u32,
            1u32,
            refs_x,
        );
    }
    let all_k = f32::from_bytes(&client.read_one(filtered_buf).expect("filtered readback failed"))
        [..filt_len]
        .to_vec();
    let mut filtered_hand_built = vec![0.0f32; refs * 64];
    for r in 0..refs {
        let src = r * 8 * 64;
        filtered_hand_built[r * 64..(r + 1) * 64].copy_from_slice(&all_k[src..src + 64]);
    }
    let weight_hand_built =
        f32::from_bytes(&client.read_one(weight_buf).expect("weight readback failed"))[..refs].to_vec();

    assert_eq!(
        filtered_zero_rho, filtered_hand_built,
        "filtered output at rho=0 must be bit-for-bit identical to a hand-built all-ones profile"
    );
    assert_eq!(
        weight_zero_rho, weight_hand_built,
        "group_weight at rho=0 must be bit-for-bit identical to a hand-built all-ones profile"
    );
}

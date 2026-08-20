use cubecl::prelude::*;
use cubecl::server::Handle;

use super::helpers::{
    R,
    deterministic_texture,
    make_client,
    make_unique_frame,
    noisy_field_over,
    plant_patch,
};
use crate::collab::geometry::{
    filtered_buf_len,
    member_buf_len,
    member_frame_buf_len,
    member_sig2_buf_len,
    ref_count,
    ref_pos,
    refs_along,
};
use crate::collab::kernels::aggregate::{ACCUM_SCALE, cross_frame_accum_scale, weight_scale};
use crate::collab::kernels::filter_ht::{collab_filter_ht, variance_ladder};
use crate::collab::kernels::group::pack_pos_host;
use crate::collab::kernels::group_temporal::collab_group_temporal;
use crate::collab::kernels::transforms::{dct_noise_profile, haar_variance_ladder};

/// Every reference top-left position on the grid, in the same raster
/// order the kernels use to compute `ref_idx`.
fn ref_positions(width: u32, height: u32) -> Vec<(u32, u32)> {
    let refs_x = refs_along(width);
    let refs_y = refs_along(height);
    let mut out = Vec::with_capacity((refs_x * refs_y) as usize);
    for ry in 0..refs_y {
        for rx in 0..refs_x {
            out.push((ref_pos(rx, width), ref_pos(ry, height)));
        }
    }
    out
}

/// Reads an 8x8 patch out of `frame` at `(px, py)`, row-major, matching
/// the `stack[k * PATCH_AREA + pos]` layout the kernel uses.
fn extract_patch(frame: &[f32], w: u32, px: u32, py: u32) -> [f32; 64] {
    let mut out = [0.0f32; 64];
    for row in 0..8u32 {
        for col in 0..8u32 {
            out[(row * 8 + col) as usize] = frame[((py + row) * w + (px + col)) as usize];
        }
    }
    out
}

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

/// Runs [`collab_group_temporal`] over a one-frame ring and keeps the
/// raw device buffers, rather than reading them back, so a filter launch
/// can consume them directly.
///
/// `radius = 0` means the ring is one frame wide and every candidate
/// comes from the spatial window around the reference patch. The motion
/// field, the confidence field, and the neighbour slot list are all
/// one-element dummies, since nothing reads them at that radius. The
/// groups these tests feed the filter are a fixed input, not the thing
/// under test, so a spatial-only search is all they need.
#[expect(
    clippy::too_many_arguments,
    reason = "mirrors collab_group_temporal's own launch arguments, so grouping them into a struct \
              would only move the same list one level out"
)]
fn run_group_raw(
    client: &ComputeClient<R>,
    reference: &Handle,
    frame_len: usize,
    width: u32,
    height: u32,
    noise_floor: f32,
    spatial_radius: u32,
    k_max: u32,
) -> Groups {
    let refs_x = refs_along(width);
    let refs_y = refs_along(height);
    let refs = ref_count(width, height);
    let pos_len = member_buf_len(width, height, k_max);
    let frame_len_out = member_frame_buf_len(width, height, k_max);
    let sig2_len = member_sig2_buf_len(width, height, k_max);

    let member_pos = client.empty(pos_len * size_of::<u32>());
    let member_frame = client.empty(frame_len_out * size_of::<u32>());
    let member_count = client.empty(refs * size_of::<u32>());
    let member_sig2 = client.empty(sig2_len * size_of::<f32>());
    let mv_dummy = client.create_from_slice(i32::as_bytes(&[0i32, 0i32]));
    let conf_dummy = client.create_from_slice(f32::as_bytes(&[1.0f32]));
    let slots_dummy = client.create_from_slice(u32::as_bytes(&[0u32]));

    let grid = CubeCount::new_2d(refs_x, refs_y);
    let dim = CubeDim::new_2d(8, 8);

    unsafe {
        collab_group_temporal::launch_unchecked::<R>(
            client,
            grid,
            dim,
            1usize,
            ArrayArg::from_raw_parts(reference.clone(), frame_len),
            ArrayArg::from_raw_parts(mv_dummy, 2),
            ArrayArg::from_raw_parts(conf_dummy, 1),
            ArrayArg::from_raw_parts(member_pos.clone(), pos_len),
            ArrayArg::from_raw_parts(member_frame, frame_len_out),
            ArrayArg::from_raw_parts(member_count.clone(), refs),
            ArrayArg::from_raw_parts(member_sig2, sig2_len),
            0u32,
            ArrayArg::from_raw_parts(slots_dummy, 1),
            noise_floor,
            0.0f32,
            1.0f32,
            1.0f32,
            0u32,
            0u32,
            2u32,
            1u32,
            8u32,
            8u32,
            1u32,
            1u32,
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

/// Runs [`collab_filter_ht`] against an already-grouped frame and reads
/// back both output buffers.
///
/// `member_frame`/`member_frame_len`/`temporal` follow the same
/// pattern `member_sig2`/`member_sig2_len`/`use_member_sigma` already
/// use. A 1-element dummy is valid whenever `temporal` is false, since
/// the kernel never reads the buffer in that case.
#[allow(clippy::too_many_arguments)]
fn run_filter_raw(
    client: &ComputeClient<R>,
    reference: &Handle,
    frame_len: usize,
    groups: &Groups,
    width: u32,
    height: u32,
    k_max: u32,
    sigma: f32,
    lambda_ht: f32,
    member_sig2: &Handle,
    member_sig2_len: usize,
    use_member_sigma: bool,
    rho: f32,
    member_frame: &Handle,
    member_frame_len: usize,
    temporal: bool,
    frame: u32,
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
    // The filter scatters into these on every launch. These tests read
    // the filtered patches themselves rather than the aggregated frame,
    // so they are allocated only to give the scatter somewhere to land.
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
            ArrayArg::from_raw_parts(reference.clone(), frame_len),
            ArrayArg::from_raw_parts(groups.member_pos.clone(), groups.pos_len),
            ArrayArg::from_raw_parts(member_frame.clone(), member_frame_len),
            ArrayArg::from_raw_parts(groups.member_count.clone(), refs),
            ArrayArg::from_raw_parts(member_sig2.clone(), member_sig2_len),
            ArrayArg::from_raw_parts(accum, pixels),
            ArrayArg::from_raw_parts(wsum, pixels),
            ArrayArg::from_raw_parts(filtered_buf.clone(), filt_len),
            ArrayArg::from_raw_parts(weight_buf.clone(), refs),
            frame,
            ArrayArg::from_raw_parts(sigma_buf, 1),
            ArrayArg::from_raw_parts(dct_profile_buf, 8),
            lambda_ht,
            weight_scale(sigma, &profile),
            ACCUM_SCALE,
            use_member_sigma,
            true,
            false,
            temporal,
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

/// Groups then filters a luma frame with `use_member_sigma` off and a
/// 1-element dummy `member_sig2`, the configuration to use whenever
/// every member's noise variance is just the channel sigma, with no
/// separate per-member estimate.
#[allow(clippy::too_many_arguments)]
fn run_filter(
    frame: &[f32],
    width: u32,
    height: u32,
    noise_floor: f32,
    spatial_radius: u32,
    k_max: u32,
    sigma: f32,
    lambda_ht: f32,
) -> (Vec<f32>, Vec<f32>) {
    run_filter_with_rho(
        frame,
        width,
        height,
        noise_floor,
        spatial_radius,
        k_max,
        sigma,
        lambda_ht,
        0.0,
    )
}

/// [`run_filter`] with an explicit `rho`, for tests that need
/// correlation shaping turned on rather than the white-noise default
/// every other test in this file uses.
#[allow(clippy::too_many_arguments)]
fn run_filter_with_rho(
    frame: &[f32],
    width: u32,
    height: u32,
    noise_floor: f32,
    spatial_radius: u32,
    k_max: u32,
    sigma: f32,
    lambda_ht: f32,
    rho: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(frame.len(), (width * height) as usize);

    let client = make_client();
    let reference = client.create_from_slice(f32::as_bytes(frame));
    let groups = run_group_raw(
        &client,
        &reference,
        frame.len(),
        width,
        height,
        noise_floor,
        spatial_radius,
        k_max,
    );
    let dummy = client.empty(size_of::<f32>());
    let frame_dummy = client.empty(size_of::<u32>());

    run_filter_raw(
        &client,
        &reference,
        frame.len(),
        &groups,
        width,
        height,
        k_max,
        sigma,
        lambda_ht,
        &dummy,
        1,
        false,
        rho,
        &frame_dummy,
        1,
        false,
        0u32,
    )
}

/// A single-member stack at `sigma = 0`, the `k_use = 1` case.
///
/// `k_max = 1` is what pins the group to one member. The search pins the
/// reference patch itself into slot 0 before selection starts, selection
/// only ever fills slots 1 and above, and the power-of-two rounding then
/// caps the count at `k_max`, so a group can hold nothing else. The
/// assertion on `member_count` below is what proves that, rather than
/// leaving it to be assumed.
///
/// With one member the Haar stack transform is a no-op and `sigma = 0`
/// makes every threshold 0, so the DCT round trip alone must reproduce
/// the reference patch exactly.
#[test]
fn zero_sigma_is_identity() {
    let (w, h) = (32u32, 32u32);
    let mut frame = vec![0.3f32; (w * h) as usize];
    let texture = deterministic_texture(11);
    plant_patch(&mut frame, w, 12, 12, &texture);

    let client = make_client();
    let reference = client.create_from_slice(f32::as_bytes(&frame));
    let groups = run_group_raw(&client, &reference, frame.len(), w, h, 0.0, 6, 1);

    let refs_x = refs_along(w);
    let ref_idx = (3 * refs_x + 3) as usize; // ref_pos(3, 32) == 12 on both axes

    let count_bytes = client
        .read_one(groups.member_count.clone())
        .expect("member_count readback failed");
    let member_count = u32::from_bytes(&count_bytes)[ref_idx];
    assert_eq!(
        member_count, 1,
        "expected k_max = 1 to hold the group at the pinned self-match alone, got {member_count}"
    );

    let dummy = client.empty(size_of::<f32>());
    let frame_dummy = client.empty(size_of::<u32>());
    let (filtered, _weights) = run_filter_raw(
        &client,
        &reference,
        frame.len(),
        &groups,
        w,
        h,
        1,
        0.0,
        2.7,
        &dummy,
        1,
        false,
        0.0,
        &frame_dummy,
        1,
        false,
        0u32,
    );

    let expected = extract_patch(&frame, w, 12, 12);
    let got = &filtered[ref_idx * 64..ref_idx * 64 + 64];

    for (idx, (&want, &have)) in expected.iter().zip(got.iter()).enumerate() {
        assert!((want - have).abs() < 1e-4, "idx={idx}: want {want} got {have}");
    }
}

/// The `k_use = 1` case above never enters `haar_fwd_stack`/
/// `haar_inv_stack`'s `while len > 1` loop, so it can't tell a correct
/// multi-level Haar butterfly from a broken one. This forces a full
/// 8-member stack instead, over a frame where every position differs
/// from every other (`make_unique_frame`), so the 8 members the search
/// keeps out of the 81 a `spatial_radius = 4` window offers carry
/// genuinely different per-position content and their Haar detail
/// coefficients are not all trivially zero. With
/// `sigma = 0` every threshold is still 0, so nothing is discarded and
/// the full forward-and-inverse chain, all 3 Haar levels included, must
/// reproduce member 0's patch exactly.
#[test]
fn zero_sigma_is_identity_for_a_full_stack() {
    let (w, h) = (32u32, 32u32);
    let frame = make_unique_frame(w, h);

    let client = make_client();
    let reference = client.create_from_slice(f32::as_bytes(&frame));
    let groups = run_group_raw(&client, &reference, frame.len(), w, h, 0.0, 4, 8);

    let refs_x = refs_along(w);
    let ref_idx = (4 * refs_x + 4) as usize; // ref_pos(4, 32) == 16 on both axes

    let count_bytes = client
        .read_one(groups.member_count.clone())
        .expect("member_count readback failed");
    let member_count = u32::from_bytes(&count_bytes)[ref_idx];
    assert_eq!(
        member_count, 8,
        "expected the reference at (16, 16) to reach a full 8-member stack under \
         unconditional admission, got {member_count}"
    );

    let dummy = client.empty(size_of::<f32>());
    let frame_dummy = client.empty(size_of::<u32>());
    let (filtered, _weights) = run_filter_raw(
        &client,
        &reference,
        frame.len(),
        &groups,
        w,
        h,
        8,
        0.0,
        2.7,
        &dummy,
        1,
        false,
        0.0,
        &frame_dummy,
        1,
        false,
        0u32,
    );

    let expected = extract_patch(&frame, w, 16, 16);
    let got = &filtered[ref_idx * 64..ref_idx * 64 + 64];

    for (idx, (&want, &have)) in expected.iter().zip(got.iter()).enumerate() {
        assert!((want - have).abs() < 1e-4, "idx={idx}: want {want} got {have}");
    }
}

#[test]
fn noise_is_suppressed_on_a_flat_field() {
    let (w, h) = (48u32, 48u32);
    let sigma = 0.04f32;
    let frame = noisy_field_over(w, h, 0.5, sigma);

    // The channel-scaled expected SSD between two independent noisy
    // copies of the same flat content, which is what a real match on
    // this frame costs. Every position on a flat field is a real match,
    // so the stack fills.
    let floor = 2.0 * 3.0 * sigma * sigma * 64.0;

    let (filtered, _weights) = run_filter(&frame, w, h, floor, 9, 8, sigma, 2.7);

    let input_pool: Vec<f32> = ref_positions(w, h)
        .iter()
        .flat_map(|&(px, py)| extract_patch(&frame, w, px, py))
        .collect();

    let input_var = variance(&input_pool);
    let output_var = variance(&filtered);

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
    let frame = noisy_field_over(w, h, 0.5, sigma);

    let floor = 2.0 * 3.0 * sigma * sigma * 64.0;
    let lambda_ht = 2.7f32;

    let (_filtered, weights) = run_filter(&frame, w, h, floor, 9, 8, sigma, lambda_ht);

    // With every member's variance equal to `sigma^2`, the ladder is a
    // fixed point (see `haar_variance_ladder`'s own `uniform_variance_
    // is_unchanged_by_the_ladder` test), so every coefficient the
    // threshold could possibly keep also carries variance `sigma^2`,
    // whatever level or spatial position it came from. `group_weight`
    // is `1 / (sigma^2 * n_ret)` exactly, not just approximately, so
    // this backs out the mean retained count the run actually produced
    // and checks it against the two things that should be true of it:
    // it must include at least the forced group DC, and a hard
    // threshold at 2.7 standard deviations only lets roughly 0.7% of
    // pure-noise coefficients through by chance (the two-tailed normal
    // tail beyond 2.7), so out of the up to `k_max * PATCH_AREA - 1`
    // coefficients besides the DC that a full 8-member group offers,
    // the mean false-positive count should be small next to that
    // ceiling, not close to it.
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
         ceiling; lambda_ht={lambda_ht}"
    );
}

#[test]
fn hetero_flag_off_never_reads_member_sig2() {
    let (w, h) = (32u32, 32u32);
    let frame = noisy_field_over(w, h, 0.5, 0.05);

    let client = make_client();
    let reference = client.create_from_slice(f32::as_bytes(&frame));
    let groups = run_group_raw(&client, &reference, frame.len(), w, h, 0.0, 6, 8);

    let refs = groups.refs;
    let dummy = client.empty(size_of::<f32>());
    let full_zero = client.create_from_slice(f32::as_bytes(&vec![0.0f32; refs * 8]));
    let frame_dummy = client.empty(size_of::<u32>());

    let (filtered_dummy, weight_dummy) = run_filter_raw(
        &client,
        &reference,
        frame.len(),
        &groups,
        w,
        h,
        8,
        0.05,
        2.7,
        &dummy,
        1,
        false,
        0.0,
        &frame_dummy,
        1,
        false,
        0u32,
    );
    let (filtered_full, weight_full) = run_filter_raw(
        &client,
        &reference,
        frame.len(),
        &groups,
        w,
        h,
        8,
        0.05,
        2.7,
        &full_zero,
        refs * 8,
        false,
        0.0,
        &frame_dummy,
        1,
        false,
        0u32,
    );

    assert_eq!(
        filtered_dummy, filtered_full,
        "filtered output must be identical whether member_sig2 is a 1-element dummy or a \
         full-size zero buffer, since use_member_sigma is false"
    );
    assert_eq!(
        weight_dummy, weight_full,
        "group_weight must be identical whether member_sig2 is a 1-element dummy or a \
         full-size zero buffer, since use_member_sigma is false"
    );
}

#[cube(launch_unchecked)]
fn variance_ladder_kernel(input: &Array<f32>, k_use: u32, output: &mut Array<f32>) {
    let mut v = Array::<f32>::new(8usize);
    #[unroll]
    for k in 0..8u32 {
        v[k as usize] = input[k as usize];
    }
    variance_ladder(&mut v, k_use);
    #[unroll]
    for k in 0..8u32 {
        output[k as usize] = v[k as usize];
    }
}

fn run_variance_ladder(sig2: &[f32; 8], k_use: u32) -> Vec<f32> {
    let client = make_client();
    let input_buf = client.create_from_slice(f32::as_bytes(sig2));
    let output_buf = client.empty(8 * size_of::<f32>());

    let grid = CubeCount::new_single();
    let dim = CubeDim::new_2d(1, 1);

    unsafe {
        variance_ladder_kernel::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            ArrayArg::from_raw_parts(input_buf, 8),
            k_use,
            ArrayArg::from_raw_parts(output_buf.clone(), 8),
        );
    }

    let bytes = client
        .read_one(output_buf)
        .expect("variance ladder readback failed");
    f32::from_bytes(&bytes)[..8].to_vec()
}

/// Pins the GPU ladder against the host mirror `haar_variance_ladder`
/// in `kernels::transforms`, for every valid `k_use`, with non-uniform
/// input variances.
/// Uniform input would leave the ladder at a fixed point regardless of
/// pairing order, so it couldn't catch a level or pairing mismatch
/// between the two implementations, only a non-uniform input can.
#[test]
fn gpu_variance_ladder_matches_the_host_mirror() {
    let sig2 = [0.7f32, 1.3, 0.2, 2.5, 0.05, 3.0, 1.1, 0.4];

    for k_use in [1u32, 2, 4, 8] {
        let host = haar_variance_ladder(&sig2, k_use);
        let gpu = run_variance_ladder(&sig2, k_use);

        for idx in 0..8usize {
            assert!(
                (host[idx] - gpu[idx]).abs() < 1e-5,
                "k_use={k_use} idx={idx}: host {} gpu {}",
                host[idx],
                gpu[idx]
            );
        }
    }
}

/// `rho = 0` must leave the kernel's output bit for bit identical to
/// what it would be with no noise-shaping profile in the computation at
/// all. This is checked two ways from the same noisy multi-member
/// group: once through the real `dct_noise_profile(0.0)` production
/// path, and once through a profile buffer built entirely by hand,
/// `[1.0; 8]`, which is mathematically the exact identity multiplier
/// and so stands in for "no profile logic at all" without needing a
/// second copy of the kernel to prove it against.
#[test]
fn dct_profile_rho_zero_matches_a_hand_built_all_ones_profile() {
    let (w, h) = (48u32, 48u32);
    let sigma = 0.04f32;
    let frame = noisy_field_over(w, h, 0.5, sigma);

    let floor = 2.0 * 3.0 * sigma * sigma * 64.0;

    let client = make_client();
    let reference = client.create_from_slice(f32::as_bytes(&frame));
    let groups = run_group_raw(&client, &reference, frame.len(), w, h, floor, 9, 8);
    let dummy = client.empty(size_of::<f32>());
    let frame_dummy = client.empty(size_of::<u32>());

    assert_eq!(
        dct_noise_profile(0.0),
        [1.0f32; 8],
        "dct_noise_profile(0.0) must be exactly [1.0; 8], the property this test's kernel-level \
         comparison relies on"
    );

    let (filtered_zero_rho, weight_zero_rho) = run_filter_raw(
        &client,
        &reference,
        frame.len(),
        &groups,
        w,
        h,
        8,
        sigma,
        2.7,
        &dummy,
        1,
        false,
        0.0,
        &frame_dummy,
        1,
        false,
        0u32,
    );

    // Bypasses run_filter_raw's own dct_noise_profile(rho) call entirely,
    // building the profile buffer by hand so this half of the comparison
    // does not depend on dct_noise_profile being correct at rho = 0, only
    // on the kernel actually treating an all-ones profile as a no-op.
    let refs_x = refs_along(w);
    let refs = groups.refs;
    let filt_len = filtered_buf_len(w, h, 8);
    let pixels = (w * h) as usize;
    let filtered_buf = client.empty(filt_len * size_of::<f32>());
    let weight_buf = client.empty(refs * size_of::<f32>());
    let sigma_buf = client.create_from_slice(f32::as_bytes(&[sigma]));
    let ones_profile_buf = client.create_from_slice(f32::as_bytes(&[1.0f32; 8]));
    let accum = client.create_from_slice(i32::as_bytes(&vec![0i32; pixels]));
    let wsum = client.create_from_slice(i32::as_bytes(&vec![0i32; pixels]));
    let grid = CubeCount::new_2d(refs_x, refs_along(h));
    let dim = CubeDim::new_2d(8, 8);
    unsafe {
        collab_filter_ht::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            1usize,
            ArrayArg::from_raw_parts(reference.clone(), frame.len()),
            ArrayArg::from_raw_parts(groups.member_pos.clone(), groups.pos_len),
            ArrayArg::from_raw_parts(frame_dummy, 1),
            ArrayArg::from_raw_parts(groups.member_count.clone(), refs),
            ArrayArg::from_raw_parts(dummy, 1),
            ArrayArg::from_raw_parts(accum, pixels),
            ArrayArg::from_raw_parts(wsum, pixels),
            ArrayArg::from_raw_parts(filtered_buf.clone(), filt_len),
            ArrayArg::from_raw_parts(weight_buf.clone(), refs),
            0u32,
            ArrayArg::from_raw_parts(sigma_buf, 1),
            ArrayArg::from_raw_parts(ones_profile_buf, 8),
            2.7f32,
            weight_scale(sigma, &[1.0f32; 8]),
            ACCUM_SCALE,
            false,
            true,
            false,
            false,
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
    // Lifted to member 0 per reference, so it lines up with what
    // `run_filter_raw` returns.
    let mut filtered_hand_built = vec![0.0f32; refs * 64];
    for r in 0..refs {
        let src = r * 8 * 64;
        filtered_hand_built[r * 64..(r + 1) * 64].copy_from_slice(&all_k[src..src + 64]);
    }
    let weight_hand_built =
        f32::from_bytes(&client.read_one(weight_buf).expect("weight readback failed"))[..refs].to_vec();

    assert_eq!(
        filtered_zero_rho, filtered_hand_built,
        "filtered output at rho=0 must be bit-for-bit identical to a hand-built all-ones \
         profile, proving correlation shaping off is exactly a no-op"
    );
    assert_eq!(
        weight_zero_rho, weight_hand_built,
        "group_weight at rho=0 must be bit-for-bit identical to a hand-built all-ones profile"
    );
}

/// Higher `rho` must retain more residual noise on a flat, noise-only
/// field than `rho = 0` does, at the same `lambda_ht`.
///
/// A positive `rho` moves variance out of the high frequencies and into
/// the low ones (`dct_noise_profile`'s own monotonic-decrease property),
/// so a fixed `lambda_ht` reaches a smaller threshold on most
/// non-DC coefficients than the white-noise assumption would, and more
/// of the pure noise sitting in those coefficients survives. This is the
/// documented, deliberate trade the shipped table's caveat describes: on
/// content where the true correlation is lower than the table assumes,
/// shaping under-shrinks rather than over-shrinks, trading a little
/// leftover noise for preserved detail. A flat, noise-only field isolates
/// that trade with nothing else going on.
#[test]
fn higher_rho_retains_more_noise_on_a_flat_field() {
    let (w, h) = (48u32, 48u32);
    let sigma = 0.04f32;
    let frame = noisy_field_over(w, h, 0.5, sigma);

    let floor = 2.0 * 3.0 * sigma * sigma * 64.0;

    let (filtered_rho0, _) = run_filter_with_rho(&frame, w, h, floor, 9, 8, sigma, 2.7, 0.0);
    let (filtered_rho_high, _) = run_filter_with_rho(&frame, w, h, floor, 9, 8, sigma, 2.7, 0.86);

    let var_rho0 = variance(&filtered_rho0);
    let var_rho_high = variance(&filtered_rho_high);

    assert!(
        var_rho_high > var_rho0 * 1.05,
        "expected rho=0.86 to leave meaningfully more residual variance than rho=0 at the same \
         lambda_ht, got rho=0 variance={var_rho0} rho=0.86 variance={var_rho_high}"
    );
}

/// A hand-built two-frame, two-member group, one member in the centre
/// frame and one in a neighbour frame, with `sigma = 0` so the hard
/// threshold keeps every coefficient and the filter is an identity.
///
/// `width = height = 16` gives a `4 x 4` `refs_x/refs_y` grid (`STEP =
/// 4`, `PATCH_SIZE = 8`), so there are 9 references. Only reference 0
/// carries any members, both frames at `rho = 0` planted so its two
/// members' 8x8 patches, at `(0, 0)` and `(8, 8)`, do not overlap. Every
/// other reference gets `member_count = 0`, so it contributes nothing.
///
/// `accum`/`wsum` here hold two frames' worth of pixels back to back,
/// the caller's own cross-frame accumulator ring
/// [`crate::nl4d::Nl4dDenoiser`] builds. With `temporal = true`, each
/// member scatters into its own `member_frame` entry's region of that
/// ring rather than always into `frame`'s, so the centre member lands in
/// region 0 and the neighbour member lands in region 1, distinguishable
/// content planted per frame proving each landed in the right one rather
/// than being conflated or dropped.
#[test]
fn temporal_members_scatter_into_their_own_frames_region() {
    let (w, h) = (16u32, 16u32);
    let pixels = (w * h) as usize;
    let refs_x = refs_along(w);
    let refs = ref_count(w, h);
    let k_max = 2u32;

    let centre_slot = 0u32;
    let neighbour_slot = 1u32;
    let centre_patch = deterministic_texture(21);
    let neighbour_patch = deterministic_texture(22);

    // Two stacked planes, `read_line`'s `frame` index selects between
    // them. Centre-frame content lives at (0, 0), neighbour-frame content
    // at (8, 8), matching where each member's own position points.
    let mut ring = vec![0.4f32; 2 * pixels];
    plant_patch(&mut ring[..pixels], w, 0, 0, &centre_patch);
    plant_patch(&mut ring[pixels..], w, 8, 8, &neighbour_patch);

    let mut member_pos = vec![0u32; refs * k_max as usize];
    let mut member_frame = vec![0u32; refs * k_max as usize];
    let mut member_count = vec![0u32; refs];
    member_pos[0] = pack_pos_host(0, 0);
    member_frame[0] = centre_slot;
    member_pos[1] = pack_pos_host(8, 8);
    member_frame[1] = neighbour_slot;
    member_count[0] = 2;

    let client = make_client();
    let ring_buf = client.create_from_slice(f32::as_bytes(&ring));
    let member_pos_buf = client.create_from_slice(u32::as_bytes(&member_pos));
    let member_frame_buf = client.create_from_slice(u32::as_bytes(&member_frame));
    let member_count_buf = client.create_from_slice(u32::as_bytes(&member_count));
    let member_sig2_dummy = client.empty(size_of::<f32>());
    let filt_len = filtered_buf_len(w, h, k_max);
    let filtered_buf = client.create_from_slice(f32::as_bytes(&vec![0.0f32; filt_len]));
    let weight_buf = client.empty(refs * size_of::<f32>());
    let sigma_buf = client.create_from_slice(f32::as_bytes(&[0.0f32]));
    let profile = dct_noise_profile(0.0);
    let dct_profile_buf = client.create_from_slice(f32::as_bytes(&profile));
    // Two frames' worth of accumulators, region 0 for the centre slot
    // and region 1 for the neighbour slot, the same layout
    // `scatter_patch`'s `frame_slot` addresses.
    let accum = client.create_from_slice(i32::as_bytes(&vec![0i32; 2 * pixels]));
    let wsum = client.create_from_slice(i32::as_bytes(&vec![0i32; 2 * pixels]));

    let grid = CubeCount::new_2d(refs_x, refs_along(h));
    let dim = CubeDim::new_2d(8, 8);
    unsafe {
        collab_filter_ht::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            1usize,
            ArrayArg::from_raw_parts(ring_buf, 2 * pixels),
            ArrayArg::from_raw_parts(member_pos_buf, refs * k_max as usize),
            ArrayArg::from_raw_parts(member_frame_buf, refs * k_max as usize),
            ArrayArg::from_raw_parts(member_count_buf, refs),
            ArrayArg::from_raw_parts(member_sig2_dummy, 1),
            ArrayArg::from_raw_parts(accum, 2 * pixels),
            ArrayArg::from_raw_parts(wsum.clone(), 2 * pixels),
            ArrayArg::from_raw_parts(filtered_buf.clone(), filt_len),
            ArrayArg::from_raw_parts(weight_buf, refs),
            centre_slot,
            ArrayArg::from_raw_parts(sigma_buf, 1),
            ArrayArg::from_raw_parts(dct_profile_buf, 8),
            2.7f32,
            weight_scale(0.0, &profile),
            // Any real denoiser's own `spatial_radius`/`temporal_radius`
            // works here, this test only checks the frame-slot addressing,
            // not the scale's magnitude, so `nl4d`'s defaults stand in.
            cross_frame_accum_scale(9, 2),
            false,
            true,
            false,
            true,
            w,
            h,
            1u32,
            k_max,
            1u32,
            refs_x,
        );
    }

    let filtered = f32::from_bytes(&client.read_one(filtered_buf).expect("filtered readback failed"))
        [..filt_len]
        .to_vec();
    let wsum_out =
        i32::from_bytes(&client.read_one(wsum).expect("wsum readback failed"))[..2 * pixels].to_vec();
    let (centre_region, neighbour_region) = wsum_out.split_at(pixels);

    // Both members were filtered, sigma = 0 makes the threshold a no-op,
    // so each patch reads back exactly as planted.
    let centre_out = &filtered[0..64];
    let neighbour_out = &filtered[64..128];
    for (idx, (&want, &have)) in centre_patch.iter().zip(centre_out.iter()).enumerate() {
        assert!(
            (want - have).abs() < 1e-4,
            "centre member idx={idx}: want {want} got {have}"
        );
    }
    for (idx, (&want, &have)) in neighbour_patch.iter().zip(neighbour_out.iter()).enumerate() {
        assert!(
            (want - have).abs() < 1e-4,
            "neighbour member idx={idx}: want {want} got {have}"
        );
    }

    // The centre member's own 8x8 region, (0,0)-(7,7), was scattered into
    // region 0 (the centre slot), and nowhere in region 1 (the neighbour
    // slot).
    for y in 0..8u32 {
        for x in 0..8u32 {
            let idx = (y * 16 + x) as usize;
            assert!(
                centre_region[idx] > 0,
                "centre region ({x},{y}) in the centre slot's own accumulator: expected wsum > 0, got \
                 {}",
                centre_region[idx]
            );
            assert_eq!(
                neighbour_region[idx], 0,
                "centre region ({x},{y}) leaked into the neighbour slot's accumulator: got {}",
                neighbour_region[idx]
            );
        }
    }
    // The neighbour member's own 8x8 region, (8,8)-(15,15), was
    // scattered into region 1 (the neighbour slot), and nowhere in
    // region 0 (the centre slot), the reverse of the centre member's
    // check above. No other reference reaches into either region, so
    // both sides of both checks hold without any competing contribution
    // to account for.
    for y in 8..16u32 {
        for x in 8..16u32 {
            let idx = (y * 16 + x) as usize;
            assert!(
                neighbour_region[idx] > 0,
                "neighbour region ({x},{y}) in the neighbour slot's own accumulator: expected wsum > \
                 0, got {}",
                neighbour_region[idx]
            );
            assert_eq!(
                centre_region[idx], 0,
                "neighbour region ({x},{y}) leaked into the centre slot's accumulator: got {}",
                centre_region[idx]
            );
        }
    }
}

/// With `temporal = false` the kernel must behave exactly as it did
/// before `member_frame` existed, ignoring the buffer's contents
/// entirely. This is the substitute this task's report describes for
/// the "bit-identical to the pre-change kernel" comparison the plan
/// asks for, which cannot be captured retroactively. Instead, this
/// proves the property that comparison would have proven, that
/// `temporal = false` never reads `member_frame`, by filling it with
/// values that would visibly corrupt the result if they were ever read.
///
/// The same two-member, two-frame setup as
/// [`temporal_members_scatter_into_their_own_frames_region`] runs with
/// `temporal = false`, `frame = centre_slot`, and `member_frame` filled
/// entirely with `neighbour_slot`, the wrong slot for every member. If
/// `temporal = false` read `member_frame` for the scatter guard or the
/// per-member source frame, either the neighbour member's region would
/// stay unscattered (as in the temporal test) or the centre member's
/// patch would read back as the neighbour's content. Neither happens,
/// every member scatters and every member reads from `frame`.
#[test]
fn single_frame_path_is_unchanged_when_temporal_is_false() {
    let (w, h) = (16u32, 16u32);
    let pixels = (w * h) as usize;
    let refs_x = refs_along(w);
    let refs = ref_count(w, h);
    let k_max = 2u32;

    let centre_slot = 0u32;
    let neighbour_slot = 1u32;
    let centre_patch = deterministic_texture(21);
    let neighbour_patch = deterministic_texture(22);

    let mut ring = vec![0.4f32; 2 * pixels];
    plant_patch(&mut ring[..pixels], w, 0, 0, &centre_patch);
    plant_patch(&mut ring[pixels..], w, 8, 8, &neighbour_patch);

    let mut member_pos = vec![0u32; refs * k_max as usize];
    // Every entry lies about its frame. Both members are read from the
    // centre plane at run time (since temporal = false), but member_frame
    // claims the neighbour slot for both, which is wrong for member 0 too.
    let member_frame = vec![neighbour_slot; refs * k_max as usize];
    let mut member_count = vec![0u32; refs];
    member_pos[0] = pack_pos_host(0, 0);
    member_pos[1] = pack_pos_host(8, 8);
    member_count[0] = 2;

    let client = make_client();
    let ring_buf = client.create_from_slice(f32::as_bytes(&ring));
    let member_pos_buf = client.create_from_slice(u32::as_bytes(&member_pos));
    let member_frame_buf = client.create_from_slice(u32::as_bytes(&member_frame));
    let member_count_buf = client.create_from_slice(u32::as_bytes(&member_count));
    let member_sig2_dummy = client.empty(size_of::<f32>());
    let filt_len = filtered_buf_len(w, h, k_max);
    let filtered_buf = client.create_from_slice(f32::as_bytes(&vec![0.0f32; filt_len]));
    let weight_buf = client.empty(refs * size_of::<f32>());
    let sigma_buf = client.create_from_slice(f32::as_bytes(&[0.0f32]));
    let profile = dct_noise_profile(0.0);
    let dct_profile_buf = client.create_from_slice(f32::as_bytes(&profile));
    let accum = client.create_from_slice(i32::as_bytes(&vec![0i32; pixels]));
    let wsum = client.create_from_slice(i32::as_bytes(&vec![0i32; pixels]));

    let grid = CubeCount::new_2d(refs_x, refs_along(h));
    let dim = CubeDim::new_2d(8, 8);
    unsafe {
        collab_filter_ht::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            1usize,
            ArrayArg::from_raw_parts(ring_buf, 2 * pixels),
            ArrayArg::from_raw_parts(member_pos_buf, refs * k_max as usize),
            ArrayArg::from_raw_parts(member_frame_buf, refs * k_max as usize),
            ArrayArg::from_raw_parts(member_count_buf, refs),
            ArrayArg::from_raw_parts(member_sig2_dummy, 1),
            ArrayArg::from_raw_parts(accum, pixels),
            ArrayArg::from_raw_parts(wsum.clone(), pixels),
            ArrayArg::from_raw_parts(filtered_buf.clone(), filt_len),
            ArrayArg::from_raw_parts(weight_buf, refs),
            centre_slot,
            ArrayArg::from_raw_parts(sigma_buf, 1),
            ArrayArg::from_raw_parts(dct_profile_buf, 8),
            2.7f32,
            weight_scale(0.0, &profile),
            ACCUM_SCALE,
            false,
            true,
            false,
            false, // temporal
            w,
            h,
            1u32,
            k_max,
            1u32,
            refs_x,
        );
    }

    let filtered = f32::from_bytes(&client.read_one(filtered_buf).expect("filtered readback failed"))
        [..filt_len]
        .to_vec();
    let wsum_out = i32::from_bytes(&client.read_one(wsum).expect("wsum readback failed"))[..pixels].to_vec();

    // Both members are read from the centre plane (frame = centre_slot),
    // whatever member_frame claims, so member 1's filtered patch matches
    // the centre plane's content at (8, 8), the flat 0.4 background, not
    // the neighbour_patch actually planted at (8, 8) on the neighbour
    // plane.
    let centre_out = &filtered[0..64];
    let member1_out = &filtered[64..128];
    for (idx, (&want, &have)) in centre_patch.iter().zip(centre_out.iter()).enumerate() {
        assert!(
            (want - have).abs() < 1e-4,
            "member 0 idx={idx}: want {want} got {have}"
        );
    }
    for (idx, &have) in member1_out.iter().enumerate() {
        assert!(
            (0.4f32 - have).abs() < 1e-4,
            "member 1 idx={idx}: expected the centre plane's flat background 0.4, got {have}, \
             proving temporal=false read member_frame instead of ignoring it"
        );
    }

    // Every member scatters when temporal = false, whatever member_frame
    // says, so both members' own 8x8 regions receive a contribution.
    // The other two quadrants of this 16x16 frame are covered by
    // neither member's patch, only by other references' zero-count
    // groups, so they are not part of this assertion.
    for y in 0..8u32 {
        for x in 0..8u32 {
            let w = wsum_out[(y * 16 + x) as usize];
            assert!(w > 0, "member 0 region ({x},{y}): expected wsum > 0, got {w}");
        }
    }
    for y in 8..16u32 {
        for x in 8..16u32 {
            let w = wsum_out[(y * 16 + x) as usize];
            assert!(w > 0, "member 1 region ({x},{y}): expected wsum > 0, got {w}");
        }
    }
}

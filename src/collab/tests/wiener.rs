use cubecl::prelude::*;
use cubecl::server::Handle;

use super::helpers::{R, deterministic_texture, make_client, noisy_field_over, plant_patch};
use crate::collab::geometry::{filtered_buf_len, member_buf_len, ref_count, refs_along};
use crate::collab::kernels::filter_ht::collab_filter_ht;
use crate::collab::kernels::filter_wiener::collab_filter_wiener;
use crate::collab::kernels::group::collab_group_spatial;

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
) -> (Vec<f32>, Vec<f32>) {
    let refs_x = refs_along(width);
    let refs_y = refs_along(height);
    let refs = groups.refs;
    let filt_len = filtered_buf_len(width, height);

    let filtered_buf = client.empty(filt_len * size_of::<f32>());
    let weight_buf = client.empty(refs * size_of::<f32>());
    let sigma_buf = client.create_from_slice(f32::as_bytes(&[sigma]));
    let dummy = client.empty(size_of::<f32>());

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
            ArrayArg::from_raw_parts(filtered_buf.clone(), filt_len),
            ArrayArg::from_raw_parts(weight_buf.clone(), refs),
            0u32,
            0u32,
            ArrayArg::from_raw_parts(sigma_buf, 1),
            false,
            width,
            height,
            1u32,
            k_max,
            refs_x,
        );
    }

    let filtered_bytes = client.read_one(filtered_buf).expect("filtered readback failed");
    let weight_bytes = client.read_one(weight_buf).expect("group_weight readback failed");

    (
        f32::from_bytes(&filtered_bytes)[..filt_len].to_vec(),
        f32::from_bytes(&weight_bytes)[..refs].to_vec(),
    )
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
    let filt_len = filtered_buf_len(width, height);

    let filtered_buf = client.empty(filt_len * size_of::<f32>());
    let weight_buf = client.empty(refs * size_of::<f32>());
    let sigma_buf = client.create_from_slice(f32::as_bytes(&[sigma]));
    let dummy = client.empty(size_of::<f32>());

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
            ArrayArg::from_raw_parts(groups.member_count.clone(), refs),
            ArrayArg::from_raw_parts(dummy, 1),
            ArrayArg::from_raw_parts(filtered_buf.clone(), filt_len),
            ArrayArg::from_raw_parts(weight_buf, refs),
            0u32,
            ArrayArg::from_raw_parts(sigma_buf, 1),
            lambda_ht,
            false,
            width,
            height,
            1u32,
            k_max,
            refs_x,
        );
    }

    let filtered_bytes = client.read_one(filtered_buf).expect("filtered readback failed");
    f32::from_bytes(&filtered_bytes)[..filt_len].to_vec()
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

    let (filtered, _weights) = run_wiener_raw(&client, &handle, &handle, frame.len(), &groups, w, h, 8, 0.0);

    let refs_x = refs_along(w);
    let ref_idx = (3 * refs_x + 3) as usize; // ref_pos(3, 32) == 12 on both axes

    let expected = extract_patch(&frame, w, 12, 12);
    let got = &filtered[ref_idx * 64..ref_idx * 64 + 64];

    for (idx, (&want, &have)) in expected.iter().zip(got.iter()).enumerate() {
        assert!((want - have).abs() < 1e-4, "idx={idx}: want {want} got {have}");
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

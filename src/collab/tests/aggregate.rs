use cubecl::prelude::*;

use super::helpers::{R, make_client, noisy_field_over};
use crate::collab::geometry::{
    member_buf_len,
    member_frame_buf_len,
    member_sig2_buf_len,
    ref_count,
    refs_along,
};
use crate::collab::kernels::aggregate::{ACCUM_SCALE, collab_normalise, collab_zero_accum, weight_scale};
use crate::collab::kernels::filter_ht::collab_filter_ht;
use crate::collab::kernels::group_temporal::collab_group_temporal;
use crate::collab::kernels::transforms::dct_noise_profile;
use crate::nlmeans::{BLOCK_X, BLOCK_Y};

/// Runs [`collab_normalise`] over hand-built accumulators.
fn run_normalise(accum_host: &[i32], wsum_host: &[i32], width: u32, height: u32) -> Vec<f32> {
    let pixels = (width * height) as usize;
    assert_eq!(accum_host.len(), pixels);
    assert_eq!(wsum_host.len(), pixels);

    let client = make_client();
    let accum = client.create_from_slice(i32::as_bytes(accum_host));
    let wsum = client.create_from_slice(i32::as_bytes(wsum_host));
    let output = client.empty(pixels * size_of::<f32>());

    unsafe {
        collab_normalise::launch_unchecked::<R>(
            &client,
            CubeCount::new_2d(width.div_ceil(BLOCK_X), height.div_ceil(BLOCK_Y)),
            CubeDim::new_2d(BLOCK_X, BLOCK_Y),
            1usize,
            ArrayArg::from_raw_parts(accum, pixels),
            ArrayArg::from_raw_parts(wsum, pixels),
            ArrayArg::from_raw_parts(output.clone(), pixels),
            0u32,
            width,
            height,
            1u32,
            1u32,
        );
    }

    let bytes = client.read_one(output).expect("normalise readback failed");
    f32::from_bytes(&bytes)[..pixels].to_vec()
}

#[test]
fn normalise_divides_one_accumulator_by_the_other() {
    let (w, h) = (21u32, 16u32);
    let pixels = (w * h) as usize;

    // Varied, non-constant fills, so a transposed index or a dropped
    // pixel changes the answer rather than vanishing into a fixed point.
    let accum: Vec<i32> = (0..pixels).map(|i| (i as i32 % 97) * 1000 - 4000).collect();
    let wsum: Vec<i32> = (0..pixels).map(|i| (i as i32 % 13) + 1).collect();

    let got = run_normalise(&accum, &wsum, w, h);

    for i in 0..pixels {
        let want = accum[i] as f32 / wsum[i] as f32;
        // Relative, because the ratios here run into the thousands and
        // a single-precision divide is only good to about 1e-7 of the
        // value either way.
        assert!(
            (got[i] - want).abs() <= want.abs() * 1e-6,
            "idx={i}: want {want} got {}",
            got[i]
        );
    }
}

/// The fixed-point scale cancels, so a pixel whose accumulator and
/// weight sum were both built at the same scale reads back as the plain
/// ratio with no scale factor left in it.
#[test]
fn normalise_cancels_the_fixed_point_scale() {
    let (w, h) = (16u32, 16u32);
    let pixels = (w * h) as usize;

    let value = 0.375f32;
    let weight = 0.25f32;
    let covering = 7;

    let accum = vec![((value * weight * ACCUM_SCALE) as i32) * covering; pixels];
    let wsum = vec![((weight * ACCUM_SCALE) as i32) * covering; pixels];

    let got = run_normalise(&accum, &wsum, w, h);
    for (i, &v) in got.iter().enumerate() {
        assert!((v - value).abs() < 1e-4, "idx={i}: want {value} got {v}");
    }
}

#[test]
fn a_zero_weight_sum_returns_the_accumulator_rather_than_a_nan() {
    let (w, h) = (16u32, 16u32);
    let pixels = (w * h) as usize;

    let accum = vec![1234i32; pixels];
    let wsum = vec![0i32; pixels];

    let got = run_normalise(&accum, &wsum, w, h);
    for (i, &v) in got.iter().enumerate() {
        assert!(v.is_finite(), "idx={i}: expected a finite value, got {v}");
        assert_eq!(v, 1234.0, "idx={i}");
    }
}

#[test]
fn zero_accum_clears_both_buffers() {
    let (w, h) = (16u32, 16u32);
    let pixels = (w * h) as usize;

    let client = make_client();
    let accum = client.create_from_slice(i32::as_bytes(&vec![42i32; pixels]));
    let wsum = client.create_from_slice(i32::as_bytes(&vec![7i32; pixels]));

    let dim = 256u32;
    let grid = (pixels as u32).div_ceil(dim);
    unsafe {
        collab_zero_accum::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(grid),
            CubeDim::new_1d(dim),
            ArrayArg::from_raw_parts(accum.clone(), pixels),
            ArrayArg::from_raw_parts(wsum.clone(), pixels),
            0u32,
            pixels as u32,
            1u32,
            grid * dim,
        );
    }

    let a = client.read_one(accum).expect("accum readback failed");
    let s = client.read_one(wsum).expect("wsum readback failed");
    assert!(i32::from_bytes(&a)[..pixels].iter().all(|&v| v == 0));
    assert!(i32::from_bytes(&s)[..pixels].iter().all(|&v| v == 0));
}

/// A buffer sized past the GPU's 65,535-workgroups-per-dimension
/// dispatch limit at the 256-thread block size the caller launches
/// this kernel with, and not a multiple of that block size either, so
/// the tail both needs the grid clamp and lands mid-block.
///
/// A one-thread-per-slot launch clamped to that limit stops short of
/// `pixels`, leaving the tail un-zeroed, which is exactly the silent
/// under-zeroing the task's clamp-without-striding trap describes.
/// `collab_zero_accum` is grid-strided so a clamped launch still walks
/// every slot in a second pass, this buffer needs a real 65,626-thread
/// unclamped grid, only 65,535 of which the dispatch actually starts.
#[test]
fn zero_accum_clears_every_slot_of_a_buffer_past_the_grid_clamp() {
    const MAX_GRID_1D: u32 = 65_535;
    let dim = 256u32;
    let pixels = 16_800_005usize;
    assert!(pixels as u32 > MAX_GRID_1D * dim, "test buffer must exceed the clamp point");
    assert_ne!(pixels as u32 % dim, 0, "test buffer must not be a multiple of the block size");

    let client = make_client();
    let accum = client.create_from_slice(i32::as_bytes(&vec![42i32; pixels]));
    let wsum = client.create_from_slice(i32::as_bytes(&vec![7i32; pixels]));

    let grid = (pixels as u32).div_ceil(dim).min(MAX_GRID_1D);
    unsafe {
        collab_zero_accum::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(grid),
            CubeDim::new_1d(dim),
            ArrayArg::from_raw_parts(accum.clone(), pixels),
            ArrayArg::from_raw_parts(wsum.clone(), pixels),
            0u32,
            pixels as u32,
            1u32,
            grid * dim,
        );
    }

    let a = client.read_one(accum).expect("accum readback failed");
    let s = client.read_one(wsum).expect("wsum readback failed");
    let a = i32::from_bytes(&a);
    let s = i32::from_bytes(&s);
    for i in 0..pixels {
        assert_eq!(a[i], 0, "accum[{i}] left un-zeroed past the clamp point");
        assert_eq!(s[i], 0, "wsum[{i}] left un-zeroed past the clamp point");
    }
}

/// Groups, filters, and aggregates a frame end to end, returning the
/// finished plane and the weight sum behind it.
///
/// The grouping search runs at `radius = 0`, a one-frame ring with no
/// neighbours, so this covers the single-frame scatter path the
/// aggregation kernels are being checked on here.
fn run_scatter_stage(frame: &[f32], width: u32, height: u32, sigma: f32) -> (Vec<f32>, Vec<i32>) {
    let client = make_client();
    let refs_x = refs_along(width);
    let refs_y = refs_along(height);
    let refs = ref_count(width, height);
    let k_max = 8u32;
    let pos_len = member_buf_len(width, height, k_max);
    let member_frame_len = member_frame_buf_len(width, height, k_max);
    let sig2_len = member_sig2_buf_len(width, height, k_max);
    let pixels = (width * height) as usize;

    let input = client.create_from_slice(f32::as_bytes(frame));
    let member_pos = client.empty(pos_len * size_of::<u32>());
    let member_frame = client.empty(member_frame_len * size_of::<u32>());
    let member_sig2 = client.empty(sig2_len * size_of::<f32>());
    let mv_dummy = client.create_from_slice(i32::as_bytes(&[0i32, 0i32]));
    let conf_dummy = client.create_from_slice(f32::as_bytes(&[1.0f32]));
    let slots_dummy = client.create_from_slice(u32::as_bytes(&[0u32]));
    let member_frame_dummy = client.empty(size_of::<u32>());
    let member_count = client.empty(refs * size_of::<u32>());
    let accum = client.empty(pixels * size_of::<i32>());
    let wsum = client.empty(pixels * size_of::<i32>());
    let dummy = client.empty(size_of::<f32>());
    let filtered_dummy = client.empty(size_of::<f32>());
    let group_weight = client.empty(refs * size_of::<f32>());
    let sigma_buf = client.create_from_slice(f32::as_bytes(&[sigma]));
    let profile = dct_noise_profile(0.0);
    let profile_buf = client.create_from_slice(f32::as_bytes(&profile));
    let output = client.empty(pixels * size_of::<f32>());

    let floor = 2.0 * 3.0 * sigma * sigma * 64.0;

    let zero_dim = 256u32;
    unsafe {
        collab_group_temporal::launch_unchecked::<R>(
            &client,
            CubeCount::new_2d(refs_x, refs_y),
            CubeDim::new_2d(8, 8),
            1usize,
            ArrayArg::from_raw_parts(input.clone(), pixels),
            ArrayArg::from_raw_parts(mv_dummy, 2),
            ArrayArg::from_raw_parts(conf_dummy, 1),
            ArrayArg::from_raw_parts(member_pos.clone(), pos_len),
            ArrayArg::from_raw_parts(member_frame, member_frame_len),
            ArrayArg::from_raw_parts(member_count.clone(), refs),
            ArrayArg::from_raw_parts(member_sig2, sig2_len),
            0u32,
            ArrayArg::from_raw_parts(slots_dummy, 1),
            floor,
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
            9u32,
            refs_x,
        );
        let zero_grid = (pixels as u32).div_ceil(zero_dim);
        collab_zero_accum::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(zero_grid),
            CubeDim::new_1d(zero_dim),
            ArrayArg::from_raw_parts(accum.clone(), pixels),
            ArrayArg::from_raw_parts(wsum.clone(), pixels),
            0u32,
            pixels as u32,
            1u32,
            zero_grid * zero_dim,
        );
        collab_filter_ht::launch_unchecked::<R>(
            &client,
            CubeCount::new_2d(refs_x, refs_y),
            CubeDim::new_2d(8, 8),
            1usize,
            ArrayArg::from_raw_parts(input.clone(), pixels),
            ArrayArg::from_raw_parts(member_pos.clone(), pos_len),
            ArrayArg::from_raw_parts(member_frame_dummy, 1),
            ArrayArg::from_raw_parts(member_count.clone(), refs),
            ArrayArg::from_raw_parts(dummy, 1),
            ArrayArg::from_raw_parts(accum.clone(), pixels),
            ArrayArg::from_raw_parts(wsum.clone(), pixels),
            ArrayArg::from_raw_parts(filtered_dummy, 1),
            ArrayArg::from_raw_parts(group_weight, refs),
            0u32,
            ArrayArg::from_raw_parts(sigma_buf, 1),
            ArrayArg::from_raw_parts(profile_buf, 8),
            2.7f32,
            weight_scale(sigma, &profile),
            ACCUM_SCALE,
            false,
            false,
            false,
            false,
            width,
            height,
            1u32,
            k_max,
            1u32,
            refs_x,
        );
        collab_normalise::launch_unchecked::<R>(
            &client,
            CubeCount::new_2d(width.div_ceil(BLOCK_X), height.div_ceil(BLOCK_Y)),
            CubeDim::new_2d(BLOCK_X, BLOCK_Y),
            1usize,
            ArrayArg::from_raw_parts(accum, pixels),
            ArrayArg::from_raw_parts(wsum.clone(), pixels),
            ArrayArg::from_raw_parts(output.clone(), pixels),
            0u32,
            width,
            height,
            1u32,
            1u32,
        );
    }

    let out = client.read_one(output).expect("output readback failed");
    let ws = client.read_one(wsum).expect("wsum readback failed");
    (
        f32::from_bytes(&out)[..pixels].to_vec(),
        i32::from_bytes(&ws)[..pixels].to_vec(),
    )
}

/// The sharpest check the scatter has, and it does not depend on the
/// filter doing anything in particular.
///
/// At `sigma = 0` the hard threshold keeps every coefficient, so each
/// member's filtered patch comes back as an exact copy of the input at
/// that member's own position. A member patch sitting at `q` contributes
/// its pixel `q + offset` to output pixel `q + offset`, so every single
/// contribution any pixel receives is that pixel's own input value,
/// whatever group it travelled through. The weighted mean of a set of
/// identical values is that value, so the whole scatter and normalise
/// path has to reproduce the input exactly.
///
/// Any addressing mistake breaks this. A member written to the reference
/// patch's position instead of its own, a transposed `x`/`y`, or an
/// off-by-one in the pixel index all pull in a neighbouring pixel's
/// value and move the result.
#[test]
fn scattering_every_member_at_zero_sigma_reproduces_the_input() {
    let (w, h) = (48u32, 40u32);
    let frame = noisy_field_over(w, h, 0.5, 0.05);

    let (output, _) = run_scatter_stage(&frame, w, h, 0.0);

    for (idx, (&want, &have)) in frame.iter().zip(output.iter()).enumerate() {
        assert!(
            (want - have).abs() < 2e-3,
            "idx={idx}: want {want} got {have}, the scatter moved a pixel"
        );
    }
}

/// Proves the aggregation really covers every member and not just the
/// reference patch of each group.
///
/// Reference patches alone sit on a grid of stride `STEP` and are
/// `PATCH_SIZE` wide, so they can cover any one pixel at most nine
/// times. Members are drawn from a window of radius 9 around their
/// reference, so once every member is written back an interior pixel
/// picks up far more contributions than that ceiling allows.
///
/// The weight sum is read rather than a contribution count, because
/// that is what aggregation actually divides by. Every group here
/// carries the same weight, since a flat noise field gives every group
/// the same retained variance, so the sum is proportional to the number
/// of covering patches.
#[test]
fn every_member_reaches_the_weight_sum_not_only_the_reference_patch() {
    let (w, h) = (64u32, 64u32);
    let frame = noisy_field_over(w, h, 0.5, 0.02);

    let (_, wsum) = run_scatter_stage(&frame, w, h, 0.02);

    // Away from the edges, where the search window is not truncated.
    let mut interior: Vec<i32> = Vec::new();
    for y in 16..h - 16 {
        for x in 16..w - 16 {
            interior.push(wsum[(y * w + x) as usize]);
        }
    }
    assert!(!interior.is_empty());

    let smallest = *interior.iter().min().expect("interior is non-empty");
    assert!(
        smallest > 0,
        "every interior pixel must receive at least one contribution"
    );

    // One group's weight, taken as the largest single contribution any
    // pixel could have received, bounds the count from above. Nine of
    // them is the reference-only ceiling.
    let per_patch = interior.iter().map(|&v| v as f64).fold(f64::INFINITY, f64::min);
    let biggest = *interior.iter().max().expect("interior is non-empty") as f64;
    assert!(
        biggest / per_patch > 9.0,
        "expected some interior pixel to collect more than the nine covering reference \
         patches a member-0-only writeback could manage, got a spread of {}",
        biggest / per_patch,
    );
}

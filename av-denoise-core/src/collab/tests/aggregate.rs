use cubecl::prelude::*;

use super::helpers::{R, make_client, noisy_flat_field};
use crate::collab::geometry::{fused_cubes_x, ref_count, refs_along, strength_map_dims};
use crate::collab::kernels::aggregate::{
    ACCUM_SCALE,
    WEIGHT_GAIN,
    collab_normalise,
    collab_zero_accum,
    kaiser_window,
    weight_scale,
};
use crate::collab::kernels::fused::{STRENGTH_MAP_OFF, collab_fused};
use crate::collab::kernels::transforms::dct_noise_profile;
use crate::collab::{COLLAB_GROUPS, PATCH_SIZE, grid_frames, needs_warp_uniform_search};
use crate::nlmeans::{BLOCK_X, BLOCK_Y, NOISE_CURVE_BINS};

/// Runs [collab_normalise] over hand-built accumulators.
fn run_normalise(accum_host: &[i32], wsum_host: &[i32], width: u32, height: u32) -> Vec<f32> {
    let pixels = (width * height) as usize;
    assert_eq!(accum_host.len(), pixels);
    assert_eq!(wsum_host.len(), pixels);

    let client = make_client();
    let accum_bytes = i32::as_bytes(accum_host);
    let wsum_bytes = i32::as_bytes(wsum_host);
    let accum = client.create_from_slice(accum_bytes);
    let wsum = client.create_from_slice(wsum_bytes);
    let output = client.empty(pixels * size_of::<f32>());

    let blocks_x = width.div_ceil(BLOCK_X);
    let blocks_y = height.div_ceil(BLOCK_Y);
    let grid = CubeCount::new_2d(blocks_x, blocks_y);
    let dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);

    unsafe {
        collab_normalise::launch_unchecked::<R>(
            &client,
            grid,
            dim,
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

    let output_bytes = client.read_one(output).expect("normalise readback failed");

    f32::from_bytes(&output_bytes)[..pixels].to_vec()
}

#[test]
fn normalise_divides_one_accumulator_by_the_other() {
    let (width, height) = (21u32, 16u32);
    let pixels = (width * height) as usize;

    // Varied fills, so a transposed index or a dropped pixel changes the answer rather than vanishing
    // into a fixed point.
    let accum: Vec<i32> = (0..pixels).map(|i| (i as i32 % 97) * 1000 - 4000).collect();
    let wsum: Vec<i32> = (0..pixels).map(|i| (i as i32 % 13) + 1).collect();

    let got = run_normalise(&accum, &wsum, width, height);

    for i in 0..pixels {
        // `wsum` counts at `WEIGHT_GAIN` times `accum`'s scale, the one factor that does not cancel.
        let want = accum[i] as f32 * WEIGHT_GAIN / wsum[i] as f32;
        // Relative, because the ratios run into the thousands and a single-precision divide is only
        // good to about 1e-7 of the value.
        assert!(
            (got[i] - want).abs() <= want.abs() * 1e-6,
            "idx={i}: want {want} got {}",
            got[i]
        );
    }
}

#[test]
fn normalise_cancels_the_fixed_point_scale() {
    let (width, height) = (16u32, 16u32);
    let pixels = (width * height) as usize;

    let value = 0.375f32;
    let weight = 0.25f32;
    let covering = 7;

    let accum = vec![((value * weight * ACCUM_SCALE) as i32) * covering; pixels];
    let wsum = vec![((weight * ACCUM_SCALE * WEIGHT_GAIN) as i32) * covering; pixels];

    let got = run_normalise(&accum, &wsum, width, height);

    for (i, &pixel) in got.iter().enumerate() {
        assert!((pixel - value).abs() < 1e-4, "idx={i}: want {value} got {pixel}");
    }
}

#[test]
fn a_zero_weight_sum_returns_the_accumulator_rather_than_a_nan() {
    let (width, height) = (16u32, 16u32);
    let pixels = (width * height) as usize;

    let accum = vec![1234i32; pixels];
    let wsum = vec![0i32; pixels];

    let got = run_normalise(&accum, &wsum, width, height);

    for (i, &pixel) in got.iter().enumerate() {
        assert!(pixel.is_finite(), "idx={i}: expected a finite value, got {pixel}");
        assert_eq!(pixel, 1234.0, "idx={i}");
    }
}

#[test]
fn zero_accum_clears_both_buffers() {
    let (width, height) = (16u32, 16u32);
    let pixels = (width * height) as usize;

    let client = make_client();
    let filled_accum = vec![42i32; pixels];
    let filled_wsum = vec![7i32; pixels];
    let filled_accum_bytes = i32::as_bytes(&filled_accum);
    let filled_wsum_bytes = i32::as_bytes(&filled_wsum);
    let accum = client.create_from_slice(filled_accum_bytes);
    let wsum = client.create_from_slice(filled_wsum_bytes);

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

    let accum_bytes = client.read_one(accum).expect("accum readback failed");
    let wsum_bytes = client.read_one(wsum).expect("wsum readback failed");
    let accum_cleared = i32::from_bytes(&accum_bytes)[..pixels]
        .iter()
        .all(|&value| value == 0);
    let wsum_cleared = i32::from_bytes(&wsum_bytes)[..pixels]
        .iter()
        .all(|&value| value == 0);
    assert!(accum_cleared);
    assert!(wsum_cleared);
}

/// The buffer needs 65,626 workgroups of 256 threads, past the 65,535 dispatch limit, and is not a
/// multiple of 256, so the tail both needs the grid clamp and lands mid-block.
///
/// A clamped one-thread-per-slot launch would stop short and leave the tail un-zeroed.
/// `collab_zero_accum` is grid-strided, so the clamped launch still walks every slot.
#[test]
fn zero_accum_clears_every_slot_of_a_buffer_past_the_grid_clamp() {
    const MAX_GRID_1D: u32 = 65_535;
    let dim = 256u32;
    let pixels = 16_800_005usize;
    assert!(
        pixels as u32 > MAX_GRID_1D * dim,
        "test buffer must exceed the clamp point"
    );
    assert_ne!(
        pixels as u32 % dim,
        0,
        "test buffer must not be a multiple of the block size"
    );

    let client = make_client();
    let filled_accum = vec![42i32; pixels];
    let filled_wsum = vec![7i32; pixels];
    let filled_accum_bytes = i32::as_bytes(&filled_accum);
    let filled_wsum_bytes = i32::as_bytes(&filled_wsum);
    let accum = client.create_from_slice(filled_accum_bytes);
    let wsum = client.create_from_slice(filled_wsum_bytes);

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

    let accum_bytes = client.read_one(accum).expect("accum readback failed");
    let wsum_bytes = client.read_one(wsum).expect("wsum readback failed");
    let accum_values = i32::from_bytes(&accum_bytes);
    let wsum_values = i32::from_bytes(&wsum_bytes);

    for i in 0..pixels {
        assert_eq!(
            accum_values[i], 0,
            "accum[{i}] left un-zeroed past the clamp point"
        );
        assert_eq!(wsum_values[i], 0, "wsum[{i}] left un-zeroed past the clamp point");
    }
}

/// Groups, filters and aggregates a frame end to end, returning the finished plane and its weight sum.
///
/// The search runs at radius 0, a one-frame ring with no neighbours, so only the single-frame scatter
/// path runs.
fn run_scatter_stage(frame: &[f32], width: u32, height: u32, sigma: f32) -> (Vec<f32>, Vec<i32>) {
    run_scatter_stage_windowed(frame, width, height, sigma, 0.0)
}

/// [run_scatter_stage] with the aggregation window's beta chosen by the caller.
///
/// A beta of `0.0` is a uniform blend.
fn run_scatter_stage_windowed(
    frame: &[f32],
    width: u32,
    height: u32,
    sigma: f32,
    kaiser_beta: f32,
) -> (Vec<f32>, Vec<i32>) {
    let client = make_client();
    let refs_y = refs_along(height);
    let refs = ref_count(width, height);
    let k_max = 8u32;
    let pixels = (width * height) as usize;

    let frame_bytes = f32::as_bytes(frame);
    let mv_dummy_bytes = i32::as_bytes(&[0i32, 0i32]);
    let conf_dummy_bytes = f32::as_bytes(&[1.0f32]);
    let slots_dummy_bytes = u32::as_bytes(&[0u32]);
    let input = client.create_from_slice(frame_bytes);
    let mv_dummy = client.create_from_slice(mv_dummy_bytes);
    let conf_dummy = client.create_from_slice(conf_dummy_bytes);
    let slots_dummy = client.create_from_slice(slots_dummy_bytes);
    let accum = client.empty(pixels * size_of::<i32>());
    let wsum = client.empty(pixels * size_of::<i32>());
    let group_weight = client.empty(refs * size_of::<f32>());

    let sigma_values = [sigma];
    let sigma_bytes = f32::as_bytes(&sigma_values);
    let sigma_buf = client.create_from_slice(sigma_bytes);
    let profile = dct_noise_profile(0.0);
    let profile_bytes = f32::as_bytes(&profile);
    let profile_buf = client.create_from_slice(profile_bytes);
    let kaiser = kaiser_window(kaiser_beta);
    let kaiser_bytes = f32::as_bytes(&kaiser);
    let kaiser_buf = client.create_from_slice(kaiser_bytes);
    let zeroed_curve = [0.0f32; NOISE_CURVE_BINS];
    let curve_bytes = f32::as_bytes(&zeroed_curve);
    let zero_curve = client.create_from_slice(curve_bytes);

    let (map_cols, map_rows) = strength_map_dims(width, height);
    let map_len = (map_cols * map_rows) as usize;
    let unit_map = vec![1.0f32; map_len];
    let unit_map_bytes = f32::as_bytes(&unit_map);
    let unit_map_buf = client.create_from_slice(unit_map_bytes);
    let output = client.empty(pixels * size_of::<f32>());

    let zero_dim = 256u32;
    let zero_grid = (pixels as u32).div_ceil(zero_dim);
    let cubes_x = fused_cubes_x(width);
    let fused_grid = CubeCount::new_2d(cubes_x, refs_y);
    let fused_dim = CubeDim::new_1d(64);
    let scale = weight_scale(sigma, &profile);
    let warp_uniform = needs_warp_uniform_search(&client);
    let grid_frame_count = grid_frames(0);
    let refs_x = refs_along(width);
    let blocks_x = width.div_ceil(BLOCK_X);
    let blocks_y = height.div_ceil(BLOCK_Y);
    let normalise_grid = CubeCount::new_2d(blocks_x, blocks_y);
    let normalise_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);

    let stored_ch = 1usize;

    unsafe {
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
        collab_fused::launch_unchecked::<f32, R>(
            &client,
            fused_grid,
            fused_dim,
            stored_ch,
            ArrayArg::from_raw_parts(input.clone(), pixels),
            ArrayArg::from_raw_parts(input.clone(), stored_ch),
            ArrayArg::from_raw_parts(mv_dummy, 2),
            ArrayArg::from_raw_parts(conf_dummy, 1),
            ArrayArg::from_raw_parts(slots_dummy, 1),
            ArrayArg::from_raw_parts(sigma_buf, 1),
            ArrayArg::from_raw_parts(zero_curve, NOISE_CURVE_BINS),
            ArrayArg::from_raw_parts(unit_map_buf, map_len),
            ArrayArg::from_raw_parts(profile_buf, 8),
            ArrayArg::from_raw_parts(kaiser_buf, PATCH_SIZE as usize),
            ArrayArg::from_raw_parts(accum.clone(), pixels),
            ArrayArg::from_raw_parts(wsum.clone(), pixels),
            ArrayArg::from_raw_parts(group_weight, refs),
            0u32,
            0.0f32,
            2.7f32,
            0u32,
            STRENGTH_MAP_OFF,
            scale,
            ACCUM_SCALE,
            warp_uniform,
            false,
            0u32,
            grid_frame_count,
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
            1u32,
            9u32,
            refs_x,
            map_cols,
            map_rows,
            0.0f32,
            false,
            COLLAB_GROUPS,
        );
        collab_normalise::launch_unchecked::<R>(
            &client,
            normalise_grid,
            normalise_dim,
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

    let output_bytes = client.read_one(output).expect("output readback failed");
    let wsum_bytes = client.read_one(wsum).expect("wsum readback failed");
    let plane = f32::from_bytes(&output_bytes)[..pixels].to_vec();
    let weight_sum = i32::from_bytes(&wsum_bytes)[..pixels].to_vec();

    (plane, weight_sum)
}

/// At `sigma = 0` the hard threshold keeps every coefficient, so each member's filtered patch is an
/// exact copy of the input at that member's own position.
///
/// Every contribution a pixel receives is then its own input value, and the weighted mean of
/// identical values is that value, so the scatter and normalise path must reproduce the input. A
/// member written to the reference's position, a transposed `x`/`y` or an off-by-one pixel index
/// all pull in a neighbouring pixel's value and move the result.
#[test]
fn scattering_every_member_at_zero_sigma_reproduces_the_input() {
    let (width, height) = (48u32, 40u32);
    let frame = noisy_flat_field(width, height, 0.5, 0.05);

    let (output, _) = run_scatter_stage(&frame, width, height, 0.0);

    for (idx, (&want, &have)) in frame.iter().zip(output.iter()).enumerate() {
        assert!(
            (want - have).abs() < 2e-3,
            "idx={idx}: want {want} got {have}, the scatter moved a pixel"
        );
    }
}

/// Reference patches sit on a stride-`STEP` grid and are `PATCH_SIZE` wide, so they cover any pixel
/// at most nine times. Members come from a radius-9 window around their reference, so writing every
/// member back gives an interior pixel far more contributions than that.
///
/// The weight sum is read because aggregation divides by it. A flat noise field gives every group the
/// same retained variance and so the same weight, which makes the sum proportional to the number of
/// covering patches.
#[test]
fn every_member_reaches_the_weight_sum_not_only_the_reference_patch() {
    let (width, height) = (64u32, 64u32);
    let frame = noisy_flat_field(width, height, 0.5, 0.02);

    let (_, wsum) = run_scatter_stage(&frame, width, height, 0.02);

    // Away from the edges, where the search window is not truncated.
    let mut interior: Vec<i32> = Vec::new();
    for y in 16..height - 16 {
        for x in 16..width - 16 {
            interior.push(wsum[(y * width + x) as usize]);
        }
    }

    assert!(!interior.is_empty());

    let smallest = *interior.iter().min().expect("interior is non-empty");
    assert!(
        smallest > 0,
        "every interior pixel must receive at least one contribution"
    );

    // The smallest interior sum is at least one group's weight, so the spread is a lower bound on the
    // largest contribution count. Nine is the reference-only ceiling.
    let per_patch = interior
        .iter()
        .map(|&weight| weight as f64)
        .fold(f64::INFINITY, f64::min);
    let biggest = *interior.iter().max().expect("interior is non-empty") as f64;
    let spread = biggest / per_patch;
    assert!(
        spread > 9.0,
        "expected some interior pixel to collect more than the nine covering reference \
         patches a member-0-only writeback could manage, got a spread of {spread}",
    );
}

/// A window applied to the value but not the weight would pull pixels toward zero, hardest at the
/// patch edges. Flat content shows that exactly, since the weighted mean of one value is that value.
#[test]
fn the_aggregation_window_leaves_flat_content_flat() {
    let (width, height) = (64u32, 64u32);
    let level = 0.5f32;
    let frame = vec![level; (width * height) as usize];

    let (plane, wsum) = run_scatter_stage_windowed(&frame, width, height, 0.02, 2.0);

    for (idx, (&got, &weight)) in plane.iter().zip(wsum.iter()).enumerate() {
        assert!(weight > 0, "pixel {idx} collected no weight at all");
        assert!(
            (got - level).abs() < 1e-3,
            "pixel {idx} came back at {got}, not the {level} every patch carried",
        );
    }
}

/// The window enters only at the scatter, so the two runs differ by the taper alone.
#[test]
fn the_aggregation_window_reweights_the_blend() {
    let (width, height) = (64u32, 64u32);
    let frame = noisy_flat_field(width, height, 0.5, 0.05);

    let (uniform, _) = run_scatter_stage_windowed(&frame, width, height, 0.02, 0.0);
    let (windowed, _) = run_scatter_stage_windowed(&frame, width, height, 0.02, 2.0);

    let moved = uniform
        .iter()
        .zip(windowed.iter())
        .filter(|(uniform_value, windowed_value)| (*uniform_value - *windowed_value).abs() > 1e-4)
        .count();
    assert!(
        moved > uniform.len() / 100,
        "expected the taper to move a real share of the plane, only {moved} of {} pixels \
         moved at all",
        uniform.len(),
    );
}

use cubecl::prelude::*;

use super::helpers::{R, make_client};
use crate::collab::kernels::fused::grid::{
    grid_fwd,
    grid_fwd_host,
    grid_inv,
    grid_inv_host,
    grid_variance,
    grid_variance_host,
};
use crate::collab::{MAX_K, MAX_TEMPORAL_RADIUS, PATCH_AREA, PATCH_SIZE, grid_frames};

#[cube(launch_unchecked)]
fn grid_probe(
    input: &Array<f32>,
    output: &mut Array<f32>,
    #[comptime] grid_frames: u32,
    #[comptime] inverse: bool,
) {
    let mut stack = Array::<f32>::new(PATCH_AREA as usize);
    #[unroll]
    for i in 0..PATCH_AREA {
        stack[i as usize] = input[i as usize];
    }

    if inverse {
        grid_inv(&mut stack, grid_frames);
    } else {
        grid_fwd(&mut stack, grid_frames);
    }

    #[unroll]
    for i in 0..PATCH_AREA {
        output[i as usize] = stack[i as usize];
    }
}

#[cube(launch_unchecked)]
fn grid_variance_probe(input: &Array<f32>, output: &mut Array<f32>, #[comptime] grid_frames: u32) {
    let mut variances = Array::<f32>::new(MAX_K as usize);
    #[unroll]
    for m in 0..MAX_K {
        variances[m as usize] = input[m as usize];
    }

    grid_variance(&mut variances, grid_frames);

    #[unroll]
    for m in 0..MAX_K {
        output[m as usize] = variances[m as usize];
    }
}

fn run_grid(stack: &[f32], grid_frames: u32, inverse: bool) -> Vec<f32> {
    let client = make_client();
    let stack_bytes = f32::as_bytes(stack);
    let input = client.create_from_slice(stack_bytes);
    let stack_size = size_of_val(stack);
    let output = client.empty(stack_size);

    unsafe {
        grid_probe::launch_unchecked::<R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(input, stack.len()),
            ArrayArg::from_raw_parts(output.clone(), stack.len()),
            grid_frames,
            inverse,
        );
    }

    let output_bytes = client.read_one(output).expect("grid readback failed");

    f32::from_bytes(&output_bytes)[..stack.len()].to_vec()
}

fn run_grid_variance(variances: &[f32; 8], grid_frames: u32) -> Vec<f32> {
    let client = make_client();
    let variance_bytes = f32::as_bytes(variances);
    let input = client.create_from_slice(variance_bytes);
    let output = client.empty(8 * size_of::<f32>());

    unsafe {
        grid_variance_probe::launch_unchecked::<R>(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(input, 8),
            ArrayArg::from_raw_parts(output.clone(), 8),
            grid_frames,
        );
    }

    let output_bytes = client.read_one(output).expect("grid variance readback failed");

    f32::from_bytes(&output_bytes)[..8].to_vec()
}

/// A stack of distinct, non-symmetric values so a wrong pairing or level order shows.
fn distinct_stack() -> Vec<f32> {
    (0..PATCH_AREA)
        .map(|i| ((i * 37 + 11) % 64) as f32 / 64.0 - 0.3)
        .collect()
}

/// Every member's value at spatial position `pos`, gathered into one column.
fn column(stack: &[f32], pos: u32) -> [f32; 8] {
    let mut values = [0.0f32; 8];
    for (m, value) in values.iter_mut().enumerate() {
        *value = stack[m * PATCH_SIZE as usize + pos as usize];
    }

    values
}

#[test]
fn gpu_grid_forward_matches_the_host_mirror() {
    let stack = distinct_stack();

    for grid_frames in [2u32, 4] {
        let gpu = run_grid(&stack, grid_frames, false);

        for pos in 0..PATCH_SIZE {
            let input_column = column(&stack, pos);
            let host = grid_fwd_host(&input_column, grid_frames);
            let device = column(&gpu, pos);

            for m in 0..8 {
                assert!(
                    (host[m] - device[m]).abs() < 1e-5,
                    "T={grid_frames} pos={pos} m={m}: host {} gpu {}",
                    host[m],
                    device[m]
                );
            }
        }
    }
}

#[test]
fn grid_inverse_undoes_the_forward_on_the_gpu() {
    let stack = distinct_stack();

    for grid_frames in [2u32, 4] {
        let forward = run_grid(&stack, grid_frames, false);
        let round_trip = run_grid(&forward, grid_frames, true);

        for (i, (&want, &have)) in stack.iter().zip(round_trip.iter()).enumerate() {
            assert!(
                (want - have).abs() < 1e-5,
                "T={grid_frames} index {i}: want {want} got {have}"
            );
        }
    }
}

#[test]
fn host_grid_inverse_undoes_the_host_forward() {
    let input = [0.7f32, -1.3, 0.2, 2.5, 0.05, -3.0, 1.1, 0.4];

    for grid_frames in [2u32, 4] {
        let forward = grid_fwd_host(&input, grid_frames);
        let round_trip = grid_inv_host(&forward, grid_frames);

        for m in 0..8 {
            assert!((input[m] - round_trip[m]).abs() < 1e-5, "T={grid_frames} m={m}");
        }
    }
}

#[test]
fn gpu_grid_variance_matches_the_host_mirror() {
    let sig2 = [0.7f32, 1.3, 0.2, 2.5, 0.05, 3.0, 1.1, 0.4];

    for grid_frames in [2u32, 4] {
        let host = grid_variance_host(&sig2, grid_frames);
        let gpu = run_grid_variance(&sig2, grid_frames);

        for m in 0..8 {
            assert!(
                (host[m] - gpu[m]).abs() < 1e-5,
                "T={grid_frames} m={m}: host {} gpu {}",
                host[m],
                gpu[m]
            );
        }
    }
}

/// The DC of a Haar grid averages every member, so its variance is the members' mean.
#[test]
fn grid_dc_variance_is_the_mean_member_variance() {
    let sig2 = [0.7f32, 1.3, 0.2, 2.5, 0.05, 3.0, 1.1, 0.4];
    let mean = sig2.iter().sum::<f32>() / 8.0;

    for grid_frames in [2u32, 4] {
        let propagated = grid_variance_host(&sig2, grid_frames);
        assert!((propagated[0] - mean).abs() < 1e-6, "T={grid_frames}");
    }
}

/// Coefficient `s * T + t` with `t > 0` is temporal detail after both passes.
#[test]
fn a_volume_constant_in_time_has_no_temporal_detail() {
    for grid_frames in [2u32, 4] {
        let volumes = 8 / grid_frames;
        let mut input = [0.0f32; 8];
        for volume in 0..volumes {
            for frame in 0..grid_frames {
                input[(volume * grid_frames + frame) as usize] = 0.3 + volume as f32 * 0.25;
            }
        }

        let output = grid_fwd_host(&input, grid_frames);

        for volume in 0..volumes {
            for frame in 1..grid_frames {
                let index = (volume * grid_frames + frame) as usize;
                assert!(
                    output[index].abs() < 1e-6,
                    "T={grid_frames} coefficient {index} should be zero, got {}",
                    output[index]
                );
            }
        }
    }
}

#[test]
fn grid_frames_follows_the_temporal_radius() {
    assert_eq!(grid_frames(0), 1);
    assert_eq!(grid_frames(1), 2);
    assert_eq!(grid_frames(2), 4);
    assert_eq!(grid_frames(MAX_TEMPORAL_RADIUS), 4);
}

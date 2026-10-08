use cubecl::prelude::*;

use super::helpers::*;
use super::noise_reference::{
    reference_noise_partial,
    reference_noise_reduce,
    reference_temporal_noise_stats,
};
use crate::nlmeans::kernels::{nlm_noise_partial, nlm_noise_reduce, nlm_temporal_noise_stats};
use crate::nlmeans::noise::{TEMPORAL_NOISE_BLOCK, temporal_stats_record_len};
use crate::nlmeans::{BLOCK_1D, BLOCK_X, BLOCK_Y};

#[derive(Clone, Copy)]
enum Kernel {
    Reference,
    Production,
}

fn next_random(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as u32
}

/// `frames` frames of uniform noise in `stored_ch` lanes, with lanes past `channels` left at zero.
fn noisy_ring(
    width: u32,
    height: u32,
    channels: u32,
    stored_ch: u32,
    frames: u32,
    state: &mut u64,
) -> Vec<f32> {
    let pixels = (width * height * frames) as usize;
    let mut ring = vec![0.0f32; pixels * stored_ch as usize];
    for pixel in 0..pixels {
        for channel in 0..channels as usize {
            let sample = next_random(state) % 4096;
            let offset = 0.0137 * (channel as f32 + 1.0);
            ring[pixel * stored_ch as usize + channel] = sample as f32 / 4095.0 + offset;
        }
    }

    ring
}

fn to_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Runs the spatial pair and returns the bit patterns of the partials and the reduced totals.
fn run_spatial_pair(
    kernel: Kernel,
    width: u32,
    height: u32,
    channels: u32,
    stored_ch: u32,
    ring: &[f32],
) -> (Vec<u32>, Vec<u32>) {
    let client = make_client();
    let cubes_x = width.div_ceil(BLOCK_X);
    let cubes_y = height.div_ceil(BLOCK_Y);
    let block_count = cubes_x * cubes_y;
    let partials_len = (block_count * 4) as usize;

    let ring_bytes = f32::as_bytes(ring);
    let input = client.create_from_slice(ring_bytes);
    let partial_fill = vec![f32::NAN; partials_len];
    let partial_fill_bytes = f32::as_bytes(&partial_fill);
    let partials = client.create_from_slice(partial_fill_bytes);
    let result_fill = vec![f32::NAN; 4];
    let result_fill_bytes = f32::as_bytes(&result_fill);
    let results = client.create_from_slice(result_fill_bytes);

    unsafe {
        let input_arg = ArrayArg::from_raw_parts(input.clone(), ring.len());
        let partials_arg = ArrayArg::from_raw_parts(partials.clone(), partials_len);
        let partial_grid = CubeCount::new_2d(cubes_x, cubes_y);
        let partial_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);
        match kernel {
            Kernel::Reference => reference_noise_partial::launch_unchecked::<R>(
                &client,
                partial_grid,
                partial_dim,
                stored_ch as usize,
                input_arg,
                partials_arg,
                0u32,
                width,
                height,
                channels,
                BLOCK_X,
                BLOCK_Y,
            ),
            Kernel::Production => nlm_noise_partial::launch_unchecked::<R>(
                &client,
                partial_grid,
                partial_dim,
                stored_ch as usize,
                input_arg,
                partials_arg,
                0u32,
                width,
                height,
                channels,
                BLOCK_X,
                BLOCK_Y,
            ),
        }

        let reduce_in = ArrayArg::from_raw_parts(partials.clone(), partials_len);
        let reduce_out = ArrayArg::from_raw_parts(results.clone(), 4);
        let reduce_grid = CubeCount::new_1d(1);
        let reduce_dim = CubeDim::new_1d(BLOCK_1D);
        match kernel {
            Kernel::Reference => reference_noise_reduce::launch_unchecked::<R>(
                &client,
                reduce_grid,
                reduce_dim,
                reduce_in,
                reduce_out,
                0u32,
                block_count,
                BLOCK_1D,
            ),
            Kernel::Production => nlm_noise_reduce::launch_unchecked::<R>(
                &client,
                reduce_grid,
                reduce_dim,
                reduce_in,
                reduce_out,
                0u32,
                block_count,
                BLOCK_1D,
            ),
        }
    }

    let partial_bytes = client.read_one(partials).expect("partials readback failed");
    let result_bytes = client.read_one(results).expect("results readback failed");
    let partial_values = f32::from_bytes(&partial_bytes);
    let result_values = f32::from_bytes(&result_bytes);
    let partial_bits = to_bits(&partial_values[..partials_len]);
    let result_bits = to_bits(&result_values[..4]);
    (partial_bits, result_bits)
}

#[test]
fn noise_partial_and_reduce_match_the_reference_bit_for_bit() {
    let (width, height) = (643u32, 197u32);
    let mut state = 5u64;

    for (channels, stored_ch) in [(1u32, 1u32), (2, 2), (3, 4)] {
        let ring = noisy_ring(width, height, channels, stored_ch, 1, &mut state);
        let (expected_partials, expected_totals) =
            run_spatial_pair(Kernel::Reference, width, height, channels, stored_ch, &ring);
        let (got_partials, got_totals) =
            run_spatial_pair(Kernel::Production, width, height, channels, stored_ch, &ring);

        assert_eq!(
            got_partials, expected_partials,
            "partials differ at {channels} channels"
        );
        assert_eq!(
            got_totals, expected_totals,
            "totals differ at {channels} channels"
        );
    }
}

fn run_temporal(
    kernel: Kernel,
    width: u32,
    height: u32,
    stored_ch: u32,
    ring: &[f32],
    luma_fields: bool,
) -> Vec<u32> {
    let client = make_client();
    let block = TEMPORAL_NOISE_BLOCK;
    let blocks_x = width.div_ceil(block);
    let blocks_y = height.div_ceil(block);
    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let stats_len = (blocks_x * blocks_y) as usize * record_len;

    let ring_bytes = f32::as_bytes(ring);
    let input = client.create_from_slice(ring_bytes);
    let stats_fill = vec![f32::NAN; stats_len];
    let stats_fill_bytes = f32::as_bytes(&stats_fill);
    let stats = client.create_from_slice(stats_fill_bytes);

    unsafe {
        let input_arg = ArrayArg::from_raw_parts(input.clone(), ring.len());
        let stats_arg = ArrayArg::from_raw_parts(stats.clone(), stats_len);
        let grid = CubeCount::new_2d(blocks_x, blocks_y);
        let dim = CubeDim::new_2d(block, block);
        match kernel {
            Kernel::Reference => reference_temporal_noise_stats::launch_unchecked::<R>(
                &client,
                grid,
                dim,
                stored_ch as usize,
                input_arg,
                stats_arg,
                1u32,
                0u32,
                width,
                height,
                stored_ch,
                block,
                luma_fields,
            ),
            Kernel::Production => nlm_temporal_noise_stats::launch_unchecked::<R>(
                &client,
                grid,
                dim,
                stored_ch as usize,
                input_arg,
                stats_arg,
                1u32,
                0u32,
                width,
                height,
                stored_ch,
                block,
                luma_fields,
            ),
        }
    }

    let stats_bytes = client.read_one(stats).expect("stats readback failed");
    let stats_values = f32::from_bytes(&stats_bytes);
    to_bits(&stats_values[..stats_len])
}

#[test]
fn temporal_noise_stats_match_the_reference_bit_for_bit() {
    let (width, height) = (70u32, 45u32);
    let mut state = 9u64;

    for (channels, stored_ch) in [(1u32, 1u32), (2, 2), (3, 4)] {
        for luma_fields in [false, true] {
            let ring = noisy_ring(width, height, channels, stored_ch, 2, &mut state);
            let expected = run_temporal(Kernel::Reference, width, height, stored_ch, &ring, luma_fields);
            let got = run_temporal(Kernel::Production, width, height, stored_ch, &ring, luma_fields);

            assert_eq!(
                got, expected,
                "stats differ at {channels} channels, luma_fields {luma_fields}"
            );
        }
    }
}

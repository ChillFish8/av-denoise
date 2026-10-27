use cubecl::prelude::*;

use crate::nlmeans::kernels::{nlm_vertical_weight, nlm_vweight_pair_accumulate};
use crate::nlmeans::{BLOCK_X, BLOCK_Y};
use crate::noise::tests::nlm_noise_buffer;
use crate::noise::{DERIVED_H2_INV_NORM, DERIVED_WORDS, spatial_offset_index, spatial_offset_lut_len};
use crate::tune::gpu::{bind, read_f32};
use crate::tune::tests::fixtures::{hashed_unit, upload};
use crate::tune::tests::runtime::test_client;

const WIDTH: u32 = 40;
const HEIGHT: u32 = 20;
const PATCH_RADIUS: u32 = 2;
const SEARCH_RADIUS: u32 = 3;

/// Distinct from each other and from every table entry, so a swap or a wrong source shows.
const H2_INV_NORM: f32 = 0.6;
const NOISE_OFFSET: f32 = 1.13;

/// The largest relative gap a kernel weight may show against the host weight.
const MAX_RELATIVE: f32 = 1.0e-5;

fn host_welsch(sum: f32, h2_inv_norm: f32, noise_offset: f32) -> f32 {
    let excess = f32::max(sum - noise_offset, 0.0);
    f32::exp(-excess * h2_inv_norm)
}

fn clamp_coord(value: i32, limit: u32) -> usize {
    value.clamp(0, limit as i32 - 1) as usize
}

/// The column sum of `2 * PATCH_RADIUS + 1` row sums the vertical pass reads at `x`, `y`.
fn host_column_sum(row_sums: &[f32], x: i32, y: i32) -> f32 {
    let clamped_x = clamp_coord(x, WIDTH);
    let mut sum = 0.0f32;

    for offset_y in 0..=2 * PATCH_RADIUS as i32 {
        let source_y = y - PATCH_RADIUS as i32 + offset_y;
        let clamped_y = clamp_coord(source_y, HEIGHT);
        sum += row_sums[clamped_y * WIDTH as usize + clamped_x];
    }

    sum
}

fn assert_close(actual: f32, expected: f32, what: &str) {
    let gap = (actual - expected).abs() / expected.abs();
    assert!(gap <= MAX_RELATIVE, "{what}: {actual} against {expected}");
}

fn cube_count() -> CubeCount {
    let cubes_x = WIDTH.div_ceil(BLOCK_X);
    let cubes_y = HEIGHT.div_ceil(BLOCK_Y);

    CubeCount::new_2d(cubes_x, cubes_y)
}

fn hashed_plane(seed: u32) -> Vec<f32> {
    (0..WIDTH * HEIGHT)
        .map(|index| hashed_unit(index, seed))
        .collect()
}

#[test]
fn the_vertical_weight_reads_the_strength_word_and_its_table_entry() {
    let client = test_client();
    let pixels = (WIDTH * HEIGHT) as usize;
    let row_sums = hashed_plane(1);
    let input = upload(&client, &row_sums);
    let output = client.empty(pixels * size_of::<f32>());
    let noise = nlm_noise_buffer(&client, DERIVED_H2_INV_NORM, H2_INV_NORM, NOISE_OFFSET);

    let offsets_len = spatial_offset_lut_len(SEARCH_RADIUS);
    let table: Vec<f32> = (0..offsets_len).map(|index| 0.32 + 0.05 * index as f32).collect();
    let offsets = upload(&client, &table);
    let offset_index = spatial_offset_index(SEARCH_RADIUS, 1, -2);

    unsafe {
        nlm_vertical_weight::launch_unchecked(
            &client,
            cube_count(),
            CubeDim::new_2d(BLOCK_X, BLOCK_Y),
            bind(&input, pixels),
            bind(&output, pixels),
            bind(&noise, DERIVED_WORDS as usize),
            bind(&offsets, offsets_len),
            offset_index,
            WIDTH,
            HEIGHT,
            PATCH_RADIUS,
            BLOCK_X,
            BLOCK_Y,
        );
    }

    let weights = read_f32(&client, &output, pixels).expect("the weights read back");
    let entry = table[offset_index as usize];

    for y in 0..HEIGHT as i32 {
        for x in 0..WIDTH as i32 {
            let sum = host_column_sum(&row_sums, x, y);
            let expected = host_welsch(sum, H2_INV_NORM, entry);
            let actual = weights[y as usize * WIDTH as usize + x as usize];
            let what = format!("pixel {x}, {y}");
            assert_close(actual, expected, &what);
        }
    }
}

#[test]
fn the_paired_vertical_weight_reads_the_strength_and_offset_words() {
    let client = test_client();
    let pixels = (WIDTH * HEIGHT) as usize;
    let (q_x, q_y) = (1i32, -1i32);
    let forward_sums = hashed_plane(2);
    let backward_sums = hashed_plane(3);
    let frame = hashed_plane(4);
    let zeros = vec![0.0f32; pixels];

    let hsum_fwd = upload(&client, &forward_sums);
    let hsum_bwd = upload(&client, &backward_sums);
    let input = upload(&client, &frame);
    let accum = upload(&client, &zeros);
    let weight_sum = upload(&client, &zeros);
    let max_weight = upload(&client, &zeros);
    let confidence = upload(&client, &[1.0f32]);
    let noise = nlm_noise_buffer(&client, DERIVED_H2_INV_NORM, H2_INV_NORM, NOISE_OFFSET);

    unsafe {
        nlm_vweight_pair_accumulate::launch_unchecked(
            &client,
            cube_count(),
            CubeDim::new_2d(BLOCK_X, BLOCK_Y),
            1,
            bind(&hsum_fwd, pixels),
            bind(&hsum_bwd, pixels),
            bind(&input, pixels),
            bind(&accum, pixels),
            bind(&weight_sum, pixels),
            bind(&max_weight, pixels),
            bind(&confidence, 1),
            bind(&confidence, 1),
            false,
            0u32,
            0u32,
            q_x,
            q_y,
            bind(&noise, DERIVED_WORDS as usize),
            WIDTH,
            HEIGHT,
            PATCH_RADIUS,
            BLOCK_X,
            BLOCK_Y,
            1u32,
            1u32,
            1u32,
        );
    }

    let sums = read_f32(&client, &weight_sum, pixels).expect("the weight sums read back");
    let maxima = read_f32(&client, &max_weight, pixels).expect("the maxima read back");

    for y in 0..HEIGHT as i32 {
        for x in 0..WIDTH as i32 {
            let forward_sum = host_column_sum(&forward_sums, x, y);
            let backward_sum = host_column_sum(&backward_sums, x - q_x, y - q_y);
            let forward = host_welsch(forward_sum, H2_INV_NORM, NOISE_OFFSET);
            let backward = host_welsch(backward_sum, H2_INV_NORM, NOISE_OFFSET);
            let pixel = y as usize * WIDTH as usize + x as usize;
            let what = format!("pixel {x}, {y}");

            assert_close(sums[pixel], forward + backward, &what);
            assert_close(maxima[pixel], forward.max(backward), &what);
        }
    }
}

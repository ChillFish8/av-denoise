use cubecl::prelude::*;

use super::pair_pyramid;
use super::reference::{reference_block_match_coarse, reference_block_match_fine};
use crate::motion::kernels::{BLOCK_MATCH_THREADS, mc_block_match_coarse, mc_block_match_fine};
use crate::noise::tests::sad_floor_buffer;
use crate::noise::{DERIVED_SAD_FLOOR, DERIVED_WORDS};
use crate::tune::tests::runtime::{TestRuntime, test_client};

const FINE_STEP: u32 = 8;
const NOISE_FLOOR: f32 = 0.1;
const THSAD: f32 = 5.12;

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

/// A frame of uniform noise, or a flat frame when `flat` is set.
fn test_frame(width: u32, height: u32, flat: bool, state: &mut u64) -> Vec<f32> {
    let pixels = (width * height) as usize;
    if flat {
        return vec![0.25f32; pixels];
    }

    let mut frame = Vec::with_capacity(pixels);
    for _ in 0..pixels {
        let sample = next_random(state) % 1000;
        frame.push(sample as f32 / 1000.0);
    }
    frame
}

/// Seeds between -16 and 16, so many windows leave the frame.
fn test_seeds(blocks: usize, state: &mut u64) -> Vec<i32> {
    let mut seeds = Vec::with_capacity(blocks * 2);
    for _ in 0..blocks * 2 {
        let offset = next_random(state) % 33;
        seeds.push(offset as i32 - 16);
    }
    seeds
}

#[expect(clippy::too_many_arguments, reason = "each argument is one launch parameter")]
fn run_fine(
    kernel: Kernel,
    dim: CubeDim,
    width: u32,
    height: u32,
    blksize: u32,
    search_radius: u32,
    use_seed: u32,
    centre: &[f32],
    neighbour: &[f32],
    seeds: &[i32],
) -> (Vec<i32>, Vec<u32>) {
    let client = test_client();
    let blocks_x = width.div_ceil(FINE_STEP);
    let blocks_y = height.div_ceil(FINE_STEP);
    let level_len = (width * height) as usize;
    let mv_len = (blocks_x * blocks_y * 2) as usize;
    let conf_len = (blocks_x * blocks_y) as usize;

    let seed_bytes = i32::as_bytes(seeds);
    let mv_field = client.create_from_slice(seed_bytes);
    let unwritten = vec![f32::NAN; conf_len];
    let unwritten_bytes = f32::as_bytes(&unwritten);
    let confidence = client.create_from_slice(unwritten_bytes);

    let grid = CubeCount::new_2d(blocks_x, blocks_y);

    unsafe {
        let mv_arg = ArrayArg::from_raw_parts(mv_field.clone(), mv_len);
        let conf_arg = ArrayArg::from_raw_parts(confidence.clone(), conf_len);

        match kernel {
            Kernel::Reference => {
                let centre_bytes = f32::as_bytes(centre);
                let neighbour_bytes = f32::as_bytes(neighbour);
                let centre_buf = client.create_from_slice(centre_bytes);
                let neighbour_buf = client.create_from_slice(neighbour_bytes);

                let centre_arg = ArrayArg::from_raw_parts(centre_buf, level_len);
                let neighbour_arg = ArrayArg::from_raw_parts(neighbour_buf, level_len);

                reference_block_match_fine::launch_unchecked::<TestRuntime>(
                    &client,
                    grid,
                    dim,
                    centre_arg,
                    neighbour_arg,
                    mv_arg,
                    conf_arg,
                    true,
                    NOISE_FLOOR,
                    THSAD,
                    width,
                    height,
                    blksize,
                    FINE_STEP,
                    search_radius,
                    use_seed,
                    blocks_x,
                )
            },
            Kernel::Production => {
                let (pyramid, neighbour_slots) = pair_pyramid(&client, centre, neighbour);

                let pyramid_arg = ArrayArg::from_raw_parts(pyramid, 2 * level_len);
                let neighbour_slots_arg = ArrayArg::from_raw_parts(neighbour_slots, 1);

                let noise = sad_floor_buffer(&client, DERIVED_SAD_FLOOR, NOISE_FLOOR);
                let noise_arg = ArrayArg::from_raw_parts(noise, DERIVED_WORDS as usize);

                mc_block_match_fine::launch_unchecked::<TestRuntime>(
                    &client,
                    grid,
                    dim,
                    pyramid_arg,
                    neighbour_slots_arg,
                    mv_arg,
                    conf_arg,
                    0u32,
                    true,
                    noise_arg,
                    DERIVED_SAD_FLOOR,
                    THSAD,
                    width,
                    height,
                    level_len as u32,
                    blksize,
                    FINE_STEP,
                    search_radius,
                    use_seed,
                    blocks_x,
                    mv_len as u32,
                    conf_len as u32,
                )
            },
        }
    }

    let mv_bytes = client.read_one(mv_field).expect("mv readback failed");
    let conf_bytes = client.read_one(confidence).expect("confidence readback failed");
    let vectors = i32::from_bytes(&mv_bytes)[..mv_len].to_vec();
    let confidence_bits = f32::from_bytes(&conf_bytes)[..conf_len]
        .iter()
        .map(|value| value.to_bits())
        .collect();
    (vectors, confidence_bits)
}

#[test]
fn blocked_fine_matches_the_reference_bit_for_bit_at_the_production_threads() {
    let height = 140u32;
    let reference_dim = CubeDim::new_2d(8, 8);
    let production_dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);
    let mut state = 7u64;

    let mut cases = Vec::new();
    for width in [260u32, 250] {
        for search_radius in [0u32, 1, 2, 4, 8] {
            for use_seed in [0u32, 1] {
                cases.push((width, 16u32, search_radius, use_seed, false));
            }
        }
    }
    cases.push((260, 16, 4, 1, true));
    cases.push((260, 32, 8, 1, false));

    for (width, blksize, search_radius, use_seed, flat) in cases {
        let centre = test_frame(width, height, flat, &mut state);
        let neighbour = test_frame(width, height, flat, &mut state);
        let blocks = (width.div_ceil(FINE_STEP) * height.div_ceil(FINE_STEP)) as usize;
        let seeds = test_seeds(blocks, &mut state);

        let expected = run_fine(
            Kernel::Reference,
            reference_dim,
            width,
            height,
            blksize,
            search_radius,
            use_seed,
            &centre,
            &neighbour,
            &seeds,
        );

        let label =
            format!("width {width} blksize {blksize} radius {search_radius} seed {use_seed} flat {flat}");

        let got = run_fine(
            Kernel::Production,
            production_dim,
            width,
            height,
            blksize,
            search_radius,
            use_seed,
            &centre,
            &neighbour,
            &seeds,
        );

        assert_eq!(
            got.0, expected.0,
            "motion vectors differ at {label} dim {production_dim:?}"
        );
        assert_eq!(
            got.1, expected.1,
            "confidence bits differ at {label} dim {production_dim:?}"
        );
    }
}

#[expect(clippy::too_many_arguments, reason = "each argument is one launch parameter")]
fn run_coarse(
    kernel: Kernel,
    dim: CubeDim,
    level_width: u32,
    level_height: u32,
    blksize: u32,
    step: u32,
    search_radius: u32,
    centre: &[f32],
    neighbour: &[f32],
) -> Vec<i32> {
    let client = test_client();
    let level_scale = 2u32;
    let fine_blocks_x = (level_width * level_scale).div_ceil(FINE_STEP);
    let fine_blocks_y = (level_height * level_scale).div_ceil(FINE_STEP);
    let coarse_blocks_x = level_width.div_ceil(step);
    let coarse_blocks_y = level_height.div_ceil(step);
    let level_len = (level_width * level_height) as usize;
    let mv_len = (fine_blocks_x * fine_blocks_y * 2) as usize;

    let sentinels = vec![i32::MIN; mv_len];
    let sentinel_bytes = i32::as_bytes(&sentinels);
    let mv_field = client.create_from_slice(sentinel_bytes);

    let grid = CubeCount::new_2d(coarse_blocks_x, coarse_blocks_y);

    unsafe {
        let mv_arg = ArrayArg::from_raw_parts(mv_field.clone(), mv_len);

        match kernel {
            Kernel::Reference => {
                let centre_bytes = f32::as_bytes(centre);
                let neighbour_bytes = f32::as_bytes(neighbour);
                let centre_buf = client.create_from_slice(centre_bytes);
                let neighbour_buf = client.create_from_slice(neighbour_bytes);

                let centre_arg = ArrayArg::from_raw_parts(centre_buf, level_len);
                let neighbour_arg = ArrayArg::from_raw_parts(neighbour_buf, level_len);

                reference_block_match_coarse::launch_unchecked::<TestRuntime>(
                    &client,
                    grid,
                    dim,
                    centre_arg,
                    neighbour_arg,
                    mv_arg,
                    level_width,
                    level_height,
                    blksize,
                    step,
                    search_radius,
                    level_scale,
                    fine_blocks_x,
                    fine_blocks_y,
                    FINE_STEP,
                )
            },
            Kernel::Production => {
                let (pyramid, neighbour_slots) = pair_pyramid(&client, centre, neighbour);

                let pyramid_arg = ArrayArg::from_raw_parts(pyramid, 2 * level_len);
                let neighbour_slots_arg = ArrayArg::from_raw_parts(neighbour_slots, 1);

                mc_block_match_coarse::launch_unchecked::<TestRuntime>(
                    &client,
                    grid,
                    dim,
                    pyramid_arg,
                    neighbour_slots_arg,
                    mv_arg,
                    0u32,
                    level_width,
                    level_height,
                    0,
                    level_len as u32,
                    blksize,
                    step,
                    search_radius,
                    level_scale,
                    fine_blocks_x,
                    fine_blocks_y,
                    FINE_STEP,
                    mv_len as u32,
                )
            },
        }
    }

    let mv_bytes = client.read_one(mv_field).expect("mv readback failed");
    i32::from_bytes(&mv_bytes)[..mv_len].to_vec()
}

#[test]
fn blocked_coarse_matches_the_reference_bit_for_bit_at_the_production_threads() {
    let level_height = 70u32;
    let reference_dim = CubeDim::new_2d(8, 8);
    let production_dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);
    let mut state = 11u64;

    for level_width in [130u32, 125] {
        for (blksize, step) in [(8u32, 4u32), (2, 1)] {
            for search_radius in [1u32, 2, 4, 8] {
                for flat in [false, true] {
                    let centre = test_frame(level_width, level_height, flat, &mut state);
                    let neighbour = test_frame(level_width, level_height, flat, &mut state);

                    let expected = run_coarse(
                        Kernel::Reference,
                        reference_dim,
                        level_width,
                        level_height,
                        blksize,
                        step,
                        search_radius,
                        &centre,
                        &neighbour,
                    );

                    let got = run_coarse(
                        Kernel::Production,
                        production_dim,
                        level_width,
                        level_height,
                        blksize,
                        step,
                        search_radius,
                        &centre,
                        &neighbour,
                    );

                    assert_eq!(
                        got, expected,
                        "coarse vectors differ at width {level_width} blksize {blksize} step \
                         {step} radius {search_radius} flat {flat} dim {production_dim:?}"
                    );
                }
            }
        }
    }
}

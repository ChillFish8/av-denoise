use cubecl::prelude::*;

use crate::nlmeans::kernels::motion::{nlm_mc_block_match_coarse, nlm_mc_block_match_fine};
use crate::tune::gpu::{read_i32, zero_words};
use crate::tune::tests::fixtures::{hashed_unit, upload};
use crate::tune::tests::runtime::{TestRuntime, test_client};

/// Cube sizes the argmin runs at. 16 threads leave a 32-wide plane partly empty, 32 is the shipped
/// size, and 128 spans two to eight planes depending on the device.
const CUBE_THREADS: [u32; 3] = [16, 32, 128];

const WIDTH: u32 = 70;
const HEIGHT: u32 = 45;
const BLKSIZE: u32 = 16;
const STEP: u32 = 8;

/// Search radii with fewer and more candidates than the widest cube has threads.
const SEARCH_RADII: [u32; 2] = [4, 7];

/// The content both frames of a case hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Content {
    /// Every pixel the same, so every SAD ties at zero.
    Flat,
    /// A 4x4 tile repeated, with the neighbour shifted by `(2, 1)`, so several offsets share the
    /// lowest SAD and the zero offset is not among them.
    Periodic,
    /// Distinct values, with the neighbour shifted by `(3, -2)` and lightly perturbed.
    Textured,
}

fn frames(content: Content) -> (Vec<f32>, Vec<f32>) {
    let pixels = WIDTH * HEIGHT;

    match content {
        Content::Flat => {
            let flat = vec![0.25f32; pixels as usize];
            (flat.clone(), flat)
        },
        Content::Periodic => {
            let tile: Vec<f32> = (0..16).map(|index| hashed_unit(index, 21)).collect();
            let sample = |x: i32, y: i32| {
                let tile_x = x.rem_euclid(4) as usize;
                let tile_y = y.rem_euclid(4) as usize;
                tile[tile_y * 4 + tile_x]
            };
            let centre: Vec<f32> = (0..pixels)
                .map(|index| {
                    let x = (index % WIDTH) as i32;
                    let y = (index / WIDTH) as i32;
                    sample(x, y)
                })
                .collect();
            let neighbour: Vec<f32> = (0..pixels)
                .map(|index| {
                    let x = (index % WIDTH) as i32;
                    let y = (index / WIDTH) as i32;
                    sample(x - 2, y - 1)
                })
                .collect();
            (centre, neighbour)
        },
        Content::Textured => {
            let centre: Vec<f32> = (0..pixels).map(|index| hashed_unit(index, 22)).collect();
            let neighbour: Vec<f32> = (0..pixels)
                .map(|index| {
                    let x = (index % WIDTH) as i32;
                    let y = (index / WIDTH) as i32;
                    let source_x = (x - 3).clamp(0, WIDTH as i32 - 1) as u32;
                    let source_y = (y + 2).clamp(0, HEIGHT as i32 - 1) as u32;
                    let source = centre[(source_y * WIDTH + source_x) as usize];
                    let perturbation = (hashed_unit(index, 23) - 0.5) * 0.02;
                    source + perturbation
                })
                .collect();
            (centre, neighbour)
        },
    }
}

fn blocks_x() -> u32 {
    WIDTH.div_ceil(STEP)
}

fn blocks_y() -> u32 {
    HEIGHT.div_ceil(STEP)
}

/// Seeds between -6 and 6 that vary by block.
fn seeds() -> Vec<i32> {
    let blocks = blocks_x() * blocks_y();
    (0..2 * blocks).map(|index| (index % 13) as i32 - 6).collect()
}

/// Clamps a coordinate into the frame, as the kernels clamp their reads.
fn clamp_coord(value: i32, limit: u32) -> u32 {
    value.clamp(0, limit as i32 - 1) as u32
}

/// Every candidate's SAD for one block, row-major over the window.
///
/// Each SAD sums its pixels one block row at a time, the order the kernels use, so every sum rounds
/// the same way and matches the GPU bit for bit.
fn block_sads(
    centre: &[f32],
    neighbour: &[f32],
    origin: (i32, i32),
    window_origin: (i32, i32),
    search_radius: u32,
) -> Vec<f32> {
    let side = 2 * search_radius + 1;
    let mut sads = Vec::with_capacity((side * side) as usize);

    for dy in 0..side {
        for dx in 0..side {
            let mut sad = 0.0f32;

            for iy in 0..BLKSIZE {
                for ix in 0..BLKSIZE {
                    let centre_x = clamp_coord(origin.0 + ix as i32, WIDTH);
                    let centre_y = clamp_coord(origin.1 + iy as i32, HEIGHT);
                    let neighbour_x = clamp_coord(window_origin.0 + (dx + ix) as i32, WIDTH);
                    let neighbour_y = clamp_coord(window_origin.1 + (dy + iy) as i32, HEIGHT);
                    let centre_value = centre[(centre_y * WIDTH + centre_x) as usize];
                    let neighbour_value = neighbour[(neighbour_y * WIDTH + neighbour_x) as usize];
                    let difference = centre_value - neighbour_value;
                    sad += difference.abs();
                }
            }

            sads.push(sad);
        }
    }

    sads
}

/// A row-major scan that keeps the first strictly lower SAD, starting from `1.0e30`.
fn serial_argmin(sads: &[f32]) -> Option<usize> {
    let mut best_sad = 1.0e30f32;
    let mut best = None;

    for (index, &sad) in sads.iter().enumerate() {
        if sad < best_sad {
            best_sad = sad;
            best = Some(index);
        }
    }

    best
}

/// The vector the serial scan picks, offset from `seed`, with a tie at the window centre going
/// back to `seed`.
fn expected_vector(sads: &[f32], seed: (i32, i32), search_radius: u32) -> (i32, i32) {
    let side = (2 * search_radius + 1) as usize;
    let radius = search_radius as i32;
    let centre_sad = sads[search_radius as usize * side + search_radius as usize];
    let winner = serial_argmin(sads);
    let mut best = seed;
    let mut best_sad = 1.0e30f32;

    if let Some(index) = winner {
        best_sad = sads[index];
        let offset_x = (index % side) as i32 - radius;
        let offset_y = (index / side) as i32 - radius;
        best = (seed.0 + offset_x, seed.1 + offset_y);
    }

    if centre_sad <= best_sad {
        best = seed;
    }

    best
}

fn run_fine(content: Content, search_radius: u32, threads: u32) -> Vec<i32> {
    let client = test_client();
    let (centre, neighbour) = frames(content);
    let seeds = seeds();
    let centre_buf = upload(&client, &centre);
    let neighbour_buf = upload(&client, &neighbour);
    let seed_bytes = i32::as_bytes(&seeds);
    let mv_field = client.create_from_slice(seed_bytes);
    let confidence = client.empty(size_of::<f32>());
    zero_words(&client, &confidence, 1);

    let pixels = (WIDTH * HEIGHT) as usize;
    let columns = blocks_x();
    let rows = blocks_y();
    let cube_count = CubeCount::new_2d(columns, rows);
    let cube_dim = CubeDim::new_1d(threads);

    // SAFETY: every argument covers the element count it was allocated for.
    let centre_arg = unsafe { ArrayArg::from_raw_parts(centre_buf, pixels) };
    let neighbour_arg = unsafe { ArrayArg::from_raw_parts(neighbour_buf, pixels) };
    let field_arg = unsafe { ArrayArg::from_raw_parts(mv_field.clone(), seeds.len()) };
    let confidence_arg = unsafe { ArrayArg::from_raw_parts(confidence, 1) };

    unsafe {
        nlm_mc_block_match_fine::launch_unchecked::<TestRuntime>(
            &client,
            cube_count,
            cube_dim,
            centre_arg,
            neighbour_arg,
            field_arg,
            confidence_arg,
            false,
            0.0,
            1.0,
            WIDTH,
            HEIGHT,
            BLKSIZE,
            STEP,
            search_radius,
            1u32,
            columns,
        );
    }

    read_i32(&client, &mv_field, seeds.len()).expect("the field reads back")
}

fn expected_fine(content: Content, search_radius: u32) -> Vec<i32> {
    let (centre, neighbour) = frames(content);
    let seeds = seeds();
    let columns = blocks_x();
    let rows = blocks_y();
    let mut field = Vec::with_capacity(seeds.len());

    for block_row in 0..rows {
        for block_col in 0..columns {
            let block = (block_row * columns + block_col) as usize;
            let seed = (seeds[2 * block], seeds[2 * block + 1]);
            let origin = ((block_col * STEP) as i32, (block_row * STEP) as i32);
            let window_origin = (
                origin.0 + seed.0 - search_radius as i32,
                origin.1 + seed.1 - search_radius as i32,
            );
            let sads = block_sads(&centre, &neighbour, origin, window_origin, search_radius);
            let (vector_x, vector_y) = expected_vector(&sads, seed, search_radius);
            field.push(vector_x);
            field.push(vector_y);
        }
    }

    field
}

/// Runs the coarse kernel with the fine grid equal to its own, so each block writes only itself.
fn run_coarse(content: Content, search_radius: u32, threads: u32) -> Vec<i32> {
    let client = test_client();
    let (centre, neighbour) = frames(content);
    let centre_buf = upload(&client, &centre);
    let neighbour_buf = upload(&client, &neighbour);
    let columns = blocks_x();
    let rows = blocks_y();
    let field_len = (2 * columns * rows) as usize;
    let mv_field = client.empty(field_len * size_of::<i32>());
    zero_words(&client, &mv_field, field_len);

    let pixels = (WIDTH * HEIGHT) as usize;
    let cube_count = CubeCount::new_2d(columns, rows);
    let cube_dim = CubeDim::new_1d(threads);

    // SAFETY: every argument covers the element count it was allocated for.
    let centre_arg = unsafe { ArrayArg::from_raw_parts(centre_buf, pixels) };
    let neighbour_arg = unsafe { ArrayArg::from_raw_parts(neighbour_buf, pixels) };
    let field_arg = unsafe { ArrayArg::from_raw_parts(mv_field.clone(), field_len) };

    unsafe {
        nlm_mc_block_match_coarse::launch_unchecked::<TestRuntime>(
            &client,
            cube_count,
            cube_dim,
            centre_arg,
            neighbour_arg,
            field_arg,
            WIDTH,
            HEIGHT,
            BLKSIZE,
            STEP,
            search_radius,
            1,
            columns,
            rows,
            STEP,
        );
    }

    read_i32(&client, &mv_field, field_len).expect("the field reads back")
}

fn expected_coarse(content: Content, search_radius: u32) -> Vec<i32> {
    let (centre, neighbour) = frames(content);
    let mut field = Vec::new();

    for block_row in 0..blocks_y() {
        for block_col in 0..blocks_x() {
            let origin = ((block_col * STEP) as i32, (block_row * STEP) as i32);
            let window_origin = (origin.0 - search_radius as i32, origin.1 - search_radius as i32);
            let sads = block_sads(&centre, &neighbour, origin, window_origin, search_radius);
            let (vector_x, vector_y) = expected_vector(&sads, (0, 0), search_radius);
            field.push(vector_x);
            field.push(vector_y);
        }
    }

    field
}

fn assert_fine_matches_the_serial_scan(content: Content) {
    for search_radius in SEARCH_RADII {
        let expected = expected_fine(content, search_radius);

        for threads in CUBE_THREADS {
            let got = run_fine(content, search_radius, threads);
            assert_eq!(
                got, expected,
                "{content:?} radius {search_radius} on {threads} threads"
            );
        }
    }
}

fn assert_coarse_matches_the_serial_scan(content: Content) {
    for search_radius in SEARCH_RADII {
        let expected = expected_coarse(content, search_radius);

        for threads in CUBE_THREADS {
            let got = run_coarse(content, search_radius, threads);
            assert_eq!(
                got, expected,
                "{content:?} radius {search_radius} on {threads} threads"
            );
        }
    }
}

#[test]
fn periodic_content_has_blocks_with_several_lowest_sads() {
    let (centre, neighbour) = frames(Content::Periodic);
    let origin = (24, 16);
    let window_origin = (20, 12);

    let sads = block_sads(&centre, &neighbour, origin, window_origin, 4);
    let lowest = sads.iter().copied().fold(f32::INFINITY, f32::min);
    let tied = sads.iter().filter(|&&sad| sad == lowest).count();

    assert!(tied > 1, "the periodic case must force ties, found {tied}");
}

#[test]
fn fine_argmin_matches_the_serial_scan_on_flat_frames() {
    assert_fine_matches_the_serial_scan(Content::Flat);
}

#[test]
fn fine_argmin_matches_the_serial_scan_on_periodic_frames() {
    assert_fine_matches_the_serial_scan(Content::Periodic);
}

#[test]
fn fine_argmin_matches_the_serial_scan_on_textured_frames() {
    assert_fine_matches_the_serial_scan(Content::Textured);
}

#[test]
fn coarse_argmin_matches_the_serial_scan_on_flat_frames() {
    assert_coarse_matches_the_serial_scan(Content::Flat);
}

#[test]
fn coarse_argmin_matches_the_serial_scan_on_periodic_frames() {
    assert_coarse_matches_the_serial_scan(Content::Periodic);
}

#[test]
fn coarse_argmin_matches_the_serial_scan_on_textured_frames() {
    assert_coarse_matches_the_serial_scan(Content::Textured);
}

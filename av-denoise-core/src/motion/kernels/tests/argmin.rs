use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::motion::kernels::{mc_block_match_coarse, mc_block_match_fine};
use crate::noise::tests::sad_floor_buffer;
use crate::noise::{DERIVED_SAD_FLOOR, DERIVED_WORDS};
use crate::tune::gpu::{read_i32, zero_words};
use crate::tune::tests::fixtures::{hashed_unit, upload};
use crate::tune::tests::runtime::test_client;

/// Cube sizes the argmin runs at. 16 threads leave a 32-wide plane partly empty, 32 is the shipped
/// size, and 128 spans two to eight planes depending on the device.
const CUBE_THREADS: [u32; 3] = [16, 32, 128];

const WIDTH: u32 = 70;
const HEIGHT: u32 = 45;
const BLKSIZE: u32 = 16;
const STEP: u32 = 8;

/// Search radii with fewer and more candidates than the widest cube has threads.
const SEARCH_RADII: [u32; 2] = [4, 7];

/// Neighbours each argmin launch covers on z, each matching its own frame.
const NEIGHBOURS: usize = 3;

/// The frame slot each neighbour's frame sits in, out of order so a wrong slot lookup shows.
const NEIGHBOUR_SLOTS: [u32; NEIGHBOURS] = [2, 0, 3];

/// The frame slot the centre frame sits in.
const CENTRE_SLOT: u32 = 1;

/// Frame slots in the buffer, the centre's included.
const SLOT_COUNT: usize = NEIGHBOURS + 1;

/// How far each periodic neighbour is shifted. None is a multiple of the 4x4 tile.
const PERIODIC_SHIFTS: [(i32, i32); NEIGHBOURS] = [(2, 1), (1, 3), (3, 2)];

/// How far each textured neighbour is shifted.
const TEXTURED_SHIFTS: [(i32, i32); NEIGHBOURS] = [(3, -2), (-2, 3), (1, 1)];

/// The content the frames of a case hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Content {
    /// Every frame flat at its own level, so every SAD of a block ties.
    Flat,
    /// A 4x4 tile repeated, with each neighbour shifted by its [PERIODIC_SHIFTS] entry, so several
    /// offsets share the lowest SAD and the zero offset is not among them.
    Periodic,
    /// Distinct values, with each neighbour shifted by its [TEXTURED_SHIFTS] entry and lightly
    /// perturbed.
    Textured,
}

/// The centre frame and one frame per neighbour.
fn frames(content: Content) -> (Vec<f32>, Vec<Vec<f32>>) {
    let pixels = WIDTH * HEIGHT;

    match content {
        Content::Flat => {
            let centre = vec![0.25f32; pixels as usize];
            let neighbours = (0..NEIGHBOURS)
                .map(|neighbour| {
                    let level = 0.3 + 0.05 * neighbour as f32;
                    vec![level; pixels as usize]
                })
                .collect();
            (centre, neighbours)
        },
        Content::Periodic => {
            let tile: Vec<f32> = (0..16).map(|index| hashed_unit(index, 21)).collect();
            let sample = |x: i32, y: i32| {
                let tile_x = x.rem_euclid(4) as usize;
                let tile_y = y.rem_euclid(4) as usize;
                tile[tile_y * 4 + tile_x]
            };
            let shifted = |shift: (i32, i32)| -> Vec<f32> {
                (0..pixels)
                    .map(|index| {
                        let x = (index % WIDTH) as i32;
                        let y = (index / WIDTH) as i32;
                        sample(x - shift.0, y - shift.1)
                    })
                    .collect()
            };
            let centre = shifted((0, 0));
            let neighbours = PERIODIC_SHIFTS.into_iter().map(shifted).collect();
            (centre, neighbours)
        },
        Content::Textured => {
            let centre: Vec<f32> = (0..pixels).map(|index| hashed_unit(index, 22)).collect();
            let shifted = |(neighbour, shift): (usize, (i32, i32))| -> Vec<f32> {
                let perturbation_seed = 23 + neighbour as u32;

                (0..pixels)
                    .map(|index| {
                        let x = (index % WIDTH) as i32;
                        let y = (index / WIDTH) as i32;
                        let source_x = (x - shift.0).clamp(0, WIDTH as i32 - 1) as u32;
                        let source_y = (y - shift.1).clamp(0, HEIGHT as i32 - 1) as u32;
                        let source = centre[(source_y * WIDTH + source_x) as usize];
                        let perturbation = (hashed_unit(index, perturbation_seed) - 0.5) * 0.02;
                        source + perturbation
                    })
                    .collect()
            };
            let neighbours = TEXTURED_SHIFTS.into_iter().enumerate().map(shifted).collect();
            (centre, neighbours)
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

/// Lays out `centre` at [CENTRE_SLOT] and each neighbour at its [NEIGHBOUR_SLOTS] entry, with that
/// table uploaded beside the frames.
fn slot_frames(client: &Client, centre: &[f32], neighbours: &[Vec<f32>]) -> (Handle, Handle) {
    let pixels = centre.len();
    let mut frames = vec![0.0f32; pixels * SLOT_COUNT];
    let centre_start = CENTRE_SLOT as usize * pixels;
    frames[centre_start..centre_start + pixels].copy_from_slice(centre);

    for (neighbour, &slot) in neighbours.iter().zip(NEIGHBOUR_SLOTS.iter()) {
        let start = slot as usize * pixels;
        frames[start..start + pixels].copy_from_slice(neighbour);
    }

    let pyramid = upload(client, &frames);
    let table_bytes = u32::as_bytes(&NEIGHBOUR_SLOTS);
    let neighbour_slots = client.create_from_slice(table_bytes);

    (pyramid, neighbour_slots)
}

/// Runs the fine kernel over [NEIGHBOURS] neighbours and returns each neighbour's field.
fn run_fine(content: Content, search_radius: u32, threads: u32) -> Vec<Vec<i32>> {
    let client = test_client();
    let (centre, neighbours) = frames(content);
    let (pyramid, neighbour_slots) = slot_frames(&client, &centre, &neighbours);
    let seeds = seeds();
    let mv_stride = seeds.len();
    let field_values = seeds.repeat(NEIGHBOURS);
    let field_len = field_values.len();
    let seed_bytes = i32::as_bytes(&field_values);
    let mv_field = client.create_from_slice(seed_bytes);
    let confidence = client.empty(size_of::<f32>());
    zero_words(&client, &confidence, 1);

    let pixels = (WIDTH * HEIGHT) as usize;
    let pyramid_len = pixels * SLOT_COUNT;
    let columns = blocks_x();
    let rows = blocks_y();
    let cube_count = CubeCount::new_3d(columns, rows, NEIGHBOURS as u32);
    let cube_dim = CubeDim::new_1d(threads);
    let mv_field_handle = mv_field.clone();

    // SAFETY: every argument covers the element count it was allocated for.
    let pyramid_arg = unsafe { BufferArg::from_raw_parts(pyramid, pyramid_len) };
    let slots_arg = unsafe { BufferArg::from_raw_parts(neighbour_slots, NEIGHBOURS) };
    let field_arg = unsafe { BufferArg::from_raw_parts(mv_field_handle, field_len) };
    let confidence_arg = unsafe { BufferArg::from_raw_parts(confidence, 1) };

    unsafe {
        let noise = sad_floor_buffer(&client, DERIVED_SAD_FLOOR, 0.0);
        let noise_arg = BufferArg::from_raw_parts(noise, DERIVED_WORDS as usize);

        mc_block_match_fine::launch_unchecked(
            &client,
            cube_count,
            cube_dim,
            pyramid_arg,
            slots_arg,
            field_arg,
            confidence_arg,
            CENTRE_SLOT,
            false,
            noise_arg,
            DERIVED_SAD_FLOOR,
            1.0,
            WIDTH,
            HEIGHT,
            pixels as u32,
            BLKSIZE,
            STEP,
            search_radius,
            1u32,
            columns,
            mv_stride as u32,
            1u32,
        );
    }

    let field = read_i32(&client, &mv_field, field_len).expect("the field reads back");
    field.chunks(mv_stride).map(<[i32]>::to_vec).collect()
}

/// Each neighbour's field from the serial scan against that neighbour's own frame.
fn expected_fine(content: Content, search_radius: u32) -> Vec<Vec<i32>> {
    let (centre, neighbours) = frames(content);

    neighbours
        .iter()
        .map(|neighbour| expected_fine_field(&centre, neighbour, search_radius))
        .collect()
}

fn expected_fine_field(centre: &[f32], neighbour: &[f32], search_radius: u32) -> Vec<i32> {
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
            let sads = block_sads(centre, neighbour, origin, window_origin, search_radius);
            let (vector_x, vector_y) = expected_vector(&sads, seed, search_radius);
            field.push(vector_x);
            field.push(vector_y);
        }
    }

    field
}

/// Runs the coarse kernel over [NEIGHBOURS] neighbours and returns each neighbour's field.
///
/// The fine grid equals the coarse one, so each block writes only itself.
fn run_coarse(content: Content, search_radius: u32, threads: u32) -> Vec<Vec<i32>> {
    let client = test_client();
    let (centre, neighbours) = frames(content);
    let (pyramid, neighbour_slots) = slot_frames(&client, &centre, &neighbours);
    let columns = blocks_x();
    let rows = blocks_y();
    let mv_stride = (2 * columns * rows) as usize;
    let field_len = mv_stride * NEIGHBOURS;
    let mv_field = client.empty(field_len * size_of::<i32>());
    zero_words(&client, &mv_field, field_len);

    let pixels = (WIDTH * HEIGHT) as usize;
    let pyramid_len = pixels * SLOT_COUNT;
    let cube_count = CubeCount::new_3d(columns, rows, NEIGHBOURS as u32);
    let cube_dim = CubeDim::new_1d(threads);
    let mv_field_handle = mv_field.clone();

    // SAFETY: every argument covers the element count it was allocated for.
    let pyramid_arg = unsafe { BufferArg::from_raw_parts(pyramid, pyramid_len) };
    let slots_arg = unsafe { BufferArg::from_raw_parts(neighbour_slots, NEIGHBOURS) };
    let field_arg = unsafe { BufferArg::from_raw_parts(mv_field_handle, field_len) };

    unsafe {
        mc_block_match_coarse::launch_unchecked(
            &client,
            cube_count,
            cube_dim,
            pyramid_arg,
            slots_arg,
            field_arg,
            CENTRE_SLOT,
            WIDTH,
            HEIGHT,
            0,
            pixels as u32,
            BLKSIZE,
            STEP,
            search_radius,
            1,
            columns,
            rows,
            STEP,
            mv_stride as u32,
        );
    }

    let field = read_i32(&client, &mv_field, field_len).expect("the field reads back");
    field.chunks(mv_stride).map(<[i32]>::to_vec).collect()
}

/// Each neighbour's field from the serial scan against that neighbour's own frame.
fn expected_coarse(content: Content, search_radius: u32) -> Vec<Vec<i32>> {
    let (centre, neighbours) = frames(content);

    neighbours
        .iter()
        .map(|neighbour| expected_coarse_field(&centre, neighbour, search_radius))
        .collect()
}

fn expected_coarse_field(centre: &[f32], neighbour: &[f32], search_radius: u32) -> Vec<i32> {
    let mut field = Vec::new();

    for block_row in 0..blocks_y() {
        for block_col in 0..blocks_x() {
            let origin = ((block_col * STEP) as i32, (block_row * STEP) as i32);
            let window_origin = (origin.0 - search_radius as i32, origin.1 - search_radius as i32);
            let sads = block_sads(centre, neighbour, origin, window_origin, search_radius);
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
            let fields = run_fine(content, search_radius, threads);

            assert_eq!(fields.len(), expected.len());

            let pairs = fields.iter().zip(expected.iter()).enumerate();
            for (neighbour, (got, want)) in pairs {
                let label =
                    format!("{content:?} radius {search_radius} on {threads} threads, neighbour {neighbour}");
                assert_eq!(got, want, "{label}");
            }
        }
    }
}

fn assert_coarse_matches_the_serial_scan(content: Content) {
    for search_radius in SEARCH_RADII {
        let expected = expected_coarse(content, search_radius);

        for threads in CUBE_THREADS {
            let fields = run_coarse(content, search_radius, threads);

            assert_eq!(fields.len(), expected.len());

            let pairs = fields.iter().zip(expected.iter()).enumerate();
            for (neighbour, (got, want)) in pairs {
                let label =
                    format!("{content:?} radius {search_radius} on {threads} threads, neighbour {neighbour}");
                assert_eq!(got, want, "{label}");
            }
        }
    }
}

#[test]
fn periodic_content_has_blocks_with_several_lowest_sads() {
    let (centre, neighbours) = frames(Content::Periodic);
    let origin = (24, 16);
    let window_origin = (20, 12);

    for (index, neighbour) in neighbours.iter().enumerate() {
        let sads = block_sads(&centre, neighbour, origin, window_origin, 4);
        let lowest = sads.iter().copied().fold(f32::INFINITY, f32::min);
        let tied = sads.iter().filter(|&&sad| sad == lowest).count();

        assert!(tied > 1, "neighbour {index} must force ties, found {tied}");
    }
}

#[test]
fn every_neighbour_expects_a_different_field() {
    for content in [Content::Periodic, Content::Textured] {
        let fine = expected_fine(content, 4);
        let coarse = expected_coarse(content, 4);

        for first in 0..NEIGHBOURS {
            for second in first + 1..NEIGHBOURS {
                assert_ne!(
                    fine[first], fine[second],
                    "{content:?} fine neighbours {first} and {second}"
                );
                assert_ne!(
                    coarse[first], coarse[second],
                    "{content:?} coarse neighbours {first} and {second}"
                );
            }
        }
    }
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

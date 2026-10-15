use cubecl::prelude::*;
use cubecl::server::Handle;

use super::reference::{reference_block_match_coarse, reference_block_match_fine};
use crate::nlmeans::kernels::motion::{
    BLOCK_MATCH_THREADS,
    nlm_mc_block_match_coarse,
    nlm_mc_block_match_fine,
};
use crate::nlmeans::tests::helpers::{R, make_client};

const WIDTH: u32 = 72;
const HEIGHT: u32 = 56;
const FINE_BLKSIZE: u32 = 16;
const FINE_STEP: u32 = 8;
const SEARCH_RADIUS: u32 = 3;
const NOISE_FLOOR: f32 = 0.01;
const THSAD: f32 = 5.12;
/// Values between frame slots, which no launch may read.
const SLOT_PADDING: u32 = 37;
/// A value far outside the frames, so a stray read of the padding changes every SAD.
const POISON: f32 = 1.0e6;
const CENTRE_SLOT: u32 = 2;
const NEIGHBOUR_SLOTS: [u32; 4] = [0, 1, 3, 4];
const SHIFTS: [(i32, i32); 4] = [(2, -1), (-3, 0), (0, 2), (1, 1)];
/// Cube sizes each merged launch runs at.
///
/// 16 threads leave a plane partly empty and 128 span several planes, so the argmin's cross-plane
/// path runs on every z.
const MERGED_THREADS: [u32; 3] = [16, BLOCK_MATCH_THREADS, 128];

fn textured(width: u32, height: u32) -> Vec<f32> {
    (0..width * height)
        .map(|index| {
            let mut hash = index.wrapping_mul(2_654_435_761);
            hash ^= hash >> 15;
            hash = hash.wrapping_mul(0x85EB_CA6B);
            hash ^= hash >> 13;
            0.2 + 0.6 * (hash as f32 / u32::MAX as f32)
        })
        .collect()
}

fn shifted(frame: &[f32], width: u32, height: u32, shift: (i32, i32)) -> Vec<f32> {
    let mut moved = Vec::with_capacity(frame.len());

    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let source_x = (x - shift.0).clamp(0, width as i32 - 1);
            let source_y = (y - shift.1).clamp(0, height as i32 - 1);
            moved.push(frame[(source_y * width as i32 + source_x) as usize]);
        }
    }

    moved
}

/// One level of five frame slots, padded apart, with `slots_before` poisoned values in front.
///
/// It returns the flat buffer, each slot's plane and the slot stride.
fn frame_slots(width: u32, height: u32, slots_before: usize) -> (Vec<f32>, Vec<Vec<f32>>, u32) {
    let centre = textured(width, height);
    let slot_pixels = width * height + SLOT_PADDING;
    let mut frames = vec![POISON; slots_before];
    let mut planes = Vec::new();

    for slot in 0..5u32 {
        let neighbour_index = NEIGHBOUR_SLOTS.iter().position(|&neighbour| neighbour == slot);
        let plane = match neighbour_index {
            Some(index) => shifted(&centre, width, height, SHIFTS[index]),
            None => centre.clone(),
        };

        let slot_end = slots_before + ((slot + 1) * slot_pixels) as usize;
        frames.extend_from_slice(&plane);
        frames.resize(slot_end, POISON);
        planes.push(plane);
    }

    (frames, planes, slot_pixels)
}

fn upload_f32(client: &ComputeClient<R>, values: &[f32]) -> Handle {
    let bytes = f32::as_bytes(values);
    client.create_from_slice(bytes)
}

fn upload_i32(client: &ComputeClient<R>, values: &[i32]) -> Handle {
    let bytes = i32::as_bytes(values);
    client.create_from_slice(bytes)
}

fn upload_u32(client: &ComputeClient<R>, values: &[u32]) -> Handle {
    let bytes = u32::as_bytes(values);
    client.create_from_slice(bytes)
}

fn confidence_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn run_fine_merged_against_reference(use_seed: u32, threads: u32) {
    let client = make_client();
    let (frames, planes, slot_pixels) = frame_slots(WIDTH, HEIGHT, 0);
    let blocks_x = WIDTH.div_ceil(FINE_STEP);
    let blocks_y = HEIGHT.div_ceil(FINE_STEP);
    let blocks = (blocks_x * blocks_y) as usize;
    let mv_stride = 2 * blocks + 6;
    let conf_stride = blocks + 5;
    let neighbours = NEIGHBOUR_SLOTS.len();

    let mut seeds = vec![i32::MIN; neighbours * mv_stride];
    for (index, shift) in SHIFTS.iter().enumerate() {
        for block in 0..blocks {
            seeds[index * mv_stride + 2 * block] = shift.0 - 1;
            seeds[index * mv_stride + 2 * block + 1] = shift.1;
        }
    }

    let pyramid = upload_f32(&client, &frames);
    let table = upload_u32(&client, &NEIGHBOUR_SLOTS);
    let mv_field = upload_i32(&client, &seeds);
    let unwritten = vec![f32::NAN; neighbours * conf_stride];
    let confidence = upload_f32(&client, &unwritten);
    let grid = CubeCount::new_3d(blocks_x, blocks_y, neighbours as u32);
    let dim = CubeDim::new_1d(threads);

    unsafe {
        let pyramid_arg = ArrayArg::from_raw_parts(pyramid, frames.len());
        let table_arg = ArrayArg::from_raw_parts(table, neighbours);
        let mv_field_arg = ArrayArg::from_raw_parts(mv_field.clone(), seeds.len());
        let confidence_arg = ArrayArg::from_raw_parts(confidence.clone(), unwritten.len());

        nlm_mc_block_match_fine::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            pyramid_arg,
            table_arg,
            mv_field_arg,
            confidence_arg,
            CENTRE_SLOT,
            true,
            NOISE_FLOOR,
            THSAD,
            WIDTH,
            HEIGHT,
            slot_pixels,
            FINE_BLKSIZE,
            FINE_STEP,
            SEARCH_RADIUS,
            use_seed,
            blocks_x,
            mv_stride as u32,
            conf_stride as u32,
        );
    }

    let merged_mv_bytes = client.read_one(mv_field).expect("mv reads back");
    let merged_conf_bytes = client.read_one(confidence).expect("confidence reads back");
    let merged_mv = i32::from_bytes(&merged_mv_bytes);
    let merged_conf = f32::from_bytes(&merged_conf_bytes);

    for (index, &slot) in NEIGHBOUR_SLOTS.iter().enumerate() {
        let centre = upload_f32(&client, &planes[CENTRE_SLOT as usize]);
        let neighbour = upload_f32(&client, &planes[slot as usize]);
        let own_seeds = &seeds[index * mv_stride..index * mv_stride + 2 * blocks];
        let reference_mv = upload_i32(&client, own_seeds);
        let reference_unwritten = vec![f32::NAN; blocks];
        let reference_conf = upload_f32(&client, &reference_unwritten);
        let level_len = (WIDTH * HEIGHT) as usize;
        let reference_grid = CubeCount::new_2d(blocks_x, blocks_y);

        unsafe {
            let centre_arg = ArrayArg::from_raw_parts(centre, level_len);
            let neighbour_arg = ArrayArg::from_raw_parts(neighbour, level_len);
            let reference_mv_arg = ArrayArg::from_raw_parts(reference_mv.clone(), 2 * blocks);
            let reference_conf_arg = ArrayArg::from_raw_parts(reference_conf.clone(), blocks);

            reference_block_match_fine::launch_unchecked::<R>(
                &client,
                reference_grid,
                dim,
                centre_arg,
                neighbour_arg,
                reference_mv_arg,
                reference_conf_arg,
                true,
                NOISE_FLOOR,
                THSAD,
                WIDTH,
                HEIGHT,
                FINE_BLKSIZE,
                FINE_STEP,
                SEARCH_RADIUS,
                use_seed,
                blocks_x,
            );
        }

        let expected_mv_bytes = client.read_one(reference_mv).expect("mv reads back");
        let expected_conf_bytes = client.read_one(reference_conf).expect("confidence reads back");
        let expected_mv = &i32::from_bytes(&expected_mv_bytes)[..2 * blocks];
        let expected_conf_values = f32::from_bytes(&expected_conf_bytes);
        let expected_conf = confidence_bits(&expected_conf_values[..blocks]);

        let mv_start = index * mv_stride;
        let conf_start = index * conf_stride;
        let actual_mv = &merged_mv[mv_start..mv_start + 2 * blocks];
        let actual_conf = confidence_bits(&merged_conf[conf_start..conf_start + blocks]);
        let mv_padding = &merged_mv[mv_start + 2 * blocks..mv_start + mv_stride];
        let conf_padding = &merged_conf[conf_start + blocks..conf_start + conf_stride];

        let label = format!("neighbour {index} on {threads} threads");

        assert_eq!(actual_mv, expected_mv, "{label} vectors");
        assert_eq!(actual_conf, expected_conf, "{label} confidence");
        assert!(
            mv_padding.iter().all(|&value| value == i32::MIN),
            "{label} field padding"
        );
        assert!(
            conf_padding.iter().all(|value| value.is_nan()),
            "{label} confidence padding"
        );
    }
}

fn run_coarse_merged_against_reference(threads: u32) {
    let client = make_client();
    let level_width = WIDTH / 2;
    let level_height = HEIGHT / 2;
    let level_offset = 1_000usize;
    let (frames, planes, slot_pixels) = frame_slots(level_width, level_height, level_offset);
    let coarse_blksize = FINE_BLKSIZE / 2;
    let coarse_step = FINE_STEP / 2;
    let level_scale = 2u32;
    let coarse_blocks_x = level_width.div_ceil(coarse_step);
    let coarse_blocks_y = level_height.div_ceil(coarse_step);
    let fine_blocks_x = WIDTH.div_ceil(FINE_STEP);
    let fine_blocks_y = HEIGHT.div_ceil(FINE_STEP);
    let fine_blocks = (fine_blocks_x * fine_blocks_y) as usize;
    let mv_stride = 2 * fine_blocks + 6;
    let neighbours = NEIGHBOUR_SLOTS.len();

    let pyramid = upload_f32(&client, &frames);
    let table = upload_u32(&client, &NEIGHBOUR_SLOTS);
    let unwritten = vec![i32::MIN; neighbours * mv_stride];
    let mv_field = upload_i32(&client, &unwritten);
    let grid = CubeCount::new_3d(coarse_blocks_x, coarse_blocks_y, neighbours as u32);
    let dim = CubeDim::new_1d(threads);

    unsafe {
        let pyramid_arg = ArrayArg::from_raw_parts(pyramid, frames.len());
        let table_arg = ArrayArg::from_raw_parts(table, neighbours);
        let mv_field_arg = ArrayArg::from_raw_parts(mv_field.clone(), unwritten.len());

        nlm_mc_block_match_coarse::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            pyramid_arg,
            table_arg,
            mv_field_arg,
            CENTRE_SLOT,
            level_width,
            level_height,
            level_offset as u32,
            slot_pixels,
            coarse_blksize,
            coarse_step,
            SEARCH_RADIUS,
            level_scale,
            fine_blocks_x,
            fine_blocks_y,
            FINE_STEP,
            mv_stride as u32,
        );
    }

    let merged_bytes = client.read_one(mv_field).expect("mv reads back");
    let merged = i32::from_bytes(&merged_bytes);

    for (index, &slot) in NEIGHBOUR_SLOTS.iter().enumerate() {
        let centre = upload_f32(&client, &planes[CENTRE_SLOT as usize]);
        let neighbour = upload_f32(&client, &planes[slot as usize]);
        let reference_unwritten = vec![i32::MIN; 2 * fine_blocks];
        let reference_mv = upload_i32(&client, &reference_unwritten);
        let level_len = (level_width * level_height) as usize;
        let reference_grid = CubeCount::new_2d(coarse_blocks_x, coarse_blocks_y);

        unsafe {
            let centre_arg = ArrayArg::from_raw_parts(centre, level_len);
            let neighbour_arg = ArrayArg::from_raw_parts(neighbour, level_len);
            let reference_mv_arg = ArrayArg::from_raw_parts(reference_mv.clone(), 2 * fine_blocks);

            reference_block_match_coarse::launch_unchecked::<R>(
                &client,
                reference_grid,
                dim,
                centre_arg,
                neighbour_arg,
                reference_mv_arg,
                level_width,
                level_height,
                coarse_blksize,
                coarse_step,
                SEARCH_RADIUS,
                level_scale,
                fine_blocks_x,
                fine_blocks_y,
                FINE_STEP,
            );
        }

        let expected_bytes = client.read_one(reference_mv).expect("mv reads back");
        let expected = &i32::from_bytes(&expected_bytes)[..2 * fine_blocks];
        let start = index * mv_stride;
        let actual = &merged[start..start + 2 * fine_blocks];
        let padding = &merged[start + 2 * fine_blocks..start + mv_stride];

        let label = format!("neighbour {index} on {threads} threads");

        assert_eq!(actual, expected, "{label} vectors");
        assert!(
            padding.iter().all(|&value| value == i32::MIN),
            "{label} field padding"
        );
    }
}

#[test]
fn a_fine_launch_over_four_neighbours_matches_one_reference_launch_each() {
    for threads in MERGED_THREADS {
        run_fine_merged_against_reference(0, threads);
    }
}

#[test]
fn a_seeded_fine_launch_over_four_neighbours_matches_one_reference_launch_each() {
    for threads in MERGED_THREADS {
        run_fine_merged_against_reference(1, threads);
    }
}

#[test]
fn a_coarse_launch_over_four_neighbours_matches_one_reference_launch_each() {
    for threads in MERGED_THREADS {
        run_coarse_merged_against_reference(threads);
    }
}

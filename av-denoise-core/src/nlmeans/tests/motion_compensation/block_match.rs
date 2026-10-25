use cubecl::prelude::*;

use super::frame_shifted_by;
use crate::motion::kernels::{BLOCK_MATCH_THREADS, mc_block_match_coarse, mc_block_match_fine};
use crate::nlmeans::motion::{DEFAULT_BLKSIZE, DEFAULT_SEARCH_RADIUS};
use crate::nlmeans::tests::helpers::*;
use crate::noise::tests::sad_floor_buffer;
use crate::noise::{DERIVED_SAD_FLOOR, DERIVED_WORDS};

/// Runs the fine block-match kernel over one block covering the whole buffer.
///
/// It returns the winning motion vector and its confidence, with no seed and no pyramid.
fn run_fine_block_match_single_block(
    blksize: u32,
    search_radius: u32,
    centre: &[f32],
    neighbour: &[f32],
    sad_noise_floor: f32,
    thsad: f32,
) -> (i32, i32, f32) {
    let client = make_client();
    let level_len = (blksize * blksize) as usize;
    assert_eq!(centre.len(), level_len);
    assert_eq!(neighbour.len(), level_len);

    let (pyramid, neighbour_slots) = pair_pyramid(&client, centre, neighbour);
    let mv_field = client.empty(2 * size_of::<i32>());
    let confidence = client.empty(size_of::<f32>());

    let grid = CubeCount::new_2d(1, 1);
    let dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);

    unsafe {
        let mv_field_handle = mv_field.clone();
        let confidence_handle = confidence.clone();
        let pyramid_arg = ArrayArg::from_raw_parts(pyramid, 2 * level_len);
        let neighbour_slots_arg = ArrayArg::from_raw_parts(neighbour_slots, 1);
        let mv_field_arg = ArrayArg::from_raw_parts(mv_field_handle, 2);
        let confidence_arg = ArrayArg::from_raw_parts(confidence_handle, 1);

        let noise = sad_floor_buffer(&client, DERIVED_SAD_FLOOR, sad_noise_floor);
        let noise_arg = ArrayArg::from_raw_parts(noise, DERIVED_WORDS as usize);

        mc_block_match_fine::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            pyramid_arg,
            neighbour_slots_arg,
            mv_field_arg,
            confidence_arg,
            0u32,
            true,
            noise_arg,
            DERIVED_SAD_FLOOR,
            thsad,
            blksize,
            blksize,
            level_len as u32,
            blksize,
            blksize,
            search_radius,
            0u32,
            1,
            2,
            1,
        );
    }

    let mv_bytes = client.read_one(mv_field).expect("mv readback failed");
    let mv = i32::from_bytes(&mv_bytes);
    let confidence_bytes = client.read_one(confidence).expect("confidence readback failed");
    let confidence = f32::from_bytes(&confidence_bytes)[0];
    (mv[0], mv[1], confidence)
}

/// Recovers the fine kernel's best SAD by inverting its confidence formula.
///
/// `confidence = (thsad² - S²) / (thsad² + S²)` inverts to `S = thsad · sqrt((1 - confidence) /
/// (1 + confidence))`. It needs `sad_noise_floor = 0.0` and a `thsad` well above the expected SAD,
/// which keeps the confidence clear of cancellation near 1 and of the clamp at 0.
fn recover_sad_from_confidence(confidence: f32, thsad: f32) -> f32 {
    thsad * ((1.0 - confidence) / (1.0 + confidence)).sqrt()
}

/// A racy shared-memory SAD reduction drops contributions and undercounts by orders of magnitude.
#[test]
fn block_match_fine_exact_sad_uniform_mismatch() {
    let blksize = 16u32;
    let mismatch = 0.1f32;
    let centre = vec![0.25f32; (blksize * blksize) as usize];
    let neighbour = vec![0.25f32 + mismatch; (blksize * blksize) as usize];

    let expected_sad = (blksize * blksize) as f32 * mismatch;
    // Well above the expected SAD so the confidence stays clear of both precision corners.
    let thsad = 3.0 * expected_sad;

    let (_, _, confidence) = run_fine_block_match_single_block(blksize, 0, &centre, &neighbour, 0.0, thsad);
    let measured_sad = recover_sad_from_confidence(confidence, thsad);

    assert!(
        (measured_sad - expected_sad).abs() < expected_sad * 0.01,
        "uniform |Δ|={mismatch} over a {blksize}x{blksize} block should give best_sad \
         = {expected_sad} (blksize²·d), measured {measured_sad} (confidence={confidence})",
    );
}

/// The neighbour is the centre shifted by `(+2, +1)`, so the argmin must land exactly there.
///
/// Rich content keeps every other candidate's SAD strictly larger. A 3x3 grid at the library
/// defaults gives the middle block the same clamped addressing the production dispatch uses.
/// Confidence is turned off, which also covers the path where its write is compiled out.
#[test]
fn block_match_fine_argmin_finds_clean_shift() {
    let width = 64u32;
    let height = 64u32;
    let blksize = DEFAULT_BLKSIZE;
    let step = blksize;
    let search_radius = DEFAULT_SEARCH_RADIUS;
    let blocks_x = 3u32;
    let blocks_y = 3u32;

    let centre = noisy_copy(width, 0.5, 0.2, 123);
    let neighbour = frame_shifted_by(&centre, width, height, 2, 1);

    let client = make_client();
    let level_len = (width * height) as usize;
    let (pyramid, neighbour_slots) = pair_pyramid(&client, &centre, &neighbour);
    let mv_len = (blocks_x * blocks_y * 2) as usize;
    let mv_field = client.empty(mv_len * size_of::<i32>());
    // Confidence is off below, so this placeholder is never indexed.
    let confidence = client.empty(size_of::<f32>());

    let grid = CubeCount::new_2d(blocks_x, blocks_y);
    let dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);

    unsafe {
        let mv_field_handle = mv_field.clone();
        let pyramid_arg = ArrayArg::from_raw_parts(pyramid, 2 * level_len);
        let neighbour_slots_arg = ArrayArg::from_raw_parts(neighbour_slots, 1);
        let mv_field_arg = ArrayArg::from_raw_parts(mv_field_handle, mv_len);
        let confidence_arg = ArrayArg::from_raw_parts(confidence, 1);

        let noise = sad_floor_buffer(&client, DERIVED_SAD_FLOOR, 0.0);
        let noise_arg = ArrayArg::from_raw_parts(noise, DERIVED_WORDS as usize);

        mc_block_match_fine::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            pyramid_arg,
            neighbour_slots_arg,
            mv_field_arg,
            confidence_arg,
            0u32,
            false,
            noise_arg,
            DERIVED_SAD_FLOOR,
            1.0,
            width,
            height,
            level_len as u32,
            blksize,
            step,
            search_radius,
            0u32,
            blocks_x,
            mv_len as u32,
            1,
        );
    }

    let bytes = client.read_one(mv_field).expect("mv readback failed");
    let mv = i32::from_bytes(&bytes);

    // The middle block and its search window sit `blksize` pixels from every edge, so no clamped
    // read is hit.
    let middle_block_x = 1u32;
    let middle_block_y = 1u32;
    let mv_index = ((middle_block_y * blocks_x + middle_block_x) * 2) as usize;
    assert_eq!(
        (mv[mv_index], mv[mv_index + 1]),
        (2, 1),
        "a clean (+2, +1) shift of the centre content should give exactly \
         MV=(2, 1) at default blksize={blksize}/search_radius={search_radius}, got ({}, {})",
        mv[mv_index],
        mv[mv_index + 1],
    );
}

/// Runs the coarse block-match kernel over one block covering the whole buffer.
///
/// The block seeds exactly one fine block at a level scale of 1, so the returned vector is the raw
/// coarse offset.
fn run_coarse_block_match_single_block(
    blksize: u32,
    search_radius: u32,
    centre: &[f32],
    neighbour: &[f32],
) -> (i32, i32) {
    let client = make_client();
    let level_len = (blksize * blksize) as usize;
    assert_eq!(centre.len(), level_len);
    assert_eq!(neighbour.len(), level_len);

    let (pyramid, neighbour_slots) = pair_pyramid(&client, centre, neighbour);
    let mv_field = client.empty(2 * size_of::<i32>());

    let grid = CubeCount::new_2d(1, 1);
    let dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);

    unsafe {
        let mv_field_handle = mv_field.clone();
        let pyramid_arg = ArrayArg::from_raw_parts(pyramid, 2 * level_len);
        let neighbour_slots_arg = ArrayArg::from_raw_parts(neighbour_slots, 1);
        let mv_field_arg = ArrayArg::from_raw_parts(mv_field_handle, 2);

        mc_block_match_coarse::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            pyramid_arg,
            neighbour_slots_arg,
            mv_field_arg,
            0u32,
            blksize,
            blksize,
            0,
            level_len as u32,
            blksize,
            blksize,
            search_radius,
            1,
            1,
            1,
            blksize,
            2,
        );
    }

    let mv_bytes = client.read_one(mv_field).expect("mv readback failed");
    let mv = i32::from_bytes(&mv_bytes);
    (mv[0], mv[1])
}

/// Every candidate ties at a SAD of 0, and the raster scan reaches the window corner first.
#[test]
fn block_match_fine_flat_region_tie_resolves_to_zero_motion() {
    let blksize = 16u32;
    let search_radius = 4u32;
    let value = 0.5f32;
    let centre = vec![value; (blksize * blksize) as usize];
    let neighbour = vec![value; (blksize * blksize) as usize];

    let (mv_x, mv_y, confidence) =
        run_fine_block_match_single_block(blksize, search_radius, &centre, &neighbour, 0.0, 1.0);

    assert_eq!(
        (mv_x, mv_y),
        (0, 0),
        "a flat region gives an exact SAD tie at every candidate, which must \
         resolve to the zero-motion seed, not the window corner \
         (-{search_radius}, -{search_radius}); got ({mv_x}, {mv_y})",
    );

    // Confidence is 1.0 whichever candidate wins the tie, so only the vector above pins the
    // tie-break. This guards the exact-zero case of the confidence formula.
    assert_eq!(
        confidence, 1.0,
        "an exact SAD=0 match should give full confidence"
    );
}

/// A corner-biased tie here mis-seeds every fine block under it, which compounds across pyramid
/// levels under `Chained` estimation.
#[test]
fn block_match_coarse_flat_region_tie_resolves_to_zero_motion() {
    let blksize = 16u32;
    let search_radius = 4u32;
    let value = 0.5f32;
    let centre = vec![value; (blksize * blksize) as usize];
    let neighbour = vec![value; (blksize * blksize) as usize];

    let (mv_x, mv_y) = run_coarse_block_match_single_block(blksize, search_radius, &centre, &neighbour);

    assert_eq!(
        (mv_x, mv_y),
        (0, 0),
        "a flat region gives an exact SAD tie at every candidate, which the \
         coarse pass must resolve to the zero-motion candidate, not the window \
         corner (-{search_radius}, -{search_radius}); got ({mv_x}, {mv_y})",
    );
}

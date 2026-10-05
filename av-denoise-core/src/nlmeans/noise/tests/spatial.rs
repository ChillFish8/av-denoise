use crate::nlmeans::align::StorageAlign;
use crate::nlmeans::noise::spatial::{
    noise_partials_slot_stride_bytes,
    partials_len,
    sigma_block_p25_from_partials,
    sigma_from_abs_sum,
};

/// Hand-computed interior areas of a 70x20 frame's ragged 3x3 block grid, summing to 1224.
const RAGGED_CUBE_AREAS: [[f32; 3]; 3] = [[217.0, 224.0, 35.0], [248.0, 256.0, 40.0], [93.0, 96.0, 15.0]];

#[test]
fn sigma_from_abs_sum_zero_for_zero_response() {
    let sigma = sigma_from_abs_sum(0.0, 64, 64);
    assert_eq!(sigma, 0.0);
}

#[test]
fn partials_len_matches_cube_grid() {
    let one_cube = partials_len(32, 8);
    let spilled = partials_len(33, 9);
    let full_hd = partials_len(1920, 1080);

    assert_eq!(one_cube, 4); // exactly one BLOCK_X x BLOCK_Y cube
    assert_eq!(spilled, 16); // spills into a 2x2 cube grid
    assert_eq!(full_hd, 60 * 135 * 4);
}

#[test]
fn noise_partials_slot_stride_bytes_pads_odd_cube_count() {
    // One block's 16 bytes pad up to the next 32-byte multiple.
    let align = StorageAlign::new(32);
    let stride = noise_partials_slot_stride_bytes(32, 8, align);
    assert_eq!(stride, 32);
}

#[test]
fn noise_partials_slot_stride_bytes_aligned_count_unchanged() {
    // A 2x2 grid's 64 bytes are already a multiple of 32.
    let align = StorageAlign::new(32);
    let stride = noise_partials_slot_stride_bytes(33, 9, align);
    assert_eq!(stride, 64);
}

/// The area cancels out of each block's sigma, so a uniform response gives every block the same
/// sigma as the frame-wide estimate.
#[test]
fn sigma_block_p25_from_partials_uniform_response_matches_frame_wide() {
    let width = 70;
    let height = 20;
    let channels = 1;
    let response = 0.02f32;

    let mut partials = vec![0.0f32; 3 * 3 * 4];
    let mut total_sum = 0.0f32;
    for (cube_y, row) in RAGGED_CUBE_AREAS.iter().enumerate() {
        for (cube_x, &area) in row.iter().enumerate() {
            let sum = response * area;
            partials[(cube_y * 3 + cube_x) * 4] = sum;
            total_sum += sum;
        }
    }

    let sigma_low = sigma_block_p25_from_partials(&partials, channels, width, height);
    let expected = sigma_from_abs_sum(total_sum, width, height);

    assert!(
        (sigma_low[0] - expected).abs() < expected * 1e-4,
        "uniform response should reproduce the frame-wide estimate {expected}, got {}",
        sigma_low[0]
    );
}

/// Nine blocks get a shuffled 1..=9 run of 8-bit sigmas, so the lower quartile lands exactly on
/// the third smallest.
#[test]
fn sigma_block_p25_from_partials_distinct_sums_pick_expected_cube() {
    let width = 70;
    let height = 20;
    let channels = 1;
    let sigma_targets_255 = [5.0f32, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0, 4.0, 6.0];

    let mut partials = vec![0.0f32; 3 * 3 * 4];
    for (cube_y, row) in RAGGED_CUBE_AREAS.iter().enumerate() {
        for (cube_x, &area) in row.iter().enumerate() {
            let cube = cube_y * 3 + cube_x;
            let sigma_target = sigma_targets_255[cube] / 255.0;
            let sum = sigma_target * 6.0 * area / std::f32::consts::FRAC_PI_2.sqrt();
            partials[cube * 4] = sum;
        }
    }

    let sigma_low = sigma_block_p25_from_partials(&partials, channels, width, height);
    let expected = 3.0 / 255.0;
    assert!(
        (sigma_low[0] - expected).abs() < 1e-4,
        "expected the lower quartile to land on the third-smallest cube sigma {expected}, got {}",
        sigma_low[0]
    );
}

use cubecl::prelude::*;

use crate::collab::PATCH_SIZE;
use crate::collab::kernels::aggregate::{collab_normalise, kaiser_window};
use crate::nlmeans::{BLOCK_X, BLOCK_Y};
use crate::tune::gpu::read_i32;
use crate::tune::tests::runtime::test_client;

/// The weight map is read by the pixels around each origin, so normalise must leave it for the caller's
/// `collab_zero_accum` launch rather than clear it as it goes.
#[test]
fn normalise_zeroes_the_accumulator_region_and_leaves_the_weight_map() {
    let (width, height) = (21u32, 16u32);
    let pixels = (width * height) as usize;
    let channels = 2u32;
    let region_len = pixels * channels as usize;

    let accum_host: Vec<i32> = (0..2 * region_len as i32).map(|value| value + 1).collect();
    let wsum_host: Vec<i32> = (0..2 * pixels as i32).map(|value| value + 7).collect();

    let client = test_client();
    let accum_bytes = i32::as_bytes(&accum_host);
    let wsum_bytes = i32::as_bytes(&wsum_host);
    let accum = client.create_from_slice(accum_bytes);
    let wsum = client.create_from_slice(wsum_bytes);
    let kaiser = kaiser_window(0.0);
    let kaiser_buf = client.create_from_slice(f32::as_bytes(&kaiser));
    let output = client.empty(region_len * size_of::<f32>());

    let blocks_x = width.div_ceil(BLOCK_X);
    let blocks_y = height.div_ceil(BLOCK_Y);
    let grid = CubeCount::new_2d(blocks_x, blocks_y);
    let dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);
    let accum_launched = accum.clone();
    let wsum_launched = wsum.clone();

    unsafe {
        let accum_arg = BufferArg::from_raw_parts(accum_launched, 2 * region_len);
        let wsum_arg = BufferArg::from_raw_parts(wsum_launched, 2 * pixels);
        let output_arg = BufferArg::from_raw_parts(output, region_len);

        collab_normalise::launch_unchecked(
            &client,
            grid,
            dim,
            channels as usize,
            accum_arg,
            wsum_arg,
            BufferArg::from_raw_parts(kaiser_buf, PATCH_SIZE as usize),
            output_arg,
            pixels as u32,
            width,
            height,
            channels,
        );
    }

    let accum_after = read_i32(&client, &accum, 2 * region_len).expect("accum reads back");
    let wsum_after = read_i32(&client, &wsum, 2 * pixels).expect("wsum reads back");

    assert_eq!(
        &accum_after[..region_len],
        &accum_host[..region_len],
        "the other region is untouched"
    );
    assert!(
        accum_after[region_len..].iter().all(|&value| value == 0),
        "every channel of the read region is zeroed"
    );
    assert_eq!(
        wsum_after, wsum_host,
        "normalise leaves the whole weight map as it was"
    );
}

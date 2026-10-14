use cubecl::prelude::*;

use crate::collab::kernels::aggregate::collab_normalise;
use crate::nlmeans::{BLOCK_X, BLOCK_Y};
use crate::tune::gpu::read_i32;
use crate::tune::tests::runtime::{TestRuntime, test_client};

#[test]
fn normalise_leaves_the_region_it_read_zeroed() {
    let (width, height) = (21u32, 16u32);
    let pixels = (width * height) as usize;
    let stored_ch = 4u32;
    let channels = 3u32;
    let region_len = pixels * stored_ch as usize;

    let accum_host: Vec<i32> = (0..2 * region_len as i32).map(|value| value + 1).collect();
    let wsum_host: Vec<i32> = (0..2 * pixels as i32).map(|value| value + 7).collect();

    let client = test_client();
    let accum_bytes = i32::as_bytes(&accum_host);
    let wsum_bytes = i32::as_bytes(&wsum_host);
    let accum = client.create_from_slice(accum_bytes);
    let wsum = client.create_from_slice(wsum_bytes);
    let output = client.empty(region_len * size_of::<f32>());

    let blocks_x = width.div_ceil(BLOCK_X);
    let blocks_y = height.div_ceil(BLOCK_Y);
    let grid = CubeCount::new_2d(blocks_x, blocks_y);
    let dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);

    unsafe {
        collab_normalise::launch_unchecked::<TestRuntime>(
            &client,
            grid,
            dim,
            stored_ch as usize,
            ArrayArg::from_raw_parts(accum.clone(), 2 * region_len),
            ArrayArg::from_raw_parts(wsum.clone(), 2 * pixels),
            ArrayArg::from_raw_parts(output, region_len),
            pixels as u32,
            width,
            height,
            channels,
            stored_ch,
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
        "every stored channel lane of the read region is zeroed"
    );
    assert_eq!(
        &wsum_after[..pixels],
        &wsum_host[..pixels],
        "the other region is untouched"
    );
    assert!(
        wsum_after[pixels..].iter().all(|&value| value == 0),
        "the read region's weight sum is zeroed"
    );
}

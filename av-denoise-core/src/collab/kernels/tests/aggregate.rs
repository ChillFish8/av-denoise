use cubecl::prelude::*;

use crate::collab::kernels::aggregate::collab_normalise;
use crate::collab::kernels::packed::{collab_normalise_packed, pack_accum_host};
use crate::nlmeans::{BLOCK_X, BLOCK_Y};
use crate::tune::gpu::read_i32;
use crate::tune::tests::runtime::{packed_supported, test_client};

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
            stored_ch as usize,
            accum_arg,
            wsum_arg,
            output_arg,
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

#[test]
fn packed_normalise_leaves_the_region_it_read_zeroed() {
    if !packed_supported() {
        eprintln!("skipped, this device has no 64-bit atomic add");
        return;
    }

    let (width, height) = (21u32, 16u32);
    let pixels = (width * height) as usize;
    let client = test_client();

    for (channels, stored_ch) in [(1u32, 1u32), (2, 2), (3, 4)] {
        let rest_channels = (stored_ch - 1) as usize;
        let output_len = pixels * stored_ch as usize;
        let rest_region = pixels * rest_channels;

        let packed_host: Vec<i64> = (0..2 * pixels as i32)
            .map(|index| pack_accum_host(index + 1, index + 7))
            .collect();
        let rest_host: Vec<i32> = (0..(2 * rest_region).max(1) as i32)
            .map(|value| value + 1)
            .collect();
        let packed_bytes = i64::as_bytes(&packed_host);
        let rest_bytes = i32::as_bytes(&rest_host);
        let packed = client.create_from_slice(packed_bytes);
        let rest = client.create_from_slice(rest_bytes);
        let output = client.empty(output_len * size_of::<f32>());

        let blocks_x = width.div_ceil(BLOCK_X);
        let blocks_y = height.div_ceil(BLOCK_Y);
        let grid = CubeCount::new_2d(blocks_x, blocks_y);
        let dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);
        let packed_launched = packed.clone();
        let rest_launched = rest.clone();
        let packed_len = packed_host.len();
        let rest_len = rest_host.len();

        unsafe {
            let packed_arg = BufferArg::from_raw_parts(packed_launched, packed_len);
            let rest_arg = BufferArg::from_raw_parts(rest_launched, rest_len);
            let output_arg = BufferArg::from_raw_parts(output, output_len);

            collab_normalise_packed::launch_unchecked(
                &client,
                grid,
                dim,
                stored_ch as usize,
                packed_arg,
                rest_arg,
                output_arg,
                pixels as u32,
                width,
                height,
                channels,
                stored_ch,
            );
        }

        let packed_bytes_after = client.read_one(packed).expect("packed reads back");
        let rest_bytes_after = client.read_one(rest).expect("rest reads back");
        let packed_after = i64::from_bytes(&packed_bytes_after);
        let rest_after = i32::from_bytes(&rest_bytes_after);

        assert_eq!(
            &packed_after[..pixels],
            &packed_host[..pixels],
            "{stored_ch}: the other packed region is untouched"
        );
        assert!(
            packed_after[pixels..2 * pixels].iter().all(|&word| word == 0),
            "{stored_ch}: the read packed region is zeroed"
        );
        assert_eq!(
            &rest_after[..rest_region],
            &rest_host[..rest_region],
            "{stored_ch}: the other rest region is untouched"
        );
        assert!(
            rest_after[rest_region..2 * rest_region]
                .iter()
                .all(|&value| value == 0),
            "{stored_ch}: every rest channel lane of the read region is zeroed"
        );
    }
}

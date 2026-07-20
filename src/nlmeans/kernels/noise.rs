use cubecl::prelude::*;

use super::helpers::read_line;

/// Per-cube stage of the Immerkær noise estimate. Every interior thread
/// applies the 3×3 Laplacian-difference mask to its pixel and the cube
/// reduces the absolute responses per channel into one partial sum.
/// Border pixels (and threads that fall outside the image because the
/// grid overshoots on the last row/column of cubes) contribute zero.
/// The result layout is `partials[cube_index * 4 + lane]` with unused
/// lanes left at zero.
#[cube(launch_unchecked)]
pub fn nlm_noise_partial<N: Size>(
    input: &Array<Vector<f32, N>>,
    partials: &mut Array<f32>,
    frame: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] channels: u32,
    #[comptime] block_x: u32,
    #[comptime] block_y: u32,
) {
    let threads = comptime!(block_x * block_y);
    let mut scratch = SharedMemory::<f32>::new(comptime!(block_x * block_y * 4) as usize);

    let x = ABSOLUTE_POS_X;
    let y = ABSOLUTE_POS_Y;
    let tid = UNIT_POS_Y * block_x + UNIT_POS_X;

    let interior = x >= 1 && x < width - 1 && y >= 1 && y < height - 1;

    let mut response = Vector::<f32, N>::empty();
    if interior {
        let c = read_line(input, x, y, frame, width, height);
        let l = read_line(input, x - 1, y, frame, width, height);
        let r = read_line(input, x + 1, y, frame, width, height);
        let u = read_line(input, x, y - 1, frame, width, height);
        let d = read_line(input, x, y + 1, frame, width, height);
        let ul = read_line(input, x - 1, y - 1, frame, width, height);
        let ur = read_line(input, x + 1, y - 1, frame, width, height);
        let dl = read_line(input, x - 1, y + 1, frame, width, height);
        let dr = read_line(input, x + 1, y + 1, frame, width, height);
        let four = Vector::<f32, N>::empty().fill(4.0f32);
        let two = Vector::<f32, N>::empty().fill(2.0f32);
        response = c * four - (l + r + u + d) * two + (ul + ur + dl + dr);
    }

    #[unroll]
    for ch in 0..channels {
        scratch[(tid * 4 + ch) as usize] = f32::abs(response[ch as usize]);
    }
    #[unroll]
    for ch in channels..4u32 {
        scratch[(tid * 4 + ch) as usize] = 0.0f32;
    }

    sync_cube();

    if tid == 0 {
        let cube_index = CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X;
        #[unroll]
        for ch in 0..4u32 {
            let mut sum = 0.0f32;
            for t in 0..threads {
                sum += scratch[(t * 4 + ch) as usize];
            }
            partials[(cube_index * 4 + ch) as usize] = sum;
        }
    }
}

/// Final stage of the Immerkær noise estimate. One cube sums every
/// per-cube partial into the per-channel totals for the given ring
/// slot. Each thread accumulates a strided share of the partials and
/// thread zero folds the shares together.
#[cube(launch_unchecked)]
pub fn nlm_noise_reduce(
    partials: &Array<f32>,
    results: &mut Array<f32>,
    slot: u32,
    num_partials: u32,
    #[comptime] block: u32,
) {
    let mut scratch = SharedMemory::<f32>::new(comptime!(block * 4) as usize);
    let tid = UNIT_POS_X;

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;
    let mut i = tid;
    while i < num_partials {
        sum0 += partials[(i * 4) as usize];
        sum1 += partials[(i * 4 + 1) as usize];
        sum2 += partials[(i * 4 + 2) as usize];
        sum3 += partials[(i * 4 + 3) as usize];
        i += block;
    }
    scratch[(tid * 4) as usize] = sum0;
    scratch[(tid * 4 + 1) as usize] = sum1;
    scratch[(tid * 4 + 2) as usize] = sum2;
    scratch[(tid * 4 + 3) as usize] = sum3;

    sync_cube();

    if tid == 0 {
        #[unroll]
        for ch in 0..4u32 {
            let mut total = 0.0f32;
            for t in 0..block {
                total += scratch[(t * 4 + ch) as usize];
            }
            results[(slot * 4 + ch) as usize] = total;
        }
    }
}

use cubecl::prelude::*;

use super::helpers::*;
use crate::bench_api::HostIo;
use crate::nlmeans::kernels::gpu_cast_f16;
use crate::nlmeans::*;

const WIDTH_PX: u32 = 64;
const HEIGHT_PX: u32 = 48;

fn ramp(len: usize, start: f32) -> Vec<f32> {
    let mut values = Vec::with_capacity(len);
    for index in 0..len {
        let step = (index % 997) as f32 / 997.0;
        values.push(start + step * 0.9);
    }
    values
}

fn ring_params(channels: ChannelMode) -> NlmParams {
    NlmParams {
        temporal_radius: 2,
        channels,
        ..NlmParams::default()
    }
}

/// Reads both rings back and checks every slot of the search ring is the f16 of the input ring.
fn assert_search_ring_mirrors_input(front: &NlmDenoiser<R>, client: &ComputeClient<R>) {
    let input_handle = front.input_ring().clone();
    let search_handle = front.search_ring().expect("search ring enabled").clone();
    let input_bytes = client.read_one(input_handle).expect("input readback failed");
    let search_bytes = client.read_one(search_handle).expect("search readback failed");
    let input = f32::from_bytes(&input_bytes);
    let search = half::f16::from_bytes(&search_bytes);

    for (index, value) in input.iter().enumerate() {
        let expected = half::f16::from_f32(*value);
        assert_eq!(search[index].to_bits(), expected.to_bits(), "element {index}");
    }
}

fn launch_cast_over_middle_slot(cubes: u32, threads: u32, total_threads: u32) {
    let client = make_client();
    let slot_len = 4096usize;
    let source = ramp(slot_len * 3, 0.05);
    let source_bytes = f32::as_bytes(&source);
    let source_buf = client.create_from_slice(source_bytes);
    let sentinel = vec![half::f16::from_f32(-1.0); slot_len * 3];
    let sentinel_bytes = half::f16::as_bytes(&sentinel);
    let target_buf = client.create_from_slice(sentinel_bytes);

    unsafe {
        gpu_cast_f16::launch_unchecked::<R>(
            &client,
            CubeCount::new_1d(cubes),
            CubeDim::new_1d(threads),
            ArrayArg::from_raw_parts(source_buf, slot_len * 3),
            ArrayArg::from_raw_parts(target_buf.clone(), slot_len * 3),
            slot_len as u32,
            slot_len as u32,
            total_threads,
        );
    }

    let target_bytes = client.read_one(target_buf).expect("cast readback failed");
    let target = half::f16::from_bytes(&target_bytes);
    for (index, value) in target.iter().enumerate() {
        let in_slot = (slot_len..2 * slot_len).contains(&index);
        let expected = if in_slot {
            half::f16::from_f32(source[index])
        } else {
            half::f16::from_f32(-1.0)
        };
        assert_eq!(value.to_bits(), expected.to_bits(), "element {index}");
    }
}

#[test]
fn gpu_cast_f16_writes_one_slot_and_leaves_the_rest() {
    launch_cast_over_middle_slot(16, 256, 16 * 256);
}

#[test]
fn gpu_cast_f16_strided_loop_covers_a_slot_larger_than_the_grid() {
    launch_cast_over_middle_slot(1, 256, 256);
}

#[test]
fn the_search_ring_mirrors_every_pushed_and_duplicated_slot() {
    let client = make_client();

    for channels in [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv] {
        let params = ring_params(channels);
        let mut front = NlmDenoiser::<R>::new(&client, params, WIDTH_PX, HEIGHT_PX);
        front.enable_search_ring();

        let frame_len = (WIDTH_PX * HEIGHT_PX * channels.count()) as usize;
        let first = ramp(frame_len, 0.05);
        let second = ramp(frame_len, 0.3);
        front.push_frame(&first);
        front.push_frame(&second);

        // The first push primes two leading copies, so the fill adds one more copy of the second push.
        front.fill_ring_with_last_frame().expect("fill failed");

        assert_search_ring_mirrors_input(&front, &client);
    }
}

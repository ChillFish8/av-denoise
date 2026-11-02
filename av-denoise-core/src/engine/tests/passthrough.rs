use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::passthrough::PassthroughRing;
use crate::engine::{DevicePlane, SampleFormat};

const WIDTH: u32 = 7;
const HEIGHT: u32 = 5;

fn client() -> Client {
    crate::tests::compilation_cache::install_test_compilation_cache();
    crate::testing::test_client()
}

fn plane_bytes() -> usize {
    let pixels = u64::from(WIDTH * HEIGHT);
    SampleFormat::U8.plane_bytes(pixels) as usize
}

/// A plane whose every byte, padding included, is tagged by `frame` and `plane`.
fn tagged_plane(frame: u32, plane: u32) -> Vec<u8> {
    (0..plane_bytes())
        .map(|index| (index as u32 * 7 + frame * 31 + plane * 101) as u8)
        .collect()
}

fn upload_frame(client: &Client, frame: u32) -> Vec<Handle> {
    (0..2)
        .map(|plane| {
            let bytes = tagged_plane(frame, plane);
            client.create_from_slice(&bytes)
        })
        .collect()
}

fn as_planes(handles: &[Handle]) -> Vec<DevicePlane<'_>> {
    handles
        .iter()
        .map(|handle| DevicePlane::new(handle, WIDTH, HEIGHT))
        .collect()
}

fn load_frame(client: &Client, ring: &PassthroughRing, frame: u32) -> Vec<Vec<u8>> {
    let outputs: Vec<Handle> = (0..2).map(|_| client.empty(plane_bytes())).collect();
    let planes = as_planes(&outputs);
    ring.load(frame, &planes);

    outputs
        .iter()
        .map(|handle| {
            let bytes = client.read_one(handle.clone()).expect("read plane");
            bytes.to_vec()
        })
        .collect()
}

fn store_frame(client: &Client, ring: &PassthroughRing, frame: u32) {
    let inputs = upload_frame(client, frame);
    let planes = as_planes(&inputs);
    ring.store(frame, &planes);
}

#[test]
fn a_stored_frame_loads_back_word_for_word() {
    let client = client();
    let bytes = plane_bytes() as u64;
    let ring = PassthroughRing::new(&client, 3, 2, bytes);

    store_frame(&client, &ring, 0);

    let loaded = load_frame(&client, &ring, 0);
    assert_eq!(loaded[0], tagged_plane(0, 0));
    assert_eq!(loaded[1], tagged_plane(0, 1));
}

#[test]
fn every_held_frame_keeps_its_own_slot() {
    let client = client();
    let bytes = plane_bytes() as u64;
    let ring = PassthroughRing::new(&client, 3, 2, bytes);

    for frame in 0..3 {
        store_frame(&client, &ring, frame);
    }

    for frame in 0..3 {
        let loaded = load_frame(&client, &ring, frame);
        assert_eq!(loaded[0], tagged_plane(frame, 0), "frame {frame}");
        assert_eq!(loaded[1], tagged_plane(frame, 1), "frame {frame}");
    }
}

#[test]
fn a_frame_takes_the_slot_of_the_frame_a_ring_length_before() {
    let client = client();
    let bytes = plane_bytes() as u64;
    let ring = PassthroughRing::new(&client, 3, 2, bytes);

    for frame in 0..5 {
        store_frame(&client, &ring, frame);
    }

    for frame in 2..5 {
        let loaded = load_frame(&client, &ring, frame);
        assert_eq!(loaded[0], tagged_plane(frame, 0), "frame {frame}");
        assert_eq!(loaded[1], tagged_plane(frame, 1), "frame {frame}");
    }
}

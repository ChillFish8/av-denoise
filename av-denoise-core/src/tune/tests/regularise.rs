use cubecl::prelude::*;

use crate::nlmeans::tests::helpers::make_client;
use crate::tune::regularise::{
    REGULARISE_CANDIDATES,
    RegulariseKey,
    RegulariseLaunch,
    RegulariseShape,
    launch,
};

const WIDTH: u32 = 200;
const HEIGHT: u32 = 120;
const BLKSIZE: u32 = 16;
const STEP: u32 = 8;
const SENTINEL: i32 = 0x5EED;

fn shape() -> RegulariseShape {
    RegulariseShape {
        level_width: WIDTH,
        level_height: HEIGHT,
        blksize: BLKSIZE,
        step: STEP,
        blocks_x: WIDTH.div_ceil(STEP),
        blocks_y: HEIGHT.div_ceil(STEP),
    }
}

fn frame(seed: u32) -> Vec<f32> {
    (0..WIDTH * HEIGHT)
        .map(|index| {
            let hash = index.wrapping_mul(2654435761) ^ seed.wrapping_mul(0x9E37_79B9);
            (hash >> 8) as f32 / (1u32 << 24) as f32
        })
        .collect()
}

fn build<R: Runtime>(client: &ComputeClient<R>) -> RegulariseLaunch<R> {
    let shape = shape();
    let blocks = (shape.blocks_x * shape.blocks_y) as usize;
    let centre_frame = frame(1);
    let neighbour_frame = frame(2);
    let field: Vec<i32> = (0..2 * blocks).map(|index| (index * 7 % 11) as i32 - 5).collect();
    let sentinel_field = vec![SENTINEL; 2 * blocks];

    let centre_bytes = f32::as_bytes(&centre_frame);
    let neighbour_bytes = f32::as_bytes(&neighbour_frame);
    let field_bytes = i32::as_bytes(&field);
    let sentinel_bytes = i32::as_bytes(&sentinel_field);

    RegulariseLaunch {
        client: client.clone(),
        centre: client.create_from_slice(centre_bytes),
        neighbour: client.create_from_slice(neighbour_bytes),
        level_len: (WIDTH * HEIGHT) as usize,
        mv_in: client.create_from_slice(field_bytes),
        mv_out: client.create_from_slice(sentinel_bytes),
        conf_out: client.empty(blocks * size_of::<f32>()),
        lambda_pixel: 0.5 * (BLKSIZE * BLKSIZE) as f32 * 0.02,
        sad_noise_floor: 0.0,
        thsad: (BLKSIZE * BLKSIZE) as f32 * 0.02,
        shape,
    }
}

fn read_field<R: Runtime>(launch: &RegulariseLaunch<R>) -> Vec<i32> {
    let bytes = launch
        .client
        .read_one(launch.mv_out.clone())
        .expect("mv readback");
    let blocks = (launch.shape.blocks_x * launch.shape.blocks_y) as usize;
    i32::from_bytes(&bytes)[..2 * blocks].to_vec()
}

#[test]
fn tuning_scratch_leaves_real_buffers_untouched() {
    let client = make_client();
    let real = build(&client);
    let scratch = real.with_scratch();

    for index in 0..REGULARISE_CANDIDATES.len() {
        scratch.launch_candidate(index).expect("candidate launches");
    }

    let untouched = read_field(&real);
    assert!(untouched.iter().all(|&value| value == SENTINEL));
}

#[test]
fn tuned_launch_matches_candidate_zero() {
    let client = make_client();
    let direct = build(&client);
    direct.launch_candidate(0).expect("candidate 0 launches");
    let want = read_field(&direct);

    let tuned = build(&client);
    let tuned_out = tuned.mv_out.clone();
    launch(tuned);
    let bytes = client.read_one(tuned_out).expect("mv readback");
    let got = i32::from_bytes(&bytes)[..want.len()].to_vec();

    assert_eq!(got, want);
}

#[test]
fn every_candidate_matches_candidate_zero() {
    let client = make_client();
    let baseline = build(&client);
    baseline.launch_candidate(0).expect("candidate 0 launches");
    let want = read_field(&baseline);

    for index in 1..REGULARISE_CANDIDATES.len() {
        let candidate = build(&client);
        candidate.launch_candidate(index).expect("candidate launches");
        let got = read_field(&candidate);
        assert_eq!(got, want, "candidate {index} differs");
    }
}

#[test]
fn nearby_resolutions_share_a_key() {
    let full_hd = RegulariseKey::new(16, 8, 1920, 1080);
    let near = RegulariseKey::new(16, 8, 1900, 1200);
    let half = RegulariseKey::new(16, 8, 960, 540);

    assert_eq!(full_hd, near);
    assert_ne!(full_hd, half);
}

#[test]
fn different_steps_have_different_keys() {
    let dense = RegulariseKey::new(16, 8, 1920, 1080);
    let sparse = RegulariseKey::new(16, 16, 1920, 1080);

    assert_ne!(dense, sparse);
}

use cubecl::prelude::*;

use crate::nlmeans::noise::{build_spatial_offset_lut, spatial_offset_lut_len};
use crate::nlmeans::tests::helpers::make_client;
use crate::tune::nlm_window::{
    WINDOW_CANDIDATES,
    WindowConfidence,
    WindowKey,
    WindowLaunch,
    WindowPass,
    WindowShape,
};

const WIDTH: u32 = 200;
const HEIGHT: u32 = 120;

#[derive(Clone, Copy, Debug)]
struct Variant {
    pair: bool,
    yuv: bool,
    confidence: bool,
}

impl Variant {
    const fn single(yuv: bool) -> Self {
        Self {
            pair: false,
            yuv,
            confidence: false,
        }
    }

    const fn pair(yuv: bool, confidence: bool) -> Self {
        Self {
            pair: true,
            yuv,
            confidence,
        }
    }
}

fn shape(search_radius: u32, yuv: bool) -> WindowShape {
    let (channels, stored_ch) = if yuv { (3, 4) } else { (1, 1) };

    WindowShape {
        width: WIDTH,
        height: HEIGHT,
        channels,
        stored_ch,
        patch_radius: 3,
        search_radius,
    }
}

fn ring(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let hash = (index as u32).wrapping_mul(2654435761);
            (hash >> 8) as f32 / (1u32 << 24) as f32
        })
        .collect()
}

fn zero_buffer<R: Runtime>(client: &ComputeClient<R>, len: usize) -> cubecl::server::Handle {
    let zeros = vec![0.0f32; len];
    let bytes = f32::as_bytes(&zeros);
    client.create_from_slice(bytes)
}

fn build_confidence<R: Runtime>(client: &ComputeClient<R>, enabled: bool) -> WindowConfidence {
    let step = 8;
    let blocks_x = WIDTH.div_ceil(step);
    let blocks_y = HEIGHT.div_ceil(step);
    let blocks = (blocks_x * blocks_y) as usize;
    let conf_len = if enabled { blocks } else { 1 };
    let forward: Vec<f32> = ring(conf_len);
    let backward: Vec<f32> = ring(conf_len + 1)[1..].to_vec();
    let forward_bytes = f32::as_bytes(&forward);
    let backward_bytes = f32::as_bytes(&backward);

    WindowConfidence {
        use_confidence: enabled,
        conf_fwd: client.create_from_slice(forward_bytes),
        conf_bwd: client.create_from_slice(backward_bytes),
        conf_len,
        step,
        blocks_x,
        blocks_y,
    }
}

fn build<R: Runtime>(client: &ComputeClient<R>, pair: bool) -> WindowLaunch<R> {
    let variant = Variant {
        pair,
        yuv: false,
        confidence: false,
    };
    build_variant(client, variant)
}

fn build_variant<R: Runtime>(client: &ComputeClient<R>, variant: Variant) -> WindowLaunch<R> {
    let shape = shape(4, variant.yuv);
    let pixels = (WIDTH * HEIGHT) as usize;
    let frame_size = pixels * shape.stored_ch as usize;
    let frames = 3;
    let input = ring(frame_size * frames);
    let input_bytes = f32::as_bytes(&input);

    let pass = if variant.pair {
        WindowPass::Pair {
            frame_t: 1,
            frame_fwd: 2,
            frame_bwd: 0,
            noise_offset: 0.0,
            confidence: build_confidence(client, variant.confidence),
        }
    } else {
        let lut = build_spatial_offset_lut(shape.search_radius, 0.0, 0.0);
        let lut_bytes = f32::as_bytes(&lut);
        WindowPass::Single {
            frame_t: 1,
            offset_lut: client.create_from_slice(lut_bytes),
            offset_lut_len: spatial_offset_lut_len(shape.search_radius),
        }
    };

    WindowLaunch {
        client: client.clone(),
        input: client.create_from_slice(input_bytes),
        input_len: input.len(),
        accum: zero_buffer(client, frame_size),
        frame_size,
        weight_sum: zero_buffer(client, pixels),
        max_weight: zero_buffer(client, pixels),
        pixels,
        h2_inv_norm: 40.0,
        pass,
        shape,
    }
}

fn read_all<R: Runtime>(launch: &WindowLaunch<R>) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let read = |handle: &cubecl::server::Handle, len: usize| {
        let bytes = launch.client.read_one(handle.clone()).expect("readback");
        f32::from_bytes(&bytes)[..len].to_vec()
    };

    let accum = read(&launch.accum, launch.frame_size);
    let weight_sum = read(&launch.weight_sum, launch.pixels);
    let max_weight = read(&launch.max_weight, launch.pixels);
    (accum, weight_sum, max_weight)
}

fn assert_candidates_bit_exact(variant: Variant) {
    let client = make_client();
    let baseline = build_variant(&client, variant);
    baseline.launch_candidate(0).expect("candidate 0 launches");
    let want = read_all(&baseline);

    for index in 1..WINDOW_CANDIDATES.len() {
        let candidate = build_variant(&client, variant);
        candidate.launch_candidate(index).expect("candidate launches");
        let got = read_all(&candidate);
        assert_eq!(got, want, "candidate {index} differs, {variant:?}");
    }
}

#[test]
fn single_window_candidates_are_bit_exact() {
    assert_candidates_bit_exact(Variant::single(false));
}

#[test]
fn pair_window_candidates_are_bit_exact() {
    assert_candidates_bit_exact(Variant::pair(false, false));
}

#[test]
fn single_yuv_window_candidates_are_bit_exact() {
    assert_candidates_bit_exact(Variant::single(true));
}

#[test]
fn pair_yuv_window_candidates_are_bit_exact() {
    assert_candidates_bit_exact(Variant::pair(true, false));
}

#[test]
fn pair_window_candidates_are_bit_exact_with_confidence() {
    assert_candidates_bit_exact(Variant::pair(false, true));
}

#[test]
fn pair_yuv_window_candidates_are_bit_exact_with_confidence() {
    assert_candidates_bit_exact(Variant::pair(true, true));
}

#[test]
fn window_candidate_rejects_oversized_tile() {
    let client = make_client();
    let launch = build(&client, true);
    let limit = client.properties().hardware.max_shared_memory_size;
    let mut oversized = launch.clone();
    oversized.shape.search_radius = 64;
    oversized.shape.stored_ch = 4;

    let needed = oversized.shared_bytes(32, 16);
    assert!(needed > limit, "test radius must exceed the device limit");

    let result = oversized.launch_candidate(1);
    assert!(result.is_err());
}

#[test]
fn window_scratch_leaves_real_buffers_untouched() {
    let client = make_client();
    let real = build(&client, true);
    let scratch = real.with_scratch();

    for index in 0..WINDOW_CANDIDATES.len() {
        scratch.launch_candidate(index).expect("candidate launches");
    }

    let (accum, weight_sum, max_weight) = read_all(&real);
    assert!(accum.iter().all(|&value| value == 0.0));
    assert!(weight_sum.iter().all(|&value| value == 0.0));
    assert!(max_weight.iter().all(|&value| value == 0.0));
}

#[test]
fn window_keys_split_by_pass_and_radius() {
    let single = WindowKey::new(false, 1, 3, 4, false, 1920, 1080);
    let pair = WindowKey::new(true, 1, 3, 4, false, 1920, 1080);
    let wider = WindowKey::new(false, 1, 3, 6, false, 1920, 1080);

    assert_ne!(single, pair);
    assert_ne!(single, wider);
}

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::planar::PlanarEngine;
use crate::engine::{ChromaLayout, ChromaSiting, DevicePlane, Engine, EngineSpec, Geometry, SampleFormat};
use crate::nl4d::Nl4dOptions;
use crate::noise::colour::COLOUR_CELLS;
use crate::tune::gpu::read_f32;

const WIDTH: u32 = 128;
const HEIGHT: u32 = 96;
const CHROMA_WIDTH: u32 = WIDTH / 2;
const CHROMA_HEIGHT: u32 = HEIGHT / 2;
/// Enough frames for a whole reseed window around several targets.
const FRAMES: u32 = 40;

/// The luma of every frame, in 8-bit codes.
const LUMA: f32 = 110.0;
/// The peak luma grain of the neutral half, in 8-bit codes.
const NEUTRAL_GRAIN: f32 = 4.0;
/// The peak luma grain of the coloured half, three times the neutral half's.
const COLOURED_GRAIN: f32 = 12.0;
/// The Cb and Cr of the coloured half, in 8-bit codes.
const COLOURED: (f32, f32) = (170.0, 90.0);
/// The Cb and Cr of the neutral half, in 8-bit codes.
const NEUTRAL: (f32, f32) = (128.0, 128.0);

type Planes = Vec<Vec<u8>>;

fn client() -> Client {
    crate::tests::compilation_cache::install_test_compilation_cache();
    crate::testing::test_client()
}

fn engine(client: &Client, noise_map: bool) -> PlanarEngine {
    let options = Nl4dOptions {
        noise_map,
        ..Nl4dOptions::default()
    };
    let geometry = Geometry {
        width: WIDTH,
        height: HEIGHT,
        chroma: ChromaLayout::Yuv420,
        input: SampleFormat::U8,
        output: SampleFormat::U8,
        chroma_siting: ChromaSiting::Unknown,
    };
    let spec = EngineSpec::Nl4d {
        luma: Some(options),
        chroma: Some(options),
        geometry,
    };

    spec.build_planar_for_test(client).expect("the engine builds")
}

/// A value between -0.5 and 0.5 from a SplitMix64 hash of `seed` and `index`.
fn grain_unit(seed: u64, index: u64) -> f32 {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ index;
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    state = (state ^ (state >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    state ^= state >> 31;

    (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
}

/// A `width` by `height` plane whose left half reads `left` and right half `right`, each with its
/// own peak grain, drawn from `seed`.
fn split_plane(width: u32, height: u32, left: (f32, f32), right: (f32, f32), seed: u64) -> Vec<u8> {
    let mut plane = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let (code, grain) = if x < width / 2 { left } else { right };
            let index = u64::from(y * width + x);
            let sample = code + 2.0 * grain * grain_unit(seed, index);
            plane.push(sample.round() as u8);
        }
    }

    plane
}

/// Frame `index` of a static clip whose luma grain is heavier where the chroma is coloured.
///
/// With `swapped`, the chroma of the two halves trades places and the luma stays the same.
fn frame(index: u32, swapped: bool) -> Planes {
    let seed = 4 * u64::from(index);
    let luma = split_plane(WIDTH, HEIGHT, (LUMA, NEUTRAL_GRAIN), (LUMA, COLOURED_GRAIN), seed);

    let (left, right) = if swapped {
        (COLOURED, NEUTRAL)
    } else {
        (NEUTRAL, COLOURED)
    };
    let cb = split_plane(
        CHROMA_WIDTH,
        CHROMA_HEIGHT,
        (left.0, 2.0),
        (right.0, 2.0),
        seed + 1,
    );
    let cr = split_plane(
        CHROMA_WIDTH,
        CHROMA_HEIGHT,
        (left.1, 2.0),
        (right.1, 2.0),
        seed + 2,
    );

    vec![luma, cb, cr]
}

const PLANE_DIMS: [(u32, u32); 3] = [
    (WIDTH, HEIGHT),
    (CHROMA_WIDTH, CHROMA_HEIGHT),
    (CHROMA_WIDTH, CHROMA_HEIGHT),
];

fn as_planes(handles: &[Handle]) -> Vec<DevicePlane<'_>> {
    handles
        .iter()
        .zip(PLANE_DIMS)
        .map(|(handle, (width, height))| DevicePlane::new(handle, width, height))
        .collect()
}

fn upload(client: &Client, planes: &Planes) -> Vec<Handle> {
    planes
        .iter()
        .map(|plane| client.create_from_slice(plane))
        .collect()
}

/// Writes out `ready` frames and reads them back.
fn emit(engine: &mut PlanarEngine, client: &Client, ready: usize) -> Vec<Planes> {
    let mut outputs = Vec::with_capacity(ready);
    for _ in 0..ready {
        let handles: Vec<Handle> = PLANE_DIMS
            .iter()
            .map(|&(width, height)| client.empty((width * height) as usize))
            .collect();
        let planes = as_planes(&handles);
        engine.emit_into(&planes).expect("emit");

        let read: Planes = handles
            .into_iter()
            .zip(PLANE_DIMS)
            .map(|(handle, (width, height))| {
                let bytes = client.read_one(handle).expect("the output reads back");
                bytes[..(width * height) as usize].to_vec()
            })
            .collect();
        outputs.push(read);
    }

    outputs
}

fn push(engine: &mut PlanarEngine, client: &Client, planes: &Planes) -> Vec<Planes> {
    let handles = upload(client, planes);
    let device = as_planes(&handles);
    let ready = engine.push(&device).expect("push");

    emit(engine, client, ready)
}

fn push_context(engine: &mut PlanarEngine, client: &Client, planes: &Planes) {
    let handles = upload(client, planes);
    let device = as_planes(&handles);
    engine.push_context(&device).expect("push_context");
}

/// Every output of the whole clip, its flushed tail included.
fn stream(engine: &mut PlanarEngine, client: &Client, swapped: bool) -> Vec<Planes> {
    let mut outputs = Vec::new();
    for index in 0..FRAMES {
        let planes = frame(index, swapped);
        let emitted = push(engine, client, &planes);
        outputs.extend(emitted);
    }

    let tail = engine.finish().expect("finish");
    let flushed = emit(engine, client, tail);
    outputs.extend(flushed);

    assert_eq!(outputs.len(), FRAMES as usize, "one output per frame");
    outputs
}

/// The largest log gain in the luma pass's table.
fn largest_luma_log_gain(client: &Client, engine: &PlanarEngine) -> f32 {
    let [luma, _] = engine.colour_gains_for_test();
    let luma = luma.expect("the luma pass fits gains");
    let gains = read_f32(client, luma, COLOUR_CELLS as usize).expect("the luma gains read back");

    gains.iter().fold(0.0f32, |largest, gain| largest.max(gain.abs()))
}

#[test]
fn reseed_equals_streaming() {
    let client = client();
    let mut engine = engine(&client, true);
    let span = engine.window_span();
    let behind = span.behind as u32;
    let ahead = span.ahead as u32;

    let mut streamed = Vec::new();
    let mut largest_gain = 0.0f32;
    for index in 0..FRAMES {
        let planes = frame(index, false);
        let emitted = push(&mut engine, &client, &planes);
        streamed.extend(emitted);

        let gain = largest_luma_log_gain(&client, &engine);
        largest_gain = largest_gain.max(gain);
    }

    assert!(
        largest_gain > 0.1,
        "the luma gains stay near 1, at most {largest_gain}"
    );

    for target in [behind, FRAMES / 2, FRAMES - 1 - ahead] {
        engine.reset();
        for index in target - behind..target {
            let planes = frame(index, false);
            push_context(&mut engine, &client, &planes);
        }

        let mut reseeded = Vec::new();
        for index in target..=target + ahead {
            let planes = frame(index, false);
            let emitted = push(&mut engine, &client, &planes);
            reseeded.extend(emitted);
        }

        // A continuation emits every frame from `radius` past its first through the target, so the
        // target's region is the last.
        let radius = Nl4dOptions::default().temporal_radius;
        let expected_emits = (behind - radius + 1) as usize;
        assert_eq!(reseeded.len(), expected_emits, "target {target}: emitted frames");

        let got = reseeded.last().expect("the reseed emits its target");
        let want = &streamed[target as usize];
        for (plane, name) in ["y", "u", "v"].into_iter().enumerate() {
            assert!(
                got[plane] == want[plane],
                "target {target}: {name} differs from streaming"
            );
        }
    }
}

#[test]
fn no_noise_map_turns_off_luma_gains_only() {
    let client = client();

    let mut mapped = engine(&client, true);
    let mapped_plain = stream(&mut mapped, &client, false);
    mapped.reset();
    let mapped_swapped = stream(&mut mapped, &client, true);

    let mut unmapped = engine(&client, false);
    let unmapped_plain = stream(&mut unmapped, &client, false);
    unmapped.reset();
    let unmapped_swapped = stream(&mut unmapped, &client, true);

    let mapped_luma_moves = mapped_plain
        .iter()
        .zip(&mapped_swapped)
        .any(|(plain, swapped)| plain[0] != swapped[0]);
    assert!(mapped_luma_moves, "with the noise map, luma ignores the colour");

    for (index, (plain, swapped)) in unmapped_plain.iter().zip(&unmapped_swapped).enumerate() {
        assert!(
            plain[0] == swapped[0],
            "frame {index}: luma follows the colour without the noise map"
        );
    }

    for (index, (mapped, unmapped)) in mapped_plain.iter().zip(&unmapped_plain).enumerate() {
        assert!(mapped[1] == unmapped[1], "frame {index}: u follows the noise map");
        assert!(mapped[2] == unmapped[2], "frame {index}: v follows the noise map");
    }

    let [_, chroma] = unmapped.colour_gains_for_test();
    let chroma = chroma.expect("the chroma pass fits gains without the noise map");
    let chroma_len = 2 * COLOUR_CELLS as usize;
    let chroma_gains = read_f32(&client, chroma, chroma_len).expect("the chroma gains read back");
    let chroma_fitted = chroma_gains.iter().any(|&log_gain| log_gain != 0.0);
    assert!(chroma_fitted, "the chroma gains stay at 1 without the noise map");
}

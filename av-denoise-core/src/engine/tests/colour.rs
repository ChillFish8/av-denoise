use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::planar::PlanarEngine;
use crate::engine::{ChromaLayout, DevicePlane, Engine, EngineSpec, Geometry, PlaneSource, SampleFormat};
use crate::nl4d::Nl4dOptions;
use crate::noise::colour::{
    COLOUR_CELL_COUNT,
    COLOUR_CELL_WORDS,
    COLOUR_CELLS,
    COLOUR_CELLS_PER_BRIGHTNESS,
    ColourCells,
};
use crate::tune::gpu::{read_f32, read_i32};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
const CHROMA_WIDTH: u32 = WIDTH / 2;
const CHROMA_HEIGHT: u32 = HEIGHT / 2;
/// Enough pushes for a radius 1 engine to run passes through its noise look-ahead.
const STREAM_FRAMES: u32 = 14;
/// Enough frames to hold a whole reseed window mid-clip.
const RESEED_FRAMES: u32 = 30;

type Planes = Vec<Vec<u8>>;

/// The flat colour of a test frame, in 8-bit codes.
#[derive(Debug, Clone, Copy)]
struct FrameColour {
    luma: u8,
    cb: u8,
    cr: u8,
}

const NEUTRAL: FrameColour = FrameColour {
    luma: 120,
    cb: 128,
    cr: 128,
};

const COLOURED: FrameColour = FrameColour {
    luma: 120,
    cb: 170,
    cr: 90,
};

fn client() -> Client {
    crate::tests::compilation_cache::install_test_compilation_cache();
    crate::testing::test_client()
}

fn engine(client: &Client) -> PlanarEngine {
    let options = Nl4dOptions {
        temporal_radius: 1,
        ..Nl4dOptions::default()
    };
    let geometry = Geometry {
        width: WIDTH,
        height: HEIGHT,
        chroma: ChromaLayout::Yuv420,
        input: SampleFormat::U8,
        output: SampleFormat::U8,
    };
    let spec = EngineSpec::Nl4d {
        luma: Some(options),
        chroma: Some(options),
        geometry,
    };

    spec.build_planar_for_test(client).expect("the engine builds")
}

/// A value between 0 and 1 from a hash of `seed` and `index`.
fn hashed_unit(seed: u32, index: u32) -> f32 {
    let mut hash = index.wrapping_mul(0x9E37_79B9) ^ seed.wrapping_mul(0x85EB_CA6B);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x2C1B_3C6D);
    hash ^= hash >> 12;

    (hash >> 8) as f32 / (1u32 << 24) as f32
}

/// A plane of `code` with about 2.3 codes of hashed noise from `seed`.
fn noisy_plane(width: u32, height: u32, code: u8, seed: u32) -> Vec<u8> {
    (0..width * height)
        .map(|index| {
            let unit = hashed_unit(seed, index);
            let sample = f32::from(code) + 8.0 * (unit - 0.5);
            sample.round() as u8
        })
        .collect()
}

/// Frame `index` of a clip of flat `colour`, with its own noise.
fn frame(colour: FrameColour, index: u32) -> Planes {
    let seed = 3 * index;
    vec![
        noisy_plane(WIDTH, HEIGHT, colour.luma, seed),
        noisy_plane(CHROMA_WIDTH, CHROMA_HEIGHT, colour.cb, seed + 1),
        noisy_plane(CHROMA_WIDTH, CHROMA_HEIGHT, colour.cr, seed + 2),
    ]
}

fn upload(client: &Client, planes: &Planes) -> Vec<Handle> {
    planes
        .iter()
        .map(|plane| client.create_from_slice(plane))
        .collect()
}

fn as_planes(handles: &[Handle]) -> Vec<DevicePlane<'_>> {
    let dims = [
        (WIDTH, HEIGHT),
        (CHROMA_WIDTH, CHROMA_HEIGHT),
        (CHROMA_WIDTH, CHROMA_HEIGHT),
    ];
    handles
        .iter()
        .zip(dims)
        .map(|(handle, (width, height))| DevicePlane::new(handle, width, height))
        .collect()
}

/// Writes out `ready` frames, discarding them.
fn emit(engine: &mut PlanarEngine, client: &Client, ready: usize) {
    for _ in 0..ready {
        let outputs: Vec<Handle> = [
            (WIDTH, HEIGHT),
            (CHROMA_WIDTH, CHROMA_HEIGHT),
            (CHROMA_WIDTH, CHROMA_HEIGHT),
        ]
        .iter()
        .map(|&(width, height)| {
            let pixels = u64::from(width * height);
            let bytes = SampleFormat::U8.plane_bytes(pixels);
            client.empty(bytes as usize)
        })
        .collect();
        let planes = as_planes(&outputs);
        engine.emit_into(&planes).expect("emit");
    }
}

fn push(engine: &mut PlanarEngine, client: &Client, planes: &Planes) {
    let handles = upload(client, planes);
    let device = as_planes(&handles);
    let ready = engine.push(&device).expect("push");
    emit(engine, client, ready);
}

fn push_context(engine: &mut PlanarEngine, client: &Client, planes: &Planes) {
    let handles = upload(client, planes);
    let device = as_planes(&handles);
    engine.push_context(&device).expect("push_context");
}

/// Hands a frame's planes over one group at a time.
struct Source<'a> {
    client: &'a Client,
    planes: &'a Planes,
}

impl PlaneSource for Source<'_> {
    fn upload(&mut self, planes: std::ops::Range<usize>) -> Vec<Handle> {
        self.planes[planes]
            .iter()
            .map(|plane| self.client.create_from_slice(plane))
            .collect()
    }
}

fn push_from(engine: &mut PlanarEngine, client: &Client, planes: &Planes) {
    let mut source = Source { client, planes };
    let ready = engine.push_from(&mut source).expect("push_from");
    emit(engine, client, ready);
}

/// Both passes' colour cells.
fn both_cells(engine: &PlanarEngine) -> [&ColourCells; 2] {
    engine
        .colour_cells_for_test()
        .map(|cells| cells.expect("both passes pool colour cells"))
}

/// The blocks stream frame `frame` pooled into the colour cells of every plane of `cells`.
fn frame_blocks(client: &Client, cells: &ColourCells, frame: u32) -> i32 {
    let words = cells.words();
    let slot = cells.slot(frame) as usize;
    let ring_len = words * (slot + 1);
    let ring = read_i32(client, cells.frames(), ring_len).expect("the ring reads back");
    let slot_words = &ring[slot * words..(slot + 1) * words];
    let planes = cells.records() / COLOUR_CELLS;
    let colour_words = (planes * COLOUR_CELLS * COLOUR_CELL_WORDS) as usize;

    slot_words[..colour_words]
        .chunks(COLOUR_CELL_WORDS as usize)
        .map(|record| record[COLOUR_CELL_COUNT as usize])
        .sum()
}

fn window(client: &Client, cells: &ColourCells) -> Vec<f32> {
    read_f32(client, cells.window(), cells.words()).expect("the window reads back")
}

/// The window's blocks in colour cell record `cell`.
fn window_cell_blocks(sums: &[f32], cell: usize) -> f32 {
    sums[cell * COLOUR_CELL_WORDS as usize + COLOUR_CELL_COUNT as usize]
}

/// The neutral cell of a frame of luma 120, and the band of coloured cells at the same brightness.
fn neutral_and_coloured_cells() -> (usize, std::ops::Range<usize>) {
    let per_brightness = COLOUR_CELLS_PER_BRIGHTNESS as usize;
    let neutral = 7 * per_brightness;

    (neutral, neutral + 1..neutral + per_brightness)
}

#[test]
fn every_push_path_pools_both_passes() {
    let client = client();
    let mut engine = engine(&client);

    push_context(&mut engine, &client, &frame(NEUTRAL, 0));
    push_context(&mut engine, &client, &frame(NEUTRAL, 1));
    push(&mut engine, &client, &frame(NEUTRAL, 2));
    push_from(&mut engine, &client, &frame(NEUTRAL, 3));

    // Luma holds 12 blocks, and each chroma channel 4 blocks of its half-size plane.
    let most = [12, 8];
    for (pass, cells) in both_cells(&engine).into_iter().enumerate() {
        for stream_frame in 0..4 {
            let blocks = frame_blocks(&client, cells, stream_frame);
            let limit = most[pass];
            assert!(blocks > 0, "pass {pass} frame {stream_frame} pools no blocks");
            assert!(
                blocks <= limit,
                "pass {pass} frame {stream_frame} pools {blocks} blocks"
            );
        }
    }
}

/// Runs a neutral stream, ends it with `end`, then pushes a coloured stream and checks its window
/// holds only the coloured frames.
fn assert_new_stream_window(end: impl FnOnce(&mut PlanarEngine, &Client)) {
    let client = client();
    let mut engine = engine(&client);
    let (neutral, coloured) = neutral_and_coloured_cells();

    for index in 0..STREAM_FRAMES {
        push(&mut engine, &client, &frame(NEUTRAL, index));
    }

    let first_window = window(&client, both_cells(&engine)[0]);
    assert!(
        window_cell_blocks(&first_window, neutral) > 0.0,
        "the neutral stream pools"
    );

    end(&mut engine, &client);

    for index in 0..STREAM_FRAMES {
        push(&mut engine, &client, &frame(COLOURED, 100 + index));
    }

    for cells in both_cells(&engine) {
        let sums = window(&client, cells);
        let planes = cells.records() / COLOUR_CELLS;
        for plane in 0..planes as usize {
            let base = plane * COLOUR_CELLS as usize;
            let old = window_cell_blocks(&sums, base + neutral);
            let new: f32 = coloured
                .clone()
                .map(|cell| window_cell_blocks(&sums, base + cell))
                .sum();

            assert_eq!(old, 0.0, "plane {plane} still holds the neutral stream");
            assert!(new > 0.0, "plane {plane} holds no coloured frames");
        }
    }
}

#[test]
fn an_engine_reset_leaves_the_old_stream_out_of_the_window() {
    assert_new_stream_window(|engine, _| engine.reset());
}

#[test]
fn a_finished_stream_leaves_its_frames_out_of_the_next_window() {
    assert_new_stream_window(|engine, client| {
        let tail = engine.finish().expect("finish");
        emit(engine, client, tail);
    });
}

/// Frame `index` of a clip whose colour drifts a little every frame.
fn drifting_frame(index: u32) -> Planes {
    let colour = FrameColour {
        luma: 100 + index as u8,
        cb: 128 + 2 * index as u8,
        cr: 128,
    };

    frame(colour, index)
}

#[test]
fn a_reseed_sees_the_streamed_window_bit_for_bit() {
    let client = client();
    let mut engine = engine(&client);
    let span = engine.window_span();
    let target = RESEED_FRAMES / 2;
    let first = target - span.behind as u32;
    let last = target + span.ahead as u32;

    for index in 0..=last {
        push(&mut engine, &client, &drifting_frame(index));
    }

    let streamed = both_cells(&engine).map(|cells| window(&client, cells));

    engine.reset();
    for index in first..target {
        push_context(&mut engine, &client, &drifting_frame(index));
    }

    for index in target..=last {
        push(&mut engine, &client, &drifting_frame(index));
    }

    let reseeded = both_cells(&engine).map(|cells| window(&client, cells));

    for pass in 0..2 {
        let pooled: f32 = streamed[pass]
            .chunks(COLOUR_CELL_WORDS as usize)
            .map(|record| record[COLOUR_CELL_COUNT as usize])
            .sum();
        assert!(pooled > 0.0, "pass {pass}'s window pools blocks");

        let streamed_bits: Vec<u32> = streamed[pass].iter().map(|value| value.to_bits()).collect();
        let reseeded_bits: Vec<u32> = reseeded[pass].iter().map(|value| value.to_bits()).collect();
        assert_eq!(reseeded_bits, streamed_bits, "pass {pass}'s window");
    }
}

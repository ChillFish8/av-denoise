#![cfg(any(feature = "vulkan", feature = "metal"))]

use std::collections::HashMap;

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::bench_api::HostIo;
use crate::engine::{
    ChromaLayout,
    ChromaSiting,
    DevicePlane,
    FramePass,
    Geometry,
    PassEngine,
    PassGeometry,
    SampleFormat,
};
use crate::nl4d::grain::GrainChunk;
use crate::nl4d::grain::consts::{CHROMA_RECORD_LEN, FRAME_RECORD_LEN};
use crate::nl4d::grain::snapshot::colour_frame_len;
use crate::nl4d::tests::helpers::noisy_frames;
use crate::nl4d::{Nl4d, Nl4dDenoiser, Nl4dOptions, resolve_params};
use crate::nlmeans::ChannelMode;
use crate::nlmeans::tests::helpers::{make_client, upload_planes};
use crate::noise::colour::{COLOUR_CELLS, COLOUR_FIT_PLANE_WORDS, ColourBlocks, ColourFit, ColourGrid};
use crate::noise::{READING_WORDS, history_capacity};

const WIDTH: u32 = 128;
const HEIGHT: u32 = 96;
const RADIUS: u32 = 2;
/// Fewer frames than the reading history holds, so it still holds every frame at the end.
const FRAMES: usize = 18;
/// The frames whose fit the colour test sees as they are filtered, those a push's middle pass
/// centres on.
const OBSERVED_FRAMES: usize = 6;

fn options(noise_map: bool) -> Nl4dOptions {
    Nl4dOptions {
        temporal_radius: RADIUS,
        noise_map,
        grain_data: true,
        ..Nl4dOptions::default()
    }
}

/// The plane size a pass over `channels` reads, half each way for 4:2:0 chroma.
fn plane_size(channels: ChannelMode) -> (u32, u32) {
    match channels {
        ChannelMode::Luma => (WIDTH, HEIGHT),
        ChannelMode::Chroma => (WIDTH / 2, HEIGHT / 2),
    }
}

fn pass_geometry(channels: ChannelMode) -> PassGeometry {
    let (width, height) = plane_size(channels);

    PassGeometry {
        width,
        height,
        channels,
        input: SampleFormat::F32,
        output: SampleFormat::F32,
    }
}

fn frame_geometry() -> Geometry {
    Geometry {
        width: WIDTH,
        height: HEIGHT,
        chroma: ChromaLayout::Yuv420,
        input: SampleFormat::F32,
        output: SampleFormat::F32,
        chroma_siting: ChromaSiting::Left,
    }
}

/// An engine fed the block colour of every frame, as the planar engine feeds its passes.
struct ColourRun {
    client: Client,
    channels: ChannelMode,
    engine: Nl4d,
    blocks: ColourBlocks,
    /// Each frame's luma plane, then its Cb and Cr planes.
    planes: Vec<Vec<Handle>>,
    /// The fit words seen right after each frame was filtered, by stream frame.
    observed: HashMap<u32, Vec<f32>>,
}

impl ColourRun {
    fn new(options: Nl4dOptions, channels: ChannelMode) -> Self {
        let client = make_client();
        let geometry = pass_geometry(channels);
        let mut engine = Nl4d::new(&client, options, geometry).expect("build");
        let held_frames = engine.max_held_frames();
        let clip_geometry = frame_geometry();
        let blocks = ColourBlocks::new(&client, clip_geometry, held_frames).expect("a colour clip");
        let grid = match channels {
            ChannelMode::Luma => ColourGrid::Luma,
            ChannelMode::Chroma => ColourGrid::Chroma,
        };
        let source = blocks.source(grid);
        engine.measure_colour(source);

        let luma_frames = noisy_frames(WIDTH, HEIGHT, 1, FRAMES);
        let chroma_frames = noisy_frames(WIDTH / 2, HEIGHT / 2, 2, FRAMES);
        let mut planes = Vec::with_capacity(FRAMES);
        for (luma, chroma) in luma_frames.iter().zip(&chroma_frames) {
            let mut handles = upload_planes(&client, luma, 1);
            let chroma_handles = upload_planes(&client, chroma, 2);
            handles.extend(chroma_handles);
            planes.push(handles);
        }

        Self {
            client,
            channels,
            engine,
            blocks,
            planes,
            observed: HashMap::new(),
        }
    }

    /// Pushes every frame and the tail, emitting each frame, and drains the grain chunks.
    fn drive(&mut self) -> Vec<GrainChunk> {
        for index in 0..FRAMES {
            let ready = self.push(index);
            self.observe();
            self.emit(ready);
        }

        let tail = self.engine.finish().expect("finish");
        self.emit(tail);

        self.engine.drain_grain_chunks().expect("drain")
    }

    fn push(&mut self, index: usize) -> usize {
        let frame = self.engine.next_frame();
        let handles = &self.planes[index];
        let chroma_width = WIDTH / 2;
        let chroma_height = HEIGHT / 2;
        let planes = [
            DevicePlane::new(&handles[0], WIDTH, HEIGHT),
            DevicePlane::new(&handles[1], chroma_width, chroma_height),
            DevicePlane::new(&handles[2], chroma_width, chroma_height),
        ];
        self.blocks.queue(&planes, frame);

        let pass_planes = match self.channels {
            ChannelMode::Luma => &planes[..1],
            ChannelMode::Chroma => &planes[1..],
        };

        self.engine.push(pass_planes).expect("push")
    }

    /// Keeps the fit of the frame the last pass of a push was centred on.
    fn observe(&mut self) {
        let Some(centre) = self.engine.centre_frame_for_test() else {
            return;
        };

        let Some(fit) = self.engine.colour_fit_for_test() else {
            return;
        };

        let words = fit_words(&self.client, fit);
        self.observed.insert(centre, words);
    }

    fn emit(&mut self, count: usize) {
        let (width, height) = plane_size(self.channels);
        let pixels = (width * height) as usize;
        let plane_count = self.channels.count() as usize;
        for _ in 0..count {
            let outputs: Vec<Handle> = (0..plane_count)
                .map(|_| self.client.empty(pixels * size_of::<f32>()))
                .collect();
            let planes: Vec<_> = outputs
                .iter()
                .map(|output| DevicePlane::new(output, width, height))
                .collect();
            self.engine.emit_into(&planes).expect("emit");
        }
    }
}

fn read_words(client: &Client, handle: &Handle) -> Vec<f32> {
    let bytes = client.read_one(handle.clone()).expect("read back");
    let words = f32::from_bytes(&bytes);

    words.to_vec()
}

/// The fit's words in snapshot order, each plane's table, gains and fit header in turn.
fn fit_words(client: &Client, fit: &ColourFit) -> Vec<f32> {
    let table = read_words(client, fit.table());
    let gains = read_words(client, fit.gains());
    let headers = read_words(client, fit.fit_planes());
    let cells = COLOUR_CELLS as usize;
    let header_len = COLOUR_FIT_PLANE_WORDS as usize;

    let mut words = Vec::new();
    for plane in 0..fit.planes() as usize {
        let cells_start = plane * cells;
        let header_start = plane * header_len;
        words.extend_from_slice(&table[cells_start..cells_start + cells]);
        words.extend_from_slice(&gains[cells_start..cells_start + cells]);
        words.extend_from_slice(&headers[header_start..header_start + header_len]);
    }

    words
}

/// Every chunk's frames of `field`, joined in frame order.
fn joined(chunks: &[GrainChunk], field: fn(&GrainChunk) -> &Vec<f32>) -> Vec<f32> {
    chunks.iter().flat_map(|chunk| field(chunk).clone()).collect()
}

fn frame_count(chunks: &[GrainChunk]) -> usize {
    chunks.iter().map(|chunk| chunk.frames as usize).sum()
}

/// Checks every frame's `frame_len` values hold something and differ from the frame before.
fn assert_each_frame_filled(values: &[f32], frame_len: usize, label: &str) {
    let (frames, rest) = values.split_at(values.len() / frame_len * frame_len);
    assert!(rest.is_empty(), "{label} holds whole frames");

    let mut previous: Option<&[f32]> = None;
    for (frame, words) in frames.chunks(frame_len).enumerate() {
        let filled = words.iter().any(|&word| word != 0.0);
        assert!(filled, "{label} of frame {frame} is all zeros");

        let repeats = previous.is_some_and(|previous| previous == words);
        assert!(!repeats, "{label} of frame {frame} repeats the frame before");
        previous = Some(words);
    }
}

#[test]
fn the_documented_frame_lengths_hold() {
    assert_eq!(READING_WORDS, 30);
    assert_eq!(colour_frame_len(1), 1974);
    assert_eq!(FRAME_RECORD_LEN, 24);
    assert_eq!(CHROMA_RECORD_LEN, 6);
}

#[test]
fn a_grain_data_run_keeps_a_noise_reading_per_frame() {
    assert!(FRAMES <= history_capacity(RADIUS) as usize);

    let client = make_client();
    let params = resolve_params(&options(true), ChannelMode::Luma).expect("resolve");
    let mut denoiser = Nl4dDenoiser::new(&client, params, WIDTH, HEIGHT).expect("build");
    let frames = noisy_frames(WIDTH, HEIGHT, 1, FRAMES);
    for frame in &frames {
        denoiser.push_frame(frame);
        denoiser.denoise().expect("denoise");
    }

    denoiser.flush(|_| {}).expect("flush");

    let chunks = denoiser.drain_grain_chunks().expect("drain");
    let noise = denoiser
        .front_for_test()
        .measured_noise()
        .expect("the noise is measured");
    let history = noise.reading_history().expect("a measured state keeps a history");
    let records = read_words(&client, history.records());
    let frame_noise = joined(&chunks, |chunk| &chunk.frame_noise);
    let reading_words = READING_WORDS as usize;

    assert_eq!(frame_count(&chunks), FRAMES);
    assert_eq!(frame_noise.len(), FRAMES * reading_words);
    for frame in 0..FRAMES {
        let slot = frame % history.capacity() as usize;
        let recorded = &records[slot * reading_words..(slot + 1) * reading_words];
        let kept = &frame_noise[frame * reading_words..(frame + 1) * reading_words];
        assert_eq!(kept, recorded, "frame {frame}");
    }
}

#[test]
fn a_pinned_sigma_keeps_no_noise_snapshot() {
    let client = make_client();
    let pinned = Nl4dOptions {
        sigma: Some(4.0 / 255.0),
        ..options(true)
    };
    let params = resolve_params(&pinned, ChannelMode::Luma).expect("resolve");
    let mut denoiser = Nl4dDenoiser::new(&client, params, WIDTH, HEIGHT).expect("build");
    let frames = noisy_frames(WIDTH, HEIGHT, 1, 9);
    for frame in &frames {
        denoiser.push_frame(frame);
        denoiser.denoise().expect("denoise");
    }

    denoiser.flush(|_| {}).expect("flush");

    let chunks = denoiser.drain_grain_chunks().expect("drain");

    assert_eq!(frame_count(&chunks), 9);
    assert!(chunks.iter().all(|chunk| chunk.frame_noise.is_empty()));
    assert!(chunks.iter().all(|chunk| chunk.frame_colour.is_empty()));
    assert!(chunks.iter().all(|chunk| !chunk.frame_tables.is_empty()));
}

#[test]
fn a_grain_data_run_keeps_the_colour_fit_per_frame() {
    let mut run = ColourRun::new(options(true), ChannelMode::Luma);
    let chunks = run.drive();
    let frame_colour = joined(&chunks, |chunk| &chunk.frame_colour);
    let frame_len = colour_frame_len(1);

    assert_eq!(frame_count(&chunks), FRAMES);
    assert_eq!(frame_colour.len(), FRAMES * frame_len);
    assert_each_frame_filled(&frame_colour, frame_len, "the colour fit");
    assert_eq!(run.observed.len(), OBSERVED_FRAMES);

    let mut fits_differ = false;
    let mut previous: Option<&Vec<f32>> = None;
    let mut observed: Vec<_> = run.observed.iter().collect();
    observed.sort_by_key(|(frame, _)| **frame);
    for (&frame, words) in observed {
        let start = frame as usize * frame_len;
        let kept = &frame_colour[start..start + frame_len];
        assert_eq!(kept, words.as_slice(), "frame {frame}");

        fits_differ |= previous.is_some_and(|previous| previous != words);
        previous = Some(words);
    }

    assert!(fits_differ, "the fit should change between frames");
}

#[test]
fn a_run_without_the_colour_model_keeps_no_colour_snapshot() {
    let mut run = ColourRun::new(options(false), ChannelMode::Luma);
    let chunks = run.drive();
    let reading_words = READING_WORDS as usize;

    assert_eq!(frame_count(&chunks), FRAMES);
    for chunk in &chunks {
        let frames = chunk.frames as usize;
        assert!(chunk.frame_colour.is_empty());
        assert_eq!(chunk.frame_noise.len(), frames * reading_words);
        assert!(!chunk.frame_tables.is_empty());
        assert!(!chunk.frame_records.is_empty());
    }
}

#[test]
fn a_chroma_pass_keeps_the_reading_and_both_planes_fit_per_frame() {
    let mut run = ColourRun::new(options(true), ChannelMode::Chroma);
    let chunks = run.drive();
    let frame_colour = joined(&chunks, |chunk| &chunk.frame_colour);
    let frame_noise = joined(&chunks, |chunk| &chunk.frame_noise);
    let colour_len = colour_frame_len(2);
    let reading_words = READING_WORDS as usize;

    assert_eq!(colour_len, 3948);
    assert_eq!(frame_count(&chunks), FRAMES);
    assert_eq!(frame_colour.len(), FRAMES * colour_len);
    assert_eq!(frame_noise.len(), FRAMES * reading_words);
    assert_each_frame_filled(&frame_colour, colour_len, "the colour fit");
    assert_each_frame_filled(&frame_noise, reading_words, "the noise reading");
    assert!(chunks.iter().all(|chunk| chunk.frame_tables.is_empty()));
}

mod order;
mod roundtrip;
mod synthesize;

use std::fs::File;
use std::io::Write;

use super::{AvdsFrame, AvdsMeta, PlanesMeasured, write_avds};
use crate::nl4d::grain::consts::{
    CHROMA_RECORD_LEN,
    CHUNK_FRAMES,
    FRAME_RECORD_LEN,
    FRAME_TABLE_LEN,
    HIST_LEN,
};
use crate::nl4d::grain::snapshot::colour_frame_len;
use crate::noise::READING_WORDS;
use crate::{ChromaLayout, ChromaSiting, GrainChunk, SceneGrain};

/// Which groups a test scene measures.
#[derive(Debug, Clone, Copy)]
pub(super) struct Measured {
    pub persistence: bool,
    pub noise: bool,
    pub colour: bool,
    pub chroma: bool,
}

impl Measured {
    pub(super) const ALL: Measured = Measured {
        persistence: true,
        noise: true,
        colour: true,
        chroma: true,
    };

    pub(super) const TABLE_ONLY: Measured = Measured {
        persistence: false,
        noise: false,
        colour: false,
        chroma: false,
    };
}

pub(super) fn sample_meta(frames: u64) -> AvdsMeta {
    AvdsMeta {
        writer: "av-denoise test".to_owned(),
        frame_rate: (24000, 1001),
        width: 64,
        height: 48,
        bit_depth: 10,
        subsampling: ChromaLayout::Yuv420,
        chroma_siting: ChromaSiting::Left,
        frames,
        planes_measured: PlanesMeasured::LumaChroma,
    }
}

/// A scene of `frames` frames from `first_frame`, chunked as the export chunks it.
///
/// Every value differs from its neighbours, and histogram counts are whole numbers.
pub(super) fn scene(first_frame: u64, frames: u32, measured: Measured) -> SceneGrain {
    let mut chunks = Vec::new();
    let mut chroma_chunks = Vec::new();
    let mut chunk_first = first_frame;
    let mut remaining = frames;
    while remaining > 0 {
        let chunk_frames = remaining.min(CHUNK_FRAMES);
        let luma = luma_chunk(chunk_first, chunk_frames, measured);
        chunks.push(luma);

        if measured.chroma {
            let chroma = chroma_chunk(chunk_first, chunk_frames);
            chroma_chunks.push(chroma);
        }

        chunk_first += chunk_frames as u64;
        remaining -= chunk_frames;
    }

    SceneGrain {
        first_frame,
        chunks,
        chroma_chunks,
    }
}

/// Scenes of the given lengths, one after another from frame 0.
pub(super) fn scenes_of(lengths: &[u32], measured: Measured) -> Vec<SceneGrain> {
    let mut scenes = Vec::new();
    let mut first_frame = 0;
    for &length in lengths {
        let scene = scene(first_frame, length, measured);
        scenes.push(scene);
        first_frame += length as u64;
    }

    scenes
}

fn luma_chunk(first_frame: u64, frames: u32, measured: Measured) -> GrainChunk {
    let mut chunk = GrainChunk::empty();
    chunk.frames = frames;
    chunk.frame_tables = table_values(first_frame, frames);

    if measured.persistence {
        chunk.frame_records = double_values(first_frame, frames, FRAME_RECORD_LEN, 1.0);
    }

    if measured.noise {
        chunk.frame_noise = float_values(first_frame, frames, READING_WORDS as usize, 2.0);
    }

    if measured.colour {
        chunk.frame_colour = float_values(first_frame, frames, colour_frame_len(1), 3.0);
    }

    chunk
}

fn chroma_chunk(first_frame: u64, frames: u32) -> GrainChunk {
    let mut chunk = GrainChunk::empty();
    chunk.frames = frames;
    chunk.frame_records = double_values(first_frame, frames, CHROMA_RECORD_LEN, 4.0);
    chunk.frame_noise = float_values(first_frame, frames, READING_WORDS as usize, 5.0);
    chunk.frame_colour = float_values(first_frame, frames, colour_frame_len(2), 6.0);

    chunk
}

fn table_values(first_frame: u64, frames: u32) -> Vec<f32> {
    let mut values = Vec::with_capacity(frames as usize * FRAME_TABLE_LEN);
    for frame in first_frame..first_frame + frames as u64 {
        for index in 0..FRAME_TABLE_LEN {
            let value = if index < 2 * HIST_LEN {
                ((frame as usize * 7 + index) % 23) as f32
            } else {
                frame as f32 * 0.375 + index as f32 * 0.001
            };
            values.push(value);
        }
    }

    values
}

fn float_values(first_frame: u64, frames: u32, frame_len: usize, offset: f32) -> Vec<f32> {
    let mut values = Vec::with_capacity(frames as usize * frame_len);
    for frame in first_frame..first_frame + frames as u64 {
        for index in 0..frame_len {
            let value = offset + frame as f32 * 0.125 - index as f32 * 0.003;
            values.push(value);
        }
    }

    values
}

fn double_values(first_frame: u64, frames: u32, frame_len: usize, offset: f64) -> Vec<f64> {
    let mut values = Vec::with_capacity(frames as usize * frame_len);
    for frame in first_frame..first_frame + frames as u64 {
        for index in 0..frame_len {
            let value = offset + frame as f64 * 0.1 + index as f64 / 3.0;
            values.push(value);
        }
    }

    values
}

/// `bytes` in a temporary file a reader can open.
pub(super) fn file_of(bytes: &[u8]) -> File {
    let mut file = tempfile::tempfile().expect("a temporary file");
    file.write_all(bytes).expect("the bytes are written");

    file
}

/// A whole file of `rows` with `meta`, ready to read.
pub(super) fn written_file(meta: &AvdsMeta, rows: &[AvdsFrame]) -> File {
    let mut bytes = Vec::new();
    write_avds(&mut bytes, meta, rows).expect("the file is written");

    file_of(&bytes)
}

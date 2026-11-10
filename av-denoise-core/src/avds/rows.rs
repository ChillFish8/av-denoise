use super::AvdsFrame;
use super::schema::AvdsGroup;
use crate::nl4d::grain::consts::CHUNK_FRAMES;
use crate::{GrainChunk, SceneGrain};

/// The rows of every scene, in frame order.
pub fn rows_from_scenes(scenes: &[SceneGrain]) -> Vec<AvdsFrame> {
    let mut ordered: Vec<&SceneGrain> = scenes.iter().collect();
    ordered.sort_by_key(|scene| scene.first_frame);

    let mut rows = Vec::new();
    for (scene_index, scene) in ordered.into_iter().enumerate() {
        let scene_rows = scene_rows(scene, scene_index as u64);
        rows.extend(scene_rows);
    }

    rows
}

/// The rows of one scene, which is the `scene_index`th scene in frame order.
///
/// A scene has one row per frame its luma chunks cover. Chroma values fill the same rows in turn.
///
/// Panics when a chunk field does not hold every frame of its chunk, or when the chroma chunks
/// cover a different number of frames than the luma chunks.
pub(super) fn scene_rows(scene: &SceneGrain, scene_index: u64) -> Vec<AvdsFrame> {
    let luma_frames: u32 = scene.chunks.iter().map(|chunk| chunk.frames).sum();
    let chroma_frames: u32 = scene.chroma_chunks.iter().map(|chunk| chunk.frames).sum();
    if !scene.chroma_chunks.is_empty() {
        let first_frame = scene.first_frame;
        if chroma_frames != luma_frames {
            panic!("scene at frame {first_frame} has {chroma_frames} chroma and {luma_frames} luma frames");
        }
    }

    for chunk in &scene.chunks {
        check_field(
            "frame_tables",
            &chunk.frame_tables,
            chunk.frames,
            AvdsGroup::Table,
        );
        check_field(
            "frame_records",
            &chunk.frame_records,
            chunk.frames,
            AvdsGroup::Persistence,
        );
        check_field("frame_noise", &chunk.frame_noise, chunk.frames, AvdsGroup::Noise);
        check_field(
            "frame_colour",
            &chunk.frame_colour,
            chunk.frames,
            AvdsGroup::Colour,
        );
    }

    for chunk in &scene.chroma_chunks {
        check_field(
            "chroma frame_records",
            &chunk.frame_records,
            chunk.frames,
            AvdsGroup::ChromaPersistence,
        );
        check_field(
            "chroma frame_noise",
            &chunk.frame_noise,
            chunk.frames,
            AvdsGroup::ChromaNoise,
        );
        check_field(
            "chroma frame_colour",
            &chunk.frame_colour,
            chunk.frames,
            AvdsGroup::ChromaColour,
        );
    }

    let mut rows = Vec::with_capacity(luma_frames as usize);
    for chunk in &scene.chunks {
        for slot in 0..chunk.frames as usize {
            let frame = scene.first_frame + rows.len() as u64;
            let first_of_scene = rows.is_empty();
            let table = frame_values(&chunk.frame_tables, slot, AvdsGroup::Table);
            let persistence = frame_values(&chunk.frame_records, slot, AvdsGroup::Persistence);
            let noise = frame_values(&chunk.frame_noise, slot, AvdsGroup::Noise);
            let colour = frame_values(&chunk.frame_colour, slot, AvdsGroup::Colour);

            let row = AvdsFrame {
                frame,
                scene: scene_index,
                first_of_scene,
                table,
                persistence,
                chroma_persistence: None,
                noise,
                chroma_noise: None,
                colour,
                chroma_colour: None,
            };
            rows.push(row);
        }
    }

    let chroma_slots = scene
        .chroma_chunks
        .iter()
        .flat_map(|chunk| (0..chunk.frames as usize).map(move |slot| (chunk, slot)));
    for (row, (chunk, slot)) in rows.iter_mut().zip(chroma_slots) {
        row.chroma_persistence = frame_values(&chunk.frame_records, slot, AvdsGroup::ChromaPersistence);
        row.chroma_noise = frame_values(&chunk.frame_noise, slot, AvdsGroup::ChromaNoise);
        row.chroma_colour = frame_values(&chunk.frame_colour, slot, AvdsGroup::ChromaColour);
    }

    rows
}

/// Panics unless the chunk field `name` is empty or holds `frames` frames of `group`.
fn check_field<T>(name: &str, values: &[T], frames: u32, group: AvdsGroup) {
    if values.is_empty() {
        return;
    }

    let expected = frames as usize * group.len();
    let len = values.len();
    if len != expected {
        panic!("chunk field {name} holds {len} values, {frames} frames need {expected}");
    }
}

/// Slot `slot` of a chunk field of `group`, `None` when the field is empty.
fn frame_values<T: Copy>(values: &[T], slot: usize, group: AvdsGroup) -> Option<Vec<T>> {
    if values.is_empty() {
        return None;
    }

    let frame_len = group.len();
    let start = slot * frame_len;
    let frame_values = &values[start..start + frame_len];

    Some(frame_values.to_vec())
}

/// The scenes the rows describe, chunked as the export chunks them.
///
/// A scene starts at each row marked as the first of its scene. Each scene is cut into chunks of
/// up to 24 frames from its first frame. A chunk keeps a group when any of its rows has it, and
/// rows without it hold zeros there.
///
/// Each scene's rows are dropped once its chunks are built.
pub fn scenes_from_rows(rows: Vec<AvdsFrame>) -> Vec<SceneGrain> {
    let mut scenes = Vec::new();
    let mut scene_rows = Vec::new();
    for row in rows {
        if row.first_of_scene && !scene_rows.is_empty() {
            let scene = scene_of(&scene_rows);
            scenes.push(scene);
            scene_rows.clear();
        }

        scene_rows.push(row);
    }

    if !scene_rows.is_empty() {
        let scene = scene_of(&scene_rows);
        scenes.push(scene);
    }

    scenes
}

fn scene_of(rows: &[AvdsFrame]) -> SceneGrain {
    let has_chroma = rows.iter().any(|row| {
        row.chroma_persistence.is_some() || row.chroma_noise.is_some() || row.chroma_colour.is_some()
    });

    let mut chunks = Vec::new();
    let mut chroma_chunks = Vec::new();
    for chunk_rows in rows.chunks(CHUNK_FRAMES as usize) {
        let chunk = luma_chunk(chunk_rows);
        chunks.push(chunk);

        if has_chroma {
            let chroma_chunk = chroma_chunk(chunk_rows);
            chroma_chunks.push(chroma_chunk);
        }
    }

    SceneGrain {
        first_frame: rows[0].frame,
        chunks,
        chroma_chunks,
    }
}

fn luma_chunk(rows: &[AvdsFrame]) -> GrainChunk {
    let frame_tables = joined(rows, AvdsGroup::Table, AvdsFrame::floats);
    let frame_records = joined(rows, AvdsGroup::Persistence, AvdsFrame::doubles);
    let frame_noise = joined(rows, AvdsGroup::Noise, AvdsFrame::floats);
    let frame_colour = joined(rows, AvdsGroup::Colour, AvdsFrame::floats);

    GrainChunk {
        frames: rows.len() as u32,
        frame_tables,
        frame_records,
        frame_noise,
        frame_colour,
    }
}

fn chroma_chunk(rows: &[AvdsFrame]) -> GrainChunk {
    let frame_records = joined(rows, AvdsGroup::ChromaPersistence, AvdsFrame::doubles);
    let frame_noise = joined(rows, AvdsGroup::ChromaNoise, AvdsFrame::floats);
    let frame_colour = joined(rows, AvdsGroup::ChromaColour, AvdsFrame::floats);

    GrainChunk {
        frames: rows.len() as u32,
        frame_tables: Vec::new(),
        frame_records,
        frame_noise,
        frame_colour,
    }
}

/// Every row's values of `group` in turn, empty when no row has the group.
fn joined<T: Copy + Default>(
    rows: &[AvdsFrame],
    group: AvdsGroup,
    values_of: impl Fn(&AvdsFrame, AvdsGroup) -> Option<&[T]>,
) -> Vec<T> {
    let measured = rows.iter().any(|row| values_of(row, group).is_some());
    if !measured {
        return Vec::new();
    }

    let frame_len = group.len();
    let mut joined = Vec::with_capacity(rows.len() * frame_len);
    for row in rows {
        match values_of(row, group) {
            Some(values) => joined.extend_from_slice(values),
            None => joined.resize(joined.len() + frame_len, T::default()),
        }
    }

    joined
}

use super::chunk::GrainChunk;
use super::consts::{
    CHROMA_LAGS,
    CHROMA_PLANES,
    CHROMA_RECORD_LEN,
    FRAME_RECORD_LEN,
    LAG_RECORD_LEN,
    MIN_FRAME_CELLS,
    PERSISTENCE_LAGS,
};
use super::segment::SceneGrain;

/// The lag a chroma strength reads, the first part of each plane's record.
const STRENGTH_LAG: usize = 1;

/// One frame's grain strength and persistence.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct FrameGrain {
    /// The output index of the frame.
    pub frame: u64,
    /// The position of the frame's scene, counting scenes in frame order.
    pub scene: u64,
    /// How many 8x8 cells the reading rests on, counting the pairs on both sides of the frame.
    pub cells: u32,
    /// The grain std in 8-bit code values, `None` with too few cells or no grain.
    pub strength: Option<f64>,
    /// The Cb grain std in 8-bit code values, `None` with too few cells or no chroma reading.
    pub strength_cb: Option<f64>,
    /// The Cr grain std in 8-bit code values, `None` with too few cells or no chroma reading.
    pub strength_cr: Option<f64>,
    /// The share of the grain unchanged from 1 to 8 frames away, each between 0 and 1.
    ///
    /// Each entry pools the frame that far before and after the frame.
    ///
    /// An entry is `None` when its lag has too few cells or no grain.
    pub persistence: [Option<f64>; PERSISTENCE_LAGS],
}

/// One row per frame of every scene, in frame order.
pub fn frame_rows(scenes: &[SceneGrain]) -> Vec<FrameGrain> {
    let mut ordered: Vec<&SceneGrain> = scenes.iter().collect();
    ordered.sort_by_key(|scene| scene.first_frame);

    let mut rows = Vec::new();
    for (scene_index, scene) in ordered.into_iter().enumerate() {
        let records = joined_records(&scene.chunks, FRAME_RECORD_LEN);
        let chroma_records = joined_records(&scene.chroma_chunks, CHROMA_RECORD_LEN);
        let frame_count = records.len() / FRAME_RECORD_LEN;
        for index in 0..frame_count {
            let sums = window_sums(&records, index);
            let chroma_strengths = chroma_strength(&chroma_records, index);
            let frame = scene.first_frame + index as u64;
            let row = row_of(frame, scene_index as u64, &sums, chroma_strengths);
            rows.push(row);
        }
    }

    rows
}

/// The chunks' frame records joined in frame order, with zeros for any that are missing.
fn joined_records(chunks: &[GrainChunk], record_len: usize) -> Vec<f64> {
    let total_frames: usize = chunks.iter().map(|chunk| chunk.frames as usize).sum();
    let mut records = vec![0.0; total_frames * record_len];

    let mut offset = 0;
    for chunk in chunks {
        let chunk_len = chunk.frames as usize * record_len;
        let available = chunk.frame_records.len().min(chunk_len);
        let target = &mut records[offset..offset + available];
        target.copy_from_slice(&chunk.frame_records[..available]);
        offset += chunk_len;
    }

    records
}

/// The cell count, auto sum and cross sum of each lag, pooled over the pairs behind and ahead.
fn window_sums(records: &[f64], index: usize) -> [[f64; LAG_RECORD_LEN]; PERSISTENCE_LAGS] {
    let mut sums = [[0.0; LAG_RECORD_LEN]; PERSISTENCE_LAGS];
    for (lag_index, lag_sums) in sums.iter_mut().enumerate() {
        let lag = lag_index + 1;
        let part_offset = lag_index * LAG_RECORD_LEN;
        *lag_sums = pooled_sums(records, FRAME_RECORD_LEN, index, part_offset, lag);
    }

    sums
}

/// The Cb and Cr strength of frame `index`, pooled over the pairs behind and ahead.
fn chroma_strength(chroma_records: &[f64], index: usize) -> [Option<f64>; CHROMA_PLANES] {
    let mut strengths = [None; CHROMA_PLANES];
    for (plane, strength) in strengths.iter_mut().enumerate() {
        let part_offset = plane * CHROMA_LAGS * LAG_RECORD_LEN;
        let sums = pooled_sums(
            chroma_records,
            CHROMA_RECORD_LEN,
            index,
            part_offset,
            STRENGTH_LAG,
        );
        *strength = strength_of(&sums);
    }

    strengths
}

/// The `[cells, auto, cross]` part at `part_offset` of frame `index`'s record, plus the same part
/// of the record `lag` frames ahead when it exists.
///
/// All zeros when the records end before frame `index`.
fn pooled_sums(
    records: &[f64],
    record_len: usize,
    index: usize,
    part_offset: usize,
    lag: usize,
) -> [f64; LAG_RECORD_LEN] {
    let frame_count = records.len() / record_len;
    let mut sums = [0.0; LAG_RECORD_LEN];
    if index >= frame_count {
        return sums;
    }

    let behind_start = index * record_len + part_offset;
    let behind = &records[behind_start..behind_start + LAG_RECORD_LEN];
    for (sum, value) in sums.iter_mut().zip(behind) {
        *sum += value;
    }

    if index + lag >= frame_count {
        return sums;
    }

    let ahead_start = (index + lag) * record_len + part_offset;
    let ahead = &records[ahead_start..ahead_start + LAG_RECORD_LEN];
    for (sum, value) in sums.iter_mut().zip(ahead) {
        *sum += value;
    }

    sums
}

fn row_of(
    frame: u64,
    scene: u64,
    sums: &[[f64; LAG_RECORD_LEN]; PERSISTENCE_LAGS],
    chroma_strengths: [Option<f64>; CHROMA_PLANES],
) -> FrameGrain {
    let mut persistence = [None; PERSISTENCE_LAGS];
    for (reading, lag_sums) in persistence.iter_mut().zip(sums) {
        let [cells, auto_sum, cross_sum] = *lag_sums;
        if readable(cells, auto_sum) {
            let share = share_kept(auto_sum, cross_sum);
            *reading = Some(share);
        }
    }

    let [cells, _, _] = sums[0];
    let strength = strength_of(&sums[0]);
    let [strength_cb, strength_cr] = chroma_strengths;

    FrameGrain {
        frame,
        scene,
        cells: cells.round() as u32,
        strength,
        strength_cb,
        strength_cr,
        persistence,
    }
}

/// The grain std in 8-bit code values of pooled lag-1 sums, `None` when they are not readable.
fn strength_of(sums: &[f64; LAG_RECORD_LEN]) -> Option<f64> {
    let [cells, auto_sum, _] = *sums;
    let lag_one_readable = readable(cells, auto_sum);

    lag_one_readable.then(|| (auto_sum / cells).sqrt() * 255.0)
}

fn readable(cells: f64, auto_sum: f64) -> bool {
    cells >= MIN_FRAME_CELLS && auto_sum > 0.0
}

fn share_kept(auto_sum: f64, cross_sum: f64) -> f64 {
    (cross_sum.max(0.0) / auto_sum).min(1.0)
}

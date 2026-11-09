use super::consts::{FRAME_RECORD_LEN, LAG_RECORD_LEN, MIN_FRAME_CELLS, PERSISTENCE_LAGS};
use super::segment::SceneGrain;

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
        let records = scene_records(scene);
        let frame_count = records.len() / FRAME_RECORD_LEN;
        for index in 0..frame_count {
            let sums = window_sums(&records, index);
            let frame = scene.first_frame + index as u64;
            let row = row_of(frame, scene_index as u64, &sums);
            rows.push(row);
        }
    }

    rows
}

/// The scene's frame records joined in frame order, with zeros for any that are missing.
fn scene_records(scene: &SceneGrain) -> Vec<f64> {
    let total_frames: usize = scene.chunks.iter().map(|chunk| chunk.frames as usize).sum();
    let mut records = vec![0.0; total_frames * FRAME_RECORD_LEN];

    let mut offset = 0;
    for chunk in &scene.chunks {
        let chunk_len = chunk.frames as usize * FRAME_RECORD_LEN;
        let available = chunk.frame_records.len().min(chunk_len);
        let target = &mut records[offset..offset + available];
        target.copy_from_slice(&chunk.frame_records[..available]);
        offset += chunk_len;
    }

    records
}

/// The cell count, auto sum and cross sum of each lag, pooled over the pairs behind and ahead.
fn window_sums(records: &[f64], index: usize) -> [[f64; LAG_RECORD_LEN]; PERSISTENCE_LAGS] {
    let frame_count = records.len() / FRAME_RECORD_LEN;
    let mut sums = [[0.0; LAG_RECORD_LEN]; PERSISTENCE_LAGS];
    for (lag_index, lag_sums) in sums.iter_mut().enumerate() {
        let lag = lag_index + 1;
        let behind_start = index * FRAME_RECORD_LEN + lag_index * LAG_RECORD_LEN;
        let behind = &records[behind_start..behind_start + LAG_RECORD_LEN];
        for (sum, value) in lag_sums.iter_mut().zip(behind) {
            *sum += value;
        }

        if index + lag >= frame_count {
            continue;
        }

        let ahead_start = (index + lag) * FRAME_RECORD_LEN + lag_index * LAG_RECORD_LEN;
        let ahead = &records[ahead_start..ahead_start + LAG_RECORD_LEN];
        for (sum, value) in lag_sums.iter_mut().zip(ahead) {
            *sum += value;
        }
    }

    sums
}

fn row_of(frame: u64, scene: u64, sums: &[[f64; LAG_RECORD_LEN]; PERSISTENCE_LAGS]) -> FrameGrain {
    let mut persistence = [None; PERSISTENCE_LAGS];
    for (reading, lag_sums) in persistence.iter_mut().zip(sums) {
        let [cells, auto_sum, cross_sum] = *lag_sums;
        if readable(cells, auto_sum) {
            let share = share_kept(auto_sum, cross_sum);
            *reading = Some(share);
        }
    }

    let [cells, auto_sum, _] = sums[0];
    let lag_one_readable = readable(cells, auto_sum);
    let strength = lag_one_readable.then(|| (auto_sum / cells).sqrt() * 255.0);

    FrameGrain {
        frame,
        scene,
        cells: cells.round() as u32,
        strength,
        persistence,
    }
}

fn readable(cells: f64, auto_sum: f64) -> bool {
    cells >= MIN_FRAME_CELLS && auto_sum > 0.0
}

fn share_kept(auto_sum: f64, cross_sum: f64) -> f64 {
    (cross_sum.max(0.0) / auto_sum).min(1.0)
}

use std::collections::BTreeMap;
use std::io::Write;

use super::rows::scene_rows;
use super::write::AvdsWriter;
use crate::SceneGrain;

/// Writes scenes that finish in any order to a grain data file in frame order.
///
/// A scene waits only until every earlier frame is written. Its rows are then written and the
/// scene is dropped, so the scenes held are only those waiting for an earlier one.
pub struct AvdsSceneWriter<W: Write + Send> {
    writer: AvdsWriter<W>,
    /// Scenes waiting for an earlier frame, by their first frame.
    held: BTreeMap<u64, SceneGrain>,
    /// Scenes written so far.
    scenes_written: u64,
}

impl<W: Write + Send> AvdsSceneWriter<W> {
    pub fn new(writer: AvdsWriter<W>) -> Self {
        Self {
            writer,
            held: BTreeMap::new(),
            scenes_written: 0,
        }
    }

    /// Writes `scene` and every held scene that follows it, or holds it until its turn.
    ///
    /// Refuses a scene with no frames, or one whose frames overlap frames already written or held.
    pub fn push(&mut self, scene: SceneGrain) -> Result<(), anyhow::Error> {
        let first_frame = scene.first_frame;
        let frames = scene_frames(&scene);
        let end_frame = first_frame + frames;
        if frames == 0 {
            anyhow::bail!("the scene at frame {first_frame} holds no frames");
        }

        let written = self.writer.frames_written();
        if first_frame < written {
            anyhow::bail!("the scene at frame {first_frame} overlaps the {written} frames already written");
        }

        for (&held_first, held) in &self.held {
            let held_frames = scene_frames(held);
            let held_end = held_first + held_frames;
            if first_frame < held_end && held_first < end_frame {
                anyhow::bail!("scene at frame {first_frame} overlaps the held scene at frame {held_first}");
            }
        }

        self.held.insert(first_frame, scene);

        loop {
            let next_frame = self.writer.frames_written();
            let Some(next) = self.held.remove(&next_frame) else {
                break;
            };

            let rows = scene_rows(&next, self.scenes_written);
            self.writer.write_rows(&rows)?;
            self.scenes_written += 1;
        }

        Ok(())
    }

    /// Scenes waiting for an earlier frame.
    #[cfg(test)]
    pub(crate) fn held_scenes(&self) -> usize {
        self.held.len()
    }

    pub fn frames_written(&self) -> u64 {
        self.writer.frames_written()
    }

    /// Finishes the file, refusing when a scene still waits for frames no scene covered.
    pub fn finish(self) -> Result<W, anyhow::Error> {
        if let Some(&first_held) = self.held.keys().next() {
            // A held scene always starts past the frames written, else it would have been written.
            let written = self.writer.frames_written();
            let missing = missing_range(written, first_held);
            anyhow::bail!("no scene covers {missing}, so the grain data has a gap");
        }

        self.writer.finish()
    }
}

/// The frames a scene's luma chunks cover.
fn scene_frames(scene: &SceneGrain) -> u64 {
    scene.chunks.iter().map(|chunk| chunk.frames as u64).sum()
}

/// The frames from `first` up to but not including `end`, which is past `first`.
fn missing_range(first: u64, end: u64) -> String {
    let last = end - 1;
    if first == last {
        return format!("frame {first}");
    }

    format!("frames {first} to {last}")
}

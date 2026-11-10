use std::fs::File;
use std::path::Path;

use av_denoise::SceneGrain;
use av_denoise::avds::{AvdsMeta, AvdsSceneWriter, AvdsWriter};

use super::staged_file::StagedFile;

/// The grain data file, written beside its final path until the run succeeds.
pub struct AvdsPath {
    /// The temporary file the rows are written to.
    file: StagedFile,
}

/// A grain data file being written scene by scene.
pub struct AvdsStream {
    /// The writer that puts scenes in frame order, declared first so it drops before `file`.
    writer: AvdsSceneWriter<File>,
    /// The temporary file the rows are written to.
    file: StagedFile,
}

impl AvdsPath {
    /// Creates the temporary file beside `path`.
    pub fn create(path: &Path) -> Result<Self, anyhow::Error> {
        let file = StagedFile::create(path, "grain data file")?;

        Ok(Self { file })
    }

    /// Starts writing the file with the metadata of `meta`.
    pub fn stream(self, meta: &AvdsMeta) -> Result<AvdsStream, anyhow::Error> {
        let handle = self.file.writer()?;
        let rows = AvdsWriter::new(handle, meta)?;
        let writer = AvdsSceneWriter::new(rows);

        Ok(AvdsStream {
            writer,
            file: self.file,
        })
    }
}

impl AvdsStream {
    /// Writes the scene once every earlier frame is written, holding it until then.
    pub fn push(&mut self, scene: SceneGrain) -> Result<(), anyhow::Error> {
        self.writer.push(scene)
    }

    /// Finishes the file and moves it into place, once it covers all `frames` frames of the run.
    pub fn finish(self, frames: u64) -> Result<(), anyhow::Error> {
        let written = self.writer.frames_written();
        if written != frames {
            anyhow::bail!("the grain data file covers {written} frames but the run wrote {frames}");
        }

        let handle = self.writer.finish()?;

        self.file.finish(&handle)
    }
}

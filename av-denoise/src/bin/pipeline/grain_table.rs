use std::path::Path;

use av_denoise::{SceneGrain, build_table};

use super::staged_file::StagedFile;

/// The AV1 film grain table, written beside its final path until the run succeeds.
pub struct TablePath {
    /// The temporary file the table is written to.
    file: StagedFile,
}

impl TablePath {
    /// Creates the temporary file beside `path`.
    pub fn create(path: &Path) -> Result<Self, anyhow::Error> {
        let file = StagedFile::create(path, "grain table")?;

        Ok(Self { file })
    }

    /// Writes the table and moves it into place.
    pub fn write(self, scenes: &[SceneGrain], frame_rate: (u64, u64)) -> Result<(), anyhow::Error> {
        let text = build_table(scenes, frame_rate);
        if text == "filmgrn1\n" {
            tracing::warn!("no grain could be measured, writing an empty grain table");
        }

        self.file.commit(text.as_bytes())
    }
}

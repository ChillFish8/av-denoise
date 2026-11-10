use std::path::Path;

use anyhow::Context;
use av_denoise::{FrameGrain, SceneGrain, frame_rows};

use super::staged_file::StagedFile;

/// The per-frame grain report, written beside its final path until the run succeeds.
pub struct ReportPath {
    /// The temporary file the report is written to.
    file: StagedFile,
}

impl ReportPath {
    /// Creates the temporary file beside `path`.
    pub fn create(path: &Path) -> Result<Self, anyhow::Error> {
        let file = StagedFile::create(path, "grain report")?;

        Ok(Self { file })
    }

    /// Writes one JSON line per frame and moves the report into place.
    pub fn write(self, scenes: &[SceneGrain]) -> Result<(), anyhow::Error> {
        let rows = frame_rows(scenes);
        let text = report_text(&rows)?;

        self.file.commit(text.as_bytes())
    }
}

/// The report text, one JSON line per row.
pub fn report_text(rows: &[FrameGrain]) -> Result<String, anyhow::Error> {
    let mut text = String::new();
    for row in rows {
        let line = serde_json::to_string(row).context("unable to encode a grain report row")?;
        text.push_str(&line);
        text.push('\n');
    }

    Ok(text)
}

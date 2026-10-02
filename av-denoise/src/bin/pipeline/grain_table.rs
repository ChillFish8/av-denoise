use std::fs::{File, remove_file, rename};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Context;
use av_denoise::{SceneGrain, build_table};

/// The table's final path and the temporary file written beside it.
///
/// Dropping it before [Self::write] succeeds removes the temporary file.
pub struct TablePath {
    path: PathBuf,
    temporary: PathBuf,
    written: bool,
}

impl TablePath {
    /// Creates the temporary file beside `path`.
    pub fn create(path: &Path) -> Result<Self, anyhow::Error> {
        let mut temporary = path.as_os_str().to_owned();
        temporary.push(".tmp");
        let temporary = PathBuf::from(temporary);

        File::create(&temporary)
            .with_context(|| format!("unable to create the grain table at {}", path.display()))?;

        Ok(Self {
            path: path.to_path_buf(),
            temporary,
            written: false,
        })
    }

    /// Writes the table and moves it into place.
    pub fn write(mut self, scenes: &[SceneGrain], frame_rate: (u64, u64)) -> Result<(), anyhow::Error> {
        let text = build_table(scenes, frame_rate);
        if text == "filmgrn1\n" {
            tracing::warn!("no grain could be measured, writing an empty grain table");
        }

        let mut file = File::create(&self.temporary)
            .with_context(|| format!("unable to write the grain table at {}", self.path.display()))?;
        file.write_all(text.as_bytes())
            .with_context(|| format!("unable to write the grain table at {}", self.path.display()))?;
        file.sync_all()
            .with_context(|| format!("unable to write the grain table at {}", self.path.display()))?;
        rename(&self.temporary, &self.path)
            .with_context(|| format!("unable to move the grain table to {}", self.path.display()))?;

        self.written = true;
        Ok(())
    }
}

impl Drop for TablePath {
    fn drop(&mut self) {
        if !self.written {
            let _ = remove_file(&self.temporary);
        }
    }
}

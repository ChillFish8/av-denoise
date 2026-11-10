use std::fs::{File, remove_file, rename};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Context;

/// A file written beside its final path and moved into place once complete.
pub struct StagedFile {
    /// Where the finished file lands.
    path: PathBuf,
    /// The file written before it moves to `path`.
    temporary: PathBuf,
    /// What the file holds, used in error messages.
    label: &'static str,
    /// Whether the temporary file has moved to `path`.
    written: bool,
}

impl StagedFile {
    /// Creates the empty temporary file beside `path`.
    pub fn create(path: &Path, label: &'static str) -> Result<Self, anyhow::Error> {
        let mut temporary_name = path.as_os_str().to_owned();
        temporary_name.push(".tmp");
        let temporary = PathBuf::from(temporary_name);

        let shown_path = path.display();
        File::create(&temporary).with_context(|| format!("unable to create the {label} at {shown_path}"))?;

        Ok(Self {
            path: path.to_path_buf(),
            temporary,
            label,
            written: false,
        })
    }

    /// Writes `bytes` to the temporary file and moves it to the final path.
    pub fn commit(mut self, bytes: &[u8]) -> Result<(), anyhow::Error> {
        let label = self.label;
        let shown_path = self.path.display();

        let mut file = File::create(&self.temporary)
            .with_context(|| format!("unable to write the {label} at {shown_path}"))?;
        file.write_all(bytes)
            .with_context(|| format!("unable to write the {label} at {shown_path}"))?;
        file.sync_all()
            .with_context(|| format!("unable to write the {label} at {shown_path}"))?;
        rename(&self.temporary, &self.path)
            .with_context(|| format!("unable to move the {label} to {shown_path}"))?;

        self.written = true;

        Ok(())
    }
}

/// Removes the temporary file unless it was moved into place.
impl Drop for StagedFile {
    fn drop(&mut self) {
        if !self.written {
            let _ = remove_file(&self.temporary);
        }
    }
}

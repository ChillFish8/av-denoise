use std::fmt;
use std::io::Read;
use std::path::PathBuf;
use std::str::FromStr;

/// Where frames are read from, as named by `-i`/`--input`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputSource {
    /// This process's standard input, written as `-` or `pipe:0`.
    Stdin,
    /// An inherited file descriptor, written as `pipe:N`. `N` is always
    /// 3 or above because 0, 1, and 2 are handled on their own.
    Fd(u32),
    /// A path on disk.
    File(PathBuf),
}

impl InputSource {
    /// Opens the stream this source names.
    ///
    /// Only the piped variants are readable as a stream. A path is an error.
    pub fn open_reader(&self) -> Result<Box<dyn Read>, anyhow::Error> {
        match self {
            InputSource::Stdin => {
                let stdin = std::io::stdin().lock();
                Ok(Box::new(stdin))
            },
            InputSource::Fd(fd) => open_fd(*fd),
            InputSource::File(path) => anyhow::bail!(
                "`{}` is a file path and is opened with ffms2, not read as a stream",
                path.display(),
            ),
        }
    }
}

impl FromStr for InputSource {
    type Err = String;

    /// Accepts the same input spellings as ffmpeg.
    ///
    /// - `-` and `pipe:0` are standard input
    /// - `pipe:N` for `N` of 3 or above is an inherited descriptor
    /// - anything else is a path on disk
    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        if raw == "-" {
            return Ok(InputSource::Stdin);
        }

        if let Some(rest) = raw.strip_prefix("pipe:") {
            let fd: u32 = rest
                .parse()
                .map_err(|_| format!("pipe: expects a file descriptor number (got `{raw}`)"))?;

            return match fd {
                0 => Ok(InputSource::Stdin),
                1 => Err("pipe:1 is this process's stdout, which carries the denoised y4m".to_string()),
                2 => Err("pipe:2 is this process's stderr, which carries log output".to_string()),
                inherited => Ok(InputSource::Fd(inherited)),
            };
        }

        if raw.is_empty() {
            return Err("expected a file path, `-`, or `pipe:N`".to_string());
        }

        let path = PathBuf::from(raw);
        Ok(InputSource::File(path))
    }
}

impl fmt::Display for InputSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InputSource::Stdin => formatter.write_str("stdin"),
            InputSource::Fd(fd) => write!(formatter, "pipe:{fd}"),
            InputSource::File(path) => write!(formatter, "{}", path.display()),
        }
    }
}

/// Reopens an inherited descriptor through `/dev/fd`.
#[cfg(unix)]
fn open_fd(fd: u32) -> Result<Box<dyn Read>, anyhow::Error> {
    let path = format!("/dev/fd/{fd}");
    let file = std::fs::File::open(&path)
        .map_err(|error| anyhow::anyhow!("--input pipe:{fd} could not open {path}: {error}"))?;

    Ok(Box::new(file))
}

#[cfg(not(unix))]
fn open_fd(fd: u32) -> Result<Box<dyn Read>, anyhow::Error> {
    anyhow::bail!("--input pipe:{fd} needs a Unix platform, use `-` for stdin instead")
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    #[cfg(unix)]
    use std::os::fd::AsRawFd;

    use super::*;

    fn parse(raw: &str) -> Result<InputSource, String> {
        raw.parse()
    }

    #[test]
    fn dash_is_stdin() {
        let source = parse("-").unwrap();

        assert_eq!(source, InputSource::Stdin);
    }

    #[test]
    fn pipe_zero_is_stdin() {
        let source = parse("pipe:0").unwrap();

        assert_eq!(source, InputSource::Stdin);
    }

    #[test]
    fn pipe_three_is_an_inherited_descriptor() {
        let source = parse("pipe:3").unwrap();

        assert_eq!(source, InputSource::Fd(3));
    }

    #[test]
    fn our_own_output_descriptors_are_rejected() {
        let stdout_error = parse("pipe:1").unwrap_err();
        let stderr_error = parse("pipe:2").unwrap_err();

        assert!(stdout_error.contains("stdout"));
        assert!(stderr_error.contains("stderr"));
    }

    #[test]
    fn non_numeric_descriptor_is_rejected() {
        let error = parse("pipe:x").unwrap_err();

        assert!(error.contains("file descriptor"));
    }

    #[test]
    fn anything_else_is_a_path() {
        let plain = parse("noisy.mkv").unwrap();
        let dash = parse("./-").unwrap();
        let pipe_lookalike = parse("./pipe:3").unwrap();

        let plain_path = PathBuf::from("noisy.mkv");
        let dash_path = PathBuf::from("./-");
        let pipe_lookalike_path = PathBuf::from("./pipe:3");

        assert_eq!(plain, InputSource::File(plain_path));
        assert_eq!(dash, InputSource::File(dash_path));
        assert_eq!(pipe_lookalike, InputSource::File(pipe_lookalike_path));
    }

    #[test]
    fn empty_is_rejected() {
        let result = parse("");

        assert!(result.is_err());
    }

    #[test]
    fn display_round_trips_the_typed_spelling() {
        let path = PathBuf::from("noisy.mkv");
        let file = InputSource::File(path);

        assert_eq!(InputSource::Stdin.to_string(), "stdin");
        assert_eq!(InputSource::Fd(3).to_string(), "pipe:3");
        assert_eq!(file.to_string(), "noisy.mkv");
    }

    /// `/dev/fd/N` reopens whatever the descriptor points at, so a temp
    /// file stands in for an inherited pipe.
    #[cfg(unix)]
    #[test]
    fn open_reader_reads_an_inherited_descriptor() {
        let file_name = format!("av-denoise-fd-{}.bin", std::process::id());
        let path = std::env::temp_dir().join(file_name);

        let mut file = std::fs::File::create(&path).expect("temp file should create");
        file.write_all(b"YUV4MPEG2 frames").expect("payload should write");
        file.sync_all().expect("payload should flush");
        drop(file);

        let file = std::fs::File::open(&path).expect("temp file should reopen");
        let fd = file.as_raw_fd() as u32;

        let mut reader = InputSource::Fd(fd)
            .open_reader()
            .expect("inherited descriptor should open");

        let mut read_back = String::new();
        reader
            .read_to_string(&mut read_back)
            .expect("payload should read back");

        drop(reader);
        drop(file);
        let _ = std::fs::remove_file(&path);

        assert_eq!(read_back, "YUV4MPEG2 frames");
    }

    #[test]
    fn open_reader_rejects_a_path() {
        let path = PathBuf::from("noisy.mkv");
        let source = InputSource::File(path);

        // `Box<dyn Read>` isn't `Debug`, so `expect_err` can't be used here.
        let error = match source.open_reader() {
            Ok(_) => panic!("paths are opened with ffms2, not read as a stream"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("noisy.mkv"));
    }
}

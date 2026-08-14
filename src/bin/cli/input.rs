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
    /// Only the piped variants are readable here. A path goes through
    /// the file pipeline, which opens it with ffms2 instead.
    pub fn open_reader(&self) -> Result<Box<dyn Read>, anyhow::Error> {
        match self {
            InputSource::Stdin => Ok(Box::new(std::io::stdin().lock())),
            InputSource::Fd(fd) => open_fd(*fd),
            InputSource::File(path) => anyhow::bail!(
                "`{}` is a file path and is read through the file pipeline, not as a stream",
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
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "-" {
            return Ok(InputSource::Stdin);
        }

        if let Some(rest) = s.strip_prefix("pipe:") {
            let fd: u32 = rest
                .parse()
                .map_err(|_| format!("pipe: expects a file descriptor number (got `{s}`)"))?;

            return match fd {
                0 => Ok(InputSource::Stdin),
                1 => Err("pipe:1 is this process's stdout, which carries the denoised y4m".to_string()),
                2 => Err("pipe:2 is this process's stderr, which carries log output".to_string()),
                n => Ok(InputSource::Fd(n)),
            };
        }

        if s.is_empty() {
            return Err("expected a file path, `-`, or `pipe:N`".to_string());
        }

        Ok(InputSource::File(PathBuf::from(s)))
    }
}

impl fmt::Display for InputSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InputSource::Stdin => f.write_str("stdin"),
            InputSource::Fd(fd) => write!(f, "pipe:{fd}"),
            InputSource::File(path) => write!(f, "{}", path.display()),
        }
    }
}

/// Reopens an inherited descriptor through `/dev/fd`.
#[cfg(unix)]
fn open_fd(fd: u32) -> Result<Box<dyn Read>, anyhow::Error> {
    let path = format!("/dev/fd/{fd}");
    let file = std::fs::File::open(&path)
        .map_err(|e| anyhow::anyhow!("--input pipe:{fd} could not open {path}: {e}"))?;

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

    fn parse(s: &str) -> Result<InputSource, String> {
        s.parse()
    }

    #[test]
    fn dash_is_stdin() {
        assert_eq!(parse("-").unwrap(), InputSource::Stdin);
    }

    #[test]
    fn pipe_zero_is_stdin() {
        assert_eq!(parse("pipe:0").unwrap(), InputSource::Stdin);
    }

    #[test]
    fn pipe_three_is_an_inherited_descriptor() {
        assert_eq!(parse("pipe:3").unwrap(), InputSource::Fd(3));
    }

    #[test]
    fn our_own_output_descriptors_are_rejected() {
        assert!(parse("pipe:1").unwrap_err().contains("stdout"));
        assert!(parse("pipe:2").unwrap_err().contains("stderr"));
    }

    #[test]
    fn non_numeric_descriptor_is_rejected() {
        assert!(parse("pipe:x").unwrap_err().contains("file descriptor"));
    }

    #[test]
    fn anything_else_is_a_path() {
        assert_eq!(
            parse("noisy.mkv").unwrap(),
            InputSource::File(PathBuf::from("noisy.mkv"))
        );
        assert_eq!(parse("./-").unwrap(), InputSource::File(PathBuf::from("./-")));
        assert_eq!(
            parse("./pipe:3").unwrap(),
            InputSource::File(PathBuf::from("./pipe:3"))
        );
    }

    #[test]
    fn empty_is_rejected() {
        assert!(parse("").is_err());
    }

    #[test]
    fn display_round_trips_the_typed_spelling() {
        assert_eq!(InputSource::Stdin.to_string(), "stdin");
        assert_eq!(InputSource::Fd(3).to_string(), "pipe:3");
        assert_eq!(
            InputSource::File(PathBuf::from("noisy.mkv")).to_string(),
            "noisy.mkv"
        );
    }

    /// `/dev/fd/N` reopens whatever the descriptor points at, so a temp
    /// file stands in for the inherited pipe a harness would hand us.
    #[cfg(unix)]
    #[test]
    fn open_reader_reads_an_inherited_descriptor() {
        let path = std::env::temp_dir().join(format!("av-denoise-fd-{}.bin", std::process::id()));

        let mut file = std::fs::File::create(&path).expect("temp file should create");
        file.write_all(b"YUV4MPEG2 frames").expect("payload should write");
        file.sync_all().expect("payload should flush");
        drop(file);

        let file = std::fs::File::open(&path).expect("temp file should reopen");
        let fd = file.as_raw_fd() as u32;

        let mut reader = InputSource::Fd(fd)
            .open_reader()
            .expect("inherited descriptor should open");

        let mut got = String::new();
        reader.read_to_string(&mut got).expect("payload should read back");

        drop(reader);
        drop(file);
        let _ = std::fs::remove_file(&path);

        assert_eq!(got, "YUV4MPEG2 frames");
    }

    #[test]
    fn open_reader_rejects_a_path() {
        // `Box<dyn Read>` isn't `Debug`, so `expect_err` can't be used here.
        let err = match InputSource::File(PathBuf::from("noisy.mkv")).open_reader() {
            Ok(_) => panic!("paths go through the file pipeline"),
            Err(e) => e,
        };

        assert!(err.to_string().contains("noisy.mkv"));
    }
}

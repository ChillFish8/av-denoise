use super::InputSource;

/// The flags every denoising family takes, whatever it does with the
/// frames once it has them.
#[derive(Debug, Clone, clap::Args)]
pub struct CommonArgs {
    /// Where to read frames from.
    ///
    /// A path opens the file with ffms2 and splits the work by
    /// scene. Any container or codec supported by ffmpeg works.
    ///
    /// `-` or `pipe:0` reads a y4m stream from standard input.
    ///
    /// `pipe:N` for `N` of 3 or above reads a y4m stream from an
    /// inherited file descriptor.
    ///
    /// Piped input has no scene detection, so the temporal window
    /// slides across the whole stream.
    ///
    /// A file whose name would otherwise be read as a pipe is
    /// reachable by prefixing it, for example `./-`.
    ///
    /// The source's bit depth is detected automatically. 8, 10, and
    /// 12-bit sources are supported and the output keeps the source's
    /// depth. Other depths are rejected with a clear error message.
    #[arg(short, long)]
    pub input: InputSource,

    /// How many scenes to clean in parallel.
    ///
    /// Each worker uses its own GPU memory for the frame ring
    /// buffer, so higher values trade GPU memory for throughput.
    ///
    /// `1` is valid and useful for debugging. Defaults to 2 when
    /// unset.
    ///
    /// Ignored for piped input, which cannot be split by scene.
    #[arg(short = 'W', long)]
    pub workers: Option<usize>,
}

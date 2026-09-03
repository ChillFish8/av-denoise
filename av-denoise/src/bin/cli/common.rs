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

    /// How much memory frames in flight may occupy.
    ///
    /// A frame counts against this from the moment it is decoded until
    /// it has been written, so the budget covers the staging queues,
    /// each worker's GPU pipeline, and the reordering buffer together.
    ///
    /// Takes a size such as `8GB`, `4.5GiB`, or `512MB`, or a plain
    /// byte count. `GB` is 10^9 while `GiB` is 2^30, and the two differ
    /// by 7%.
    ///
    /// Lower it when several `av-denoise` processes share a machine, as
    /// Av1an chunking does. Raise it when one run has the machine to
    /// itself and the decoder is the bottleneck.
    ///
    /// Defaults to 1GiB when unset.
    ///
    /// Ignored for piped input, which cannot be split by scene.
    #[arg(long, value_name = "SIZE", value_parser = parse_frame_budget)]
    pub frame_budget: Option<u64>,
}

/// Reads a size string such as `8GB` into a byte count.
fn parse_frame_budget(raw: &str) -> Result<u64, String> {
    raw.parse::<bytesize::ByteSize>().map(|size| size.as_u64()).map_err(|_| {
        format!("invalid frame budget '{raw}'. Give a size such as 8GB, 4.5GiB, or 512MB, or a plain byte count")
    })
}

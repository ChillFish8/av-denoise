use std::path::PathBuf;

use av_denoise::accelerate::{Accelerator, get_default_accelerators};
use av_denoise::{DenoisingMode, Device, PrefilterMode};
use clap::{Parser, Subcommand};
use strum_macros::EnumString;

mod file_mode;
mod ingest;
mod stdin_mode;

use ingest::{BinaryChannelIntent, CliOptions};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// The denoising algorithm to use.
///
/// Currently only Non-Local Means is implemented; the flag exists so
/// future algorithms can be added without breaking the CLI surface.
#[derive(Debug, Copy, Clone, Default, EnumString)]
#[strum(ascii_case_insensitive)]
pub enum Algorithm {
    #[default]
    Nlmeans,
}

/// Which channels of each frame should be denoised.
#[derive(Debug, Copy, Clone, PartialEq, Eq, clap::ValueEnum)]
pub enum CliChannelMode {
    /// Denoise only the luma (Y) plane. Chroma is passed through.
    Luma,
    /// Denoise only the chroma (U, V) planes. Luma is passed through.
    Chroma,
    /// Single-pass fused YUV denoising via the library's 3-channel
    /// kernel. Requires a YUV444 source and cannot be combined with
    /// other modes.
    Yuv,
}

fn resolve_channel_intent(modes: &[CliChannelMode]) -> Result<BinaryChannelIntent, anyhow::Error> {
    if modes.is_empty() {
        anyhow::bail!("--channel-mode must contain at least one value");
    }

    let has_yuv = modes.contains(&CliChannelMode::Yuv);
    if has_yuv && modes.len() > 1 {
        anyhow::bail!("--channel-mode `yuv` cannot be combined with other modes");
    }

    let has_luma = modes.contains(&CliChannelMode::Luma);
    let has_chroma = modes.contains(&CliChannelMode::Chroma);
    let luma_count = modes.iter().filter(|m| **m == CliChannelMode::Luma).count();
    let chroma_count = modes.iter().filter(|m| **m == CliChannelMode::Chroma).count();
    let yuv_count = modes.iter().filter(|m| **m == CliChannelMode::Yuv).count();

    if luma_count > 1 || chroma_count > 1 || yuv_count > 1 {
        anyhow::bail!("--channel-mode entries must be unique");
    }

    Ok(match (has_yuv, has_luma, has_chroma) {
        (true, _, _) => BinaryChannelIntent::YuvFused,
        (false, true, true) => BinaryChannelIntent::LumaChroma,
        (false, true, false) => BinaryChannelIntent::Luma,
        (false, false, true) => BinaryChannelIntent::Chroma,
        (false, false, false) => unreachable!("empty list rejected above"),
    })
}

/// Fast and efficient video denoising.
///
/// Reads a video from either a file (with scene-aware parallel
/// denoising) or a y4m stream on stdin, runs NLMeans on the GPU via
/// cubecl, and writes the denoised result as y4m to stdout.
///
/// Pick the input source with a subcommand:
///
///   av-denoise file --input clip.mkv | x265 ...
///   ffmpeg -i clip.mkv -f yuv4mpegpipe - | av-denoise stdin | x265 ...
#[derive(Debug, Parser)]
#[command(about = "Fast and efficient video denoising", long_about = None)]
struct Args {
    /// Denoising algorithm.
    ///
    /// Only `nlmeans` is currently implemented.
    #[arg(short, long, default_value = "nlmeans")]
    algorithm: Algorithm,

    /// Hardware accelerator priority list (comma-delimited).
    ///
    /// The runtime is selected by probing each accelerator in order
    /// and taking the first one that initialises successfully. If
    /// none work, the binary exits with an error.
    ///
    /// Defaults to every backend the binary was compiled with.
    #[arg(short = 'A', long, value_delimiter = ',', default_values_t = get_default_accelerators())]
    accelerators: Vec<Accelerator>,

    /// Specific device to bind to on the selected accelerator.
    ///
    /// Accepted forms:
    ///
    /// `default` — backend-chosen default device.
    ///
    /// `discrete[:N]` — discrete GPU at ordinal N (default 0).
    /// Honoured by CUDA, ROCm, and wgpu.
    ///
    /// `integrated[:N]` — integrated GPU at ordinal N. wgpu only.
    ///
    /// `virtual[:N]` — virtual GPU at ordinal N. wgpu only.
    ///
    /// `cpu` — software/CPU device.
    #[arg(short, long, default_value = "default")]
    device: Device,

    /// Which channels of each frame to denoise (comma-delimited).
    ///
    /// `luma` denoises only Y; `chroma` only U/V at the source's
    /// native subsampled resolution. `luma,chroma` runs both as two
    /// independent denoisers (full-res Y + subsampled UV).
    ///
    /// `yuv` invokes the library's fused 3-channel kernel in one
    /// pass. It requires a YUV444 source and cannot be combined with
    /// any other mode.
    #[arg(long, value_enum, value_delimiter = ',', default_values_t = vec![CliChannelMode::Luma])]
    channel_mode: Vec<CliChannelMode>,

    /// Reference clip used for NLM weight calculation.
    ///
    /// `none` disables prefiltering and uses the noisy input directly
    /// for both weight calculation and pixel accumulation.
    ///
    /// `bilateral:<sigma_s>,<sigma_r>` runs an on-GPU bilateral
    /// prefilter; `sigma_s` is the spatial sigma in pixels and
    /// `sigma_r` is the range sigma in `[0, 1]` intensity units.
    /// A sensible starting point is `bilateral:3.0,0.02`.
    #[arg(long, default_value = "none")]
    prefilter: String,

    /// Temporal radius for temporal-aware denoising.
    ///
    /// `0` (default) runs spatial-only denoising — each output frame
    /// depends only on the matching input frame. Values `> 0` enable
    /// temporal denoising over a `2 * radius + 1` frame window
    /// centred on the current frame; higher values give stronger
    /// noise reduction at the cost of latency and memory.
    ///
    /// In `file` mode, temporal context is reset at every scene
    /// boundary detected by av-scenechange, so increasing the radius
    /// never blends frames across cuts.
    #[arg(long, default_value_t = 0)]
    temporal_radius: u32,

    #[command(subcommand)]
    command: Command,
}

/// Input-source selector.
#[derive(Debug, Subcommand)]
enum Command {
    /// Denoise a video file with scene-aware parallel processing.
    ///
    /// Opens the input with ffms2, runs scene detection
    /// (av-scenechange), and dispatches scenes across N worker
    /// threads. Each worker holds its own `Denoiser` and rebuilds it
    /// at every scene boundary, so temporal context never leaks
    /// across cuts. Workers emit frames to a coordinator that
    /// re-orders them and writes y4m to stdout.
    File {
        /// Path to the input video file.
        ///
        /// Any container/codec supported by ffmpeg (and therefore
        /// ffms2) is accepted. The source must be 8-bit; 10/12-bit
        /// inputs are rejected with a clear error.
        #[arg(short, long)]
        input: PathBuf,

        /// Number of denoiser workers running in parallel.
        ///
        /// Each worker holds its own `Denoiser` and processes one
        /// scene at a time. Higher values increase GPU utilisation on
        /// content with many short scenes, at the cost of more GPU
        /// memory (each worker's denoiser allocates its own temporal
        /// ring buffer).
        ///
        /// `1` is valid and useful for debugging.
        #[arg(short = 'W', long, default_value_t = 2)]
        workers: usize,
    },
    /// Denoise a y4m stream from stdin and emit y4m to stdout.
    ///
    /// Useful for piping through ffmpeg/x264/x265/etc. No scene
    /// detection is performed — the temporal window slides
    /// continuously across the entire stream. Only 8-bit 4:2:0 /
    /// 4:2:2 / 4:4:4 y4m is supported for v1.
    Stdin,
}

fn parse_prefilter(s: &str) -> Result<PrefilterMode, anyhow::Error> {
    if s == "none" || s.is_empty() {
        return Ok(PrefilterMode::None);
    }

    if let Some(rest) = s.strip_prefix("bilateral:") {
        let parts: Vec<&str> = rest.split(',').collect();

        if parts.len() != 2 {
            anyhow::bail!("--prefilter bilateral expects two values: bilateral:<sigma_s>,<sigma_r>");
        }

        let sigma_s: f32 = parts[0].trim().parse()?;
        let sigma_r: f32 = parts[1].trim().parse()?;

        return Ok(PrefilterMode::Bilateral { sigma_s, sigma_r });
    }

    anyhow::bail!("unknown prefilter '{s}'; expected `none` or `bilateral:<sigma_s>,<sigma_r>`")
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if std::env::var("RUST_LOG").is_err() {
        unsafe { std::env::set_var("RUST_LOG", "info") };
    }

    tracing_subscriber::fmt().with_writer(std::io::stderr).init();

    let mode = if args.temporal_radius == 0 {
        DenoisingMode::Spacial
    } else {
        DenoisingMode::Temporal {
            radius: args.temporal_radius,
        }
    };

    let prefilter = parse_prefilter(&args.prefilter)?;
    let intent = resolve_channel_intent(&args.channel_mode)?;

    let opts = CliOptions {
        accelerators: args.accelerators,
        device: args.device,
        intent,
        mode,
        prefilter,
    };

    match args.command {
        Command::File { input, workers } => file_mode::run_file(&opts, &input, workers)?,
        Command::Stdin => stdin_mode::run_stdin(&opts)?,
    }

    Ok(())
}

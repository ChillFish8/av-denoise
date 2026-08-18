mod input;
mod nl3d;
mod nlmeans;

use av_denoise::Device;
use av_denoise::accelerate::{Accelerator, get_default_accelerators};
use clap::{Parser, Subcommand};
use strum_macros::EnumString;

pub use self::input::InputSource;
pub use self::nl3d::Nl3dArgs;
pub use self::nlmeans::NlmeansArgs;
use crate::ingest::BinaryChannelIntent;

/// Speed vs quality dial.
///
/// Each denoising family reads the same dial and fills in its own
/// knobs from it. For `nlmeans` that is the variant, the temporal
/// radius, and the search radius.
#[derive(Debug, Copy, Clone, Default, EnumString)]
#[strum(ascii_case_insensitive)]
pub enum Preset {
    /// Fastest and lowest quality.
    Veryfast,
    /// One step up from `veryfast`.
    Fast,
    /// The default, favouring quality over speed.
    #[default]
    Base,
    /// One step down from `veryslow`.
    Slow,
    /// Slowest and highest quality.
    Veryslow,
}

/// Which planes to clean up.
#[derive(Debug, Copy, Clone, PartialEq, Eq, clap::ValueEnum)]
pub enum CliChannelMode {
    /// Clean only the brightness plane (Y). Colour passes through.
    Luma,
    /// Clean only the colour planes (U, V). Brightness passes through.
    Chroma,
    /// Clean all three planes together in one pass. Needs a YUV444
    /// source and cannot be combined with the other modes.
    Yuv,
}

pub fn resolve_channel_intent(modes: &[CliChannelMode]) -> Result<BinaryChannelIntent, anyhow::Error> {
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

#[derive(Debug, Parser)]
#[command(about = "Fast and efficient video denoising", long_about = None, version)]
pub struct Args {
    /// Speed vs quality dial.
    ///
    /// `veryfast` is the fastest and lowest-quality end of the dial.
    /// For `nlmeans` it runs the `fast` variant with no temporal window
    /// and matches this tool's original default behavior.
    ///
    /// `fast`, `base`, `slow`, and `veryslow` all run the `hq` variant
    /// and widen the temporal window going up the list, from a 1-frame
    /// radius at `fast` to an 8-frame radius at `veryslow`. `slow` and
    /// `veryslow` also widen the search radius.
    ///
    /// `base` is the default.
    #[arg(long, default_value = "base", global = true)]
    pub preset: Preset,

    /// Which hardware backends to try, in order of preference.
    ///
    /// The first backend that initialises is used. If none work the
    /// program exits with an error.
    ///
    /// The list is comma-separated, for example `cuda,vulkan`.
    #[arg(short = 'A', long, value_delimiter = ',', default_values_t = get_default_accelerators(), global = true)]
    pub accelerators: Vec<Accelerator>,

    /// Which device to use on the chosen backend.
    ///
    /// `default` lets the backend pick.
    ///
    /// `discrete[:N]` picks the Nth discrete GPU (default 0).
    /// Works on CUDA, ROCm, and Vulkan.
    ///
    /// `integrated[:N]` picks the Nth integrated GPU. Vulkan only.
    ///
    /// `virtual[:N]` picks the Nth virtual GPU. Vulkan only.
    ///
    /// `cpu` picks a software device where the platform offers one,
    /// such as lavapipe under Vulkan. It is for testing the pipeline,
    /// not for real encodes.
    #[arg(short, long, default_value = "default", global = true)]
    pub device: Device,

    /// Which planes of the video to clean (comma-separated).
    ///
    /// `luma` cleans only the brightness plane. Colour passes through
    /// untouched, which is cheaper when only luma carries grain.
    ///
    /// `chroma` cleans only the colour planes at their native size.
    ///
    /// `luma,chroma` cleans both as two independent passes. This is the
    /// default and is usually what you want for noisy footage.
    ///
    /// `yuv` cleans all three planes in one fused pass.
    ///
    /// `yuv` needs a YUV444 source and cannot be combined with the
    /// other modes.
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "luma,chroma",
        global = true
    )]
    pub channel_mode: Vec<CliChannelMode>,

    /// Shows a progress bar for the denoising pass when `--input`
    /// names a file.
    ///
    /// Off by default because that bar runs for the whole encode, and
    /// anything else writing to the terminal, such as the ffmpeg the
    /// output is usually piped into, scrambles it. Scene detection
    /// shows its bar without this flag, since it finishes before any
    /// output is written.
    ///
    /// Neither bar is drawn unless stderr is a terminal, and there is
    /// nothing to show a bar for on piped input.
    #[arg(long, global = true)]
    pub progress: bool,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Denoise with the non-local means family.
    ///
    /// `nlmeans` compares small patches of pixels and averages the ones
    /// that look alike, either inside a single frame or across a
    /// temporal window.
    Nlmeans(NlmeansArgs),

    /// Denoise with non-local means, then clean up what it leaves behind
    /// with a collaborative filter.
    ///
    /// `nl3d` runs the same `hq` front end `nlmeans` does, tuned to
    /// filter a little more gently than usual. That leaves some
    /// structured noise in its output on purpose. A second pass then
    /// groups matching patches from that output into stacks, runs each
    /// stack through a joint transform, and shrinks away the
    /// coefficients noise is most likely to have produced. That reaches
    /// noise the first pass's averaging could not remove without
    /// smoothing away real detail along with it.
    ///
    /// Always runs the `hq` front end. `--variant fast` is rejected.
    Nl3d(Nl3dArgs),
}

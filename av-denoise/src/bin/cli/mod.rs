mod common;
mod input;
mod list_devices;
mod motion;
mod nl4d;
mod nlmeans;

pub use av_denoise::Preset;
use av_denoise::accelerate::{Accelerator, get_default_accelerators};
use av_denoise::{ChannelIntent, Device, PlaneOptions};
use clap::{Parser, Subcommand};

pub use self::common::CommonArgs;
pub use self::input::InputSource;
pub use self::list_devices::run_list_devices;
pub use self::motion::MotionArgs;
pub use self::nl4d::Nl4dArgs;
pub use self::nlmeans::NlmeansArgs;

/// The options `main` runs a denoising pass with.
///
/// `planes` is the library-side option set that drives `PlanarDenoiser`.
/// `progress` is a CLI concern the library layer has no use for, so it
/// stays here rather than on `PlaneOptions`.
#[derive(Debug, Clone)]
pub struct RunOptions {
    pub planes: PlaneOptions,
    /// Draws the denoising progress bar for file input.
    pub progress: bool,
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

pub fn resolve_channel_intent(modes: &[CliChannelMode]) -> Result<ChannelIntent, anyhow::Error> {
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
        (true, _, _) => ChannelIntent::YuvFused,
        (false, true, true) => ChannelIntent::LumaChroma,
        (false, true, false) => ChannelIntent::Luma,
        (false, false, true) => ChannelIntent::Chroma,
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
    /// and matches this tool's original default behavior. For `nl4d` it
    /// keeps a 1-frame temporal window, which that algorithm needs, and
    /// narrows the spatial search instead.
    ///
    /// Going up the list widens the temporal window, from a 1-frame
    /// radius at `fast` to an 8-frame radius at `veryslow`. `fast` and
    /// above also run the `hq` variant of `nlmeans`, and `slow` and
    /// `veryslow` widen its search radius.
    ///
    /// `base` is the default.
    #[arg(long, env = "AVD_PRESET", default_value = "base", global = true)]
    pub preset: Preset,

    /// Which hardware backends to try, in order of preference.
    ///
    /// The first backend that initialises is used. If none work the
    /// program exits with an error.
    ///
    /// The list is comma-separated, for example `cuda,vulkan`.
    #[arg(
        short = 'A',
        long,
        env = "AVD_ACCELERATORS",
        value_delimiter = ',',
        default_values_t = get_default_accelerators(),
        global = true
    )]
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
    #[arg(short, long, env = "AVD_DEVICE", default_value = "default", global = true)]
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
        env = "AVD_CHANNEL_MODE",
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
    #[arg(long, env = "AVD_PROGRESS", global = true)]
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

    /// Denoise by grouping matching patches across several noisy frames
    /// directly, rather than filtering with non-local means first.
    ///
    /// `nl4d` measures the noise level, tracks motion, and scores how
    /// well each neighbour frame matches, the same way `nlmeans hq`
    /// does. No NLM weighting pass ever runs. Instead, patches are
    /// grouped straight out of the noisy frames, searching both the
    /// centre frame spatially and each neighbour frame around where a
    /// patch is predicted to have moved, then each group's coefficients
    /// are shrunk jointly.
    ///
    /// Motion tracking is always on, and every preset keeps a temporal
    /// window, which this algorithm needs.
    Nl4d(Nl4dArgs),

    /// List the devices each backend can see on this machine.
    ///
    /// Every row names a device in the spelling `--device` takes, next
    /// to the backends that offer it. `--accelerators` narrows which
    /// backends are asked.
    ///
    /// Ordinals are counted per backend, so the same row under two
    /// backends is not always the same physical card.
    ListDevices,
}

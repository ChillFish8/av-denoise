use std::path::PathBuf;

use av_denoise::accelerate::{Accelerator, get_default_accelerators};
use av_denoise::{DenoisingMode, Device, MotionCompensationMode, NlmTuning, PrefilterMode};
use clap::{Parser, Subcommand};
use strum_macros::EnumString;

mod file_mode;
mod ingest;
mod stdin_mode;

use ingest::{BinaryChannelIntent, CliOptions};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Denoising algorithm. Only `nlmeans` is currently implemented.
#[derive(Debug, Copy, Clone, Default, EnumString)]
#[strum(ascii_case_insensitive)]
pub enum Algorithm {
    #[default]
    Nlmeans,
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

#[derive(Debug, Parser)]
#[command(about = "Fast and efficient video denoising", long_about = None)]
struct Args {
    /// Denoising algorithm to run.
    ///
    /// Only `nlmeans` is currently available.
    #[arg(short, long, default_value = "nlmeans", global = true)]
    algorithm: Algorithm,

    /// Which hardware backends to try, in order of preference.
    ///
    /// The first backend that initialises is used. If none work the
    /// program exits with an error. The list is comma-separated,
    /// for example `vulkan,cpu`.
    #[arg(short = 'A', long, value_delimiter = ',', default_values_t = get_default_accelerators(), global = true)]
    accelerators: Vec<Accelerator>,

    /// Which device to use on the chosen backend.
    ///
    /// Accepted values:
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
    /// `cpu` uses the software backend.
    #[arg(short, long, default_value = "default", global = true)]
    device: Device,

    /// Which planes of the video to clean (comma-separated).
    ///
    /// `luma` cleans only the brightness plane.
    ///
    /// `chroma` cleans only the colour planes at their native size.
    ///
    /// `luma,chroma` cleans both as two independent passes, which is
    /// usually what you want for noisy footage.
    ///
    /// `yuv` cleans all three planes in one fused pass. This needs a
    /// YUV444 source and cannot be combined with the other modes.
    #[arg(long, value_enum, value_delimiter = ',', default_values_t = vec![CliChannelMode::Luma], global = true)]
    channel_mode: Vec<CliChannelMode>,

    /// Reference image used when comparing patches.
    ///
    /// `none` uses the noisy input directly (the cheapest option).
    ///
    /// `bilateral:<sigma_s>,<sigma_r>` runs a quick on-GPU bilateral
    /// blur first, then compares patches against that cleaner image.
    /// `sigma_s` is the spatial blur radius in pixels and `sigma_r`
    /// is the colour-similarity threshold in `[0, 1]`. A good
    /// starting point is `bilateral:3.0,0.02`.
    ///
    /// Prefiltering keeps more detail at the cost of one extra GPU
    /// pass per frame.
    #[arg(long, default_value = "none", global = true)]
    prefilter: String,

    /// How many neighbouring frames to look at on each side when
    /// cleaning a frame.
    ///
    /// `0` (default) means no temporal blending: each frame is
    /// cleaned on its own.
    ///
    /// Values above `0` look at that many frames before and after
    /// the current one. Larger values give stronger cleanup but use
    /// more memory and add latency.
    ///
    /// In `file` mode this is reset at every scene change, so
    /// raising it never causes blending across cuts.
    #[arg(long, default_value_t = 0, global = true)]
    temporal_radius: u32,

    /// How far away to look for similar patches inside a frame.
    ///
    /// Larger values find more matches but cost quadratically more
    /// work. Library default is 2.
    #[arg(long, global = true)]
    search_radius: Option<u32>,

    /// Size of each patch being compared. The patch is
    /// `(2*patch_radius + 1)` pixels square.
    ///
    /// Larger patches preserve fine structure better but cost more
    /// GPU memory. Library default is 4.
    #[arg(long, global = true)]
    patch_radius: Option<u32>,

    /// Cleaning strength. Higher numbers smooth more.
    ///
    /// Must be a finite number greater than 0. Library default is
    /// 1.2.
    ///
    /// This value applies to both planes unless `--luma-strength`
    /// or `--chroma-strength` is set.
    #[arg(long, global = true)]
    strength: Option<f32>,

    /// Strength override for the brightness plane only.
    ///
    /// Falls back to `--strength` (or the library default) when not
    /// set. Ignored when luma is not being denoised, or when
    /// `--channel-mode yuv` is used.
    #[arg(long, global = true)]
    luma_strength: Option<f32>,

    /// Strength override for the colour planes only.
    ///
    /// Falls back to `--strength` (or the library default) when not
    /// set. Ignored when chroma is not being denoised, or when
    /// `--channel-mode yuv` is used.
    #[arg(long, global = true)]
    chroma_strength: Option<f32>,

    /// How much weight to give the centre pixel itself when
    /// averaging.
    ///
    /// Library default is 1.0. Must be a finite number `>= 0`.
    /// Setting to 0 gives pure NLM (centre pixel only counts if a
    /// similar patch was found nearby).
    #[arg(long)]
    self_weight: Option<f32>,

    /// Turn on motion compensation for temporal denoising.
    ///
    /// When the camera or content moves between frames, the
    /// brightness at the same `(x, y)` is different content in each
    /// frame. Without help, temporal cleanup will blur moving edges.
    ///
    /// Motion compensation looks at where each block of pixels
    /// moved between frames, then shifts neighbour frames to line up
    /// with the current frame before cleaning. This keeps detail
    /// sharp on anime, fast pans, and action footage.
    ///
    /// Has no effect when `--temporal-radius 0`.
    #[arg(long, global = true)]
    motion_compensation: bool,

    /// Size of each motion-search block, in pixels. Must be even.
    ///
    /// Larger blocks are more stable but track motion less
    /// accurately on small details.
    #[arg(long, default_value = "16", global = true)]
    mc_blksize: u32,

    /// How many pixels neighbouring motion blocks may overlap.
    ///
    /// Must be less than `--mc-blksize`. Higher overlap smooths the
    /// transitions between blocks but does more work.
    #[arg(long, default_value = "8", global = true)]
    mc_overlap: u32,

    /// How many pixels of motion to search for at the finest level.
    ///
    /// The coarse pyramid pass reaches further (search radius times
    /// 2 for a 2-level pyramid), so for typical content the default
    /// is fine. Raise it for very fast motion.
    #[arg(long, default_value = "4", global = true)]
    mc_search: u32,

    /// How many levels the motion-search pyramid uses.
    ///
    /// `1` does a single full-resolution search (cheaper, weaker on
    /// large motion).
    ///
    /// `2` (default) does a coarse pass on a half-size image first,
    /// then refines at full resolution. This handles much larger
    /// motion at modest extra cost.
    #[arg(long, default_value = "2", global = true)]
    mc_pyramid_levels: u32,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Denoise a video file, splitting work by scene.
    ///
    /// Opens the file with ffms2, finds scene boundaries with
    /// `av-scenechange`, and runs each scene on its own worker
    /// thread. Temporal context is reset between scenes so the
    /// denoiser never blends frames across a cut.
    File {
        /// Path to the input video file.
        ///
        /// Any container or codec supported by ffmpeg works. The
        /// source must be 8-bit; 10 or 12-bit inputs are rejected
        /// with a clear error message.
        #[arg(short, long)]
        input: PathBuf,

        /// How many scenes to clean in parallel.
        ///
        /// Each worker uses its own GPU memory for the frame ring
        /// buffer, so higher values trade GPU memory for throughput.
        /// `1` is valid and useful for debugging.
        #[arg(short = 'W', long, default_value_t = 2)]
        workers: usize,
    },
    /// Denoise a y4m stream coming in on stdin, writing y4m on
    /// stdout.
    ///
    /// Useful for piping through ffmpeg or an encoder. There is no
    /// scene detection in this mode, so temporal denoising slides
    /// across the whole stream. Only 8-bit 4:2:0 / 4:2:2 / 4:4:4
    /// y4m is supported right now.
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
    // cubecl spawns its per-device worker thread with no explicit stack
    // size (uses Rust's default 2 MiB). GPU kernel codegen runs on that
    // thread; at large --search-radius the (2R+1)^2 unrolled body in
    // the windowed NLM kernels in src/nlmeans/kernels/fused.rs
    // overflows the default stack. RUST_MIN_STACK is cached on first
    // read, so set it here before any GPU thread spawns.
    if std::env::var_os("RUST_MIN_STACK").is_none() {
        // SAFETY: still single-threaded, no other thread can race the env mutation.
        unsafe { std::env::set_var("RUST_MIN_STACK", "16777216") };
    }

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

    let motion_compensation = if args.motion_compensation {
        if args.temporal_radius == 0 {
            tracing::warn!(
                "--motion-compensation has no effect when --temporal-radius is 0; \
                 the spatial path doesn't use temporal neighbours"
            );
        }
        MotionCompensationMode::Mvtools {
            blksize: args.mc_blksize,
            overlap: args.mc_overlap,
            search_radius: args.mc_search,
            pyramid_levels: args.mc_pyramid_levels,
        }
    } else {
        MotionCompensationMode::None
    };

    let nlm_tuning = if args.search_radius.is_some()
        || args.patch_radius.is_some()
        || args.strength.is_some()
        || args.self_weight.is_some()
    {
        Some(NlmTuning {
            search_radius: args.search_radius,
            patch_radius: args.patch_radius,
            strength: args.strength,
            self_weight: args.self_weight,
        })
    } else {
        None
    };

    let opts = CliOptions {
        accelerators: args.accelerators,
        device: args.device,
        intent,
        mode,
        prefilter,
        motion_compensation,
        nlm_tuning,
        luma_strength: args.luma_strength,
        chroma_strength: args.chroma_strength,
    };

    match args.command {
        Command::File { input, workers } => file_mode::run_file(&opts, &input, workers)?,
        Command::Stdin => stdin_mode::run_stdin(&opts)?,
    }

    Ok(())
}

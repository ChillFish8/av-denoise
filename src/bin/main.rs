use std::path::PathBuf;

use av_denoise::accelerate::{Accelerator, get_default_accelerators};
use av_denoise::{
    DEFAULT_PILOT_STRENGTH_SCALE,
    DenoisingMode,
    Device,
    MotionCompensationMode,
    MotionEstimation,
    NlmTuning,
    PrefilterMode,
};
use clap::{Parser, Subcommand};
use strum_macros::EnumString;

mod file_mode;
mod ingest;
mod progress;
mod stdin_mode;

use ingest::{BinaryChannelIntent, CliOptions};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Denoising algorithm. `nlmeans` is the fast path. `nlmeans-hq`
/// trades some speed for noise-calibrated quality.
#[derive(Debug, Copy, Clone, Default, EnumString)]
#[strum(ascii_case_insensitive)]
pub enum Algorithm {
    #[default]
    Nlmeans,
    #[strum(serialize = "nlmeans-hq")]
    NlmeansHq,
}

/// Speed vs quality dial.
///
/// Fills in `--algorithm`, `--temporal-radius`, and `--search-radius`
/// with a matched set of values.
///
/// Passing any of those three flags explicitly overrides just that value.
/// The rest of the preset still applies.
#[derive(Debug, Copy, Clone, Default, EnumString)]
#[strum(ascii_case_insensitive)]
pub enum Preset {
    /// Fastest. `nlmeans`, no temporal window, search radius 2. Matches
    /// this tool's original default behavior.
    Veryfast,
    /// `nlmeans-hq`, temporal radius 1, search radius 2.
    Fast,
    /// `nlmeans-hq`, temporal radius 2, search radius 2. The default,
    /// favouring quality over the old default's speed.
    #[default]
    Base,
    /// `nlmeans-hq`, temporal radius 4, search radius 4.
    Slow,
    /// Slowest. `nlmeans-hq`, temporal radius 8, search radius 4.
    Veryslow,
}

impl Preset {
    fn algorithm(self) -> Algorithm {
        match self {
            Preset::Veryfast => Algorithm::Nlmeans,
            Preset::Fast | Preset::Base | Preset::Slow | Preset::Veryslow => Algorithm::NlmeansHq,
        }
    }

    fn temporal_radius(self) -> u32 {
        match self {
            Preset::Veryfast => 0,
            Preset::Fast => 1,
            Preset::Base => 2,
            Preset::Slow => 4,
            Preset::Veryslow => 8,
        }
    }

    fn search_radius(self) -> u32 {
        match self {
            Preset::Veryfast | Preset::Fast | Preset::Base => 2,
            Preset::Slow | Preset::Veryslow => 4,
        }
    }
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
    /// Speed vs quality dial.
    ///
    /// `veryfast` is the fastest and lowest-quality end of the dial. It
    /// runs the plain `nlmeans` algorithm with no temporal window and
    /// matches this tool's original default behavior.
    ///
    /// `fast`, `base`, `slow`, and `veryslow` all run `nlmeans-hq` and
    /// widen the temporal window going up the list, from a 1-frame
    /// radius at `fast` to an 8-frame radius at `veryslow`. `slow` and
    /// `veryslow` also widen the search radius.
    ///
    /// `base` is the default.
    #[arg(long, default_value = "base", global = true)]
    preset: Preset,

    /// Denoising algorithm to run.
    ///
    /// `nlmeans` is the fast path. `nlmeans-hq` is a quality-focused
    /// variant that calibrates its weighting to the noise level,
    /// measured automatically per frame (see `--hq-sigma` to override).
    ///
    /// Defaults to whatever `--preset` selects.
    #[arg(short, long, global = true)]
    algorithm: Option<Algorithm>,

    /// Which hardware backends to try, in order of preference.
    ///
    /// The first backend that initialises is used. If none work the
    /// program exits with an error.
    ///
    /// The list is comma-separated, for example `vulkan,cpu`.
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
    /// `yuv` cleans all three planes in one fused pass.
    ///
    /// `yuv` needs a YUV444 source and cannot be combined with the
    /// other modes.
    #[arg(long, value_enum, value_delimiter = ',', default_values_t = vec![CliChannelMode::Luma], global = true)]
    channel_mode: Vec<CliChannelMode>,

    /// Reference image used when comparing patches.
    ///
    /// Omitted (the default) means no prefilter, for both `nlmeans`
    /// and `nlmeans-hq`.
    ///
    /// `none` forces the noisy input directly (the cheapest option).
    /// This is the same as leaving the flag unset.
    ///
    /// `nlm` or `nlm:<strength_scale>` runs a windowed spatial NLM
    /// pass first and compares patches against that cleaner image.
    /// `strength_scale` multiplies the main pass strength for the
    /// pilot pass. Bare `nlm` uses the calibrated default.
    ///
    /// `bilateral:<sigma_s>,<sigma_r>` runs a quick on-GPU bilateral
    /// blur first, then compares patches against that cleaner image.
    ///
    /// `sigma_s` is the spatial blur radius in pixels.
    ///
    /// `sigma_r` is the colour-similarity threshold in `[0, 1]`.
    ///
    /// A good starting point is `bilateral:3.0,0.02`.
    ///
    /// Prefiltering keeps more detail at the cost of one extra GPU
    /// pass per frame.
    #[arg(long, global = true)]
    prefilter: Option<String>,

    /// How many neighbouring frames to look at on each side when
    /// cleaning a frame.
    ///
    /// `0` means no temporal blending. Each frame is cleaned on its
    /// own.
    ///
    /// Values above `0` look at that many frames before and after
    /// the current one.
    ///
    /// Larger values give stronger cleanup but use more memory and
    /// add latency.
    ///
    /// In `file` mode this is reset at every scene change, so
    /// raising it never causes blending across cuts.
    ///
    /// Defaults to whatever `--preset` selects.
    #[arg(long, global = true)]
    temporal_radius: Option<u32>,

    /// How far away to look for similar patches inside a frame.
    ///
    /// Larger values find more matches but cost quadratically more
    /// work.
    ///
    /// Defaults to whatever `--preset` selects.
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
    /// Must be a finite number greater than 0.
    ///
    /// The default depends on the algorithm. `nlmeans` defaults to
    /// 1.2. `nlmeans-hq` interprets strength as a multiplier on the
    /// measured noise level. Its default is calibrated automatically,
    /// adapting to the temporal radius and to which plane (luma or
    /// chroma) is being denoised, so lower and higher radii each get
    /// their own measured value.
    ///
    /// This value applies to both planes unless `--luma-strength`
    /// or `--chroma-strength` is set.
    #[arg(long, global = true)]
    strength: Option<f32>,

    /// Strength override for the brightness plane only.
    ///
    /// Falls back to `--strength` (or the library default) when not
    /// set.
    ///
    /// Ignored when luma is not being denoised, or when
    /// `--channel-mode yuv` is used.
    #[arg(long, global = true)]
    luma_strength: Option<f32>,

    /// Strength override for the colour planes only.
    ///
    /// Falls back to `--strength` (or the library default) when not
    /// set.
    ///
    /// Ignored when chroma is not being denoised, or when
    /// `--channel-mode yuv` is used.
    #[arg(long, global = true)]
    chroma_strength: Option<f32>,

    /// How much weight to give the centre pixel itself when
    /// averaging.
    ///
    /// Library default is 1.0. Must be a finite number `>= 0`.
    ///
    /// Setting to 0 gives pure NLM (centre pixel only counts if a
    /// similar patch was found nearby).
    #[arg(long, global = true)]
    self_weight: Option<f32>,

    /// How noisy the source is. Leave it unset for almost all uses.
    ///
    /// The noise level is measured automatically per scene when this
    /// is not set. Set it only when the automatic estimate misjudges
    /// a source and you want to pin the value.
    ///
    /// Small values mean light grain and larger values mean heavier
    /// noise. `3` is subtle grain, `6` is clearly visible grain, `12`
    /// and up is heavy noise.
    #[arg(long, global = true)]
    hq_sigma: Option<f32>,

    /// Treat `--strength` as an absolute value instead of a
    /// multiplier on `--hq-sigma`.
    #[arg(long, global = true)]
    hq_no_auto_strength: bool,

    /// Keep the expected-noise floor inside patch distances instead
    /// of subtracting it.
    #[arg(long, global = true)]
    hq_no_noise_floor: bool,

    /// Disable per-block temporal confidence weighting for `nlmeans-hq`.
    ///
    /// By default HQ block-matches each temporal neighbour against the
    /// centre frame and lets a poor match suppress that neighbour's
    /// contribution, instead of blurring in occluded or changed
    /// content. Setting this applies temporal weights uniformly no
    /// matter how well a neighbour matches.
    ///
    /// Only takes effect when `--temporal-radius` is above 0.
    #[arg(long, global = true)]
    hq_no_temporal_confidence: bool,

    /// Multiplier on the per-block mismatch threshold temporal
    /// confidence weighting tolerates before a neighbour's contribution
    /// starts dropping.
    ///
    /// Higher values tolerate larger mismatches. Library default is
    /// 1.0. Ignored when `--hq-no-temporal-confidence` is set.
    #[arg(long, global = true)]
    hq_thsad_scale: Option<f32>,

    /// Nudges the automatically measured noise level up or down.
    ///
    /// `1.0` (the library default) keeps the measurement as-is. Raise
    /// it a little when the cleaned result still looks noisy. Lower
    /// it when detail is getting scrubbed.
    ///
    /// This differs from `--strength` because the noise level also
    /// sets the patch-distance noise floor and the motion-confidence
    /// floor, not just the weighting.
    ///
    /// Has no effect when `--hq-sigma` pins the noise level.
    #[arg(long, global = true)]
    hq_sigma_scale: Option<f32>,

    /// Turn on motion compensation for temporal denoising.
    ///
    /// When the camera or content moves between frames, the
    /// brightness at the same `(x, y)` is different content in each
    /// frame.
    ///
    /// Without help, temporal cleanup will blur moving edges.
    ///
    /// Motion compensation looks at where each block of pixels
    /// moved between frames, then shifts neighbour frames to line up
    /// with the current frame before cleaning.
    ///
    /// This keeps detail sharp on anime, fast pans, and action
    /// footage.
    ///
    /// The tracking strategy adapts automatically to `--temporal-radius`.
    ///
    /// Has no effect when `--temporal-radius 0`.
    #[arg(long, global = true)]
    motion_compensation: bool,

    /// Size of each motion-search block, in pixels. Must be even.
    ///
    /// Larger blocks are more stable but track motion less
    /// accurately on small details.
    ///
    /// Only takes effect with `--motion-compensation`. Defaults to 16
    /// when unset.
    #[arg(long, global = true)]
    mc_blksize: Option<u32>,

    /// How many pixels neighbouring motion blocks may overlap.
    ///
    /// Must be less than `--mc-blksize`. Higher overlap smooths the
    /// transitions between blocks but does more work.
    ///
    /// Only takes effect with `--motion-compensation`. Defaults to 8
    /// when unset.
    #[arg(long, global = true)]
    mc_overlap: Option<u32>,

    /// How many pixels of motion to search for at the finest level.
    ///
    /// The coarse pyramid pass reaches further (search radius times
    /// 2 for a 2-level pyramid), so for typical content the default
    /// is fine.
    ///
    /// Raise it for very fast motion.
    ///
    /// Only takes effect with `--motion-compensation`. Defaults to 4
    /// when unset.
    #[arg(long, global = true)]
    mc_search: Option<u32>,

    /// How many levels the motion-search pyramid uses.
    ///
    /// `1` does a single full-resolution search (cheaper, weaker on
    /// large motion).
    ///
    /// `2` (default) does a coarse pass on a half-size image first,
    /// then refines at full resolution.
    ///
    /// This handles much larger motion at modest extra cost.
    ///
    /// Only takes effect with `--motion-compensation`. Defaults to 2
    /// when unset.
    #[arg(long, global = true)]
    mc_pyramid_levels: Option<u32>,

    /// Hides the progress bar.
    ///
    /// The progress bar is otherwise shown when the output terminal
    /// supports it. It only appears during scene detection in `file`
    /// mode. There is nothing to show a bar for in `stdin` mode.
    #[arg(long, global = true)]
    no_progress: bool,

    #[command(subcommand)]
    command: Command,
}

/// `--algorithm`, `--temporal-radius`, and `--search-radius` resolved
/// from either an explicit flag or the active `--preset`.
#[derive(Debug, Copy, Clone)]
struct ResolvedPreset {
    algorithm: Algorithm,
    temporal_radius: u32,
    search_radius: u32,
}

impl Args {
    fn resolve_preset(&self) -> ResolvedPreset {
        ResolvedPreset {
            algorithm: self.algorithm.unwrap_or_else(|| self.preset.algorithm()),
            temporal_radius: self
                .temporal_radius
                .unwrap_or_else(|| self.preset.temporal_radius()),
            search_radius: self.search_radius.unwrap_or_else(|| self.preset.search_radius()),
        }
    }

    /// Builds the library's [`av_denoise::Algorithm`] from the resolved
    /// preset and the `--hq-*` flags, warning about flags that are set
    /// but have no effect given the rest of the resolved configuration.
    fn resolve_algorithm(&self, resolved: ResolvedPreset) -> Result<av_denoise::Algorithm, anyhow::Error> {
        let sigma_scale_is_set = self.hq_sigma_scale.is_some_and(|v| v != 1.0);

        match resolved.algorithm {
            Algorithm::Nlmeans => {
                if self.hq_sigma.is_some()
                    || self.hq_no_auto_strength
                    || self.hq_no_noise_floor
                    || self.hq_no_temporal_confidence
                    || self.hq_thsad_scale.is_some()
                    || sigma_scale_is_set
                {
                    tracing::warn!("--hq-* options are ignored unless --algorithm nlmeans-hq is selected");
                }
                Ok(av_denoise::Algorithm::Nlmeans)
            },
            Algorithm::NlmeansHq => {
                // Check the raw 8-bit value here so an out-of-range
                // `--hq-sigma` reports the number the user typed. The
                // library re-validates the same bound after the /255
                // normalisation, but its message speaks in [0, 1] units.
                if let Some(sigma) = self.hq_sigma
                    && (!sigma.is_finite() || sigma <= 0.0 || sigma > 255.0)
                {
                    anyhow::bail!("--hq-sigma must be a finite value in (0, 255] 8-bit units (got {sigma})");
                }

                if self.hq_sigma.is_some() && sigma_scale_is_set {
                    tracing::warn!("--hq-sigma-scale has no effect when --hq-sigma pins the noise level");
                }

                Ok(av_denoise::Algorithm::NlmeansHq(av_denoise::HqParams {
                    auto_strength: !self.hq_no_auto_strength,
                    noise_floor: !self.hq_no_noise_floor,
                    sigma_override: self.hq_sigma.map(|s| s / 255.0),
                    temporal_confidence: !self.hq_no_temporal_confidence,
                    thsad_scale: self.hq_thsad_scale.unwrap_or(1.0),
                    sigma_scale: self.hq_sigma_scale.unwrap_or(1.0),
                }))
            },
        }
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Denoise a video file, splitting work by scene.
    ///
    /// Opens the file with ffms2, finds scene boundaries with
    /// `av-scenechange`, and runs each scene on its own worker
    /// thread.
    ///
    /// Temporal context is reset between scenes so the denoiser
    /// never blends frames across a cut.
    File {
        /// Path to the input video file.
        ///
        /// Any container or codec supported by ffmpeg works.
        ///
        /// The source must be 8-bit; 10 or 12-bit inputs are
        /// rejected with a clear error message.
        #[arg(short, long)]
        input: PathBuf,

        /// How many scenes to clean in parallel.
        ///
        /// Each worker uses its own GPU memory for the frame ring
        /// buffer, so higher values trade GPU memory for throughput.
        ///
        /// `1` is valid and useful for debugging.
        #[arg(short = 'W', long, default_value_t = 2)]
        workers: usize,
    },
    /// Denoise a y4m stream coming in on stdin, writing y4m on
    /// stdout.
    ///
    /// Useful for piping through ffmpeg or an encoder.
    ///
    /// There is no scene detection in this mode, so temporal
    /// denoising slides across the whole stream.
    ///
    /// Only 8-bit 4:2:0 / 4:2:2 / 4:4:4 y4m is supported right now.
    Stdin,
}

fn parse_prefilter(s: &str) -> Result<PrefilterMode, anyhow::Error> {
    if s == "none" || s.is_empty() {
        return Ok(PrefilterMode::None);
    }

    if s == "nlm" {
        return Ok(PrefilterMode::NlmSpatial {
            strength_scale: DEFAULT_PILOT_STRENGTH_SCALE,
        });
    }

    if let Some(rest) = s.strip_prefix("nlm:") {
        let strength_scale: f32 = rest
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("--prefilter nlm expects a number: nlm:<strength_scale>"))?;

        return Ok(PrefilterMode::NlmSpatial { strength_scale });
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

    anyhow::bail!(
        "unknown prefilter '{s}'; expected `none`, `nlm[:<strength_scale>]`, or `bilateral:<sigma_s>,<sigma_r>`"
    )
}

#[cfg(test)]
mod parse_prefilter_tests {
    use super::*;

    #[test]
    fn none_and_empty() {
        assert!(matches!(parse_prefilter("none").unwrap(), PrefilterMode::None));
        assert!(matches!(parse_prefilter("").unwrap(), PrefilterMode::None));
    }

    #[test]
    fn bilateral_with_values() {
        let mode = parse_prefilter("bilateral:3.0,0.02").unwrap();
        assert!(matches!(
            mode,
            PrefilterMode::Bilateral {
                sigma_s,
                sigma_r,
            } if (sigma_s - 3.0).abs() < f32::EPSILON && (sigma_r - 0.02).abs() < f32::EPSILON
        ));
    }

    #[test]
    fn bare_nlm_uses_default_strength_scale() {
        let mode = parse_prefilter("nlm").unwrap();
        assert!(matches!(
            mode,
            PrefilterMode::NlmSpatial { strength_scale }
                if (strength_scale - DEFAULT_PILOT_STRENGTH_SCALE).abs() < f32::EPSILON
        ));
    }

    #[test]
    fn nlm_with_explicit_strength_scale() {
        let mode = parse_prefilter("nlm:0.8").unwrap();
        assert!(matches!(
            mode,
            PrefilterMode::NlmSpatial { strength_scale } if (strength_scale - 0.8).abs() < f32::EPSILON
        ));
    }

    #[test]
    fn malformed_nlm_scale_is_rejected() {
        let err = parse_prefilter("nlm:x").expect_err("expected parse failure");
        assert!(err.to_string().contains("nlm"));
    }

    #[test]
    fn unknown_prefilter_is_rejected() {
        assert!(parse_prefilter("garbage").is_err());
    }
}

#[cfg(test)]
mod preset_tests {
    use super::*;

    /// Parses `Args` from CLI-style tokens, always terminated with the
    /// `stdin` subcommand since these tests only exercise the top-level
    /// preset resolution.
    fn parse(extra: &[&str]) -> Args {
        let mut argv = vec!["av-denoise"];
        argv.extend_from_slice(extra);
        argv.push("stdin");
        Args::parse_from(argv)
    }

    #[test]
    fn default_preset_is_base() {
        let resolved = parse(&[]).resolve_preset();
        assert!(matches!(resolved.algorithm, Algorithm::NlmeansHq));
        assert_eq!(resolved.temporal_radius, 2);
        assert_eq!(resolved.search_radius, 2);
    }

    #[test]
    fn veryfast_matches_former_defaults() {
        let resolved = parse(&["--preset", "veryfast"]).resolve_preset();
        assert!(matches!(resolved.algorithm, Algorithm::Nlmeans));
        assert_eq!(resolved.temporal_radius, 0);
        assert_eq!(resolved.search_radius, 2);
    }

    #[test]
    fn fast_grid_row() {
        let resolved = parse(&["--preset", "fast"]).resolve_preset();
        assert!(matches!(resolved.algorithm, Algorithm::NlmeansHq));
        assert_eq!(resolved.temporal_radius, 1);
        assert_eq!(resolved.search_radius, 2);
    }

    #[test]
    fn base_grid_row() {
        let resolved = parse(&["--preset", "base"]).resolve_preset();
        assert!(matches!(resolved.algorithm, Algorithm::NlmeansHq));
        assert_eq!(resolved.temporal_radius, 2);
        assert_eq!(resolved.search_radius, 2);
    }

    #[test]
    fn slow_grid_row() {
        let resolved = parse(&["--preset", "slow"]).resolve_preset();
        assert!(matches!(resolved.algorithm, Algorithm::NlmeansHq));
        assert_eq!(resolved.temporal_radius, 4);
        assert_eq!(resolved.search_radius, 4);
    }

    #[test]
    fn veryslow_grid_row() {
        let resolved = parse(&["--preset", "veryslow"]).resolve_preset();
        assert!(matches!(resolved.algorithm, Algorithm::NlmeansHq));
        assert_eq!(resolved.temporal_radius, 8);
        assert_eq!(resolved.search_radius, 4);
    }

    #[test]
    fn explicit_algorithm_overrides_preset() {
        let resolved = parse(&["--preset", "base", "--algorithm", "nlmeans"]).resolve_preset();
        assert!(matches!(resolved.algorithm, Algorithm::Nlmeans));
        assert_eq!(resolved.temporal_radius, 2);
        assert_eq!(resolved.search_radius, 2);
    }

    #[test]
    fn explicit_temporal_radius_overrides_preset() {
        let resolved = parse(&["--preset", "base", "--temporal-radius", "6"]).resolve_preset();
        assert!(matches!(resolved.algorithm, Algorithm::NlmeansHq));
        assert_eq!(resolved.temporal_radius, 6);
        assert_eq!(resolved.search_radius, 2);
    }

    #[test]
    fn explicit_search_radius_overrides_preset() {
        let resolved = parse(&["--preset", "veryslow", "--search-radius", "1"]).resolve_preset();
        assert!(matches!(resolved.algorithm, Algorithm::NlmeansHq));
        assert_eq!(resolved.temporal_radius, 8);
        assert_eq!(resolved.search_radius, 1);
    }
}

#[cfg(test)]
mod hq_sigma_scale_tests {
    use super::*;

    /// Parses `Args` from CLI-style tokens, always terminated with the
    /// `stdin` subcommand. Mirrors `preset_tests::parse`.
    fn parse(extra: &[&str]) -> Args {
        let mut argv = vec!["av-denoise"];
        argv.extend_from_slice(extra);
        argv.push("stdin");
        Args::parse_from(argv)
    }

    #[test]
    fn flag_parses_to_the_typed_value() {
        let args = parse(&["--hq-sigma-scale", "2.0"]);
        assert_eq!(args.hq_sigma_scale, Some(2.0));
    }

    #[test]
    fn unset_flag_resolves_to_the_library_default_of_one() {
        let args = parse(&["--algorithm", "nlmeans-hq"]);
        let resolved = args.resolve_preset();
        let algorithm = args
            .resolve_algorithm(resolved)
            .expect("resolution should succeed");

        match algorithm {
            av_denoise::Algorithm::NlmeansHq(hq) => assert_eq!(hq.sigma_scale, 1.0),
            other => panic!("expected NlmeansHq, got {other:?}"),
        }
    }

    /// Setting `--hq-sigma-scale` without `--algorithm nlmeans-hq` (nor
    /// a preset that resolves to it) should warn, not fail; the CLI
    /// still resolves to the fast path.
    #[test]
    fn set_without_hq_algorithm_still_resolves_to_nlmeans() {
        let args = parse(&["--algorithm", "nlmeans", "--hq-sigma-scale", "2.0"]);
        let resolved = args.resolve_preset();
        let algorithm = args
            .resolve_algorithm(resolved)
            .expect("resolution should succeed");

        assert!(matches!(algorithm, av_denoise::Algorithm::Nlmeans));
    }

    /// Combined with `--hq-sigma`, both values should still resolve
    /// through into `HqParams` (the override warns rather than errors;
    /// `sigma_scale` staying inert at runtime is the library's job, not
    /// the CLI's).
    #[test]
    fn combined_with_hq_sigma_still_resolves_both_values() {
        let args = parse(&[
            "--algorithm",
            "nlmeans-hq",
            "--hq-sigma",
            "6",
            "--hq-sigma-scale",
            "2.0",
        ]);
        let resolved = args.resolve_preset();
        let algorithm = args
            .resolve_algorithm(resolved)
            .expect("resolution should succeed");

        match algorithm {
            av_denoise::Algorithm::NlmeansHq(hq) => {
                assert!(
                    matches!(hq.sigma_override, Some(s) if (s - 6.0 / 255.0).abs() < f32::EPSILON),
                    "expected sigma_override Some(6/255), got {:?}",
                    hq.sigma_override
                );
                assert_eq!(hq.sigma_scale, 2.0);
            },
            other => panic!("expected NlmeansHq, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod no_progress_tests {
    use super::*;

    /// Parses `Args` from CLI-style tokens, always terminated with the
    /// `stdin` subcommand. Mirrors `preset_tests::parse`.
    fn parse(extra: &[&str]) -> Args {
        let mut argv = vec!["av-denoise"];
        argv.extend_from_slice(extra);
        argv.push("stdin");
        Args::parse_from(argv)
    }

    #[test]
    fn defaults_to_false() {
        let args = parse(&[]);
        assert!(!args.no_progress);
    }

    #[test]
    fn flag_sets_it_true() {
        let args = parse(&["--no-progress"]);
        assert!(args.no_progress);
    }
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

    // Honor AV_DENOISE_COMPILATION_CACHE. Must run before Denoiser::create
    // since the first CubeCL client lazily locks the global config.
    match av_denoise::apply_compilation_cache_env() {
        Ok(Some(path)) => {
            tracing::info!(?path, "AV_DENOISE_COMPILATION_CACHE override active")
        },
        Ok(None) => {},
        Err(_) => anyhow::bail!("unable to apply AV_DENOISE_COMPILATION_CACHE, this is a bug."),
    }

    let resolved = args.resolve_preset();

    let mode = if resolved.temporal_radius == 0 {
        DenoisingMode::Spacial
    } else {
        DenoisingMode::Temporal {
            radius: resolved.temporal_radius,
        }
    };

    let prefilter = args.prefilter.as_deref().map(parse_prefilter).transpose()?;
    let intent = resolve_channel_intent(&args.channel_mode)?;

    let motion_compensation = if args.motion_compensation {
        if resolved.temporal_radius == 0 {
            tracing::warn!(
                "--motion-compensation has no effect when --temporal-radius is 0; \
                 the spatial path doesn't use temporal neighbours"
            );
        }
        MotionCompensationMode::Mvtools {
            blksize: args.mc_blksize.unwrap_or(16),
            overlap: args.mc_overlap.unwrap_or(8),
            search_radius: args.mc_search.unwrap_or(4),
            pyramid_levels: args.mc_pyramid_levels.unwrap_or(2),
            estimation: MotionEstimation::default(),
        }
    } else {
        if args.mc_blksize.is_some()
            || args.mc_overlap.is_some()
            || args.mc_search.is_some()
            || args.mc_pyramid_levels.is_some()
        {
            tracing::warn!("--mc-* options are ignored unless --motion-compensation is set");
        }
        MotionCompensationMode::None
    };

    let algorithm = args.resolve_algorithm(resolved)?;

    // search_radius always has a resolved value (explicit flag or the
    // active preset), so it's always carried into the tuning override.
    let nlm_tuning = Some(NlmTuning {
        search_radius: Some(resolved.search_radius),
        patch_radius: args.patch_radius,
        strength: args.strength,
        self_weight: args.self_weight,
    });

    let opts = CliOptions {
        accelerators: args.accelerators,
        device: args.device,
        intent,
        mode,
        prefilter,
        motion_compensation,
        algorithm,
        nlm_tuning,
        luma_strength: args.luma_strength,
        chroma_strength: args.chroma_strength,
        no_progress: args.no_progress,
    };

    match args.command {
        Command::File { input, workers } => file_mode::run_file(&opts, &input, workers)?,
        Command::Stdin => stdin_mode::run_stdin(&opts)?,
    }

    Ok(())
}

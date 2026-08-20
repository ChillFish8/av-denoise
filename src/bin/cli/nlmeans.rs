use av_denoise::{
    DEFAULT_PILOT_STRENGTH_SCALE,
    DenoisingMode,
    MotionCompensationMode,
    NlmTuning,
    NlmeansHqOptions,
    NlmeansOptions,
    PrefilterMode,
};
use strum_macros::EnumString;

use super::{Args, CommonArgs, MotionArgs, Preset, resolve_channel_intent};
use crate::ingest::CliOptions;

/// Which non-local means variant to run.
#[derive(Debug, Copy, Clone, PartialEq, Eq, EnumString)]
#[strum(ascii_case_insensitive)]
pub enum Variant {
    /// The fast path. Fixed weighting, no noise measurement.
    Fast,
    /// Quality focused. Calibrates its weighting to the noise level,
    /// measured automatically per frame.
    Hq,
}

#[derive(Debug, Clone, clap::Args)]
pub struct NlmeansArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Which variant to run.
    ///
    /// `fast` uses fixed weighting and is the cheapest option. `hq`
    /// calibrates its weighting to the noise level, measured
    /// automatically per frame (see `--hq-sigma` to override).
    ///
    /// Defaults to whatever `--preset` selects.
    #[arg(long)]
    pub variant: Option<Variant>,

    /// Reference image used when comparing patches.
    ///
    /// Omitted (the default) means no prefilter, for both variants.
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
    /// `sigma_s` is the spatial blur radius in pixels, greater than 0
    /// and at most 11.0 (anything beyond this is insane.)
    ///
    /// `sigma_r` is the colour-similarity threshold, greater than 0.
    /// `(0, 1]` is the typical range for normalised pixel data. There
    /// is no enforced upper bound.
    ///
    /// A good starting point is `bilateral:3.0,0.02`.
    ///
    /// Prefiltering keeps more detail at the cost of one extra GPU
    /// pass per frame.
    #[arg(long)]
    pub prefilter: Option<String>,

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
    /// When `--input` names a file this is reset at every scene
    /// change, so raising it never causes blending across cuts.
    ///
    /// Defaults to whatever `--preset` selects.
    #[arg(long)]
    pub temporal_radius: Option<u32>,

    /// How far away to look for similar patches inside a frame.
    ///
    /// Larger values find more matches but cost quadratically more
    /// work.
    ///
    /// Defaults to whatever `--preset` selects.
    #[arg(long)]
    pub search_radius: Option<u32>,

    /// Size of each patch being compared. The patch is
    /// `(2*patch_radius + 1)` pixels square.
    ///
    /// Larger patches preserve fine structure better but cost more
    /// GPU memory. Library default is 4.
    #[arg(long)]
    pub patch_radius: Option<u32>,

    /// Cleaning strength. Higher numbers smooth more.
    ///
    /// Must be a finite number greater than 0.
    ///
    /// The default depends on the variant. `fast` defaults to 1.2.
    /// `hq` interprets strength as a multiplier on the measured noise
    /// level. Its default is calibrated automatically, adapting to the
    /// temporal radius and to which plane (luma or chroma) is being
    /// denoised, so lower and higher radii each get their own measured
    /// value.
    ///
    /// This value applies to both planes unless `--luma-strength`
    /// or `--chroma-strength` is set.
    #[arg(long)]
    pub strength: Option<f32>,

    /// Strength override for the brightness plane only.
    ///
    /// Falls back to `--strength` (or the library default) when not
    /// set.
    ///
    /// Ignored when luma is not being denoised, or when
    /// `--channel-mode yuv` is used.
    #[arg(long)]
    pub luma_strength: Option<f32>,

    /// Strength override for the colour planes only.
    ///
    /// Falls back to `--strength` (or the library default) when not
    /// set.
    ///
    /// Ignored when chroma is not being denoised, or when
    /// `--channel-mode yuv` is used.
    #[arg(long)]
    pub chroma_strength: Option<f32>,

    /// How much weight to give the centre pixel itself when
    /// averaging.
    ///
    /// Library default is 1.0. Must be a finite number `>= 0`.
    ///
    /// Setting to 0 gives pure NLM (centre pixel only counts if a
    /// similar patch was found nearby).
    #[arg(long)]
    pub self_weight: Option<f32>,

    /// How noisy the source is. Leave it unset for almost all uses.
    ///
    /// The noise level is measured automatically per scene when this
    /// is not set. Set it only when the automatic estimate misjudges
    /// a source and you want to pin the value.
    ///
    /// Small values mean light grain and larger values mean heavier
    /// noise. `3` is subtle grain, `6` is clearly visible grain, `12`
    /// and up is heavy noise.
    ///
    /// Always expressed on an 8-bit 0-255 scale, no matter the
    /// source's actual bit depth.
    #[arg(long)]
    pub hq_sigma: Option<f32>,

    /// Treat `--strength` as an absolute value instead of a
    /// multiplier on `--hq-sigma`.
    #[arg(long)]
    pub hq_no_auto_strength: bool,

    /// Keep the expected-noise floor inside patch distances instead
    /// of subtracting it.
    #[arg(long)]
    pub hq_no_noise_floor: bool,

    /// Disable per-block temporal confidence weighting for the `hq`
    /// variant.
    ///
    /// By default HQ block-matches each temporal neighbour against the
    /// centre frame and lets a poor match suppress that neighbour's
    /// contribution, instead of blurring in occluded or changed
    /// content. Setting this applies temporal weights uniformly no
    /// matter how well a neighbour matches.
    ///
    /// Only takes effect when `--temporal-radius` is above 0.
    #[arg(long)]
    pub hq_no_temporal_confidence: bool,

    /// Multiplier on the per-block mismatch threshold temporal
    /// confidence weighting tolerates before a neighbour's contribution
    /// starts dropping.
    ///
    /// Higher values tolerate larger mismatches. Library default is
    /// 1.0. Ignored when `--hq-no-temporal-confidence` is set.
    #[arg(long)]
    pub hq_thsad_scale: Option<f32>,

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
    #[arg(long)]
    pub hq_sigma_scale: Option<f32>,

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
    /// The `--mc-*` flags only take effect with this set. Has no effect
    /// when `--temporal-radius 0`.
    #[arg(long)]
    pub motion_compensation: bool,

    #[command(flatten)]
    pub motion: MotionArgs,
}

/// `--variant`, `--temporal-radius`, and `--search-radius` resolved
/// from either an explicit flag or the active `--preset`.
#[derive(Debug, Copy, Clone)]
pub struct ResolvedPreset {
    pub variant: Variant,
    pub temporal_radius: u32,
    pub search_radius: u32,
}

impl NlmeansArgs {
    /// Fills the unset dial-driven flags in from `preset`.
    pub fn resolve_preset(&self, preset: Preset) -> ResolvedPreset {
        ResolvedPreset {
            variant: self.variant.unwrap_or_else(|| variant_for(preset)),
            temporal_radius: self
                .temporal_radius
                .unwrap_or_else(|| temporal_radius_for(preset)),
            search_radius: self.search_radius.unwrap_or_else(|| search_radius_for(preset)),
        }
    }

    /// Builds the library's [`av_denoise::Algorithm`] from the resolved
    /// preset and the flags the chosen variant reads.
    ///
    /// Flags that are set but do nothing for this configuration are
    /// reported as warnings.
    pub fn resolve_algorithm(
        &self,
        resolved: ResolvedPreset,
        nlm: NlmeansOptions,
    ) -> Result<av_denoise::Algorithm, anyhow::Error> {
        let sigma_scale_is_set = self.hq_sigma_scale.is_some_and(|v| v != 1.0);

        match resolved.variant {
            Variant::Fast => {
                if self.hq_sigma.is_some()
                    || self.hq_no_auto_strength
                    || self.hq_no_noise_floor
                    || self.hq_no_temporal_confidence
                    || self.hq_thsad_scale.is_some()
                    || sigma_scale_is_set
                {
                    tracing::warn!("--hq-* options are ignored unless --variant hq is selected");
                }
                Ok(av_denoise::Algorithm::Nlmeans(nlm))
            },
            Variant::Hq => {
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

                Ok(av_denoise::Algorithm::NlmeansHq(NlmeansHqOptions {
                    nlm,
                    hq: av_denoise::HqParams {
                        auto_strength: !self.hq_no_auto_strength,
                        noise_floor: !self.hq_no_noise_floor,
                        sigma_override: self.hq_sigma.map(|s| s / 255.0),
                        temporal_confidence: !self.hq_no_temporal_confidence,
                        thsad_scale: self.hq_thsad_scale.unwrap_or(1.0),
                        sigma_scale: self.hq_sigma_scale.unwrap_or(1.0),
                    },
                }))
            },
        }
    }

    /// Turns the parsed flags plus the shared globals into the options
    /// the ingest pipeline takes.
    pub fn build_options(&self, globals: &Args) -> Result<CliOptions, anyhow::Error> {
        let resolved = self.resolve_preset(globals.preset);

        let mode = if resolved.temporal_radius == 0 {
            DenoisingMode::Spacial
        } else {
            DenoisingMode::Temporal {
                radius: resolved.temporal_radius,
            }
        };

        let prefilter = self
            .prefilter
            .as_deref()
            .map(parse_prefilter)
            .transpose()?
            .unwrap_or(PrefilterMode::None);
        let intent = resolve_channel_intent(&globals.channel_mode)?;

        let motion_compensation = if self.motion_compensation {
            if resolved.temporal_radius == 0 {
                tracing::warn!(
                    "--motion-compensation has no effect when --temporal-radius is 0; \
                     the spatial path doesn't use temporal neighbours"
                );
            }
            self.motion.to_motion_search().into()
        } else {
            if self.motion.any_set() {
                tracing::warn!("--mc-* options are ignored unless --motion-compensation is set");
            }
            MotionCompensationMode::None
        };

        // search_radius always has a resolved value (explicit flag or the
        // active preset), so it's always carried into the tuning override.
        let nlm = NlmeansOptions {
            prefilter,
            motion_compensation,
            tuning: NlmTuning {
                search_radius: Some(resolved.search_radius),
                patch_radius: self.patch_radius,
                strength: self.strength,
                self_weight: self.self_weight,
            },
        };

        Ok(CliOptions {
            accelerators: globals.accelerators.clone(),
            device: globals.device.clone(),
            intent,
            mode,
            algorithm: self.resolve_algorithm(resolved, nlm)?,
            luma_strength: self.luma_strength,
            chroma_strength: self.chroma_strength,
            // `nlmeans` has no grouping stage, so these stay unset
            // here. `Nl4dArgs::build_options` fills them in afterwards.
            luma_lambda_ht: None,
            chroma_lambda_ht: None,
            progress: globals.progress,
        })
    }
}

fn variant_for(preset: Preset) -> Variant {
    match preset {
        Preset::Veryfast => Variant::Fast,
        Preset::Fast | Preset::Base | Preset::Slow | Preset::Veryslow => Variant::Hq,
    }
}

fn temporal_radius_for(preset: Preset) -> u32 {
    match preset {
        Preset::Veryfast => 0,
        Preset::Fast => 1,
        Preset::Base => 2,
        Preset::Slow => 4,
        Preset::Veryslow => 8,
    }
}

fn search_radius_for(preset: Preset) -> u32 {
    match preset {
        Preset::Veryfast | Preset::Fast | Preset::Base => 2,
        Preset::Slow | Preset::Veryslow => 4,
    }
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
        "unknown prefilter '{s}', expected `none`, `nlm[:<strength_scale>]`, or `bilateral:<sigma_s>,<sigma_r>`"
    )
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::super::{Args, CliChannelMode, Command, InputSource, Preset};
    use super::*;

    /// Parses a full argv into the `nlmeans` subcommand's args plus the
    /// globals they resolve against.
    ///
    /// `extra` follows the subcommand because the family-owned flags
    /// only exist there. The globals parse in that position too.
    fn parse(extra: &[&str]) -> (Args, NlmeansArgs) {
        let mut argv = vec!["av-denoise", "nlmeans", "-i", "-"];
        argv.extend_from_slice(extra);

        let args = Args::parse_from(argv);
        let nlm = match &args.command {
            Command::Nlmeans(nlm) => NlmeansArgs::clone(nlm),
            Command::Nl4d(_) => unreachable!("parse() always builds a nlmeans argv"),
        };

        (args, nlm)
    }

    /// Parses an argv whose `-i`/`-W` values are supplied by the caller,
    /// unlike [`parse`] which pins the input to stdin.
    fn parse_input(extra: &[&str]) -> NlmeansArgs {
        let mut argv = vec!["av-denoise", "nlmeans"];
        argv.extend_from_slice(extra);

        let args = Args::parse_from(argv);
        match &args.command {
            Command::Nlmeans(nlm) => NlmeansArgs::clone(nlm),
            Command::Nl4d(_) => unreachable!("parse_input() always builds a nlmeans argv"),
        }
    }

    fn resolve(extra: &[&str]) -> ResolvedPreset {
        let (args, nlm) = parse(extra);
        nlm.resolve_preset(args.preset)
    }

    #[test]
    fn default_preset_is_base() {
        let resolved = resolve(&[]);
        assert_eq!(resolved.variant, Variant::Hq);
        assert_eq!(resolved.temporal_radius, 2);
        assert_eq!(resolved.search_radius, 2);
    }

    #[test]
    fn veryfast_matches_former_defaults() {
        let resolved = resolve(&["--preset", "veryfast"]);
        assert_eq!(resolved.variant, Variant::Fast);
        assert_eq!(resolved.temporal_radius, 0);
        assert_eq!(resolved.search_radius, 2);
    }

    #[test]
    fn fast_grid_row() {
        let resolved = resolve(&["--preset", "fast"]);
        assert_eq!(resolved.variant, Variant::Hq);
        assert_eq!(resolved.temporal_radius, 1);
        assert_eq!(resolved.search_radius, 2);
    }

    #[test]
    fn base_grid_row() {
        let resolved = resolve(&["--preset", "base"]);
        assert_eq!(resolved.variant, Variant::Hq);
        assert_eq!(resolved.temporal_radius, 2);
        assert_eq!(resolved.search_radius, 2);
    }

    #[test]
    fn slow_grid_row() {
        let resolved = resolve(&["--preset", "slow"]);
        assert_eq!(resolved.variant, Variant::Hq);
        assert_eq!(resolved.temporal_radius, 4);
        assert_eq!(resolved.search_radius, 4);
    }

    #[test]
    fn veryslow_grid_row() {
        let resolved = resolve(&["--preset", "veryslow"]);
        assert_eq!(resolved.variant, Variant::Hq);
        assert_eq!(resolved.temporal_radius, 8);
        assert_eq!(resolved.search_radius, 4);
    }

    #[test]
    fn explicit_variant_overrides_the_preset() {
        let resolved = resolve(&["--preset", "base", "--variant", "fast"]);
        assert_eq!(resolved.variant, Variant::Fast);
        assert_eq!(resolved.temporal_radius, 2);
        assert_eq!(resolved.search_radius, 2);
    }

    #[test]
    fn explicit_temporal_radius_overrides_the_preset() {
        let resolved = resolve(&["--preset", "base", "--temporal-radius", "6"]);
        assert_eq!(resolved.variant, Variant::Hq);
        assert_eq!(resolved.temporal_radius, 6);
        assert_eq!(resolved.search_radius, 2);
    }

    #[test]
    fn explicit_search_radius_overrides_the_preset() {
        let resolved = resolve(&["--preset", "veryslow", "--search-radius", "1"]);
        assert_eq!(resolved.variant, Variant::Hq);
        assert_eq!(resolved.temporal_radius, 8);
        assert_eq!(resolved.search_radius, 1);
    }

    #[test]
    fn globals_parse_before_the_subcommand() {
        let args = Args::parse_from(["av-denoise", "--preset", "slow", "nlmeans", "-i", "-"]);
        assert!(matches!(args.preset, Preset::Slow));
    }

    /// The five tests below check that every global flag is accepted
    /// before the `nlmeans` token. That is the shape the `Justfile`
    /// uses, passing `-A vulkan,cpu` ahead of the subcommand.
    ///
    /// Despite the name they do not guard `global = true`. Clap accepts
    /// `Args`'s own fields in the leading position either way, because
    /// `global` only starts to matter once the subcommand token has
    /// been read. Dropping `global = true` from `accelerators`,
    /// `device`, `channel_mode`, and `progress` in turn left all five
    /// tests passing.
    ///
    /// What `global = true` really controls is the trailing position,
    /// after `nlmeans`. The `*_parses_after_the_subcommand` tests
    /// further down cover that.
    ///
    /// The repeated `vulkan` is deliberate. What this pins is that the
    /// comma delimiter splits the value into a list, and `Vulkan` is
    /// the only accelerator variant a default build is guaranteed to
    /// have, so naming it twice tests the split without depending on a
    /// second backend feature being enabled.
    ///
    /// Feature-gated because it names the `Vulkan` accelerator variant,
    /// which only exists when its feature is enabled.
    #[cfg(feature = "vulkan")]
    #[test]
    fn accelerators_are_accepted_before_the_subcommand() {
        let args = Args::parse_from([
            "av-denoise",
            "--accelerators",
            "vulkan,vulkan",
            "nlmeans",
            "-i",
            "-",
        ]);
        assert_eq!(
            args.accelerators,
            vec![
                av_denoise::accelerate::Accelerator::Vulkan,
                av_denoise::accelerate::Accelerator::Vulkan
            ]
        );
    }

    /// Gated because it names the `Vulkan` accelerator variant, which
    /// only exists when the `vulkan` feature is enabled.
    #[cfg(feature = "vulkan")]
    #[test]
    fn short_accelerators_flag_is_accepted_before_the_subcommand() {
        let args = Args::parse_from(["av-denoise", "-A", "vulkan", "nlmeans", "-i", "-"]);
        assert_eq!(
            args.accelerators,
            vec![av_denoise::accelerate::Accelerator::Vulkan]
        );
    }

    #[test]
    fn device_is_accepted_before_the_subcommand() {
        let args = Args::parse_from(["av-denoise", "--device", "cpu", "nlmeans", "-i", "-"]);
        assert!(matches!(args.device, av_denoise::Device::Cpu));
    }

    #[test]
    fn channel_mode_is_accepted_before_the_subcommand() {
        let args = Args::parse_from(["av-denoise", "--channel-mode", "chroma", "nlmeans", "-i", "-"]);
        assert_eq!(args.channel_mode, vec![CliChannelMode::Chroma]);
    }

    #[test]
    fn progress_flag_is_accepted_before_the_subcommand() {
        let args = Args::parse_from(["av-denoise", "--progress", "nlmeans", "-i", "-"]);
        assert!(args.progress);
    }

    /// The other half of the rule. `--strength` is owned by the
    /// `nlmeans` subcommand rather than being global, so it must be
    /// rejected before `nlmeans` even though the globals above are
    /// accepted there.
    #[test]
    fn a_subcommand_owned_flag_is_rejected_before_the_subcommand() {
        let err = Args::try_parse_from(["av-denoise", "--strength", "1.2", "nlmeans", "-i", "-"])
            .expect_err("--strength is subcommand-owned, not global");
        assert!(err.to_string().contains("strength"), "got {err}");
    }

    /// This is the test that really depends on `--accelerators` being
    /// `global = true`. Without it, a flag placed after `nlmeans` is
    /// rejected as unknown to the subcommand's own parser. The
    /// `denoise-file` recipe in the `Justfile` passes user flags in
    /// exactly this position.
    ///
    /// Feature-gated because it names the `Vulkan` accelerator variant,
    /// which only exists when its feature is enabled.
    #[cfg(feature = "vulkan")]
    #[test]
    fn accelerators_parses_after_the_subcommand() {
        let (args, _) = parse(&["--accelerators", "vulkan"]);
        assert_eq!(
            args.accelerators,
            vec![av_denoise::accelerate::Accelerator::Vulkan]
        );
    }

    /// Same `global = true` dependency as
    /// [`accelerators_parses_after_the_subcommand`], for `--device`.
    #[test]
    fn device_parses_after_the_subcommand() {
        let (args, _) = parse(&["--device", "cpu"]);
        assert!(matches!(args.device, av_denoise::Device::Cpu));
    }

    /// Same `global = true` dependency as
    /// [`accelerators_parses_after_the_subcommand`], for
    /// `--channel-mode`.
    #[test]
    fn channel_mode_parses_after_the_subcommand() {
        let (args, _) = parse(&["--channel-mode", "chroma"]);
        assert_eq!(args.channel_mode, vec![CliChannelMode::Chroma]);
    }

    #[test]
    fn channel_mode_defaults_to_luma_and_chroma() {
        let (args, _) = parse(&[]);
        assert_eq!(
            args.channel_mode,
            vec![CliChannelMode::Luma, CliChannelMode::Chroma]
        );
    }

    #[test]
    fn default_channel_mode_resolves_to_the_luma_chroma_intent() {
        let (args, _) = parse(&[]);
        let intent = resolve_channel_intent(&args.channel_mode).expect("default should resolve");
        assert!(
            matches!(intent, crate::ingest::BinaryChannelIntent::LumaChroma),
            "expected LumaChroma, got {intent:?}",
        );
    }

    #[test]
    fn variant_flag_is_case_insensitive() {
        let (_, nlm) = parse(&["--variant", "HQ"]);
        assert_eq!(nlm.variant, Some(Variant::Hq));
    }

    #[test]
    fn hq_sigma_scale_parses_to_the_typed_value() {
        let (_, nlm) = parse(&["--hq-sigma-scale", "2.0"]);
        assert_eq!(nlm.hq_sigma_scale, Some(2.0));
    }

    #[test]
    fn unset_hq_sigma_scale_resolves_to_the_library_default_of_one() {
        let (args, nlm) = parse(&["--variant", "hq"]);
        let resolved = nlm.resolve_preset(args.preset);
        let algorithm = nlm
            .resolve_algorithm(resolved, NlmeansOptions::default())
            .expect("resolution should succeed");

        match algorithm {
            av_denoise::Algorithm::NlmeansHq(opts) => assert_eq!(opts.hq.sigma_scale, 1.0),
            other => panic!("expected NlmeansHq, got {other:?}"),
        }
    }

    #[test]
    fn hq_flags_on_the_fast_variant_still_resolve_to_nlmeans() {
        let (args, nlm) = parse(&["--variant", "fast", "--hq-sigma-scale", "2.0"]);
        let resolved = nlm.resolve_preset(args.preset);
        let algorithm = nlm
            .resolve_algorithm(resolved, NlmeansOptions::default())
            .expect("resolution should succeed");

        assert!(matches!(algorithm, av_denoise::Algorithm::Nlmeans(_)));
    }

    #[test]
    fn hq_sigma_and_sigma_scale_both_resolve() {
        let (args, nlm) = parse(&["--variant", "hq", "--hq-sigma", "6", "--hq-sigma-scale", "2.0"]);
        let resolved = nlm.resolve_preset(args.preset);
        let algorithm = nlm
            .resolve_algorithm(resolved, NlmeansOptions::default())
            .expect("resolution should succeed");

        match algorithm {
            av_denoise::Algorithm::NlmeansHq(opts) => {
                assert!(
                    matches!(opts.hq.sigma_override, Some(s) if (s - 6.0 / 255.0).abs() < f32::EPSILON),
                    "expected sigma_override Some(6/255), got {:?}",
                    opts.hq.sigma_override,
                );
                assert_eq!(opts.hq.sigma_scale, 2.0);
            },
            other => panic!("expected NlmeansHq, got {other:?}"),
        }
    }

    #[test]
    fn out_of_range_hq_sigma_is_rejected() {
        let (args, nlm) = parse(&["--variant", "hq", "--hq-sigma", "300"]);
        let resolved = nlm.resolve_preset(args.preset);
        let err = nlm
            .resolve_algorithm(resolved, NlmeansOptions::default())
            .expect_err("300 is out of range");

        assert!(err.to_string().contains("--hq-sigma"), "got {err}");
    }

    #[test]
    fn progress_defaults_to_false() {
        let (args, _) = parse(&[]);
        assert!(!args.progress);
    }

    #[test]
    fn progress_flag_sets_it_true() {
        let (args, _) = parse(&["--progress"]);
        assert!(args.progress);
    }

    #[test]
    fn none_and_empty_prefilters() {
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

    #[test]
    fn the_old_subcommands_are_rejected() {
        assert!(Args::try_parse_from(["av-denoise", "stdin"]).is_err());
        assert!(Args::try_parse_from(["av-denoise", "file", "-i", "noisy.mkv"]).is_err());
    }

    #[test]
    fn a_path_parses_as_a_file_source() {
        let nlm = parse_input(&["--input", "noisy.mkv"]);
        assert_eq!(
            nlm.common.input,
            InputSource::File(std::path::PathBuf::from("noisy.mkv"))
        );
    }

    #[test]
    fn a_dash_parses_as_stdin() {
        let nlm = parse_input(&["-i", "-"]);
        assert_eq!(nlm.common.input, InputSource::Stdin);
    }

    #[test]
    fn a_pipe_parses_as_a_descriptor() {
        let nlm = parse_input(&["-i", "pipe:3"]);
        assert_eq!(nlm.common.input, InputSource::Fd(3));
    }

    #[test]
    fn an_unreadable_descriptor_fails_to_parse() {
        let err = Args::try_parse_from(["av-denoise", "nlmeans", "-i", "pipe:1"])
            .expect_err("pipe:1 is our own stdout");
        assert!(err.to_string().contains("stdout"), "got {err}");
    }

    #[test]
    fn workers_is_unset_by_default() {
        let nlm = parse_input(&["-i", "noisy.mkv"]);
        assert_eq!(nlm.common.workers, None);
    }

    #[test]
    fn workers_carries_the_typed_value() {
        let nlm = parse_input(&["-i", "noisy.mkv", "--workers", "4"]);
        assert_eq!(nlm.common.workers, Some(4));
    }
}

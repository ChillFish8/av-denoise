use super::Args;
use super::nlmeans::{NlmeansArgs, Variant};
use crate::ingest::{BinaryChannelIntent, CliOptions};

/// Flags for `nl4d`, which groups 8x8 patches across the
/// motion-compensated temporal window itself, rather than filtering
/// with `nlmeans` first and grouping within one frame afterward the way
/// `nl3d` does.
///
/// Everything `nlmeans` accepts is inherited through `nlm`. The fields
/// below tune the temporal grouping and shrinkage stage that runs on
/// top of it.
#[derive(Debug, Clone, clap::Args)]
pub struct Nl4dArgs {
    #[command(flatten)]
    pub nlm: NlmeansArgs,

    /// Half-width of the refine window searched around each neighbour
    /// frame's motion-predicted position.
    ///
    /// In 1..=4. Library default is 2.
    #[arg(long)]
    pub refine: Option<u32>,

    /// Half-width of the spatial candidate window searched in the
    /// centre frame.
    ///
    /// In 1..=16. Library default is 9.
    #[arg(long)]
    pub spatial_radius: Option<u32>,

    /// How aggressively the temporal grouping's hard-threshold stage
    /// zeroes out small transform coefficients.
    ///
    /// Multiplies the propagated coefficient sigma. The library default
    /// is now per plane rather than a single shared number, see
    /// `nl4d_default_lambda_ht`'s docs for both values and how
    /// differently certain they are. Setting this flag applies the same
    /// explicit value to both planes, overriding the per-plane default
    /// for each. Applies to both planes unless `--luma-lambda-ht` or
    /// `--chroma-lambda-ht` is set.
    #[arg(long)]
    pub lambda_ht: Option<f32>,

    /// `--lambda-ht` override for the brightness plane only.
    ///
    /// Falls back to `--lambda-ht` (or the calibrated per-plane library
    /// default) when not set.
    ///
    /// Ignored when luma is not being denoised, or when `--channel-mode
    /// yuv` is used.
    #[arg(long)]
    pub luma_lambda_ht: Option<f32>,

    /// `--lambda-ht` override for the colour planes only.
    ///
    /// Falls back to `--lambda-ht` (or the calibrated per-plane library
    /// default) when not set.
    ///
    /// Ignored when chroma is not being denoised, or when
    /// `--channel-mode yuv` is used.
    #[arg(long)]
    pub chroma_lambda_ht: Option<f32>,

    /// The confidence floor below which a whole neighbour block is
    /// skipped rather than scored.
    ///
    /// In [0, 1). Library default is 0.05. Only affects how much
    /// compute a submit spends, never which candidates are admitted
    /// once they are scored.
    #[arg(long)]
    pub c_min: Option<f32>,
}

impl Nl4dArgs {
    /// Turns the parsed flags plus the shared globals into the options
    /// the ingest pipeline takes.
    ///
    /// nl4d always runs the hq front end with an active
    /// motion-compensated ring, because the temporal grouping kernel
    /// reads the motion field and confidence scores that front end
    /// builds. A resolved variant of `fast`, whether from an explicit
    /// `--variant fast` or from a preset that resolves to it, is
    /// rejected here rather than left to fail deep inside construction,
    /// the same way `nl3d` rejects it. Motion compensation is turned on
    /// unconditionally rather than left to `--motion-compensation`,
    /// which defaults off and has no preset-driven default of its own.
    pub fn build_options(&self, globals: &Args) -> Result<CliOptions, anyhow::Error> {
        let resolved = self.nlm.resolve_preset(globals.preset);
        if resolved.variant == Variant::Fast {
            anyhow::bail!(
                "nl4d requires the hq front end, but the resolved variant is fast. Pass \
                 `--variant hq` explicitly, or drop `--variant` and pick a `--preset` above \
                 `veryfast`, which is the only preset that resolves to fast"
            );
        }

        let mut opts = self.nlm.build_options(globals)?;

        let hq = match opts.algorithm {
            av_denoise::Algorithm::NlmeansHq(hq) => hq,
            other => unreachable!(
                "resolved variant is hq, so resolve_algorithm always returns NlmeansHq, got {other:?}"
            ),
        };

        // The temporal grouping kernel reads the motion field and
        // confidence scores that `submit_machinery` only builds when
        // both motion compensation and temporal confidence are active
        // (see `Nl4dParams::validate`), so this is forced on here
        // rather than left to `--motion-compensation`.
        opts.motion_compensation = av_denoise::MotionCompensationMode::Mvtools {
            blksize: self.nlm.mc_blksize.unwrap_or(16),
            overlap: self.nlm.mc_overlap.unwrap_or(8),
            search_radius: self.nlm.mc_search.unwrap_or(4),
            pyramid_levels: self.nlm.mc_pyramid_levels.unwrap_or(2),
            estimation: av_denoise::MotionEstimation::default(),
        };

        let defaults = av_denoise::Nl4dOptions::default();

        opts.algorithm = av_denoise::Algorithm::Nl4d(av_denoise::Nl4dOptions {
            hq,
            // Has to equal the temporal radius `resolve_preset` picked
            // above, which is what `opts.mode` was just built from.
            // Both the front end's frame ring and the outer
            // `Denoiser`'s own push/flush bookkeeping depend on that
            // radius agreeing everywhere it is read, see
            // `Nl4dOptions`'s doc comment in `src/denoiser.rs`.
            temporal_radius: resolved.temporal_radius,
            refine: self.refine.unwrap_or(defaults.refine),
            spatial_radius: self.spatial_radius.unwrap_or(defaults.spatial_radius),
            // Left unresolved when unset, rather than eagerly picked
            // from `defaults` here, because the calibrated default now
            // depends on which plane is being denoised, which is not
            // known yet at this point. `CliOptions::algorithm_for`
            // (src/bin/ingest.rs) applies `--luma-`/`--chroma-lambda-ht`
            // on top of this once the plane is known, and construction
            // itself picks the calibrated per-plane default for
            // whatever is still unset.
            lambda_ht: self.lambda_ht,
            c_min: self.c_min.unwrap_or(defaults.c_min),
        });

        opts.luma_lambda_ht = self.luma_lambda_ht;
        opts.chroma_lambda_ht = self.chroma_lambda_ht;

        self.warn_on_dead_per_plane_flags(opts.intent);

        Ok(opts)
    }

    /// Warns when a per-plane override was set but the resolved
    /// `--channel-mode` means it has no effect, the same way
    /// `Nl3dArgs::warn_on_dead_per_plane_flags` does.
    fn warn_on_dead_per_plane_flags(&self, intent: BinaryChannelIntent) {
        match intent {
            BinaryChannelIntent::Luma if self.chroma_lambda_ht.is_some() => {
                tracing::warn!("--chroma-lambda-ht is ignored when chroma is not being denoised");
            },
            BinaryChannelIntent::Chroma if self.luma_lambda_ht.is_some() => {
                tracing::warn!("--luma-lambda-ht is ignored when luma is not being denoised");
            },
            BinaryChannelIntent::YuvFused if self.luma_lambda_ht.is_some() || self.chroma_lambda_ht.is_some() => {
                tracing::warn!(
                    "--luma-lambda-ht and --chroma-lambda-ht are ignored when --channel-mode yuv is used"
                );
            },
            _ => {},
        }
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::super::Command;
    use super::*;

    /// Parses a full argv into the `nl4d` subcommand's args plus the
    /// globals they resolve against.
    fn parse(extra: &[&str]) -> (Args, Nl4dArgs) {
        let mut argv = vec!["av-denoise", "nl4d", "-i", "-"];
        argv.extend_from_slice(extra);

        let args = Args::parse_from(argv);
        let nl4d = match &args.command {
            Command::Nl4d(nl4d) => Nl4dArgs::clone(nl4d),
            Command::Nlmeans(_) => unreachable!("parse() always builds a nl4d argv"),
            Command::Nl3d(_) => unreachable!("parse() always builds a nl4d argv"),
        };

        (args, nl4d)
    }

    #[test]
    fn nl4d_subcommand_parses_with_no_extra_flags() {
        let (_, nl4d) = parse(&[]);
        assert_eq!(nl4d.refine, None);
        assert_eq!(nl4d.spatial_radius, None);
        assert_eq!(nl4d.lambda_ht, None);
        assert_eq!(nl4d.luma_lambda_ht, None);
        assert_eq!(nl4d.chroma_lambda_ht, None);
        assert_eq!(nl4d.c_min, None);
    }

    #[test]
    fn nl4d_specific_flags_parse_to_the_typed_values() {
        let (_, nl4d) = parse(&[
            "--refine",
            "3",
            "--spatial-radius",
            "12",
            "--lambda-ht",
            "2.0",
            "--c-min",
            "0.1",
        ]);

        assert_eq!(nl4d.refine, Some(3));
        assert_eq!(nl4d.spatial_radius, Some(12));
        assert!((nl4d.lambda_ht.unwrap() - 2.0).abs() < f32::EPSILON);
        assert!((nl4d.c_min.unwrap() - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn nlmeans_flags_flatten_into_nl4d() {
        let (_, nl4d) = parse(&["--temporal-radius", "3", "--search-radius", "1"]);
        assert_eq!(nl4d.nlm.temporal_radius, Some(3));
        assert_eq!(nl4d.nlm.search_radius, Some(1));
    }

    #[test]
    fn explicit_variant_fast_is_rejected() {
        let (args, nl4d) = parse(&["--variant", "fast"]);
        let err = nl4d
            .build_options(&args)
            .expect_err("fast must be rejected under nl4d");
        assert!(err.to_string().contains("fast"), "got {err}");
    }

    #[test]
    fn preset_resolving_to_fast_is_rejected() {
        let (args, nl4d) = parse(&["--preset", "veryfast"]);
        let err = nl4d
            .build_options(&args)
            .expect_err("a preset resolving to fast must be rejected under nl4d");
        assert!(err.to_string().contains("fast"), "got {err}");
    }

    #[test]
    fn explicit_variant_hq_is_accepted() {
        let (args, nl4d) = parse(&["--variant", "hq"]);
        let opts = nl4d.build_options(&args);
        assert!(opts.is_ok(), "explicit hq should be accepted under nl4d: {opts:?}");
    }

    #[test]
    fn default_preset_builds_the_nl4d_algorithm_with_library_defaults() {
        let (args, nl4d) = parse(&[]);
        let opts = nl4d
            .build_options(&args)
            .expect("default preset should resolve for nl4d");

        match opts.algorithm {
            av_denoise::Algorithm::Nl4d(nl4d_opts) => {
                let defaults = av_denoise::Nl4dOptions::default();
                assert_eq!(
                    nl4d_opts.temporal_radius, 2,
                    "base preset resolves to temporal_radius 2"
                );
                assert_eq!(nl4d_opts.refine, defaults.refine);
                assert_eq!(nl4d_opts.spatial_radius, defaults.spatial_radius);
                assert_eq!(
                    nl4d_opts.lambda_ht, defaults.lambda_ht,
                    "unset --lambda-ht should stay None here, resolved later per plane"
                );
                assert!((nl4d_opts.c_min - defaults.c_min).abs() < f32::EPSILON);
            },
            other => panic!("expected Algorithm::Nl4d, got {other:?}"),
        }
    }

    #[test]
    fn motion_compensation_is_forced_on_regardless_of_the_flag() {
        let (args, nl4d) = parse(&[]);
        let opts = nl4d.build_options(&args).expect("build_options should succeed");

        assert!(
            matches!(
                opts.motion_compensation,
                av_denoise::MotionCompensationMode::Mvtools { .. }
            ),
            "nl4d must force motion compensation on even when --motion-compensation is not \
             passed, got {:?}",
            opts.motion_compensation
        );
    }

    #[test]
    fn explicit_collab_flags_override_the_library_defaults() {
        let (args, nl4d) = parse(&["--refine", "4", "--spatial-radius", "6", "--c-min", "0.2"]);
        let opts = nl4d.build_options(&args).expect("build_options should succeed");

        match opts.algorithm {
            av_denoise::Algorithm::Nl4d(nl4d_opts) => {
                assert_eq!(nl4d_opts.refine, 4);
                assert_eq!(nl4d_opts.spatial_radius, 6);
                assert!((nl4d_opts.c_min - 0.2).abs() < f32::EPSILON);
            },
            other => panic!("expected Algorithm::Nl4d, got {other:?}"),
        }
    }

    #[test]
    fn per_plane_lambda_ht_flags_parse_to_the_typed_values() {
        let (_, nl4d) = parse(&["--luma-lambda-ht", "2.0", "--chroma-lambda-ht", "3.5"]);
        assert!((nl4d.luma_lambda_ht.unwrap() - 2.0).abs() < f32::EPSILON);
        assert!((nl4d.chroma_lambda_ht.unwrap() - 3.5).abs() < f32::EPSILON);
    }

    /// `build_options` carries the two per-plane `lambda_ht` overrides
    /// straight through onto `CliOptions`, unresolved, the same shape
    /// nl3d's per-plane overrides use.
    /// `CliOptions::algorithm_for` (`src/bin/ingest.rs`) is what
    /// actually resolves them per plane, so this test only checks the
    /// flow into `CliOptions`.
    #[test]
    fn luma_lambda_ht_alone_flows_into_cli_options_luma_field_only() {
        let (args, nl4d) = parse(&["--luma-lambda-ht", "2.0"]);
        let opts = nl4d.build_options(&args).expect("build_options should succeed");

        assert!((opts.luma_lambda_ht.unwrap() - 2.0).abs() < f32::EPSILON);
        assert_eq!(opts.chroma_lambda_ht, None);
    }

    #[test]
    fn chroma_lambda_ht_alone_flows_into_cli_options_chroma_field_only() {
        let (args, nl4d) = parse(&["--chroma-lambda-ht", "3.5"]);
        let opts = nl4d.build_options(&args).expect("build_options should succeed");

        assert!((opts.chroma_lambda_ht.unwrap() - 3.5).abs() < f32::EPSILON);
        assert_eq!(opts.luma_lambda_ht, None);
    }

    #[test]
    fn both_planes_lambda_ht_overrides_flow_into_cli_options_independently() {
        let (args, nl4d) = parse(&["--luma-lambda-ht", "2.0", "--chroma-lambda-ht", "3.5"]);
        let opts = nl4d.build_options(&args).expect("build_options should succeed");

        assert!((opts.luma_lambda_ht.unwrap() - 2.0).abs() < f32::EPSILON);
        assert!((opts.chroma_lambda_ht.unwrap() - 3.5).abs() < f32::EPSILON);
    }

    #[test]
    fn unset_per_plane_overrides_leave_cli_options_fields_none() {
        let (args, nl4d) = parse(&[]);
        let opts = nl4d.build_options(&args).expect("build_options should succeed");

        assert_eq!(opts.luma_lambda_ht, None);
        assert_eq!(opts.chroma_lambda_ht, None);
    }
}

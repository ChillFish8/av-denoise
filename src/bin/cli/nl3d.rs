use super::Args;
use super::nlmeans::{NlmeansArgs, Variant};
use crate::ingest::CliOptions;

/// Flags for `nl3d`, the non-local means front end followed by a
/// collaborative-filter cleanup pass.
///
/// Everything `nlmeans` accepts is inherited through `nlm`. The fields
/// below are the collaborative stage's own knobs.
#[derive(Debug, Clone, clap::Args)]
pub struct Nl3dArgs {
    #[command(flatten)]
    pub nlm: NlmeansArgs,

    /// How gently the front end filters before the collaborative stage
    /// cleans up what it leaves behind.
    ///
    /// This multiplies the hq front end's calibrated strength. A value
    /// below 1 leaves more structured noise in the front end's output,
    /// which the collaborative stage then removes through its own
    /// grouped shrinkage. That recovers more real detail than pushing
    /// the front end harder on its own would. Library default is 0.5,
    /// calibrated by sweep, see `Nl3dOptions`'s docs for the numbers.
    #[arg(long)]
    pub front_strength_scale: Option<f32>,

    /// Largest number of matching patches grouped into one stack for the
    /// collaborative filter.
    ///
    /// Must be a power of two, up to 8. Larger stacks pool more
    /// candidates together, which removes more noise but costs more GPU
    /// work per group. Library default is 8.
    #[arg(long)]
    pub k_max: Option<u32>,

    /// How closely a candidate patch has to match the reference patch
    /// before it joins the reference's group.
    ///
    /// Multiplies the noise floor each patch is judged against. Lower
    /// values admit only close matches, higher values admit more distant
    /// ones. Library default is 3.0.
    #[arg(long)]
    pub tau_match: Option<f32>,

    /// How aggressively the collaborative filter's first stage zeroes
    /// out small transform coefficients.
    ///
    /// Multiplies the noise sigma each coefficient is compared against.
    /// Higher values remove more noise but risk taking fine detail with
    /// it. Library default is 2.7, checked by sweep against 2.0 and 3.5,
    /// see `Nl3dOptions`'s docs for the numbers.
    #[arg(long)]
    pub lambda_ht: Option<f32>,

    /// Nudges the noise level the collaborative stage shrinks its
    /// coefficients by.
    ///
    /// The collaborative stage estimates how much noise the front end
    /// left behind and shrinks by that amount. `1.9`, the library
    /// default, corrects for the front end's real residual noise running
    /// almost twice the analytic estimate, see `Nl3dOptions`'s docs for
    /// the sweep this came from. Raise it further if the result still
    /// looks noisy, lower it if fine detail is getting scrubbed.
    #[arg(long)]
    pub residual_sigma_scale: Option<f32>,
}

impl Nl3dArgs {
    /// Turns the parsed flags plus the shared globals into the options
    /// the ingest pipeline takes.
    ///
    /// nl3d always runs the hq front end, because the collaborative
    /// stage needs the front end's estimated noise sigma to shrink its
    /// own coefficients by. A resolved variant of `fast`, whether from
    /// an explicit `--variant fast` or from a preset that resolves to
    /// it, is rejected here rather than left to fail deep inside
    /// construction.
    pub fn build_options(&self, globals: &Args) -> Result<CliOptions, anyhow::Error> {
        let resolved = self.nlm.resolve_preset(globals.preset);
        if resolved.variant == Variant::Fast {
            anyhow::bail!(
                "nl3d requires the hq front end, but the resolved variant is fast. Pass \
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

        let defaults = av_denoise::Nl3dOptions::default();

        opts.algorithm = av_denoise::Algorithm::Nl3d(av_denoise::Nl3dOptions {
            hq,
            front_strength_scale: self.front_strength_scale.unwrap_or(defaults.front_strength_scale),
            k_max: self.k_max.unwrap_or(defaults.k_max),
            tau_match: self.tau_match.unwrap_or(defaults.tau_match),
            lambda_ht: self.lambda_ht.unwrap_or(defaults.lambda_ht),
            residual_sigma_scale: self.residual_sigma_scale.unwrap_or(defaults.residual_sigma_scale),
        });

        Ok(opts)
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::super::Command;
    use super::*;

    /// Parses a full argv into the `nl3d` subcommand's args plus the
    /// globals they resolve against.
    fn parse(extra: &[&str]) -> (Args, Nl3dArgs) {
        let mut argv = vec!["av-denoise", "nl3d", "-i", "-"];
        argv.extend_from_slice(extra);

        let args = Args::parse_from(argv);
        let nl3d = match &args.command {
            Command::Nl3d(nl3d) => Nl3dArgs::clone(nl3d),
            Command::Nlmeans(_) => unreachable!("parse() always builds a nl3d argv"),
        };

        (args, nl3d)
    }

    #[test]
    fn nl3d_subcommand_parses_with_no_extra_flags() {
        let (_, nl3d) = parse(&[]);
        assert_eq!(nl3d.k_max, None);
        assert_eq!(nl3d.front_strength_scale, None);
        assert_eq!(nl3d.tau_match, None);
        assert_eq!(nl3d.lambda_ht, None);
        assert_eq!(nl3d.residual_sigma_scale, None);
    }

    #[test]
    fn nl3d_specific_flags_parse_to_the_typed_values() {
        let (_, nl3d) = parse(&[
            "--k-max",
            "8",
            "--front-strength-scale",
            "0.7",
            "--tau-match",
            "2.5",
            "--lambda-ht",
            "2.0",
            "--residual-sigma-scale",
            "1.2",
        ]);

        assert_eq!(nl3d.k_max, Some(8));
        assert!((nl3d.front_strength_scale.unwrap() - 0.7).abs() < f32::EPSILON);
        assert!((nl3d.tau_match.unwrap() - 2.5).abs() < f32::EPSILON);
        assert!((nl3d.lambda_ht.unwrap() - 2.0).abs() < f32::EPSILON);
        assert!((nl3d.residual_sigma_scale.unwrap() - 1.2).abs() < f32::EPSILON);
    }

    #[test]
    fn nlmeans_flags_flatten_into_nl3d() {
        let (_, nl3d) = parse(&["--temporal-radius", "3", "--search-radius", "1"]);
        assert_eq!(nl3d.nlm.temporal_radius, Some(3));
        assert_eq!(nl3d.nlm.search_radius, Some(1));
    }

    #[test]
    fn explicit_variant_fast_is_rejected() {
        let (args, nl3d) = parse(&["--variant", "fast"]);
        let err = nl3d
            .build_options(&args)
            .expect_err("fast must be rejected under nl3d");
        assert!(err.to_string().contains("fast"), "got {err}");
    }

    #[test]
    fn preset_resolving_to_fast_is_rejected() {
        let (args, nl3d) = parse(&["--preset", "veryfast"]);
        let err = nl3d
            .build_options(&args)
            .expect_err("a preset resolving to fast must be rejected under nl3d");
        assert!(err.to_string().contains("fast"), "got {err}");
    }

    #[test]
    fn default_preset_builds_the_nl3d_algorithm_with_library_defaults() {
        let (args, nl3d) = parse(&[]);
        let opts = nl3d
            .build_options(&args)
            .expect("default preset should resolve for nl3d");

        match opts.algorithm {
            av_denoise::Algorithm::Nl3d(nl3d_opts) => {
                let defaults = av_denoise::Nl3dOptions::default();
                assert!(
                    (nl3d_opts.front_strength_scale - defaults.front_strength_scale).abs() < f32::EPSILON
                );
                assert_eq!(nl3d_opts.k_max, defaults.k_max);
                assert!((nl3d_opts.tau_match - defaults.tau_match).abs() < f32::EPSILON);
                assert!((nl3d_opts.lambda_ht - defaults.lambda_ht).abs() < f32::EPSILON);
                assert!(
                    (nl3d_opts.residual_sigma_scale - defaults.residual_sigma_scale).abs() < f32::EPSILON
                );
            },
            other => panic!("expected Algorithm::Nl3d, got {other:?}"),
        }
    }

    #[test]
    fn explicit_collab_flags_override_the_library_defaults() {
        let (args, nl3d) = parse(&["--k-max", "4", "--tau-match", "1.5"]);
        let opts = nl3d.build_options(&args).expect("build_options should succeed");

        match opts.algorithm {
            av_denoise::Algorithm::Nl3d(nl3d_opts) => {
                assert_eq!(nl3d_opts.k_max, 4);
                assert!((nl3d_opts.tau_match - 1.5).abs() < f32::EPSILON);
            },
            other => panic!("expected Algorithm::Nl3d, got {other:?}"),
        }
    }

    #[test]
    fn explicit_variant_hq_is_accepted() {
        let (args, nl3d) = parse(&["--variant", "hq"]);
        let opts = nl3d.build_options(&args);
        assert!(
            opts.is_ok(),
            "explicit hq should be accepted under nl3d: {opts:?}"
        );
    }
}

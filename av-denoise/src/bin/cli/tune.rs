use std::fmt::Write;

use av_denoise::tune::{
    BudgetRequest,
    DeviceIdentity,
    TuneOutcome,
    TuneRequest,
    TuneResult,
    TuneStore,
    selected_device,
};
use av_denoise_core::tune::{SiteRanking, choose};

use super::{Args, Nl4dArgs, NlmeansArgs};
use crate::pipeline::{budget_request, tune_input};

/// Tunes kernels for this device ahead of time, or shows or clears what is stored.
#[derive(Debug, clap::Args)]
pub struct TuneArgs {
    /// Tunes again even when results are already stored.
    #[arg(long, conflicts_with_all = ["show", "clear"])]
    pub force: bool,

    /// Prints the stored results for the selected device and exits.
    #[arg(long, conflicts_with = "clear")]
    pub show: bool,

    /// Deletes the stored results for the selected device and exits.
    #[arg(long)]
    pub clear: bool,

    /// The algorithm to tune, with the same options and input a denoising run takes.
    #[command(subcommand)]
    pub algorithm: Option<TuneAlgorithm>,
}

#[derive(Debug, clap::Subcommand)]
pub enum TuneAlgorithm {
    /// Tunes the kernels `av-denoise nlmeans` runs with these options.
    Nlmeans(NlmeansArgs),
    /// Tunes the kernels `av-denoise nl4d` runs with these options.
    Nl4d(Nl4dArgs),
}

/// Runs `av-denoise tune`.
pub fn run_tune(tune: &TuneArgs, globals: &Args) -> Result<(), anyhow::Error> {
    if tune.show || tune.clear {
        return show_or_clear(tune, globals);
    }

    let Some(algorithm) = &tune.algorithm else {
        anyhow::bail!("name an algorithm to tune, for example `av-denoise tune nl4d -i clip.mkv`");
    };

    let (opts, input) = match algorithm {
        TuneAlgorithm::Nlmeans(nlmeans) => (nlmeans.build_options(globals)?, &nlmeans.common.input),
        TuneAlgorithm::Nl4d(nl4d) => (nl4d.build_options(globals)?, &nl4d.common.input),
    };

    let budget = budget_request(opts.tune_budget, BudgetRequest::Full);
    let request = TuneRequest {
        budget,
        force: tune.force,
    };
    let report = tune_input(&opts.planes, input, &request)?;

    match report.outcome {
        TuneOutcome::Tuned => tracing::info!(
            keys = report.tuned_keys,
            partial_keys = report.partial_keys,
            seconds = report.elapsed.as_secs_f64(),
            compile_seconds = report.compile.as_secs_f64(),
            saved = report.saved,
            "tuning finished"
        ),
        TuneOutcome::AlreadyTuned => {
            tracing::info!("this configuration is already tuned, pass --force to tune it again")
        },
        TuneOutcome::Skipped(reason) => anyhow::bail!("tuning did not run: {reason}"),
    }

    Ok(())
}

fn show_or_clear(tune: &TuneArgs, globals: &Args) -> Result<(), anyhow::Error> {
    let device = selected_device(&globals.accelerators, &globals.device)?;
    let Some(store) = TuneStore::open_default() else {
        anyhow::bail!("kernel caching is off, so no tuning results are stored");
    };

    if tune.clear {
        let removed = store.clear(&device.identity)?;
        let message = if removed {
            "deleted the stored tuning results for this device"
        } else {
            "no tuning results were stored for this device"
        };
        println!("{message}");

        return Ok(());
    }

    let result = store.load(&device.identity);
    let table = format_stored(&device.identity, &result, false);
    print!("{table}");

    Ok(())
}

/// Renders the stored winner of every site key, one row per key.
///
/// A key with a partial ranking is marked `(partial)`, with how many of its candidates the compile
/// ceiling left when that is known.
pub fn format_stored(identity: &DeviceIdentity, result: &TuneResult, deterministic: bool) -> String {
    let mut table = String::new();
    let _ = writeln!(
        table,
        "{} on {} ({})",
        identity.adapter, identity.backend, identity.driver
    );

    if result.sites.is_empty() {
        let _ = writeln!(table, "no tuned launch sites are stored");
        return table;
    }

    for (key, ranking) in &result.sites {
        let chosen = choose(&ranking.candidates, deterministic);
        let Some(record) = chosen else {
            continue;
        };

        let millis = record.median_ns as f64 / 1_000_000.0;
        let note = partial_note(ranking);
        let _ = writeln!(table, "  {key}  {}  {millis:.2} ms{note}", record.name);
    }

    table
}

fn partial_note(ranking: &SiteRanking) -> String {
    if !ranking.partial {
        return String::new();
    }

    let progress = ranking.progress;
    if progress.total == 0 {
        return "  (partial)".to_string();
    }

    format!("  (partial, {} of {} left)", progress.left, progress.total)
}

#[cfg(test)]
mod tests {
    use av_denoise::TuningMode;
    use av_denoise::tune::{DeviceIdentity, TuneResult};
    use av_denoise_core::tune::{CandidateRecord, CompileProgress, FidelityClass, SiteKey, SiteRanking};
    use clap::Parser;

    use super::*;
    use crate::cli::{Args, Command};

    #[test]
    fn denoise_commands_take_the_tuning_flags() {
        let args = Args::try_parse_from([
            "av-denoise",
            "nl4d",
            "-i",
            "clip.mkv",
            "--tune",
            "cached",
            "--deterministic",
            "--tune-budget",
            "30",
        ])
        .expect("the flags parse");

        let Command::Nl4d(nl4d) = args.command else {
            panic!("expected nl4d");
        };
        assert_eq!(nl4d.common.tune, TuningMode::Cached);
        assert!(nl4d.common.deterministic);
        assert_eq!(nl4d.common.tune_budget, Some(30.0));
    }

    #[test]
    fn tuning_defaults_to_auto() {
        let args =
            Args::try_parse_from(["av-denoise", "nlmeans", "-i", "clip.mkv"]).expect("the flags parse");

        let Command::Nlmeans(nlmeans) = args.command else {
            panic!("expected nlmeans");
        };
        assert_eq!(nlmeans.common.tune, TuningMode::Auto);
        assert!(!nlmeans.common.deterministic);
        assert_eq!(nlmeans.common.tune_budget, None);
    }

    #[test]
    fn an_unknown_tuning_mode_is_rejected() {
        let result = Args::try_parse_from(["av-denoise", "nl4d", "-i", "clip.mkv", "--tune", "sometimes"]);

        assert!(result.is_err());
    }

    #[test]
    fn a_zero_tuning_budget_is_accepted() {
        let args = Args::try_parse_from(["av-denoise", "nl4d", "-i", "clip.mkv", "--tune-budget", "0"])
            .expect("zero parses");

        let Command::Nl4d(nl4d) = args.command else {
            panic!("expected nl4d");
        };
        assert_eq!(
            nl4d.common.checked_tune_budget().expect("zero is valid"),
            Some(0.0)
        );
    }

    #[test]
    fn bad_tuning_budgets_are_rejected_when_parsing() {
        for bad in ["-5", "-0.5", "NaN", "inf", "-inf", "abc"] {
            let result = Args::try_parse_from(["av-denoise", "nl4d", "-i", "clip.mkv", "--tune-budget", bad]);

            assert!(result.is_err(), "{bad} should be rejected");
        }
    }

    #[test]
    fn a_huge_tuning_budget_is_rejected_when_parsing() {
        let result = Args::try_parse_from(["av-denoise", "nl4d", "-i", "clip.mkv", "--tune-budget", "1e300"]);

        assert!(result.is_err());
    }

    #[test]
    fn checking_a_budget_set_by_hand_rejects_it_too() {
        let mut args =
            Args::try_parse_from(["av-denoise", "nl4d", "-i", "clip.mkv"]).expect("the flags parse");
        let Command::Nl4d(nl4d) = &mut args.command else {
            panic!("expected nl4d");
        };

        nl4d.common.tune_budget = Some(f64::NAN);
        assert!(nl4d.common.checked_tune_budget().is_err());

        nl4d.common.tune_budget = Some(-1.0);
        assert!(nl4d.common.checked_tune_budget().is_err());
    }

    #[test]
    fn the_tune_subcommand_takes_an_algorithm_and_its_options() {
        let args = Args::try_parse_from(["av-denoise", "tune", "--force", "nl4d", "-i", "clip.mkv"])
            .expect("the subcommand parses");

        let Command::Tune(tune) = args.command else {
            panic!("expected tune");
        };
        assert!(tune.force);
        assert!(matches!(tune.algorithm, Some(TuneAlgorithm::Nl4d(_))));
    }

    #[test]
    fn tuning_and_denoising_build_identical_plane_options() {
        let flags = [
            "-i",
            "clip.mkv",
            "--tune",
            "off",
            "--deterministic",
            "--tune-budget",
            "12.5",
        ];
        let mut denoise_args = vec!["av-denoise", "nl4d"];
        denoise_args.extend(flags);

        let mut tune_args = vec!["av-denoise", "tune", "nl4d"];
        tune_args.extend(flags);

        let denoise = Args::try_parse_from(denoise_args).expect("denoise parses");
        let tune = Args::try_parse_from(tune_args).expect("tune parses");
        let Command::Nl4d(denoise_nl4d) = &denoise.command else {
            panic!("expected nl4d");
        };
        let Command::Tune(TuneArgs {
            algorithm: Some(TuneAlgorithm::Nl4d(tune_nl4d)),
            ..
        }) = &tune.command
        else {
            panic!("expected tune nl4d");
        };

        let denoise_options = denoise_nl4d.build_options(&denoise).expect("denoise options");
        let tune_options = tune_nl4d.build_options(&tune).expect("tune options");

        assert_eq!(
            format!("{:?}", denoise_options.planes),
            format!("{:?}", tune_options.planes)
        );
        assert!(tune_options.planes.deterministic);
        assert_eq!(tune_options.planes.tuning, TuningMode::Off);
        assert_eq!(tune_options.tune_budget, Some(12.5));
    }

    #[test]
    fn show_and_clear_need_no_algorithm() {
        let show = Args::try_parse_from(["av-denoise", "tune", "--show"]).expect("show parses");
        let clear = Args::try_parse_from(["av-denoise", "tune", "--clear"]).expect("clear parses");

        assert!(matches!(show.command, Command::Tune(TuneArgs { show: true, .. })));
        assert!(matches!(
            clear.command,
            Command::Tune(TuneArgs { clear: true, .. })
        ));
    }

    #[test]
    fn force_cannot_be_combined_with_show() {
        let result = Args::try_parse_from(["av-denoise", "tune", "--force", "--show"]);

        assert!(result.is_err());
    }

    #[test]
    fn stored_winners_print_one_row_per_key() {
        let identity = DeviceIdentity {
            backend: "vulkan".to_string(),
            adapter: "Test Adapter".to_string(),
            vendor_id: 0x1002,
            device_id: 0x7551,
            driver: "test".to_string(),
            os: "linux".to_string(),
            compute_capability: None,
        };
        let mut result = TuneResult::default();
        let key = SiteKey::builder("nlm_window", 1920, 1080).param("pr", 2).build();
        let ranking = SiteRanking {
            candidates: vec![
                CandidateRecord {
                    name: "32x16".to_string(),
                    fidelity: FidelityClass::Exact,
                    baseline: false,
                    median_ns: 800_000,
                    samples: 40,
                },
                CandidateRecord {
                    name: "32x8".to_string(),
                    fidelity: FidelityClass::Exact,
                    baseline: true,
                    median_ns: 1_000_000,
                    samples: 40,
                },
            ],
            partial: false,
            progress: CompileProgress::default(),
            identity: String::new(),
            search: None,
        };
        result.sites.insert(key, ranking);

        let table = format_stored(&identity, &result, false);

        assert!(table.contains("Test Adapter"), "{table}");
        assert!(table.contains("nlm_window|w2048|h2048|pr=2"), "{table}");
        assert!(table.contains("32x16"), "{table}");
        assert!(table.contains("0.80 ms"), "{table}");
    }

    #[test]
    fn a_partial_key_is_marked_in_the_table() {
        let identity = DeviceIdentity {
            backend: "vulkan".to_string(),
            adapter: "Test Adapter".to_string(),
            vendor_id: 0,
            device_id: 0,
            driver: "test".to_string(),
            os: "linux".to_string(),
            compute_capability: None,
        };
        let record = CandidateRecord {
            name: "32x8".to_string(),
            fidelity: FidelityClass::Exact,
            baseline: true,
            median_ns: 1_000_000,
            samples: 40,
        };
        let ranking = SiteRanking {
            candidates: vec![record],
            partial: true,
            progress: CompileProgress::default(),
            identity: String::new(),
            search: None,
        };
        let key = SiteKey::builder("nlm_window", 1920, 1080).build();
        let mut result = TuneResult::default();
        result.sites.insert(key, ranking);

        let table = format_stored(&identity, &result, false);

        assert!(table.contains("1.00 ms  (partial)"), "{table}");
    }

    #[test]
    fn a_partial_key_shows_how_many_candidates_it_has_left() {
        let identity = DeviceIdentity {
            backend: "vulkan".to_string(),
            adapter: "Test Adapter".to_string(),
            vendor_id: 0,
            device_id: 0,
            driver: "test".to_string(),
            os: "linux".to_string(),
            compute_capability: None,
        };
        let record = CandidateRecord {
            name: "32x8".to_string(),
            fidelity: FidelityClass::Exact,
            baseline: true,
            median_ns: 1_000_000,
            samples: 40,
        };
        let progress = CompileProgress { left: 9, total: 26 };
        let ranking = SiteRanking {
            candidates: vec![record],
            partial: true,
            progress,
            identity: String::new(),
            search: None,
        };
        let key = SiteKey::builder("nlm_window", 1920, 1080).build();
        let mut result = TuneResult::default();
        result.sites.insert(key, ranking);

        let table = format_stored(&identity, &result, false);

        assert!(table.contains("1.00 ms  (partial, 9 of 26 left)"), "{table}");
    }

    #[test]
    fn an_empty_store_says_so() {
        let identity = DeviceIdentity {
            backend: "vulkan".to_string(),
            adapter: "Test Adapter".to_string(),
            vendor_id: 0,
            device_id: 0,
            driver: "test".to_string(),
            os: "linux".to_string(),
            compute_capability: None,
        };

        let table = format_stored(&identity, &TuneResult::default(), false);

        assert!(table.contains("no tuned launch sites"), "{table}");
    }
}

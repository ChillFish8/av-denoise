use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::Context;
use av_denoise::FrameGrain;
use av_denoise::avds::{AvdsFrame, AvdsMeta, FitOptions, av1_table, read_avds, report_rows};

use crate::pipeline::StagedFile;

/// Builds grain outputs from a grain data file.
#[derive(Debug, clap::Args)]
pub struct SynthesizeArgs {
    /// The output to build.
    #[command(subcommand)]
    pub kind: SynthesizeKind,
}

#[derive(Debug, clap::Subcommand)]
pub enum SynthesizeKind {
    /// Write an AV1 film grain table.
    Av1Fgs(Av1FgsArgs),
    /// Write the per-frame grain report as JSON lines.
    Report(ReportArgs),
}

/// The options of `synthesize av1-fgs`.
#[derive(Debug, clap::Args)]
pub struct Av1FgsArgs {
    /// The grain data file written by `--export-avds`.
    #[arg(long, value_name = "PATH")]
    pub input: PathBuf,

    /// Where to write the `filmgrn1` table.
    #[arg(long, value_name = "PATH")]
    pub output: PathBuf,

    /// Which planes get grain (comma-separated).
    ///
    /// Only `luma` is accepted.
    #[arg(long, value_enum, value_delimiter = ',', default_value = "luma")]
    pub planes: Vec<GrainPlane>,

    /// The share of the fine, fresh scan noise taken out of the table, between 0 and 1.
    ///
    /// 0 keeps everything measured.
    #[arg(
        long,
        value_name = "AMOUNT",
        default_value_t = 0.5,
        value_parser = parse_fine_noise,
        allow_negative_numbers = true
    )]
    pub fine_noise: f64,
}

/// The options of `synthesize report`.
#[derive(Debug, clap::Args)]
pub struct ReportArgs {
    /// The grain data file written by `--export-avds`.
    #[arg(long, value_name = "PATH")]
    pub input: PathBuf,

    /// Where to write the report, one JSON line per frame.
    ///
    /// Each line holds the source's luma grain strength, the Cb and Cr strengths when chroma was
    /// denoised, and how much grain stays unchanged from 1 to 8 frames either side of the frame.
    #[arg(long, value_name = "PATH")]
    pub output: PathBuf,
}

/// A plane that can get grain.
#[derive(Debug, Copy, Clone, PartialEq, Eq, clap::ValueEnum)]
pub enum GrainPlane {
    Luma,
    Chroma,
}

impl Av1FgsArgs {
    /// The fit options the arguments ask for, refusing planes other than luma alone.
    fn fit_options(&self) -> Result<FitOptions, anyhow::Error> {
        if self.planes != [GrainPlane::Luma] {
            anyhow::bail!("--planes accepts luma only, chroma grain is not synthesised yet");
        }

        let mut options = FitOptions::default();
        options.fine_noise = self.fine_noise;

        Ok(options)
    }
}

/// Runs `av-denoise synthesize`.
pub fn run_synthesize(args: &SynthesizeArgs) -> Result<(), anyhow::Error> {
    match &args.kind {
        SynthesizeKind::Av1Fgs(av1_fgs) => write_av1_table(av1_fgs),
        SynthesizeKind::Report(report) => write_report(report),
    }
}

fn write_av1_table(args: &Av1FgsArgs) -> Result<(), anyhow::Error> {
    let options = args.fit_options()?;
    let output = StagedFile::create(&args.output, "grain table")?;
    let (meta, rows) = read_file(&args.input)?;

    let table = av1_table(&meta, &rows, &options)?;
    warn_on_empty_table(&table);

    output.commit(table.as_bytes())
}

fn write_report(args: &ReportArgs) -> Result<(), anyhow::Error> {
    let output = StagedFile::create(&args.output, "grain report")?;
    let (_, rows) = read_file(&args.input)?;

    let report = report_rows(&rows);
    let text = report_text(&report)?;

    output.commit(text.as_bytes())
}

/// Warns when `table` is a `filmgrn1` table with no entries.
fn warn_on_empty_table(table: &str) {
    if table == "filmgrn1\n" {
        tracing::warn!("no grain could be measured, writing an empty grain table");
    }
}

/// The report text, one JSON line per row.
pub fn report_text(rows: &[FrameGrain]) -> Result<String, anyhow::Error> {
    let mut text = String::new();
    for row in rows {
        let line = serde_json::to_string(row).context("unable to encode a grain report row")?;
        text.push_str(&line);
        text.push('\n');
    }

    Ok(text)
}

fn read_file(path: &Path) -> Result<(AvdsMeta, Vec<AvdsFrame>), anyhow::Error> {
    let shown_path = path.display();
    let file =
        File::open(path).with_context(|| format!("unable to open the grain data file at {shown_path}"))?;

    read_avds(file).with_context(|| format!("unable to read the grain data file at {shown_path}"))
}

fn parse_fine_noise(text: &str) -> Result<f64, String> {
    let amount: f64 = text.parse().map_err(|_| format!("{text} is not a number"))?;
    if !(0.0..=1.0).contains(&amount) {
        return Err(format!("{amount} is not between 0 and 1"));
    }

    Ok(amount)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use clap::Parser;
    use parquet::arrow::ArrowWriter;

    use super::*;
    use crate::cli::{Args, Command};

    /// A Parquet file of one integer column, with none of a grain data file's metadata.
    fn write_foreign_parquet(path: &Path) {
        let field = Field::new("value", DataType::Int32, false);
        let schema = Schema::new(vec![field]);
        let schema = Arc::new(schema);
        let values = Int32Array::from(vec![1, 2, 3]);
        let column: ArrayRef = Arc::new(values);
        let columns = vec![column];
        let batch = RecordBatch::try_new(schema.clone(), columns).expect("the batch builds");

        let file = File::create(path).expect("the file is created");
        let mut writer = ArrowWriter::try_new(file, schema, None).expect("the writer opens");
        writer.write(&batch).expect("the rows are written");
        writer.close().expect("the file closes");
    }

    fn temporary_of(path: &Path) -> PathBuf {
        let mut name = path.as_os_str().to_owned();
        name.push(".tmp");

        PathBuf::from(name)
    }

    fn av1_fgs(input: PathBuf, output: PathBuf, planes: Vec<GrainPlane>) -> SynthesizeArgs {
        let av1_fgs = Av1FgsArgs {
            input,
            output,
            planes,
            fine_noise: 0.0,
        };
        let kind = SynthesizeKind::Av1Fgs(av1_fgs);

        SynthesizeArgs { kind }
    }

    fn parse(extra: &[&str]) -> Result<Args, clap::Error> {
        let mut argv = vec!["av-denoise", "synthesize"];
        argv.extend_from_slice(extra);

        Args::try_parse_from(argv)
    }

    fn parsed_av1_fgs(extra: &[&str]) -> Av1FgsArgs {
        let args = parse(extra).expect("the arguments parse");
        let Command::Synthesize(synthesize) = args.command else {
            panic!("expected synthesize");
        };
        let SynthesizeKind::Av1Fgs(av1_fgs) = synthesize.kind else {
            panic!("expected av1-fgs");
        };

        av1_fgs
    }

    #[test]
    fn av1_fgs_defaults_to_luma_and_half_the_fine_noise() {
        let av1_fgs = parsed_av1_fgs(&["av1-fgs", "--input", "run.avds", "--output", "grain.tbl"]);

        assert_eq!(av1_fgs.input, PathBuf::from("run.avds"));
        assert_eq!(av1_fgs.output, PathBuf::from("grain.tbl"));
        assert_eq!(av1_fgs.planes, [GrainPlane::Luma]);
        assert_eq!(av1_fgs.fine_noise, 0.5);
    }

    #[test]
    fn fine_noise_reaches_the_fit_options() {
        let extra = ["av1-fgs", "--input", "a", "--output", "b", "--fine-noise", "0.6"];
        let av1_fgs = parsed_av1_fgs(&extra);

        let options = av1_fgs.fit_options().expect("luma is accepted");

        assert_eq!(options.fine_noise, 0.6);
    }

    #[test]
    fn av1_fgs_parses_planes_and_fine_noise() {
        let extra = [
            "av1-fgs",
            "--input",
            "run.avds",
            "--output",
            "grain.tbl",
            "--planes",
            "luma,chroma",
            "--fine-noise",
            "0.75",
        ];
        let av1_fgs = parsed_av1_fgs(&extra);

        assert_eq!(av1_fgs.planes, [GrainPlane::Luma, GrainPlane::Chroma]);
        assert_eq!(av1_fgs.fine_noise, 0.75);
    }

    #[test]
    fn report_text_has_one_line_per_row_in_order() {
        let rows: Vec<FrameGrain> = (0..3)
            .map(|frame| FrameGrain {
                frame,
                scene: 0,
                cells: 0,
                strength: None,
                strength_cb: None,
                strength_cr: None,
                persistence: [None; 8],
            })
            .collect();

        let text = report_text(&rows).expect("the rows encode");
        let lines: Vec<&str> = text.lines().collect();

        assert!(text.ends_with('\n'));
        assert_eq!(lines.len(), 3);

        for (index, line) in lines.iter().enumerate() {
            let value: serde_json::Value = serde_json::from_str(line).expect("each line is JSON");
            let object = value.as_object().expect("each line is an object");
            let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
            keys.sort_unstable();
            let persistence = object["persistence"].as_array().expect("persistence is a list");

            let expected_keys = [
                "cells",
                "frame",
                "persistence",
                "scene",
                "strength",
                "strength_cb",
                "strength_cr",
            ];
            assert_eq!(keys, expected_keys);
            assert_eq!(object["frame"], index as u64);
            assert_eq!(persistence.len(), 8);
        }
    }

    #[test]
    fn report_parses_its_paths() {
        let args = parse(&["report", "--input", "run.avds", "--output", "report.jsonl"])
            .expect("the arguments parse");
        let Command::Synthesize(synthesize) = args.command else {
            panic!("expected synthesize");
        };
        let SynthesizeKind::Report(report) = synthesize.kind else {
            panic!("expected report");
        };

        assert_eq!(report.input, PathBuf::from("run.avds"));
        assert_eq!(report.output, PathBuf::from("report.jsonl"));
    }

    #[test]
    fn fine_noise_outside_zero_to_one_is_refused() {
        for amount in ["-0.1", "1.5", "lots", "NaN"] {
            let extra = ["av1-fgs", "--input", "a", "--output", "b", "--fine-noise", amount];
            let error = parse(&extra).expect_err("the amount is refused");
            let message = error.to_string();

            assert!(message.contains("--fine-noise"), "got {message}");
        }

        let edges = ["0", "1"];
        for amount in edges {
            let extra = ["av1-fgs", "--input", "a", "--output", "b", "--fine-noise", amount];
            let parsed = parse(&extra);

            assert!(parsed.is_ok(), "{amount} is refused");
        }
    }

    #[test]
    fn chroma_planes_are_refused() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let input = dir.path().join("run.avds");
        let output = dir.path().join("grain.tbl");

        for planes in [
            vec![GrainPlane::Chroma],
            vec![GrainPlane::Luma, GrainPlane::Chroma],
        ] {
            let args = av1_fgs(input.clone(), output.clone(), planes);
            let error = run_synthesize(&args).expect_err("chroma is refused");
            let message = error.to_string();

            assert!(message.contains("--planes accepts luma only"), "got {message}");
            assert!(!output.exists());
            assert!(!temporary_of(&output).exists());
        }
    }

    #[test]
    fn a_foreign_parquet_file_is_refused_and_writes_nothing() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let input = dir.path().join("foreign.parquet");
        let table = dir.path().join("grain.tbl");
        let report = dir.path().join("report.jsonl");
        write_foreign_parquet(&input);

        let table_args = av1_fgs(input.clone(), table.clone(), vec![GrainPlane::Luma]);
        let table_error = run_synthesize(&table_args).expect_err("the file is refused");
        let report_args = ReportArgs {
            input,
            output: report.clone(),
        };
        let kind = SynthesizeKind::Report(report_args);
        let report_args = SynthesizeArgs { kind };
        let report_error = run_synthesize(&report_args).expect_err("the file is refused");

        let table_message = format!("{table_error:#}");
        let report_message = format!("{report_error:#}");

        assert!(
            table_message.contains("no avds.version metadata"),
            "got {table_message}"
        );
        assert!(
            report_message.contains("no avds.version metadata"),
            "got {report_message}"
        );
        assert!(!table.exists());
        assert!(!temporary_of(&table).exists());
        assert!(!report.exists());
        assert!(!temporary_of(&report).exists());
    }

    #[test]
    fn an_unwritable_output_fails_before_reading() {
        let input = PathBuf::from("/nonexistent-directory/run.avds");
        let table = PathBuf::from("/nonexistent-directory/grain.tbl");
        let report = PathBuf::from("/nonexistent-directory/report.jsonl");

        let table_args = av1_fgs(input.clone(), table, vec![GrainPlane::Luma]);
        let table_error = run_synthesize(&table_args).expect_err("cannot write there");
        let report_args = ReportArgs {
            input,
            output: report,
        };
        let kind = SynthesizeKind::Report(report_args);
        let report_args = SynthesizeArgs { kind };
        let report_error = run_synthesize(&report_args).expect_err("cannot write there");

        let table_message = table_error.to_string();
        let report_message = report_error.to_string();

        assert!(
            table_message.contains("unable to create the grain table"),
            "got {table_message}"
        );
        assert!(
            report_message.contains("unable to create the grain report"),
            "got {report_message}"
        );
    }
}

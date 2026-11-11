use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::Context;
use av_denoise::FrameGrain;
use av_denoise::avds::{
    AV1_CHROMA_TABLE_GROUPS,
    AV1_TABLE_GROUPS,
    AvdsFrame,
    AvdsGroup,
    AvdsMeta,
    DEFAULT_FINE_NOISE,
    FitOptions,
    REPORT_GROUPS,
    av1_table,
    chroma_unavailable,
    read_avds_groups,
    report_rows,
};

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
    ///
    /// Apply it with `regrain apply`, since SVT-AV1 and grav1synth apply only its first entry. Its
    /// timestamps suit constant frame rate video.
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

    /// Which planes get grain (comma-separated), `luma` or `luma,chroma`.
    ///
    /// Chroma grain needs 4:2:0 video with chroma measured. Otherwise a warning is logged and only
    /// luma gets grain.
    #[arg(long, value_enum, value_delimiter = ',', default_value = "luma,chroma")]
    pub planes: Vec<GrainPlane>,

    /// The share of the fine, fresh scan noise taken out of the table, between 0 and 1.
    ///
    /// 0 keeps everything measured.
    #[arg(
        long,
        value_name = "AMOUNT",
        default_value_t = DEFAULT_FINE_NOISE,
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
    /// The fit options the arguments ask for, refusing chroma without luma.
    fn fit_options(&self) -> Result<FitOptions, anyhow::Error> {
        if !self.planes.contains(&GrainPlane::Luma) {
            anyhow::bail!("--planes needs luma, chroma grain follows the luma entries");
        }

        let chroma = self.planes.contains(&GrainPlane::Chroma);
        let options = FitOptions::with_fine_noise(self.fine_noise).with_chroma(chroma);

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
    let mut options = args.fit_options()?;
    let output = StagedFile::create(&args.output, "grain table")?;
    let groups: &[AvdsGroup] = if options.chroma {
        &AV1_CHROMA_TABLE_GROUPS
    } else {
        &AV1_TABLE_GROUPS
    };
    let (meta, rows) = read_file(&args.input, groups)?;

    if options.chroma {
        let unavailable = chroma_unavailable(&meta, &rows);
        if let Some(reason) = unavailable {
            tracing::warn!("chroma grain is not written, {reason}");
            options = options.with_chroma(false);
        }
    }

    let table = av1_table(&meta, rows, &options)?;
    warn_on_empty_table(&table);

    output.commit(table.as_bytes())
}

fn write_report(args: &ReportArgs) -> Result<(), anyhow::Error> {
    let output = StagedFile::create(&args.output, "grain report")?;
    let (_, rows) = read_file(&args.input, &REPORT_GROUPS)?;

    let report = report_rows(rows);
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

/// The file's metadata and rows, with only the columns of `groups`.
fn read_file(path: &Path, groups: &[AvdsGroup]) -> Result<(AvdsMeta, Vec<AvdsFrame>), anyhow::Error> {
    let shown_path = path.display();
    let file =
        File::open(path).with_context(|| format!("unable to open the grain data file at {shown_path}"))?;

    read_avds_groups(file, groups)
        .with_context(|| format!("unable to read the grain data file at {shown_path}"))
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
    use std::sync::{Arc, Mutex};

    use arrow_array::{ArrayRef, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use av_denoise::avds::{PlanesMeasured, write_avds};
    use av_denoise::{ChromaLayout, ChromaSiting};
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

    /// The luma bins and std buckets of a table's histograms.
    const LUMA_BINS: usize = 16;
    const STD_BUCKETS: usize = 64;
    /// The strength groups of a table, each a record of lag sums then a pixel count.
    const STRENGTH_GROUPS: usize = 16;
    const BUCKETS_PER_GROUP: usize = STD_BUCKETS / STRENGTH_GROUPS;
    const LAG_SUMS: usize = 46;
    const RECORD_LEN: usize = LAG_SUMS + 1;
    /// Values in one plane's frame table, two histograms and the grouped records.
    const HISTOGRAM_LEN: usize = LUMA_BINS * STD_BUCKETS;
    const TABLE_LEN: usize = 2 * HISTOGRAM_LEN + STRENGTH_GROUPS * RECORD_LEN;

    /// The luma bins the table's blocks sit in, and the std bucket they share.
    const BLOCK_BINS: std::ops::Range<usize> = 4..12;
    const BLOCK_BUCKET: usize = 24;
    const BLOCKS_PER_BIN: f32 = 200.0;
    /// The pixels of one measured 8x8 block.
    const CELL_PIXELS: f32 = 64.0;

    /// The brightness bins of a colour fit, with the cells of each and the header words after them.
    const BRIGHTNESS_BINS: usize = 16;
    const CELLS_PER_BRIGHTNESS: usize = 61;
    const COLOUR_CELLS: usize = BRIGHTNESS_BINS * CELLS_PER_BRIGHTNESS;
    const HEADER_LEN: usize = 22;
    const POPULATED_WORD: usize = 18;
    const BLOCKS_WORD: usize = 19;
    const LUMA_LOW_WORD: usize = 20;
    const LUMA_HIGH_WORD: usize = 21;
    /// Values in a frame's chroma colour fit, both planes.
    const CHROMA_COLOUR_LEN: usize = 2 * (2 * COLOUR_CELLS + HEADER_LEN);

    /// `planes` tables of uncorrelated grain with the variance `variance`.
    fn table_values(variance: f32, planes: usize) -> Vec<f32> {
        let pixels = BLOCKS_PER_BIN * BLOCK_BINS.len() as f32 * CELL_PIXELS;
        let group = BLOCK_BUCKET / BUCKETS_PER_GROUP;
        let group_start = 2 * HISTOGRAM_LEN + group * RECORD_LEN;
        let mut values = vec![0.0; planes * TABLE_LEN];

        for plane in 0..planes {
            let start = plane * TABLE_LEN;
            for bin in BLOCK_BINS {
                values[start + bin * STD_BUCKETS + BLOCK_BUCKET] = BLOCKS_PER_BIN;
            }

            values[start + group_start] = variance * pixels;
            values[start + group_start + LAG_SUMS] = pixels;
        }

        values
    }

    /// A colour fit of both planes, each with grain of 4 codes over luma 40 to 150.
    fn chroma_colour_values() -> Vec<f32> {
        let mut values = Vec::with_capacity(CHROMA_COLOUR_LEN);
        for _plane in 0..2 {
            let log_grain = vec![4.0f32.ln(); COLOUR_CELLS];
            let positions = vec![0.0; COLOUR_CELLS];
            let mut header = vec![0.0; HEADER_LEN];
            header[POPULATED_WORD] = 1.0;
            header[BLOCKS_WORD] = 600.0;
            header[LUMA_LOW_WORD] = 40.0;
            header[LUMA_HIGH_WORD] = 150.0;

            values.extend(log_grain);
            values.extend(positions);
            values.extend(header);
        }

        values
    }

    fn table_row(frame: u64, chroma: bool) -> AvdsFrame {
        let chroma_table = chroma.then(|| table_values(0.0005, 2));
        let chroma_colour = chroma.then(chroma_colour_values);

        AvdsFrame {
            frame,
            scene: 0,
            first_of_scene: frame == 0,
            table: Some(table_values(0.001, 1)),
            chroma_table,
            persistence: None,
            chroma_persistence: None,
            noise: None,
            chroma_noise: None,
            colour: None,
            chroma_colour,
        }
    }

    /// Writes a 4:2:0 grain data file of `frames` frames, with chroma rows when `chroma` is on.
    fn write_avds_file(path: &Path, frames: u64, chroma: bool) {
        let meta = AvdsMeta {
            writer: "av-denoise test".to_owned(),
            frame_rate: (24, 1),
            width: 64,
            height: 48,
            bit_depth: 10,
            subsampling: ChromaLayout::Yuv420,
            chroma_siting: ChromaSiting::Left,
            frames,
            planes_measured: PlanesMeasured::LumaChroma,
        };
        let rows: Vec<AvdsFrame> = (0..frames).map(|frame| table_row(frame, chroma)).collect();

        let file = File::create(path).expect("the file is created");
        write_avds(file, &meta, &rows).expect("the file is written");
    }

    #[derive(Clone, Default)]
    struct LogSink(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for LogSink {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            let mut buffer = self.0.lock().expect("the sink is not poisoned");
            buffer.extend_from_slice(bytes);

            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// The log text written while `work` runs.
    fn capture_warnings<T>(work: impl FnOnce() -> T) -> String {
        let sink = LogSink::default();
        let writer = sink.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        tracing::subscriber::with_default(subscriber, work);

        let buffer = sink.0.lock().expect("the sink is not poisoned");

        String::from_utf8_lossy(&buffer).into_owned()
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
    fn planes_default_to_luma_and_chroma() {
        let av1_fgs = parsed_av1_fgs(&["av1-fgs", "--input", "run.avds", "--output", "grain.tbl"]);

        assert_eq!(av1_fgs.input, PathBuf::from("run.avds"));
        assert_eq!(av1_fgs.output, PathBuf::from("grain.tbl"));
        assert_eq!(av1_fgs.planes, [GrainPlane::Luma, GrainPlane::Chroma]);
        assert_eq!(av1_fgs.fine_noise, 0.5);
    }

    #[test]
    fn planes_reach_the_fit_options() {
        let both = [GrainPlane::Luma, GrainPlane::Chroma];
        let reversed = [GrainPlane::Chroma, GrainPlane::Luma];
        let luma = [GrainPlane::Luma];

        for (planes, chroma) in [(&both[..], true), (&reversed[..], true), (&luma[..], false)] {
            let args = av1_fgs(PathBuf::new(), PathBuf::new(), planes.to_vec());
            let SynthesizeKind::Av1Fgs(av1_fgs) = args.kind else {
                panic!("expected av1-fgs");
            };

            let options = av1_fgs.fit_options().expect("the planes are accepted");

            assert_eq!(options.chroma, chroma);
        }
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
    fn chroma_alone_is_refused() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let input = dir.path().join("run.avds");
        let output = dir.path().join("grain.tbl");

        let args = av1_fgs(input, output.clone(), vec![GrainPlane::Chroma]);
        let error = run_synthesize(&args).expect_err("chroma alone is refused");
        let message = error.to_string();

        assert!(message.contains("--planes needs luma"), "got {message}");
        assert!(!output.exists());
        assert!(!temporary_of(&output).exists());
    }

    #[test]
    fn a_luma_only_file_writes_the_luma_table_and_warns() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let input = dir.path().join("run.avds");
        let both_output = dir.path().join("both.tbl");
        let luma_output = dir.path().join("luma.tbl");
        write_avds_file(&input, 40, false);
        let both_planes = vec![GrainPlane::Luma, GrainPlane::Chroma];
        let both_args = av1_fgs(input.clone(), both_output.clone(), both_planes);
        let luma_args = av1_fgs(input, luma_output.clone(), vec![GrainPlane::Luma]);

        let warnings = capture_warnings(|| run_synthesize(&both_args).expect("the table is written"));
        run_synthesize(&luma_args).expect("the table is written");

        let both = std::fs::read_to_string(both_output).expect("the table is written");
        let luma = std::fs::read_to_string(luma_output).expect("the table is written");

        assert_ne!(luma, "filmgrn1\n");
        assert_eq!(both, luma);
        assert!(warnings.contains("chroma grain is not written"), "got {warnings}");
    }

    #[test]
    fn luma_planes_match_the_luma_only_table() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let input = dir.path().join("run.avds");
        let output = dir.path().join("grain.tbl");
        write_avds_file(&input, 40, true);
        let args = av1_fgs(input.clone(), output.clone(), vec![GrainPlane::Luma]);
        run_synthesize(&args).expect("the table is written");

        let (meta, rows) = read_file(&input, &AV1_TABLE_GROUPS).expect("the file reads");
        let options = FitOptions::with_fine_noise(0.0);
        let expected = av1_table(&meta, rows, &options).expect("the table builds");

        let written = std::fs::read_to_string(output).expect("the table is written");
        assert_eq!(written, expected);
    }

    #[test]
    fn chroma_planes_match_the_chroma_table() {
        let dir = tempfile::tempdir().expect("a temporary directory");
        let input = dir.path().join("run.avds");
        let output = dir.path().join("grain.tbl");
        write_avds_file(&input, 40, true);
        let both_planes = vec![GrainPlane::Luma, GrainPlane::Chroma];
        let args = av1_fgs(input.clone(), output.clone(), both_planes);
        run_synthesize(&args).expect("the table is written");

        let (meta, rows) = read_file(&input, &AV1_CHROMA_TABLE_GROUPS).expect("the file reads");
        let options = FitOptions::with_fine_noise(0.0).with_chroma(true);
        let expected = av1_table(&meta, rows, &options).expect("the table builds");
        let (_, luma_rows) = read_file(&input, &AV1_TABLE_GROUPS).expect("the file reads");
        let luma_options = FitOptions::with_fine_noise(0.0);
        let luma_only = av1_table(&meta, luma_rows, &luma_options).expect("the table builds");

        let written = std::fs::read_to_string(output).expect("the table is written");

        assert_eq!(written, expected);
        assert_ne!(written, luma_only);
        for prefix in ["\tsCb ", "\tsCr "] {
            let line = written
                .lines()
                .find(|line| line.starts_with(prefix))
                .expect("the table has the plane's scaling line");
            let values: Vec<&str> = line.split_whitespace().skip(2).collect();
            let has_strength = values.iter().skip(1).step_by(2).any(|value| *value != "0");

            assert!(has_strength, "got {line}");
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

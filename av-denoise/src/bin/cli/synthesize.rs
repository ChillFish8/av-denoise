use std::fs::File;
use std::ops::RangeInclusive;
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
    DEFAULT_STRENGTH,
    FitOptions,
    GrainMode,
    GrainRun,
    MAX_STRENGTH,
    REPORT_GROUPS,
    StaticGrain,
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
    /// Apply it with `regrain apply --static-grain`, which the table needs to hold static grain
    /// still. SVT-AV1 and grav1synth apply only its first entry. Its timestamps suit constant
    /// frame rate video.
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

    /// The share of the fine, fresh scan noise moved out of the grain's texture, between 0 and 1.
    ///
    /// The grain keeps its measured strength. 0 leaves the texture as measured.
    #[arg(
        long,
        value_name = "AMOUNT",
        default_value_t = DEFAULT_FINE_NOISE,
        value_parser = parse_fine_noise,
        allow_negative_numbers = true
    )]
    pub fine_noise: f64,

    /// How strong the grain is, as a multiple of the source's, between 0 and 4.
    ///
    /// 1 matches the source's grain. A lower value suits an encode that keeps some grain itself.
    #[arg(
        long,
        value_name = "FACTOR",
        default_value_t = DEFAULT_STRENGTH,
        value_parser = parse_strength,
        allow_negative_numbers = true
    )]
    pub strength: f64,

    /// Treat the whole clip's grain as static, one pattern held on every frame.
    #[arg(
        long = "static",
        conflicts_with_all = ["no_static", "static_scenes", "static_frames"]
    )]
    pub static_grain: bool,

    /// Turn static grain detection off, so only declared scenes and frames are static.
    #[arg(long)]
    pub no_static: bool,

    /// Scenes whose grain is static (comma-separated), each a scene number or a `first:last` range.
    ///
    /// Scene numbers are the ones `synthesize report` prints. A range includes both ends.
    #[arg(
        long,
        value_name = "SCENES",
        value_delimiter = ',',
        value_parser = parse_scene_range
    )]
    pub static_scenes: Vec<RangeInclusive<u64>>,

    /// Frames whose grain is static (comma-separated), each a frame number or a `first:last` range.
    ///
    /// Frame numbers count output frames from 0. A range includes both ends.
    #[arg(
        long,
        value_name = "FRAMES",
        value_delimiter = ',',
        value_parser = parse_frame_range
    )]
    pub static_frames: Vec<RangeInclusive<u64>>,
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
    pub(super) fn fit_options(&self) -> Result<FitOptions, anyhow::Error> {
        if !self.planes.contains(&GrainPlane::Luma) {
            anyhow::bail!("--planes needs luma, chroma grain follows the luma entries");
        }

        let chroma = self.planes.contains(&GrainPlane::Chroma);
        let static_grain = if self.static_grain {
            StaticGrain::All
        } else if self.no_static {
            StaticGrain::Off
        } else {
            StaticGrain::Detect
        };
        let options = FitOptions::with_fine_noise(self.fine_noise)
            .with_chroma(chroma)
            .with_strength(self.strength)
            .with_static_grain(static_grain)
            .with_static_scenes(self.static_scenes.clone())
            .with_static_frames(self.static_frames.clone());

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
    warn_on_empty_table(&table.text);
    log_runs(&table.runs);

    output.commit(table.text.as_bytes())
}

/// Logs one line for every run of frames.
fn log_runs(runs: &[GrainRun]) {
    for run in runs {
        let first_frame = run.first_frame;
        let last_frame = run.last_frame;
        let mode = match run.mode {
            GrainMode::Fresh => "fresh",
            GrainMode::Frozen => "frozen",
        };

        tracing::info!("frames {first_frame} to {last_frame} {mode}");
    }
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
pub(super) fn read_file(
    path: &Path,
    groups: &[AvdsGroup],
) -> Result<(AvdsMeta, Vec<AvdsFrame>), anyhow::Error> {
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

fn parse_strength(text: &str) -> Result<f64, String> {
    let factor: f64 = text.parse().map_err(|_| format!("{text} is not a number"))?;
    if !(0.0..=MAX_STRENGTH).contains(&factor) {
        return Err(format!("{factor} is not between 0 and {MAX_STRENGTH}"));
    }

    Ok(factor)
}

fn parse_scene_range(text: &str) -> Result<RangeInclusive<u64>, String> {
    parse_range(text, "scene")
}

fn parse_frame_range(text: &str) -> Result<RangeInclusive<u64>, String> {
    parse_range(text, "frame")
}

/// A `first:last` range, or a lone number as a range of one.
fn parse_range(text: &str, kind: &str) -> Result<RangeInclusive<u64>, String> {
    let not_a_range = || format!("{text:?} is not a {kind} range");
    let (first_text, last_text) = text.split_once(':').unwrap_or((text, text));

    let first: u64 = first_text.parse().map_err(|_| not_a_range())?;
    let last: u64 = last_text.parse().map_err(|_| not_a_range())?;
    if last < first {
        return Err(format!("the {kind} range {text:?} ends before it starts"));
    }

    Ok(first..=last)
}

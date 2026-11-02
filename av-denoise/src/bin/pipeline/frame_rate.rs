use av_decoders::Rational32;

/// How far the span may sit from the whole periods it holds, as a share of those periods.
const SPAN_TOLERANCE: f64 = 0.01;

/// The share of frame intervals that may sit off a whole period.
const OFF_PERIOD_SHARE: f64 = 0.01;

/// The share of frame intervals that may span two or more periods, as dropped frames do.
const SKIPPED_SHARE: f64 = 0.1;

/// The fewest intervals spanning two or more periods that mark a multiple, so one dropped frame
/// in a short clip is not mistaken for one.
const MIN_SKIPPED: usize = 2;

/// The fewest ticks a period may last for the timestamps to describe the rate.
const MIN_PERIOD_TICKS: f64 = 2.0;

/// Why a file's frames do not run at its declared rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VariableRate {
    /// Every frame carries the same timestamp, so the frames span no time.
    NoSpan { frames: usize },
    /// The span disagrees with the whole periods it holds.
    Average {
        /// The rate over the span, with each gap counted as the frames it skipped.
        measured_fps: f64,
        declared: Rational32,
    },
    /// Too many intervals miss a whole period.
    OffPeriod {
        /// The share of intervals that miss, between 0 and 1.
        share: f64,
        declared: Rational32,
    },
    /// Too many intervals span two or more periods, so the declared rate looks like a multiple of
    /// the real one.
    Multiple {
        /// The share of intervals spanning two or more periods, between 0 and 1.
        share: f64,
        /// The plain rate between kept frames.
        frame_fps: f64,
        declared: Rational32,
    },
}

/// Checks that frames stamped at `pts` run at the constant `declared` rate.
///
/// `pts` holds the kept frames' timestamps in order, each unit lasting `tick_seconds`. Each
/// interval counts as its nearest whole number of periods, at least one, so a dropped frame is not
/// mistaken for a rate change. The rate is variable when the span misses those periods by more
/// than 1% or one tick, when more than 1% of the intervals miss a whole period, or when at least
/// two and more than 10% of them span two or more periods. Frames that all share one timestamp are
/// rejected too. Timestamps too coarse for the period are not checked.
pub fn check_constant_rate(pts: &[i64], tick_seconds: f64, declared: Rational32) -> Result<(), VariableRate> {
    let Some((first, last)) = pts.first().zip(pts.last()) else {
        return Ok(());
    };

    if pts.len() < 2 {
        return Ok(());
    }

    let span_ticks = last.saturating_sub(*first);
    if span_ticks <= 0 {
        let frames = pts.len();
        return Err(VariableRate::NoSpan { frames });
    }

    let declared_fps = rational_to_f64(declared);
    let period_ticks = 1.0 / (declared_fps * tick_seconds);
    if period_ticks < MIN_PERIOD_TICKS {
        tracing::debug!(
            period_ticks,
            "the timestamps are too coarse for the frame rate, so the frame rate is not checked",
        );
        return Ok(());
    }

    let intervals: Vec<i64> = pts
        .windows(2)
        .map(|pair| pair[1].saturating_sub(pair[0]))
        .collect();
    let interval_count = intervals.len() as f64;

    let periods: f64 = intervals
        .iter()
        .map(|&interval| whole_periods(interval, period_ticks))
        .sum();
    let expected_ticks = periods * period_ticks;
    let span_miss = (span_ticks as f64 - expected_ticks).abs();
    let allowed_miss = (expected_ticks * SPAN_TOLERANCE).max(1.0);
    if span_miss > allowed_miss {
        let span_seconds = span_ticks as f64 * tick_seconds;
        let measured_fps = periods / span_seconds;
        return Err(VariableRate::Average {
            measured_fps,
            declared,
        });
    }

    let off_period = intervals
        .iter()
        .filter(|&&interval| !fits_period(interval, period_ticks))
        .count();
    let off_share = off_period as f64 / interval_count;
    if off_share > OFF_PERIOD_SHARE {
        return Err(VariableRate::OffPeriod {
            share: off_share,
            declared,
        });
    }

    let skipped = intervals
        .iter()
        .filter(|&&interval| whole_periods(interval, period_ticks) >= 2.0)
        .count();
    let skipped_share = skipped as f64 / interval_count;
    if skipped >= MIN_SKIPPED && skipped_share > SKIPPED_SHARE {
        let span_seconds = span_ticks as f64 * tick_seconds;
        let frame_fps = interval_count / span_seconds;
        return Err(VariableRate::Multiple {
            share: skipped_share,
            frame_fps,
            declared,
        });
    }

    Ok(())
}

/// Whether an `interval` of ticks spans a whole number of periods, at least one.
///
/// It may miss by a quarter of a period, or by one tick when that is wider, so timestamps rounded
/// to the container's tick still fit.
pub fn fits_period(interval: i64, period_ticks: f64) -> bool {
    let periods = whole_periods(interval, period_ticks);
    let miss = (interval as f64 - periods * period_ticks).abs();
    let tolerance = (period_ticks / 4.0).max(1.0);

    miss <= tolerance
}

/// The whole number of periods nearest an `interval` of ticks, at least one.
fn whole_periods(interval: i64, period_ticks: f64) -> f64 {
    let periods = interval as f64 / period_ticks;

    periods.round().max(1.0)
}

/// The error a variable frame rate file stops the run with.
pub fn variable_rate_error(rate: VariableRate) -> anyhow::Error {
    match rate {
        VariableRate::NoSpan { frames } => anyhow::anyhow!(
            "the input stamps all {frames} of its frames with the same time, so y4m output cannot \
             keep its timing. Convert it to a constant rate first, for example \
             `ffmpeg -i in.mkv -fps_mode cfr -r <rate> ...`"
        ),
        VariableRate::Average {
            measured_fps,
            declared,
        } => {
            let declared_fps = rational_to_f64(declared);
            let numerator = *declared.numer();
            let denominator = *declared.denom();

            anyhow::anyhow!(
                "the input has a variable frame rate, which y4m output cannot keep. Counting dropped \
                 frames, it averages {measured_fps:.3} fps against a declared {declared_fps:.3} fps. \
                 Convert it to a constant rate first, for example \
                 `ffmpeg -i in.mkv -fps_mode cfr -r {numerator}/{denominator} ...`"
            )
        },
        VariableRate::OffPeriod { share, declared } => {
            let percent = share * 100.0;
            let declared_fps = rational_to_f64(declared);
            let numerator = *declared.numer();
            let denominator = *declared.denom();

            anyhow::anyhow!(
                "the input has a variable frame rate, which y4m output cannot keep. {percent:.1}% of \
                 its frames fall off the period of the declared {declared_fps:.3} fps. Convert it to a \
                 constant rate first, for example \
                 `ffmpeg -i in.mkv -fps_mode cfr -r {numerator}/{denominator} ...`"
            )
        },
        VariableRate::Multiple {
            share,
            frame_fps,
            declared,
        } => {
            let percent = share * 100.0;
            let declared_fps = rational_to_f64(declared);

            anyhow::anyhow!(
                "the input declares {declared_fps:.3} fps, which looks like a multiple of its real rate. \
                 {percent:.1}% of its frames sit two or more periods apart, and frames arrive at \
                 {frame_fps:.3} fps. Convert it to its real constant rate first, for example \
                 `ffmpeg -i in.mkv -fps_mode cfr -r <rate> ...`"
            )
        },
    }
}

fn rational_to_f64(rate: Rational32) -> f64 {
    let numerator = f64::from(*rate.numer());
    let denominator = f64::from(*rate.denom());

    numerator / denominator
}

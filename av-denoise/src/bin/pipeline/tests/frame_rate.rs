use av_decoders::Rational32;

use crate::pipeline::frame_rate::{VariableRate, check_constant_rate, fits_period, variable_rate_error};

const MILLISECOND: f64 = 0.001;

fn ntsc_film() -> Rational32 {
    Rational32::new(24000, 1001)
}

/// `count` frames at `fps`, stamped in whole milliseconds as Matroska stores them.
fn millisecond_stamps(count: usize, fps: f64) -> Vec<i64> {
    (0..count)
        .map(|frame| {
            let seconds = frame as f64 / fps;
            (seconds * 1000.0).round() as i64
        })
        .collect()
}

#[test]
fn a_steady_stream_is_constant() {
    let pts = millisecond_stamps(900, 24000.0 / 1001.0);

    assert_eq!(check_constant_rate(&pts, MILLISECOND, ntsc_film()), Ok(()));
}

#[test]
fn a_single_dropped_frame_is_constant() {
    let mut pts = millisecond_stamps(900, 24000.0 / 1001.0);
    pts.remove(450);

    assert_eq!(check_constant_rate(&pts, MILLISECOND, ntsc_film()), Ok(()));
}

#[test]
fn a_short_stream_with_one_dropped_frame_is_constant() {
    let mut pts = millisecond_stamps(41, 24000.0 / 1001.0);
    pts.remove(20);

    assert_eq!(pts.len(), 40);
    assert_eq!(check_constant_rate(&pts, MILLISECOND, ntsc_film()), Ok(()));
}

#[test]
fn one_tick_of_jitter_is_constant() {
    let pts: Vec<i64> = millisecond_stamps(900, 24000.0 / 1001.0)
        .into_iter()
        .enumerate()
        .map(|(frame, stamp)| stamp + (frame % 2) as i64)
        .collect();

    assert_eq!(check_constant_rate(&pts, MILLISECOND, ntsc_film()), Ok(()));
}

#[test]
fn a_mixed_rate_stream_is_variable() {
    let mut pts = millisecond_stamps(300, 30000.0 / 1001.0);
    let switch = *pts.last().expect("the first segment has frames");
    let tail = millisecond_stamps(101, 24000.0 / 1001.0);
    pts.extend(tail.into_iter().skip(1).map(|stamp| switch + stamp));

    let declared = Rational32::new(30, 1);
    let result = check_constant_rate(&pts, MILLISECOND, declared);
    let Err(VariableRate::Average { measured_fps, .. }) = result else {
        panic!("a mixed rate stream must fail on its average, got {result:?}");
    };

    assert!((measured_fps - 28.24).abs() < 0.1, "measured {measured_fps}");
}

#[test]
fn short_constant_rate_clips_on_millisecond_ticks_are_constant() {
    let cases = [
        (30000, 1001, 2),
        (60000, 1001, 2),
        (60000, 1001, 3),
        (120, 1, 2),
        (120, 1, 3),
        (120, 1, 5),
        (120000, 1001, 2),
        (120000, 1001, 3),
        (120000, 1001, 5),
    ];

    for (numerator, denominator, frames) in cases {
        let declared = Rational32::new(numerator, denominator);
        let fps = f64::from(numerator) / f64::from(denominator);
        let pts = millisecond_stamps(frames, fps);
        let result = check_constant_rate(&pts, MILLISECOND, declared);

        assert_eq!(result, Ok(()), "{numerator}/{denominator} with {frames} frames");
    }
}

#[test]
fn a_declared_rate_twice_the_real_one_is_rejected_as_a_multiple() {
    let pts = millisecond_stamps(200, 25.0);
    let declared = Rational32::new(50, 1);
    let result = check_constant_rate(&pts, MILLISECOND, declared);

    let Err(VariableRate::Multiple { share, frame_fps, .. }) = result else {
        panic!("25 fps declared as 50 must be a multiple, got {result:?}");
    };

    assert!(share > 0.99, "share {share}");
    assert!((frame_fps - 25.0).abs() < 0.01, "frame rate {frame_fps}");
}

#[test]
fn timestamps_too_coarse_for_the_period_are_not_checked() {
    // A 1/60 s tick cannot describe a 120 fps period.
    let pts: Vec<i64> = (0..100).map(|frame: i64| frame / 2).collect();
    let declared = Rational32::new(120, 1);

    assert_eq!(check_constant_rate(&pts, 1.0 / 60.0, declared), Ok(()));
}

#[test]
fn a_single_frame_is_constant() {
    assert_eq!(check_constant_rate(&[0], MILLISECOND, ntsc_film()), Ok(()));
}

#[test]
fn no_timestamps_are_constant() {
    assert_eq!(check_constant_rate(&[], MILLISECOND, ntsc_film()), Ok(()));
}

#[test]
fn duplicate_timestamps_are_variable() {
    let mut pts = millisecond_stamps(100, 24000.0 / 1001.0);
    let repeated = pts[50];
    pts.insert(50, repeated);
    pts.insert(80, pts[80]);

    assert!(check_constant_rate(&pts, MILLISECOND, ntsc_film()).is_err());
}

#[test]
fn frames_sharing_one_timestamp_have_their_own_error() {
    let pts = [40, 40, 40];
    let result = check_constant_rate(&pts, MILLISECOND, ntsc_film());

    let Err(rate) = result else {
        panic!("frames sharing one timestamp must be rejected");
    };
    let message = variable_rate_error(rate).to_string();

    assert_eq!(rate, VariableRate::NoSpan { frames: 3 });
    assert!(
        message.contains("all 3 of its frames with the same time"),
        "got {message}"
    );
    assert!(!message.contains("inf"), "got {message}");
}

#[test]
fn a_ten_frame_clip_with_one_dropped_frame_is_constant() {
    let mut pts = millisecond_stamps(11, 24000.0 / 1001.0);
    pts.remove(5);

    assert_eq!(pts.len(), 10);
    assert_eq!(check_constant_rate(&pts, MILLISECOND, ntsc_film()), Ok(()));
}

#[test]
fn intervals_off_the_period_are_variable_even_with_the_right_average() {
    // Alternating half and one and a half periods keeps the average on 24 fps.
    let pts: Vec<i64> = (0..400)
        .map(|frame: i64| {
            let pairs = frame / 2;
            let half_step = if frame % 2 == 1 { 21 } else { 0 };
            pairs * 83 + half_step
        })
        .collect();
    let declared = Rational32::new(24, 1);
    let result = check_constant_rate(&pts, MILLISECOND, declared);

    let Err(VariableRate::OffPeriod { share, .. }) = result else {
        panic!("alternating intervals must fall off the period, got {result:?}");
    };

    assert!(share > 0.99, "share {share}");
}

#[test]
fn whole_multiples_of_the_period_fit() {
    let period = 1000.0 / 24.0;

    assert!(fits_period(42, period));
    assert!(fits_period(83, period));
    assert!(fits_period(125, period));
    assert!(!fits_period(21, period));
    assert!(!fits_period(62, period));
}

#[test]
fn an_average_error_names_both_rates_and_a_conversion() {
    let rate = VariableRate::Average {
        measured_fps: 28.24,
        declared: Rational32::new(30, 1),
    };

    let message = variable_rate_error(rate).to_string();

    assert!(message.contains("variable frame rate"), "got {message}");
    assert!(message.contains("averages 28.240 fps"), "got {message}");
    assert!(message.contains("declared 30.000 fps"), "got {message}");
    assert!(message.contains("-fps_mode cfr -r 30/1"), "got {message}");
}

#[test]
fn an_off_period_error_names_the_share_of_frames() {
    let rate = VariableRate::OffPeriod {
        share: 0.25,
        declared: Rational32::new(24, 1),
    };

    let message = variable_rate_error(rate).to_string();

    assert!(message.contains("variable frame rate"), "got {message}");
    assert!(message.contains("25.0% of its frames fall off"), "got {message}");
    assert!(message.contains("24.000 fps"), "got {message}");
    assert!(!message.contains("averages"), "got {message}");
}

#[test]
fn a_multiple_error_says_the_declared_rate_looks_like_a_multiple() {
    let rate = VariableRate::Multiple {
        share: 1.0,
        frame_fps: 25.0,
        declared: Rational32::new(50, 1),
    };

    let message = variable_rate_error(rate).to_string();

    assert!(message.contains("declares 50.000 fps"), "got {message}");
    assert!(message.contains("multiple of its real rate"), "got {message}");
    assert!(message.contains("25.000 fps"), "got {message}");
}

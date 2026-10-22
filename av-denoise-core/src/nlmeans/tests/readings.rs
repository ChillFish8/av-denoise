use cubecl::prelude::*;

use super::noise_curve::{frame_dims, quarters_at, synthetic_records};
use crate::nlmeans::denoiser::readings::{ReadingShape, SlotNoiseReading};
use crate::noise::{QuarterSettings, partials_len, temporal_noise_reading};

#[test]
fn a_reading_with_temporal_stats_matches_the_temporal_reading() {
    let mut quarters = quarters_at(0.15, 0.02, 160);
    let middle_band = quarters_at(0.35, 0.01, 160);
    let bright_band = quarters_at(0.6, 0.005, 160);
    quarters.extend(middle_band);
    quarters.extend(bright_band);
    let quarter_count = quarters.len();
    let (width, height) = frame_dims(quarter_count);
    let records = synthetic_records(&quarters);

    let row = [0.1f32, 0.2, 0.3, 0.4];
    let partials_count = partials_len(width, height);
    let partials = vec![0.5f32; partials_count];
    let row_bytes = f32::as_bytes(&row);
    let partials_bytes = f32::as_bytes(&partials);
    let records_bytes = f32::as_bytes(&records);
    let buffers = [row_bytes, partials_bytes, records_bytes];
    let shape = ReadingShape {
        width,
        height,
        channels: 1,
        stored_ch: 1,
        with_curve: true,
        settings: QuarterSettings::default(),
    };

    let reading = SlotNoiseReading::from_bytes(&buffers, &shape);

    let settings = QuarterSettings::default();
    let expected = temporal_noise_reading(&records, 1, 1, width, height, true, settings);
    assert!(expected.sample.is_some());
    assert!(expected.curve.is_some());
    assert_eq!(reading.temporal.sample, expected.sample);
    assert_eq!(reading.temporal.curve, expected.curve);
    assert_eq!(reading.temporal.classes, expected.classes);
}

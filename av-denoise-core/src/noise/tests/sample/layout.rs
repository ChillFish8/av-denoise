use super::assert_query_run;
use crate::noise::NOISE_CHANNELS;
use crate::noise::curve::{NOISE_CURVE_BINS, NoiseCurve};
use crate::noise::reading::{
    BLOCK_SET_QUERIES,
    BLOCK_SET_TAGS,
    IMMERKAER_CHANNEL_QUERIES,
    IMMERKAER_QUERY_EXCLUDED_BASE,
    IMMERKAER_QUERY_REFERENCE_BASE,
    READING_CURVE_VALID,
    READING_SAMPLE_VALID_BASE,
    ReadingRecord,
    SAMPLE_CHANNEL_QUERIES,
    SAMPLE_QUERY_ACCEPTED,
    SAMPLE_QUERY_QUARTER,
    SAMPLE_QUERY_RHO,
    TAG_ACCEPTED,
    TAG_EXCLUDED_BASE,
    TAG_QUARTER,
    TAG_REFERENCE_BASE,
    TAG_RHO,
    immerkaer_queries,
    immerkaer_query_count,
    immerkaer_query_texture,
    sample_queries,
    sample_query_count,
    tag_moved_texture,
};
use crate::noise::select::RankRule;
use crate::noise::temporal::TemporalNoiseSample;

#[test]
fn the_query_indices_name_their_queries() {
    let quartile_pair = [RankRule::QuartileLow, RankRule::QuartileHigh];
    let median_pair = [RankRule::MedianLow, RankRule::MedianHigh];
    let sample_channel = [
        RankRule::MedianLow,
        RankRule::MedianHigh,
        RankRule::QuartileLow,
        RankRule::QuartileHigh,
    ];
    let reference = [
        RankRule::QuartileLow,
        RankRule::QuartileHigh,
        RankRule::First,
        RankRule::Last,
    ];

    for channels in 1..=NOISE_CHANNELS {
        let spatial = immerkaer_queries(false, channels);
        let immerkaer = immerkaer_queries(true, channels);
        let sample = sample_queries(channels);

        assert_eq!(spatial.len(), IMMERKAER_QUERY_REFERENCE_BASE as usize);
        assert_eq!(immerkaer.len(), immerkaer_query_count(channels) as usize);
        assert_eq!(sample.len(), sample_query_count(channels) as usize);

        for channel in 0..NOISE_CHANNELS {
            let immerkaer_first = IMMERKAER_CHANNEL_QUERIES * channel;
            let sample_first = SAMPLE_CHANNEL_QUERIES * channel;

            assert_query_run(&spatial, immerkaer_first, channel, &quartile_pair);
            assert_query_run(&immerkaer, immerkaer_first, channel, &quartile_pair);
            assert_query_run(&sample, sample_first, channel, &sample_channel);
        }

        for channel in 0..channels {
            let reference_query = IMMERKAER_QUERY_REFERENCE_BASE + BLOCK_SET_QUERIES * channel;
            let excluded_query = IMMERKAER_QUERY_EXCLUDED_BASE + BLOCK_SET_QUERIES * channel;
            let reference_tag = TAG_REFERENCE_BASE + BLOCK_SET_TAGS * channel;
            let excluded_tag = TAG_EXCLUDED_BASE + BLOCK_SET_TAGS * channel;
            let accepted_query = SAMPLE_QUERY_ACCEPTED + channel;
            let accepted_tag = TAG_ACCEPTED + channel;

            let texture_query = immerkaer_query_texture(channels) + channel;
            let texture_tag = tag_moved_texture(channels) + channel;

            assert_query_run(&immerkaer, reference_query, reference_tag, &reference);
            assert_query_run(&immerkaer, excluded_query, excluded_tag, &[RankRule::First]);
            assert_query_run(&immerkaer, texture_query, texture_tag, &[RankRule::TextureShare]);
            assert_query_run(&sample, accepted_query, accepted_tag, &[RankRule::First]);
        }

        assert_query_run(&sample, SAMPLE_QUERY_RHO, TAG_RHO, &median_pair);
        assert_query_run(&sample, SAMPLE_QUERY_QUARTER, TAG_QUARTER, &median_pair);
    }
}

#[test]
fn a_reading_survives_its_words() {
    let sample = TemporalNoiseSample {
        valid: [true; NOISE_CHANNELS as usize],
        sigma: [0.01, 0.02],
        sigma_low: [0.005, 0.015],
        rho: 0.3,
        static_fraction: 0.75,
    };
    let mut ratios = [1.0f32; NOISE_CURVE_BINS];
    ratios[3] = 1.25;
    let curve = NoiseCurve {
        ratios,
        sigma_quarter_median: 0.012,
    };
    let record = ReadingRecord {
        immerkaer_low: [0.004, 0.003],
        sample: Some(sample),
        accepted: 1234,
        curve: Some(curve),
        populated_bins: 5,
    };

    let words = record.to_words();
    let back = ReadingRecord::from_words(&words);
    assert_eq!(back, record);

    let empty = ReadingRecord {
        sample: None,
        curve: None,
        ..record
    };
    let empty_words = empty.to_words();
    let valid_start = READING_SAMPLE_VALID_BASE as usize;
    let valid_end = valid_start + NOISE_CHANNELS as usize;
    let sample_valid = &empty_words[valid_start..valid_end];
    let curve_valid = empty_words[READING_CURVE_VALID as usize];
    let empty_back = ReadingRecord::from_words(&empty_words);
    assert_eq!(sample_valid, [0.0; NOISE_CHANNELS as usize]);
    assert_eq!(curve_valid, 0.0);
    assert_eq!(empty_back, empty);

    let channel_1_only = TemporalNoiseSample {
        valid: [false, true],
        sigma: [0.0, 0.02],
        sigma_low: [0.0, 0.015],
        rho: 0.0,
        static_fraction: 0.0,
    };
    let partial = ReadingRecord {
        sample: Some(channel_1_only),
        curve: None,
        ..record
    };
    let partial_words = partial.to_words();
    let partial_valid = &partial_words[valid_start..valid_end];
    let partial_back = ReadingRecord::from_words(&partial_words);
    assert_eq!(partial_valid, [0.0, 1.0]);
    assert_eq!(partial_back, partial);
}

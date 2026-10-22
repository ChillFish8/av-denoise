use super::curve::{NOISE_CURVE_BINS, NoiseCurve};
use super::select::{RankRule, SelectQuery};
use super::temporal::TemporalNoiseSample;

/// The per-channel lower quartile of the Immerkær cube sigmas, 3 words.
pub(crate) const READING_IMMERKAER_LOW: u32 = 0;
/// 1.0 when the slot has a temporal sample, else 0.0.
pub(crate) const READING_SAMPLE_VALID: u32 = 3;
/// The sample's per-channel median sigma, 3 words.
pub(crate) const READING_SIGMA: u32 = 4;
/// The sample's per-channel lower-quartile sigma, 3 words.
pub(crate) const READING_SIGMA_LOW: u32 = 7;
/// The sample's median lag-1 correlation.
pub(crate) const READING_RHO: u32 = 10;
/// The sample's fraction of accepted blocks.
pub(crate) const READING_STATIC_FRACTION: u32 = 11;
/// How many blocks were accepted as static noise.
pub(crate) const READING_ACCEPTED: u32 = 12;
/// 1.0 when the slot has a noise curve, else 0.0.
pub(crate) const READING_CURVE_VALID: u32 = 13;
/// The median sigma of the quarters the curve was built from.
pub(crate) const READING_QUARTER_MEDIAN: u32 = 14;
/// How many curve bins held enough quarters for a median.
pub(crate) const READING_POPULATED_BINS: u32 = 15;
/// The curve's ratio per luma bin, 16 words.
pub(crate) const READING_RATIOS: u32 = 16;
/// The `f32` words of one slot's reading.
pub(crate) const READING_WORDS: u32 = READING_RATIOS + NOISE_CURVE_BINS as u32;

/// One slot's reading as a readback sees it.
///
/// Flags read as 0.0 or 1.0 and counts as exact `f32` values.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "no production code reads a reading back")
)]
pub(crate) struct ReadingRecord {
    pub immerkaer_low: [f32; 3],
    pub sample: Option<TemporalNoiseSample>,
    pub accepted: u32,
    pub curve: Option<NoiseCurve>,
    pub populated_bins: u32,
}

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "no production code reads a reading back")
)]
impl ReadingRecord {
    pub(crate) fn from_words(words: &[f32]) -> Self {
        let word_count = words.len();
        assert_eq!(word_count, READING_WORDS as usize, "one reading's words");

        let immerkaer_low = triple(words, READING_IMMERKAER_LOW);
        let sample_valid = words[READING_SAMPLE_VALID as usize] != 0.0;
        let sample = sample_valid.then(|| TemporalNoiseSample {
            sigma: triple(words, READING_SIGMA),
            sigma_low: triple(words, READING_SIGMA_LOW),
            rho: words[READING_RHO as usize],
            static_fraction: words[READING_STATIC_FRACTION as usize],
        });

        let curve_valid = words[READING_CURVE_VALID as usize] != 0.0;
        let ratios_start = READING_RATIOS as usize;
        let ratios_end = ratios_start + NOISE_CURVE_BINS;
        let curve = curve_valid.then(|| {
            let mut ratios = [0.0f32; NOISE_CURVE_BINS];
            ratios.copy_from_slice(&words[ratios_start..ratios_end]);

            NoiseCurve {
                ratios,
                sigma_quarter_median: words[READING_QUARTER_MEDIAN as usize],
            }
        });

        Self {
            immerkaer_low,
            sample,
            accepted: words[READING_ACCEPTED as usize] as u32,
            curve,
            populated_bins: words[READING_POPULATED_BINS as usize] as u32,
        }
    }

    /// The words of the reading, with every value of a missing sample or curve at 0.0.
    pub(crate) fn to_words(&self) -> [f32; READING_WORDS as usize] {
        let mut words = [0.0f32; READING_WORDS as usize];

        set_triple(&mut words, READING_IMMERKAER_LOW, self.immerkaer_low);

        if let Some(sample) = self.sample {
            words[READING_SAMPLE_VALID as usize] = 1.0;
            set_triple(&mut words, READING_SIGMA, sample.sigma);
            set_triple(&mut words, READING_SIGMA_LOW, sample.sigma_low);
            words[READING_RHO as usize] = sample.rho;
            words[READING_STATIC_FRACTION as usize] = sample.static_fraction;
        }

        words[READING_ACCEPTED as usize] = self.accepted as f32;

        if let Some(curve) = self.curve {
            let ratios_start = READING_RATIOS as usize;
            let ratios_end = ratios_start + NOISE_CURVE_BINS;
            words[READING_CURVE_VALID as usize] = 1.0;
            words[READING_QUARTER_MEDIAN as usize] = curve.sigma_quarter_median;
            words[ratios_start..ratios_end].copy_from_slice(&curve.ratios);
        }

        words[READING_POPULATED_BINS as usize] = self.populated_bins as f32;

        words
    }
}

fn triple(words: &[f32], start: u32) -> [f32; 3] {
    let start = start as usize;
    [words[start], words[start + 1], words[start + 2]]
}

fn set_triple(words: &mut [f32], start: u32, values: [f32; 3]) {
    let start = start as usize;
    words[start..start + 3].copy_from_slice(&values);
}

/// The immerkaer batch tag of a candidate block above the measurable-noise gate.
pub(crate) const TAG_REFERENCE: u32 = 3;
/// The immerkaer batch tag of a candidate block at or below the measurable-noise gate.
pub(crate) const TAG_EXCLUDED: u32 = 4;
/// The immerkaer batch's tags with temporal stats, three channels and the two block sets.
pub(crate) const IMMERKAER_TAGS: u32 = 5;
/// The immerkaer batch's tags without temporal stats, one per channel.
pub(crate) const IMMERKAER_SPATIAL_TAGS: u32 = 3;
/// The immerkaer batch's queries with temporal stats.
pub(crate) const IMMERKAER_QUERIES: u32 = 11;
/// The query of the reference set's lower quartile floor, followed by its ceiling, its first and
/// its last element.
pub(crate) const IMMERKAER_QUERY_REFERENCE: u32 = 6;
/// The query whose count says how many candidates the reference set left out.
pub(crate) const IMMERKAER_QUERY_EXCLUDED: u32 = 10;

/// The sample batch tag of an accepted block's clamped correlation.
pub(crate) const TAG_RHO: u32 = 3;
/// The sample batch tag of a static quarter's sigma.
pub(crate) const TAG_QUARTER: u32 = 4;
/// The sample batch's tags, three channels, the correlation and the quarters.
pub(crate) const SAMPLE_TAGS: u32 = 5;
/// The sample batch's queries.
pub(crate) const SAMPLE_QUERIES: u32 = 16;
/// The query of the correlation's lower median, followed by its upper median.
pub(crate) const SAMPLE_QUERY_RHO: u32 = 12;

/// The immerkaer batch's queries, a quartile pair per channel and, with temporal stats, the
/// reference set's quartile pair, first and last element, then the excluded set's first.
pub(crate) fn immerkaer_queries(temporal: bool) -> Vec<SelectQuery> {
    let mut queries = Vec::with_capacity(IMMERKAER_QUERIES as usize);

    for tag in 0..3 {
        queries.push(SelectQuery {
            tag,
            rule: RankRule::QuartileLow,
        });
        queries.push(SelectQuery {
            tag,
            rule: RankRule::QuartileHigh,
        });
    }

    if !temporal {
        return queries;
    }

    let reference_rules = [
        RankRule::QuartileLow,
        RankRule::QuartileHigh,
        RankRule::First,
        RankRule::Last,
    ];
    for rule in reference_rules {
        queries.push(SelectQuery {
            tag: TAG_REFERENCE,
            rule,
        });
    }

    queries.push(SelectQuery {
        tag: TAG_EXCLUDED,
        rule: RankRule::First,
    });

    queries
}

/// The sample batch's queries, a median pair and a quartile pair per channel, then the median
/// pairs of the correlation and the quarters.
pub(crate) fn sample_queries() -> Vec<SelectQuery> {
    let percentile_rules = [
        RankRule::MedianLow,
        RankRule::MedianHigh,
        RankRule::QuartileLow,
        RankRule::QuartileHigh,
    ];
    let median_rules = [RankRule::MedianLow, RankRule::MedianHigh];
    let mut queries = Vec::with_capacity(SAMPLE_QUERIES as usize);

    for tag in 0..3 {
        for rule in percentile_rules {
            queries.push(SelectQuery { tag, rule });
        }
    }

    for tag in [TAG_RHO, TAG_QUARTER] {
        for rule in median_rules {
            queries.push(SelectQuery { tag, rule });
        }
    }

    queries
}

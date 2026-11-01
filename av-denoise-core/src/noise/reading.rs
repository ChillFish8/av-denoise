use super::NOISE_CHANNELS;
use super::curve::NOISE_CURVE_BINS;
#[cfg(test)]
use super::curve::NoiseCurve;
use super::select::{RankRule, SelectQuery};
#[cfg(test)]
use super::temporal::TemporalNoiseSample;

/// The per-channel lower quartile of the Immerkær cube sigmas, one word per channel.
pub(crate) const READING_IMMERKAER_LOW: u32 = 0;
/// 1.0 when the slot has a temporal sample, else 0.0.
pub(crate) const READING_SAMPLE_VALID: u32 = READING_IMMERKAER_LOW + NOISE_CHANNELS;
/// The sample's per-channel median sigma, one word per channel.
pub(crate) const READING_SIGMA: u32 = READING_SAMPLE_VALID + 1;
/// The sample's per-channel lower-quartile sigma, one word per channel.
pub(crate) const READING_SIGMA_LOW: u32 = READING_SIGMA + NOISE_CHANNELS;
/// The sample's median lag-1 correlation.
pub(crate) const READING_RHO: u32 = READING_SIGMA_LOW + NOISE_CHANNELS;
/// The sample's fraction of accepted blocks.
pub(crate) const READING_STATIC_FRACTION: u32 = READING_RHO + 1;
/// How many blocks were accepted as static noise.
pub(crate) const READING_ACCEPTED: u32 = READING_STATIC_FRACTION + 1;
/// 1.0 when the slot has a noise curve, else 0.0.
pub(crate) const READING_CURVE_VALID: u32 = READING_ACCEPTED + 1;
/// The median sigma of the quarters the curve was built from.
pub(crate) const READING_QUARTER_MEDIAN: u32 = READING_CURVE_VALID + 1;
/// How many curve bins held enough quarters for a median.
pub(crate) const READING_POPULATED_BINS: u32 = READING_QUARTER_MEDIAN + 1;
/// The curve's ratio per luma bin, 16 words.
pub(crate) const READING_RATIOS: u32 = READING_POPULATED_BINS + 1;
/// The `f32` words of one slot's reading.
pub(crate) const READING_WORDS: u32 = READING_RATIOS + NOISE_CURVE_BINS as u32;

/// One slot's reading as a readback sees it.
///
/// Flags read as 0.0 or 1.0 and counts as exact `f32` values.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ReadingRecord {
    pub immerkaer_low: [f32; NOISE_CHANNELS as usize],
    pub sample: Option<TemporalNoiseSample>,
    pub accepted: u32,
    pub curve: Option<NoiseCurve>,
    pub populated_bins: u32,
}

#[cfg(test)]
impl ReadingRecord {
    pub(crate) fn from_words(words: &[f32]) -> Self {
        let word_count = words.len();
        assert_eq!(word_count, READING_WORDS as usize, "one reading's words");

        let immerkaer_low = channel_words(words, READING_IMMERKAER_LOW);
        let sample_valid = words[READING_SAMPLE_VALID as usize] != 0.0;
        let sample = sample_valid.then(|| TemporalNoiseSample {
            sigma: channel_words(words, READING_SIGMA),
            sigma_low: channel_words(words, READING_SIGMA_LOW),
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

        set_channel_words(&mut words, READING_IMMERKAER_LOW, self.immerkaer_low);

        if let Some(sample) = self.sample {
            words[READING_SAMPLE_VALID as usize] = 1.0;
            set_channel_words(&mut words, READING_SIGMA, sample.sigma);
            set_channel_words(&mut words, READING_SIGMA_LOW, sample.sigma_low);
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

#[cfg(test)]
fn channel_words(words: &[f32], start: u32) -> [f32; NOISE_CHANNELS as usize] {
    let start = start as usize;
    let end = start + NOISE_CHANNELS as usize;
    let mut values = [0.0f32; NOISE_CHANNELS as usize];
    values.copy_from_slice(&words[start..end]);

    values
}

#[cfg(test)]
fn set_channel_words(words: &mut [f32], start: u32, values: [f32; NOISE_CHANNELS as usize]) {
    let start = start as usize;
    let end = start + NOISE_CHANNELS as usize;
    words[start..end].copy_from_slice(&values);
}

/// The rules of each channel's immerkaer queries, its lower quartile pair.
const IMMERKAER_CHANNEL_RULES: [RankRule; 2] = [RankRule::QuartileLow, RankRule::QuartileHigh];
/// The rules of the reference set's queries, its lower quartile pair, its first and its last.
const REFERENCE_RULES: [RankRule; 4] = [
    RankRule::QuartileLow,
    RankRule::QuartileHigh,
    RankRule::First,
    RankRule::Last,
];
/// The rules of each channel's sample queries, its median pair then its lower quartile pair.
const SAMPLE_CHANNEL_RULES: [RankRule; 4] = [
    RankRule::MedianLow,
    RankRule::MedianHigh,
    RankRule::QuartileLow,
    RankRule::QuartileHigh,
];
/// The rules of the correlation's and the quarters' queries, a median pair each.
const MEDIAN_RULES: [RankRule; 2] = [RankRule::MedianLow, RankRule::MedianHigh];

/// The immerkaer batch's queries per channel.
pub(crate) const IMMERKAER_CHANNEL_QUERIES: u32 = IMMERKAER_CHANNEL_RULES.len() as u32;
/// The sample batch's queries per channel.
pub(crate) const SAMPLE_CHANNEL_QUERIES: u32 = SAMPLE_CHANNEL_RULES.len() as u32;

/// The immerkaer batch tags each channel's block sets take, its reference set then its excluded set.
pub(crate) const BLOCK_SET_TAGS: u32 = 2;
/// The immerkaer batch tag of channel 0's candidate blocks above the measurable-noise gate.
///
/// Channel `c`'s tag is `BLOCK_SET_TAGS * c` past it.
pub(crate) const TAG_REFERENCE_BASE: u32 = NOISE_CHANNELS;
/// The immerkaer batch tag of channel 0's candidate blocks at or below the measurable-noise gate.
///
/// Channel `c`'s tag is `BLOCK_SET_TAGS * c` past it.
pub(crate) const TAG_EXCLUDED_BASE: u32 = TAG_REFERENCE_BASE + 1;
/// The immerkaer batch's tags without temporal stats, one per channel.
pub(crate) const IMMERKAER_SPATIAL_TAGS: u32 = NOISE_CHANNELS;
/// The immerkaer batch's queries each channel's block sets take, the reference rules then the
/// excluded set's first.
pub(crate) const BLOCK_SET_QUERIES: u32 = REFERENCE_RULES.len() as u32 + 1;
/// The query of channel 0's reference lower quartile floor, followed by its ceiling, its first and
/// its last element.
///
/// Channel `c`'s queries start `BLOCK_SET_QUERIES * c` past it.
pub(crate) const IMMERKAER_QUERY_REFERENCE_BASE: u32 = NOISE_CHANNELS * IMMERKAER_CHANNEL_QUERIES;
/// The query whose count says how many of channel 0's candidates its reference set left out.
///
/// Channel `c`'s query is `BLOCK_SET_QUERIES * c` past it.
pub(crate) const IMMERKAER_QUERY_EXCLUDED_BASE: u32 =
    IMMERKAER_QUERY_REFERENCE_BASE + REFERENCE_RULES.len() as u32;

/// The immerkaer batch tag of a moved candidate block's texture, after each channel's block sets.
pub(crate) const fn tag_moved_texture(channels: u32) -> u32 {
    TAG_REFERENCE_BASE + BLOCK_SET_TAGS * channels
}

/// The immerkaer batch's tags with temporal stats, one per channel, each channel's block sets and
/// the moved texture set.
pub(crate) const fn immerkaer_tags(channels: u32) -> u32 {
    tag_moved_texture(channels) + 1
}

/// The query of the moved blocks' texture threshold, after each channel's block set queries.
pub(crate) const fn immerkaer_query_texture(channels: u32) -> u32 {
    IMMERKAER_QUERY_REFERENCE_BASE + BLOCK_SET_QUERIES * channels
}

/// The immerkaer batch's queries with temporal stats.
pub(crate) const fn immerkaer_query_count(channels: u32) -> u32 {
    immerkaer_query_texture(channels) + 1
}

/// The sample batch tag of an accepted block's clamped correlation.
pub(crate) const TAG_RHO: u32 = NOISE_CHANNELS;
/// The sample batch tag of a static quarter's sigma.
pub(crate) const TAG_QUARTER: u32 = TAG_RHO + 1;
/// The sample batch tag every block channel 0 accepts joins, so its count is the accepted count.
///
/// Channel `c`'s tag is `c` past it.
pub(crate) const TAG_ACCEPTED: u32 = TAG_QUARTER + 1;
/// The query of the correlation's lower median, followed by its upper median.
pub(crate) const SAMPLE_QUERY_RHO: u32 = NOISE_CHANNELS * SAMPLE_CHANNEL_QUERIES;
/// The query of the static quarters' lower median, followed by its upper median.
pub(crate) const SAMPLE_QUERY_QUARTER: u32 = SAMPLE_QUERY_RHO + MEDIAN_RULES.len() as u32;
/// The query whose count says how many blocks channel 0 accepted.
///
/// Channel `c`'s query is `c` past it.
pub(crate) const SAMPLE_QUERY_ACCEPTED: u32 = SAMPLE_QUERY_QUARTER + MEDIAN_RULES.len() as u32;

/// The sample batch's tags, one per channel, the correlation, the quarters and each channel's
/// accepted blocks.
pub(crate) const fn sample_tags(channels: u32) -> u32 {
    TAG_ACCEPTED + channels
}

/// The sample batch's queries.
pub(crate) const fn sample_query_count(channels: u32) -> u32 {
    SAMPLE_QUERY_ACCEPTED + channels
}

/// The curve batch's tags, one per luma bin.
pub(crate) const CURVE_TAGS: u32 = NOISE_CURVE_BINS as u32;
/// The curve batch's queries, a median pair per bin.
pub(crate) const CURVE_QUERIES: u32 = 2 * CURVE_TAGS;

/// The immerkaer batch's queries, a quartile pair per channel and, with temporal stats, each active
/// channel's block set queries, then the moved texture set's 40th percentile.
///
/// A channel's block set queries are its reference set's quartile pair, first and last element,
/// then its excluded set's first.
pub(crate) fn immerkaer_queries(temporal: bool, channels: u32) -> Vec<SelectQuery> {
    let mut queries = Vec::with_capacity(immerkaer_query_count(channels) as usize);

    for tag in 0..NOISE_CHANNELS {
        for rule in IMMERKAER_CHANNEL_RULES {
            queries.push(SelectQuery { tag, rule });
        }
    }

    if !temporal {
        return queries;
    }

    for channel in 0..channels {
        let reference = TAG_REFERENCE_BASE + BLOCK_SET_TAGS * channel;
        let excluded = TAG_EXCLUDED_BASE + BLOCK_SET_TAGS * channel;

        for rule in REFERENCE_RULES {
            queries.push(SelectQuery { tag: reference, rule });
        }

        queries.push(SelectQuery {
            tag: excluded,
            rule: RankRule::First,
        });
    }

    queries.push(SelectQuery {
        tag: tag_moved_texture(channels),
        rule: RankRule::TextureShare,
    });

    queries
}

/// The sample batch's queries, a median pair and a quartile pair per channel, the median pairs of
/// the correlation and the quarters, then each active channel's accepted set's first.
pub(crate) fn sample_queries(channels: u32) -> Vec<SelectQuery> {
    let mut queries = Vec::with_capacity(sample_query_count(channels) as usize);

    for tag in 0..NOISE_CHANNELS {
        for rule in SAMPLE_CHANNEL_RULES {
            queries.push(SelectQuery { tag, rule });
        }
    }

    for tag in [TAG_RHO, TAG_QUARTER] {
        for rule in MEDIAN_RULES {
            queries.push(SelectQuery { tag, rule });
        }
    }

    for channel in 0..channels {
        queries.push(SelectQuery {
            tag: TAG_ACCEPTED + channel,
            rule: RankRule::First,
        });
    }

    queries
}

/// The curve batch's queries, the lower then the upper median of each bin in turn.
pub(crate) fn curve_queries() -> Vec<SelectQuery> {
    let mut queries = Vec::with_capacity(CURVE_QUERIES as usize);

    for tag in 0..CURVE_TAGS {
        queries.push(SelectQuery {
            tag,
            rule: RankRule::MedianLow,
        });
        queries.push(SelectQuery {
            tag,
            rule: RankRule::MedianHigh,
        });
    }

    queries
}

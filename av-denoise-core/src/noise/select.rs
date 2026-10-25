use cubecl::prelude::*;
use cubecl::server::Handle;

use super::kernels::noise_select_histogram;
use crate::engine::MAX_GRID_1D;

/// The digit width of each of the three radix passes, from the top bit down.
pub(crate) const SELECT_DIGIT_BITS: [u32; 3] = [11, 11, 10];

/// How far each pass shifts a key right before taking its digit.
pub(crate) const SELECT_SHIFTS: [u32; 3] = [
    SELECT_DIGIT_BITS[1] + SELECT_DIGIT_BITS[2],
    SELECT_DIGIT_BITS[2],
    0,
];

/// Bins in one histogram row, enough for the widest digit.
pub(crate) const SELECT_BINS: u32 = 1 << SELECT_DIGIT_BITS[0];

/// Threads in one histogram cube, one element each.
pub const SELECT_CUBE_THREADS: u32 = 256;

/// Threads that walk one query's row together.
pub(crate) const SELECT_GROUP_THREADS: u32 = 32;

/// Bins each thread of a group sums.
pub(crate) const SELECT_BINS_PER_THREAD: u32 = SELECT_BINS / SELECT_GROUP_THREADS;

/// The words of one published slot, which are the prefix, the remaining rank, the count and the
/// written flag.
pub(crate) const SELECT_SLOT_WORDS: u32 = 4;

/// Published slots per query, one after each of the first two digits.
pub(crate) const SELECT_SLOTS: u32 = 2;

/// The words of one query in the query table, its tag and its rule.
pub const SELECT_QUERY_WORDS: u32 = 2;

/// The words of one tag in the tag table, its first query, its query count and its minimum count.
pub const SELECT_TAG_WORDS: u32 = 3;

/// The written flag of a slot no cube resolved, which reads as an empty set.
pub(crate) const SLOT_EMPTY: u32 = 0;

/// The written flag of a resolved slot.
pub(crate) const SLOT_WRITTEN: u32 = 1;

/// The written flag of a query whose tag holds fewer elements than its minimum count.
pub(crate) const SLOT_STOPPED: u32 = 2;

/// The tag that keeps an element out of every query.
pub const UNTAGGED: u32 = u32::MAX;

/// Most queries one batch answers.
pub(crate) const MAX_QUERIES: u32 = 32;

/// Which element of a sorted set a query picks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum RankRule {
    First = 0,
    Last = 1,
    /// `(count - 1) / 2`, the lower of the two middle elements.
    MedianLow = 2,
    /// `count / 2`.
    MedianHigh = 3,
    /// `(count - 1) / 4`, the floor of the quarter point.
    QuartileLow = 4,
    /// `(count + 2) / 4`, the ceiling of the quarter point.
    QuartileHigh = 5,
}

#[cfg(test)]
impl RankRule {
    /// The 0-based rank the rule picks from `count` elements. `count` must be above 0.
    pub(crate) fn rank(self, count: u32) -> u32 {
        assert!(count > 0, "a rank needs at least one element, got {count}");

        match self {
            RankRule::First => 0,
            RankRule::Last => count - 1,
            RankRule::MedianLow => (count - 1) / 2,
            RankRule::MedianHigh => count / 2,
            RankRule::QuartileLow => (count - 1) / 4,
            RankRule::QuartileHigh => (count + 2) / 4,
        }
    }
}

/// One order statistic, the element `rule` picks among the elements tagged `tag`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectQuery {
    pub tag: u32,
    pub rule: RankRule,
}

/// Order statistics over one tagged key array, answered by three radix passes.
pub(crate) struct SelectBatch {
    keys: Handle,
    tags: Handle,
    len: u32,
    query_table: Handle,
    tag_table: Handle,
    tag_count: u32,
    query_count: u32,
    histogram: Handle,
    state: Handle,
}

impl SelectBatch {
    /// Allocates the arrays for `len` elements, with every tag `UNTAGGED`, a zeroed histogram and
    /// an empty state.
    ///
    /// `min_counts` holds one entry per tag. `queries` must be grouped by tag.
    pub(crate) fn new<R: Runtime>(
        client: &ComputeClient<R>,
        len: u32,
        min_counts: &[u32],
        queries: &[SelectQuery],
    ) -> Self {
        let keys = vec![0u32; len as usize];
        let tags = vec![UNTAGGED; len as usize];

        Self::from_elements(client, &keys, &tags, min_counts, queries)
    }

    /// A batch whose keys and tags start as `keys` and `tags`.
    pub(crate) fn from_elements<R: Runtime>(
        client: &ComputeClient<R>,
        keys: &[u32],
        tags: &[u32],
        min_counts: &[u32],
        queries: &[SelectQuery],
    ) -> Self {
        let len = keys.len() as u32;
        let tag_count = min_counts.len() as u32;
        let query_count = queries.len() as u32;
        let cubes = len.div_ceil(SELECT_CUBE_THREADS);

        assert_eq!(tags.len(), keys.len(), "every key needs a tag");
        assert!(len > 0, "a select batch needs at least one element");
        assert!(cubes <= MAX_GRID_1D, "{cubes} cubes exceed the 1D grid limit");
        assert!(tag_count > 0, "a select batch needs at least one tag");
        assert!(
            tag_count <= SELECT_CUBE_THREADS,
            "{tag_count} tags exceed one flag per histogram thread"
        );
        assert!(query_count > 0, "a select batch needs at least one query");
        assert!(
            query_count <= MAX_QUERIES,
            "{query_count} queries exceed the batch limit of {MAX_QUERIES}"
        );

        let query_words = query_table(queries, tag_count);
        let tag_words = tag_table(queries, min_counts);

        let histogram_len = histogram_len(tag_count, query_count);
        let histogram = vec![0i32; histogram_len as usize];
        let state_len = state_len(query_count);
        let state = vec![SLOT_EMPTY; state_len as usize];

        let histogram_bytes = i32::as_bytes(&histogram);

        Self {
            keys: upload_words(client, keys),
            tags: upload_words(client, tags),
            len,
            query_table: upload_words(client, &query_words),
            tag_table: upload_words(client, &tag_words),
            tag_count,
            query_count,
            histogram: client.create_from_slice(histogram_bytes),
            state: upload_words(client, &state),
        }
    }

    pub(crate) fn keys(&self) -> &Handle {
        &self.keys
    }

    pub(crate) fn tags(&self) -> &Handle {
        &self.tags
    }

    pub(crate) fn histogram(&self) -> &Handle {
        &self.histogram
    }

    pub(crate) fn state(&self) -> &Handle {
        &self.state
    }

    pub(crate) fn query_table(&self) -> &Handle {
        &self.query_table
    }

    pub(crate) fn tag_table(&self) -> &Handle {
        &self.tag_table
    }

    pub(crate) fn len(&self) -> u32 {
        self.len
    }

    pub(crate) fn tag_count(&self) -> u32 {
        self.tag_count
    }

    pub(crate) fn query_count(&self) -> u32 {
        self.query_count
    }

    /// Words in the histogram buffer.
    pub(crate) fn histogram_len(&self) -> u32 {
        histogram_len(self.tag_count, self.query_count)
    }

    /// Words in the published state buffer.
    pub(crate) fn state_len(&self) -> u32 {
        state_len(self.query_count)
    }

    /// Queues the three histogram passes, returning how many launches that took.
    pub(crate) fn queue<R: Runtime>(&self, client: &ComputeClient<R>) -> u32 {
        let cubes = self.len.div_ceil(SELECT_CUBE_THREADS);
        let len = self.len as usize;
        let query_words = (self.query_count * SELECT_QUERY_WORDS) as usize;
        let tag_words = (self.tag_count * SELECT_TAG_WORDS) as usize;
        let state_len = self.state_len() as usize;

        let passes = SELECT_DIGIT_BITS.len() as u32;
        for pass in 0..passes {
            let reads = read_pass(pass);
            let (rows_in, rows_in_len) = self.pass_rows(reads);
            let (rows_out, rows_out_len) = self.pass_rows(pass);

            unsafe {
                noise_select_histogram::launch_unchecked::<R>(
                    client,
                    CubeCount::new_1d(cubes),
                    CubeDim::new_1d(SELECT_CUBE_THREADS),
                    ArrayArg::from_raw_parts(self.keys.clone(), len),
                    ArrayArg::from_raw_parts(self.tags.clone(), len),
                    ArrayArg::from_raw_parts(self.query_table.clone(), query_words),
                    ArrayArg::from_raw_parts(self.tag_table.clone(), tag_words),
                    ArrayArg::from_raw_parts(rows_in, rows_in_len),
                    ArrayArg::from_raw_parts(rows_out, rows_out_len),
                    ArrayArg::from_raw_parts(self.state.clone(), state_len),
                    pass,
                    self.len,
                    self.tag_count,
                    self.query_count,
                );
            }
        }

        passes
    }

    fn pass_rows(&self, pass: u32) -> (Handle, usize) {
        pass_rows(&self.histogram, self.tag_count, self.query_count, pass)
    }
}

/// The pass whose rows pass `pass` reads, which is pass 2 for pass 0, since it reads none.
pub fn read_pass(pass: u32) -> u32 {
    if pass == 0 { 2 } else { pass - 1 }
}

/// A view of the histogram rows pass `pass` adds to, with its length in words.
///
/// Pass 0 adds to the tag rows, and passes 1 and 2 add to one row per query.
pub fn pass_rows(histogram: &Handle, tag_count: u32, query_count: u32, pass: u32) -> (Handle, usize) {
    let (first_row, rows) = match pass {
        0 => (0, tag_count),
        1 => (tag_count, query_count),
        _ => (tag_count + query_count, query_count),
    };
    let total_rows = tag_count + 2 * query_count;
    let row_bytes = u64::from(SELECT_BINS) * size_of::<i32>() as u64;
    let start = u64::from(first_row) * row_bytes;
    let end_trim = u64::from(total_rows - first_row - rows) * row_bytes;
    let view = histogram.clone().offset_start(start).offset_end(end_trim);
    let words = (rows * SELECT_BINS) as usize;

    (view, words)
}

/// Threads in a cube that resolves a whole batch, capped at 1024.
///
/// A device offering fewer than 1024 threads per cube gets 256, which resolves 8 queries per round.
pub fn resolve_cube_threads<R: Runtime>(client: &ComputeClient<R>) -> u32 {
    let units = client.properties().hardware.max_units_per_cube;

    if units >= 1024 { 1024 } else { SELECT_CUBE_THREADS }
}

fn upload_words<R: Runtime>(client: &ComputeClient<R>, words: &[u32]) -> Handle {
    let bytes = u32::as_bytes(words);
    client.create_from_slice(bytes)
}

/// Histogram words for `tag_count` pass-0 rows and two rows per query.
pub fn histogram_len(tag_count: u32, query_count: u32) -> u32 {
    let rows = tag_count + 2 * query_count;
    rows * SELECT_BINS
}

/// State words for two slots per query.
pub fn state_len(query_count: u32) -> u32 {
    query_count * SELECT_SLOTS * SELECT_SLOT_WORDS
}

/// The `[tag, rule]` words of every query.
pub fn query_table(queries: &[SelectQuery], tag_count: u32) -> Vec<u32> {
    let mut words = Vec::with_capacity(queries.len() * SELECT_QUERY_WORDS as usize);

    for query in queries {
        assert!(
            query.tag < tag_count,
            "query tag {} is past {tag_count} tags",
            query.tag
        );
        words.push(query.tag);
        words.push(query.rule as u32);
    }

    words
}

/// The `[first_query, query_count, min_count]` words of every tag.
pub fn tag_table(queries: &[SelectQuery], min_counts: &[u32]) -> Vec<u32> {
    let grouped = queries.windows(2).all(|pair| pair[0].tag <= pair[1].tag);
    assert!(grouped, "queries must be grouped by tag");

    let mut words = Vec::with_capacity(min_counts.len() * SELECT_TAG_WORDS as usize);

    for (tag, &min_count) in min_counts.iter().enumerate() {
        let tag = tag as u32;
        let first = queries.partition_point(|query| query.tag < tag);
        let end = queries.partition_point(|query| query.tag <= tag);
        let count = end - first;

        words.push(first as u32);
        words.push(count as u32);
        words.push(min_count);
    }

    words
}

/// The order key of `value`, which sorts as the float does.
pub fn order_key_host(value: f32) -> u32 {
    let bits = value.to_bits();

    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000
    }
}

/// The float whose order key is `key`.
#[cfg(test)]
pub(crate) fn key_value_host(key: u32) -> f32 {
    let bits = if key & 0x8000_0000 != 0 {
        key ^ 0x8000_0000
    } else {
        !key
    };

    f32::from_bits(bits)
}

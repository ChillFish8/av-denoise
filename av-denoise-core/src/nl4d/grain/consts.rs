/// Side length of a measured block in pixels.
pub(crate) const CELL: u32 = 8;
pub(crate) const LUMA_BINS: usize = 16;
pub(crate) const STD_BUCKETS: usize = 64;
/// Counts in one histogram, luma bins by std buckets.
pub(crate) const HIST_LEN: usize = LUMA_BINS * STD_BUCKETS;
/// The 46 half-plane autocovariance lags, with `dy` in `0..=3` and `dx` in `-6..=6`.
pub(crate) const LAG_COUNT: usize = 46;
/// The lag sums plus the accepted pixel count.
pub(crate) const AUTOCOV_LEN: usize = LAG_COUNT + 1;
/// Groups of neighbouring std buckets, each with its own autocovariance record.
pub(crate) const STRENGTH_GROUPS: usize = 16;
pub(crate) const BUCKETS_PER_GROUP: usize = STD_BUCKETS / STRENGTH_GROUPS;
/// One autocovariance record per strength group.
pub(crate) const GROUPED_AUTOCOV_LEN: usize = STRENGTH_GROUPS * AUTOCOV_LEN;
/// Values in one frame's table statistics, the two histograms, the grouped lag sums and pixels.
pub(crate) const FRAME_TABLE_LEN: usize = 2 * HIST_LEN + GROUPED_AUTOCOV_LEN;
/// One cell's autocovariance record followed by its strength group.
pub(crate) const PARTIAL_LEN: usize = AUTOCOV_LEN + 1;
/// Threads per cube that reduce one lane of every cell's partial.
pub(crate) const REDUCE_THREADS: u32 = 128;
pub(crate) const MAX_LAG_Y: i32 = 3;
pub(crate) const MAX_LAG_X: i32 = 6;
pub(crate) const AR_COEFFS: usize = 24;

/// The measured lags in record order, `dy` first and then `dx`.
pub(crate) const LAGS: [(i32, i32); LAG_COUNT] = lag_table();

/// AV1's 24 causal lag-3 offsets in coefficient order, as `(dy, dx)`.
pub(crate) const AR_OFFSETS: [(i32, i32); AR_COEFFS] = ar_offset_table();

// A measured block needs a motion confidence of at least CONF_MIN, a denoised range below
// FLAT_RANGE, a denoised mean between LUMA_LOW and LUMA_HIGH, no pixel past CLIP_LOW or CLIP_HIGH,
// and a grain std above STD_MIN. A chroma persistence cell skips the luma band.
pub(crate) const FLAT_RANGE: f32 = 5.0 / 255.0;
pub(crate) const LUMA_LOW: f32 = 20.0 / 255.0;
pub(crate) const LUMA_HIGH: f32 = 235.0 / 255.0;
pub(crate) const CLIP_LOW: f32 = 4.0 / 255.0;
pub(crate) const CLIP_HIGH: f32 = 251.0 / 255.0;
pub(crate) const CONF_MIN: f32 = 0.8;
pub(crate) const STD_MIN: f32 = 0.05 / 255.0;
/// The top std bucket edge.
pub(crate) const STD_MAX: f32 = 32.0 / 255.0;

/// The furthest frame back a persistence reading compares. Every lag from 1 to this is measured.
pub(crate) const PERSISTENCE_LAGS: usize = 8;
/// Values in one lag's part of a frame record, the cell count, the auto sum and the cross sum.
pub(crate) const LAG_RECORD_LEN: usize = 3;
/// Values in one frame's record, one part per lag.
pub(crate) const FRAME_RECORD_LEN: usize = LAG_RECORD_LEN * PERSISTENCE_LAGS;
/// The furthest frame back a chroma persistence reading compares, lag 1 only.
pub(crate) const CHROMA_LAGS: usize = 1;
/// The chroma planes a chroma record holds, Cb then Cr.
pub(crate) const CHROMA_PLANES: usize = 2;
/// Values in one chroma frame's record, each chroma plane's lag parts in turn.
pub(crate) const CHROMA_RECORD_LEN: usize = CHROMA_PLANES * CHROMA_LAGS * LAG_RECORD_LEN;
/// Fewest cells for a frame to give a reading.
pub(crate) const MIN_FRAME_CELLS: f64 = 200.0;
/// Projections kept of each cell's source pixels.
pub(crate) const PROJECTIONS: usize = 8;
/// Completed frames the persistence history holds.
pub(crate) const HISTORY_FRAMES: u32 = 9;
/// Values kept of each cell per history frame, the projections and then the flat flag.
pub(crate) const HISTORY_ENTRY_LEN: usize = PROJECTIONS + 1;
/// Threads per cube that add one block of cells at one lag.
pub(crate) const PERSIST_THREADS: u32 = 256;
/// Cells each persisting thread adds.
pub(crate) const PERSIST_CELLS_PER_THREAD: u32 = 4;
/// Cells one persisting cube adds.
pub(crate) const PERSIST_BLOCK_CELLS: u32 = PERSIST_CELLS_PER_THREAD * PERSIST_THREADS;
/// The sign of each pixel in each projection, bit `y * 8 + x` set where the pixel counts plus one.
///
/// Each pattern has 32 set bits, and no two are equal or complementary. Each pattern ignores a
/// cell's mean and its horizontal and vertical slope.
pub(crate) const PATTERNS: [u64; PROJECTIONS] = [
    0x7866_974C_A28F_35D2,
    0x235B_8DA5_D0DE_5C91,
    0xC95E_0BA4_E0DF_D207,
    0xDC58_6702_1D7D_FA18,
    0x3E14_CCEC_2BB9_13D1,
    0xC379_12DD_286E_55C6,
    0x1277_D975_0989_CEE4,
    0x56D2_61D7_728A_23AD,
];

/// Fewest blocks for a luma bin to give a strength reading.
///
/// A bin's kept reading counts as 0 below it.
pub(crate) const MIN_BLOCKS_PER_BIN: u64 = 200;
/// Fewest readable luma bins for a segment to have its own strength.
pub(crate) const MIN_POPULATED_BINS: usize = 3;
/// Fewest source pixels for a segment to solve its own texture.
pub(crate) const MIN_AR_PIXELS: f64 = 50_000.0;
/// Most completed frames per [GrainChunk](crate::nl4d::grain::GrainChunk).
pub(crate) const CHUNK_FRAMES: u32 = 24;
/// The median source std ratio, either way, past which a chunk starts a new segment.
pub(crate) const DRIFT: f64 = 1.15;
/// Fewest source blocks for a chunk to start a new segment.
pub(crate) const MIN_CHUNK_BLOCKS: u64 = 2_000;
/// Templates measured when calibrating a texture's std.
pub(crate) const TEMPLATE_SEEDS: u32 = 24;
/// AV1 allows at most 14 luma scaling points.
pub(crate) const MAX_POINTS: usize = 14;
/// AV1 allows at most 10 scaling points per chroma plane.
pub(crate) const MAX_CHROMA_POINTS: usize = 10;

const fn lag_table() -> [(i32, i32); LAG_COUNT] {
    let mut table = [(0, 0); LAG_COUNT];
    let mut next = 0;
    let mut dy = 0;
    while dy <= MAX_LAG_Y {
        let mut dx = -MAX_LAG_X;
        while dx <= MAX_LAG_X {
            if dy > 0 || dx >= 0 {
                table[next] = (dy, dx);
                next += 1;
            }

            dx += 1;
        }

        dy += 1;
    }

    table
}

const fn ar_offset_table() -> [(i32, i32); AR_COEFFS] {
    let mut table = [(0, 0); AR_COEFFS];
    let mut next = 0;
    let mut dy = -3;
    while dy <= 0 {
        let mut dx = -3;
        while dx <= 3 {
            if dy < 0 || dx < 0 {
                table[next] = (dy, dx);
                next += 1;
            }

            dx += 1;
        }

        dy += 1;
    }

    table
}

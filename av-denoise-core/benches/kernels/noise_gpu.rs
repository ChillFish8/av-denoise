use std::ops::Range;
use std::sync::Arc;

use av_denoise_core::bench_api::MAX_GRID_1D;
use av_denoise_core::bench_api::noise_kernels::{noise_select_histogram, noise_select_resolve};
use av_denoise_core::bench_api::noise_reading::{NoiseReading, ReadingMap, ReadingStage};
use av_denoise_core::bench_api::noise_select::{
    RankRule,
    SELECT_CUBE_THREADS,
    SELECT_QUERY_WORDS,
    SELECT_TAG_WORDS,
    SelectQuery,
    UNTAGGED,
    histogram_len,
    order_key_host,
    pass_rows,
    query_table,
    read_pass,
    resolve_cube_threads,
    state_len,
    tag_table,
};
use cubecl::benchmark::{Benchmark, ProfileDuration};
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{HEIGHT, WIDTH, block_sync, make_padded_frame, stored_channels};

/// The 16x16 noise blocks of a 1080p plane.
const BLOCKS: u32 = (WIDTH / 16) * (HEIGHT / 16);
/// Noise blocks across one 1080p row.
const BLOCKS_X: u32 = WIDTH / 16;
/// Luma bins of the noise curve.
const CURVE_BINS: u32 = 16;

const PERCENTILE_RULES: [RankRule; 4] = [
    RankRule::MedianLow,
    RankRule::MedianHigh,
    RankRule::QuartileLow,
    RankRule::QuartileHigh,
];

const MEDIAN_RULES: [RankRule; 2] = [RankRule::MedianLow, RankRule::MedianHigh];

/// The luma strength map the class and map rows build, with the line ring and the texture veto on.
const READING_MAP: ReadingMap = ReadingMap {
    flat_boost: 1.5,
    shadow_soften: 0.65,
    texture_cut: 0.21,
    line_ring: 2,
};

/// Copies `length` words of `source` over `target`.
#[cube(launch_unchecked)]
fn refill_words(
    source: &Array<u32>,
    target: &mut Array<u32>,
    #[comptime] length: u32,
    #[comptime] total_threads: u32,
) {
    let mut index = ABSOLUTE_POS_X;
    while index < length {
        target[index as usize] = source[index as usize];
        index += total_threads;
    }
}

/// Which launch of a select batch a row times.
#[derive(Clone, Copy)]
pub enum SelectStage {
    /// One histogram pass, 0, 1 or 2.
    Pass(u32),
    /// The one-cube resolve of every query's last digit.
    Resolve,
}

/// The batch a row selects over.
#[derive(Clone, Copy)]
pub enum SelectShape {
    /// The luma sample batch, with 8 lanes per block, 5 tags and 16 queries.
    LumaSample,
    /// The curve batch, with 4 quarters per block over a luma gradient, 16 tags and 32 queries.
    ///
    /// Every minimum count is 0, so every bin walks its rows.
    Curve,
}

/// One launch of the radix select on a 1080p batch shape.
///
/// Values are sigma-like, between 2 and 12 out of 255, with rho between 0 and 0.6, so most keys share
/// a few pass-0 bins and same-address contention shows. The histogram and state are refilled from
/// the batch as it stood before the timed launch, outside the timed span.
pub struct SelectBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub stage: SelectStage,
    pub shape: SelectShape,
}

/// The arrays of one batch and the copies its histogram and state are refilled from.
#[derive(Clone)]
pub struct SelectInput {
    pub keys: Handle,
    pub tags: Handle,
    pub queries: Handle,
    pub tag_ranges: Handle,
    pub histogram: Handle,
    pub state: Handle,
    pub values: Handle,
    pub counts: Handle,
    pub histogram_source: Handle,
    pub state_source: Handle,
}

/// The elements and queries of one batch shape.
struct BatchData {
    values: Vec<f32>,
    tags: Vec<u32>,
    tag_count: u32,
    queries: Vec<SelectQuery>,
}

impl BatchData {
    fn len(&self) -> u32 {
        self.values.len() as u32
    }

    fn query_count(&self) -> u32 {
        self.queries.len() as u32
    }

    fn histogram_len(&self) -> u32 {
        histogram_len(self.tag_count, self.query_count())
    }

    fn state_len(&self) -> u32 {
        state_len(self.query_count())
    }

    /// The query table and the tag table, with every minimum count at 0.
    fn tables(&self) -> (Vec<u32>, Vec<u32>) {
        let min_counts = vec![0u32; self.tag_count as usize];
        let query_words = query_table(&self.queries, self.tag_count);
        let tag_words = tag_table(&self.queries, &min_counts);

        (query_words, tag_words)
    }
}

/// Every rule in `rules` for every tag in `tags`, grouped by tag.
fn queries_for(tags: Range<u32>, rules: &[RankRule]) -> Vec<SelectQuery> {
    let mut queries = Vec::new();

    for tag in tags {
        for &rule in rules {
            queries.push(SelectQuery { tag, rule });
        }
    }

    queries
}

/// A value between 0 and 1 from a hash of `index`.
fn hashed_unit(index: u32) -> f32 {
    let mut hash = index.wrapping_mul(0x9E37_79B9);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x2C1B_3C6D);
    hash ^= hash >> 12;

    (hash >> 8) as f32 / (1u32 << 24) as f32
}

fn sigma_like(index: u32) -> f32 {
    let unit = hashed_unit(index);
    (2.0 + 10.0 * unit) / 255.0
}

/// The luma sample batch of each accepted block's sigma, rho and four quarter sigmas.
fn luma_sample() -> BatchData {
    let lanes = (8 * BLOCKS) as usize;
    let mut values = vec![0.0f32; lanes];
    let mut tags = vec![UNTAGGED; lanes];

    for block in 0..BLOCKS {
        let accepted = block % 10 != 0;
        if !accepted {
            continue;
        }

        let sigma_lane = block as usize;
        values[sigma_lane] = sigma_like(block);
        tags[sigma_lane] = 0;

        let rho_lane = (3 * BLOCKS + block) as usize;
        let rho_unit = hashed_unit(block + BLOCKS);
        values[rho_lane] = 0.6 * rho_unit;
        tags[rho_lane] = 3;

        for quarter in 0..4 {
            let quarter_lane = (4 * BLOCKS + 4 * block + quarter) as usize;
            values[quarter_lane] = sigma_like(4 * block + quarter + 2 * BLOCKS);
            tags[quarter_lane] = 4;
        }
    }

    let mut queries = queries_for(0..3, &PERCENTILE_RULES);
    let median_queries = queries_for(3..5, &MEDIAN_RULES);
    queries.extend(median_queries);

    BatchData {
        values,
        tags,
        tag_count: 5,
        queries,
    }
}

/// The curve batch of every flat quarter's sigma, tagged with its bin on a horizontal luma gradient.
fn curve() -> BatchData {
    let lanes = (4 * BLOCKS) as usize;
    let mut values = vec![0.0f32; lanes];
    let mut tags = vec![UNTAGGED; lanes];
    let quarters_x = 2 * BLOCKS_X;

    for block in 0..BLOCKS {
        for quarter in 0..4 {
            let lane = 4 * block + quarter;
            let flat = lane % 7 != 0;
            if !flat {
                continue;
            }

            let quarter_x = 2 * (block % BLOCKS_X) + quarter % 2;
            let luma = (quarter_x as f32 + 0.5) / quarters_x as f32;
            let bin = (luma * CURVE_BINS as f32) as u32;
            values[lane as usize] = sigma_like(lane);
            tags[lane as usize] = bin.min(CURVE_BINS - 1);
        }
    }

    let queries = queries_for(0..CURVE_BINS, &MEDIAN_RULES);

    BatchData {
        values,
        tags,
        tag_count: CURVE_BINS,
        queries,
    }
}

impl<R: Runtime> SelectBench<R> {
    fn data(&self) -> BatchData {
        match self.shape {
            SelectShape::LumaSample => luma_sample(),
            SelectShape::Curve => curve(),
        }
    }

    fn upload_words(&self, words: &[u32]) -> Handle {
        let bytes = u32::as_bytes(words);
        self.client.create_from_slice(bytes)
    }

    fn queue_pass(&self, input: &SelectInput, data: &BatchData, pass: u32) {
        let len = data.len();
        let cubes = len.div_ceil(SELECT_CUBE_THREADS);
        let query_words = (data.query_count() * SELECT_QUERY_WORDS) as usize;
        let tag_words = (data.tag_count * SELECT_TAG_WORDS) as usize;
        let query_count = data.query_count();
        let reads = read_pass(pass);
        let (rows_in, rows_in_len) = pass_rows(&input.histogram, data.tag_count, query_count, reads);
        let (rows_out, rows_out_len) = pass_rows(&input.histogram, data.tag_count, query_count, pass);

        unsafe {
            noise_select_histogram::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(cubes),
                CubeDim::new_1d(SELECT_CUBE_THREADS),
                ArrayArg::from_raw_parts(input.keys.clone(), len as usize),
                ArrayArg::from_raw_parts(input.tags.clone(), len as usize),
                ArrayArg::from_raw_parts(input.queries.clone(), query_words),
                ArrayArg::from_raw_parts(input.tag_ranges.clone(), tag_words),
                ArrayArg::from_raw_parts(rows_in, rows_in_len),
                ArrayArg::from_raw_parts(rows_out, rows_out_len),
                ArrayArg::from_raw_parts(input.state.clone(), data.state_len() as usize),
                pass,
                len,
                data.tag_count,
                data.query_count(),
            );
        }
    }

    fn queue_resolve(&self, input: &SelectInput, data: &BatchData) {
        let threads = resolve_cube_threads(&self.client);
        let query_count = data.query_count();
        let query_words = (query_count * SELECT_QUERY_WORDS) as usize;
        let tag_words = (data.tag_count * SELECT_TAG_WORDS) as usize;

        unsafe {
            noise_select_resolve::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(1),
                CubeDim::new_1d(threads),
                ArrayArg::from_raw_parts(input.histogram.clone(), data.histogram_len() as usize),
                ArrayArg::from_raw_parts(input.state.clone(), data.state_len() as usize),
                ArrayArg::from_raw_parts(input.queries.clone(), query_words),
                ArrayArg::from_raw_parts(input.tag_ranges.clone(), tag_words),
                ArrayArg::from_raw_parts(input.values.clone(), query_count as usize),
                ArrayArg::from_raw_parts(input.counts.clone(), query_count as usize),
                data.tag_count,
                query_count,
                threads,
            );
        }
    }

    fn refill(&self, source: &Handle, target: &Handle, length: u32) {
        let dim = 256u32;
        let grid = length.div_ceil(dim).min(MAX_GRID_1D);
        let source_arg = unsafe { ArrayArg::from_raw_parts(source.clone(), length as usize) };
        let target_arg = unsafe { ArrayArg::from_raw_parts(target.clone(), length as usize) };

        unsafe {
            refill_words::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(dim),
                source_arg,
                target_arg,
                length,
                grid * dim,
            );
        }
    }

    fn queue_stage(&self, input: &SelectInput, data: &BatchData) {
        match self.stage {
            SelectStage::Pass(pass) => self.queue_pass(input, data, pass),
            SelectStage::Resolve => self.queue_resolve(input, data),
        }
    }

    /// The passes that run before the timed stage.
    fn passes_before(&self) -> u32 {
        match self.stage {
            SelectStage::Pass(pass) => pass,
            SelectStage::Resolve => 3,
        }
    }
}

impl<R: Runtime> Benchmark for SelectBench<R> {
    type Input = SelectInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let data = self.data();
        let keys: Vec<u32> = data.values.iter().copied().map(order_key_host).collect();
        let (query_words, tag_words) = data.tables();
        let histogram_zeros = vec![0u32; data.histogram_len() as usize];
        let state_zeros = vec![0u32; data.state_len() as usize];
        let result_zeros = vec![0u32; data.query_count() as usize];

        let input = SelectInput {
            keys: self.upload_words(&keys),
            tags: self.upload_words(&data.tags),
            queries: self.upload_words(&query_words),
            tag_ranges: self.upload_words(&tag_words),
            histogram: self.upload_words(&histogram_zeros),
            state: self.upload_words(&state_zeros),
            values: self.upload_words(&result_zeros),
            counts: self.upload_words(&result_zeros),
            histogram_source: self.upload_words(&histogram_zeros),
            state_source: self.upload_words(&state_zeros),
        };

        for pass in 0..self.passes_before() {
            self.queue_pass(&input, &data, pass);
        }

        self.refill(&input.histogram, &input.histogram_source, data.histogram_len());
        self.refill(&input.state, &input.state_source, data.state_len());
        self.sync();

        input
    }

    /// Refills the histogram and state, then times the stage on the device with the refill
    /// outside the span.
    fn profile_full(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        let data = self.data();
        self.refill(&args.histogram_source, &args.histogram, data.histogram_len());
        self.refill(&args.state_source, &args.state, data.state_len());
        self.sync();

        let profiled = self
            .client
            .profile(|| self.queue_stage(&args, &data), "noise_select");
        let (_, duration) = profiled.map_err(|error| format!("profile failed: {error:?}"))?;

        Ok(duration)
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        let data = self.data();
        self.queue_stage(&args, &data);

        Ok(())
    }

    fn name(&self) -> String {
        let stage = match self.stage {
            SelectStage::Pass(pass) => format!("histogram_pass{pass}"),
            SelectStage::Resolve => "resolve".to_string(),
        };
        let shape = match self.shape {
            SelectShape::LumaSample => "luma",
            SelectShape::Curve => "curve",
        };

        format!("noise_select/{stage}/1080p_{shape}")
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        let data = self.data();
        let len = data.len() as usize;
        let tag_count = data.tag_count as usize;
        let query_count = data.queries.len();

        vec![vec![len, tag_count, query_count]]
    }
}

/// One launch of a slot's noise reading on a 1080p luma or chroma plane.
///
/// The stats come from the real stats kernels over a frame and a copy with hashed noise added, so
/// most blocks are static noise. The launches before the timed one run in `prepare`. The luma rows
/// measure the quarter fields, as nl4d does, except the plain finish row, which leaves the curve
/// out. The class and map rows also build a luma strength map. The curve batch's passes add to its
/// histogram, so it is refilled from a copy outside the timed span.
pub struct ReadingBench<R: Runtime> {
    pub client: ComputeClient<R>,
    pub stage: ReadingStage,
    pub chroma: bool,
    /// Whether a luma row reads a smooth luma gradient, whose quarters fill every curve bin, and
    /// builds the curve.
    pub curve: bool,
}

/// A prepared reading and the copies of its batch buffers.
#[derive(Clone)]
pub struct ReadingInput<R: Runtime> {
    pub reading: Arc<NoiseReading<R>>,
    /// Each batch buffer, its copy and its length in words.
    pub buffers: Vec<(Handle, Handle, u32)>,
}

impl<R: Runtime> ReadingBench<R> {
    /// The width, height, channels and stored channels of the plane.
    fn plane(&self) -> (u32, u32, u32, u32) {
        if self.chroma {
            (WIDTH / 2, HEIGHT / 2, 2, stored_channels(2))
        } else {
            (WIDTH, HEIGHT, 1, stored_channels(1))
        }
    }

    /// The strength map the reading builds, for the class and map rows only.
    fn map(&self) -> Option<ReadingMap> {
        let builds_map = matches!(self.stage, ReadingStage::Classify | ReadingStage::StrengthMap);
        builds_map.then_some(READING_MAP)
    }

    /// Whether the stats carry the luma quarter fields.
    fn luma_fields(&self) -> bool {
        let finish = self.stage == ReadingStage::SampleFinish;
        let plain_finish = finish && !self.curve;

        !self.chroma && !plain_finish
    }

    /// A horizontal luma gradient between 0.05 and 0.95, gently bent along each column.
    fn gradient_frame(&self) -> Vec<f32> {
        let mut frame = Vec::with_capacity((WIDTH * HEIGHT) as usize);
        let span = (WIDTH - 1) as f32;

        for y in 0..HEIGHT {
            let bend = 0.02 * (y as f32 * 0.01).sin();

            for x in 0..WIDTH {
                let ramp = 0.05 + 0.9 * x as f32 / span;
                let sample = (ramp + bend).clamp(0.05, 0.95);
                frame.push(sample);
            }
        }

        frame
    }

    /// Two frames, the second with up to 0.02 of hashed noise added to every sample.
    fn ring(&self) -> Handle {
        let (width, height, channels, _) = self.plane();
        let frame = if self.curve {
            self.gradient_frame()
        } else {
            make_padded_frame(width, height, channels)
        };
        let mut ring = frame.clone();

        for (index, &sample) in frame.iter().enumerate() {
            let unit = hashed_unit(index as u32);
            let noisy = sample + 0.04 * (unit - 0.5);
            ring.push(noisy);
        }

        let bytes = f32::as_bytes(&ring);
        self.client.create_from_slice(bytes)
    }

    fn refill_buffers(&self, buffers: &[(Handle, Handle, u32)], restore: bool) {
        for (buffer, copy, length) in buffers {
            if restore {
                self.refill(copy, buffer, *length);
            } else {
                self.refill(buffer, copy, *length);
            }
        }
    }

    fn refill(&self, source: &Handle, target: &Handle, length: u32) {
        let dim = 256u32;
        let grid = length.div_ceil(dim).min(MAX_GRID_1D);
        let source_arg = unsafe { ArrayArg::from_raw_parts(source.clone(), length as usize) };
        let target_arg = unsafe { ArrayArg::from_raw_parts(target.clone(), length as usize) };

        unsafe {
            refill_words::launch_unchecked::<R>(
                &self.client,
                CubeCount::new_1d(grid),
                CubeDim::new_1d(dim),
                source_arg,
                target_arg,
                length,
                grid * dim,
            );
        }
    }
}

impl<R: Runtime> Benchmark for ReadingBench<R> {
    type Input = ReadingInput<R>;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        let (width, height, channels, stored_ch) = self.plane();
        let ring = self.ring();
        let luma_fields = self.luma_fields();
        let map = self.map();
        let reading = NoiseReading::new(
            &self.client,
            width,
            height,
            channels,
            stored_ch,
            &ring,
            luma_fields,
            map,
        )
        .expect("the stats queue");
        reading.queue_before(self.stage);

        let mut buffers = Vec::new();
        if self.stage == ReadingStage::CurveBatch {
            for (buffer, length) in reading.curve_batch_buffers() {
                let bytes = length as usize * size_of::<u32>();
                let copy = self.client.empty(bytes);
                buffers.push((buffer, copy, length));
            }
        }

        self.refill_buffers(&buffers, false);
        self.sync();

        ReadingInput {
            reading: Arc::new(reading),
            buffers,
        }
    }

    /// Refills the curve batch's buffers, then times the stage on the device with the refill
    /// outside the span.
    fn profile_full(&self, args: Self::Input) -> Result<ProfileDuration, String> {
        self.refill_buffers(&args.buffers, true);
        self.sync();

        let profiled = self
            .client
            .profile(|| args.reading.queue_stage(self.stage), "noise_reading");
        let (_, duration) = profiled.map_err(|error| format!("profile failed: {error:?}"))?;

        Ok(duration)
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        args.reading.queue_stage(self.stage);

        Ok(())
    }

    fn name(&self) -> String {
        let stage = match (self.stage, self.curve) {
            (ReadingStage::BlockDerive, _) => "block_derive",
            (ReadingStage::BlockAccept, _) => "block_accept",
            (ReadingStage::CurveBins, _) => "curve_bins",
            (ReadingStage::CurveBatch, _) => "curve_batch",
            (ReadingStage::SampleFinish, false) => "sample_finish",
            (ReadingStage::SampleFinish, true) => "sample_finish_curve",
            (ReadingStage::Classify, _) => "classify",
            (ReadingStage::StrengthMap, _) => "strength_map",
        };
        let plane = if self.chroma { "chroma" } else { "luma" };

        format!("noise_reading/{stage}/1080p_{plane}")
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        let (width, height, channels, _) = self.plane();

        vec![vec![width as usize, height as usize, channels as usize]]
    }
}

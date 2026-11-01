use av_denoise_core::bench_api::kernels::motion::{
    BLOCK_MATCH_THREADS,
    FLAG_THREADS,
    NEIGHBOUR_DISAGREEMENT,
    STATE_SECTIONS,
    mc_block_match_fine,
    mc_block_match_fine_tracked,
    mc_escalate_search,
    mc_flag_failures,
};
use av_denoise_core::bench_api::{DERIVED_SAD_FLOOR, DERIVED_WORDS, DerivedValues};
use cubecl::benchmark::Benchmark;
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{HEIGHT, WIDTH, block_sync, make_synthetic_frame, neighbour_bench_name, shapes_with_channels};

const FINE_BLKSIZE: u32 = 16;
const FINE_STEP: u32 = 8;
/// The library's `thsad` at a 16-pixel block and a scale of 1.
const THSAD: f32 = 5.12;
/// The grain added to every frame of the grain rows, 4 code values at 8 bits.
const GRAIN_SIGMA: f32 = 4.0 / 255.0;
/// The first-pass radius of the adaptive grain row.
const GRAIN_FIRST_RADIUS: u32 = 2;
/// The radius of each widening stage the adaptive grain row runs after its first pass.
const GRAIN_STAGE_RADII: [u32; 2] = [3, 4];
/// The vector a disagreeing block holds, past [NEIGHBOUR_DISAGREEMENT] from its zero neighbours.
const OUTLIER_DX: i32 = NEIGHBOUR_DISAGREEMENT + 2;

/// How many blocks fail, through vectors that disagree with their neighbours.
#[derive(Clone, Copy)]
pub enum FailingShare {
    None,
    /// Every tenth block, none of them grid neighbours.
    Tenth,
    /// A checkerboard of disagreeing vectors, so every block fails.
    All,
}

impl FailingShare {
    pub const ALL: &[FailingShare] = &[FailingShare::None, FailingShare::Tenth, FailingShare::All];

    fn label(self) -> &'static str {
        match self {
            FailingShare::None => "none",
            FailingShare::Tenth => "10pct",
            FailingShare::All => "all",
        }
    }

    fn is_outlier(self, col: u32, row: u32) -> bool {
        match self {
            FailingShare::None => false,
            FailingShare::Tenth => (col + 3 * row).is_multiple_of(10),
            FailingShare::All => (col + row).is_multiple_of(2),
        }
    }
}

fn blocks_x() -> u32 {
    WIDTH.div_ceil(FINE_STEP)
}

fn blocks_y() -> u32 {
    HEIGHT.div_ceil(FINE_STEP)
}

fn block_count() -> u32 {
    blocks_x() * blocks_y()
}

#[derive(Clone)]
pub struct EscalateInput {
    /// The centre frame in slot 0 and an identical neighbour in each slot after it.
    pub pyramid: Handle,
    pub neighbour_slots: Handle,
    pub mv_field: Handle,
    pub confidence: Handle,
    pub sad: Handle,
    pub state: Handle,
    /// The derived noise values, with SAD floors of 0.
    pub noise: Handle,
    /// Bound as the unread predictions.
    pub placeholder: Handle,
}

/// Static frames with `share` of blocks holding a disagreeing vector, each kept SAD at 0.
///
/// No search beats a kept SAD of 0, so the field never changes and every run repeats the same work.
fn escalate_input(client: &Client, neighbours: u32, share: FailingShare) -> EscalateInput {
    let frame = make_synthetic_frame(WIDTH, HEIGHT, 1);
    let frames = frame.repeat(neighbours as usize + 1);
    let frame_bytes = f32::as_bytes(&frames);
    let pyramid = client.create_from_slice(frame_bytes);
    let table: Vec<u32> = (1..=neighbours).collect();
    let table_bytes = u32::as_bytes(&table);
    let neighbour_slots = client.create_from_slice(table_bytes);

    let blocks = block_count() as usize;
    let mut vectors = vec![0i32; 2 * blocks];
    for row in 0..blocks_y() {
        for col in 0..blocks_x() {
            if share.is_outlier(col, row) {
                let block = (row * blocks_x() + col) as usize;
                vectors[2 * block] = OUTLIER_DX;
            }
        }
    }

    let mut state = vec![0.0f32; STATE_SECTIONS as usize * blocks];
    for block in 0..blocks {
        state[2 * blocks + block] = vectors[2 * block] as f32;
        state[3 * blocks + block] = vectors[2 * block + 1] as f32;
    }

    let all_vectors = vectors.repeat(neighbours as usize);
    let all_state = neighbour_major_state(&state, blocks, neighbours);
    let vector_bytes = i32::as_bytes(&all_vectors);
    let state_bytes = f32::as_bytes(&all_state);
    let neighbours = neighbours as usize;

    let noise_words = DerivedValues::default().to_words();
    let noise_bytes = f32::as_bytes(&noise_words);

    EscalateInput {
        pyramid,
        neighbour_slots,
        mv_field: client.create_from_slice(vector_bytes),
        confidence: client.empty(neighbours * blocks * size_of::<f32>()),
        sad: client.empty(2 * neighbours * blocks * size_of::<f32>()),
        state: client.create_from_slice(state_bytes),
        noise: client.create_from_slice(noise_bytes),
        placeholder: client.empty(size_of::<f32>()),
    }
}

/// Repeats each section of one neighbour's state once per neighbour, sections outermost.
fn neighbour_major_state(state: &[f32], blocks: usize, neighbours: u32) -> Vec<f32> {
    let mut all_state = Vec::with_capacity(state.len() * neighbours as usize);

    for section in state.chunks(blocks) {
        for _ in 0..neighbours {
            all_state.extend_from_slice(section);
        }
    }

    all_state
}

fn launch_flags(client: &Client, input: &EscalateInput, neighbours: u32) {
    let blocks = block_count();
    let mv_len = (2 * blocks * neighbours) as usize;
    let state_len = (STATE_SECTIONS * blocks * neighbours) as usize;
    let cubes_x = blocks_x().div_ceil(FLAG_THREADS);
    let grid = CubeCount::new_3d(cubes_x, blocks_y(), neighbours);
    let dim = CubeDim::new_1d(FLAG_THREADS);

    unsafe {
        let mv_field_arg = BufferArg::from_raw_parts(input.mv_field.clone(), mv_len);
        let state_arg = BufferArg::from_raw_parts(input.state.clone(), state_len);
        let noise_arg = BufferArg::from_raw_parts(input.noise.clone(), DERIVED_WORDS as usize);

        mc_flag_failures::launch_unchecked(
            client,
            grid,
            dim,
            mv_field_arg,
            state_arg,
            noise_arg,
            DERIVED_SAD_FLOOR,
            THSAD,
            blocks_x(),
            blocks_y(),
            2 * blocks,
            blocks,
        );
    }
}

fn launch_escalate(client: &Client, input: &EscalateInput, neighbours: u32, radius: u32) {
    let blocks = block_count();
    let level_len = WIDTH * HEIGHT;
    let pyramid_len = ((neighbours + 1) * level_len) as usize;
    let mv_len = (2 * blocks * neighbours) as usize;
    let conf_len = (blocks * neighbours) as usize;
    let state_len = (STATE_SECTIONS * blocks * neighbours) as usize;
    let grid = CubeCount::new_3d(blocks_x(), blocks_y(), neighbours);
    let dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);

    unsafe {
        let pyramid_arg = BufferArg::from_raw_parts(input.pyramid.clone(), pyramid_len);
        let slots_arg = BufferArg::from_raw_parts(input.neighbour_slots.clone(), neighbours as usize);
        let mv_field_arg = BufferArg::from_raw_parts(input.mv_field.clone(), mv_len);
        let confidence_arg = BufferArg::from_raw_parts(input.confidence.clone(), conf_len);
        let sad_arg = BufferArg::from_raw_parts(input.sad.clone(), 2 * conf_len);
        let state_arg = BufferArg::from_raw_parts(input.state.clone(), state_len);
        let noise_arg = BufferArg::from_raw_parts(input.noise.clone(), DERIVED_WORDS as usize);

        mc_escalate_search::launch_unchecked(
            client,
            grid,
            dim,
            pyramid_arg,
            slots_arg,
            mv_field_arg,
            confidence_arg,
            sad_arg,
            state_arg,
            0u32,
            true,
            true,
            noise_arg,
            DERIVED_SAD_FLOOR,
            THSAD,
            WIDTH,
            HEIGHT,
            level_len,
            FINE_BLKSIZE,
            FINE_STEP,
            radius,
            blocks_x(),
            2 * blocks,
            blocks,
        );
    }
}

fn launch_tracked(client: &Client, input: &EscalateInput, neighbours: u32, radius: u32) {
    let blocks = block_count();
    let level_len = WIDTH * HEIGHT;
    let pyramid_len = ((neighbours + 1) * level_len) as usize;
    let mv_len = (2 * blocks * neighbours) as usize;
    let conf_len = (blocks * neighbours) as usize;
    let state_len = (STATE_SECTIONS * blocks * neighbours) as usize;
    let grid = CubeCount::new_3d(blocks_x(), blocks_y(), neighbours);
    let dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);

    unsafe {
        let pyramid_arg = BufferArg::from_raw_parts(input.pyramid.clone(), pyramid_len);
        let slots_arg = BufferArg::from_raw_parts(input.neighbour_slots.clone(), neighbours as usize);
        let mv_field_arg = BufferArg::from_raw_parts(input.mv_field.clone(), mv_len);
        let confidence_arg = BufferArg::from_raw_parts(input.confidence.clone(), conf_len);
        let sad_arg = BufferArg::from_raw_parts(input.placeholder.clone(), 1);
        let state_arg = BufferArg::from_raw_parts(input.state.clone(), state_len);
        let predictions_arg = BufferArg::from_raw_parts(input.placeholder.clone(), 1);
        let noise_arg = BufferArg::from_raw_parts(input.noise.clone(), DERIVED_WORDS as usize);

        mc_block_match_fine_tracked::launch_unchecked(
            client,
            grid,
            dim,
            pyramid_arg,
            slots_arg,
            mv_field_arg,
            confidence_arg,
            sad_arg,
            state_arg,
            predictions_arg,
            0u32,
            true,
            false,
            false,
            noise_arg,
            DERIVED_SAD_FLOOR,
            THSAD,
            WIDTH,
            HEIGHT,
            level_len,
            FINE_BLKSIZE,
            FINE_STEP,
            radius,
            0u32,
            blocks_x(),
            2 * blocks,
            blocks,
        );
    }
}

/// The failure flags over a 1080p luma field.
pub struct FlagFailuresBench {
    pub client: Client,
    pub neighbours: u32,
}

impl Benchmark for FlagFailuresBench {
    type Input = EscalateInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        escalate_input(&self.client, self.neighbours, FailingShare::Tenth)
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        launch_flags(&self.client, &args, self.neighbours);
        Ok(())
    }

    fn name(&self) -> String {
        neighbour_bench_name("mc_flag_failures_1080p_luma", self.neighbours)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(1)
    }
}

/// One escalation stage at `stage_radius`, its failure flags then its search, over a 1080p field.
///
/// Every launch covers the whole grid, so the `none` row is the floor an all-passing frame costs.
/// Subtract [FlagFailuresBench] for the search alone.
pub struct EscalateStageBench {
    pub client: Client,
    pub neighbours: u32,
    pub stage_radius: u32,
    pub share: FailingShare,
}

impl Benchmark for EscalateStageBench {
    type Input = EscalateInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        escalate_input(&self.client, self.neighbours, self.share)
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        launch_flags(&self.client, &args, self.neighbours);
        launch_escalate(&self.client, &args, self.neighbours, self.stage_radius);
        Ok(())
    }

    fn name(&self) -> String {
        let label = self.share.label();
        let base = format!("mc_escalate_stage_r{}_{label}_1080p_luma", self.stage_radius);
        neighbour_bench_name(&base, self.neighbours)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(1)
    }
}

/// The fine first pass that records escalation state, unseeded at `search_radius`.
///
/// Compare it with the `mc_block_match_fine` row at the same radius for the cost of the record.
pub struct TrackedFineBench {
    pub client: Client,
    pub neighbours: u32,
    pub search_radius: u32,
}

impl Benchmark for TrackedFineBench {
    type Input = EscalateInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        escalate_input(&self.client, self.neighbours, FailingShare::None)
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        launch_tracked(&self.client, &args, self.neighbours, self.search_radius);
        Ok(())
    }

    fn name(&self) -> String {
        let base = format!("mc_block_match_fine_tracked_r{}_1080p_luma", self.search_radius);
        neighbour_bench_name(&base, self.neighbours)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(1)
    }
}

/// An approximately Gaussian sample with unit spread, from four hashed uniforms.
fn hashed_gaussian(index: u32, seed: u32) -> f32 {
    let mut total = 0.0f32;

    for round in 0..4u32 {
        let mixed = index.wrapping_mul(4).wrapping_add(round);
        let mut hash = mixed ^ seed.wrapping_mul(0x9E37_79B9);
        hash = hash.wrapping_mul(2_654_435_761);
        hash ^= hash >> 15;
        hash = hash.wrapping_mul(0x85EB_CA6B);
        hash ^= hash >> 13;
        total += hash as f32 / u32::MAX as f32;
    }

    // Four uniforms sum to a mean of 2 and a variance of 1/3.
    (total - 2.0) * 3.0f32.sqrt()
}

/// The synthetic frame panned `shift_x` pixels, with independent grain from `seed`.
fn grainy_frame(base: &[f32], shift_x: i32, seed: u32) -> Vec<f32> {
    let mut frame = Vec::with_capacity(base.len());

    for y in 0..HEIGHT as i32 {
        for x in 0..WIDTH as i32 {
            let source_x = (x - shift_x).clamp(0, WIDTH as i32 - 1);
            let index = (y * WIDTH as i32 + x) as u32;
            let clean = base[(y * WIDTH as i32 + source_x) as usize];
            let grain = GRAIN_SIGMA * hashed_gaussian(index, seed);
            frame.push(clean + grain);
        }
    }

    frame
}

/// Grainy frames panned one pixel per neighbour, with the derived SAD floor that grain implies.
fn grain_input(client: &Client, neighbours: u32) -> EscalateInput {
    let base = make_synthetic_frame(WIDTH, HEIGHT, 1);
    let mut frames = grainy_frame(&base, 0, 0);
    for neighbour in 1..=neighbours {
        let neighbour_frame = grainy_frame(&base, neighbour as i32, neighbour);
        frames.extend_from_slice(&neighbour_frame);
    }

    let frame_bytes = f32::as_bytes(&frames);
    let table: Vec<u32> = (1..=neighbours).collect();
    let table_bytes = u32::as_bytes(&table);
    let blocks = block_count() as usize;
    let neighbours = neighbours as usize;

    let block_area = (FINE_BLKSIZE * FINE_BLKSIZE) as f32;
    let sad_floor = block_area * 2.0 * GRAIN_SIGMA / std::f32::consts::PI.sqrt();
    let mut noise_words = DerivedValues::default().to_words();
    noise_words[DERIVED_SAD_FLOOR as usize] = sad_floor;
    let noise_bytes = f32::as_bytes(&noise_words);

    EscalateInput {
        pyramid: client.create_from_slice(frame_bytes),
        neighbour_slots: client.create_from_slice(table_bytes),
        mv_field: client.empty(2 * neighbours * blocks * size_of::<i32>()),
        confidence: client.empty(neighbours * blocks * size_of::<f32>()),
        sad: client.empty(2 * neighbours * blocks * size_of::<f32>()),
        state: client.empty(STATE_SECTIONS as usize * neighbours * blocks * size_of::<f32>()),
        noise: client.create_from_slice(noise_bytes),
        placeholder: client.empty(size_of::<f32>()),
    }
}

/// The plain unseeded fine pass at `radius`, writing confidence.
fn launch_plain(client: &Client, input: &EscalateInput, neighbours: u32, radius: u32) {
    let blocks = block_count();
    let level_len = WIDTH * HEIGHT;
    let pyramid_len = ((neighbours + 1) * level_len) as usize;
    let mv_len = (2 * blocks * neighbours) as usize;
    let conf_len = (blocks * neighbours) as usize;
    let grid = CubeCount::new_3d(blocks_x(), blocks_y(), neighbours);
    let dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);

    unsafe {
        let pyramid_arg = BufferArg::from_raw_parts(input.pyramid.clone(), pyramid_len);
        let slots_arg = BufferArg::from_raw_parts(input.neighbour_slots.clone(), neighbours as usize);
        let mv_field_arg = BufferArg::from_raw_parts(input.mv_field.clone(), mv_len);
        let confidence_arg = BufferArg::from_raw_parts(input.confidence.clone(), conf_len);
        let sad_arg = BufferArg::from_raw_parts(input.placeholder.clone(), 1);
        let predictions_arg = BufferArg::from_raw_parts(input.placeholder.clone(), 1);
        let noise_arg = BufferArg::from_raw_parts(input.noise.clone(), DERIVED_WORDS as usize);

        mc_block_match_fine::launch_unchecked(
            client,
            grid,
            dim,
            pyramid_arg,
            slots_arg,
            mv_field_arg,
            confidence_arg,
            sad_arg,
            predictions_arg,
            0u32,
            true,
            false,
            false,
            noise_arg,
            DERIVED_SAD_FLOOR,
            THSAD,
            WIDTH,
            HEIGHT,
            level_len,
            FINE_BLKSIZE,
            FINE_STEP,
            radius,
            0u32,
            blocks_x(),
            2 * blocks,
            blocks,
        );
    }
}

/// The whole fine search over grainy panning frames, fixed at radius 4 or adaptive from radius 2.
///
/// The adaptive row runs the tracked first pass, then flags and a search at each stage radius, so
/// its escalation share is whatever the grain and the thresholds produce.
pub struct GrainSearchBench {
    pub client: Client,
    pub neighbours: u32,
    pub adaptive: bool,
}

impl Benchmark for GrainSearchBench {
    type Input = EscalateInput;
    type Output = ();

    fn prepare(&self) -> Self::Input {
        grain_input(&self.client, self.neighbours)
    }

    fn execute(&self, args: Self::Input) -> Result<(), String> {
        if !self.adaptive {
            launch_plain(&self.client, &args, self.neighbours, 4);
            return Ok(());
        }

        launch_tracked(&self.client, &args, self.neighbours, GRAIN_FIRST_RADIUS);
        for stage_radius in GRAIN_STAGE_RADII {
            launch_flags(&self.client, &args, self.neighbours);
            launch_escalate(&self.client, &args, self.neighbours, stage_radius);
        }

        Ok(())
    }

    fn name(&self) -> String {
        let base = if self.adaptive {
            "mc_fine_adaptive_r2_to_r4_grain_1080p_luma"
        } else {
            "mc_fine_fixed_r4_grain_1080p_luma"
        };

        neighbour_bench_name(base, self.neighbours)
    }

    fn sync(&self) {
        block_sync(&self.client);
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        shapes_with_channels(1)
    }
}

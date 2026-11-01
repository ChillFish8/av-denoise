use cubecl::prelude::*;
use cubecl::server::Handle;

use super::prediction_buffer;
use crate::engine::align::StorageAlign;
use crate::motion::analyse::{escalation_state_len, launch_fine};
use crate::motion::context::SearchSchedule;
use crate::motion::escalate::{launch_flags, run_escalation};
use crate::motion::kernels::{NEIGHBOUR_DISAGREEMENT, STATE_SECTIONS};
use crate::motion::{
    MatchTargets,
    MotionCompensationMode,
    MotionCtx,
    MotionEstimation,
    pyramid_slot_pixels,
    run_analyse,
    sad_noise_floor,
};
use crate::noise::DERIVED_SAD_FLOOR;
use crate::noise::tests::sad_floor_buffer;
use crate::tune::gpu::{read_f32, read_i32};
use crate::tune::tests::fixtures::{noisy_field_over, upload};
use crate::tune::tests::runtime::test_client;

const WIDTH: u32 = 128;
const HEIGHT: u32 = 64;
const BLKSIZE: u32 = 16;
const STEP: u32 = 8;
/// The library's `thsad` at a 16-pixel block and a scale of 1.
const THSAD: f32 = 5.12;
/// A `thsad` no block in these tests falls short of, so only the edge and neighbour tests flag.
const LENIENT_THSAD: f32 = 1.0e4;
/// A kept SAD far past `THSAD`, which scores a confidence of 0.
const POOR_SAD: f32 = 1.0e3;
/// Pixels of world around each frame, so shifted crops never clamp.
const MARGIN: i32 = 16;
/// The `thsad` and derived SAD floor one search scores with.
#[derive(Clone, Copy)]
struct Scoring {
    thsad: f32,
    sad_floor: f32,
}

const EXACT: Scoring = Scoring {
    thsad: THSAD,
    sad_floor: 0.0,
};
const LENIENT: Scoring = Scoring {
    thsad: LENIENT_THSAD,
    sad_floor: 0.0,
};

/// The value every output starts at, so a block that never writes is visible.
const UNWRITTEN: f32 = -7.0;
const SAD_SECTION: usize = 0;
const FLAG_SECTION: usize = 1;
const CENTRE_X_SECTION: usize = 2;
const CENTRE_Y_SECTION: usize = 3;

/// A hashed value between 0 and 1.
fn hashed_unit(index: u32, seed: u32) -> f32 {
    let seed_offset = seed.wrapping_mul(0x9E37_79B9);
    let mut hash = index.wrapping_add(seed_offset);
    hash = hash.wrapping_mul(2_654_435_761);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x85EB_CA6B);
    hash ^= hash >> 13;
    hash as f32 / u32::MAX as f32
}

/// A hashed texture with every value between 0.2 and 0.8.
fn hashed_world(seed: u32) -> World {
    let width = WIDTH + 2 * MARGIN as u32;
    let height = HEIGHT + 2 * MARGIN as u32;
    let values = (0..width * height)
        .map(|index| 0.2 + 0.6 * hashed_unit(index, seed))
        .collect();

    World { values, width }
}

/// A horizontal wave with a hashed phase per row.
///
/// A block's SAD falls steadily towards its true horizontal offset, and any vertical miss lands on
/// unrelated rows.
fn smooth_world() -> World {
    let width = WIDTH + 2 * MARGIN as u32;
    let height = HEIGHT + 2 * MARGIN as u32;
    let mut values = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        let phase = std::f32::consts::TAU * hashed_unit(y, 1);
        for x in 0..width {
            let angle = 0.1 * x as f32 + phase;
            values.push(0.5 + 0.3 * angle.sin());
        }
    }

    World { values, width }
}

struct World {
    values: Vec<f32>,
    width: u32,
}

impl World {
    /// The frame whose content sits `(shift_x, shift_y)` pixels further on than the centre frame's.
    fn frame(&self, shift_x: i32, shift_y: i32) -> Vec<f32> {
        let mut frame = Vec::with_capacity((WIDTH * HEIGHT) as usize);

        for y in 0..HEIGHT as i32 {
            for x in 0..WIDTH as i32 {
                let source_x = (x + MARGIN - shift_x) as u32;
                let source_y = (y + MARGIN - shift_y) as u32;
                frame.push(self.values[(source_y * self.width + source_x) as usize]);
            }
        }

        frame
    }
}

fn schedule(first_radius: u32, ceiling: u32, steps: u32) -> SearchSchedule {
    SearchSchedule {
        first_radius,
        ceiling,
        steps,
    }
}

/// A single-level geometry over the test frame with `schedule`.
fn ctx_with(client: &Client, schedule: SearchSchedule) -> MotionCtx {
    let mode = MotionCompensationMode::Mvtools {
        blksize: BLKSIZE,
        overlap: BLKSIZE - STEP,
        search_radius: schedule.ceiling,
        pyramid_levels: 1,
        estimation: MotionEstimation::Direct,
    };
    let align = StorageAlign::from_client(client);
    let ctx = MotionCtx::new(mode, WIDTH, HEIGHT, align).expect("the geometry is valid");

    MotionCtx { schedule, ..ctx }
}

/// Every block whose window and content stay inside the frame for shifts up to `reach`.
fn interior_blocks(ctx: &MotionCtx, reach: u32) -> Vec<usize> {
    let mut blocks = Vec::new();

    for row in 0..ctx.blocks_y {
        for col in 0..ctx.blocks_x {
            let right = col * STEP + BLKSIZE + reach;
            let bottom = row * STEP + BLKSIZE + reach;
            let clear = col * STEP >= reach && row * STEP >= reach;
            if clear && right <= WIDTH && bottom <= HEIGHT {
                blocks.push((row * ctx.blocks_x + col) as usize);
            }
        }
    }

    blocks
}

/// The centre frame in slot 0 and each neighbour after it, padded as the context lays slots out.
struct Frames {
    pyramid: Handle,
    count: u32,
}

fn upload_frames(client: &Client, ctx: &MotionCtx, frames: &[Vec<f32>]) -> Frames {
    let slot_pixels = pyramid_slot_pixels(WIDTH, HEIGHT, 0, ctx.align) as usize;
    let mut values = Vec::new();

    for frame in frames {
        let padding = slot_pixels - frame.len();
        values.extend_from_slice(frame);
        values.extend(std::iter::repeat_n(0.0, padding));
    }

    Frames {
        pyramid: upload(client, &values),
        count: frames.len() as u32,
    }
}

/// Every buffer one search writes, read back after it.
struct Outputs {
    vectors: Vec<i32>,
    confidence: Vec<f32>,
    /// Winning SADs, then window means, in the confidence layout.
    sad: Vec<f32>,
    state: Vec<f32>,
}

impl Outputs {
    fn vector(&self, block: usize) -> (i32, i32) {
        (self.vectors[2 * block], self.vectors[2 * block + 1])
    }

    fn state_word(&self, ctx: &MotionCtx, section: usize, block: usize) -> f32 {
        let neighbours = self.state.len() / (STATE_SECTIONS as usize * ctx.conf_stride() as usize);
        let section_len = neighbours * ctx.conf_stride() as usize;
        self.state[section * section_len + block]
    }

    fn flagged(&self, ctx: &MotionCtx, block: usize) -> bool {
        self.state_word(ctx, FLAG_SECTION, block) != 0.0
    }
}

/// The device buffers of one search, with every output at [UNWRITTEN].
struct Buffers {
    neighbour_slots: Handle,
    neighbours: u32,
    mv_field: Handle,
    mv_len: usize,
    confidence: Handle,
    conf_len: usize,
    sad: Handle,
    state: Handle,
    state_len: usize,
    noise: Handle,
}

impl Buffers {
    fn new(client: &Client, ctx: &MotionCtx, table: &[u32], vectors: Option<&[i32]>, sad_floor: f32) -> Self {
        let neighbours = table.len() as u32;
        let table_bytes = u32::as_bytes(table);
        let neighbour_slots = client.create_from_slice(table_bytes);
        let mv_len = (neighbours * ctx.mv_stride()) as usize;
        let zero_vectors = vec![0i32; mv_len];
        let initial_vectors = vectors.unwrap_or(&zero_vectors);
        let vector_bytes = i32::as_bytes(initial_vectors);
        let mv_field = client.create_from_slice(vector_bytes);
        let conf_len = (neighbours * ctx.conf_stride()) as usize;
        let unwritten = vec![UNWRITTEN; 2 * conf_len];
        let state_len = escalation_state_len(ctx, neighbours);
        let unwritten_state = vec![UNWRITTEN; state_len];

        Self {
            neighbour_slots,
            neighbours,
            mv_field,
            mv_len,
            confidence: upload(client, &unwritten[..conf_len]),
            conf_len,
            sad: upload(client, &unwritten),
            state: upload(client, &unwritten_state),
            state_len,
            noise: sad_floor_buffer(client, DERIVED_SAD_FLOOR, sad_floor),
        }
    }

    fn targets<'a>(&'a self, frames: &'a Frames, tracked: bool) -> MatchTargets<'a> {
        MatchTargets {
            pyramid: &frames.pyramid,
            frame_count: frames.count,
            centre_slot: 0,
            neighbour_slots: &self.neighbour_slots,
            neighbours: self.neighbours,
            mv_field: &self.mv_field,
            mv_len: self.mv_len,
            confidence: &self.confidence,
            conf_len: self.conf_len,
            write_confidence: true,
            predictions: None,
            sad: Some(&self.sad),
            escalation: tracked.then_some(&self.state),
        }
    }

    fn read(&self, client: &Client) -> Outputs {
        Outputs {
            vectors: read_i32(client, &self.mv_field, self.mv_len).expect("the field reads back"),
            confidence: read_f32(client, &self.confidence, self.conf_len).expect("confidence reads back"),
            sad: read_f32(client, &self.sad, 2 * self.conf_len).expect("the SADs read back"),
            state: read_f32(client, &self.state, self.state_len).expect("the state reads back"),
        }
    }
}

/// Runs the whole analyse, escalation included when `tracked`, against every slot in `table`.
fn analyse(
    client: &Client,
    ctx: &MotionCtx,
    frames: &Frames,
    table: &[u32],
    search: Scoring,
    tracked: bool,
) -> Outputs {
    let buffers = Buffers::new(client, ctx, table, None, search.sad_floor);
    let targets = buffers.targets(frames, tracked);
    run_analyse(
        client,
        ctx,
        WIDTH,
        HEIGHT,
        &targets,
        &buffers.noise,
        DERIVED_SAD_FLOOR,
        search.thsad,
    )
    .expect("the analyse runs");

    buffers.read(client)
}

/// Runs only the tracked first pass and the failure flags over one neighbour in slot 1.
fn first_pass_flags(client: &Client, ctx: &MotionCtx, frames: &Frames, search: Scoring) -> Outputs {
    let buffers = Buffers::new(client, ctx, &[1], None, search.sad_floor);
    let thsad = search.thsad;
    let targets = buffers.targets(frames, true);
    let first_radius = ctx.schedule.first_radius;
    launch_fine(
        client,
        ctx,
        WIDTH,
        HEIGHT,
        &targets,
        first_radius,
        0,
        &buffers.noise,
        DERIVED_SAD_FLOOR,
        thsad,
    );
    launch_flags(
        client,
        ctx,
        &targets,
        &buffers.state,
        &buffers.noise,
        DERIVED_SAD_FLOOR,
        thsad,
    );

    buffers.read(client)
}

/// Runs the escalation over a first pass that left `vectors` with `kept_sads` and no edge winners.
fn escalate_planted(
    client: &Client,
    ctx: &MotionCtx,
    frames: &Frames,
    vectors: &[i32],
    kept_sads: &[f32],
) -> Outputs {
    let buffers = Buffers::new(client, ctx, &[1], Some(vectors), 0.0);
    let section_len = ctx.conf_stride() as usize;
    let mut state = vec![0.0f32; buffers.state_len];
    for (block, kept_sad) in kept_sads.iter().enumerate() {
        state[SAD_SECTION * section_len + block] = *kept_sad;
        state[CENTRE_X_SECTION * section_len + block] = vectors[2 * block] as f32;
        state[CENTRE_Y_SECTION * section_len + block] = vectors[2 * block + 1] as f32;
    }

    let state_bytes = f32::as_bytes(&state);
    let state_handle = client.create_from_slice(state_bytes);
    let planted = Buffers {
        state: state_handle,
        ..buffers
    };
    let targets = planted.targets(frames, true);
    run_escalation(
        client,
        ctx,
        WIDTH,
        HEIGHT,
        &targets,
        &planted.noise,
        DERIVED_SAD_FLOOR,
        THSAD,
    )
    .expect("the escalation runs");

    planted.read(client)
}

fn block_count(ctx: &MotionCtx) -> usize {
    (ctx.blocks_x * ctx.blocks_y) as usize
}

/// The block in the middle of the grid, well clear of every frame edge.
fn middle_block(ctx: &MotionCtx) -> usize {
    let row = ctx.blocks_y / 2;
    let col = ctx.blocks_x / 2;
    (row * ctx.blocks_x + col) as usize
}

fn uniform_vectors(ctx: &MotionCtx, vector: (i32, i32)) -> Vec<i32> {
    let mv_len = ctx.mv_stride() as usize;
    (0..mv_len)
        .map(|index| if index % 2 == 0 { vector.0 } else { vector.1 })
        .collect()
}

fn assert_bits_eq(left: &[f32], right: &[f32], what: &str) {
    let left_bits: Vec<u32> = left.iter().map(|value| value.to_bits()).collect();
    let right_bits: Vec<u32> = right.iter().map(|value| value.to_bits()).collect();
    assert_eq!(left_bits, right_bits, "{what}");
}

#[test]
fn in_window_motion_needs_no_escalation() {
    let client = test_client();
    let ctx = ctx_with(&client, schedule(2, 6, 2));
    let world = hashed_world(3);
    let centre = world.frame(0, 0);
    let neighbour = world.frame(1, 1);
    let frames = upload_frames(&client, &ctx, &[centre, neighbour]);

    let flags = first_pass_flags(&client, &ctx, &frames, EXACT);
    let escalated = analyse(&client, &ctx, &frames, &[1], EXACT, true);
    let fixed_ctx = MotionCtx {
        schedule: SearchSchedule::fixed(2),
        ..ctx.clone()
    };
    let first_only = analyse(&client, &fixed_ctx, &frames, &[1], EXACT, false);

    for block in interior_blocks(&ctx, 1) {
        assert!(!flags.flagged(&ctx, block), "block {block} should pass");
        assert_eq!(escalated.vector(block), (1, 1), "block {block}");
        assert_eq!(escalated.vector(block), first_only.vector(block), "block {block}");
        assert_eq!(escalated.sad[block].to_bits(), first_only.sad[block].to_bits());
        assert_eq!(
            escalated.confidence[block].to_bits(),
            first_only.confidence[block].to_bits()
        );
    }
}

#[test]
fn edge_minimum_escalates_and_finds_the_true_motion() {
    let client = test_client();
    let ctx = ctx_with(&client, schedule(2, 4, 1));
    let world = smooth_world();
    let centre = world.frame(0, 0);
    let neighbour = world.frame(3, 0);
    let frames = upload_frames(&client, &ctx, &[centre, neighbour]);

    let flags = first_pass_flags(&client, &ctx, &frames, LENIENT);
    let escalated = analyse(&client, &ctx, &frames, &[1], LENIENT, true);

    for block in interior_blocks(&ctx, 4) {
        assert_eq!(
            flags.vector(block),
            (2, 0),
            "premise: block {block} stops at the edge"
        );
        assert!(
            flags.flagged(&ctx, block),
            "block {block} should fail on the edge"
        );
        assert_eq!(escalated.vector(block), (3, 0), "block {block}");
    }
}

#[test]
fn low_confidence_blocks_escalate() {
    let client = test_client();
    let ctx = ctx_with(&client, schedule(2, 4, 1));
    let world = hashed_world(5);
    let still = world.frame(0, 0);
    let frames = upload_frames(&client, &ctx, &[still.clone(), still]);
    let poor = middle_block(&ctx);
    let mut vectors = uniform_vectors(&ctx, (0, 0));
    vectors[2 * poor] = 1;
    let mut kept_sads = vec![0.0f32; block_count(&ctx)];
    kept_sads[poor] = POOR_SAD;

    let escalated = escalate_planted(&client, &ctx, &frames, &vectors, &kept_sads);

    assert_eq!(escalated.vector(poor), (0, 0));
    assert_eq!(escalated.sad[poor], 0.0);
    assert_eq!(escalated.confidence[poor], 1.0);
    for block in (0..block_count(&ctx)).filter(|block| *block != poor) {
        assert_eq!(
            escalated.confidence[block], UNWRITTEN,
            "block {block} should not search again"
        );
    }
}

#[test]
fn neighbour_disagreement_escalates() {
    let client = test_client();
    let ctx = ctx_with(&client, schedule(2, 8, 1));
    let world = hashed_world(7);
    let still = world.frame(0, 0);
    let frames = upload_frames(&client, &ctx, &[still.clone(), still]);
    let outlier = middle_block(&ctx);
    // Confident enough that only the disagreement can fail the block.
    let mut kept_sads = vec![0.0f32; block_count(&ctx)];
    kept_sads[outlier] = 0.5;

    let mut far = uniform_vectors(&ctx, (0, 0));
    far[2 * outlier] = NEIGHBOUR_DISAGREEMENT + 2;
    let mut near = uniform_vectors(&ctx, (0, 0));
    near[2 * outlier] = NEIGHBOUR_DISAGREEMENT;

    let far_result = escalate_planted(&client, &ctx, &frames, &far, &kept_sads);
    let near_result = escalate_planted(&client, &ctx, &frames, &near, &kept_sads);

    assert_eq!(far_result.vector(outlier), (0, 0));
    assert_eq!(near_result.vector(outlier), (NEIGHBOUR_DISAGREEMENT, 0));
    assert_eq!(near_result.confidence[outlier], UNWRITTEN);
    for block in (0..block_count(&ctx)).filter(|block| *block != outlier) {
        assert_eq!(
            far_result.confidence[block], UNWRITTEN,
            "block {block} should not search again"
        );
    }
}

#[test]
fn escalation_reaches_the_ceiling_and_no_further() {
    let client = test_client();
    let ctx = ctx_with(&client, schedule(2, 4, 2));
    let world = smooth_world();
    let centre = world.frame(0, 0);
    let within = world.frame(4, 0);
    let beyond = world.frame(5, 0);
    let frames = upload_frames(&client, &ctx, &[centre, within, beyond]);

    let escalated = analyse(&client, &ctx, &frames, &[1, 2], LENIENT, true);
    let conf_stride = ctx.conf_stride() as usize;
    let mv_stride = ctx.mv_stride() as usize;

    // Every window centres on the zero seed, so the ceiling of 4 bounds the total reach.
    for block in interior_blocks(&ctx, 7) {
        let beyond_slot = mv_stride + 2 * block;
        let beyond_vector = (escalated.vectors[beyond_slot], escalated.vectors[beyond_slot + 1]);

        assert_eq!(
            escalated.vector(block),
            (4, 0),
            "block {block} within the ceiling"
        );
        assert_eq!(beyond_vector, (4, 0), "block {block} beyond the ceiling");
        assert_eq!(escalated.sad[block], 0.0, "block {block} within the ceiling");
        assert!(
            escalated.sad[conf_stride + block] > 0.0,
            "block {block} beyond the ceiling"
        );
    }
}

#[test]
fn output_is_independent_of_neighbour_order_and_repeats() {
    let client = test_client();
    let ctx = ctx_with(&client, schedule(2, 6, 2));
    let world = smooth_world();
    let centre = world.frame(0, 0);
    let right = world.frame(5, 1);
    let up = world.frame(-1, -4);
    let frames = upload_frames(&client, &ctx, &[centre, right, up]);

    let forward = analyse(&client, &ctx, &frames, &[1, 2], EXACT, true);
    let repeat = analyse(&client, &ctx, &frames, &[1, 2], EXACT, true);
    let reversed = analyse(&client, &ctx, &frames, &[2, 1], EXACT, true);

    assert_eq!(forward.vectors, repeat.vectors);
    assert_bits_eq(&forward.confidence, &repeat.confidence, "confidence run to run");
    assert_bits_eq(&forward.sad, &repeat.sad, "SADs run to run");

    let mv_stride = ctx.mv_stride() as usize;
    let conf_stride = ctx.conf_stride() as usize;
    let blocks = block_count(&ctx);
    for (forward_index, reversed_index) in [(0usize, 1usize), (1, 0)] {
        let forward_mv = forward_index * mv_stride;
        let reversed_mv = reversed_index * mv_stride;
        let forward_conf = forward_index * conf_stride;
        let reversed_conf = reversed_index * conf_stride;
        let means = 2 * conf_stride;

        assert_eq!(
            forward.vectors[forward_mv..forward_mv + 2 * blocks],
            reversed.vectors[reversed_mv..reversed_mv + 2 * blocks],
        );
        assert_bits_eq(
            &forward.confidence[forward_conf..forward_conf + blocks],
            &reversed.confidence[reversed_conf..reversed_conf + blocks],
            "confidence",
        );
        assert_bits_eq(
            &forward.sad[forward_conf..forward_conf + blocks],
            &reversed.sad[reversed_conf..reversed_conf + blocks],
            "SADs",
        );
        assert_bits_eq(
            &forward.sad[means + forward_conf..means + forward_conf + blocks],
            &reversed.sad[means + reversed_conf..means + reversed_conf + blocks],
            "window means",
        );
    }
}

#[test]
fn no_failing_blocks_and_all_failing_blocks_both_work() {
    let client = test_client();
    let ctx = ctx_with(&client, schedule(2, 4, 2));
    let world = hashed_world(9);
    let still = world.frame(0, 0);
    let frames = upload_frames(&client, &ctx, &[still.clone(), still]);
    let blocks = block_count(&ctx);

    let flags = first_pass_flags(&client, &ctx, &frames, EXACT);
    let none_fail = analyse(&client, &ctx, &frames, &[1], EXACT, true);
    let fixed_ctx = MotionCtx {
        schedule: SearchSchedule::fixed(2),
        ..ctx.clone()
    };
    let first_only = analyse(&client, &fixed_ctx, &frames, &[1], EXACT, false);

    assert!((0..blocks).all(|block| !flags.flagged(&ctx, block)));
    assert_eq!(none_fail.vectors, first_only.vectors);
    assert_bits_eq(&none_fail.confidence, &first_only.confidence, "confidence");
    assert_bits_eq(&none_fail.sad, &first_only.sad, "SADs");

    let vectors = uniform_vectors(&ctx, (1, 0));
    let kept_sads = vec![POOR_SAD; blocks];
    let all_fail = escalate_planted(&client, &ctx, &frames, &vectors, &kept_sads);

    for block in 0..blocks {
        assert_eq!(all_fail.vector(block), (0, 0), "block {block}");
        assert_eq!(all_fail.sad[block], 0.0, "block {block}");
        assert_eq!(all_fail.confidence[block], 1.0, "block {block}");
    }
}

#[test]
fn first_radius_equal_to_ceiling_is_bit_identical() {
    let client = test_client();
    let world = hashed_world(11);
    let centre = world.frame(0, 0);
    let first = world.frame(3, 2);
    let second = world.frame(-2, 5);
    let fixed_ctx = ctx_with(&client, SearchSchedule::fixed(4));
    let frames = upload_frames(&client, &fixed_ctx, &[centre, first, second]);
    let equal_ctx = ctx_with(&client, schedule(4, 4, 3));

    let plain = analyse(&client, &fixed_ctx, &frames, &[1, 2], EXACT, false);
    let tracked = analyse(&client, &equal_ctx, &frames, &[1, 2], EXACT, true);

    assert_eq!(plain.vectors, tracked.vectors);
    assert_bits_eq(&plain.confidence, &tracked.confidence, "confidence");
    assert_bits_eq(&plain.sad, &tracked.sad, "SADs");
    assert!(
        tracked.state.iter().all(|value| *value == UNWRITTEN),
        "nothing records state"
    );
}

#[test]
fn tracked_first_pass_matches_the_plain_fine_pass() {
    let client = test_client();
    let ctx = ctx_with(&client, schedule(3, 6, 1));
    let world = hashed_world(13);
    let centre = world.frame(0, 0);
    let neighbour = world.frame(2, -3);
    let frames = upload_frames(&client, &ctx, &[centre, neighbour]);
    let pan = [1.0f32, 0.0, 0.0, 1.0, 12.0, 0.0];
    let (predictions, _, _) = prediction_buffer(&client, Some(&pan));

    let mut outputs = Vec::new();
    for tracked in [false, true] {
        let buffers = Buffers::new(&client, &ctx, &[1], None, 0.0);
        let targets = MatchTargets {
            predictions: Some(&predictions),
            ..buffers.targets(&frames, tracked)
        };
        launch_fine(
            &client,
            &ctx,
            WIDTH,
            HEIGHT,
            &targets,
            3,
            0,
            &buffers.noise,
            DERIVED_SAD_FLOOR,
            THSAD,
        );
        outputs.push(buffers.read(&client));
    }

    let plain = &outputs[0];
    let tracked = &outputs[1];
    assert_eq!(plain.vectors, tracked.vectors);
    assert_bits_eq(&plain.confidence, &tracked.confidence, "confidence");
    assert_bits_eq(&plain.sad, &tracked.sad, "SADs");
    for block in 0..block_count(&ctx) {
        let (dx, dy) = tracked.vector(block);
        let kept_sad = tracked.state_word(&ctx, SAD_SECTION, block);
        let centre_x = tracked.state_word(&ctx, CENTRE_X_SECTION, block);
        let centre_y = tracked.state_word(&ctx, CENTRE_Y_SECTION, block);
        let from_seed = (centre_x, centre_y) == (0.0, 0.0);
        let from_model = (centre_x, centre_y) == (12.0, 0.0);
        let reach_x = (dx as f32 - centre_x).abs();
        let reach_y = (dy as f32 - centre_y).abs();

        assert_eq!(kept_sad.to_bits(), tracked.sad[block].to_bits(), "block {block}");
        assert!(
            from_seed || from_model,
            "block {block} centre ({centre_x}, {centre_y})"
        );
        assert!(
            reach_x <= 3.0 && reach_y <= 3.0,
            "block {block} sits inside its window"
        );
    }
}

#[test]
fn flat_grain_rarely_escalates_and_keeps_near_vectors() {
    let client = test_client();
    let ctx = ctx_with(&client, schedule(2, 6, 2));
    let sigma = 4.0 / 255.0;
    let flat = vec![0.5f32; (WIDTH * HEIGHT) as usize];
    let centre = noisy_field_over(&flat, WIDTH, HEIGHT, sigma, 21);
    let neighbour = noisy_field_over(&flat, WIDTH, HEIGHT, sigma, 22);
    let frames = upload_frames(&client, &ctx, &[centre, neighbour]);
    let grain = Scoring {
        thsad: THSAD,
        sad_floor: sad_noise_floor(BLKSIZE, sigma),
    };
    let blocks = block_count(&ctx);

    let flags = first_pass_flags(&client, &ctx, &frames, grain);
    let escalated = analyse(&client, &ctx, &frames, &[1], grain, true);

    let flagged = (0..blocks).filter(|block| flags.flagged(&ctx, *block)).count();
    assert!(
        flagged * 10 <= blocks,
        "{flagged} of {blocks} blocks flagged on grain"
    );

    let first_radius = ctx.schedule.first_radius as i32;
    for block in 0..blocks {
        let (dx, dy) = escalated.vector(block);

        assert!(
            dx.abs() <= first_radius && dy.abs() <= first_radius,
            "block {block} moved to ({dx}, {dy})"
        );
        assert_eq!(escalated.vector(block), flags.vector(block), "block {block}");
    }
}

#[test]
fn a_flagged_block_that_finds_nothing_better_keeps_its_first_pass() {
    let client = test_client();
    let ctx = ctx_with(&client, schedule(2, 8, 1));
    let world = hashed_world(15);
    let still = world.frame(0, 0);
    let frames = upload_frames(&client, &ctx, &[still.clone(), still]);
    let outlier = middle_block(&ctx);
    let mut vectors = uniform_vectors(&ctx, (0, 0));
    vectors[2 * outlier] = NEIGHBOUR_DISAGREEMENT + 2;
    // No offset scores below 0, so the wider search cannot win.
    let kept_sads = vec![0.0f32; block_count(&ctx)];

    let escalated = escalate_planted(&client, &ctx, &frames, &vectors, &kept_sads);

    assert_eq!(escalated.vector(outlier), (NEIGHBOUR_DISAGREEMENT + 2, 0));
    assert_eq!(escalated.sad[outlier], UNWRITTEN);
    assert_eq!(escalated.confidence[outlier], UNWRITTEN);
    assert_eq!(escalated.state_word(&ctx, SAD_SECTION, outlier), 0.0);
    assert_eq!(escalated.state_word(&ctx, FLAG_SECTION, outlier), 0.0);
}

#[test]
fn an_escalating_schedule_rejects_a_zero_thsad() {
    let client = test_client();
    let ctx = ctx_with(&client, schedule(2, 4, 1));
    let world = hashed_world(17);
    let still = world.frame(0, 0);
    let frames = upload_frames(&client, &ctx, &[still.clone(), still]);
    let buffers = Buffers::new(&client, &ctx, &[1], None, 0.0);
    let targets = buffers.targets(&frames, true);

    let result = run_escalation(
        &client,
        &ctx,
        WIDTH,
        HEIGHT,
        &targets,
        &buffers.noise,
        DERIVED_SAD_FLOOR,
        0.0,
    );

    assert!(result.is_err());
}

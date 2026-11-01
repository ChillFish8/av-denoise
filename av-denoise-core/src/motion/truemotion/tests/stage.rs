use std::collections::HashSet;

use cubecl::prelude::*;

use super::harness::{
    BLKSIZE,
    STEP,
    SearchShape,
    StageInput,
    StageOutput,
    add_noise,
    noise_floor,
    run_search,
    run_stage,
    shifted,
    textured,
    thsad,
    translation,
    uniform_field,
};
use super::make_client;
use crate::motion::context::SearchSchedule;
use crate::motion::truemotion::TruemotionParams;
use crate::motion::truemotion::kernels::tests::reference::nl4d_mv_regularise_reference;
use crate::motion::truemotion::kernels::{
    LIST_CAPACITY,
    META_WORDS,
    PassBlock,
    STAGED_WORDS,
    gather_candidates,
    neighbourhood_median,
    stage_neighbourhood,
};
use crate::noise::tests::sad_floor_buffer;
use crate::noise::{DERIVED_SAD_FLOOR, DERIVED_WORDS};
use crate::tune::gpu::{bind, read_f32, read_i32};
use crate::tune::tests::fixtures::upload;

fn blocks_along(pixels: u32) -> u32 {
    pixels.div_ceil(STEP)
}

/// A field whose vectors vary block to block between about -3 and 3.
fn patterned_field(blocks: usize) -> Vec<i32> {
    (0..2 * blocks)
        .map(|index| (index % 7) as i32 - 3 + ((index / 14) % 3) as i32)
        .collect()
}

/// Runs the verbatim old regulariser kernel over one neighbour and returns its field and confidence.
fn reference_pass(
    width: u32,
    height: u32,
    centre: &[f32],
    neighbour: &[f32],
    field: &[i32],
    lambda: f32,
    sad_floor: f32,
) -> StageOutput {
    let client = make_client();
    let blocks_x = blocks_along(width);
    let blocks_y = blocks_along(height);
    let blocks = (blocks_x * blocks_y) as usize;
    let pixels = centre.len();
    let centre_buf = upload(&client, centre);
    let neighbour_buf = upload(&client, neighbour);
    let field_bytes = i32::as_bytes(field);
    let mv_in = client.create_from_slice(field_bytes);
    let mv_out = client.empty(2 * blocks * size_of::<i32>());
    let confidence = client.empty(blocks * size_of::<f32>());
    let noise = sad_floor_buffer(&client, DERIVED_SAD_FLOOR, sad_floor);
    let lambda_pixel = lambda * thsad();
    let cube_count = CubeCount::new_2d(blocks_x, blocks_y);
    let cube_dim = CubeDim::new_2d(8, 8);

    unsafe {
        nl4d_mv_regularise_reference::launch_unchecked(
            &client,
            cube_count,
            cube_dim,
            bind(&centre_buf, pixels),
            bind(&neighbour_buf, pixels),
            bind(&mv_in, 2 * blocks),
            bind(&mv_out, 2 * blocks),
            bind(&confidence, blocks),
            lambda_pixel,
            bind(&noise, DERIVED_WORDS as usize),
            thsad(),
            width,
            height,
            BLKSIZE,
            STEP,
            blocks_x,
            blocks_y,
        );
    }

    let field = read_i32(&client, &mv_out, 2 * blocks).expect("the field reads back");
    let confidence = read_f32(&client, &confidence, blocks).expect("the confidence reads back");

    StageOutput { field, confidence }
}

#[test]
fn with_noise_scaling_off_it_reproduces_the_old_regulariser_choices() {
    let client = make_client();
    let (width, height) = (128u32, 64u32);
    let blocks = (blocks_along(width) * blocks_along(height)) as usize;
    let centre = textured(width, height, 21);
    let mut neighbour = shifted(&centre, width, height, 2, 1);
    add_noise(&mut neighbour, 0.01, 22);
    let flat = vec![0.5f32; (width * height) as usize];
    let field = patterned_field(blocks);
    let sad_floor = 0.4 * thsad();

    for (content, centre_frame, neighbour_frame) in
        [("textured", &centre, &neighbour), ("flat", &flat, &flat)]
    {
        for lambda in [0.0f32, 1.0, 4.0] {
            let label = format!("{content} at lambda {lambda}");
            let input = StageInput {
                width,
                height,
                centre: centre_frame,
                neighbour: neighbour_frame,
                field: &field,
                seeds: None,
                prediction: None,
                sad_floor,
                reads_floor: true,
            };
            let params = TruemotionParams::legacy(lambda);

            let got = run_stage(&client, &input, params);
            let expected = reference_pass(
                width,
                height,
                centre_frame,
                neighbour_frame,
                &field,
                lambda,
                sad_floor,
            );

            // The stage sums each row before adding the rows, so confidence differs by rounding.
            let difference = got
                .confidence
                .iter()
                .zip(&expected.confidence)
                .map(|(got_score, expected_score)| (got_score - expected_score).abs())
                .fold(0.0f32, f32::max);

            assert_eq!(got.field, expected.field, "{label}: vectors");
            assert!(
                difference <= 1.0e-5,
                "{label}: confidence differs by {difference:e}"
            );
        }
    }
}

/// The SAD of the block at `origin` in `centre` against `neighbour` moved by `vector`.
fn block_sad(
    centre: &[f32],
    neighbour: &[f32],
    width: u32,
    height: u32,
    origin: [u32; 2],
    vector: [i32; 2],
) -> f32 {
    let mut sad = 0.0f32;
    for y in origin[1]..origin[1] + BLKSIZE {
        for x in origin[0]..origin[0] + BLKSIZE {
            let moved_x = (x as i32 + vector[0]).clamp(0, width as i32 - 1) as u32;
            let moved_y = (y as i32 + vector[1]).clamp(0, height as i32 - 1) as u32;
            let centre_value = centre[(y * width + x) as usize];
            let neighbour_value = neighbour[(moved_y * width + moved_x) as usize];
            sad += (centre_value - neighbour_value).abs();
        }
    }

    sad
}

/// A frame of `seed` texture with a flat square `0.5` over `flat` in both axes.
fn texture_with_flat_square(width: u32, height: u32, seed: u32, flat: std::ops::Range<u32>) -> Vec<f32> {
    let mut frame = textured(width, height, seed);
    for y in flat.clone() {
        for x in flat.clone() {
            frame[(y * width + x) as usize] = 0.5;
        }
    }

    frame
}

#[test]
fn the_penalty_prefers_the_consensus_on_a_flat_noisy_block() {
    let client = make_client();
    let (width, height) = (128u32, 128u32);
    let blocks_x = blocks_along(width);
    let blocks = (blocks_x * blocks_along(height)) as usize;
    let sigma = 4.0 / 255.0;
    let clean = texture_with_flat_square(width, height, 31, 32..96);
    let mut centre = clean.clone();
    add_noise(&mut centre, sigma, 32);
    let mut neighbour = shifted(&clean, width, height, 2, 0);
    add_noise(&mut neighbour, sigma, 33);

    // Grain makes some far vector beat the true one on the flat block, as a plain search finds.
    let origin = [56u32, 56u32];
    let truth = [2, 0];
    let mut grain_best = truth;
    let mut grain_best_sad = f32::INFINITY;
    for dy in -6..=6 {
        for dx in -6..=6 {
            let vector = [dx, dy];
            if vector == truth {
                continue;
            }

            let sad = block_sad(&centre, &neighbour, width, height, origin, vector);
            if sad < grain_best_sad {
                grain_best_sad = sad;
                grain_best = vector;
            }
        }
    }

    let truth_sad = block_sad(&centre, &neighbour, width, height, origin, truth);
    assert!(grain_best_sad < truth_sad, "premise: grain beats the true vector");

    let block = ((origin[1] / STEP) * blocks_x + origin[0] / STEP) as usize;
    let mut field = uniform_field(blocks, truth);
    field[2 * block] = grain_best[0];
    field[2 * block + 1] = grain_best[1];
    let input = StageInput {
        width,
        height,
        centre: &centre,
        neighbour: &neighbour,
        field: &field,
        seeds: None,
        prediction: None,
        sad_floor: noise_floor(sigma),
        reads_floor: true,
    };

    let rescored = run_stage(&client, &input, TruemotionParams::legacy(0.0));
    let penalised = run_stage(&client, &input, TruemotionParams::default());

    assert_eq!(
        rescored.vector(block),
        grain_best,
        "premise: without a penalty grain wins"
    );
    assert_eq!(penalised.vector(block), truth);
}

#[test]
fn the_penalty_keeps_the_true_motion_on_a_textured_block() {
    let client = make_client();
    let (width, height) = (128u32, 128u32);
    let blocks_x = blocks_along(width);
    let blocks = (blocks_x * blocks_along(height)) as usize;
    let sigma = 1.0 / 255.0;
    let clean = textured(width, height, 41);
    let object_texture = textured(width, height, 44);
    let background = [2, 0];
    let object = [-4, 3];
    let mut centre = clean.clone();
    let mut neighbour = shifted(&clean, width, height, background[0], background[1]);

    // One block-sized object of its own texture moves on its own.
    let origin = [56u32, 56u32];
    for y in origin[1]..origin[1] + BLKSIZE {
        for x in origin[0]..origin[0] + BLKSIZE {
            let moved_x = (x as i32 + object[0]) as u32;
            let moved_y = (y as i32 + object[1]) as u32;
            let value = object_texture[(y * width + x) as usize];
            centre[(y * width + x) as usize] = value;
            neighbour[(moved_y * width + moved_x) as usize] = value;
        }
    }

    add_noise(&mut centre, sigma, 42);
    add_noise(&mut neighbour, sigma, 43);

    let block = ((origin[1] / STEP) * blocks_x + origin[0] / STEP) as usize;
    let mut field = uniform_field(blocks, background);
    field[2 * block] = object[0];
    field[2 * block + 1] = object[1];
    let input = StageInput {
        width,
        height,
        centre: &centre,
        neighbour: &neighbour,
        field: &field,
        seeds: None,
        prediction: None,
        sad_floor: noise_floor(sigma),
        reads_floor: true,
    };
    let strong = TruemotionParams {
        field_lambda: 4.0,
        ..TruemotionParams::default()
    };
    let strong_without_fade = TruemotionParams {
        fade_scale: 0.0,
        ..strong
    };

    let shipped = run_stage(&client, &input, TruemotionParams::default());
    let faded = run_stage(&client, &input, strong);
    let unfaded = run_stage(&client, &input, strong_without_fade);

    assert_eq!(shipped.vector(block), object);
    assert_eq!(
        faded.vector(block),
        object,
        "the fade keeps it under a strong penalty"
    );
    assert_ne!(
        unfaded.vector(block),
        object,
        "premise: without the fade the strong penalty pulls it to the consensus"
    );
}

/// The blocks within `reach` steps of `centre` along the grid, counting both axes.
fn diamond(centre: [i32; 2], reach: i32, blocks_x: u32, blocks_y: u32) -> HashSet<usize> {
    let mut blocks = HashSet::new();
    for row in 0..blocks_y as i32 {
        for col in 0..blocks_x as i32 {
            let distance = (col - centre[0]).abs() + (row - centre[1]).abs();
            if distance <= reach {
                blocks.insert((row as u32 * blocks_x + col as u32) as usize);
            }
        }
    }

    blocks
}

#[test]
fn propagation_spreads_a_planted_correct_vector() {
    let client = make_client();
    let (width, height) = (192u32, 128u32);
    let blocks_x = blocks_along(width);
    let blocks_y = blocks_along(height);
    let blocks = (blocks_x * blocks_y) as usize;
    let truth = [5, -3];
    let centre = textured(width, height, 51);
    let neighbour = shifted(&centre, width, height, truth[0], truth[1]);

    let planted = [12i32, 8i32];
    let planted_block = (planted[1] as u32 * blocks_x + planted[0] as u32) as usize;
    let mut field = uniform_field(blocks, [0, 0]);
    field[2 * planted_block] = truth[0];
    field[2 * planted_block + 1] = truth[1];
    let input = StageInput {
        width,
        height,
        centre: &centre,
        neighbour: &neighbour,
        field: &field,
        seeds: None,
        prediction: None,
        sad_floor: 0.0,
        reads_floor: false,
    };

    for passes in [1u32, 2] {
        let params = TruemotionParams {
            passes,
            ..TruemotionParams::default()
        };
        let output = run_stage(&client, &input, params);
        let found: HashSet<usize> = (0..blocks)
            .filter(|&block| output.vector(block) == truth)
            .collect();
        let expected = diamond(planted, passes as i32, blocks_x, blocks_y);

        assert_eq!(found, expected, "after {passes} passes");
    }
}

/// Gathers the candidate list of block `(col, row)` and writes its length, the median's entry and
/// the list to `out`.
#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "every argument is a buffer, position or comptime shape the probe binds"
)]
fn gather_probe(
    mv_in: &[i32],
    seeds: &[i32],
    predictions: &[f32],
    out: &mut [i32],
    col: u32,
    row: u32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blocks_x: u32,
    #[comptime] blocks_y: u32,
) {
    let mut staged = Shared::<[i32]>::new_slice(STAGED_WORDS as usize);
    let mut list = Shared::<[i32]>::new_slice(comptime!(2 * LIST_CAPACITY) as usize);
    let mut meta = Shared::<[i32]>::new_slice(META_WORDS as usize);
    let block = PassBlock {
        block_col: col,
        block_row: row,
        neighbour_index: 0u32,
        neighbour_base: 0u32,
        origin_x: (col * STEP) as i32,
        origin_y: (row * STEP) as i32,
        thread_id: 0u32,
        threads: 1u32,
    };

    stage_neighbourhood(mv_in, seeds, 0u32, &mut staged, &block, true, blocks_x, blocks_y);
    neighbourhood_median(&staged, &mut meta, &block, blocks_x, blocks_y);
    gather_candidates(
        &staged,
        predictions,
        &mut list,
        &mut meta,
        &block,
        true,
        true,
        width,
        height,
        BLKSIZE,
        blocks_x,
        blocks_y,
    );

    let count = meta[0] as u32;
    out[0] = meta[0];
    out[1] = meta[1];
    let mut index = 0u32;
    while index < 2u32 * count {
        out[(2u32 + index) as usize] = list[index as usize];
        index += 1u32;
    }
}

/// The candidate list the gather should build for block `(col, row)`, in order, without repeats.
fn expected_candidates(
    field: &[i32],
    seeds: &[i32],
    model: [i32; 2],
    col: i32,
    row: i32,
    blocks_x: i32,
) -> Vec<[i32; 2]> {
    let vector = |values: &[i32], col: i32, row: i32| {
        let index = (2 * (row * blocks_x + col)) as usize;
        [values[index], values[index + 1]]
    };
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx != 0 || dy != 0 {
                let [x, y] = vector(field, col + dx, row + dy);
                xs.push(x);
                ys.push(y);
            }
        }
    }

    xs.sort();
    ys.sort();
    let median = [xs[3], ys[3]];
    let raw = [
        vector(field, col, row),
        median,
        vector(field, col - 1, row),
        vector(field, col + 1, row),
        vector(field, col, row - 1),
        vector(field, col, row + 1),
        [0, 0],
        model,
        vector(seeds, col, row),
        vector(seeds, col - 1, row - 1),
        vector(seeds, col + 1, row - 1),
        vector(seeds, col - 1, row + 1),
        vector(seeds, col + 1, row + 1),
    ];

    let mut unique = Vec::new();
    for candidate in raw {
        if !unique.contains(&candidate) {
            unique.push(candidate);
        }
    }

    unique
}

/// The gathered list of block (1, 1) of a 3x3 grid.
fn gathered(field: &[i32], seeds: &[i32], prediction: [f32; 6]) -> Vec<[i32; 2]> {
    let client = make_client();
    let (width, height) = (24u32, 24u32);
    let field_buf = client.create_from_slice(i32::as_bytes(field));
    let seeds_buf = client.create_from_slice(i32::as_bytes(seeds));
    let predictions = upload(&client, &prediction);
    let out_len = 2 + 2 * LIST_CAPACITY as usize;
    let out = client.empty(out_len * size_of::<i32>());

    unsafe {
        gather_probe::launch_unchecked(
            &client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(1),
            bind(&field_buf, field.len()),
            bind(&seeds_buf, seeds.len()),
            bind(&predictions, 6),
            bind(&out, out_len),
            1u32,
            1u32,
            width,
            height,
            3u32,
            3u32,
        );
    }

    let words = read_i32(&client, &out, out_len).expect("the list reads back");
    let count = words[0] as usize;
    let median_entry = words[1] as usize;
    let list: Vec<[i32; 2]> = (0..count)
        .map(|entry| [words[2 + 2 * entry], words[3 + 2 * entry]])
        .collect();
    assert!(median_entry < count, "the median's entry lies in the list");

    list
}

#[test]
fn duplicate_candidates_are_evaluated_once() {
    let zeroes = vec![0i32; 18];
    let identity = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    let still = gathered(&zeroes, &zeroes, identity);

    assert_eq!(still, vec![[0, 0]], "a still block scores one vector");

    // Every source repeats. The right block repeats the left, the lower block repeats the block's
    // own vector, and the seeds repeat field vectors.
    #[rustfmt::skip]
    let field = vec![
        3, 1,   1, 0,   3, 1,
        2, 0,   1, 0,   2, 0,
        4, -1,  1, 0,   4, -1,
    ];
    #[rustfmt::skip]
    let seeds = vec![
        2, 0,   9, 9,   1, 0,
        9, 9,   5, 5,   9, 9,
        3, 1,   9, 9,   -2, 2,
    ];
    let prediction = translation(2.0, 0.0);

    let got = gathered(&field, &seeds, prediction);
    let expected = expected_candidates(&field, &seeds, [2, 0], 1, 1, 3);

    assert_eq!(got, expected);
    assert_eq!(got.len(), 6, "13 candidates hold 6 distinct vectors");
}

/// A pan of `pan` px with an independently moving textured layer over its middle.
///
/// The layer moves by `layer` px and has a flat horizontal stripe one block tall at `y = 72`.
struct LayeredPan {
    width: u32,
    height: u32,
    centre: Vec<f32>,
    neighbour: Vec<f32>,
    sigma: f32,
}

const PAN: [i32; 2] = [30, 0];
const LAYER: [i32; 2] = [-6, 1];
const LAYER_X: std::ops::Range<u32> = 96..224;
const LAYER_Y: std::ops::Range<u32> = 32..128;
const STRIPE_X: std::ops::Range<u32> = 104..216;
const STRIPE_Y: std::ops::Range<u32> = 72..88;

fn layered_pan() -> LayeredPan {
    let (width, height) = (320u32, 160u32);
    let sigma = 2.0 / 255.0;
    let background = textured(width, height, 61);
    let layer = textured(width, height, 62);
    let mut centre = background.clone();
    let mut neighbour = shifted(&background, width, height, PAN[0], PAN[1]);

    for y in LAYER_Y {
        for x in LAYER_X {
            let in_stripe = STRIPE_X.contains(&x) && STRIPE_Y.contains(&y);
            let value = if in_stripe {
                0.5
            } else {
                layer[(y * width + x) as usize]
            };
            let moved_x = (x as i32 + LAYER[0]) as u32;
            let moved_y = (y as i32 + LAYER[1]) as u32;
            centre[(y * width + x) as usize] = value;
            neighbour[(moved_y * width + moved_x) as usize] = value;
        }
    }

    add_noise(&mut centre, sigma, 63);
    add_noise(&mut neighbour, sigma, 64);

    LayeredPan {
        width,
        height,
        centre,
        neighbour,
        sigma,
    }
}

impl LayeredPan {
    fn input<'a>(&'a self, field: &'a [i32]) -> StageInput<'a> {
        StageInput {
            width: self.width,
            height: self.height,
            centre: &self.centre,
            neighbour: &self.neighbour,
            field,
            seeds: None,
            prediction: Some(translation(PAN[0] as f32, PAN[1] as f32)),
            sad_floor: noise_floor(self.sigma),
            reads_floor: true,
        }
    }

    fn blocks(&self) -> usize {
        (blocks_along(self.width) * blocks_along(self.height)) as usize
    }

    fn block_at(&self, x: u32, y: u32) -> usize {
        ((y / STEP) * blocks_along(self.width) + x / STEP) as usize
    }
}

#[test]
fn an_independent_layer_is_found() {
    let client = make_client();
    let pan = layered_pan();
    let field = vec![0i32; 2 * pan.blocks()];
    let input = pan.input(&field);
    let shipped = SearchShape {
        search_radius: 8,
        pyramid_levels: 1,
        params: TruemotionParams::default(),
        scratch_fill: 0,
    };
    let unstaged = SearchShape {
        params: TruemotionParams {
            field_lambda: 0.0,
            ..TruemotionParams::default()
        },
        ..shipped
    };

    let with_stage = run_search(&client, &input, shipped);
    let without_stage = run_search(&client, &input, unstaged);

    // Flat stripe blocks far enough from its ends that every vector a search tries stays flat
    // along x.
    let stripe_blocks: Vec<usize> = (STRIPE_X.start + 16..=STRIPE_X.end - 32)
        .step_by(STEP as usize)
        .map(|x| pan.block_at(x, STRIPE_Y.start))
        .collect();
    let missed_without = stripe_blocks
        .iter()
        .filter(|&&block| without_stage.vector(block) != LAYER)
        .count();
    assert!(
        missed_without > 0,
        "premise: grain misleads the stripe without the stage"
    );

    for &block in &stripe_blocks {
        assert_eq!(with_stage.vector(block), LAYER, "stripe block {block}");
    }

    // Textured layer blocks keep the layer's motion against the pan around them.
    for x in (LAYER_X.start..=LAYER_X.end - BLKSIZE).step_by(STEP as usize) {
        let block = pan.block_at(x, LAYER_Y.start);
        assert_eq!(with_stage.vector(block), LAYER, "layer block {block}");
    }

    // The pan beside the layer still follows the camera.
    for y in (LAYER_Y.start..LAYER_Y.end).step_by(STEP as usize) {
        let block = pan.block_at(32, y);
        assert_eq!(with_stage.vector(block), PAN, "pan block {block}");
    }
}

#[test]
fn reseed_equals_streaming() {
    let client = make_client();
    let pan = layered_pan();
    let field = vec![0i32; 2 * pan.blocks()];
    let input = pan.input(&field);

    // Two passes make the second pass read the change flags the first pass wrote.
    for passes in 1..=2u32 {
        let params = TruemotionParams {
            passes,
            ..TruemotionParams::default()
        };
        let fresh = SearchShape {
            search_radius: 4,
            pyramid_levels: 2,
            params,
            scratch_fill: 0,
        };
        // A stream's stage buffers still hold an earlier search's fields and flags when the next
        // search starts.
        let streamed = SearchShape {
            scratch_fill: 77,
            ..fresh
        };

        let reseeded = run_search(&client, &input, fresh);
        let streaming = run_search(&client, &input, streamed);

        assert_eq!(reseeded.field, streaming.field, "{passes} passes");
        assert_eq!(
            reseeded.confidence_bits(),
            streaming.confidence_bits(),
            "{passes} passes"
        );
    }
}

#[test]
fn deterministic_runs_are_bit_identical() {
    let client = make_client();
    let pan = layered_pan();
    let field = vec![0i32; 2 * pan.blocks()];
    let input = pan.input(&field);

    for passes in 1..=2u32 {
        let params = TruemotionParams {
            passes,
            ..TruemotionParams::default()
        };
        let shape = SearchShape {
            search_radius: 4,
            pyramid_levels: 2,
            params,
            scratch_fill: 77,
        };

        let first = run_search(&client, &input, shape);
        let second = run_search(&client, &input, shape);

        assert_eq!(first.field, second.field, "{passes} passes");
        assert_eq!(
            first.confidence_bits(),
            second.confidence_bits(),
            "{passes} passes"
        );
    }
}

#[test]
fn a_search_that_escalates_runs_a_second_pass() {
    let fixed = SearchSchedule::fixed(4);
    let escalating = SearchSchedule {
        first_radius: 4,
        ceiling: 8,
    };

    let fixed_stage = TruemotionParams::for_schedule(fixed);
    let escalating_stage = TruemotionParams::for_schedule(escalating);

    assert_eq!(fixed_stage.passes, 1);
    assert_eq!(escalating_stage.passes, 2);
}

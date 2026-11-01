use std::collections::HashSet;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{frame_shifted_by, prediction_buffer};
use crate::engine::align::StorageAlign;
use crate::motion::kernels::{
    BLOCK_MATCH_THREADS,
    MODEL_CANDIDATE_BIAS,
    mc_block_match_coarse,
    mc_block_match_fine,
};
use crate::motion::{
    MatchTargets,
    MotionCompensationMode,
    MotionCtx,
    MotionEstimation,
    level_dims,
    pyramid_slot_pixels,
    run_analyse,
};
use crate::noise::tests::sad_floor_buffer;
use crate::noise::{DERIVED_SAD_FLOOR, DERIVED_WORDS};
use crate::tune::gpu::{bind, read_f32, read_i32, zeroed_words};
use crate::tune::tests::fixtures::upload;
use crate::tune::tests::runtime::test_client;

const THSAD: f32 = 5.12;
/// Unwritten values between each neighbour's confidence and SAD slices.
const CONF_PADDING: u32 = 3;

/// A hashed texture with every value between 0.2 and 0.8.
fn texture(width: u32, height: u32, seed: u32) -> Vec<f32> {
    (0..width * height)
        .map(|index| {
            let seed_offset = seed.wrapping_mul(0x9E37_79B9);
            let mut hash = index.wrapping_add(seed_offset);
            hash = hash.wrapping_mul(2_654_435_761);
            hash ^= hash >> 15;
            hash = hash.wrapping_mul(0x85EB_CA6B);
            hash ^= hash >> 13;
            0.2 + 0.6 * (hash as f32 / u32::MAX as f32)
        })
        .collect()
}

fn translation(tx: f32, ty: f32) -> [f32; 6] {
    [1.0, 0.0, 0.0, 1.0, tx, ty]
}

/// Uploads `centre` as slot 0 and each neighbour after it, with a table naming slots 1 onward.
fn slot_frames(client: &Client, centre: &[f32], neighbours: &[Vec<f32>]) -> (Handle, Handle, usize) {
    let mut frames = centre.to_vec();
    for neighbour in neighbours {
        frames.extend_from_slice(neighbour);
    }

    let pyramid = upload(client, &frames);
    let table: Vec<u32> = (1..=neighbours.len() as u32).collect();
    let table_bytes = u32::as_bytes(&table);
    let neighbour_slots = client.create_from_slice(table_bytes);

    (pyramid, neighbour_slots, frames.len())
}

struct FineSearch {
    width: u32,
    height: u32,
    blksize: u32,
    step: u32,
    search_radius: u32,
    sad_floor: f32,
    write_sad: bool,
}

struct FineOutput {
    vectors: Vec<i32>,
    confidence: Vec<f32>,
    /// The winning SADs, then the window means, each in the confidence layout.
    sad: Vec<f32>,
}

impl FineSearch {
    fn blocks_x(&self) -> u32 {
        self.width.div_ceil(self.step)
    }

    fn blocks(&self) -> u32 {
        let blocks_y = self.height.div_ceil(self.step);
        self.blocks_x() * blocks_y
    }

    fn conf_stride(&self) -> u32 {
        self.blocks() + CONF_PADDING
    }

    /// Runs the unseeded fine kernel against each neighbour, with the model candidate on when
    /// `predictions` holds six words per neighbour.
    fn run(&self, centre: &[f32], neighbours: &[Vec<f32>], predictions: Option<&[f32]>) -> FineOutput {
        self.run_seeded(centre, neighbours, predictions, None)
    }

    /// [Self::run] with every block seeded at `seed` when given.
    fn run_seeded(
        &self,
        centre: &[f32],
        neighbours: &[Vec<f32>],
        predictions: Option<&[f32]>,
        seed: Option<(i32, i32)>,
    ) -> FineOutput {
        let client = test_client();
        let neighbour_count = neighbours.len() as u32;
        let (pyramid, neighbour_slots, pyramid_len) = slot_frames(&client, centre, neighbours);
        let (prediction_words, prediction_len, model_candidate) = prediction_buffer(&client, predictions);
        let noise = sad_floor_buffer(&client, DERIVED_SAD_FLOOR, self.sad_floor);

        let mv_stride = 2 * self.blocks();
        let mv_len = (mv_stride * neighbour_count) as usize;
        let conf_len = (self.conf_stride() * neighbour_count) as usize;
        let sad_len = 2 * conf_len;
        let (seed_dx, seed_dy) = seed.unwrap_or((0, 0));
        let seeds: Vec<i32> = (0..mv_len)
            .map(|index| if index % 2 == 0 { seed_dx } else { seed_dy })
            .collect();
        let seed_bytes = i32::as_bytes(&seeds);
        let mv_field = client.create_from_slice(seed_bytes);
        let use_seed = u32::from(seed.is_some());
        let unwritten = vec![f32::NAN; sad_len];
        let confidence = upload(&client, &unwritten[..conf_len]);
        let sad = upload(&client, &unwritten);

        let blocks_y = self.height.div_ceil(self.step);
        let blocks_x = self.blocks_x();
        let cube_count = CubeCount::new_3d(blocks_x, blocks_y, neighbour_count);
        let cube_dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);
        let slot_pixels = self.width * self.height;
        let pyramid_arg = bind(&pyramid, pyramid_len);
        let slots_arg = bind(&neighbour_slots, neighbours.len());
        let mv_field_arg = bind(&mv_field, mv_len);
        let confidence_arg = bind(&confidence, conf_len);
        let sad_arg = bind(&sad, sad_len);
        let predictions_arg = bind(&prediction_words, prediction_len);
        let noise_arg = bind(&noise, DERIVED_WORDS as usize);

        // SAFETY: every binding covers the values its buffer was allocated with.
        unsafe {
            mc_block_match_fine::launch_unchecked(
                &client,
                cube_count,
                cube_dim,
                pyramid_arg,
                slots_arg,
                mv_field_arg,
                confidence_arg,
                sad_arg,
                predictions_arg,
                0u32,
                true,
                self.write_sad,
                model_candidate,
                noise_arg,
                DERIVED_SAD_FLOOR,
                THSAD,
                self.width,
                self.height,
                slot_pixels,
                self.blksize,
                self.step,
                self.search_radius,
                use_seed,
                self.blocks_x(),
                mv_stride,
                self.conf_stride(),
            );
        }

        let vectors = read_i32(&client, &mv_field, mv_len).expect("the field reads back");
        let confidence = read_f32(&client, &confidence, conf_len).expect("confidence reads back");
        let sad = read_f32(&client, &sad, sad_len).expect("the SAD reads back");

        FineOutput {
            vectors,
            confidence,
            sad,
        }
    }

    fn vector(&self, output: &FineOutput, neighbour: u32, col: u32, row: u32) -> (i32, i32) {
        let block = row * self.blocks_x() + col;
        let index = (neighbour * 2 * self.blocks() + 2 * block) as usize;
        (output.vectors[index], output.vectors[index + 1])
    }

    fn confidence(&self, output: &FineOutput, col: u32, row: u32) -> f32 {
        let block = row * self.blocks_x() + col;
        output.confidence[block as usize]
    }
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// A 160x64 frame cut into 16-pixel blocks every 8 pixels, searched 4 pixels around its centre.
fn pan_search() -> FineSearch {
    FineSearch {
        width: 160,
        height: 64,
        blksize: 16,
        step: 8,
        search_radius: 4,
        sad_floor: 0.0,
        write_sad: false,
    }
}

#[test]
fn a_pan_beyond_reach_takes_the_model_candidate() {
    let search = pan_search();
    let centre = texture(search.width, search.height, 1);
    let neighbour = frame_shifted_by(&centre, search.width, search.height, 20, 0);
    let neighbours = [neighbour];
    let predictions = translation(20.0, 0.0);

    let without_model = search.run(&centre, &neighbours, None);
    let with_model = search.run(&centre, &neighbours, Some(&predictions));

    // Blocks up to x = 124 find their content fully inside the neighbour.
    for row in 0..search.height.div_ceil(search.step) {
        for col in 0..=124 / search.step {
            let unseeded = search.vector(&without_model, 0, col, row);
            let seeded = search.vector(&with_model, 0, col, row);
            let confidence = search.confidence(&with_model, col, row);

            assert_ne!(unseeded, (20, 0), "premise: block ({col}, {row}) is out of reach");
            assert_eq!(seeded, (20, 0), "block ({col}, {row})");
            assert!(confidence > 0.99, "block ({col}, {row}) confidence {confidence}");
        }
    }
}

#[test]
fn static_content_keeps_the_existing_seed() {
    let mut search = pan_search();
    search.sad_floor = 0.5;
    search.write_sad = true;
    let centre = texture(search.width, search.height, 2);
    let neighbours = [centre.clone()];
    let predictions = translation(20.0, 0.0);

    let without_model = search.run(&centre, &neighbours, None);
    let with_model = search.run(&centre, &neighbours, Some(&predictions));

    assert_eq!(with_model.vectors, without_model.vectors);
    assert_eq!(bits(&with_model.confidence), bits(&without_model.confidence));
    assert_eq!(bits(&with_model.sad), bits(&without_model.sad));
}

/// One 16x16 block at the origin, whose seed and model positions differ from it by a uniform
/// `seed_offset` and `model_offset`, searched with a radius of 0.
fn offset_pair(seed_offset: f32, model_offset: f32) -> (i32, i32) {
    let search = FineSearch {
        width: 64,
        height: 16,
        blksize: 16,
        step: 16,
        search_radius: 0,
        sad_floor: 1.0,
        write_sad: false,
    };
    let centre = texture(search.width, search.height, 3);
    let mut neighbour = texture(search.width, search.height, 4);

    for y in 0..16 {
        for x in 0..16 {
            let source = centre[y * 64 + x];
            neighbour[y * 64 + x] = source + seed_offset;
            neighbour[y * 64 + x + 20] = source + model_offset;
        }
    }

    let neighbours = [neighbour];
    let predictions = translation(20.0, 0.0);
    let output = search.run(&centre, &neighbours, Some(&predictions));

    search.vector(&output, 0, 0, 0)
}

#[test]
fn an_identity_prediction_keeps_a_non_zero_seed() {
    let search = FineSearch {
        width: 64,
        height: 16,
        blksize: 16,
        step: 16,
        search_radius: 0,
        sad_floor: 0.0,
        write_sad: true,
    };
    let centre = texture(search.width, search.height, 5);
    let neighbours = [centre.clone()];
    let identity = translation(0.0, 0.0);
    let seed = Some((20, 0));

    let without_model = search.run_seeded(&centre, &neighbours, None, seed);
    let with_model = search.run_seeded(&centre, &neighbours, Some(&identity), seed);

    let seeded_vector = search.vector(&with_model, 0, 0, 0);
    assert_eq!(seeded_vector, (20, 0), "the seed is kept, not the zero window");
    assert_eq!(with_model.vectors, without_model.vectors);
    assert_eq!(bits(&with_model.sad), bits(&without_model.sad));
}

#[test]
fn a_near_tie_keeps_the_existing_seed() {
    // 256 pixels apart by 0.0003 give a SAD gap of 0.077, inside the 0.125 bias of a unit floor.
    assert_eq!(
        MODEL_CANDIDATE_BIAS, 0.125,
        "premise: the bias the offsets straddle"
    );

    let vector = offset_pair(0.01, 0.0097);

    assert_eq!(vector, (0, 0));
}

#[test]
fn a_win_beyond_the_bias_takes_the_model_candidate() {
    // A gap of 0.001 per pixel gives a SAD gap of 0.256, outside the 0.125 bias of a unit floor.
    let vector = offset_pair(0.01, 0.009);

    assert_eq!(vector, (20, 0));
}

/// One 16x16 block at the origin whose content sits 2 pixels right in the neighbour, searched 1
/// pixel around zero, so only a model window can reach the match.
fn match_two_pixels_away(predictions: Option<&[f32]>) -> FineOutput {
    let search = FineSearch {
        width: 64,
        height: 16,
        blksize: 16,
        step: 16,
        search_radius: 1,
        sad_floor: 1.0,
        write_sad: true,
    };
    let centre = texture(search.width, search.height, 7);
    let mut neighbour = texture(search.width, search.height, 8);

    for y in 0..16 {
        for x in 0..16 {
            neighbour[y * 64 + x + 2] = centre[y * 64 + x];
        }
    }

    let neighbours = [neighbour];

    search.run(&centre, &neighbours, predictions)
}

#[test]
fn a_model_inside_the_seed_window_gives_the_seed_only_result() {
    let at_reach = translation(1.0, 0.0);

    let without_model = match_two_pixels_away(None);
    let with_model = match_two_pixels_away(Some(&at_reach));

    assert_ne!(
        &without_model.vectors[..2],
        &[2, 0],
        "premise: the seed window cannot reach the match"
    );
    assert_eq!(with_model.vectors, without_model.vectors);
    assert_eq!(bits(&with_model.confidence), bits(&without_model.confidence));
    assert_eq!(bits(&with_model.sad), bits(&without_model.sad));
}

#[test]
fn a_model_one_past_the_seed_window_searches_its_own_window() {
    let past_reach = translation(2.0, 0.0);

    let with_model = match_two_pixels_away(Some(&past_reach));

    assert_eq!(&with_model.vectors[..2], &[2, 0]);
}

/// One 16x16 block at the origin whose content sits 2 pixels down in the neighbour, searched 1
/// pixel around zero, so only a model window can reach the match.
fn match_two_pixels_down(predictions: Option<&[f32]>) -> FineOutput {
    let search = FineSearch {
        width: 16,
        height: 64,
        blksize: 16,
        step: 16,
        search_radius: 1,
        sad_floor: 1.0,
        write_sad: true,
    };
    let centre = texture(search.width, search.height, 17);
    let mut neighbour = texture(search.width, search.height, 18);

    for y in 0..16 {
        for x in 0..16 {
            neighbour[(y + 2) * 16 + x] = centre[y * 16 + x];
        }
    }

    let neighbours = [neighbour];

    search.run(&centre, &neighbours, predictions)
}

#[test]
fn a_model_inside_the_seed_window_on_y_gives_the_seed_only_result() {
    let at_reach = translation(0.0, 1.0);

    let without_model = match_two_pixels_down(None);
    let with_model = match_two_pixels_down(Some(&at_reach));

    assert_ne!(
        &without_model.vectors[..2],
        &[0, 2],
        "premise: the seed window cannot reach the match"
    );
    assert_eq!(with_model.vectors, without_model.vectors);
    assert_eq!(bits(&with_model.confidence), bits(&without_model.confidence));
    assert_eq!(bits(&with_model.sad), bits(&without_model.sad));
}

#[test]
fn a_model_one_past_the_seed_window_on_y_searches_its_own_window() {
    let past_reach = translation(0.0, 2.0);

    let with_model = match_two_pixels_down(Some(&past_reach));

    assert_eq!(&with_model.vectors[..2], &[0, 2]);
}

#[test]
fn a_static_overlay_on_a_pan_keeps_its_own_seed() {
    let search = pan_search();
    let width = search.width as usize;
    let world = texture(search.width, search.height, 5);
    let overlay = texture(search.width, search.height, 6);
    let mut centre = world.clone();
    let mut neighbour = frame_shifted_by(&world, search.width, search.height, 20, 0);

    // The overlay covers x between 48 and 96 and y between 16 and 48 in both frames.
    for y in 16..48 {
        for x in 48..96 {
            centre[y * width + x] = overlay[y * width + x];
            neighbour[y * width + x] = overlay[y * width + x];
        }
    }

    let neighbours = [neighbour];
    let predictions = translation(20.0, 0.0);
    let output = search.run(&centre, &neighbours, Some(&predictions));

    for row in 2..=4 {
        for col in 6..=10 {
            let vector = search.vector(&output, 0, col, row);
            assert_eq!(vector, (0, 0), "overlay block ({col}, {row})");
        }
    }

    for col in 0..=124 / search.step {
        let vector = search.vector(&output, 0, col, 0);
        assert_eq!(vector, (20, 0), "background block ({col}, 0)");
    }
}

#[test]
fn each_neighbour_reads_its_own_prediction() {
    let search = pan_search();
    let centre = texture(search.width, search.height, 7);
    let forward = frame_shifted_by(&centre, search.width, search.height, 20, 0);
    let backward = frame_shifted_by(&centre, search.width, search.height, -20, 0);
    let neighbours = [forward, backward];
    let backward_prediction = translation(-20.0, 0.0);
    let mut predictions = translation(20.0, 0.0).to_vec();
    predictions.extend_from_slice(&backward_prediction);

    let output = search.run(&centre, &neighbours, Some(&predictions));

    // Block 8 sits at x = 64, far enough from both edges for either pan.
    let forward_vector = search.vector(&output, 0, 8, 3);
    let backward_vector = search.vector(&output, 1, 8, 3);

    assert_eq!(forward_vector, (20, 0));
    assert_eq!(backward_vector, (-20, 0));
}

/// A roll of 0.05 radians about the frame centre followed by a shift of `(10, -6)`.
fn roll_predictions(size: f32) -> [f32; 6] {
    let (sin, cos) = 0.05f32.sin_cos();
    let pivot = size / 2.0;
    let tx = pivot - (cos * pivot - sin * pivot) + 10.0;
    let ty = pivot - (sin * pivot + cos * pivot) - 6.0;

    [cos, -sin, sin, cos, tx, ty]
}

/// The rounded displacement `predictions` gives at `(centre_x, centre_y)`, computed as the kernel
/// does.
fn predicted_vector(predictions: &[f32; 6], centre_x: f32, centre_y: f32) -> (i32, i32) {
    let mapped_x = predictions[0] * centre_x + predictions[1] * centre_y + predictions[4];
    let mapped_y = predictions[2] * centre_x + predictions[3] * centre_y + predictions[5];
    let dx = mapped_x - centre_x;
    let dy = mapped_y - centre_y;

    for value in [dx, dy] {
        let distance_to_half = (value.fract().abs() - 0.5).abs();
        assert!(
            distance_to_half > 0.02,
            "premise: {value} rounds the same on any backend"
        );
    }

    let rounded_dx = (dx + 0.5).floor();
    let rounded_dy = (dy + 0.5).floor();
    (rounded_dx as i32, rounded_dy as i32)
}

#[test]
fn roll_predictions_differ_per_block() {
    // 8-pixel blocks every 16 pixels leave gaps, so per-block copies a pixel apart never overlap.
    let search = FineSearch {
        width: 128,
        height: 128,
        blksize: 8,
        step: 16,
        search_radius: 0,
        sad_floor: 0.0,
        write_sad: false,
    };
    let size = search.width as usize;
    let predictions = roll_predictions(search.width as f32);
    let centre = texture(search.width, search.height, 8);
    let mut neighbour = texture(search.width, search.height, 9);
    let mut expected = Vec::new();

    for row in 0..search.width / search.step {
        for col in 0..search.width / search.step {
            let origin_x = (col * search.step) as i32;
            let origin_y = (row * search.step) as i32;
            let centre_x = (origin_x + 4) as f32;
            let centre_y = (origin_y + 4) as f32;
            let vector = predicted_vector(&predictions, centre_x, centre_y);
            let target_x = origin_x + vector.0;
            let target_y = origin_y + vector.1;
            let limit = (size - 8) as i32;
            if target_x < 0 || target_y < 0 || target_x > limit || target_y > limit {
                continue;
            }

            for y in 0..8 {
                for x in 0..8 {
                    let source = (origin_y as usize + y) * size + origin_x as usize + x;
                    let target = (target_y as usize + y) * size + target_x as usize + x;
                    neighbour[target] = centre[source];
                }
            }

            expected.push((col, row, vector));
        }
    }

    let neighbours = [neighbour];
    let output = search.run(&centre, &neighbours, Some(&predictions));

    let distinct: HashSet<(i32, i32)> = expected.iter().map(|entry| entry.2).collect();
    assert!(
        distinct.len() >= 4,
        "premise: the roll spreads the vectors, got {distinct:?}"
    );

    for (col, row, vector) in expected {
        let found = search.vector(&output, 0, col, row);
        assert_eq!(found, vector, "block ({col}, {row})");
    }
}

#[test]
fn the_winning_sad_lands_in_the_confidence_layout() {
    let search = FineSearch {
        width: 64,
        height: 16,
        blksize: 16,
        step: 16,
        search_radius: 0,
        sad_floor: 0.0,
        write_sad: true,
    };
    let centre = vec![0.25f32; 64 * 16];
    let first = vec![0.35f32; 64 * 16];
    let second = vec![0.3f32; 64 * 16];
    let neighbours = [first, second];

    let output = search.run(&centre, &neighbours, None);

    // A window of one offset has its winning SAD as its mean, written after both neighbours.
    let blocks = search.blocks() as usize;
    let stride = search.conf_stride() as usize;
    let means_base = 2 * stride;
    for (neighbour, expected) in [(0usize, 25.6f32), (1, 12.8)] {
        for base in [0, means_base] {
            let start = base + neighbour * stride;
            let values = &output.sad[start..start + blocks];
            let padding = &output.sad[start + blocks..start + stride];

            for value in values {
                assert!(
                    (value - expected).abs() < 0.01,
                    "neighbour {neighbour} at {base}: value {value}"
                );
            }

            assert!(
                padding.iter().all(|value| value.is_nan()),
                "neighbour {neighbour} at {base}: padding"
            );
        }
    }
}

/// The SAD of the block at `(origin_x, 0)` against `neighbour` shifted by `(dx, dy)`, read clamped
/// to the frame as the kernel stages it.
fn clamped_sad(
    centre: &[f32],
    neighbour: &[f32],
    search: &FineSearch,
    origin_x: i32,
    dx: i32,
    dy: i32,
) -> f32 {
    let width = search.width as i32;
    let height = search.height as i32;
    let mut total = 0.0f32;

    for y in 0..search.blksize as i32 {
        for x in 0..search.blksize as i32 {
            let source_x = (origin_x + x + dx).clamp(0, width - 1);
            let source_y = (y + dy).clamp(0, height - 1);
            let centre_value = centre[(y * width + origin_x + x) as usize];
            let neighbour_value = neighbour[(source_y * width + source_x) as usize];
            total += (centre_value - neighbour_value).abs();
        }
    }

    total
}

#[test]
fn the_window_mean_averages_every_offset() {
    let search = FineSearch {
        width: 64,
        height: 16,
        blksize: 16,
        step: 16,
        search_radius: 1,
        sad_floor: 0.0,
        write_sad: true,
    };
    let centre = texture(search.width, search.height, 15);
    let neighbour = texture(search.width, search.height, 16);
    let neighbours = [neighbour.clone()];

    let output = search.run(&centre, &neighbours, None);

    let means_base = search.conf_stride() as usize;
    for block in 0..search.blocks() {
        let origin_x = (block * search.step) as i32;
        let mut total = 0.0f32;
        for dy in -1..=1 {
            for dx in -1..=1 {
                total += clamped_sad(&centre, &neighbour, &search, origin_x, dx, dy);
            }
        }

        let expected = total / 9.0;
        let mean = output.sad[means_base + block as usize];
        let error = (mean - expected).abs();
        assert!(
            error < 1e-3 * expected,
            "block {block}: mean {mean}, expected {expected}"
        );
    }
}

#[test]
fn the_sad_output_stays_untouched_when_off() {
    let search = pan_search();
    let centre = texture(search.width, search.height, 10);
    let neighbours = [texture(search.width, search.height, 11)];

    let output = search.run(&centre, &neighbours, None);

    assert!(output.sad.iter().all(|value| value.is_nan()));
}

struct CoarseSearch {
    level_width: u32,
    level_height: u32,
    blksize: u32,
    step: u32,
    search_radius: u32,
    fine_step: u32,
}

impl CoarseSearch {
    fn fine_blocks_x(&self) -> u32 {
        (2 * self.level_width).div_ceil(self.fine_step)
    }

    fn fine_blocks_y(&self) -> u32 {
        (2 * self.level_height).div_ceil(self.fine_step)
    }

    /// Runs the coarse kernel at a level scale of 2 and returns the fine seeds it writes.
    fn run(&self, centre: &[f32], neighbour: &[f32], predictions: Option<&[f32]>) -> Vec<i32> {
        let client = test_client();
        let neighbours = [neighbour.to_vec()];
        let (pyramid, neighbour_slots, pyramid_len) = slot_frames(&client, centre, &neighbours);
        let (prediction_words, prediction_len, model_candidate) = prediction_buffer(&client, predictions);
        let noise = sad_floor_buffer(&client, DERIVED_SAD_FLOOR, 1.0);

        let mv_len = (2 * self.fine_blocks_x() * self.fine_blocks_y()) as usize;
        let zeros = vec![0i32; mv_len];
        let zero_bytes = i32::as_bytes(&zeros);
        let mv_field = client.create_from_slice(zero_bytes);
        let coarse_blocks_x = self.level_width.div_ceil(self.step);
        let coarse_blocks_y = self.level_height.div_ceil(self.step);
        let cube_count = CubeCount::new_3d(coarse_blocks_x, coarse_blocks_y, 1);
        let cube_dim = CubeDim::new_1d(BLOCK_MATCH_THREADS);
        let slot_pixels = self.level_width * self.level_height;
        let pyramid_arg = bind(&pyramid, pyramid_len);
        let slots_arg = bind(&neighbour_slots, 1);
        let mv_field_arg = bind(&mv_field, mv_len);
        let noise_arg = bind(&noise, DERIVED_WORDS as usize);
        let predictions_arg = bind(&prediction_words, prediction_len);

        // SAFETY: every binding covers the values its buffer was allocated with.
        unsafe {
            mc_block_match_coarse::launch_unchecked(
                &client,
                cube_count,
                cube_dim,
                pyramid_arg,
                slots_arg,
                mv_field_arg,
                noise_arg,
                predictions_arg,
                0u32,
                0.25,
                model_candidate,
                self.level_width,
                self.level_height,
                0,
                slot_pixels,
                self.blksize,
                self.step,
                self.search_radius,
                2,
                self.fine_blocks_x(),
                self.fine_blocks_y(),
                self.fine_step,
                mv_len as u32,
            );
        }

        read_i32(&client, &mv_field, mv_len).expect("the field reads back")
    }
}

/// A 160x64 frame's half-resolution level, in 8-pixel blocks every 4 pixels, searched 2 pixels out.
fn coarse_search() -> CoarseSearch {
    CoarseSearch {
        level_width: 80,
        level_height: 32,
        blksize: 8,
        step: 4,
        search_radius: 2,
        fine_step: 8,
    }
}

#[test]
fn the_coarse_pass_centres_on_the_model() {
    let search = coarse_search();
    let centre = texture(search.level_width, search.level_height, 12);
    let neighbour = frame_shifted_by(&centre, search.level_width, search.level_height, 10, 0);
    let predictions = translation(20.0, 0.0);

    let without_model = search.run(&centre, &neighbour, None);
    let with_model = search.run(&centre, &neighbour, Some(&predictions));

    // Coarse blocks up to x = 60 match fully inside the level, and seed fine columns up to 14.
    for row in 0..search.fine_blocks_y() {
        for col in 0..=14 {
            let index = (2 * (row * search.fine_blocks_x() + col)) as usize;
            let unseeded = (without_model[index], without_model[index + 1]);
            let seeded = (with_model[index], with_model[index + 1]);

            assert_ne!(
                unseeded,
                (20, 0),
                "premise: fine block ({col}, {row}) is out of reach"
            );
            assert_eq!(seeded, (20, 0), "fine block ({col}, {row})");
        }
    }
}

#[test]
fn the_coarse_pass_keeps_zero_on_static_content() {
    let search = coarse_search();
    let centre = texture(search.level_width, search.level_height, 13);
    let predictions = translation(20.0, 0.0);

    let without_model = search.run(&centre, &centre, None);
    let with_model = search.run(&centre, &centre, Some(&predictions));

    assert_eq!(with_model, without_model);
}

#[test]
fn a_coarse_model_at_the_window_edge_adds_no_window() {
    let search = coarse_search();
    let centre = texture(search.level_width, search.level_height, 14);
    let neighbour = frame_shifted_by(&centre, search.level_width, search.level_height, 3, 0);
    // 4 full-resolution pixels are 2 on the level, exactly the search radius.
    let at_reach = translation(4.0, 0.0);
    // 6 full-resolution pixels are 3 on the level, one past it.
    let past_reach = translation(6.0, 0.0);

    let without_model = search.run(&centre, &neighbour, None);
    let inside = search.run(&centre, &neighbour, Some(&at_reach));
    let outside = search.run(&centre, &neighbour, Some(&past_reach));

    assert_ne!(
        &without_model[..2],
        &[6, 0],
        "premise: the zero window cannot reach the match"
    );
    assert_eq!(inside, without_model);
    assert_eq!(&outside[..2], &[6, 0]);
}

#[test]
fn a_coarse_model_at_the_window_edge_on_y_adds_no_window() {
    let search = coarse_search();
    let centre = texture(search.level_width, search.level_height, 15);
    let neighbour = frame_shifted_by(&centre, search.level_width, search.level_height, 0, 3);
    // 4 full-resolution pixels are 2 on the level, exactly the search radius.
    let at_reach = translation(0.0, 4.0);
    // 6 full-resolution pixels are 3 on the level, one past it.
    let past_reach = translation(0.0, 6.0);

    let without_model = search.run(&centre, &neighbour, None);
    let inside = search.run(&centre, &neighbour, Some(&at_reach));
    let outside = search.run(&centre, &neighbour, Some(&past_reach));

    assert_ne!(
        &without_model[..2],
        &[0, 6],
        "premise: the zero window cannot reach the match"
    );
    assert_eq!(inside, without_model);
    assert_eq!(&outside[..2], &[0, 6]);
}

/// Halves a frame by averaging each 2x2 square, as the pyramid build does.
fn half_level(frame: &[f32], width: u32, height: u32) -> Vec<f32> {
    let (half_width, half_height) = level_dims(width, height, 1);
    let mut level = Vec::with_capacity((half_width * half_height) as usize);

    for y in 0..half_height {
        for x in 0..half_width {
            let top = (2 * y * width + 2 * x) as usize;
            let bottom = top + width as usize;
            let sum = frame[top] + frame[top + 1] + frame[bottom] + frame[bottom + 1];
            level.push(sum * 0.25);
        }
    }

    level
}

/// A two-level pyramid buffer over `frames`, each slot padded as the motion context lays it out.
fn two_level_pyramid(frames: &[Vec<f32>], width: u32, height: u32, align: StorageAlign) -> Vec<f32> {
    let mut pyramid = Vec::new();

    for level in 0..2 {
        let slot_pixels = pyramid_slot_pixels(width, height, level, align) as usize;
        for frame in frames {
            let level_frame = if level == 0 {
                frame.clone()
            } else {
                half_level(frame, width, height)
            };
            let padding = slot_pixels - level_frame.len();
            let padding_values = std::iter::repeat_n(0.0, padding);

            pyramid.extend_from_slice(&level_frame);
            pyramid.extend(padding_values);
        }
    }

    pyramid
}

/// Runs the two-level analyse over one neighbour and returns its field.
fn analyse_pan(client: &Client, ctx: &MotionCtx, pyramid: &Handle, predictions: Option<&Handle>) -> Vec<i32> {
    let neighbour_table = [1u32];
    let neighbour_bytes = u32::as_bytes(&neighbour_table);
    let neighbour_slots = client.create_from_slice(neighbour_bytes);
    let mv_len = ctx.mv_stride() as usize;
    let mv_field = zeroed_words(client, mv_len);
    let placeholder = zeroed_words(client, 1);
    let noise = zeroed_words(client, DERIVED_WORDS as usize);

    let targets = MatchTargets {
        pyramid,
        frame_count: 2,
        centre_slot: 0,
        neighbour_slots: &neighbour_slots,
        neighbours: 1,
        mv_field: &mv_field,
        mv_len,
        confidence: &placeholder,
        conf_len: 1,
        write_confidence: false,
        predictions,
        sad: None,
        escalation: None,
    };
    run_analyse(client, ctx, 192, 64, &targets, &noise, DERIVED_SAD_FLOOR, 1.0).expect("the analyse runs");

    read_i32(client, &mv_field, mv_len).expect("the field reads back")
}

#[test]
fn run_analyse_follows_a_pan_beyond_reach_with_predictions() {
    let client = test_client();
    let mode = MotionCompensationMode::Mvtools {
        blksize: 16,
        overlap: 8,
        search_radius: 4,
        max_search_escalation: 4,
        pyramid_levels: 2,
        estimation: MotionEstimation::Direct,
    };
    let align = StorageAlign::from_client(&client);
    let ctx = MotionCtx::new(mode, 192, 64, align).expect("the geometry is valid");
    let centre = texture(192, 64, 14);
    let neighbour = frame_shifted_by(&centre, 192, 64, 30, 0);
    let frames = [centre, neighbour];
    let pyramid_values = two_level_pyramid(&frames, 192, 64, align);
    let pyramid = upload(&client, &pyramid_values);
    let pan = translation(30.0, 0.0);
    let predictions = upload(&client, &pan);

    let without_model = analyse_pan(&client, &ctx, &pyramid, None);
    let with_model = analyse_pan(&client, &ctx, &pyramid, Some(&predictions));

    // Blocks up to x = 144 find their content fully inside the neighbour.
    for row in 0..ctx.blocks_y {
        for col in 0..=144 / ctx.step {
            let index = (2 * (row * ctx.blocks_x + col)) as usize;
            let unseeded = (without_model[index], without_model[index + 1]);
            let seeded = (with_model[index], with_model[index + 1]);

            assert_ne!(unseeded, (30, 0), "premise: block ({col}, {row}) is out of reach");
            assert_eq!(seeded, (30, 0), "block ({col}, {row})");
        }
    }
}

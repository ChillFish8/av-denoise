use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::engine::align::StorageAlign;
use crate::motion::global::{FitField, FitScratch, MAX_ROLL_TAN, MODEL_WORDS, ModelRing, run_model_fit};
use crate::motion::{MatchTargets, MotionCompensationMode, MotionCtx, MotionEstimation, run_analyse, thsad};
use crate::noise::{DERIVED_SAD_FLOOR, DERIVED_WORDS};
use crate::tune::gpu::{read_f32, zeroed_words};
use crate::tune::tests::fixtures::{hashed_unit, upload};
use crate::tune::tests::runtime::test_client;

const WIDTH: u32 = 960;
const HEIGHT: u32 = 540;
const RING_DEPTH: u32 = 4;
const PRIOR_SLOT: u32 = 0;
const FIT_SLOT: u32 = 1;
const RATIO_TOLERANCE: f64 = 1e-3;
const SHIFT_TOLERANCE: f64 = 0.05;

/// A camera model in luma pixels, `x' = a*x - b*y + tx` and `y' = b*x + a*y + ty`.
#[derive(Debug, Clone, Copy)]
struct Similarity {
    a: f64,
    b: f64,
    tx: f64,
    ty: f64,
}

impl Similarity {
    /// A roll and zoom about the frame centre, followed by a shift.
    fn about_centre(degrees: f64, zoom: f64, shift_x: f64, shift_y: f64) -> Self {
        let radians = degrees.to_radians();
        let a = zoom * radians.cos();
        let b = zoom * radians.sin();
        let centre_x = WIDTH as f64 / 2.0;
        let centre_y = HEIGHT as f64 / 2.0;
        let tx = centre_x - (a * centre_x - b * centre_y) + shift_x;
        let ty = centre_y - (b * centre_x + a * centre_y) + shift_y;

        Self { a, b, tx, ty }
    }

    fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let mapped_x = self.a * x - self.b * y + self.tx;
        let mapped_y = self.b * x + self.a * y + self.ty;

        (mapped_x, mapped_y)
    }

    fn inverse(&self) -> Self {
        let norm = self.a * self.a + self.b * self.b;
        let a = self.a / norm;
        let b = -self.b / norm;
        let tx = -(a * self.tx - b * self.ty);
        let ty = -(b * self.tx + a * self.ty);

        Self { a, b, tx, ty }
    }
}

/// The five words a fit writes to its ring slot.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Fitted {
    a: f32,
    b: f32,
    tx: f32,
    ty: f32,
    valid: f32,
}

impl Fitted {
    fn from_words(words: &[f32]) -> Self {
        Self {
            a: words[0],
            b: words[1],
            tx: words[2],
            ty: words[3],
            valid: words[4],
        }
    }

    fn assert_matches(&self, expected: Similarity) {
        assert_eq!(self.valid, 1.0, "the fit should be accepted, got {self:?}");

        let a_error = (self.a as f64 - expected.a).abs();
        let b_error = (self.b as f64 - expected.b).abs();
        let tx_error = (self.tx as f64 - expected.tx).abs();
        let ty_error = (self.ty as f64 - expected.ty).abs();
        assert!(
            a_error < RATIO_TOLERANCE,
            "a off by {a_error}: {self:?} vs {expected:?}"
        );
        assert!(
            b_error < RATIO_TOLERANCE,
            "b off by {b_error}: {self:?} vs {expected:?}"
        );
        assert!(
            tx_error < SHIFT_TOLERANCE,
            "tx off by {tx_error}: {self:?} vs {expected:?}"
        );
        assert!(
            ty_error < SHIFT_TOLERANCE,
            "ty off by {ty_error}: {self:?} vs {expected:?}"
        );
    }

    fn assert_identity_fallback(&self) {
        let expected = Fitted {
            a: 1.0,
            b: 0.0,
            tx: 0.0,
            ty: 0.0,
            valid: 0.0,
        };
        assert_eq!(*self, expected);
    }
}

fn fit_ctx(client: &Client, search_radius: u32, pyramid_levels: u32) -> MotionCtx {
    let mode = MotionCompensationMode::Mvtools {
        blksize: 16,
        overlap: 8,
        search_radius,
        pyramid_levels,
        estimation: MotionEstimation::Direct,
    };
    let align = StorageAlign::from_client(client);

    MotionCtx::new(mode, WIDTH, HEIGHT, align).unwrap()
}

/// The position each block's vector is measured from, `origin + blksize / 2`.
fn block_centre(ctx: &MotionCtx, block: usize) -> (f64, f64) {
    let col = block as u32 % ctx.blocks_x;
    let row = block as u32 / ctx.blocks_x;
    let half = ctx.blksize as f64 / 2.0;
    let centre_x = (col * ctx.step) as f64 + half;
    let centre_y = (row * ctx.step) as f64 + half;

    (centre_x, centre_y)
}

fn block_count(ctx: &MotionCtx) -> usize {
    (ctx.blocks_x * ctx.blocks_y) as usize
}

/// The rounded vector of each block when `model` maps its centre frame onto its neighbour.
fn camera_field(ctx: &MotionCtx, model: Similarity) -> Vec<[i32; 2]> {
    let blocks = block_count(ctx);
    let mut field = Vec::with_capacity(blocks);
    for block in 0..blocks {
        let (centre_x, centre_y) = block_centre(ctx, block);
        let (mapped_x, mapped_y) = model.apply(centre_x, centre_y);
        let mvx = (mapped_x - centre_x).round() as i32;
        let mvy = (mapped_y - centre_y).round() as i32;
        field.push([mvx, mvy]);
    }

    field
}

/// A vector between -12 and 12 per component, hashed from the block index and `seed`.
fn random_vector(block: usize, seed: u32) -> [i32; 2] {
    let unit_x = (hashed_unit(block as u32, seed) - 0.2) / 0.6;
    let unit_y = (hashed_unit(block as u32, seed + 1) - 0.2) / 0.6;
    let mvx = (unit_x * 24.99).floor() as i32 - 12;
    let mvy = (unit_y * 24.99).floor() as i32 - 12;

    [mvx, mvy]
}

fn random_field(ctx: &MotionCtx, seed: u32) -> Vec<[i32; 2]> {
    let blocks = block_count(ctx);

    (0..blocks).map(|block| random_vector(block, seed)).collect()
}

/// One pair slot with both directions at their padded offsets.
fn upload_pair(
    client: &Client,
    ctx: &MotionCtx,
    older_to_newer: &[[i32; 2]],
    newer_to_older: &[[i32; 2]],
) -> Handle {
    let stride = ctx.pair_direction_stride() as usize;
    let mut values = vec![0i32; 2 * stride];
    let directions = [older_to_newer, newer_to_older];
    for (direction, field) in directions.iter().enumerate() {
        let start = direction * stride;
        let flat = field.as_flattened();
        values[start..start + flat.len()].copy_from_slice(flat);
    }

    let bytes = i32::as_bytes(&values);
    client.create_from_slice(bytes)
}

/// A ring holding `prior` in [PRIOR_SLOT] and nothing else.
fn ring_with_prior(client: &Client, prior: Option<Similarity>) -> ModelRing {
    let mut words = vec![0.0f32; (RING_DEPTH * MODEL_WORDS) as usize];
    if let Some(model) = prior {
        let base = (PRIOR_SLOT * MODEL_WORDS) as usize;
        words[base] = model.a as f32;
        words[base + 1] = model.b as f32;
        words[base + 2] = model.tx as f32;
        words[base + 3] = model.ty as f32;
        words[base + 4] = 1.0;
    }

    let handle = upload(client, &words);
    ModelRing {
        handle,
        depth: RING_DEPTH,
    }
}

/// Fits one direction of a pair slot and reads back the model it writes.
fn fit_pairs(
    client: &Client,
    ctx: &MotionCtx,
    pairs: &Handle,
    direction: u32,
    sad: Option<&[f32]>,
    prior: Option<Similarity>,
) -> Fitted {
    let ring = ring_with_prior(client, prior);
    let blocks = ctx.blocks_x * ctx.blocks_y;
    let scratch = FitScratch::new(client, blocks);
    let sad_handle = sad.map(|values| upload(client, values));
    let field = FitField {
        pairs,
        direction,
        sad: sad_handle.as_ref(),
    };
    let prior_slot = prior.map(|_| PRIOR_SLOT);

    run_model_fit(
        client, ctx, WIDTH, HEIGHT, field, &scratch, prior_slot, &ring, FIT_SLOT,
    );

    let words = read_f32(client, ring.handle(), ring.words()).unwrap();
    let base = (FIT_SLOT * MODEL_WORDS) as usize;
    let end = base + MODEL_WORDS as usize;

    Fitted::from_words(&words[base..end])
}

/// Fits an older-to-newer field, with a random field in the other direction.
fn fit_forward(client: &Client, ctx: &MotionCtx, field: &[[i32; 2]], sad: Option<&[f32]>) -> Fitted {
    let other = random_field(ctx, 91);
    let pairs = upload_pair(client, ctx, field, &other);

    fit_pairs(client, ctx, &pairs, 0, sad, None)
}

/// A camera roll with a block group moving on its own over 30% of the blocks.
fn field_with_independent_group(ctx: &MotionCtx, camera: Similarity) -> Vec<[i32; 2]> {
    let mut field = camera_field(ctx, camera);
    for (block, vector) in field.iter_mut().enumerate() {
        let col = block as u32 % ctx.blocks_x;
        let row = block as u32 / ctx.blocks_x;
        if col < 60 && row < 41 {
            *vector = [-12, 7];
        }
    }

    field
}

#[test]
fn recovers_a_known_roll_zoom_and_shift() {
    let client = test_client();
    let ctx = fit_ctx(&client, 4, 2);
    let camera = Similarity::about_centre(2.5, 1.01, 6.0, -4.0);
    let field = camera_field(&ctx, camera);

    let fitted = fit_forward(&client, &ctx, &field, None);

    fitted.assert_matches(camera);
}

#[test]
fn fits_the_newer_to_older_direction_as_the_same_model() {
    let client = test_client();
    let ctx = fit_ctx(&client, 4, 2);
    let camera = Similarity::about_centre(-1.5, 0.99, -8.0, 3.0);
    let backward = camera.inverse();
    let field = camera_field(&ctx, backward);
    let other = random_field(&ctx, 17);
    let pairs = upload_pair(&client, &ctx, &other, &field);

    let fitted = fit_pairs(&client, &ctx, &pairs, 1, None, None);

    fitted.assert_matches(camera);
}

#[test]
fn rejects_an_independently_moving_block_group() {
    let client = test_client();
    let ctx = fit_ctx(&client, 4, 2);
    let camera = Similarity::about_centre(1.5, 1.0, 5.0, 2.0);
    let field = field_with_independent_group(&ctx, camera);

    let fitted = fit_forward(&client, &ctx, &field, None);

    fitted.assert_matches(camera);
}

#[test]
fn a_uniform_zero_field_fits_identity() {
    let client = test_client();
    let ctx = fit_ctx(&client, 4, 2);
    let field = vec![[0, 0]; block_count(&ctx)];

    let fitted = fit_forward(&client, &ctx, &field, None);

    fitted.assert_identity_fallback();
}

#[test]
fn a_nonsense_field_falls_back_to_identity() {
    let client = test_client();
    let ctx = fit_ctx(&client, 4, 2);
    let field = random_field(&ctx, 5);

    let fitted = fit_forward(&client, &ctx, &field, None);

    fitted.assert_identity_fallback();
}

#[test]
fn a_static_border_band_does_not_anchor_the_fit() {
    let client = test_client();
    let ctx = fit_ctx(&client, 4, 2);
    let camera = Similarity::about_centre(0.0, 1.0, 18.0, 3.0);
    let mut field = camera_field(&ctx, camera);
    // The bands cover most of the frame, so only finding them as bars keeps them from winning.
    let band_rows = 20;
    for (block, vector) in field.iter_mut().enumerate() {
        let row = block as u32 / ctx.blocks_x;
        if row < band_rows || row >= ctx.blocks_y - band_rows {
            *vector = [0, 0];
        }
    }

    let fitted = fit_forward(&client, &ctx, &field, None);

    fitted.assert_matches(camera);
}

#[test]
fn accepts_a_roll_inside_15_degrees() {
    let client = test_client();
    let ctx = fit_ctx(&client, 4, 2);
    let camera = Similarity::about_centre(14.0, 1.0, 0.0, 0.0);
    let field = camera_field(&ctx, camera);

    let fitted = fit_forward(&client, &ctx, &field, None);

    fitted.assert_matches(camera);
}

#[test]
fn fails_plausibility_beyond_15_degrees() {
    let client = test_client();
    let ctx = fit_ctx(&client, 4, 2);
    let camera = Similarity::about_centre(20.0, 1.0, 0.0, 0.0);
    let field = camera_field(&ctx, camera);

    let fitted = fit_forward(&client, &ctx, &field, None);

    fitted.assert_identity_fallback();
}

/// A still field with a centred group of blocks moving `(6, -2)` over about `share` of the frame.
fn still_field_with_moving_group(ctx: &MotionCtx, share: f64) -> Vec<[i32; 2]> {
    let side = share.sqrt();
    let group_cols = (ctx.blocks_x as f64 * side).round() as u32;
    let group_rows = (ctx.blocks_y as f64 * side).round() as u32;
    let first_col = (ctx.blocks_x - group_cols) / 2;
    let first_row = (ctx.blocks_y - group_rows) / 2;
    let cols = first_col..first_col + group_cols;
    let rows = first_row..first_row + group_rows;

    let mut field = vec![[0, 0]; block_count(ctx)];
    for (block, vector) in field.iter_mut().enumerate() {
        let col = block as u32 % ctx.blocks_x;
        let row = block as u32 / ctx.blocks_x;
        if cols.contains(&col) && rows.contains(&row) {
            *vector = [6, -2];
        }
    }

    field
}

#[test]
fn a_moving_subject_on_a_still_camera_falls_back_to_identity() {
    let client = test_client();
    let ctx = fit_ctx(&client, 4, 2);

    for share in [0.2, 0.3, 0.4] {
        let field = still_field_with_moving_group(&ctx, share);

        let fitted = fit_forward(&client, &ctx, &field, None);

        assert_eq!(fitted.valid, 0.0, "a {share} share was accepted as {fitted:?}");
    }
}

#[test]
fn a_moving_layer_under_a_still_top_falls_back_to_identity() {
    let client = test_client();
    let ctx = fit_ctx(&client, 4, 2);

    for share in [0.2, 0.3, 0.4, 0.5] {
        let layer_rows = (ctx.blocks_y as f64 * share).round() as u32;
        let first_row = ctx.blocks_y - layer_rows;
        let mut field = vec![[0, 0]; block_count(&ctx)];
        for (block, vector) in field.iter_mut().enumerate() {
            let row = block as u32 / ctx.blocks_x;
            if row >= first_row {
                *vector = [6, 0];
            }
        }

        let fitted = fit_forward(&client, &ctx, &field, None);

        assert_eq!(fitted.valid, 0.0, "a {share} layer was accepted as {fitted:?}");
    }
}

#[test]
fn a_moving_strip_down_one_side_falls_back_to_identity() {
    let client = test_client();
    let ctx = fit_ctx(&client, 4, 2);
    let strip_cols = (ctx.blocks_x as f64 * 0.3).round() as u32;
    let first_col = ctx.blocks_x - strip_cols;
    let mut field = vec![[0, 0]; block_count(&ctx)];
    for (block, vector) in field.iter_mut().enumerate() {
        let col = block as u32 % ctx.blocks_x;
        if col >= first_col {
            *vector = [6, 0];
        }
    }

    let fitted = fit_forward(&client, &ctx, &field, None);

    fitted.assert_identity_fallback();
}

#[test]
fn a_pillarboxed_pan_is_recovered() {
    let client = test_client();
    let ctx = fit_ctx(&client, 4, 2);
    let camera = Similarity::about_centre(0.0, 1.0, 5.0, 1.0);
    let mut field = camera_field(&ctx, camera);
    let bar_cols = (ctx.blocks_x as f64 * 0.12).round() as u32;
    for (block, vector) in field.iter_mut().enumerate() {
        let col = block as u32 % ctx.blocks_x;
        if col < bar_cols || col >= ctx.blocks_x - bar_cols {
            *vector = [0, 0];
        }
    }

    let fitted = fit_forward(&client, &ctx, &field, None);

    fitted.assert_matches(camera);
}

#[test]
fn fails_plausibility_beyond_15_percent_zoom() {
    let client = test_client();
    let ctx = fit_ctx(&client, 4, 2);
    let camera = Similarity::about_centre(0.0, 1.2, 0.0, 0.0);
    let field = camera_field(&ctx, camera);

    let fitted = fit_forward(&client, &ctx, &field, None);

    fitted.assert_identity_fallback();
}

#[test]
fn fails_plausibility_beyond_400_px_at_a_corner() {
    let client = test_client();
    let ctx = fit_ctx(&client, 4, 2);
    let camera = Similarity::about_centre(0.0, 1.0, 410.0, 0.0);
    let field = camera_field(&ctx, camera);

    let fitted = fit_forward(&client, &ctx, &field, None);

    fitted.assert_identity_fallback();
}

#[test]
fn weights_survive_high_grain() {
    let client = test_client();
    let ctx = fit_ctx(&client, 4, 2);
    let camera = Similarity::about_centre(1.0, 1.0, 9.0, -3.0);
    let mut field = camera_field(&ctx, camera);
    let threshold = thsad(ctx.blksize, 1.0);
    let mut sad = Vec::with_capacity(field.len());
    for (block, vector) in field.iter_mut().enumerate() {
        let col = block as u32 % ctx.blocks_x;
        let jitter = hashed_unit(block as u32, 33);
        let grain = threshold * (3.0 + 2.0 * jitter);

        // Half the blocks share one wrong vector, and only their SADs, three times the rest, mark them.
        if col >= 60 {
            *vector = [10, -8];
            sad.push(3.0 * grain);
        } else {
            sad.push(grain);
        }
    }

    let fitted = fit_forward(&client, &ctx, &field, Some(&sad));

    fitted.assert_matches(camera);
}

#[test]
fn a_stale_prior_does_not_bias_the_fit() {
    let client = test_client();
    let ctx = fit_ctx(&client, 4, 2);
    let camera = Similarity::about_centre(-1.0, 1.0, -3.0, 2.0);
    let stale = Similarity::about_centre(4.0, 1.0, 30.0, 0.0);
    let field = field_with_independent_group(&ctx, camera);
    let other = random_field(&ctx, 23);
    let pairs = upload_pair(&client, &ctx, &field, &other);

    let fitted = fit_pairs(&client, &ctx, &pairs, 0, None, Some(stale));

    fitted.assert_matches(camera);
}

#[test]
fn a_valid_prior_holds_the_fit_on_the_camera() {
    let client = test_client();
    let ctx = fit_ctx(&client, 4, 2);
    let camera = Similarity::about_centre(1.0, 1.0, 9.0, -3.0);
    let previous = Similarity::about_centre(1.1, 1.0, 8.5, -3.0);
    let mut field = camera_field(&ctx, camera);
    for (block, vector) in field.iter_mut().enumerate() {
        let col = block as u32 % ctx.blocks_x;
        if col >= 48 {
            *vector = [10, -8];
        }
    }

    let other = random_field(&ctx, 29);
    let pairs = upload_pair(&client, &ctx, &field, &other);

    let without_prior = fit_pairs(&client, &ctx, &pairs, 0, None, None);
    let with_prior = fit_pairs(&client, &ctx, &pairs, 0, None, Some(previous));

    let b_error = (without_prior.b as f64 - camera.b).abs();
    assert!(
        b_error > 0.01,
        "the majority group should win without a prior: {without_prior:?}"
    );
    with_prior.assert_matches(camera);
}

#[test]
fn the_fit_is_bit_identical_across_runs() {
    let client = test_client();
    let ctx = fit_ctx(&client, 4, 2);
    let camera = Similarity::about_centre(1.5, 1.0, 5.0, 2.0);
    let field = field_with_independent_group(&ctx, camera);

    let first = fit_forward(&client, &ctx, &field, None);
    let second = fit_forward(&client, &ctx, &field, None);

    let first_bits = [first.a, first.b, first.tx, first.ty, first.valid].map(f32::to_bits);
    let second_bits = [second.a, second.b, second.tx, second.ty, second.valid].map(f32::to_bits);
    assert_eq!(first_bits, second_bits);
}

/// A smooth texture with no repeat inside the search window.
fn texture(x: f64, y: f64) -> f32 {
    let wave_one = (0.31 * x + 0.17 * y).sin();
    let wave_two = (0.13 * x - 0.29 * y).sin();
    let wave_three = (0.07 * x + 0.05 * y).sin();
    let value = 0.5 + 0.18 * wave_one + 0.14 * wave_two + 0.12 * wave_three;

    value as f32
}

/// A frame of [texture], with each pixel sampled where `model` came from in the older frame.
fn render_frame(model: Similarity) -> Vec<f32> {
    let backward = model.inverse();
    let mut frame = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let (source_x, source_y) = backward.apply(x as f64, y as f64);
            frame.push(texture(source_x, source_y));
        }
    }

    frame
}

#[test]
fn a_block_matched_roll_and_shift_follows_the_field_sign() {
    let client = test_client();
    let ctx = fit_ctx(&client, 8, 1);
    let identity = Similarity::about_centre(0.0, 1.0, 0.0, 0.0);
    let camera = Similarity::about_centre(0.3, 1.002, 2.0, -1.5);
    let older = render_frame(identity);
    let newer = render_frame(camera);

    let mut frames = older;
    frames.extend_from_slice(&newer);
    let pyramid = upload(&client, &frames);
    let neighbour_table = [1u32];
    let neighbour_bytes = u32::as_bytes(&neighbour_table);
    let neighbour_slots = client.create_from_slice(neighbour_bytes);
    let pair_words = 2 * ctx.pair_direction_stride() as usize;
    let pairs = zeroed_words(&client, pair_words);
    let placeholder = zeroed_words(&client, 1);
    let noise = zeroed_words(&client, DERIVED_WORDS as usize);

    let targets = MatchTargets {
        pyramid: &pyramid,
        frame_count: 2,
        centre_slot: 0,
        neighbour_slots: &neighbour_slots,
        neighbours: 1,
        mv_field: &pairs,
        mv_len: ctx.pair_direction_len() as usize,
        confidence: &placeholder,
        conf_len: 1,
        write_confidence: false,
    };
    run_analyse(
        &client,
        &ctx,
        WIDTH,
        HEIGHT,
        &targets,
        &noise,
        DERIVED_SAD_FLOOR,
        1.0,
    )
    .unwrap();

    let fitted = fit_pairs(&client, &ctx, &pairs, 0, None, None);

    fitted.assert_matches(camera);
}

#[test]
fn the_roll_bound_is_the_tangent_of_15_degrees() {
    let expected = 15.0f32.to_radians().tan();

    assert!((MAX_ROLL_TAN - expected).abs() < 1e-6);
}

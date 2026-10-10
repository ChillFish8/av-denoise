use cubecl::prelude::*;

use super::helpers::{R, make_client};
use crate::nl4d::kernels::{nl4d_mv_regularise, nl4d_mv_regularise_coop};
use crate::nlmeans::motion::THSAD_PIXEL;

const BLKSIZE: u32 = 16;
const STEP: u32 = 8;

#[derive(Clone, Copy, Debug)]
enum Variant {
    Serial(u32, u32),
    Cooperative(u32, u32),
}

fn run(
    width: u32,
    height: u32,
    centre: &[f32],
    neighbour: &[f32],
    mv_in: &[i32],
    lambda: f32,
) -> (Vec<i32>, Vec<f32>) {
    run_variant(
        width,
        height,
        centre,
        neighbour,
        mv_in,
        lambda,
        BLKSIZE,
        Variant::Serial(8, 8),
    )
}

/// One launch over a `blocks_x x blocks_y` grid, returning the output field and confidence.
#[expect(
    clippy::too_many_arguments,
    reason = "the helper binds the kernel's inputs and shape"
)]
fn run_variant(
    width: u32,
    height: u32,
    centre: &[f32],
    neighbour: &[f32],
    mv_in: &[i32],
    lambda: f32,
    blksize: u32,
    variant: Variant,
) -> (Vec<i32>, Vec<f32>) {
    let client = make_client();
    let blocks_x = width.div_ceil(STEP);
    let blocks_y = height.div_ceil(STEP);
    let blocks = (blocks_x * blocks_y) as usize;
    assert_eq!(mv_in.len(), 2 * blocks);

    let centre_bytes = f32::as_bytes(centre);
    let neighbour_bytes = f32::as_bytes(neighbour);
    let mv_in_bytes = i32::as_bytes(mv_in);
    let centre_buf = client.create_from_slice(centre_bytes);
    let neighbour_buf = client.create_from_slice(neighbour_bytes);
    let mv_in_buf = client.create_from_slice(mv_in_bytes);
    let mv_out = client.empty(2 * blocks * size_of::<i32>());
    let conf_out = client.empty(blocks * size_of::<f32>());
    let lambda_pixel = lambda * (blksize * blksize) as f32 * THSAD_PIXEL;
    let thsad = (blksize * blksize) as f32 * THSAD_PIXEL;

    let grid = CubeCount::new_2d(blocks_x, blocks_y);
    let (dim_x, dim_y) = match variant {
        Variant::Serial(dim_x, dim_y) | Variant::Cooperative(dim_x, dim_y) => (dim_x, dim_y),
    };
    let dim = CubeDim::new_2d(dim_x, dim_y);

    match variant {
        Variant::Serial(..) => unsafe {
            nl4d_mv_regularise::launch_unchecked::<R>(
                &client,
                grid,
                dim,
                ArrayArg::from_raw_parts(centre_buf, centre.len()),
                ArrayArg::from_raw_parts(neighbour_buf, neighbour.len()),
                ArrayArg::from_raw_parts(mv_in_buf, 2 * blocks),
                ArrayArg::from_raw_parts(mv_out.clone(), 2 * blocks),
                ArrayArg::from_raw_parts(conf_out.clone(), blocks),
                lambda_pixel,
                0.0,
                thsad,
                width,
                height,
                blksize,
                STEP,
                blocks_x,
                blocks_y,
            );
        },
        Variant::Cooperative(..) => unsafe {
            nl4d_mv_regularise_coop::launch_unchecked::<R>(
                &client,
                grid,
                dim,
                ArrayArg::from_raw_parts(centre_buf, centre.len()),
                ArrayArg::from_raw_parts(neighbour_buf, neighbour.len()),
                ArrayArg::from_raw_parts(mv_in_buf, 2 * blocks),
                ArrayArg::from_raw_parts(mv_out.clone(), 2 * blocks),
                ArrayArg::from_raw_parts(conf_out.clone(), blocks),
                lambda_pixel,
                0.0,
                thsad,
                width,
                height,
                blksize,
                STEP,
                blocks_x,
                blocks_y,
                dim_x * dim_y,
            );
        },
    }

    let mv_bytes = client.read_one(mv_out).expect("mv readback");
    let conf_bytes = client.read_one(conf_out).expect("conf readback");
    let field = i32::from_bytes(&mv_bytes)[..2 * blocks].to_vec();
    let confidence = f32::from_bytes(&conf_bytes)[..blocks].to_vec();

    (field, confidence)
}

/// A frame with distinct values everywhere.
fn textured(width: u32, height: u32, seed: u32) -> Vec<f32> {
    (0..width * height)
        .map(|index| {
            let mut hash = index
                .wrapping_mul(2654435761)
                .wrapping_add(seed.wrapping_mul(0x9E37_79B9));
            hash ^= hash >> 15;
            hash = hash.wrapping_mul(0x85EB_CA6B);
            hash ^= hash >> 13;
            0.2 + 0.6 * (hash as f32 / u32::MAX as f32)
        })
        .collect()
}

/// `neighbour(x, y) = centre(x - dx, y - dy)`, so the true vector is `(dx, dy)`.
fn shifted(centre: &[f32], width: u32, height: u32, dx: i32, dy: i32) -> Vec<f32> {
    let mut frame = vec![0.0f32; (width * height) as usize];
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let source_x = (x - dx).clamp(0, width as i32 - 1) as u32;
            let source_y = (y - dy).clamp(0, height as i32 - 1) as u32;
            frame[(y as u32 * width + x as u32) as usize] = centre[(source_y * width + source_x) as usize];
        }
    }

    frame
}

fn uniform_field(blocks: usize, vector: [i32; 2]) -> Vec<i32> {
    let mut field = Vec::with_capacity(2 * blocks);
    for _ in 0..blocks {
        field.push(vector[0]);
        field.push(vector[1]);
    }

    field
}

/// A flat centre and neighbour score the same SAD at every candidate, so an outlier in a smooth
/// field moves to the neighbourhood's median as soon as the penalty is positive.
#[test]
fn an_outlier_in_a_flat_region_moves_to_the_median() {
    let (width, height) = (64u32, 64u32);
    let blocks_x = width.div_ceil(STEP);
    let blocks_y = height.div_ceil(STEP);
    let blocks = (blocks_x * blocks_y) as usize;
    let flat = vec![0.5f32; (width * height) as usize];
    let mut field = uniform_field(blocks, [3, 1]);
    let outlier = (4 * blocks_x + 4) as usize;
    field[2 * outlier] = -6;
    field[2 * outlier + 1] = 5;

    let (output, _) = run(width, height, &flat, &flat, &field, 1.0);
    assert_eq!([output[2 * outlier], output[2 * outlier + 1]], [3, 1]);

    // Every other block already sits on its median and stays put.
    for block in 0..blocks {
        if block != outlier {
            assert_eq!(
                [output[2 * block], output[2 * block + 1]],
                [3, 1],
                "block {block} moved"
            );
        }
    }
}

/// `lambda = 0` is a plain re-score, and on a flat region ties go to the block's own vector, so
/// the outlier stays.
#[test]
fn a_zero_penalty_keeps_the_input_field_on_ties() {
    let (width, height) = (64u32, 64u32);
    let blocks_x = width.div_ceil(STEP);
    let blocks_y = height.div_ceil(STEP);
    let blocks = (blocks_x * blocks_y) as usize;
    let flat = vec![0.5f32; (width * height) as usize];
    let mut field = uniform_field(blocks, [3, 1]);
    let outlier = (4 * blocks_x + 4) as usize;
    field[2 * outlier] = -6;
    field[2 * outlier + 1] = 5;

    let (output, _) = run(width, height, &flat, &flat, &field, 0.0);
    assert_eq!(output, field);
}

/// A block whose own vector matches far better than the median keeps it, because its SAD margin
/// exceeds the penalty.
#[test]
fn a_true_boundary_block_keeps_its_vector_when_the_sad_margin_wins() {
    let (width, height) = (64u32, 64u32);
    let blocks_x = width.div_ceil(STEP);
    let blocks_y = height.div_ceil(STEP);
    let blocks = (blocks_x * blocks_y) as usize;
    let centre = textured(width, height, 1);
    let neighbour = shifted(&centre, width, height, 2, 0);

    // The field says (0, 0) everywhere except one interior block that knows the truth.
    let mut field = uniform_field(blocks, [0, 0]);
    let truthful = (4 * blocks_x + 4) as usize;
    field[2 * truthful] = 2;

    let (output, confidence) = run(width, height, &centre, &neighbour, &field, 1.0);
    assert_eq!([output[2 * truthful], output[2 * truthful + 1]], [2, 0]);
    assert!(
        confidence[truthful] > 0.9,
        "an exact match scores a high confidence, got {}",
        confidence[truthful]
    );

    // Its neighbours take (2, 0) from the adjacent candidates, since textured content beats the
    // penalty of one pixel.
    let right = truthful + 1;
    assert_eq!([output[2 * right], output[2 * right + 1]], [2, 0]);
}

/// The median rule picks the lower of two middle values, and a corner block's three-member
/// neighbourhood resolves too.
///
/// The field is flat, so only the penalty against the median decides the winner. The centre
/// block's eight neighbours split four and four between `-1` and `2`, so the lower median is `-1`.
/// Corner block `(0, 0)` has only three neighbours, so its median comes from `-1, -1, 99`.
#[test]
fn the_median_rule_picks_the_lower_of_two_middle_values() {
    let (width, height) = (24u32, 24u32);
    let blocks_x = width.div_ceil(STEP);
    let blocks_y = height.div_ceil(STEP);
    assert_eq!((blocks_x, blocks_y), (3, 3));

    let flat = vec![0.5f32; (width * height) as usize];

    #[rustfmt::skip]
    let field: Vec<i32> = vec![
        -1, 0,   -1, 0,   -1, 0,
        -1, 0,   99, 99,   2, 0,
         2, 0,    2, 0,    2, 0,
    ];

    let (output, _) = run(width, height, &flat, &flat, &field, 1.0);
    let centre = (blocks_x + 1) as usize;
    assert_eq!(
        [output[2 * centre], output[2 * centre + 1]],
        [-1, 0],
        "the centre block must take the lower median, not the upper one"
    );

    let corner = 0usize;
    assert_eq!(
        [output[2 * corner], output[2 * corner + 1]],
        [-1, 0],
        "the corner block's three-member median must resolve too"
    );
}

#[test]
fn confidence_follows_the_winning_vector() {
    let (width, height) = (64u32, 64u32);
    let blocks_x = width.div_ceil(STEP);
    let blocks_y = height.div_ceil(STEP);
    let blocks = (blocks_x * blocks_y) as usize;
    let centre = textured(width, height, 2);
    let neighbour = shifted(&centre, width, height, 1, 1);
    let field = uniform_field(blocks, [1, 1]);

    let (_, confidence) = run(width, height, &centre, &neighbour, &field, 1.0);
    let interior = (3 * blocks_x + 3) as usize;
    assert!(
        confidence[interior] > 0.99,
        "a perfect match must score ~1, got {}",
        confidence[interior]
    );

    let wrong = uniform_field(blocks, [-3, -3]);
    let (_, confidence) = run(width, height, &centre, &neighbour, &wrong, 0.0);
    assert!(
        confidence[interior] < 0.5,
        "a wrong vector on texture must score low, got {}",
        confidence[interior]
    );

    // `right` has a wrong vector and wins on its left neighbour's true one, candidate 2, so this
    // does not pass by reporting candidate 0's confidence unconditionally.
    let boundary_centre = textured(width, height, 1);
    let boundary_neighbour = shifted(&boundary_centre, width, height, 2, 0);
    let mut boundary_field = uniform_field(blocks, [0, 0]);
    let truthful = (4 * blocks_x + 4) as usize;
    boundary_field[2 * truthful] = 2;
    let right = truthful + 1;

    let (output, confidence) = run(
        width,
        height,
        &boundary_centre,
        &boundary_neighbour,
        &boundary_field,
        1.0,
    );
    assert_eq!([output[2 * right], output[2 * right + 1]], [2, 0]);
    assert!(
        confidence[right] > 0.9,
        "right wins its neighbour's exact match (candidate 2), so confidence must be high, got {}",
        confidence[right]
    );
}

/// A field with a spread of vectors, so every candidate slot holds something different.
fn varied_field(blocks: usize) -> Vec<i32> {
    (0..2 * blocks).map(|index| (index * 7 % 11) as i32 - 5).collect()
}

fn assert_variant_matches_serial(blksize: u32, variant: Variant) {
    let width = 200;
    let height = 120;
    let blocks = (width.div_ceil(STEP) * height.div_ceil(STEP)) as usize;
    let centre = textured(width, height, 1);
    let neighbour = textured(width, height, 2);
    let field = varied_field(blocks);

    let baseline = Variant::Serial(8, 8);
    let (want_field, want_conf) =
        run_variant(width, height, &centre, &neighbour, &field, 0.5, blksize, baseline);
    let (got_field, got_conf) =
        run_variant(width, height, &centre, &neighbour, &field, 0.5, blksize, variant);

    assert_eq!(
        got_field, want_field,
        "{variant:?} at blksize {blksize} picked different vectors"
    );
    for (block, (got, want)) in got_conf.iter().zip(&want_conf).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff <= 1e-4,
            "{variant:?} block {block} confidence {got} vs {want}"
        );
    }
}

#[test]
fn serial_8x4_matches_8x8() {
    assert_variant_matches_serial(16, Variant::Serial(8, 4));
    assert_variant_matches_serial(8, Variant::Serial(8, 4));
}

#[test]
fn serial_4x4_matches_8x8() {
    assert_variant_matches_serial(16, Variant::Serial(4, 4));
    assert_variant_matches_serial(8, Variant::Serial(4, 4));
}

#[test]
fn cooperative_8x4_matches_serial() {
    assert_variant_matches_serial(16, Variant::Cooperative(8, 4));
    assert_variant_matches_serial(8, Variant::Cooperative(8, 4));
}

#[test]
fn cooperative_8x8_matches_serial() {
    assert_variant_matches_serial(16, Variant::Cooperative(8, 8));
    assert_variant_matches_serial(8, Variant::Cooperative(8, 8));
}

use cubecl::prelude::*;

use super::helpers::{R, make_client};
use crate::nl4d::kernels::nl4d_mv_regularise;
use crate::nlmeans::motion::THSAD_PIXEL;

const BLKSIZE: u32 = 16;
const STEP: u32 = 8;

/// One launch over a `blocks_x x blocks_y` grid, returning the output
/// field and confidence.
fn run(
    w: u32,
    h: u32,
    centre: &[f32],
    neighbour: &[f32],
    mv_in: &[i32],
    lambda: f32,
) -> (Vec<i32>, Vec<f32>) {
    let client = make_client();
    let blocks_x = w.div_ceil(STEP);
    let blocks_y = h.div_ceil(STEP);
    let blocks = (blocks_x * blocks_y) as usize;
    assert_eq!(mv_in.len(), 2 * blocks);
    let centre_buf = client.create_from_slice(f32::as_bytes(centre));
    let neighbour_buf = client.create_from_slice(f32::as_bytes(neighbour));
    let mv_in_buf = client.create_from_slice(i32::as_bytes(mv_in));
    let mv_out = client.empty(2 * blocks * size_of::<i32>());
    let conf_out = client.empty(blocks * size_of::<f32>());
    let thsad = (BLKSIZE * BLKSIZE) as f32 * THSAD_PIXEL;

    unsafe {
        nl4d_mv_regularise::launch_unchecked::<R>(
            &client,
            CubeCount::new_2d(blocks_x, blocks_y),
            CubeDim::new_2d(8, 8),
            ArrayArg::from_raw_parts(centre_buf, centre.len()),
            ArrayArg::from_raw_parts(neighbour_buf, neighbour.len()),
            ArrayArg::from_raw_parts(mv_in_buf, 2 * blocks),
            ArrayArg::from_raw_parts(mv_out.clone(), 2 * blocks),
            ArrayArg::from_raw_parts(conf_out.clone(), blocks),
            lambda * (BLKSIZE * BLKSIZE) as f32 * THSAD_PIXEL,
            0.0,
            thsad,
            w,
            h,
            BLKSIZE,
            STEP,
            blocks_x,
            blocks_y,
        );
    }

    let mv = i32::from_bytes(&client.read_one(mv_out).expect("mv readback"))[..2 * blocks].to_vec();
    let conf = f32::from_bytes(&client.read_one(conf_out).expect("conf readback"))[..blocks].to_vec();
    (mv, conf)
}

/// A frame with distinct values everywhere.
fn textured(w: u32, h: u32, seed: u32) -> Vec<f32> {
    (0..w * h)
        .map(|i| {
            let mut x = i
                .wrapping_mul(2654435761)
                .wrapping_add(seed.wrapping_mul(0x9E37_79B9));
            x ^= x >> 15;
            x = x.wrapping_mul(0x85EB_CA6B);
            x ^= x >> 13;
            0.2 + 0.6 * (x as f32 / u32::MAX as f32)
        })
        .collect()
}

/// `neighbour(x, y) = centre(x - dx, y - dy)`, so the true vector is
/// `(dx, dy)`.
fn shifted(centre: &[f32], w: u32, h: u32, dx: i32, dy: i32) -> Vec<f32> {
    let mut out = vec![0.0f32; (w * h) as usize];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let sx = (x - dx).clamp(0, w as i32 - 1) as u32;
            let sy = (y - dy).clamp(0, h as i32 - 1) as u32;
            out[(y as u32 * w + x as u32) as usize] = centre[(sy * w + sx) as usize];
        }
    }
    out
}

fn uniform_field(blocks: usize, v: [i32; 2]) -> Vec<i32> {
    let mut f = Vec::with_capacity(2 * blocks);
    for _ in 0..blocks {
        f.push(v[0]);
        f.push(v[1]);
    }
    f
}

/// A flat centre and neighbour score the same SAD at every candidate,
/// so an outlier vector in a smooth field moves to the neighbourhood's
/// median as soon as the penalty is positive.
#[test]
fn an_outlier_in_a_flat_region_moves_to_the_median() {
    let (w, h) = (64u32, 64u32);
    let blocks_x = w.div_ceil(STEP);
    let blocks = (blocks_x * h.div_ceil(STEP)) as usize;
    let flat = vec![0.5f32; (w * h) as usize];
    let mut field = uniform_field(blocks, [3, 1]);
    let outlier = (4 * blocks_x + 4) as usize;
    field[2 * outlier] = -6;
    field[2 * outlier + 1] = 5;

    let (out, _) = run(w, h, &flat, &flat, &field, 1.0);
    assert_eq!([out[2 * outlier], out[2 * outlier + 1]], [3, 1]);
    // Every other block already sits on its median and stays put.
    for b in 0..blocks {
        if b != outlier {
            assert_eq!([out[2 * b], out[2 * b + 1]], [3, 1], "block {b} moved");
        }
    }
}

/// `lambda = 0` is a plain re-score. On a clean shift with the field
/// already correct nothing moves, and on a flat region ties go to the
/// block's own vector, so the outlier stays.
#[test]
fn a_zero_penalty_keeps_the_input_field_on_ties() {
    let (w, h) = (64u32, 64u32);
    let blocks_x = w.div_ceil(STEP);
    let blocks = (blocks_x * h.div_ceil(STEP)) as usize;
    let flat = vec![0.5f32; (w * h) as usize];
    let mut field = uniform_field(blocks, [3, 1]);
    let outlier = (4 * blocks_x + 4) as usize;
    field[2 * outlier] = -6;
    field[2 * outlier + 1] = 5;

    let (out, _) = run(w, h, &flat, &flat, &field, 0.0);
    assert_eq!(out, field);
}

/// A block whose own vector matches far better than the median keeps
/// it, because its SAD margin exceeds the penalty.
#[test]
fn a_true_boundary_block_keeps_its_vector_when_the_sad_margin_wins() {
    let (w, h) = (64u32, 64u32);
    let blocks_x = w.div_ceil(STEP);
    let blocks = (blocks_x * h.div_ceil(STEP)) as usize;
    let centre = textured(w, h, 1);
    // The whole neighbour is the centre shifted by (2, 0).
    let neighbour = shifted(&centre, w, h, 2, 0);
    // The field says (0, 0) everywhere except one interior block that
    // knows the truth.
    let mut field = uniform_field(blocks, [0, 0]);
    let truthful = (4 * blocks_x + 4) as usize;
    field[2 * truthful] = 2;

    let (out, conf) = run(w, h, &centre, &neighbour, &field, 1.0);
    assert_eq!([out[2 * truthful], out[2 * truthful + 1]], [2, 0]);
    assert!(
        conf[truthful] > 0.9,
        "an exact match scores a high confidence, got {}",
        conf[truthful]
    );
    // Its neighbours see (2, 0) among the adjacent candidates and take
    // it, since textured content beats the penalty of one pixel.
    let right = truthful + 1;
    assert_eq!([out[2 * right], out[2 * right + 1]], [2, 0]);
}

/// The median rule picks the lower of two middle values, not the upper,
/// and a corner block's three-member neighbourhood gets exercised with
/// genuinely different values along the way.
///
/// The field is flat, so every candidate scores the same zero SAD and
/// only the penalty against the median decides the winner. The centre
/// block's eight neighbours split four and four between `-1` and `2`,
/// so the lower median is `-1` while the upper median would be `2`, and
/// the kernel must land on `-1`. Block `(0, 0)` is a grid corner with
/// only three neighbours, one of which is the centre block, so its
/// median comes from the mismatched values `-1, -1, 99` rather than
/// eight identical entries.
#[test]
fn the_median_rule_picks_the_lower_of_two_middle_values() {
    let (w, h) = (24u32, 24u32);
    let blocks_x = w.div_ceil(STEP);
    let blocks_y = h.div_ceil(STEP);
    assert_eq!((blocks_x, blocks_y), (3, 3));
    let flat = vec![0.5f32; (w * h) as usize];

    #[rustfmt::skip]
    let field: Vec<i32> = vec![
        -1, 0,   -1, 0,   -1, 0,
        -1, 0,   99, 99,   2, 0,
         2, 0,    2, 0,    2, 0,
    ];

    let (out, _) = run(w, h, &flat, &flat, &field, 1.0);
    let centre = (blocks_x + 1) as usize;
    assert_eq!(
        [out[2 * centre], out[2 * centre + 1]],
        [-1, 0],
        "the centre block must take the lower median, not the upper one"
    );
    let corner = 0usize;
    assert_eq!(
        [out[2 * corner], out[2 * corner + 1]],
        [-1, 0],
        "the corner block's three-member median must resolve too"
    );
}

/// Confidence is recomputed for the winner, not copied from the input.
#[test]
fn confidence_follows_the_winning_vector() {
    let (w, h) = (64u32, 64u32);
    let blocks_x = w.div_ceil(STEP);
    let blocks = (blocks_x * h.div_ceil(STEP)) as usize;
    let centre = textured(w, h, 2);
    let neighbour = shifted(&centre, w, h, 1, 1);
    let field = uniform_field(blocks, [1, 1]);

    let (_, conf) = run(w, h, &centre, &neighbour, &field, 1.0);
    let interior = (3 * blocks_x + 3) as usize;
    assert!(
        conf[interior] > 0.99,
        "a perfect match must score ~1, got {}",
        conf[interior]
    );

    let wrong = uniform_field(blocks, [-3, -3]);
    let (_, conf) = run(w, h, &centre, &neighbour, &wrong, 0.0);
    assert!(
        conf[interior] < 0.5,
        "a wrong vector on texture must score low, got {}",
        conf[interior]
    );

    // A block whose winner is not its own vector, so this test does not
    // pass merely by reporting candidate 0's confidence unconditionally.
    // `right` sits beside a block that knows the true shift, and its own
    // vector is wrong, so it wins on its left neighbour's vector, which is
    // candidate 2.
    let boundary_centre = textured(w, h, 1);
    let boundary_neighbour = shifted(&boundary_centre, w, h, 2, 0);
    let mut boundary_field = uniform_field(blocks, [0, 0]);
    let truthful = (4 * blocks_x + 4) as usize;
    boundary_field[2 * truthful] = 2;
    let right = truthful + 1;

    let (out, conf) = run(w, h, &boundary_centre, &boundary_neighbour, &boundary_field, 1.0);
    assert_eq!([out[2 * right], out[2 * right + 1]], [2, 0]);
    assert!(
        conf[right] > 0.9,
        "right wins its neighbour's exact match (candidate 2), so confidence must be high, got {}",
        conf[right]
    );
}

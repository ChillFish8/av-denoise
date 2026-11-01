use cubecl::prelude::*;

use crate::motion::chroma_map::{FieldLayout, Subsampling, map_luma_fields};
use crate::tune::gpu::read_i32;
use crate::tune::tests::runtime::test_client;

const STEP: u32 = 8;
const BLKSIZE: u32 = 16;

const FOUR_FOUR_FOUR: Subsampling = Subsampling {
    shift_x: 0,
    shift_y: 0,
};
const FOUR_TWO_TWO: Subsampling = Subsampling {
    shift_x: 1,
    shift_y: 0,
};
const FOUR_TWO_ZERO: Subsampling = Subsampling {
    shift_x: 1,
    shift_y: 1,
};

fn layout(width: u32, height: u32) -> FieldLayout {
    let blocks_x = width.div_ceil(STEP);
    let blocks_y = height.div_ceil(STEP);

    FieldLayout {
        blocks_x,
        blocks_y,
        step: STEP,
        blksize: BLKSIZE,
        stride: blocks_x * blocks_y * 2,
    }
}

fn block_count(layout: &FieldLayout) -> usize {
    (layout.blocks_x * layout.blocks_y) as usize
}

/// Gives every luma block of every field the vector `vector_of(field, block_col, block_row)`.
fn field_vectors(
    luma: &FieldLayout,
    fields: u32,
    vector_of: impl Fn(u32, u32, u32) -> (i32, i32),
) -> Vec<i32> {
    let mut values = Vec::new();
    for field in 0..fields {
        for row in 0..luma.blocks_y {
            for col in 0..luma.blocks_x {
                let (x, y) = vector_of(field, col, row);
                values.push(x);
                values.push(y);
            }
        }
    }

    values
}

fn map(
    luma_values: &[i32],
    luma: FieldLayout,
    chroma: FieldLayout,
    subsampling: Subsampling,
    fields: u32,
) -> Vec<i32> {
    let client = test_client();
    let luma_bytes = i32::as_bytes(luma_values);
    let luma_field = client.create_from_slice(luma_bytes);
    let chroma_len = (fields * chroma.stride) as usize;
    let zeroes = vec![0i32; chroma_len];
    let zero_bytes = i32::as_bytes(&zeroes);
    let chroma_field = client.create_from_slice(zero_bytes);

    map_luma_fields(
        &client,
        &luma_field,
        luma,
        &chroma_field,
        chroma,
        subsampling,
        fields,
    );

    read_i32(&client, &chroma_field, chroma_len).expect("the mapped field reads back")
}

fn vector_at(mapped: &[i32], chroma: &FieldLayout, field: u32, col: u32, row: u32) -> (i32, i32) {
    let block = (row * chroma.blocks_x + col) as usize;
    let index = field as usize * chroma.stride as usize + block * 2;

    (mapped[index], mapped[index + 1])
}

#[test]
fn four_four_four_mapping_is_the_identity() {
    let luma = layout(48, 40);
    let values = field_vectors(&luma, 1, |_, col, row| (col as i32 * 3 - 7, row as i32 * -5 + 2));

    let mapped = map(&values, luma, luma, FOUR_FOUR_FOUR, 1);

    assert_eq!(mapped, values);
}

#[test]
fn four_two_zero_halves_both_axes() {
    let luma = layout(64, 64);
    let chroma = layout(32, 32);
    let values = field_vectors(&luma, 1, |_, _, _| (12, -20));

    let mapped = map(&values, luma, chroma, FOUR_TWO_ZERO, 1);

    for block in 0..block_count(&chroma) {
        assert_eq!(&mapped[block * 2..block * 2 + 2], &[6, -10]);
    }
}

#[test]
fn four_two_two_halves_x_only() {
    let luma = layout(64, 64);
    let chroma = layout(32, 64);
    let values = field_vectors(&luma, 1, |_, _, row| (12, row as i32 * 10 - 30));

    let mapped = map(&values, luma, chroma, FOUR_TWO_TWO, 1);

    for row in 0..chroma.blocks_y {
        for col in 0..chroma.blocks_x {
            let expected = (6, row as i32 * 10 - 30);
            assert_eq!(
                vector_at(&mapped, &chroma, 0, col, row),
                expected,
                "block {col},{row}"
            );
        }
    }
}

#[test]
fn the_footprint_median_is_taken_per_component() {
    let luma = layout(64, 64);
    let chroma = layout(32, 32);
    let footprint_x = [[2, 4, 6], [8, 10, 12], [14, 16, 18]];
    let footprint_y = [[-40, 4, 8], [-8, 100, 0], [12, -4, 6]];
    let values = field_vectors(&luma, 1, |_, col, row| {
        if col < 3 && row < 3 {
            (
                footprint_x[row as usize][col as usize],
                footprint_y[row as usize][col as usize],
            )
        } else {
            (500, 500)
        }
    });

    let mapped = map(&values, luma, chroma, FOUR_TWO_ZERO, 1);

    // The lower median of x is 10 and of y is 4, and both are then halved.
    assert_eq!(vector_at(&mapped, &chroma, 0, 0, 0), (5, 2));
}

#[test]
fn odd_vectors_round_half_away_from_zero() {
    let luma = layout(64, 64);
    let chroma = layout(32, 32);
    let cases = [
        ((1, -1), (1, -1)),
        ((3, -3), (2, -2)),
        ((5, -7), (3, -4)),
        ((0, 2), (0, 1)),
    ];

    for (vector, expected) in cases {
        let values = field_vectors(&luma, 1, |_, _, _| vector);

        let mapped = map(&values, luma, chroma, FOUR_TWO_ZERO, 1);

        assert_eq!(
            vector_at(&mapped, &chroma, 0, 1, 1),
            expected,
            "vector {vector:?}"
        );
    }
}

#[test]
fn pair_fields_map_like_window_fields() {
    let luma = layout(64, 64);
    let chroma = layout(32, 32);
    let vector_of = |field: u32, col: u32, row: u32| {
        let x = (field * 14 + col * 2 + row * 6) as i32 - 20;
        let y = (field * 10 + col * 4) as i32 - row as i32 * 2;
        (x, y)
    };
    let values = field_vectors(&luma, 2, vector_of);

    let windowed = map(&values, luma, chroma, FOUR_TWO_ZERO, 2);

    // A pair pads each direction, so the same fields sit further apart than in a window.
    let padding = 8;
    let luma_pair = FieldLayout {
        stride: luma.stride + padding,
        ..luma
    };
    let chroma_pair = FieldLayout {
        stride: chroma.stride + padding,
        ..chroma
    };
    let mut padded = Vec::new();
    for field in 0..2 {
        let start = field * luma.stride as usize;
        padded.extend_from_slice(&values[start..start + luma.stride as usize]);
        let sentinels = std::iter::repeat_n(9999, padding as usize);
        padded.extend(sentinels);
    }

    let paired = map(&padded, luma_pair, chroma_pair, FOUR_TWO_ZERO, 2);

    for field in 0..2 {
        for row in 0..chroma.blocks_y {
            for col in 0..chroma.blocks_x {
                let expected = vector_at(&windowed, &chroma, field, col, row);
                let actual = vector_at(&paired, &chroma_pair, field, col, row);
                assert_eq!(actual, expected, "field {field} block {col},{row}");
            }
        }
    }
}

#[test]
fn ragged_edge_chroma_blocks_take_in_bounds_luma_vectors() {
    let luma = layout(37, 35);
    let chroma = layout(19, 18);
    assert_eq!((luma.blocks_x, luma.blocks_y), (5, 5));
    assert_eq!((chroma.blocks_x, chroma.blocks_y), (3, 3));
    let values = field_vectors(&luma, 1, |_, col, row| (col as i32 * 20, row as i32 * 40));

    let mapped = map(&values, luma, chroma, FOUR_TWO_ZERO, 1);

    // The last chroma block covers luma blocks 4..7, and only block 4 is inside the grid.
    assert_eq!(vector_at(&mapped, &chroma, 0, 2, 2), (40, 80));
    // The middle block covers luma blocks 2..5, and only blocks 2..=4 are inside, giving column 3.
    assert_eq!(vector_at(&mapped, &chroma, 0, 1, 1), (30, 60));
}

#[test]
fn an_even_count_of_in_bounds_luma_vectors_takes_the_lower_median() {
    let luma = layout(29, 16);
    let chroma = layout(15, 8);
    assert_eq!((luma.blocks_x, luma.blocks_y), (4, 2));
    assert_eq!((chroma.blocks_x, chroma.blocks_y), (2, 1));
    let values = field_vectors(&luma, 1, |_, col, row| (col as i32 * 20, row as i32 * 40));

    let mapped = map(&values, luma, chroma, FOUR_TWO_ZERO, 1);

    // The last chroma block covers luma columns 2..5 and rows 0..3, and columns 2..=3 and rows 0..=1 are
    // inside. Each axis has two live values, so the smaller one wins before halving.
    assert_eq!(vector_at(&mapped, &chroma, 0, 1, 0), (20, 0));
}

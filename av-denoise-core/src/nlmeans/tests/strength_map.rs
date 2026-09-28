use super::noise_curve::{SyntheticQuarter, frame_dims, quarter_at, synthetic_records, write_quarter};
use crate::collab::geometry::strength_map_dims;
use crate::nlmeans::noise::{
    NOISE_CURVE_BINS,
    NoiseCurve,
    QuarterClass,
    QuarterClasses,
    StrengthMapParams,
    classify_quarters,
    temporal_stats_record_len,
};

const SIGMA: f32 = 0.02;

/// A flat curve predicting [SIGMA] at every luma.
fn flat_curve() -> NoiseCurve {
    NoiseCurve {
        ratios: [1.0; NOISE_CURVE_BINS],
        sigma_quarter_median: SIGMA,
    }
}

/// Classes a single block whose four quarters are all `quarter`.
fn classify_one(quarter: SyntheticQuarter) -> QuarterClasses {
    let quarters = vec![quarter; 4];
    let (width, height) = frame_dims(quarters.len());
    let records = synthetic_records(&quarters);
    let curve = flat_curve();
    classify_quarters(&records, 1, width, height, &curve)
}

fn is_flat(quarter: SyntheticQuarter) -> bool {
    let classes = classify_one(quarter);
    let multipliers = classes.chroma_multipliers(2.0);
    multipliers.iter().all(|&multiplier| multiplier == 2.0)
}

fn with_flatness(fraction_of_variance: f32) -> SyntheticQuarter {
    SyntheticQuarter {
        flatness: fraction_of_variance * SIGMA * SIGMA,
        ..quarter_at(0.3, SIGMA)
    }
}

#[test]
fn a_flat_noisy_quarter_is_flat() {
    assert!(is_flat(with_flatness(0.4)));
}

#[test]
fn the_flat_cut_sits_at_0_55_of_the_quarters_own_variance() {
    assert!(is_flat(with_flatness(0.54)));
    assert!(!is_flat(with_flatness(0.56)));
}

#[test]
fn a_moving_quarter_is_not_flat() {
    let moving = SyntheticQuarter {
        mean_residual: 3.5 / 255.0,
        ..quarter_at(0.3, SIGMA)
    };
    assert!(!is_flat(moving));
}

#[test]
fn a_quarter_far_noisier_than_the_curve_is_not_flat() {
    assert!(is_flat(quarter_at(0.3, 2.4 * SIGMA)));
    assert!(!is_flat(quarter_at(0.3, 2.6 * SIGMA)));
}

#[test]
fn clipped_quarters_are_not_flat() {
    let clipped_low = SyntheticQuarter {
        luma_min: 2.0 / 255.0,
        ..quarter_at(0.3, SIGMA)
    };
    let clipped_high = SyntheticQuarter {
        luma_max: 253.0 / 255.0,
        ..quarter_at(0.3, SIGMA)
    };
    assert!(!is_flat(clipped_low));
    assert!(!is_flat(clipped_high));
}

#[test]
fn a_ragged_quarter_is_not_flat() {
    let ragged = SyntheticQuarter {
        flatness: 3.0e38,
        ..quarter_at(0.3, SIGMA)
    };
    assert!(!is_flat(ragged));
}

#[test]
fn a_noiseless_quarter_is_not_flat() {
    assert!(!is_flat(quarter_at(0.3, 0.0)));
}

#[test]
fn the_soften_fades_between_128_and_160() {
    let soften = 0.6;
    let params = StrengthMapParams {
        flat_boost: 1.5,
        shadow_soften: soften,
    };
    let lumas = [0.1, 128.0 / 255.0, 144.0 / 255.0, 160.0 / 255.0, 0.9];
    let classes: Vec<Option<QuarterClass>> = lumas
        .iter()
        .map(|&luma| Some(QuarterClass { flat: false, luma }))
        .collect();
    let quarters = QuarterClasses::from_classes(lumas.len(), 1, classes);

    let multipliers = quarters.luma_multipliers(params);

    assert_eq!(multipliers[0], soften);
    assert_eq!(multipliers[1], soften);
    assert!(
        (multipliers[2] - (soften + 1.0) / 2.0).abs() < 1e-6,
        "{multipliers:?}"
    );
    assert_eq!(multipliers[3], 1.0);
    assert_eq!(multipliers[4], 1.0);
}

#[test]
fn flat_quarters_take_the_boost_at_any_luma() {
    let params = StrengthMapParams {
        flat_boost: 1.5,
        shadow_soften: 0.65,
    };
    let classes = vec![
        Some(QuarterClass {
            flat: true,
            luma: 0.1,
        }),
        Some(QuarterClass {
            flat: true,
            luma: 0.9,
        }),
    ];
    let quarters = QuarterClasses::from_classes(2, 1, classes);

    assert_eq!(quarters.luma_multipliers(params), vec![1.5, 1.5]);
}

#[test]
fn chroma_multipliers_never_soften() {
    let classes = vec![
        Some(QuarterClass {
            flat: true,
            luma: 0.1,
        }),
        Some(QuarterClass {
            flat: false,
            luma: 0.1,
        }),
        None,
    ];
    let quarters = QuarterClasses::from_classes(3, 1, classes);

    assert_eq!(quarters.chroma_multipliers(1.5), vec![1.5, 1.0, 1.0]);
}

#[test]
fn unit_params_give_a_map_of_ones() {
    let params = StrengthMapParams {
        flat_boost: 1.0,
        shadow_soften: 1.0,
    };
    let classes = vec![
        Some(QuarterClass {
            flat: true,
            luma: 0.1,
        }),
        Some(QuarterClass {
            flat: false,
            luma: 0.1,
        }),
        None,
    ];
    let quarters = QuarterClasses::from_classes(3, 1, classes);

    assert!(params.is_identity());
    assert_eq!(quarters.luma_multipliers(params), vec![1.0, 1.0, 1.0]);
}

#[test]
fn classes_match_the_kernel_map_dims_on_ragged_frames() {
    for (width, height) in [(20u32, 16u32), (70, 54), (1920, 1080), (960, 540)] {
        let (blocks_x, blocks_y) = (width.div_ceil(16), height.div_ceil(16));
        let record_len = temporal_stats_record_len(1) as usize;
        let records = vec![0.0f32; (blocks_x * blocks_y) as usize * record_len];
        let curve = flat_curve();

        let classes = classify_quarters(&records, 1, width, height, &curve);

        let (cols, rows) = strength_map_dims(width, height);
        assert_eq!((classes.cols(), classes.rows()), (cols as usize, rows as usize));
    }
}

#[test]
fn a_quarter_past_the_frame_edge_gets_one() {
    // A 20 pixel wide frame holds a full block and a 4 pixel wide one, whose right quarters are
    // empty.
    let width = 20;
    let height = 16;
    let record_len = temporal_stats_record_len(1) as usize;
    let mut records = vec![0.0f32; 2 * record_len];
    for block in 0..2 {
        let record = &mut records[block * record_len..(block + 1) * record_len];
        let pixels = if block == 0 {
            [64.0, 64.0, 64.0, 64.0]
        } else {
            [32.0, 0.0, 32.0, 0.0]
        };
        for (quarter_index, &quarter_pixels) in pixels.iter().enumerate() {
            if quarter_pixels > 0.0 {
                let quarter = quarter_at(0.1, SIGMA);
                write_quarter(record, quarter_index, &quarter, quarter_pixels);
            }
        }
    }
    let curve = flat_curve();
    let params = StrengthMapParams {
        flat_boost: 1.5,
        shadow_soften: 0.65,
    };

    let classes = classify_quarters(&records, 1, width, height, &curve);
    let multipliers = classes.luma_multipliers(params);

    let cols = classes.cols();
    for row in 0..classes.rows() {
        assert_eq!(multipliers[row * cols + 3], 1.0, "row {row}");
    }
}

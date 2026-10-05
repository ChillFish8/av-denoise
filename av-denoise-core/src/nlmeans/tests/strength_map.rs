use super::helpers::{R, make_client};
use super::noise_curve::{
    SyntheticQuarter,
    banded_noisy_frame,
    frame_dims,
    quarter_at,
    quarters_at,
    synthetic_records,
    write_quarter,
};
use crate::bench_api::HostIo;
use crate::collab::geometry::strength_map_dims;
use crate::nlmeans::noise::{
    NOISE_CURVE_BINS,
    NoiseCurve,
    QuarterClass,
    QuarterClasses,
    QuarterTensor,
    StrengthMapParams,
    classify_quarters,
    temporal_noise_reading,
    temporal_stats_record_len,
};
use crate::nlmeans::{ChannelMode, HqParams, MotionCompensationMode, NlmDenoiser, NlmParams, PrefilterMode};

const SIGMA: f32 = 0.02;

const ORIENTED: QuarterTensor = QuarterTensor {
    xx: 1.0,
    yy: 0.1,
    xy: 0.0,
};
const ISOTROPIC: QuarterTensor = QuarterTensor {
    xx: 1.0,
    yy: 1.0,
    xy: 0.0,
};

/// A flat curve predicting [SIGMA] at every luma.
fn flat_curve() -> NoiseCurve {
    NoiseCurve {
        ratios: [1.0; NOISE_CURVE_BINS],
        sigma_quarter_median: SIGMA,
    }
}

/// Classifies a single block whose four quarters are all `quarter`.
fn classify_block(quarter: SyntheticQuarter, cut: Option<f32>) -> QuarterClasses {
    let quarters = vec![quarter; 4];
    let (width, height) = frame_dims(quarters.len());
    let records = synthetic_records(&quarters);
    let curve = flat_curve();
    classify_quarters(&records, 1, width, height, &curve, cut)
}

fn all_flat(classes: &QuarterClasses) -> bool {
    let multipliers = classes.chroma_multipliers(2.0);
    multipliers.iter().all(|&multiplier| multiplier == 2.0)
}

fn is_flat(quarter: SyntheticQuarter) -> bool {
    let classes = classify_block(quarter, None);
    all_flat(&classes)
}

fn with_flatness(fraction_of_variance: f32) -> SyntheticQuarter {
    SyntheticQuarter {
        flatness: fraction_of_variance * SIGMA * SIGMA,
        ..quarter_at(0.3, SIGMA)
    }
}

fn flat_grid(cols: usize, rows: usize) -> QuarterClasses {
    let class = Some(QuarterClass {
        flat: true,
        luma: 0.2,
    });
    let classes = vec![class; cols * rows];
    QuarterClasses::from_classes(cols, rows, classes)
}

/// A flat block whose four quarters carry a strongly horizontal structure tensor.
fn oriented_flat_block(cut: Option<f32>) -> QuarterClasses {
    let quarter = SyntheticQuarter {
        tensor_xx: 1.0,
        tensor_yy: 0.1,
        ..with_flatness(0.4)
    };
    classify_block(quarter, cut)
}

#[test]
fn a_flat_noisy_quarter_is_flat() {
    let quarter = with_flatness(0.4);
    let flat = is_flat(quarter);
    assert!(flat);
}

#[test]
fn the_flat_cut_sits_at_0_55_of_the_quarters_own_variance() {
    let below_cut = with_flatness(0.54);
    let above_cut = with_flatness(0.56);
    let below_cut_flat = is_flat(below_cut);
    let above_cut_flat = is_flat(above_cut);
    assert!(below_cut_flat);
    assert!(!above_cut_flat);
}

#[test]
fn a_moving_quarter_is_not_flat() {
    let moving = SyntheticQuarter {
        mean_residual: 3.5 / 255.0,
        ..quarter_at(0.3, SIGMA)
    };
    let flat = is_flat(moving);
    assert!(!flat);
}

#[test]
fn a_quarter_far_noisier_than_the_curve_is_not_flat() {
    let slightly_noisier = quarter_at(0.3, 2.4 * SIGMA);
    let far_noisier = quarter_at(0.3, 2.6 * SIGMA);
    let slightly_noisier_flat = is_flat(slightly_noisier);
    let far_noisier_flat = is_flat(far_noisier);
    assert!(slightly_noisier_flat);
    assert!(!far_noisier_flat);
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
    let clipped_low_flat = is_flat(clipped_low);
    let clipped_high_flat = is_flat(clipped_high);
    assert!(!clipped_low_flat);
    assert!(!clipped_high_flat);
}

#[test]
fn a_ragged_quarter_is_not_flat() {
    let ragged = SyntheticQuarter {
        flatness: 3.0e38,
        ..quarter_at(0.3, SIGMA)
    };
    let flat = is_flat(ragged);
    assert!(!flat);
}

#[test]
fn a_noiseless_quarter_is_not_flat() {
    let noiseless = quarter_at(0.3, 0.0);
    let flat = is_flat(noiseless);
    assert!(!flat);
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

    let multipliers = quarters.luma_multipliers(params);
    assert_eq!(multipliers, vec![1.5, 1.5]);
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

    let multipliers = quarters.chroma_multipliers(1.5);
    assert_eq!(multipliers, vec![1.5, 1.0, 1.0]);
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

    let multipliers = quarters.luma_multipliers(params);
    assert_eq!(multipliers, vec![1.0, 1.0, 1.0]);
}

#[test]
fn classes_match_the_kernel_map_dims_on_ragged_frames() {
    for (width, height) in [(20u32, 16u32), (70, 54), (1920, 1080), (960, 540)] {
        let (blocks_x, blocks_y) = (width.div_ceil(16), height.div_ceil(16));
        let record_len = temporal_stats_record_len(1) as usize;
        let records = vec![0.0f32; (blocks_x * blocks_y) as usize * record_len];
        let curve = flat_curve();

        let classes = classify_quarters(&records, 1, width, height, &curve, None);

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

    let classes = classify_quarters(&records, 1, width, height, &curve, None);
    let multipliers = classes.luma_multipliers(params);

    let cols = classes.cols();
    for row in 0..classes.rows() {
        assert_eq!(multipliers[row * cols + 3], 1.0, "row {row}");
    }
}

#[test]
fn a_reading_carries_classes_exactly_when_it_carries_a_curve() {
    let mut quarters = quarters_at(0.15, 0.02, 160);
    let middle_band = quarters_at(0.35, 0.01, 160);
    let bright_band = quarters_at(0.6, 0.005, 160);
    quarters.extend(middle_band);
    quarters.extend(bright_band);
    let (width, height) = frame_dims(quarters.len());
    let records = synthetic_records(&quarters);

    let with_curve = temporal_noise_reading(&records, 1, 1, width, height, true, None);
    let without_curve = temporal_noise_reading(&records, 1, 1, width, height, false, None);

    assert!(with_curve.curve.is_some());
    assert!(with_curve.classes.is_some());
    assert!(without_curve.curve.is_none());
    assert!(without_curve.classes.is_none());
}

#[test]
fn the_front_end_keeps_classes_beside_the_curve_and_resets_both() {
    let client = make_client();
    let width = 320;
    let height = 240;
    let params = NlmParams {
        temporal_radius: 2,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
            temporal_confidence: false,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
            windowed_noise_estimation: false,
        }),
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.set_luma_noise_fields(true);

    let mut classes_seen = false;
    for i in 0..12u32 {
        let frame = banded_noisy_frame(width, height, 100 + i);
        denoiser.push_frame(&frame);
        let _ = denoiser.denoise().unwrap();

        let curve_present = denoiser.current_noise_curve().is_some();
        let classes_present = denoiser.current_quarter_classes().is_some();
        assert_eq!(curve_present, classes_present, "push {i}");
        classes_seen |= classes_present;
    }

    assert!(classes_seen, "expected classes to form over the brightness ramp");

    denoiser.reset_stream_state();
    assert!(denoiser.current_quarter_classes().is_none());
}

#[test]
fn an_oriented_neighbourhood_is_vetoed() {
    let mut classes = flat_grid(3, 3);
    let tensors = vec![ORIENTED; 9];

    let counts = classes.veto_textured(&tensors, 0.5);

    let multipliers = classes.chroma_multipliers(2.0);

    assert_eq!((counts.flat, counts.vetoed), (9, 9));
    assert!(multipliers.iter().all(|&multiplier| multiplier == 1.0));
}

#[test]
fn an_isotropic_neighbourhood_stays_flat() {
    let mut classes = flat_grid(3, 3);
    let tensors = vec![ISOTROPIC; 9];

    let counts = classes.veto_textured(&tensors, 0.2);

    assert_eq!((counts.flat, counts.vetoed), (9, 0));

    let flat = all_flat(&classes);
    assert!(flat);
}

#[test]
fn the_veto_pools_the_three_by_three_neighbourhood() {
    // The centre quarter alone is oriented, but its eight isotropic neighbours dilute it.
    let mut classes = flat_grid(3, 3);
    let mut tensors = vec![ISOTROPIC; 9];
    tensors[4] = ORIENTED;

    let counts = classes.veto_textured(&tensors, 0.5);

    assert_eq!(counts.vetoed, 0);
}

#[test]
fn the_veto_skips_quarters_without_a_class() {
    // The oriented tensor sits on a quarter past the frame edge, so it must not pool in.
    let flat = Some(QuarterClass {
        flat: true,
        luma: 0.2,
    });
    let row = vec![flat, None];
    let mut classes = QuarterClasses::from_classes(2, 1, row);
    let huge_oriented = QuarterTensor {
        xx: 100.0,
        yy: 0.0,
        xy: 0.0,
    };
    let tensors = vec![ISOTROPIC, huge_oriented];

    let counts = classes.veto_textured(&tensors, 0.5);

    assert_eq!((counts.flat, counts.vetoed), (1, 0));
}

#[test]
fn a_vetoed_dark_quarter_takes_the_shadow_soften() {
    let mut classes = flat_grid(1, 1);
    let params = StrengthMapParams {
        flat_boost: 1.75,
        shadow_soften: 0.65,
    };

    classes.veto_textured(&[ORIENTED], 0.5);

    let multipliers = classes.luma_multipliers(params);
    assert_eq!(multipliers, vec![0.65]);
}

#[test]
fn a_zero_cut_vetoes_every_flat_quarter() {
    let mut classes = flat_grid(2, 2);
    let tensors = vec![ISOTROPIC; 4];

    let counts = classes.veto_textured(&tensors, 0.0);

    assert_eq!((counts.flat, counts.vetoed), (4, 4));
}

#[test]
fn a_neighbourhood_with_no_gradient_is_not_vetoed() {
    let mut classes = flat_grid(3, 3);
    let tensors = vec![QuarterTensor::default(); 9];

    let counts = classes.veto_textured(&tensors, 0.01);

    assert_eq!(counts.vetoed, 0);
}

#[test]
fn classify_quarters_vetoes_an_oriented_flat_block() {
    let classes = oriented_flat_block(Some(0.21));

    let multipliers = classes.chroma_multipliers(2.0);
    assert!(multipliers.iter().all(|&multiplier| multiplier == 1.0));
}

#[test]
fn classify_quarters_without_a_cut_leaves_an_oriented_flat_block_flat() {
    let classes = oriented_flat_block(None);

    let flat = all_flat(&classes);
    assert!(flat);
}

#[test]
fn a_cut_of_one_leaves_an_oriented_flat_block_flat() {
    let classes = oriented_flat_block(Some(1.0));

    let flat = all_flat(&classes);
    assert!(flat);
}

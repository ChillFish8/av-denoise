use crate::bench_api::HostIo;
use crate::nlmeans::motion::{
    DEFAULT_BLKSIZE,
    DEFAULT_OVERLAP,
    DEFAULT_PYRAMID_LEVELS,
    DEFAULT_SEARCH_RADIUS,
    MotionCtx,
};
use crate::nlmeans::tests::helpers::*;
use crate::nlmeans::*;

/// Builds a flat frame with a bright square at `(square_x, square_y)`.
fn frame_with_square(
    width: u32,
    height: u32,
    background: f32,
    square_x: u32,
    square_y: u32,
    square_size: u32,
    square_value: f32,
) -> Vec<f32> {
    let mut frame = vec![background; (width * height) as usize];
    for offset_y in 0..square_size {
        for offset_x in 0..square_size {
            let x = square_x + offset_x;
            let y = square_y + offset_y;
            if x < width && y < height {
                frame[(y * width + x) as usize] = square_value;
            }
        }
    }
    frame
}

#[test]
fn motion_compensation_uniform_passthrough() {
    let client = make_client();
    let width = 32;
    let height = 32;
    let frame = make_uniform_frame(width, height, 1, 0.5);

    let params = NlmParams {
        temporal_radius: 1,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::Mvtools {
            blksize: 8,
            overlap: 4,
            search_radius: 2,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        },
        hq: None,
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);
    denoiser.push_frame(&frame);
    denoiser.push_frame(&frame);
    let result = denoiser.denoise().unwrap().unwrap();

    assert_eq!(result.len(), (width * height) as usize);
    for (i, &value) in result.iter().enumerate() {
        assert!(value.is_finite(), "pixel {i}: non-finite output {value}");
        assert!(
            (value - 0.5).abs() < 1e-3,
            "pixel {i}: expected 0.5 (uniform input passthrough), got {value}"
        );
    }
}

#[test]
fn motion_compensation_with_bilateral_finite() {
    let client = make_client();
    let width = 32;
    let height = 32;
    let frame = make_uniform_frame(width, height, 1, 0.5);

    let params = NlmParams {
        temporal_radius: 1,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 1.0,
            sigma_r: 0.1,
        },
        motion_compensation: MotionCompensationMode::Mvtools {
            blksize: 8,
            overlap: 4,
            search_radius: 2,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        },
        hq: None,
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);
    denoiser.push_frame(&frame);
    denoiser.push_frame(&frame);
    let result = denoiser.denoise().unwrap().unwrap();

    assert_eq!(result.len(), (width * height) as usize);
    for (i, &value) in result.iter().enumerate() {
        assert!(value.is_finite(), "pixel {i}: non-finite output {value}");
        assert!(
            (-0.01..=1.01).contains(&value),
            "pixel {i}: out-of-range output {value}"
        );
    }
}

#[test]
fn motion_compensation_translating_square_preserves_centre() {
    let client = make_client();
    let width = 32u32;
    let height = 32u32;
    let background = 0.3;
    let square_value = 0.8;
    let square_size = 4u32;

    // The square moves 2 px diagonally per frame, so without motion compensation the temporal
    // kernel would see misaligned content. The centre frame's square sits at (14, 14).
    let previous_frame = frame_with_square(width, height, background, 12, 12, square_size, square_value);
    let centre_frame = frame_with_square(width, height, background, 14, 14, square_size, square_value);
    let next_frame = frame_with_square(width, height, background, 16, 16, square_size, square_value);

    let params = NlmParams {
        temporal_radius: 1,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::Mvtools {
            blksize: 8,
            overlap: 4,
            search_radius: 2,
            pyramid_levels: 2,
            estimation: MotionEstimation::Direct,
        },
        hq: None,
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&previous_frame);
    denoiser.push_frame(&centre_frame);
    denoiser.push_frame(&next_frame);
    let result = denoiser.denoise().unwrap().unwrap();

    assert_eq!(result.len(), (width * height) as usize);
    for (i, &value) in result.iter().enumerate() {
        assert!(value.is_finite(), "pixel {i}: non-finite output {value}");
        assert!(
            (-0.01..=1.01).contains(&value),
            "pixel {i}: out-of-range output {value}"
        );
    }

    // Integer-pixel warping can soften the square's edges, so the square's centre pixel only has
    // to stay above halfway between the background and the square.
    let halfway = (background + square_value) * 0.5;
    let centre_value = result[(15 * width + 15) as usize];
    assert!(
        centre_value > halfway,
        "centre of moving square should remain above halfway between bg ({background}) \
         and sq_val ({square_value}) (= {halfway}), got {centre_value}",
    );

    // A neighbour square warped to the wrong place would brighten the background.
    let background_value = result[(2 * width + 2) as usize];
    assert!(
        (background_value - background).abs() < 0.05,
        "background pixel (2, 2) should stay near {background}, got {background_value} \
         (MC may be warping neighbour squares into the background region)",
    );
}

/// A 1080x1080 frame at the defaults has 135x135 blocks, an odd count whose unpadded
/// per-neighbour stride is not a 32-byte multiple.
///
/// wgpu rejects a binding offset that is not a multiple of `min_storage_buffer_offset_alignment`,
/// so the second neighbour's dispatch fails unless
/// [mv_field_byte_offset](crate::nlmeans::motion::mv_field_byte_offset) pads the stride. 1920x1080
/// gives an even block count and never reaches this bug, which is why the test needs its own size.
#[test]
fn motion_compensation_1080_square_odd_block_count_dispatch_succeeds() {
    let client = make_client();
    let width = 1080u32;
    let height = 1080u32;
    let frame = make_uniform_frame(width, height, 1, 0.5);

    let params = NlmParams {
        temporal_radius: 1,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::mvtools_default(),
        hq: None,
    };
    let align = test_align();
    let motion_ctx = MotionCtx::new(params.motion_compensation, width, height, align).unwrap();
    assert_eq!(
        motion_ctx.blocks_x * motion_ctx.blocks_y,
        18225,
        "test premise: this geometry gives an odd block count"
    );

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);
    denoiser.push_frame(&frame);
    denoiser.push_frame(&frame);
    let result = denoiser.denoise().unwrap().unwrap();

    assert_eq!(result.len(), (width * height) as usize);
    for (i, &value) in result.iter().enumerate() {
        assert!(value.is_finite(), "pixel {i}: non-finite output {value}");
        assert!(
            (value - 0.5).abs() < 1e-3,
            "pixel {i}: expected 0.5 (uniform input passthrough), got {value}"
        );
    }
}

/// The same odd block count under `Chained` estimation, which binds into the pair ring as well.
/// The other `Chained` tests all have an even block count.
///
/// Pushing frames past the priming window binds both the zeroed duplicate slots and the real hops.
#[test]
fn motion_compensation_1080_square_odd_block_count_chained_dispatch_succeeds() {
    let client = make_client();
    let width = 1080u32;
    let height = 1080u32;
    let radius = 2u32;
    let frames: Vec<Vec<f32>> = (0..8)
        .map(|i| make_frame_with_noisy_region(width, height, 1, 0.5, 200 + i * 4, 200, 8, 0.8))
        .collect();

    let params = NlmParams {
        temporal_radius: radius,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::Mvtools {
            blksize: DEFAULT_BLKSIZE,
            overlap: DEFAULT_OVERLAP,
            search_radius: DEFAULT_SEARCH_RADIUS,
            pyramid_levels: DEFAULT_PYRAMID_LEVELS,
            estimation: MotionEstimation::chained_default(),
        },
        hq: None,
    };
    let align = test_align();
    let motion_ctx = MotionCtx::new(params.motion_compensation, width, height, align).unwrap();
    assert_eq!(
        motion_ctx.blocks_x * motion_ctx.blocks_y,
        18225,
        "test premise: this geometry gives an odd block count"
    );

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    assert!(
        denoiser.pair_ring_buf.is_some(),
        "test premise: Chained estimation must allocate the pair ring"
    );

    let check = |frame: &[f32]| {
        for (i, &value) in frame.iter().enumerate() {
            assert!(value.is_finite(), "pixel {i}: non-finite output {value}");
            assert!(
                (-0.01..=1.01).contains(&value),
                "pixel {i}: out-of-range output {value}"
            );
        }
    };

    let mut emitted = 0usize;
    for frame in &frames {
        denoiser.push_frame(frame);
        if let Some(result) = denoiser.denoise().unwrap() {
            check(&result);
            emitted += 1;
        }
    }

    denoiser
        .flush(|frame| {
            check(frame);
            emitted += 1;
        })
        .unwrap();

    assert_eq!(emitted, frames.len(), "expected one output per pushed frame");
}

/// HQ parameters with auto noise estimation, temporal confidence and `Chained` estimation.
fn chained_hq_params(radius: u32, refine_radius: u32) -> NlmParams {
    NlmParams {
        temporal_radius: radius,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::Mvtools {
            blksize: 8,
            overlap: 4,
            search_radius: 2,
            pyramid_levels: 2,
            estimation: MotionEstimation::Chained { refine_radius },
        },
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
            temporal_confidence: true,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
        }),
    }
}

/// Checks a `Chained` denoiser emits finite output in `0.0..=1.0` for every pushed frame.
fn chained_end_to_end_finite(radius: u32) {
    let client = make_client();
    let width = 32u32;
    let height = 32u32;

    let params = chained_hq_params(radius, 2);
    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);

    let frames: Vec<Vec<f32>> = (0..8)
        .map(|i| make_frame_with_noisy_region(width, height, 1, 0.5, 6 + i, 8, 2, 0.8))
        .collect();

    let mut emitted = 0usize;
    let check = |frame: &[f32]| {
        for (i, &value) in frame.iter().enumerate() {
            assert!(value.is_finite(), "pixel {i}: non-finite output {value}");
            assert!(
                (0.0..=1.0).contains(&value),
                "pixel {i}: out-of-range output {value}"
            );
        }
    };

    for frame in &frames {
        denoiser.push_frame(frame);
        if let Some(result) = denoiser.denoise().unwrap() {
            check(&result);
            emitted += 1;
        }
    }

    denoiser
        .flush(|frame| {
            check(frame);
            emitted += 1;
        })
        .unwrap();

    assert_eq!(emitted, frames.len(), "expected one output per pushed frame");
}

#[test]
fn chained_end_to_end_finite_r2() {
    chained_end_to_end_finite(2);
}

#[test]
fn chained_end_to_end_finite_r4() {
    chained_end_to_end_finite(4);
}

#[test]
fn direct_estimation_default_and_explicit_construction_match_bit_for_bit() {
    let client = make_client();
    let width = 32u32;
    let height = 32u32;
    let frame = make_frame_with_noisy_region(width, height, 1, 0.5, 16, 16, 4, 0.8);

    let run = |motion_compensation: MotionCompensationMode| {
        let params = NlmParams {
            temporal_radius: 1,
            search_radius: 2,
            patch_radius: 2,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::None,
            motion_compensation,
            hq: None,
        };
        let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
        denoiser.push_frame(&frame);
        denoiser.push_frame(&frame);
        denoiser.push_frame(&frame);
        denoiser.denoise().unwrap().unwrap()
    };

    let default_mode = MotionCompensationMode::mvtools_default();
    let via_default = run(default_mode);
    let explicit_mode = MotionCompensationMode::Mvtools {
        blksize: DEFAULT_BLKSIZE,
        overlap: DEFAULT_OVERLAP,
        search_radius: DEFAULT_SEARCH_RADIUS,
        pyramid_levels: DEFAULT_PYRAMID_LEVELS,
        estimation: MotionEstimation::Direct,
    };
    let via_explicit = run(explicit_mode);

    assert_eq!(
        via_default, via_explicit,
        "Direct estimation must give the same output regardless of which \
         constructor produced the MotionCompensationMode value"
    );
}

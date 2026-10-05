use cubecl::prelude::*;

use super::helpers::*;
use crate::bench_api::HostIo;
use crate::nlmeans::kernels::motion::nlm_mc_block_match_fine;
use crate::nlmeans::motion::{
    MotionCompensationMode,
    MotionCtx,
    pyramid_pixels_per_frame,
    run_analyse,
    run_confidence_for_neighbour,
    sad_noise_floor,
    thsad,
};
use crate::nlmeans::*;

/// Lays `frames` out as a single-level pyramid at the slot stride `run_pyramid_build` writes.
///
/// Slots are padded up to the storage-buffer offset alignment, so a frame does not always start
/// where the one before it ended.
fn pack_single_level_pyramid(frames: &[&[f32]], width: u32, height: u32, align: StorageAlign) -> Vec<f32> {
    let stride = pyramid_pixels_per_frame(width, height, 1, align);
    let mut data = vec![0.0f32; frames.len() * stride];
    for (slot, frame) in frames.iter().enumerate() {
        data[slot * stride..slot * stride + frame.len()].copy_from_slice(frame);
    }

    data
}

/// Runs the fine block-match kernel over one `blksize x blksize` block and returns its confidence.
///
/// One cube covers the whole frame with no seed and no search window. With `blksize = 1` this
/// isolates the confidence expression (floor subtraction, threshold, clamp) from the SAD reduction.
fn run_fine_confidence(
    blksize: u32,
    centre: &[f32],
    neighbour: &[f32],
    sad_noise_floor: f32,
    thsad: f32,
) -> f32 {
    let client = make_client();
    let level_len = (blksize * blksize) as usize;
    assert_eq!(centre.len(), level_len);
    assert_eq!(neighbour.len(), level_len);

    let centre_bytes = f32::as_bytes(centre);
    let neighbour_bytes = f32::as_bytes(neighbour);
    let centre_buf = client.create_from_slice(centre_bytes);
    let neighbour_buf = client.create_from_slice(neighbour_bytes);
    let mv_field = client.empty(2 * size_of::<i32>());
    let confidence = client.empty(size_of::<f32>());

    let grid = CubeCount::new_2d(1, 1);
    let dim = CubeDim::new_2d(8, 8);

    unsafe {
        nlm_mc_block_match_fine::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            ArrayArg::from_raw_parts(centre_buf, level_len),
            ArrayArg::from_raw_parts(neighbour_buf, level_len),
            ArrayArg::from_raw_parts(mv_field, 2),
            ArrayArg::from_raw_parts(confidence.clone(), 1),
            true,
            sad_noise_floor,
            thsad,
            blksize,
            blksize,
            blksize,
            blksize,
            0u32,
            0u32,
            1,
        );
    }

    let bytes = client.read_one(confidence).expect("confidence readback failed");
    f32::from_bytes(&bytes)[0]
}

#[test]
fn confidence_perfect_match_is_exactly_one() {
    let sigma = 0.1;
    let floor = sad_noise_floor(1, sigma);
    let threshold = thsad(1, 1.0);
    let confidence = run_fine_confidence(1, &[0.5], &[0.5], floor, threshold);
    assert_eq!(confidence, 1.0, "zero mismatch must give exactly full confidence");
}

#[test]
fn confidence_diff_within_noise_floor_is_exactly_one() {
    let sigma = 0.1;
    let floor = sad_noise_floor(1, sigma);
    let threshold = thsad(1, 1.0);
    // Half the floor, comfortably inside the "this is just noise" region.
    let diff = floor * 0.5;
    let confidence = run_fine_confidence(1, &[0.5], &[0.5 + diff], floor, threshold);
    assert_eq!(
        confidence, 1.0,
        "a diff under the noise floor must give exactly full confidence"
    );
}

#[test]
fn confidence_excess_at_thsad_is_exactly_zero() {
    let sigma = 0.1;
    let floor = sad_noise_floor(1, sigma);
    let threshold = thsad(1, 1.0);
    let diff = floor + threshold;
    let confidence = run_fine_confidence(1, &[0.5], &[0.5 + diff], floor, threshold);
    assert!(
        confidence < 1e-4,
        "excess reaching thsad must collapse confidence to ~zero, got {confidence}"
    );
}

#[test]
fn confidence_excess_beyond_thsad_is_exactly_zero() {
    let sigma = 0.1;
    let floor = sad_noise_floor(1, sigma);
    let threshold = thsad(1, 1.0);
    let diff = floor + 10.0 * threshold;
    let confidence = run_fine_confidence(1, &[0.5], &[0.5 + diff], floor, threshold);
    assert_eq!(
        confidence, 0.0,
        "a gross mismatch must collapse confidence to exactly zero"
    );
}

#[test]
fn confidence_decreases_as_excess_grows() {
    let sigma = 0.1;
    let floor = sad_noise_floor(1, sigma);
    let threshold = thsad(1, 1.0);

    let excess_fractions = [0.0f32, 0.2, 0.4, 0.6, 0.8, 1.0];
    let mut previous = f32::INFINITY;
    for &fraction in &excess_fractions {
        let diff = floor + fraction * threshold;
        let confidence = run_fine_confidence(1, &[0.5], &[0.5 + diff], floor, threshold);
        assert!(
            confidence <= previous + 1e-6,
            "confidence should be non-increasing as excess grows: excess={}·thsad gave {confidence}, \
             previous was {previous}",
            fraction,
        );
        previous = confidence;
    }

    assert!(
        previous < 1e-4,
        "excess reaching thsad must land at ~zero confidence, got {previous}"
    );
}

/// `sad_noise_floor` is calibrated so the chance SAD of two noisy copies sits at the floor on average.
///
/// The check only means something when `best_sad` is the full SAD. A racy reduction that
/// undercounts it passes without the floor doing any work.
#[test]
fn confidence_matched_noisy_content_at_blksize_16_is_near_one() {
    let blksize = 16;
    let sigma = 4.0 / 255.0;
    let centre = noisy_copy(blksize, 0.5, sigma, 30);
    let neighbour = noisy_copy(blksize, 0.5, sigma, 31);

    let floor = sad_noise_floor(blksize, sigma);
    let threshold = thsad(blksize, 1.0);
    let confidence = run_fine_confidence(blksize, &centre, &neighbour, floor, threshold);
    assert!(
        confidence > 0.9,
        "two independently-noisy copies of the same content at blksize=16 \
         should keep confidence near 1 (floor absorbs the noise), got {confidence}"
    );
}

/// A racy SAD reduction undercounts `best_sad` 64-fold at this block and cube size (about 1.6
/// instead of 102.4), which keeps confidence falsely high.
#[test]
fn confidence_mismatched_block_at_blksize_16_is_near_zero() {
    let blksize = 16;
    let sigma = 4.0 / 255.0;
    let centre = noisy_copy(blksize, 0.5, sigma, 40);
    let neighbour = noisy_copy(blksize, 0.9, sigma, 41);

    let floor = sad_noise_floor(blksize, sigma);
    let threshold = thsad(blksize, 1.0);
    let confidence = run_fine_confidence(blksize, &centre, &neighbour, floor, threshold);
    assert!(
        confidence < 0.1,
        "a block with a genuinely different base level (0.5 vs 0.9) should \
         collapse confidence toward 0, got {confidence}"
    );
}

/// Pins the bug where the confidence floor was sized from the raw input sigma instead of the
/// prefiltered one.
///
/// Both pairs carry the small residual noise of a cleaned reference. Under the raw-sigma floor the
/// occluded pair reads as confident as the matched one. Under a residual-scale floor only the
/// matched pair stays confident.
#[test]
fn confidence_discriminates_occluded_from_matched_at_prefilter_scale_noise() {
    let blksize = 16;
    // A prefiltered reference's residual noise, an order of magnitude below a typical raw sigma.
    let residual_sigma = 0.002f32;
    let raw_sigma = 0.02f32;
    let threshold = thsad(blksize, 1.0);

    let matched_centre = noisy_copy(blksize, 0.5, residual_sigma, 50);
    let matched_neighbour = noisy_copy(blksize, 0.5, residual_sigma, 51);
    // A base-level shift that is small next to a raw floor but large next to the residual noise.
    let occluded_centre = noisy_copy(blksize, 0.5, residual_sigma, 52);
    let occluded_neighbour = noisy_copy(blksize, 0.518, residual_sigma, 53);

    let raw_floor = sad_noise_floor(blksize, raw_sigma);
    let matched_conf_raw_floor =
        run_fine_confidence(blksize, &matched_centre, &matched_neighbour, raw_floor, threshold);
    let occluded_conf_raw_floor = run_fine_confidence(
        blksize,
        &occluded_centre,
        &occluded_neighbour,
        raw_floor,
        threshold,
    );
    assert!(
        matched_conf_raw_floor > 0.99,
        "matched pair under the raw-sigma floor should already read as confident, \
         got {matched_conf_raw_floor}"
    );
    assert!(
        occluded_conf_raw_floor > 0.9,
        "reproduces the bug: an oversized raw-sigma floor swamps thsad and clamps even the \
         occluded pair near confidence 1, indistinguishable from the matched pair; \
         got {occluded_conf_raw_floor}"
    );

    let residual_floor = sad_noise_floor(blksize, residual_sigma);
    let matched_conf_residual_floor = run_fine_confidence(
        blksize,
        &matched_centre,
        &matched_neighbour,
        residual_floor,
        threshold,
    );
    let occluded_conf_residual_floor = run_fine_confidence(
        blksize,
        &occluded_centre,
        &occluded_neighbour,
        residual_floor,
        threshold,
    );
    assert!(
        matched_conf_residual_floor > 0.9,
        "matched pair should stay confident under the residual-scale floor too, not penalised \
         just because the floor shrank; got {matched_conf_residual_floor}"
    );
    assert!(
        occluded_conf_residual_floor < 0.5,
        "occluded pair should no longer be clamped once the floor is sized to the actual \
         (small) content noise instead of raw sigma; got {occluded_conf_residual_floor}"
    );
}

/// The pyramid is packed by hand as one level instead of built by `run_pyramid_build`.
#[test]
fn run_analyse_fills_confidence_buffer() {
    let client = make_client();
    let width = 16;
    let height = 16;
    let frame_count = 2;

    let mode = MotionCompensationMode::Mvtools {
        blksize: 8,
        overlap: 4,
        search_radius: 2,
        pyramid_levels: 1,
        estimation: MotionEstimation::Direct,
    };
    let align = test_align();
    let mc = MotionCtx::new(mode, width, height, align).unwrap();

    let frame0 = noisy_copy(width, 0.5, 4.0 / 255.0, 10);
    let frame1 = noisy_copy(width, 0.5, 4.0 / 255.0, 11);
    let pyramid_data = pack_single_level_pyramid(&[&frame0, &frame1], width, height, align);
    let pyramid_bytes = f32::as_bytes(&pyramid_data);
    let pyramid = client.create_from_slice(pyramid_bytes);

    let mv_field_bytes = mc.mv_slots_per_neighbour() * 2 * size_of::<i32>();
    let mv_field = client.empty(mv_field_bytes);
    let sentinel = vec![-1.0f32; mc.mv_slots_per_neighbour()];
    let sentinel_bytes = f32::as_bytes(&sentinel);
    let confidence = client.create_from_slice(sentinel_bytes);

    let floor = sad_noise_floor(mc.blksize, 4.0 / 255.0);
    let threshold = thsad(mc.blksize, 1.0);

    run_analyse::<R>(
        &client,
        &mc,
        width,
        height,
        frame_count,
        0,
        1,
        0,
        &pyramid,
        &mv_field,
        &confidence,
        true,
        floor,
        threshold,
    )
    .expect("run_analyse dispatch failed");

    let bytes = client.read_one(confidence).expect("confidence readback failed");
    let data = f32::from_bytes(&bytes);
    assert_eq!(data.len(), mc.mv_slots_per_neighbour());
    for (i, &value) in data.iter().enumerate() {
        assert!(value.is_finite(), "block {i}: non-finite confidence {value}");
        assert!(
            (0.0..=1.0).contains(&value),
            "block {i}: out-of-range confidence {value}"
        );
        assert_ne!(
            value, -1.0,
            "block {i}: confidence left at the sentinel, kernel didn't write it"
        );
    }
}

#[test]
fn run_confidence_for_neighbour_fills_confidence_buffer() {
    let client = make_client();
    let width = 16;
    let height = 16;
    let frame_count = 2;

    let align = test_align();
    let ctx = MotionCtx::confidence_only(width, height, align);

    let frame0 = noisy_copy(width, 0.5, 4.0 / 255.0, 20);
    let frame1 = noisy_copy(width, 0.5, 4.0 / 255.0, 21);
    let pyramid_data = pack_single_level_pyramid(&[&frame0, &frame1], width, height, align);
    let pyramid_bytes = f32::as_bytes(&pyramid_data);
    let pyramid = client.create_from_slice(pyramid_bytes);

    let mv_scratch_bytes = ctx.mv_slots_per_neighbour() * 2 * size_of::<i32>();
    let mv_scratch = client.empty(mv_scratch_bytes);
    let sentinel = vec![-1.0f32; ctx.mv_slots_per_neighbour()];
    let sentinel_bytes = f32::as_bytes(&sentinel);
    let confidence = client.create_from_slice(sentinel_bytes);

    let floor = sad_noise_floor(ctx.blksize, 4.0 / 255.0);
    let threshold = thsad(ctx.blksize, 1.0);

    run_confidence_for_neighbour::<R>(
        &client,
        &ctx,
        width,
        height,
        frame_count,
        0,
        1,
        0,
        &pyramid,
        &mv_scratch,
        &confidence,
        floor,
        threshold,
    )
    .expect("run_confidence_for_neighbour dispatch failed");

    let bytes = client.read_one(confidence).expect("confidence readback failed");
    let data = f32::from_bytes(&bytes);
    assert_eq!(data.len(), ctx.mv_slots_per_neighbour());
    for (i, &value) in data.iter().enumerate() {
        assert!(value.is_finite(), "block {i}: non-finite confidence {value}");
        assert!(
            (0.0..=1.0).contains(&value),
            "block {i}: out-of-range confidence {value}"
        );
        assert_ne!(
            value, -1.0,
            "block {i}: confidence left at the sentinel, kernel didn't write it"
        );
    }
}

#[test]
fn confidence_buf_filled_without_motion_compensation() {
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
        motion_compensation: MotionCompensationMode::None,
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: Some(4.0 / 255.0),
            temporal_confidence: true,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
            windowed_noise_estimation: false,
        }),
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);
    denoiser.push_frame(&frame);
    denoiser.push_frame(&frame);
    denoiser.denoise().unwrap();

    assert!(
        denoiser.mc_ctx.is_none(),
        "this test exercises the no-MC confidence path"
    );
    let ctx = denoiser
        .confidence_ctx
        .as_ref()
        .expect("confidence_ctx must be allocated");
    let handle = denoiser
        .confidence_buf
        .as_ref()
        .expect("confidence_buf must be allocated")
        .clone();
    let bytes = denoiser
        .client
        .read_one(handle)
        .expect("confidence readback failed");
    let data = f32::from_bytes(&bytes);

    assert_eq!(data.len(), 2 * ctx.mv_slots_per_neighbour());
    for (i, &value) in data.iter().enumerate() {
        assert!(value.is_finite(), "block {i}: non-finite confidence {value}");
        assert!(
            (0.0..=1.0).contains(&value),
            "block {i}: out-of-range confidence {value}"
        );
    }
}

#[test]
fn confidence_buf_filled_with_motion_compensation() {
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
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: Some(4.0 / 255.0),
            temporal_confidence: true,
            thsad_scale: 1.0,
            sigma_scale: 1.0,
            windowed_noise_estimation: false,
        }),
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);
    denoiser.push_frame(&frame);
    denoiser.push_frame(&frame);
    denoiser.denoise().unwrap();

    let mc = denoiser
        .mc_ctx
        .as_ref()
        .expect("mc_ctx must be allocated when MC is active");
    assert!(
        denoiser.confidence_ctx.is_none(),
        "confidence_ctx is only for the no-MC path; MC-active reuses mc_ctx"
    );
    let handle = denoiser
        .confidence_buf
        .as_ref()
        .expect("confidence_buf must be allocated")
        .clone();
    let bytes = denoiser
        .client
        .read_one(handle)
        .expect("confidence readback failed");
    let data = f32::from_bytes(&bytes);

    assert_eq!(data.len(), 2 * mc.mv_slots_per_neighbour());
    for (i, &value) in data.iter().enumerate() {
        assert!(value.is_finite(), "block {i}: non-finite confidence {value}");
        assert!(
            (0.0..=1.0).contains(&value),
            "block {i}: out-of-range confidence {value}"
        );
    }
}

/// The fine block-match kernel still runs for the motion vectors but takes a placeholder confidence
/// buffer.
#[test]
fn confidence_buf_absent_with_motion_compensation_and_no_hq() {
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
    denoiser.denoise().unwrap();

    assert!(
        denoiser.mc_ctx.is_some(),
        "this test exercises the MC-active, HQ-off path"
    );
    assert!(
        denoiser.confidence_buf.is_none(),
        "confidence_buf must stay absent without HQ, even with MC active"
    );
}

/// Confidence weighting follows the flag even when motion compensation already supplies block geometry.
#[test]
fn confidence_buf_absent_with_motion_compensation_when_temporal_confidence_disabled() {
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
        hq: Some(HqParams {
            temporal_confidence: false,
            ..HqParams::with_sigma(4.0 / 255.0)
        }),
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, width, height);
    denoiser.push_frame(&frame);
    denoiser.push_frame(&frame);
    denoiser.push_frame(&frame);
    denoiser.denoise().unwrap();

    assert!(
        denoiser.mc_ctx.is_some(),
        "this test exercises the MC-active path with confidence explicitly disabled"
    );
    assert!(
        denoiser.confidence_ctx.is_none(),
        "confidence_ctx is only for the no-MC path"
    );
    assert!(
        denoiser.confidence_buf.is_none(),
        "confidence_buf must stay absent when temporal_confidence is off, even with MC active"
    );
}

#[test]
fn confidence_buf_absent_without_mc_or_hq() {
    let client = make_client();
    let params = NlmParams::default();
    let denoiser = NlmDenoiser::<R>::new(&client, params, 16, 16);

    assert!(denoiser.confidence_ctx.is_none());
    assert!(denoiser.confidence_buf.is_none());
    assert!(denoiser.confidence_pyramid.is_none());
    assert!(denoiser.confidence_mv_scratch.is_none());
}

#[test]
fn confidence_buf_absent_when_temporal_confidence_disabled() {
    let client = make_client();
    let params = NlmParams {
        temporal_radius: 1,
        hq: Some(HqParams {
            temporal_confidence: false,
            ..HqParams::with_sigma(4.0 / 255.0)
        }),
        ..NlmParams::default()
    };
    let denoiser = NlmDenoiser::<R>::new(&client, params, 16, 16);

    assert!(denoiser.confidence_ctx.is_none());
    assert!(denoiser.confidence_buf.is_none());
}

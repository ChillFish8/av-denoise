use cubecl::prelude::*;

use super::helpers::*;
use crate::nlmeans::kernels::motion::nlm_mc_block_match_fine;
use crate::nlmeans::motion::{
    MotionCompensationMode,
    MotionCtx,
    run_analyse,
    run_confidence_for_neighbour,
    sad_noise_floor,
    thsad,
};
use crate::nlmeans::*;

/// Runs the fine block-match kernel over a single `blksize × blksize`
/// block (one cube covers the whole frame, no seed, no search window)
/// and returns the resulting confidence score. Isolates the confidence
/// expression from the search behaviour already covered by
/// `tests::motion_compensation`.
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

    let centre_buf = client.create_from_slice(f32::as_bytes(centre));
    let neighbour_buf = client.create_from_slice(f32::as_bytes(neighbour));
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
            1,
        );
    }

    let bytes = client.read_one(confidence).expect("confidence readback failed");
    f32::from_bytes(&bytes)[0]
}

// NOTE on `blksize` below. Tests (a)-(e) use `blksize = 1` (a
// single-pixel block) rather than a realistic multi-pixel block size.
// This isolates the confidence *expression* (floor subtraction,
// threshold, clamp) from the SAD reduction itself, which has its own
// dedicated coverage. A prior version of this SAD reduction had every
// thread in the cube accumulate into the *same* `SharedMemory` scratch
// slot via a plain `+=` with no atomics and no per-thread
// partial/reduce split, a data race whenever more than one thread
// contributed to a block. That's now fixed (see `block_match.rs`'s
// candidate-parallel reduction and `tests::motion_compensation`'s
// `exact_sad_*`/`argmin_*` tests, which exercise it directly at a
// realistic multi-pixel `blksize`). `blksize = 1` here is kept for
// isolation, not as a workaround.

/// Zero mismatch, with a floor present. Excess clamps to zero, so
/// confidence must be exactly 1 regardless of how large the floor is.
#[test]
fn confidence_perfect_match_is_exactly_one() {
    let sigma = 0.1;
    let floor = sad_noise_floor(1, sigma);
    let th = thsad(1, 1.0);
    let confidence = run_fine_confidence(1, &[0.5], &[0.5], floor, th);
    assert_eq!(confidence, 1.0, "zero mismatch must give exactly full confidence");
}

/// A per-pixel diff at or below the noise floor must not cost any
/// confidence. The excess clamps to zero either way.
#[test]
fn confidence_diff_within_noise_floor_is_exactly_one() {
    let sigma = 0.1;
    let floor = sad_noise_floor(1, sigma);
    let th = thsad(1, 1.0);
    // Half the floor, comfortably inside the "this is just noise" region.
    let diff = floor * 0.5;
    let confidence = run_fine_confidence(1, &[0.5], &[0.5 + diff], floor, th);
    assert_eq!(
        confidence, 1.0,
        "a diff under the noise floor must give exactly full confidence"
    );
}

/// A mismatch whose excess (over the floor) reaches `thsad` must
/// collapse confidence to exactly zero, matching the documented
/// "0 at `S ≥ thsad`" behaviour.
#[test]
fn confidence_excess_at_thsad_is_exactly_zero() {
    let sigma = 0.1;
    let floor = sad_noise_floor(1, sigma);
    let th = thsad(1, 1.0);
    let diff = floor + th;
    let confidence = run_fine_confidence(1, &[0.5], &[0.5 + diff], floor, th);
    assert!(
        confidence < 1e-4,
        "excess reaching thsad must collapse confidence to ~zero, got {confidence}"
    );
}

/// A mismatch far beyond `thsad` must also collapse to zero (the
/// clamp, not just a small-but-positive value).
#[test]
fn confidence_excess_beyond_thsad_is_exactly_zero() {
    let sigma = 0.1;
    let floor = sad_noise_floor(1, sigma);
    let th = thsad(1, 1.0);
    let diff = floor + 10.0 * th;
    let confidence = run_fine_confidence(1, &[0.5], &[0.5 + diff], floor, th);
    assert_eq!(
        confidence, 0.0,
        "a gross mismatch must collapse confidence to exactly zero"
    );
}

/// Confidence must decrease monotonically (MDegrain-style) as the
/// excess over the floor grows from 0 to `thsad`.
#[test]
fn confidence_decreases_as_excess_grows() {
    let sigma = 0.1;
    let floor = sad_noise_floor(1, sigma);
    let th = thsad(1, 1.0);

    let excess_fractions = [0.0f32, 0.2, 0.4, 0.6, 0.8, 1.0];
    let mut prev = f32::INFINITY;
    for &frac in &excess_fractions {
        let diff = floor + frac * th;
        let confidence = run_fine_confidence(1, &[0.5], &[0.5 + diff], floor, th);
        assert!(
            confidence <= prev + 1e-6,
            "confidence should be non-increasing as excess grows: excess={}·thsad gave {confidence}, \
             previous was {prev}",
            frac,
        );
        prev = confidence;
    }
    assert!(
        prev < 1e-4,
        "excess reaching thsad must land at ~zero confidence, got {prev}"
    );
}

/// Realistic-blksize matched case. Two independently-noisy copies of
/// the same clean content at `blksize = 16` (the library default),
/// exercising the full multi-thread SAD reduction rather than the
/// single-pixel isolation above. `sad_noise_floor` is calibrated so
/// the SAD two noisy copies produce by chance sits at the floor on
/// average, so confidence should stay high (the floor "absorbs" the
/// noise) rather than collapsing just because real content is noisy.
/// Before the SAD-reduction race fix, every thread's contribution to a
/// shared candidate slot raced, undercounting `best_sad` and making
/// this pass "by accident" rather than because the formula is
/// discriminating. Now it holds because `best_sad` is the true SAD.
#[test]
fn confidence_matched_noisy_content_at_blksize_16_is_near_one() {
    let blksize = 16;
    let sigma = 4.0 / 255.0;
    let centre = noisy_copy(blksize, 0.5, sigma, 30);
    let neighbour = noisy_copy(blksize, 0.5, sigma, 31);

    let floor = sad_noise_floor(blksize, sigma);
    let th = thsad(blksize, 1.0);
    let confidence = run_fine_confidence(blksize, &centre, &neighbour, floor, th);
    assert!(
        confidence > 0.9,
        "two independently-noisy copies of the same content at blksize=16 \
         should keep confidence near 1 (floor absorbs the noise), got {confidence}"
    );
}

/// Realistic-blksize mismatched case. Same noise characteristics but a
/// clearly different underlying signal (a different base level), so
/// the true per-pixel diff is far larger than noise alone. Confidence
/// must collapse toward 0, not sit artificially high the way it would
/// under the pre-fix reduction race (which undercounted SAD by
/// roughly two orders of magnitude at this blksize/cube-dim
/// combination, `best_sad ≈ 2.4` instead of `≈ 153.6` for a uniform
/// 0.6 mismatch, hiding all but the most extreme real mismatches).
#[test]
fn confidence_mismatched_block_at_blksize_16_is_near_zero() {
    let blksize = 16;
    let sigma = 4.0 / 255.0;
    let centre = noisy_copy(blksize, 0.5, sigma, 40);
    let neighbour = noisy_copy(blksize, 0.9, sigma, 41);

    let floor = sad_noise_floor(blksize, sigma);
    let th = thsad(blksize, 1.0);
    let confidence = run_fine_confidence(blksize, &centre, &neighbour, floor, th);
    assert!(
        confidence < 0.1,
        "a block with a genuinely different base level (0.5 vs 0.9) should \
         collapse confidence toward 0, got {confidence}"
    );
}

/// `run_analyse` (the with-MC path) must fill the confidence buffer
/// with real, in-range values, not leave it at whatever sentinel the
/// caller pre-seeded it with. Pyramid data is supplied directly rather
/// than built via `run_pyramid_build`, since a single-level pyramid is
/// just its two frames concatenated (`[frame0][frame1]`).
#[test]
fn run_analyse_fills_confidence_buffer() {
    let client = make_client();
    let width = 16;
    let height = 16;
    let frame_count = 2;

    let mc = MotionCtx::new(
        MotionCompensationMode::Mvtools {
            blksize: 8,
            overlap: 4,
            search_radius: 2,
            pyramid_levels: 1,
            estimation: MotionEstimation::Direct,
        },
        width,
        height,
    )
    .unwrap();

    let frame0 = noisy_copy(width, 0.5, 4.0 / 255.0, 10);
    let frame1 = noisy_copy(width, 0.5, 4.0 / 255.0, 11);
    let mut pyramid_data = frame0;
    pyramid_data.extend(frame1);
    let pyramid = client.create_from_slice(f32::as_bytes(&pyramid_data));

    let mv_field = client.empty(mc.mv_slots_per_neighbour() * 2 * size_of::<i32>());
    let sentinel = vec![-1.0f32; mc.mv_slots_per_neighbour()];
    let confidence = client.create_from_slice(f32::as_bytes(&sentinel));

    let floor = sad_noise_floor(mc.blksize, 4.0 / 255.0);
    let th = thsad(mc.blksize, 1.0);

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
        th,
    )
    .expect("run_analyse dispatch failed");

    let bytes = client.read_one(confidence).expect("confidence readback failed");
    let data = f32::from_bytes(&bytes);
    assert_eq!(data.len(), mc.mv_slots_per_neighbour());
    for (i, &v) in data.iter().enumerate() {
        assert!(v.is_finite(), "block {i}: non-finite confidence {v}");
        assert!((0.0..=1.0).contains(&v), "block {i}: out-of-range confidence {v}");
        assert_ne!(
            v, -1.0,
            "block {i}: confidence left at the sentinel, kernel didn't write it"
        );
    }
}

/// `run_confidence_for_neighbour` (the no-MC path) must fill the
/// confidence buffer the same way, using the confidence-only geometry
/// instead of a real `Mvtools` configuration.
#[test]
fn run_confidence_for_neighbour_fills_confidence_buffer() {
    let client = make_client();
    let width = 16;
    let height = 16;
    let frame_count = 2;

    let ctx = MotionCtx::confidence_only(width, height);

    let frame0 = noisy_copy(width, 0.5, 4.0 / 255.0, 20);
    let frame1 = noisy_copy(width, 0.5, 4.0 / 255.0, 21);
    let mut pyramid_data = frame0;
    pyramid_data.extend(frame1);
    let pyramid = client.create_from_slice(f32::as_bytes(&pyramid_data));

    let mv_scratch = client.empty(ctx.mv_slots_per_neighbour() * 2 * size_of::<i32>());
    let sentinel = vec![-1.0f32; ctx.mv_slots_per_neighbour()];
    let confidence = client.create_from_slice(f32::as_bytes(&sentinel));

    let floor = sad_noise_floor(ctx.blksize, 4.0 / 255.0);
    let th = thsad(ctx.blksize, 1.0);

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
        th,
    )
    .expect("run_confidence_for_neighbour dispatch failed");

    let bytes = client.read_one(confidence).expect("confidence readback failed");
    let data = f32::from_bytes(&bytes);
    assert_eq!(data.len(), ctx.mv_slots_per_neighbour());
    for (i, &v) in data.iter().enumerate() {
        assert!(v.is_finite(), "block {i}: non-finite confidence {v}");
        assert!((0.0..=1.0).contains(&v), "block {i}: out-of-range confidence {v}");
        assert_ne!(
            v, -1.0,
            "block {i}: confidence left at the sentinel, kernel didn't write it"
        );
    }
}

/// End-to-end wiring check for the no-MC confidence pass. A temporal
/// HQ denoiser with `temporal_confidence: true` and no motion
/// compensation must allocate `confidence_buf` and leave it populated
/// with finite, in-range values after a submit.
#[test]
fn confidence_buf_filled_without_motion_compensation() {
    let client = make_client();
    let w = 32;
    let h = 32;
    let frame = make_uniform_frame(w, h, 1, 0.5);

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
        }),
    };

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&frame);
    d.push_frame(&frame);
    d.push_frame(&frame);
    d.denoise().unwrap();

    assert!(
        d.mc_ctx.is_none(),
        "this test exercises the no-MC confidence path"
    );
    let ctx = d
        .confidence_ctx
        .as_ref()
        .expect("confidence_ctx must be allocated");
    let handle = d
        .confidence_buf
        .as_ref()
        .expect("confidence_buf must be allocated")
        .clone();
    let bytes = d.client.read_one(handle).expect("confidence readback failed");
    let data = f32::from_bytes(&bytes);

    assert_eq!(data.len(), 2 * ctx.mv_slots_per_neighbour());
    for (i, &v) in data.iter().enumerate() {
        assert!(v.is_finite(), "block {i}: non-finite confidence {v}");
        assert!((0.0..=1.0).contains(&v), "block {i}: out-of-range confidence {v}");
    }
}

/// Same wiring check with motion compensation active. The analyse
/// fine pass must fill `confidence_buf` using `mc_ctx`'s geometry.
#[test]
fn confidence_buf_filled_with_motion_compensation() {
    let client = make_client();
    let w = 32;
    let h = 32;
    let frame = make_uniform_frame(w, h, 1, 0.5);

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
        }),
    };

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&frame);
    d.push_frame(&frame);
    d.push_frame(&frame);
    d.denoise().unwrap();

    let mc = d
        .mc_ctx
        .as_ref()
        .expect("mc_ctx must be allocated when MC is active");
    assert!(
        d.confidence_ctx.is_none(),
        "confidence_ctx is only for the no-MC path; MC-active reuses mc_ctx"
    );
    let handle = d
        .confidence_buf
        .as_ref()
        .expect("confidence_buf must be allocated")
        .clone();
    let bytes = d.client.read_one(handle).expect("confidence readback failed");
    let data = f32::from_bytes(&bytes);

    assert_eq!(data.len(), 2 * mc.mv_slots_per_neighbour());
    for (i, &v) in data.iter().enumerate() {
        assert!(v.is_finite(), "block {i}: non-finite confidence {v}");
        assert!((0.0..=1.0).contains(&v), "block {i}: out-of-range confidence {v}");
    }
}

/// Motion compensation active but HQ off (the fast path's typical MC
/// configuration). Confidence weighting requires HQ with
/// `temporal_confidence: true`, so `hq: None` must leave
/// `confidence_buf` unallocated even though `mc_ctx` is active. The
/// fine block-match kernel still runs (it always solves for the MV),
/// but with `write_confidence: false` and a placeholder buffer, so no
/// confidence-shaped allocation exists for this configuration.
#[test]
fn confidence_buf_absent_with_motion_compensation_and_no_hq() {
    let client = make_client();
    let w = 32;
    let h = 32;
    let frame = make_uniform_frame(w, h, 1, 0.5);

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

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&frame);
    d.push_frame(&frame);
    d.push_frame(&frame);
    d.denoise().unwrap();

    assert!(
        d.mc_ctx.is_some(),
        "this test exercises the MC-active, HQ-off path"
    );
    assert!(
        d.confidence_buf.is_none(),
        "confidence_buf must stay absent without HQ, even with MC active"
    );
}

/// Motion compensation active, HQ on, but `temporal_confidence: false`.
/// Confidence weighting stays gated on the flag regardless of whether
/// MC already supplies block geometry, so `confidence_buf` must be
/// absent here too, matching the no-MC gating test below.
#[test]
fn confidence_buf_absent_with_motion_compensation_when_temporal_confidence_disabled() {
    let client = make_client();
    let w = 32;
    let h = 32;
    let frame = make_uniform_frame(w, h, 1, 0.5);

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

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&frame);
    d.push_frame(&frame);
    d.push_frame(&frame);
    d.denoise().unwrap();

    assert!(
        d.mc_ctx.is_some(),
        "this test exercises the MC-active path with confidence explicitly disabled"
    );
    assert!(
        d.confidence_ctx.is_none(),
        "confidence_ctx is only for the no-MC path"
    );
    assert!(
        d.confidence_buf.is_none(),
        "confidence_buf must stay absent when temporal_confidence is off, even with MC active"
    );
}

/// No motion compensation and no HQ confidence request. Neither
/// `confidence_ctx` nor `confidence_buf` should exist. Confirms the
/// fast path allocates nothing extra.
#[test]
fn confidence_buf_absent_without_mc_or_hq() {
    let client = make_client();
    let params = NlmParams::default();
    let d = NlmDenoiser::<R>::new(&client, params, 16, 16);

    assert!(d.confidence_ctx.is_none());
    assert!(d.confidence_buf.is_none());
    assert!(d.confidence_pyramid.is_none());
    assert!(d.confidence_mv_scratch.is_none());
}

/// HQ with `temporal_confidence: false` and no motion compensation
/// must also allocate nothing. The no-MC confidence pass is the only
/// source of `confidence_buf` in the absence of MC, and it's gated on
/// the flag.
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
    let d = NlmDenoiser::<R>::new(&client, params, 16, 16);

    assert!(d.confidence_ctx.is_none());
    assert!(d.confidence_buf.is_none());
}

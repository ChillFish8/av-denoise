use cubecl::prelude::*;

use super::helpers::*;
use crate::nlmeans::kernels::motion::nlm_mc_block_match_fine;
use crate::nlmeans::motion::{DEFAULT_BLKSIZE, DEFAULT_SEARCH_RADIUS};
use crate::nlmeans::*;

/// Build a frame with a constant background and a bright square at
/// `(square_x, square_y)`. Used to simulate translating content
/// across the temporal window.
fn frame_with_square(
    w: u32,
    h: u32,
    background: f32,
    square_x: u32,
    square_y: u32,
    square_size: u32,
    square_val: f32,
) -> Vec<f32> {
    let mut frame = vec![background; (w * h) as usize];
    for dy in 0..square_size {
        for dx in 0..square_size {
            let x = square_x + dx;
            let y = square_y + dy;
            if x < w && y < h {
                frame[(y * w + x) as usize] = square_val;
            }
        }
    }
    frame
}

/// Launches `nlm_mc_block_match_fine` directly over a single block
/// covering the whole `blksize × blksize` buffer (one cube, `blocks_x
/// = blocks_y = 1`, `use_seed = 0`), returning the winning MV and
/// confidence score. Exercises the kernel's SAD reduction and argmin
/// directly, without `run_analyse`'s pyramid/geometry plumbing.
#[allow(clippy::too_many_arguments)]
fn run_fine_block_match_single_block(
    blksize: u32,
    search_radius: u32,
    centre: &[f32],
    neighbour: &[f32],
    sad_noise_floor: f32,
    thsad: f32,
) -> (i32, i32, f32) {
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
            ArrayArg::from_raw_parts(mv_field.clone(), 2),
            ArrayArg::from_raw_parts(confidence.clone(), 1),
            true,
            sad_noise_floor,
            thsad,
            blksize,
            blksize,
            blksize,
            blksize,
            search_radius,
            0u32,
            1,
            1,
        );
    }

    let mv_bytes = client.read_one(mv_field).expect("mv readback failed");
    let mv = i32::from_bytes(&mv_bytes);
    let conf_bytes = client.read_one(confidence).expect("confidence readback failed");
    let confidence = f32::from_bytes(&conf_bytes)[0];
    (mv[0], mv[1], confidence)
}

/// Recovers the fine kernel's raw `best_sad` from its confidence
/// output by inverting the confidence formula (`confidence = (thsad² -
/// S²) / (thsad² + S²)` inverts to `S = thsad · sqrt((1 - confidence) /
/// (1 + confidence))`). Requires `sad_noise_floor = 0.0` (so `excess ==
/// best_sad`) and a `thsad` comfortably larger than the expected SAD,
/// so the confidence value stays well clear of both the `≈ 1` corner
/// (catastrophic cancellation computing `1 - confidence`) and the `= 0`
/// clamp corner (all precision lost).
fn recover_sad_from_confidence(confidence: f32, thsad: f32) -> f32 {
    thsad * ((1.0 - confidence) / (1.0 + confidence)).sqrt()
}

/// Exact-SAD test. A uniform `|Δ| = d` mismatch between centre and
/// neighbour across the whole block, at zero MV (`search_radius = 0`,
/// a single candidate), must give `best_sad = blksize² · d` exactly
/// (within a tight f32 tolerance). This is the ground truth the
/// block-match kernel's SAD reduction is supposed to compute.
///
/// Before the candidate-parallel reduction fix, every thread in the
/// cube accumulated its pixel share into the *same* shared-memory
/// candidate slot with a plain `+=` and no atomics, so most
/// contributions were lost to the race. This test failed by roughly
/// two orders of magnitude on that code (see the report for the
/// measured before/after values).
#[test]
fn block_match_fine_exact_sad_uniform_mismatch() {
    let blksize = 16u32;
    let d = 0.1f32;
    let centre = vec![0.25f32; (blksize * blksize) as usize];
    let neighbour = vec![0.25f32 + d; (blksize * blksize) as usize];

    let expected_sad = (blksize * blksize) as f32 * d;
    // Comfortably above the expected SAD so the confidence readout
    // avoids both precision corners (see `recover_sad_from_confidence`).
    let thsad = 3.0 * expected_sad;

    let (_, _, confidence) = run_fine_block_match_single_block(blksize, 0, &centre, &neighbour, 0.0, thsad);
    let measured_sad = recover_sad_from_confidence(confidence, thsad);

    assert!(
        (measured_sad - expected_sad).abs() < expected_sad * 0.01,
        "uniform |Δ|={d} over a {blksize}x{blksize} block should give best_sad \
         = {expected_sad} (blksize²·d), measured {measured_sad} (confidence={confidence})",
    );
}

/// Clamps `val - delta` into `[0, limit)`. Used to build a "clean
/// shift" neighbour frame below whose out-of-range edge pixels use the
/// same clamp-to-edge convention the kernel itself applies (`clamp_i32`
/// in `block_match.rs`), even though the test's block/search geometry
/// never actually reaches those edges.
fn shift_clamped(val: i32, delta: i32, limit: i32) -> i32 {
    (val - delta).clamp(0, limit - 1)
}

/// Argmin correctness. The neighbour frame is a clean `(+2, +1)` pixel
/// shift of the centre frame's content (`neighbour(x, y) =
/// centre(x - 2, y - 1)`), so the true best match sits exactly at
/// `mv = (2, 1)`. Content is a deterministic pseudo-random pattern
/// (`noisy_copy`, reused here purely as "content-rich, no ties" filler)
/// rather than a flat value, so every other candidate in the search
/// window gives a strictly larger SAD and the argmin is unambiguous.
///
/// Runs the fine kernel over a `3×3` block grid at the library's
/// default blksize/search radius so the winning block (the centre one,
/// away from the frame edges) sees the same clamped addressing the
/// production dispatch path uses, but reads back only that one block's
/// MV. Before the reduction fix this could pick a quasi-arbitrary MV
/// because the racy SAD values didn't reflect the true per-candidate
/// cost. Recorded as a pre-fix data point in the report.
///
/// Passes `write_confidence: false` with a small placeholder
/// confidence buffer, since this test only checks the winning MV. That
/// also exercises the gated-off confidence path this fix round added.
#[test]
fn block_match_fine_argmin_finds_clean_shift() {
    let w = 64u32;
    let h = 64u32;
    let blksize = DEFAULT_BLKSIZE;
    let step = blksize;
    let search_radius = DEFAULT_SEARCH_RADIUS;
    let blocks_x = 3u32;
    let blocks_y = 3u32;

    let centre = noisy_copy(w, 0.5, 0.2, 123);
    let mut neighbour = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let sx = shift_clamped(x as i32, 2, w as i32) as u32;
            let sy = shift_clamped(y as i32, 1, h as i32) as u32;
            neighbour[(y * w + x) as usize] = centre[(sy * w + sx) as usize];
        }
    }

    let client = make_client();
    let level_len = (w * h) as usize;
    let centre_buf = client.create_from_slice(f32::as_bytes(&centre));
    let neighbour_buf = client.create_from_slice(f32::as_bytes(&neighbour));
    let mv_len = (blocks_x * blocks_y * 2) as usize;
    let mv_field = client.empty(mv_len * size_of::<i32>());
    // `write_confidence: false` below, so this placeholder never gets
    // indexed into regardless of its size.
    let confidence = client.empty(size_of::<f32>());

    let grid = CubeCount::new_2d(blocks_x, blocks_y);
    let dim = CubeDim::new_2d(8, 8);

    unsafe {
        nlm_mc_block_match_fine::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            ArrayArg::from_raw_parts(centre_buf, level_len),
            ArrayArg::from_raw_parts(neighbour_buf, level_len),
            ArrayArg::from_raw_parts(mv_field.clone(), mv_len),
            ArrayArg::from_raw_parts(confidence, 1),
            false,
            0.0,
            1.0,
            w,
            h,
            blksize,
            step,
            search_radius,
            0u32,
            blocks_x,
            blocks_y,
        );
    }

    let bytes = client.read_one(mv_field).expect("mv readback failed");
    let mv = i32::from_bytes(&bytes);
    // Middle block (bx=1, by=1). Its content and search window sit
    // `blksize` pixels away from every frame edge, well clear of the
    // `search_radius + shift` margin, so no clamped addressing is hit.
    let idx = ((1 * blocks_x + 1) * 2) as usize;
    assert_eq!(
        (mv[idx], mv[idx + 1]),
        (2, 1),
        "a clean (+2, +1) shift of the centre content should give exactly \
         MV=(2, 1) at default blksize={blksize}/search_radius={search_radius}, got ({}, {})",
        mv[idx],
        mv[idx + 1],
    );
}

/// Smoke test: a temporal denoiser with motion compensation enabled
/// must allocate, run the pyramid build, analyse, and warp dispatches
/// on every pushed frame, and produce a denoised frame that preserves
/// a uniform input. Catches plumbing mistakes (buffer allocation,
/// pyramid offsets, MC dispatch ordering) even before quality is
/// evaluated.
#[test]
fn motion_compensation_uniform_passthrough() {
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
        },
        hq: None,
    };

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&frame);
    d.push_frame(&frame);
    d.push_frame(&frame);
    let result = d.denoise().unwrap().unwrap().to_vec();

    assert_eq!(result.len(), (w * h) as usize);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "pixel {i}: non-finite output {v}");
        assert!(
            (v - 0.5).abs() < 1e-3,
            "pixel {i}: expected 0.5 (uniform input passthrough), got {v}"
        );
    }
}

/// MC + bilateral prefilter compound test: the reference clip ring
/// must also be warped, and the denoise must remain finite/in-range.
#[test]
fn motion_compensation_with_bilateral_finite() {
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
        prefilter: PrefilterMode::Bilateral {
            sigma_s: 1.0,
            sigma_r: 0.1,
        },
        motion_compensation: MotionCompensationMode::Mvtools {
            blksize: 8,
            overlap: 4,
            search_radius: 2,
            pyramid_levels: 2,
        },
        hq: None,
    };

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&frame);
    d.push_frame(&frame);
    d.push_frame(&frame);
    let result = d.denoise().unwrap().unwrap().to_vec();

    assert_eq!(result.len(), (w * h) as usize);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "pixel {i}: non-finite output {v}");
        assert!((-0.01..=1.01).contains(&v), "pixel {i}: out-of-range output {v}");
    }
}

/// Translating-square regression test. Builds a 3-frame sequence
/// where a bright square moves diagonally by 2 pixels per frame. The
/// MC-enabled temporal denoise must:
///   1. Run end-to-end without crashing.
///   2. Produce finite, in-range output.
///   3. Keep the square anchored to the centre frame's position
///      (no spatial shift introduced by misaligned temporal blending).
///
/// Catches regressions where MC mis-warps neighbours or where the
/// centre-frame copy into the compensated buffer goes wrong.
#[test]
fn motion_compensation_translating_square_preserves_centre() {
    let client = make_client();
    let w = 32u32;
    let h = 32u32;
    let bg = 0.3;
    let sq_val = 0.8;
    let sq_size = 4u32;

    // Translate the square diagonally by 2 px per frame so the
    // temporal kernel without MC would see misaligned content at the
    // same (x, y) across frames. Centre frame's square sits at (14, 14).
    let f0 = frame_with_square(w, h, bg, 12, 12, sq_size, sq_val);
    let f1 = frame_with_square(w, h, bg, 14, 14, sq_size, sq_val);
    let f2 = frame_with_square(w, h, bg, 16, 16, sq_size, sq_val);

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
        },
        hq: None,
    };

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&f0);
    d.push_frame(&f1);
    d.push_frame(&f2);
    let result = d.denoise().unwrap().unwrap().to_vec();

    assert_eq!(result.len(), (w * h) as usize);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "pixel {i}: non-finite output {v}");
        assert!((-0.01..=1.01).contains(&v), "pixel {i}: out-of-range output {v}");
    }

    // The centre of the centre-frame's square (15, 15) should still
    // look brightly square-like. We allow significant tolerance:
    // temporal blending will pull it toward the background even with
    // MC because the warping is integer-pixel and the square's edges
    // may not align perfectly across all neighbours. The assertion
    // is just that the centre pixel hasn't been smeared *below* the
    // halfway point between bg and sq_val.
    let halfway = (bg + sq_val) * 0.5;
    let centre_val = result[(15 * w + 15) as usize];
    assert!(
        centre_val > halfway,
        "centre of moving square should remain above halfway between bg ({bg}) \
         and sq_val ({sq_val}) (= {halfway}), got {centre_val}",
    );

    // Conversely, the background a few pixels away from the centre
    // frame's square must stay near `bg`. If MC mis-warped neighbour
    // squares into the wrong position, the background would brighten.
    let bg_val = result[(2 * w + 2) as usize];
    assert!(
        (bg_val - bg).abs() < 0.05,
        "background pixel (2, 2) should stay near {bg}, got {bg_val} \
         (MC may be warping neighbour squares into the background region)",
    );
}

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::helpers::*;
use crate::nlmeans::kernels::motion::{nlm_mc_block_match_coarse, nlm_mc_block_match_fine};
use crate::nlmeans::motion::{
    CHAINED_RADIUS_THRESHOLD,
    DEFAULT_BLKSIZE,
    DEFAULT_OVERLAP,
    DEFAULT_PYRAMID_LEVELS,
    DEFAULT_SEARCH_RADIUS,
    MotionCtx,
    mv_field_byte_offset,
    neighbour_idx_for_k,
    pair_byte_offset,
};
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
/// Every pixel's contribution must land in the candidate's SAD sum
/// exactly once. A racy shared-memory accumulation (multiple threads
/// `+=`-ing one candidate slot without atomics) loses most
/// contributions and undercounts the SAD by orders of magnitude.
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
/// MV. The argmin is only meaningful when each candidate's SAD
/// reflects its true cost. Corrupted per-candidate sums make the
/// winner quasi-arbitrary.
///
/// Confidence is turned off and given a small placeholder buffer, since
/// this test only checks the winning motion vector. That also covers the
/// path where the confidence write is dropped at compile time.
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
    // Confidence is turned off below, so this placeholder is never
    // indexed whatever its size.
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
        );
    }

    let bytes = client.read_one(mv_field).expect("mv readback failed");
    let mv = i32::from_bytes(&bytes);
    // Middle block (bx=1, by=1). Its content and search window sit
    // `blksize` pixels away from every frame edge, well clear of the
    // `search_radius + shift` margin, so no clamped addressing is hit.
    let (mid_bx, mid_by) = (1u32, 1u32);
    let idx = ((mid_by * blocks_x + mid_bx) * 2) as usize;
    assert_eq!(
        (mv[idx], mv[idx + 1]),
        (2, 1),
        "a clean (+2, +1) shift of the centre content should give exactly \
         MV=(2, 1) at default blksize={blksize}/search_radius={search_radius}, got ({}, {})",
        mv[idx],
        mv[idx + 1],
    );
}

/// Launches `nlm_mc_block_match_coarse` directly over a single coarse
/// block covering the whole `blksize × blksize` buffer (one cube,
/// `fine_blocks_x = fine_blocks_y = 1`, `fine_step = blksize`, so the
/// coarse block seeds exactly that one fine block), returning the
/// coarse MV it writes into `mv_field`. `level_scale` is fixed at `1`
/// so the returned MV is the raw coarse-level offset, unscaled.
fn run_coarse_block_match_single_block(
    blksize: u32,
    search_radius: u32,
    centre: &[f32],
    neighbour: &[f32],
) -> (i32, i32) {
    let client = make_client();
    let level_len = (blksize * blksize) as usize;
    assert_eq!(centre.len(), level_len);
    assert_eq!(neighbour.len(), level_len);

    let centre_buf = client.create_from_slice(f32::as_bytes(centre));
    let neighbour_buf = client.create_from_slice(f32::as_bytes(neighbour));
    let mv_field = client.empty(2 * size_of::<i32>());

    let grid = CubeCount::new_2d(1, 1);
    let dim = CubeDim::new_2d(8, 8);

    unsafe {
        nlm_mc_block_match_coarse::launch_unchecked::<R>(
            &client,
            grid,
            dim,
            ArrayArg::from_raw_parts(centre_buf, level_len),
            ArrayArg::from_raw_parts(neighbour_buf, level_len),
            ArrayArg::from_raw_parts(mv_field.clone(), 2),
            blksize,
            blksize,
            blksize,
            blksize,
            search_radius,
            1,
            1,
            1,
            blksize,
        );
    }

    let mv_bytes = client.read_one(mv_field).expect("mv readback failed");
    let mv = i32::from_bytes(&mv_bytes);
    (mv[0], mv[1])
}

/// SAD tie-break regression test (fine pass). A block lying entirely
/// inside a flat region has the same value everywhere, including at
/// every edge-clamped read the search window's candidates touch, so
/// every candidate's SAD is exactly `0.0`, an exact tie. The argmin
/// must resolve that tie to the zero-motion seed (here `(0, 0)`, since
/// `use_seed = 0`), not the window's `(-search_radius, -search_radius)`
/// corner, the first candidate the raster scan reaches.
#[test]
fn block_match_fine_flat_region_tie_resolves_to_zero_motion() {
    let blksize = 16u32;
    let search_radius = 4u32;
    let value = 0.5f32;
    let centre = vec![value; (blksize * blksize) as usize];
    let neighbour = vec![value; (blksize * blksize) as usize];

    let (mvx, mvy, confidence) =
        run_fine_block_match_single_block(blksize, search_radius, &centre, &neighbour, 0.0, 1.0);

    assert_eq!(
        (mvx, mvy),
        (0, 0),
        "a flat region gives an exact SAD tie at every candidate, which must \
         resolve to the zero-motion seed, not the window corner \
         (-{search_radius}, -{search_radius}); got ({mvx}, {mvy})",
    );
    // `best_sad == 0` here regardless of which candidate wins the tie,
    // so confidence is `1.0` either way. The MV assertion above is
    // what actually distinguishes the fix. Asserted anyway so a future
    // change to the confidence formula that breaks the exact-zero case
    // shows up here too.
    assert_eq!(
        confidence, 1.0,
        "an exact SAD=0 match should give full confidence"
    );
}

/// SAD tie-break regression test (coarse pass). Same premise as
/// `block_match_fine_flat_region_tie_resolves_to_zero_motion`, but for
/// `nlm_mc_block_match_coarse`. A corner-favouring tie here seeds every
/// fine block the coarse block covers from the wrong position, so
/// under `Chained` estimation the bias compounds across pyramid
/// levels.
#[test]
fn block_match_coarse_flat_region_tie_resolves_to_zero_motion() {
    let blksize = 16u32;
    let search_radius = 4u32;
    let value = 0.5f32;
    let centre = vec![value; (blksize * blksize) as usize];
    let neighbour = vec![value; (blksize * blksize) as usize];

    let (mvx, mvy) = run_coarse_block_match_single_block(blksize, search_radius, &centre, &neighbour);

    assert_eq!(
        (mvx, mvy),
        (0, 0),
        "a flat region gives an exact SAD tie at every candidate, which the \
         coarse pass must resolve to the zero-motion candidate, not the window \
         corner (-{search_radius}, -{search_radius}); got ({mvx}, {mvy})",
    );
}

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
            estimation: MotionEstimation::Direct,
        },
        hq: None,
    };

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&frame);
    d.push_frame(&frame);
    d.push_frame(&frame);
    let result = d
        .denoise()
        .unwrap()
        .unwrap()
        .as_f32()
        .expect("f32 denoiser")
        .to_vec();

    assert_eq!(result.len(), (w * h) as usize);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "pixel {i}: non-finite output {v}");
        assert!(
            (v - 0.5).abs() < 1e-3,
            "pixel {i}: expected 0.5 (uniform input passthrough), got {v}"
        );
    }
}

/// Motion compensation and the bilateral prefilter together.
///
/// The reference ring has to be shifted along with the input, and the
/// output has to stay finite and in range.
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
            estimation: MotionEstimation::Direct,
        },
        hq: None,
    };

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&frame);
    d.push_frame(&frame);
    d.push_frame(&frame);
    let result = d
        .denoise()
        .unwrap()
        .unwrap()
        .as_f32()
        .expect("f32 denoiser")
        .to_vec();

    assert_eq!(result.len(), (w * h) as usize);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "pixel {i}: non-finite output {v}");
        assert!((-0.01..=1.01).contains(&v), "pixel {i}: out-of-range output {v}");
    }
}

/// A three-frame sequence where a bright square moves diagonally by two
/// pixels each frame.
///
/// With motion compensation on, the denoise has to run end to end
/// without crashing, produce finite output that stays in range, and
/// leave the square exactly where the centre frame put it.
///
/// That last point is what catches misaligned temporal blending, whether
/// from a neighbour shifted wrongly or from a bad centre-frame copy into
/// the compensated buffer.
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
            estimation: MotionEstimation::Direct,
        },
        hq: None,
    };

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&f0);
    d.push_frame(&f1);
    d.push_frame(&f2);
    let result = d
        .denoise()
        .unwrap()
        .unwrap()
        .as_f32()
        .expect("f32 denoiser")
        .to_vec();

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

/// Guards the motion field's binding offsets against misalignment.
///
/// A 1080x1080 frame at the library defaults gives 135x135 blocks, an
/// odd count of 18,225, whose unpadded per-neighbour stride is not a
/// 32-byte multiple.
///
/// wgpu rejects a binding offset that is not a multiple of its
/// `min_storage_buffer_offset_alignment`, so the second neighbour's
/// dispatch fails outright at this exact size unless the stride is
/// padded. See `mv_field_byte_offset`.
///
/// The 1920x1080 size every other test uses happens to land on an even
/// block count here and never reaches the bug, which is why this needs
/// its own size.
#[test]
fn motion_compensation_1080_square_odd_block_count_dispatch_succeeds() {
    let client = make_client();
    let w = 1080u32;
    let h = 1080u32;
    let frame = make_uniform_frame(w, h, 1, 0.5);

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
    let mc = MotionCtx::new(params.motion_compensation, w, h, test_align()).unwrap();
    assert_eq!(
        mc.blocks_x * mc.blocks_y,
        18225,
        "test premise: this geometry gives an odd block count"
    );

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(&frame);
    d.push_frame(&frame);
    d.push_frame(&frame);
    let result = d
        .denoise()
        .unwrap()
        .unwrap()
        .as_f32()
        .expect("f32 denoiser")
        .to_vec();

    assert_eq!(result.len(), (w * h) as usize);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "pixel {i}: non-finite output {v}");
        assert!(
            (v - 0.5).abs() < 1e-3,
            "pixel {i}: expected 0.5 (uniform input passthrough), got {v}"
        );
    }
}

/// Guards the pair ring's binding offsets against misalignment.
///
/// This uses the same odd-block-count geometry as
/// `motion_compensation_1080_square_odd_block_count_dispatch_succeeds`
/// above, but pins `Chained` estimation so the pair ring is actually
/// allocated and used.
///
/// The defaults pin `Direct`, whose dispatch never touches the pair ring
/// at all, so the test above only ever binds into the motion field.
///
/// Every other `Chained` test in this file uses a small frame with 256
/// blocks, an even count, so none of them reach an odd one either.
///
/// Pushing enough real frames past the priming window covers both the
/// zeroed duplicate slots and the real hops, all of which bind into the
/// pair ring.
#[test]
fn motion_compensation_1080_square_odd_block_count_chained_dispatch_succeeds() {
    let client = make_client();
    let w = 1080u32;
    let h = 1080u32;
    let radius = 2u32;
    let frames: Vec<Vec<f32>> = (0..8)
        .map(|i| make_frame_with_noisy_region(w, h, 1, 0.5, 200 + i * 4, 200, 8, 0.8))
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
    let mc = MotionCtx::new(params.motion_compensation, w, h, test_align()).unwrap();
    assert_eq!(
        mc.blocks_x * mc.blocks_y,
        18225,
        "test premise: this geometry gives an odd block count"
    );

    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    assert!(
        d.pair_ring_buf.is_some(),
        "test premise: Chained estimation must allocate the pair ring"
    );

    let check = |frame: &[f32]| {
        for (i, &v) in frame.iter().enumerate() {
            assert!(v.is_finite(), "pixel {i}: non-finite output {v}");
            assert!((-0.01..=1.01).contains(&v), "pixel {i}: out-of-range output {v}");
        }
    };

    let mut emitted = 0usize;
    for frame in &frames {
        d.push_frame(frame);
        if let Some(result) = d.denoise().unwrap() {
            check(result.as_f32().expect("f32 denoiser"));
            emitted += 1;
        }
    }
    d.flush(|frame| {
        check(frame.as_f32().expect("f32 denoiser"));
        emitted += 1;
    })
    .unwrap();

    assert_eq!(emitted, frames.len(), "expected one output per pushed frame");
}

// --- Coarse-seeding tiling and pyramid level-0 extraction guards.
// Both defects live in the shared coarse+fine machinery every
// estimation mode funnels through (`run_analyse`), so they're
// exercised here through a real `Direct`-estimation `NlmDenoiser`,
// reading `mv_field_buf` back directly (the same pattern
// `composed_centre_mv` above already establishes for the Chained
// tests).

/// Builds a `w x h` frame from two independently-rich `half`-wide
/// halves (`left`, `right`), each optionally shifted locally within
/// its own half by `left_shift`/`right_shift` fine-level pixels
/// (`0` for the unshifted base frame). Local, per-half clamped
/// shifting (via `shift_clamped`) keeps each half's content a
/// self-contained translation, so a block deep inside one half never
/// legitimately depends on the other half's content.
fn split_half_frame(
    w: u32,
    h: u32,
    half: u32,
    left: &[f32],
    right: &[f32],
    left_shift: i32,
    right_shift: i32,
) -> Vec<f32> {
    let mut frame = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            if x < half {
                let lx = shift_clamped(x as i32, left_shift, half as i32) as u32;
                frame[idx] = left[(y * half + lx) as usize];
            } else {
                let rx = shift_clamped((x - half) as i32, right_shift, half as i32) as u32;
                frame[idx] = right[(y * half + rx) as usize];
            }
        }
    }
    frame
}

/// Pushes `base` twice then `neighbour` once through a fresh `Direct`
/// `NlmDenoiser` at `mode`, runs one `denoise()`, and reads back the
/// forward (`k = 1`) neighbour's MV field. Mirrors the push order
/// every other test in this file relies on (`push, push, push`, single
/// `denoise()`, centre = the second push, see
/// `motion_compensation_translating_square_preserves_centre`'s own
/// comment for the derivation), so `neighbour` (the third push) is
/// what the returned MV field is matched against.
fn direct_mv_field_for_forward_neighbour(
    mode: MotionCompensationMode,
    w: u32,
    h: u32,
    base: &[f32],
    neighbour: &[f32],
) -> Vec<i32> {
    let params = NlmParams {
        temporal_radius: 1,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: mode,
        hq: None,
    };

    let client = make_client();
    let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
    d.push_frame(base);
    d.push_frame(base);
    d.push_frame(neighbour);
    d.denoise().unwrap();

    let mc = MotionCtx::new(mode, w, h, test_align()).unwrap();
    let neighbour_idx = neighbour_idx_for_k(1, 1);
    let mv_field = d
        .mv_field_buf
        .as_ref()
        .expect("mv_field allocated when mc_ctx is Some");
    let offset = mv_field_byte_offset(&mc, neighbour_idx);
    let sliced = mv_field.clone().offset_start(offset);
    let bytes = d.client.read_one(sliced).expect("mv readback failed");
    i32::from_bytes(&bytes).to_vec()
}

/// Equal-grid seeding guard. At `blksize=8, overlap=4,
/// pyramid_levels=2` the coarse and fine grids come out to the *same*
/// block count (`coarse_blocks_x == blocks_x`, verified below
/// algebraically rather than assumed), so seeding must map by
/// position, not by index doubling. A fixed-doubling map
/// (`fine_bx_origin = bx * level_scale`) lands a coarse block's
/// displacement at fine index `2 * bx` instead of `bx`, seeding
/// roughly half the fine grid from a coarse block spatially tied to a
/// *different* region.
///
/// The content is a left half and a right half, each independently
/// detailed, and each moving by a different amount between the centre
/// frame and the one after it.
///
/// Both shifts are multiples of the pyramid scale, so the coarse level
/// sees an exact half-size version of each.
///
/// The blocks under test sit deep inside their own half, well clear of
/// both the boundary between the halves and the frame edges. A
/// correctly-seeded block therefore finds its own half's motion, while
/// one seeded from the wrong half lands outside the fine pass's small
/// search window and cannot recover.
///
/// At this geometry a left-half block always gets seeded from another
/// left-half block, which carries the same motion, so a seeding defect
/// only shows up on the right half.
#[test]
fn coarse_seeding_handles_equal_grids() {
    let w = 128u32;
    let h = 64u32;
    let half = 64u32;

    let mode = MotionCompensationMode::Mvtools {
        blksize: 8,
        overlap: 4,
        search_radius: 3,
        pyramid_levels: 2,
        estimation: MotionEstimation::Direct,
    };
    let mc = MotionCtx::new(mode, w, h, test_align()).unwrap();

    // Recompute `run_analyse`'s coarse-grid formula directly (mirrors
    // `analyse.rs`'s own derivation) rather than trusting the brief's
    // worked numbers, confirming this geometry actually lands on
    // "equal grids" before relying on it.
    let coarse_scale = 1u32 << (mc.pyramid_levels - 1);
    let coarse_step = (mc.step / coarse_scale).max(1);
    let cw = w / coarse_scale;
    let coarse_blocks_x = cw.div_ceil(coarse_step).max(1);
    assert_eq!(
        coarse_blocks_x, mc.blocks_x,
        "test premise: this geometry must give equal coarse/fine grids"
    );

    let left = noisy_copy(half, 0.5, 0.2, 201);
    let right = noisy_copy(half, 0.5, 0.2, 202);
    let left_shift = 4i32;
    let right_shift = -4i32;

    let base = split_half_frame(w, h, half, &left, &right, 0, 0);
    let shifted = split_half_frame(w, h, half, &left, &right, left_shift, right_shift);

    let data = direct_mv_field_for_forward_neighbour(mode, w, h, &base, &shifted);

    // Deep inside each half, clear of the boundary at x=64 and the
    // frame edges by well over `search_radius + blksize` (= 11).
    let by = 4u32;
    let bx_left = 8u32;
    let bx_right = 24u32;
    let idx_left = ((by * mc.blocks_x + bx_left) * 2) as usize;
    let idx_right = ((by * mc.blocks_x + bx_right) * 2) as usize;

    assert_eq!(
        (data[idx_left], data[idx_left + 1]),
        (left_shift, 0),
        "left-half block should recover the left half's own motion ({left_shift}, 0), got ({}, {})",
        data[idx_left],
        data[idx_left + 1],
    );
    assert_eq!(
        (data[idx_right], data[idx_right + 1]),
        (right_shift, 0),
        "right-half block should recover the right half's own motion ({right_shift}, 0), got \
         ({}, {}); a wrong value here means it was seeded from the wrong (left-half) coarse block",
        data[idx_right],
        data[idx_right + 1],
    );
}

/// Half-grid seeding coverage at the genuine `fine_blocks_x/y ==
/// 2 * coarse_blocks_x/y` geometry. `mc.step == 1` reaches this ratio
/// through the `.max(1)` floor clamp in `run_analyse`'s `coarse_step`
/// derivation (`coarse_step = (1 / 2).max(1) = 1`), verified
/// algebraically below rather than assumed. A coarse block's region
/// really does correspond to exactly `level_scale` fine blocks here,
/// so the position-based tiling formula must reduce to plain index
/// doubling, both for uniform motion and for the left/right split
/// (different motion per half) the equal-grid test above exercises.
#[test]
fn coarse_seeding_still_correct_at_half_grid() {
    let w = 48u32;
    let h = 16u32;
    let half = 24u32;

    let mode = MotionCompensationMode::Mvtools {
        blksize: 4,
        overlap: 3,
        search_radius: 2,
        pyramid_levels: 2,
        estimation: MotionEstimation::Direct,
    };
    let mc = MotionCtx::new(mode, w, h, test_align()).unwrap();

    let coarse_scale = 1u32 << (mc.pyramid_levels - 1);
    let coarse_step = (mc.step / coarse_scale).max(1);
    let cw = w / coarse_scale;
    let ch = h / coarse_scale;
    let coarse_blocks_x = cw.div_ceil(coarse_step).max(1);
    let coarse_blocks_y = ch.div_ceil(coarse_step).max(1);
    assert_eq!(mc.step, 1, "test premise: step must floor-clamp coarse_step to 1");
    assert_eq!(
        mc.blocks_x,
        2 * coarse_blocks_x,
        "test premise: this geometry must give a genuine 2:1 fine:coarse ratio in x"
    );
    assert_eq!(
        mc.blocks_y,
        2 * coarse_blocks_y,
        "test premise: this geometry must give a genuine 2:1 fine:coarse ratio in y"
    );

    let left = noisy_copy(half, 0.5, 0.2, 301);
    let right = noisy_copy(half, 0.5, 0.2, 302);
    // Deep inside each half, clear of the boundary at x=24 and the
    // frame edges by well over `search_radius + blksize` (= 6).
    let by = 6u32;
    let bx_left = 10u32;
    let bx_right = 32u32;

    let mv_at = |left_shift: i32, right_shift: i32| -> ((i32, i32), (i32, i32)) {
        let base = split_half_frame(w, h, half, &left, &right, 0, 0);
        let shifted = split_half_frame(w, h, half, &left, &right, left_shift, right_shift);
        let data = direct_mv_field_for_forward_neighbour(mode, w, h, &base, &shifted);
        let idx_left = ((by * mc.blocks_x + bx_left) * 2) as usize;
        let idx_right = ((by * mc.blocks_x + bx_right) * 2) as usize;
        (
            (data[idx_left], data[idx_left + 1]),
            (data[idx_right], data[idx_right + 1]),
        )
    };

    // Uniform sub-case. Both halves share one motion vector, so any
    // coarse block seeds any fine block correctly and seeding defects
    // stay invisible.
    let (uni_left, uni_right) = mv_at(2, 2);
    assert_eq!(uni_left, (2, 0), "uniform motion: left block got {uni_left:?}");
    assert_eq!(uni_right, (2, 0), "uniform motion: right block got {uni_right:?}");

    // Varying sub-case. Left and right halves move oppositely, so
    // each fine block only recovers its own half's motion when the
    // tiling formula seeds it from the coarse block covering the same
    // region.
    let (var_left, var_right) = mv_at(2, -2);
    assert_eq!(var_left, (2, 0), "varying motion: left block got {var_left:?}");
    assert_eq!(
        var_right,
        (-2, 0),
        "varying motion: right block got {var_right:?}"
    );
}

/// Pyramid level-0 extraction guard. `build_pyramid_for_slot` must
/// extract level-0 luma even when `pyramid_levels == 1`, not skip the
/// whole pyramid build. This is a real `Mvtools` MC config (not
/// `MotionCtx::confidence_only`, a separate path that doesn't share
/// this build step), so the fine pass runs unseeded straight off
/// level 0. If level 0 is never written, the recovered MV has no
/// reason to match the known shift below.
#[test]
fn pyramid_level0_extracted_at_one_level() {
    let w = 64u32;
    let h = 64u32;
    let dx = 2i32;
    let dy = 1i32;

    let mode = MotionCompensationMode::Mvtools {
        blksize: DEFAULT_BLKSIZE,
        overlap: DEFAULT_OVERLAP,
        search_radius: DEFAULT_SEARCH_RADIUS,
        pyramid_levels: 1,
        estimation: MotionEstimation::Direct,
    };
    let mc = MotionCtx::new(mode, w, h, test_align()).unwrap();

    let world = noisy_copy(w, 0.5, 0.2, 77);
    let shifted = frame_shifted_by(&world, w, h, dx, dy);

    let data = direct_mv_field_for_forward_neighbour(mode, w, h, &world, &shifted);

    // Interior block, well clear of the frame edges.
    let bx = mc.blocks_x / 2;
    let by = mc.blocks_y / 2;
    let idx = ((by * mc.blocks_x + bx) * 2) as usize;

    assert_eq!(
        (data[idx], data[idx + 1]),
        (dx, dy),
        "a clean ({dx}, {dy}) shift with pyramid_levels=1 should give exactly that MV at an \
         interior block once level-0 luma is actually extracted, got ({}, {})",
        data[idx],
        data[idx + 1],
    );
}

/// Guards the seeding of the ragged blocks at a frame's trailing edge.
///
/// The coarse and fine block counts each round up over a different width
/// and a different step, so they can round differently even where the
/// ratio between the two grids is otherwise consistent.
///
/// At the library defaults, certain frame sizes leave the coarse grid
/// exactly one fine block short of the frame's true edge. The last
/// coarse block on each axis therefore has to extend its reach to
/// absorb the remainder.
///
/// A trailing column or row no coarse block writes quietly keeps
/// whatever the motion field already holds, which is zero from a fresh
/// buffer here but a stale frame's motion in production.
///
/// The premises are verified below rather than assumed.
///
/// # Why the motion is uniform and negative
///
/// The motion here is one uniform diagonal shift, not the two-part
/// split `coarse_seeding_handles_equal_grids` uses above.
///
/// This defect leaves a block never seeded at all rather than seeded
/// from the wrong place, and that other test already covers the wrong
/// place. So what separates seeded from unseeded here is a shift large
/// enough to escape the fine-only search window while still small
/// enough to be reachable through a correct seed.
///
/// The shift is negative on both axes on purpose. A block on the
/// trailing edge has only one genuinely valid column and row to read,
/// because the rest of its tile clamps to that same edge pixel.
///
/// The search at that position can only tell apart offsets that pull
/// content in from the interior. Offsets reaching further past the edge
/// clamp to the same neighbour pixel for every candidate and contribute
/// nothing that distinguishes them.
///
/// A positive shift at a trailing edge would be unrecoverable by any
/// search, seeded or not, and would not isolate this bug.
#[test]
fn coarse_seeding_covers_ragged_last_block() {
    let shift = -6i32;
    let mode = MotionCompensationMode::Mvtools {
        blksize: DEFAULT_BLKSIZE,
        overlap: DEFAULT_OVERLAP,
        search_radius: 4,
        pyramid_levels: 2,
        estimation: MotionEstimation::Direct,
    };

    // `w_gap` (57) is congruent to 1 modulo `step` (8), giving a
    // coarse grid exactly one block short of the fine grid on that
    // axis (matching the exact arithmetic the review's own worked
    // example used, `width = 601`). `h_nice` (64) is an exact multiple
    // of `step`, giving an ordinary equal coarse/fine grid on the
    // other axis (no gap there). Each case below uses one frame ragged
    // on a single axis, rather than one frame ragged on both, because
    // two dimensions both congruent to 1 modulo an even step are both
    // odd, and an odd-by-odd pixel count can never be a multiple of
    // the ring buffers' own 32-byte frame-stride alignment
    // requirement.
    let w_gap = 57u32;
    let h_nice = 64u32;

    // Builds the MV field for a `w x h` frame under a uniform diagonal
    // `(shift, shift)` translation and returns it alongside the
    // `MotionCtx` used to index it.
    let build = |w: u32, h: u32| -> (MotionCtx, Vec<i32>) {
        let mc = MotionCtx::new(mode, w, h, test_align()).unwrap();
        let world = make_noisy_gaussian_frame(w, h, 1, 0.5, &[0.2]);
        let shifted = frame_shifted_by(&world, w, h, shift, shift);
        let data = direct_mv_field_for_forward_neighbour(mode, w, h, &world, &shifted);
        (mc, data)
    };
    let at = |mc: &MotionCtx, data: &[i32], bx: u32, by: u32| -> (i32, i32) {
        let idx = ((by * mc.blocks_x + bx) * 2) as usize;
        (data[idx], data[idx + 1])
    };
    // Test premise, verified rather than assumed. Mirrors
    // `run_analyse`'s own coarse-grid formula (see the equal-grid test
    // above for the same derivation style) to confirm this geometry
    // actually lands on the ragged, one-short-of-the-fine-grid case on
    // exactly the named axis, and an ordinary equal grid on the other.
    let assert_ragged_on = |mc: &MotionCtx, w: u32, h: u32, ragged_axis_is_x: bool| {
        let coarse_scale = 1u32 << (mc.pyramid_levels - 1);
        let coarse_step = (mc.step / coarse_scale).max(1);
        let coarse_blocks_x = (w / coarse_scale).div_ceil(coarse_step).max(1);
        let coarse_blocks_y = (h / coarse_scale).div_ceil(coarse_step).max(1);
        if ragged_axis_is_x {
            assert_eq!(w % mc.step, 1, "test premise: width must be step*k + 1");
            assert_eq!(
                coarse_blocks_x,
                mc.blocks_x - 1,
                "test premise: ragged coarse grid, one block short in x"
            );
            assert_eq!(
                coarse_blocks_y, mc.blocks_y,
                "test premise: y axis is an ordinary equal grid here"
            );
        } else {
            assert_eq!(h % mc.step, 1, "test premise: height must be step*k + 1");
            assert_eq!(
                coarse_blocks_y,
                mc.blocks_y - 1,
                "test premise: ragged coarse grid, one block short in y"
            );
            assert_eq!(
                coarse_blocks_x, mc.blocks_x,
                "test premise: x axis is an ordinary equal grid here"
            );
        }
    };

    // X-axis case, last column at a non-edge row.
    let (mc_x, data_x) = build(w_gap, h_nice);
    assert_ragged_on(&mc_x, w_gap, h_nice, true);
    let mid_bx_x = mc_x.blocks_x / 2;
    let mid_by_x = mc_x.blocks_y / 2;
    assert_eq!(
        at(&mc_x, &data_x, mid_bx_x, mid_by_x),
        (shift, shift),
        "interior control block (x-axis case) should recover ({shift}, {shift})"
    );
    assert_eq!(
        at(&mc_x, &data_x, mc_x.blocks_x - 1, mid_by_x),
        (shift, shift),
        "last-column block (x-axis coverage gap) should recover ({shift}, {shift})"
    );

    // Y-axis case, last row at a non-edge column (the same frame,
    // transposed).
    let (mc_y, data_y) = build(h_nice, w_gap);
    assert_ragged_on(&mc_y, h_nice, w_gap, false);
    let mid_bx_y = mc_y.blocks_x / 2;
    let mid_by_y = mc_y.blocks_y / 2;
    assert_eq!(
        at(&mc_y, &data_y, mid_bx_y, mid_by_y),
        (shift, shift),
        "interior control block (y-axis case) should recover ({shift}, {shift})"
    );
    assert_eq!(
        at(&mc_y, &data_y, mid_bx_y, mc_y.blocks_y - 1),
        (shift, shift),
        "last-row block (y-axis coverage gap) should recover ({shift}, {shift})"
    );
}

// --- Chained motion estimation, pair ring + composition kernel ---
//
// These tests exercise `NlmDenoiser::run_chain_compose` and the
// push-time pair analyse directly. Nothing in the submit path calls
// either yet, so every test drives them explicitly.

const CHAIN_TEST_RADIUS: u32 = 2;
const CHAIN_TEST_SIZE: u32 = 64;

fn chained_params(refine_radius: u32) -> NlmParams {
    NlmParams {
        temporal_radius: CHAIN_TEST_RADIUS,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        // Matches the MC geometry the other tests in this file already
        // exercise through the real push pipeline (blksize=8, overlap=4,
        // pyramid_levels=2).
        motion_compensation: MotionCompensationMode::Mvtools {
            blksize: 8,
            overlap: 4,
            search_radius: 2,
            pyramid_levels: 2,
            estimation: MotionEstimation::Chained { refine_radius },
        },
        hq: None,
    }
}

/// Builds a `w × h` frame that reads `world` shifted by `(dx, dy)`
/// pixels, clamped to `world`'s own edges, i.e. `frame(x, y) = world(x
/// - dx, y - dy)`. A sequence built from the same `world` with `dx = n
/// * v` for increasing `n` gives adjacent frames that differ by
/// exactly `(v, v)` everywhere except right at the frame edges.
fn frame_shifted_by(world: &[f32], w: u32, h: u32, dx: i32, dy: i32) -> Vec<f32> {
    let mut frame = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let sx = shift_clamped(x as i32, dx, w as i32) as u32;
            let sy = shift_clamped(y as i32, dy, h as i32) as u32;
            frame[(y * w + x) as usize] = world[(sy * w + sx) as usize];
        }
    }
    frame
}

/// Same idea as `frame_shifted_by`, but wraps at the frame edges
/// (`rem_euclid`) instead of clamping. `world`'s content is spatially
/// unstructured (see `noisy_copy`), so a wrapped shift keeps every
/// pixel position fully translation-invariant, with no degenerate
/// clamped border region anywhere in the frame regardless of how large
/// `dx`/`dy` grow. Used by the k4 alignment test below, which pushes
/// many frames with a steadily growing absolute shift and needs the
/// whole frame to stay a clean, uniform translation of `world`.
fn frame_shifted_wrapped(world: &[f32], w: u32, h: u32, dx: i32, dy: i32) -> Vec<f32> {
    let mut frame = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let sx = (x as i32 - dx).rem_euclid(w as i32) as u32;
            let sy = (y as i32 - dy).rem_euclid(h as i32) as u32;
            frame[(y * w + x) as usize] = world[(sy * w + sx) as usize];
        }
    }
    frame
}

/// Pushes a constant-velocity sequence (`v` pixels/frame on both axes,
/// built from one rich pattern) through a `Chained` denoiser. Pushes
/// `1 + 3 * radius + 2` real frames past the stream's first one. `1 +
/// 3 * radius` is enough for every one of the `2 * radius` pair-ring
/// slots to be overwritten by real analyse past the initial priming
/// duplicates (see `NlmDenoiser::pair_slot`'s doc comment for the
/// `2 * radius`-push lifetime this relies on), plus 2 frames of margin.
/// The chosen frame size and a centre-block readout keep every match
/// clear of both the frame edges and the accumulated drift.
fn push_constant_velocity(client: &ComputeClient<R>, radius: u32, v: i32) -> NlmDenoiser<R> {
    let w = CHAIN_TEST_SIZE;
    let h = CHAIN_TEST_SIZE;
    let world = noisy_copy(w, 0.5, 0.2, 99);

    let mut d = NlmDenoiser::<R>::new(client, chained_params(2), w, h);

    let real_pushes = 1 + 3 * radius as i32 + 2;
    for n in 0..real_pushes {
        let frame = frame_shifted_by(&world, w, h, n * v, n * v);
        d.push_frame(&frame);
    }
    d
}

/// Runs `run_chain_compose` for neighbour offset `k` and reads back the
/// composed MV at the block nearest the frame's centre.
fn composed_centre_mv(d: &NlmDenoiser<R>, radius: u32, k: i32) -> (i32, i32) {
    d.run_chain_compose(radius, k)
        .expect("chain compose dispatch failed");

    let mc = MotionCtx::new(d.params.motion_compensation, d.width, d.height, d.align).unwrap();
    let neighbour_idx = neighbour_idx_for_k(radius, k);
    let mv_field = d
        .mv_field_buf
        .as_ref()
        .expect("mv_field allocated when mc_ctx is Some");
    let offset = mv_field_byte_offset(&mc, neighbour_idx);
    let sliced = mv_field.clone().offset_start(offset);
    let bytes = d.client.read_one(sliced).expect("mv readback failed");
    let data = i32::from_bytes(&bytes);

    let bx = mc.blocks_x / 2;
    let by = mc.blocks_y / 2;
    let idx = ((by * mc.blocks_x + bx) * 2) as usize;
    (data[idx], data[idx + 1])
}

/// Asserts that the `radius` pair-ring writes starting at
/// `ring_head_before` (the `ring_head` value the first of those writes
/// saw, pre-advance) are all zero in both directions. Used to check
/// duplicated slots (priming or flush), whose pair is zero motion by
/// definition.
fn assert_pair_ring_zero_from(d: &NlmDenoiser<R>, ring_head_before: i32, radius: u32) {
    let mc = MotionCtx::new(d.params.motion_compensation, d.width, d.height, d.align).unwrap();
    let pair_ring = d
        .pair_ring_buf
        .as_ref()
        .expect("pair_ring allocated when Chained is active");
    let pair_ring_slots = 2 * radius as i32;
    let dir_len = mc.pair_direction_len() as usize;

    for i in 0..radius as i32 {
        let slot = (ring_head_before + i).rem_euclid(pair_ring_slots) as u32;
        for direction in 0..2u32 {
            let offset = pair_byte_offset(&mc, slot, direction);
            let sliced = pair_ring.clone().offset_start(offset);
            let bytes = d.client.read_one(sliced).expect("pair ring readback failed");
            let data = i32::from_bytes(&bytes);
            assert!(
                data[..dir_len].iter().all(|&v| v == 0),
                "duplicate pair slot {slot} direction {direction} should be zero-filled, got {:?}",
                &data[..dir_len],
            );
        }
    }
}

#[test]
fn chain_compose_zero_motion_gives_zero_mv() {
    let client = make_client();
    let radius = CHAIN_TEST_RADIUS;
    let d = push_constant_velocity(&client, radius, 0);

    for k in 1..=radius as i32 {
        assert_eq!(
            composed_centre_mv(&d, radius, k),
            (0, 0),
            "forward k={k} should compose to zero motion on a static sequence"
        );
        assert_eq!(
            composed_centre_mv(&d, radius, -k),
            (0, 0),
            "backward k={k} should compose to zero motion on a static sequence"
        );
    }
}

/// Constant-velocity sequence. The forward-composed MV to neighbour k
/// must equal exactly `k * v` for every k up to the temporal radius.
///
/// Uses `v = 2` rather than `1`, since the block-match geometry here
/// runs a 2-level pyramid, and a `v = 1` fine-level shift is only half
/// a pixel at the coarse level, ambiguous enough that the coarse pass
/// can lock onto the wrong candidate for either pair direction
/// (confirmed by hand-checking the pair ring directly during
/// debugging). `v = 2` gives a clean one-pixel shift at both levels.
#[test]
fn chain_compose_constant_velocity_matches_k_times_v() {
    let client = make_client();
    let radius = CHAIN_TEST_RADIUS;
    let v = 2;
    let d = push_constant_velocity(&client, radius, v);

    for k in 1..=radius as i32 {
        assert_eq!(
            composed_centre_mv(&d, radius, k),
            (k * v, k * v),
            "forward k={k} should compose to exactly k*v = ({}, {})",
            k * v,
            k * v
        );
    }
}

/// Same constant-velocity sequence walked backward. The composed MV to
/// neighbour -k must equal `-k * v`, the mirror image of the forward
/// case, since the backward pass reads the newer→older field at every
/// hop instead of older→newer. Uses `v = 2` for the same reason as
/// `chain_compose_constant_velocity_matches_k_times_v`.
#[test]
fn chain_compose_backward_direction_matches_negative_k_times_v() {
    let client = make_client();
    let radius = CHAIN_TEST_RADIUS;
    let v = 2;
    let d = push_constant_velocity(&client, radius, v);

    for k in 1..=radius as i32 {
        assert_eq!(
            composed_centre_mv(&d, radius, -k),
            (-k * v, -k * v),
            "backward k={k} should compose to exactly -k*v = ({}, {})",
            -k * v,
            -k * v
        );
    }
}

/// Duplicated ring slots (stream priming and end-of-stream flush) get
/// a zero-filled pair field rather than a real analyse. Pushes real,
/// nonzero-motion frames in between so the check isn't trivially true
/// from a freshly-allocated, still-zeroed buffer.
#[test]
fn chain_compose_duplicated_slot_pairs_are_zero_filled() {
    let client = make_client();
    let radius = CHAIN_TEST_RADIUS;
    let w = CHAIN_TEST_SIZE;
    let h = CHAIN_TEST_SIZE;
    let world = noisy_copy(w, 0.5, 0.2, 7);

    let mut d = NlmDenoiser::<R>::new(&client, chained_params(2), w, h);

    // During priming the very first push has no older partner (`ring_head ==
    // 0`), so the analyse skip leaves the window's very first gap to
    // be filled entirely by the `radius` auto-primed duplicates that
    // follow. Each of those duplicates' own pair write happens with
    // `ring_head` starting at 1 (right after the real push's
    // `advance_ring`).
    d.push_frame(&world);
    assert_pair_ring_zero_from(&d, 1, radius);

    // Overwrite every pair slot with real, nonzero motion before
    // checking the flush path, so its zero-fill isn't indistinguishable
    // from an untouched buffer.
    for n in 1..=(3 * radius) {
        let frame = frame_shifted_by(&world, w, h, n as i32, n as i32);
        d.push_frame(&frame);
    }

    let ring_head_before_flush = d.ring_head as i32;
    d.flush(|_| {}).expect("flush failed");
    assert_pair_ring_zero_from(&d, ring_head_before_flush, radius);
}

// --- Chained motion estimation, submit-path wiring (dispatch.rs) ---
//
// The tests above drive `run_chain_compose` directly. These exercise the
// real per-submit dispatch branch (`run_motion_compensation` in
// `dispatch.rs`), which composes, refines, and warps automatically on
// every `push_frame` + `denoise`/`flush` call once `estimation` is
// `Chained`.

/// Builds an HQ config with `Chained` motion estimation at the given
/// temporal radius and refinement radius. Auto noise estimation and
/// temporal confidence stay on, mirroring `hq_temporal_mc_confidence_smoke`
/// in `tests/hq.rs` but with the chained estimator instead of direct.
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
            windowed_noise_estimation: false,
        }),
    }
}

/// End-to-end smoke test. A `Chained`-estimation denoiser must produce
/// finite, `[0, 1]` output for every pushed frame, whether from pushes
/// or the trailing flush, at the given temporal radius. Exercises the
/// full `push_frame` → `denoise` → `flush` pipeline, so the compose and
/// seeded-refine dispatch branch in `run_motion_compensation` actually
/// runs (not just `run_chain_compose` in isolation).
fn chained_end_to_end_finite(radius: u32) {
    let client = make_client();
    let w = 32u32;
    let h = 32u32;

    let mut denoiser = NlmDenoiser::<R>::new(&client, chained_hq_params(radius, 2), w, h);

    let frames: Vec<Vec<f32>> = (0..8)
        .map(|i| make_frame_with_noisy_region(w, h, 1, 0.5, 6 + i, 8, 2, 0.8))
        .collect();

    let mut emitted = 0usize;
    let check = |frame: &[f32]| {
        for (i, &v) in frame.iter().enumerate() {
            assert!(v.is_finite(), "pixel {i}: non-finite output {v}");
            assert!((0.0..=1.0).contains(&v), "pixel {i}: out-of-range output {v}");
        }
    };

    for frame in &frames {
        denoiser.push_frame(frame);
        if let Some(result) = denoiser.denoise().unwrap() {
            check(result.as_f32().expect("f32 denoiser"));
            emitted += 1;
        }
    }

    denoiser
        .flush(|frame| {
            check(frame.as_f32().expect("f32 denoiser"));
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

/// `MotionCompensationMode::mvtools_default()` sets `estimation: Direct`
/// (see its own doc comment). Building the same configuration two ways
/// the codebase supports it, the convenience constructor and an
/// explicit `Mvtools` struct literal, must give bit-identical output for
/// identical input, since dispatch behaviour depends only on the
/// resulting value. Guards against the new chained dispatch branch in
/// `run_motion_compensation` accidentally changing what the Direct
/// branch does.
#[test]
fn direct_estimation_default_and_explicit_construction_match_bit_for_bit() {
    let client = make_client();
    let w = 32u32;
    let h = 32u32;
    let frame = make_frame_with_noisy_region(w, h, 1, 0.5, 16, 16, 4, 0.8);

    let run = |mc: MotionCompensationMode| {
        let params = NlmParams {
            temporal_radius: 1,
            search_radius: 2,
            patch_radius: 2,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::None,
            motion_compensation: mc,
            hq: None,
        };
        let mut d = NlmDenoiser::<R>::new(&client, params, w, h);
        d.push_frame(&frame);
        d.push_frame(&frame);
        d.push_frame(&frame);
        d.denoise()
            .unwrap()
            .unwrap()
            .as_f32()
            .expect("f32 denoiser")
            .to_vec()
    };

    let via_default = run(MotionCompensationMode::mvtools_default());
    let via_explicit = run(MotionCompensationMode::Mvtools {
        blksize: DEFAULT_BLKSIZE,
        overlap: DEFAULT_OVERLAP,
        search_radius: DEFAULT_SEARCH_RADIUS,
        pyramid_levels: DEFAULT_PYRAMID_LEVELS,
        estimation: MotionEstimation::Direct,
    });

    assert_eq!(
        via_default, via_explicit,
        "Direct estimation must give the same output regardless of which \
         constructor produced the MotionCompensationMode value"
    );
}

// --- Auto estimation, resolution allocates the pair ring exactly like
// an explicit Chained request would ---

fn auto_params(temporal_radius: u32) -> NlmParams {
    NlmParams {
        temporal_radius,
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
            estimation: MotionEstimation::Auto,
        },
        hq: None,
    }
}

#[test]
fn auto_estimation_at_high_radius_allocates_pair_ring() {
    let client = make_client();
    let params = auto_params(CHAINED_RADIUS_THRESHOLD);

    let d = NlmDenoiser::<R>::new(&client, params, 32, 32);

    assert!(
        d.pair_ring_buf.is_some(),
        "Auto at radius {CHAINED_RADIUS_THRESHOLD} (>= CHAINED_RADIUS_THRESHOLD) should \
         resolve to Chained and allocate the pair ring"
    );
}

#[test]
fn auto_estimation_at_low_radius_does_not_allocate_pair_ring() {
    let client = make_client();
    let params = auto_params(CHAINED_RADIUS_THRESHOLD - 1);

    let d = NlmDenoiser::<R>::new(&client, params, 32, 32);

    assert!(
        d.pair_ring_buf.is_none(),
        "Auto at radius {} (< CHAINED_RADIUS_THRESHOLD) should resolve to Direct \
         and skip the pair ring",
        CHAINED_RADIUS_THRESHOLD - 1
    );
}

// --- The key regression test, chained beats direct once k*v exceeds
// direct's search window ---

/// Temporal radius for the k4 alignment tests below. Chosen so k=4 is
/// reachable (needs `radius >= 4`).
const K4_RADIUS: u32 = 4;
/// Frame side length. Comfortably larger than any candidate offset the
/// coarse+fine search (or its seeded refine) can reach, so a wrapped
/// shift is never ambiguous with its aliased counterpart on the other
/// side of the torus.
const K4_SIZE: u32 = 128;
/// Per-frame diagonal shift in pixels. At this geometry (`mc.search_radius
/// = 4`, `pyramid_levels = 2`), the coarse pass doubles reach to 8px
/// (half-res search radius 4, scaled by the /2 level) and the fine pass
/// adds another 4px around that seed, so direct's reachable window is
/// 12px total. `k = 1` motion (this value) sits comfortably inside that
/// window. `k = 4` motion (`4 * K4_V = 16` px) exceeds it.
const K4_V: i32 = 4;

fn k4_params(estimation: MotionEstimation) -> NlmParams {
    NlmParams {
        temporal_radius: K4_RADIUS,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::Mvtools {
            blksize: 8,
            overlap: 4,
            search_radius: 4,
            pyramid_levels: 2,
            estimation,
        },
        hq: None,
    }
}

/// Reads back frame `slot` from a ring-buffer handle laid out
/// `[total_frames][h][w][stored_ch]` f32, matching `input_buf` /
/// `compensated_input_buf`'s shared layout.
fn read_frame_slot(
    client: &ComputeClient<R>,
    buf: &Handle,
    slot: u32,
    w: u32,
    h: u32,
    stored_ch: u32,
) -> Vec<f32> {
    let frame_size = (w * h * stored_ch) as usize;
    let byte_offset = (slot as u64) * (frame_size as u64) * (size_of::<f32>() as u64);
    let sliced = buf.clone().offset_start(byte_offset);
    let bytes = client.read_one(sliced).expect("frame readback failed");
    f32::from_bytes(&bytes)[..frame_size].to_vec()
}

/// Pushes a diagonal constant-velocity sequence (`K4_V` px/frame,
/// wrapped at the edges via `frame_shifted_wrapped` so the whole frame
/// stays a clean, uniform translation of one rich pattern with no
/// degenerate border region) through a `k4_params`-geometry denoiser,
/// then returns the mean absolute residual between the centre frame and
/// the forward `k`-neighbour's *compensated* (warped) copy, averaged
/// over the whole frame.
///
/// Pushes `2 * K4_RADIUS + 4` real frames before reading back, clearing
/// the stream's leading-edge priming duplicates (which need only `>
/// 2 * radius + 1` real pushes, see `NlmDenoiser::pair_slot`'s doc
/// comment) with a small margin to spare.
fn k4_compensated_residual(estimation: MotionEstimation, k: i32) -> f32 {
    let client = make_client();
    let w = K4_SIZE;
    let h = K4_SIZE;
    let world = noisy_copy(w, 0.5, 0.2, 55);

    let mut d = NlmDenoiser::<R>::new(&client, k4_params(estimation), w, h);

    let real_pushes = 2 * K4_RADIUS as i32 + 4;
    for n in 0..real_pushes {
        let frame = frame_shifted_wrapped(&world, w, h, n * K4_V, n * K4_V);
        d.push_frame(&frame);
    }
    d.denoise().unwrap();

    let radius = d.params.temporal_radius;
    let stored_ch = d.params.channels.storage_count();
    let centre_slot = d.phys_frame(radius as i32);
    let neighbour_slot = d.phys_frame(radius as i32 + k);
    let compensated = d
        .compensated_input_buf
        .as_ref()
        .expect("compensated buf allocated when MC is active");

    let centre_frame = read_frame_slot(&d.client, &d.input_buf, centre_slot, w, h, stored_ch);
    let warped = read_frame_slot(&d.client, compensated, neighbour_slot, w, h, stored_ch);

    let mut sum = 0.0f32;
    let mut count = 0u32;
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            sum += (centre_frame[idx] - warped[idx]).abs();
            count += 1;
        }
    }
    sum / count as f32
}

/// THE key regression test for chained motion estimation. At `k = 4`
/// the true displacement is `4 * K4_V = 16` px, beyond direct's
/// reachable window (`≈ 12` px at this geometry), while chained
/// composes four exact per-step vectors and only needs a `refine_radius
/// = 2` correction to land on the true offset. A real, wide-margin
/// assertion, not just a smoke check, since this comparison is the
/// entire reason `Chained` estimation exists.
#[test]
fn chained_beats_direct_at_k4_beyond_direct_window() {
    let direct_residual = k4_compensated_residual(MotionEstimation::Direct, 4);
    let chained_residual = k4_compensated_residual(MotionEstimation::Chained { refine_radius: 2 }, 4);

    assert!(
        direct_residual > 0.02,
        "expected direct's k=4 match to show a real misalignment residual \
         (window reach ≈12px, true motion 16px), got {direct_residual}"
    );
    assert!(
        chained_residual < direct_residual * 0.5,
        "chained's composed+refined k=4 alignment should beat direct's by a \
         wide margin: chained={chained_residual}, direct={direct_residual}"
    );
}

/// Companion sanity check. At `k = 1` the true displacement (`K4_V` = 4
/// px) sits comfortably inside direct's reach, so direct should already
/// align cleanly there. This isolates the k=4 test's effect to the
/// window-size failure mode rather than a blanket "chained is always
/// better" claim.
#[test]
fn direct_already_aligns_at_k1_inside_its_window() {
    let direct_residual = k4_compensated_residual(MotionEstimation::Direct, 1);
    assert!(
        direct_residual < 0.02,
        "direct should align cleanly at k=1 (motion {K4_V}px, well inside its ~12px reach), got {direct_residual}"
    );
}

use super::helpers::*;
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

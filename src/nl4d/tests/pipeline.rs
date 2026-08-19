use cubecl::prelude::*;

use super::helpers::{make_client, noisy_copy_of, psnr, textured_base, R};
use crate::collab::geometry::{member_buf_len, ref_count, refs_along};
use crate::collab::kernels::aggregate::{collab_normalise, collab_zero_accum, weight_scale};
use crate::collab::kernels::filter_ht::collab_filter_ht;
use crate::collab::kernels::group_temporal::collab_group_temporal;
use crate::collab::kernels::transforms::dct_noise_profile;
use crate::collab::MAX_K;
use crate::nl4d::{Nl4dDenoiser, Nl4dParams};
use crate::nlmeans::{ChannelMode, HqParams, MotionCompensationMode, MotionEstimation, NlmParams};

const SIGMA: f32 = 6.0 / 255.0;
const SPATIAL_RADIUS: u32 = 9;
const REFINE: u32 = 2;
const C_MIN: f32 = 0.05;
const LAMBDA_HT: f32 = 2.7;

fn static_clip_params(temporal_radius: u32) -> Nl4dParams {
    Nl4dParams {
        nlm: NlmParams {
            temporal_radius,
            search_radius: 2,
            patch_radius: 2,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: crate::nlmeans::PrefilterMode::None,
            motion_compensation: MotionCompensationMode::Mvtools {
                blksize: 16,
                overlap: 8,
                search_radius: 4,
                pyramid_levels: 2,
                estimation: MotionEstimation::Auto,
            },
            hq: Some(HqParams::with_sigma(SIGMA)),
            track_weight_sq: false,
        },
        temporal_radius,
        refine: REFINE,
        spatial_radius: SPATIAL_RADIUS,
        lambda_ht: LAMBDA_HT,
        c_min: C_MIN,
    }
}

/// A static clip, camera and content both still, with independent
/// per-frame noise. Every emitted frame must come out well above the
/// noisy input's own PSNR against the clean base.
#[test]
fn denoises_a_static_noisy_clip() {
    let client = make_client();
    let (w, h) = (64u32, 64u32);
    let radius = 2u32;
    let base = textured_base(w, h);
    let n = 9usize;

    let noisy_frames: Vec<Vec<f32>> = (0..n as u32).map(|seed| noisy_copy_of(&base, w, h, SIGMA, seed)).collect();

    let params = static_clip_params(radius);
    let mut d = Nl4dDenoiser::<R>::new(&client, params, w, h).expect("construction failed");

    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for frame in &noisy_frames {
        d.push_frame(frame);
        if let Some(pending) = d.denoise_submit().expect("denoise_submit failed") {
            outputs.push(pending.wait().expect("readback failed"));
        }
    }
    d.flush(|frame| outputs.push(frame.to_vec())).expect("flush failed");

    assert_eq!(
        outputs.len(),
        n,
        "expected one emitted frame per pushed frame"
    );

    for (i, out) in outputs.iter().enumerate() {
        let noisy_psnr = psnr(&noisy_frames[i], &base);
        let out_psnr = psnr(out, &base);
        assert!(
            out_psnr > noisy_psnr + 6.0,
            "frame {i}: expected at least a 6 dB PSNR improvement over the noisy input, got \
             noisy={noisy_psnr:.4} dB denoised={out_psnr:.4} dB"
        );
    }
}

/// Guards the `centre_slot`/`frame` contract `run_collab_stage` depends
/// on: `collab_group_temporal`'s `centre_slot` and `collab_filter_ht`'s
/// `frame` must read the same physical ring slot. Nothing in the type
/// system enforces that (both are plain `u32`s read from two separate
/// call sites), so this test plants content only the true centre frame
/// carries and checks it survives.
///
/// `denoises_a_static_noisy_clip` is a weak canary for this specific
/// mismatch: every ring slot there holds the same base content, so
/// feeding the filter a valid but wrong slot would barely move its PSNR
/// (a neighbour frame denoises to essentially the same clean content).
/// Here the centre frame alone carries a strong, low-frequency marker
/// block no neighbour has. If `collab_filter_ht` ever read a different
/// slot than `collab_group_temporal`'s `centre_slot`, none of the
/// marker's own members would pass the `member_frame == frame` scatter
/// gate (see `collab_filter_ht`'s doc comment), and the marker would be
/// attenuated or absent from the output.
///
/// The marker is a big flat block, not fine detail, so ordinary
/// shrinkage cannot legitimately remove it, and the assertion checks a
/// wide margin rather than an exact value, so ordinary filtering noise
/// does not trip it.
#[test]
fn output_carries_the_centre_frame_marker_no_neighbour_has() {
    let client = make_client();
    let (w, h) = (64u32, 64u32);
    let radius = 2u32;
    let base = textured_base(w, h);

    // `textured_base` never exceeds 0.65 (0.5 centre +/- 0.15 amplitude),
    // so a marker at 0.92 is unambiguous against it even before adding
    // noise or considering denoising error.
    const MARKER: f32 = 0.92;
    const MARKER_X0: u32 = 24;
    const MARKER_Y0: u32 = 24;
    const MARKER_SIZE: u32 = 24;
    // Read only the block's interior, away from its own edges, so patch
    // boundary blending against the surrounding non-marker texture can't
    // explain a low reading.
    const INTERIOR_MARGIN: u32 = 8;

    let mut marker_clean = base.clone();
    for y in MARKER_Y0..MARKER_Y0 + MARKER_SIZE {
        for x in MARKER_X0..MARKER_X0 + MARKER_SIZE {
            marker_clean[(y * w + x) as usize] = MARKER;
        }
    }

    let n_frames = 2 * radius + 1;
    let frames: Vec<Vec<f32>> = (0..n_frames)
        .map(|seed| {
            let content = if seed == radius { &marker_clean } else { &base };
            noisy_copy_of(content, w, h, SIGMA, seed)
        })
        .collect();

    let params = static_clip_params(radius);
    let mut d = Nl4dDenoiser::<R>::new(&client, params, w, h).expect("construction failed");
    for frame in &frames {
        d.push_frame(frame);
    }
    let out = d
        .denoise()
        .expect("denoise failed")
        .expect("window is exactly full, denoise should emit");

    let mut sum = 0.0f64;
    let mut count = 0usize;
    for y in (MARKER_Y0 + INTERIOR_MARGIN)..(MARKER_Y0 + MARKER_SIZE - INTERIOR_MARGIN) {
        for x in (MARKER_X0 + INTERIOR_MARGIN)..(MARKER_X0 + MARKER_SIZE - INTERIOR_MARGIN) {
            sum += out[(y * w + x) as usize] as f64;
            count += 1;
        }
    }
    let mean = sum / count as f64;
    eprintln!("output_carries_the_centre_frame_marker_no_neighbour_has: marker interior mean = {mean:.4}");

    assert!(
        mean > 0.75,
        "expected the centre frame's marker (planted at {MARKER}) to survive denoising with a \
         clear margin over textured_base's own ceiling of 0.65, got mean {mean:.4} over the \
         marker's interior; this would fail if collab_filter_ht's `frame` argument ever read a \
         different physical ring slot than collab_group_temporal's `centre_slot`"
    );
}

/// `flush` must emit exactly as many frames as were pushed, whatever mix
/// of `denoise_submit` and `flush` produced them.
#[test]
fn flush_emits_exactly_the_pushed_frame_count() {
    let client = make_client();
    let (w, h) = (64u32, 64u32);
    let radius = 2u32;
    let base = textured_base(w, h);
    let n = 7u32;

    let params = static_clip_params(radius);
    let mut d = Nl4dDenoiser::<R>::new(&client, params, w, h).expect("construction failed");

    let mut emitted = 0usize;
    for seed in 0..n {
        let frame = noisy_copy_of(&base, w, h, SIGMA, seed);
        d.push_frame(&frame);
        if d.denoise_submit().expect("denoise_submit failed").is_some() {
            emitted += 1;
        }
    }
    d.flush(|_| emitted += 1).expect("flush failed");

    assert_eq!(emitted, n as usize, "expected exactly {n} emitted frames");
}

/// Launches the same grouping, filtering, and aggregation kernels
/// [`Nl4dDenoiser::run_collab_stage`] runs, standalone, for a
/// single-frame ring at `radius = 0`. This is the "spatial-only" arm of
/// the hypothesis test below: identical grouping (no admission gate),
/// identical filter (hard threshold, same `lambda_ht`), identical noise
/// floor and `c_min`, with the only difference being that no temporal
/// candidates exist to search.
#[allow(clippy::too_many_arguments)]
fn run_spatial_only(
    client: &ComputeClient<R>,
    noisy_centre: &[f32],
    w: u32,
    h: u32,
    spatial_radius: u32,
    refine: u32,
    c_min: f32,
    lambda_ht: f32,
    sigma: f32,
) -> Vec<f32> {
    let k_max = MAX_K;
    let stored_ch = 1u32;
    let channels_count = 1u32;
    let refs_x = refs_along(w);
    let refs_y = refs_along(h);
    let refs = ref_count(w, h);
    let pos_len = member_buf_len(w, h, k_max);
    let pixels = (w * h) as usize;
    let frame_len = pixels;

    let ring_buf = client.create_from_slice(f32::as_bytes(noisy_centre));
    let mv_dummy = client.empty(size_of::<i32>());
    let conf_dummy = client.empty(size_of::<f32>());
    let neighbour_slots_dummy = client.empty(size_of::<u32>());
    let member_pos = client.empty(pos_len * size_of::<u32>());
    let member_frame = client.empty(pos_len * size_of::<u32>());
    let member_count = client.empty(refs * size_of::<u32>());
    let member_sig2_dummy = client.empty(size_of::<f32>());
    let filtered_dummy = client.empty(size_of::<f32>());
    let group_weight = client.empty(refs * size_of::<f32>());
    let sigma_buf = client.create_from_slice(f32::as_bytes(&[sigma]));
    let dct_profile = dct_noise_profile(0.0);
    let dct_profile_buf = client.create_from_slice(f32::as_bytes(&dct_profile));
    let accum = client.empty(frame_len * size_of::<i32>());
    let wsum = client.empty(pixels * size_of::<i32>());
    let output = client.empty(frame_len * size_of::<f32>());

    let group_grid = CubeCount::new_2d(refs_x, refs_y);
    let group_dim = CubeDim::new_2d(8, 8);
    let agg_grid = CubeCount::new_2d(
        w.div_ceil(crate::nlmeans::BLOCK_X),
        h.div_ceil(crate::nlmeans::BLOCK_Y),
    );
    let agg_dim = CubeDim::new_2d(crate::nlmeans::BLOCK_X, crate::nlmeans::BLOCK_Y);
    let zero_dim = 256u32;
    let zero_grid = CubeCount::new_1d((frame_len as u32).div_ceil(zero_dim));

    let wnorm = weight_scale(sigma, &dct_profile);
    let centre_slot = 0u32;

    unsafe {
        collab_zero_accum::launch_unchecked::<R>(
            client,
            zero_grid,
            CubeDim::new_1d(zero_dim),
            ArrayArg::from_raw_parts(accum.clone(), frame_len),
            ArrayArg::from_raw_parts(wsum.clone(), pixels),
            pixels as u32,
            stored_ch,
        );

        collab_group_temporal::launch_unchecked::<R>(
            client,
            group_grid.clone(),
            group_dim,
            stored_ch as usize,
            ArrayArg::from_raw_parts(ring_buf.clone(), noisy_centre.len()),
            ArrayArg::from_raw_parts(mv_dummy, 1),
            ArrayArg::from_raw_parts(conf_dummy, 1),
            ArrayArg::from_raw_parts(member_pos.clone(), pos_len),
            ArrayArg::from_raw_parts(member_frame.clone(), pos_len),
            ArrayArg::from_raw_parts(member_count.clone(), refs),
            centre_slot,
            ArrayArg::from_raw_parts(neighbour_slots_dummy, 1),
            0.0f32,
            c_min,
            0u32,
            refine,
            1u32,
            1u32,
            8u32,
            1u32,
            1u32,
            w,
            h,
            channels_count,
            k_max,
            spatial_radius,
            refs_x,
        );

        collab_filter_ht::launch_unchecked::<R>(
            client,
            group_grid,
            group_dim,
            stored_ch as usize,
            ArrayArg::from_raw_parts(ring_buf, noisy_centre.len()),
            ArrayArg::from_raw_parts(member_pos, pos_len),
            ArrayArg::from_raw_parts(member_frame, pos_len),
            ArrayArg::from_raw_parts(member_count, refs),
            ArrayArg::from_raw_parts(member_sig2_dummy, 1),
            ArrayArg::from_raw_parts(accum.clone(), frame_len),
            ArrayArg::from_raw_parts(wsum.clone(), pixels),
            ArrayArg::from_raw_parts(filtered_dummy, 1),
            ArrayArg::from_raw_parts(group_weight, refs),
            centre_slot,
            ArrayArg::from_raw_parts(sigma_buf, stored_ch as usize),
            ArrayArg::from_raw_parts(dct_profile_buf, 8),
            lambda_ht,
            wnorm,
            false,
            false,
            false,
            true,
            w,
            h,
            channels_count,
            k_max,
            stored_ch,
            refs_x,
        );

        collab_normalise::launch_unchecked::<R>(
            client,
            agg_grid,
            agg_dim,
            stored_ch as usize,
            ArrayArg::from_raw_parts(accum, frame_len),
            ArrayArg::from_raw_parts(wsum, pixels),
            ArrayArg::from_raw_parts(output.clone(), frame_len),
            w,
            h,
            channels_count,
            stored_ch,
        );
    }

    let bytes = client.read_one(output).expect("readback failed");
    f32::from_bytes(&bytes).to_vec()
}

/// The hypothesis under test, as a unit assertion: on a static clip
/// where temporal candidates are near-duplicates, grouping across the
/// temporal window should cancel more grain than a spatial-only search
/// on the same frame ever could.
///
/// Both arms run the exact same grouping and filtering kernels, at the
/// same `spatial_radius`, `c_min`, `lambda_ht`, and fixed `sigma`, over
/// the identical noisy centre frame. The only difference is whether
/// `collab_group_temporal` has a temporal window to search.
#[test]
fn temporal_grouping_beats_spatial_only_on_a_static_clip() {
    let client = make_client();
    let (w, h) = (64u32, 64u32);
    let radius = 2u32;
    let base = textured_base(w, h);

    // The radius-2 window is 5 frames; the 3rd (index 2, seed 2) is the
    // one `denoise_submit` denoises once the window is exactly full.
    let noisy_frames: Vec<Vec<f32>> = (0..(2 * radius + 1)).map(|seed| noisy_copy_of(&base, w, h, SIGMA, seed)).collect();
    let centre_index = radius as usize;

    let params = static_clip_params(radius);
    let mut d = Nl4dDenoiser::<R>::new(&client, params, w, h).expect("construction failed");
    for frame in &noisy_frames {
        d.push_frame(frame);
    }
    let temporal_out = d
        .denoise()
        .expect("denoise failed")
        .expect("window is exactly full, denoise should emit");

    let spatial_out = run_spatial_only(
        &client,
        &noisy_frames[centre_index],
        w,
        h,
        SPATIAL_RADIUS,
        REFINE,
        C_MIN,
        LAMBDA_HT,
        SIGMA,
    );

    let temporal_psnr = psnr(&temporal_out, &base);
    let spatial_psnr = psnr(&spatial_out, &base);

    eprintln!(
        "temporal_grouping_beats_spatial_only_on_a_static_clip: radius=2 PSNR={temporal_psnr:.4} dB, \
         radius=0 PSNR={spatial_psnr:.4} dB, delta={:.4} dB",
        temporal_psnr - spatial_psnr
    );

    assert!(
        temporal_psnr > spatial_psnr + 0.5,
        "expected the radius-2 arm to beat the radius-0 arm by at least 0.5 dB, got \
         radius-2={temporal_psnr:.4} dB radius-0={spatial_psnr:.4} dB"
    );
}

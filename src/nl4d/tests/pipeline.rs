use cubecl::prelude::*;

use super::helpers::{R, make_client, noisy_copy_of, psnr, textured_base};
use crate::collab::MAX_K;
use crate::collab::geometry::{filtered_buf_len, member_buf_len, ref_count, refs_along};
use crate::collab::kernels::aggregate::{ACCUM_SCALE, collab_normalise, collab_zero_accum, weight_scale};
use crate::collab::kernels::filter_ht::collab_filter_ht;
use crate::collab::kernels::group_temporal::collab_group_temporal;
use crate::collab::kernels::transforms::dct_noise_profile;
use crate::nl4d::{Nl4dDenoiser, Nl4dParams};
use crate::nlmeans::{
    ChannelMode,
    HqParams,
    MotionCompensationMode,
    MotionEstimation,
    NlmDenoiser,
    NlmParams,
};

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
        },
        temporal_radius,
        refine: REFINE,
        spatial_radius: SPATIAL_RADIUS,
        lambda_ht: LAMBDA_HT,
        c_min: C_MIN,
        mismatch_scale: 1.0,
        confidence_variance: true,
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

    let noisy_frames: Vec<Vec<f32>> = (0..n as u32)
        .map(|seed| noisy_copy_of(&base, w, h, SIGMA, seed))
        .collect();

    let params = static_clip_params(radius);
    let mut d = Nl4dDenoiser::<R>::new(&client, params, w, h).expect("construction failed");

    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for frame in &noisy_frames {
        d.push_frame(frame);
        if let Some(pending) = d.denoise_submit().expect("denoise_submit failed") {
            outputs.push(pending.wait().expect("readback failed"));
        }
    }
    d.flush(|frame| outputs.push(frame.to_vec()))
        .expect("flush failed");

    assert_eq!(outputs.len(), n, "expected one emitted frame per pushed frame");

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

/// `spatial_radius = 16` combined with `temporal_radius =
/// MAX_TEMPORAL_RADIUS` (8) is the configuration that overflowed `i32`
/// under the old fixed `CROSS_FRAME_ACCUM_SCALE = 2^15`: the worst-case
/// cross-frame accumulator value at that combination clears `i32::MAX`,
/// so the old scale silently wrapped it into a wildly wrong pixel with
/// no crash and nothing in the output to flag it.
/// `cross_frame_accum_scale` derives a scale from the real configuration
/// instead, so this combination should denoise cleanly, the same way any
/// other combination does, rather than producing the non-finite or wildly
/// out-of-range values an overflow leaves behind.
#[test]
fn denoises_at_the_previously_overflowing_spatial_and_temporal_radius() {
    let client = make_client();
    let (w, h) = (64u32, 64u32);
    let radius = crate::collab::MAX_TEMPORAL_RADIUS;
    let base = textured_base(w, h);
    let n = 3usize;

    let noisy_frames: Vec<Vec<f32>> = (0..n as u32)
        .map(|seed| noisy_copy_of(&base, w, h, SIGMA, seed))
        .collect();

    let params = Nl4dParams {
        spatial_radius: 16,
        ..static_clip_params(radius)
    };
    let mut d = Nl4dDenoiser::<R>::new(&client, params, w, h).expect("construction failed");

    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for frame in &noisy_frames {
        d.push_frame(frame);
        if let Some(pending) = d.denoise_submit().expect("denoise_submit failed") {
            outputs.push(pending.wait().expect("readback failed"));
        }
    }
    d.flush(|frame| outputs.push(frame.to_vec()))
        .expect("flush failed");

    assert_eq!(outputs.len(), n, "expected one emitted frame per pushed frame");

    for (i, out) in outputs.iter().enumerate() {
        // An `i32` overflow wraps the fixed-point accumulator into a huge
        // or negative value, which `collab_normalise` then divides
        // through into non-finite or wildly out-of-range output. Checking
        // finiteness first gives a clearer failure than letting a bad
        // value fall through into the PSNR comparison below.
        assert!(
            out.iter().all(|v| v.is_finite()),
            "frame {i}: output contains non-finite values, a symptom of the accumulator \
             overflow this test guards against"
        );

        let noisy_psnr = psnr(&noisy_frames[i], &base);
        let out_psnr = psnr(out, &base);
        assert!(
            out_psnr > noisy_psnr,
            "frame {i}: expected a PSNR improvement over the noisy input at spatial_radius=16, \
             temporal_radius={radius}, got noisy={noisy_psnr:.4} dB denoised={out_psnr:.4} dB"
        );
    }
}

/// Guards `run_collab_stage`'s pass-0 accumulator zero against the GPU's
/// per-dimension dispatch limit.
///
/// A single 1D dispatch has to stay at or under 65,535 workgroups on
/// every backend this project targets. Pass 0 used to zero the whole
/// cross-frame ring in one such dispatch, sized for `accum_ring_len`
/// (`width * height * stored_ch * (1 + 2 * temporal_radius)` elements at
/// 256 threads per workgroup), so a resolution and `temporal_radius`
/// combination large enough pushed that dispatch over the limit. A GPU
/// that rejects the oversized dispatch leaves the ring holding
/// `client.empty`'s undefined memory instead of zero, which every
/// subsequent pass's atomic scatter then adds real contributions on top
/// of, and `collab_normalise` divides through into wildly wrong output
/// for whichever frames complete before that garbage is ever cleared.
///
/// `1024 * 1024` at `temporal_radius = MAX_TEMPORAL_RADIUS` (a
/// `1 + 2 * 8 = 17`-frame ring) needs `1024 * 1024 * 17 = 17,825,792`
/// accumulator elements, `69,632` workgroups at 256 threads each,
/// comfortably over the limit. This is also the exact failure mode a
/// real run hit at `temporal_radius = 4` on a 1080p input, `72,900`
/// workgroups for the luma plane alone.
///
/// The frame count pushes the ring past a second full cycle
/// (`2 * total_frames + 3`), so this also stands as a regression guard
/// for slot reuse across more than one lap of the ring, not only the
/// pass-0 dispatch itself.
#[test]
fn survives_a_ring_size_that_would_overflow_a_single_zero_dispatch() {
    let client = make_client();
    let (w, h) = (1024u32, 1024u32);
    let radius = crate::collab::MAX_TEMPORAL_RADIUS;
    let base = textured_base(w, h);
    let total_frames = 1 + 2 * radius;
    let n = (2 * total_frames + 3) as usize;

    let noisy_frames: Vec<Vec<f32>> = (0..n as u32)
        .map(|seed| noisy_copy_of(&base, w, h, SIGMA, seed))
        .collect();

    let params = Nl4dParams {
        // Cheaper than the module defaults so the test spends its time
        // on the ring size under test, not on a wide spatial search
        // this bug has nothing to do with.
        spatial_radius: 2,
        refine: 1,
        ..static_clip_params(radius)
    };
    let mut d = Nl4dDenoiser::<R>::new(&client, params, w, h).expect("construction failed");

    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for frame in &noisy_frames {
        d.push_frame(frame);
        if let Some(pending) = d.denoise_submit().expect("denoise_submit failed") {
            outputs.push(pending.wait().expect("readback failed"));
        }
    }
    d.flush(|frame| outputs.push(frame.to_vec()))
        .expect("flush failed");

    assert_eq!(outputs.len(), n, "expected one emitted frame per pushed frame");

    for (i, out) in outputs.iter().enumerate() {
        assert!(
            out.iter().all(|v| v.is_finite()),
            "frame {i}: output contains non-finite values, a symptom of the pass-0 dispatch \
             this test guards against leaving the accumulator ring unzeroed"
        );

        let noisy_psnr = psnr(&noisy_frames[i], &base);
        let out_psnr = psnr(out, &base);
        assert!(
            out_psnr > noisy_psnr,
            "frame {i}: expected a PSNR improvement over the noisy input, got noisy={noisy_psnr:.4} dB \
             denoised={out_psnr:.4} dB; a worse-than-noisy result is what leftover garbage in the \
             accumulator ring looks like once collab_normalise divides through it"
        );
    }
}

/// Guards the `centre_slot`/`frame` contract `run_collab_stage` depends
/// on: `collab_group_temporal`'s `centre_slot` and `collab_filter_ht`'s
/// `frame` must read the same physical ring slot. Nothing in the type
/// system enforces that (both are plain `u32`s read from two separate
/// call sites), so this test plants content only one real frame carries
/// and checks it survives in that frame's own emitted output.
///
/// `denoises_a_static_noisy_clip` is a weak canary for this specific
/// mismatch: every ring slot there holds the same base content, so
/// feeding the filter a valid but wrong slot would barely move its PSNR
/// (a neighbour frame denoises to essentially the same clean content).
/// Here one frame alone carries a strong, low-frequency marker block no
/// other frame has. If `collab_filter_ht` ever read a different slot
/// than `collab_group_temporal`'s `centre_slot`, none of the marker's
/// own members would ever be grouped as part of that pass, and the
/// marker would be attenuated or absent from its frame's own completed
/// output.
///
/// Emission now lags `temporal_radius` passes behind the pass a frame is
/// the centre of (see [`Nl4dDenoiser::run_collab_stage`]), so this
/// pushes `3 * radius + 1` frames, interleaving a `denoise_submit` after
/// every push the way a real caller does, and collects every emitted
/// output in order. Emitted output `k` is always real frame `k`'s own
/// completed region (see that same doc comment for why), so the marker,
/// planted on real frame `radius`, is checked against emitted output
/// `radius`, not the first or only output the way a single-pass design
/// would have let this test check.
///
/// The marker is a big flat block, not fine detail, so ordinary
/// shrinkage cannot legitimately remove it, and the assertion checks a
/// wide margin rather than an exact value, so ordinary filtering noise
/// does not trip it.
#[test]
fn output_carries_its_own_frames_marker_no_other_frame_has() {
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

    // The pass centred on real frame `radius` only finishes contributing
    // to real frame `radius`'s own output `radius` passes later, and the
    // pass centred on frame `f` runs once `f + radius + 1` real frames
    // have been pushed, so `3 * radius + 1` real pushes are exactly
    // enough to reach that emission without needing `flush` too.
    let marker_frame = radius;
    let n_frames = 3 * radius + 1;
    let frames: Vec<Vec<f32>> = (0..n_frames)
        .map(|seed| {
            let content = if seed == marker_frame {
                &marker_clean
            } else {
                &base
            };
            noisy_copy_of(content, w, h, SIGMA, seed)
        })
        .collect();

    let params = static_clip_params(radius);
    let mut d = Nl4dDenoiser::<R>::new(&client, params, w, h).expect("construction failed");
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for frame in &frames {
        d.push_frame(frame);
        if let Some(pending) = d.denoise_submit().expect("denoise_submit failed") {
            outputs.push(pending.wait().expect("readback failed"));
        }
    }

    let out = outputs
        .get(marker_frame as usize)
        .expect("enough frames were pushed for frame `radius`'s own output to have emitted");

    let mut sum = 0.0f64;
    let mut count = 0usize;
    for y in (MARKER_Y0 + INTERIOR_MARGIN)..(MARKER_Y0 + MARKER_SIZE - INTERIOR_MARGIN) {
        for x in (MARKER_X0 + INTERIOR_MARGIN)..(MARKER_X0 + MARKER_SIZE - INTERIOR_MARGIN) {
            sum += out[(y * w + x) as usize] as f64;
            count += 1;
        }
    }
    let mean = sum / count as f64;
    eprintln!("output_carries_its_own_frames_marker_no_other_frame_has: marker interior mean = {mean:.4}");

    assert!(
        mean > 0.75,
        "expected frame {marker_frame}'s marker (planted at {MARKER}) to survive denoising with a \
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
    // `collab_group_temporal` always writes this, whatever `radius` is,
    // so it needs its real `pos_len` size here even though this
    // function's own `collab_filter_ht` call below never reads it back
    // (that call keeps `use_member_sigma` off and passes
    // `member_sig2_dummy` instead, unaffected by this buffer).
    let member_sig2 = client.empty(pos_len * size_of::<f32>());
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
    let zero_workgroups = (frame_len as u32).div_ceil(zero_dim);
    let zero_grid = CubeCount::new_1d(zero_workgroups);

    let wnorm = weight_scale(sigma, &dct_profile);
    let centre_slot = 0u32;

    unsafe {
        collab_zero_accum::launch_unchecked::<R>(
            client,
            zero_grid,
            CubeDim::new_1d(zero_dim),
            ArrayArg::from_raw_parts(accum.clone(), frame_len),
            ArrayArg::from_raw_parts(wsum.clone(), pixels),
            0u32,
            pixels as u32,
            stored_ch,
            zero_workgroups * zero_dim,
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
            ArrayArg::from_raw_parts(member_sig2, pos_len),
            centre_slot,
            ArrayArg::from_raw_parts(neighbour_slots_dummy, 1),
            0.0f32,
            c_min,
            // `radius` is 0 below, so no temporal candidate is ever
            // scored and neither of these runtime scalars is ever read.
            0.0f32,
            0.0f32,
            0u32,
            refine,
            1u32,
            1u32,
            8u32,
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
            ACCUM_SCALE,
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
            0u32,
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
///
/// `denoise_submit` is called after every push, exactly the way a real
/// caller drives this denoiser, and every emitted output is collected in
/// order. Emitted output `k` is real frame `k`'s own completed region
/// (see [`Nl4dDenoiser::run_collab_stage`]'s scheduling), which is only
/// ready `radius` passes after the pass centred on frame `k` itself, so
/// `3 * radius + 1` frames are pushed rather than the `2 * radius + 1` a
/// single-pass design would have needed.
#[test]
fn temporal_grouping_beats_spatial_only_on_a_static_clip() {
    let client = make_client();
    let (w, h) = (64u32, 64u32);
    let radius = 2u32;
    let base = textured_base(w, h);

    let noisy_frames: Vec<Vec<f32>> = (0..(3 * radius + 1))
        .map(|seed| noisy_copy_of(&base, w, h, SIGMA, seed))
        .collect();
    let centre_index = radius as usize;

    let params = static_clip_params(radius);
    let mut d = Nl4dDenoiser::<R>::new(&client, params, w, h).expect("construction failed");
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for frame in &noisy_frames {
        d.push_frame(frame);
        if let Some(pending) = d.denoise_submit().expect("denoise_submit failed") {
            outputs.push(pending.wait().expect("readback failed"));
        }
    }
    let temporal_out = outputs
        .get(centre_index)
        .expect("enough frames were pushed for frame `radius`'s own output to have emitted")
        .clone();

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

/// Isolates cross-frame aggregation's own contribution, apart from
/// temporal grouping's.
///
/// Both arms here group across the identical radius-2 temporal window,
/// with the identical `lambda_ht`, and read the identical filtered
/// member patches straight off the GPU (`emit_filtered = true`). They
/// differ only in which of those filtered members reach the aggregate
/// for real frame `radius`'s own output: every member that ever matched
/// into it, whatever pass found it (cross-frame, what `Nl4dDenoiser`
/// does now), against only the members from the one pass whose own
/// centre is real frame `radius`, and only the ones among those whose
/// own `member_frame` entry is that pass's `centre_slot` (centre-only,
/// this task's starting point, discarding every neighbour-frame
/// member's filtered pixels after they had served the group's shared
/// statistics).
///
/// The cross-frame arm is `Nl4dDenoiser` itself, already proven correct
/// by the tests above. The centre-only arm reconstructs the old discard
/// by hand, driving the same front end (`NlmDenoiser::submit_machinery`)
/// directly and aggregating only that one pass's centre-frame members,
/// on the host, in double precision, from the GPU's raw
/// `member_pos`/`member_frame`/`member_count`/`group_weight`/`filtered`
/// outputs. That reproduces exactly what a single-pass, centre-only
/// design would have produced for this frame, since a member's own
/// filtered pixels never depended on whether the scatter kept it, only
/// the choice of what to aggregate did.
#[test]
fn cross_frame_aggregation_beats_centre_only_at_the_same_lambda() {
    let client = make_client();
    let (w, h) = (64u32, 64u32);
    let radius = 2u32;
    let base = textured_base(w, h);

    let n_frames = 3 * radius + 1;
    let frames: Vec<Vec<f32>> = (0..n_frames)
        .map(|seed| noisy_copy_of(&base, w, h, SIGMA, seed))
        .collect();
    let judged_frame = radius as usize;

    // Cross-frame arm: the real denoiser, unmodified, driven the same
    // way every other test above drives it.
    let params = static_clip_params(radius);
    let mut d = Nl4dDenoiser::<R>::new(&client, params, w, h).expect("construction failed");
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for frame in &frames {
        d.push_frame(frame);
        if let Some(pending) = d.denoise_submit().expect("denoise_submit failed") {
            outputs.push(pending.wait().expect("readback failed"));
        }
    }
    let cross_frame_out = outputs
        .get(judged_frame)
        .expect("enough frames were pushed for frame `radius`'s own output to have emitted")
        .clone();

    // Centre-only arm: the same front end, driven directly rather than
    // through `Nl4dDenoiser`, so the pass centred on real frame `radius`
    // can be aggregated by hand once it runs.
    let mut nlm_params = static_clip_params(radius).nlm;
    nlm_params.temporal_radius = radius;
    let mut front = NlmDenoiser::<R>::new(&client, nlm_params, w, h);

    let pixels = (w * h) as usize;
    let refs_x = refs_along(w);
    let refs_y = refs_along(h);
    let refs = ref_count(w, h);
    let k_max = MAX_K;
    let pos_len = member_buf_len(w, h, k_max);
    let filt_len = filtered_buf_len(w, h, k_max);
    let total_frames = 1 + 2 * radius;

    let mut centre_only_out: Option<Vec<f32>> = None;
    let mut pass_index = 0u32;
    for frame in &frames {
        front.push_frame(frame);
        let Some(view) = front.submit_machinery().expect("submit_machinery failed") else {
            continue;
        };
        if pass_index != radius {
            pass_index += 1;
            continue;
        }

        let centre_slot = view.centre_slot;
        let ring_len = pixels * total_frames as usize;
        let neighbours = 2 * radius;
        let mv_len = (neighbours * view.mv_stride) as usize;
        let conf_len = (neighbours * view.conf_stride) as usize;
        let neighbour_slots_buf = client.create_from_slice(u32::as_bytes(&view.neighbour_slots));

        let sigmas = front.current_sigmas_temporal_only();
        let sigma_buf = client.create_from_slice(f32::as_bytes(&[sigmas[0]]));
        let profile = dct_noise_profile(0.0);
        let profile_buf = client.create_from_slice(f32::as_bytes(&profile));
        let wnorm = weight_scale(sigmas[0], &profile);

        let member_pos = client.empty(pos_len * size_of::<u32>());
        let member_frame = client.empty(pos_len * size_of::<u32>());
        let member_count = client.empty(refs * size_of::<u32>());
        // `collab_group_temporal` always writes this, and the
        // `collab_filter_ht` call below keeps `use_member_sigma` off
        // and reads `member_sig2_dummy` instead, unaffected by it, the
        // same split `run_spatial_only` above uses.
        let member_sig2 = client.empty(pos_len * size_of::<f32>());
        let member_sig2_dummy = client.empty(size_of::<f32>());
        let filtered_buf = client.empty(filt_len * size_of::<f32>());
        let group_weight = client.empty(refs * size_of::<f32>());
        // The kernel's scatter is unconditional now, so these have to be
        // sized for the whole ring even though this test never reads
        // them back, or the scatter's own writes land out of bounds.
        let accum_scratch = client.empty(pixels * total_frames as usize * size_of::<i32>());
        let wsum_scratch = client.empty(pixels * total_frames as usize * size_of::<i32>());

        let mc = front.motion_ctx();
        let group_grid = CubeCount::new_2d(refs_x, refs_y);
        let group_dim = CubeDim::new_2d(8, 8);

        unsafe {
            collab_group_temporal::launch_unchecked::<R>(
                &client,
                group_grid.clone(),
                group_dim,
                1usize,
                ArrayArg::from_raw_parts(view.input.clone(), ring_len),
                ArrayArg::from_raw_parts(view.mv_field.clone(), mv_len.max(1)),
                ArrayArg::from_raw_parts(view.confidence.clone(), conf_len.max(1)),
                ArrayArg::from_raw_parts(member_pos.clone(), pos_len),
                ArrayArg::from_raw_parts(member_frame.clone(), pos_len),
                ArrayArg::from_raw_parts(member_count.clone(), refs),
                ArrayArg::from_raw_parts(member_sig2, pos_len),
                centre_slot,
                ArrayArg::from_raw_parts(neighbour_slots_buf, view.neighbour_slots.len().max(1)),
                0.0f32,
                C_MIN,
                front.thsad_value(),
                1.0f32,
                radius,
                REFINE,
                view.mv_stride,
                view.conf_stride,
                mc.step,
                mc.blksize,
                mc.blocks_x,
                mc.blocks_y,
                w,
                h,
                1u32,
                k_max,
                SPATIAL_RADIUS,
                refs_x,
            );

            collab_filter_ht::launch_unchecked::<R>(
                &client,
                group_grid,
                group_dim,
                1usize,
                ArrayArg::from_raw_parts(view.input.clone(), ring_len),
                ArrayArg::from_raw_parts(member_pos.clone(), pos_len),
                ArrayArg::from_raw_parts(member_frame.clone(), pos_len),
                ArrayArg::from_raw_parts(member_count.clone(), refs),
                ArrayArg::from_raw_parts(member_sig2_dummy, 1),
                ArrayArg::from_raw_parts(accum_scratch, pixels * total_frames as usize),
                ArrayArg::from_raw_parts(wsum_scratch, pixels * total_frames as usize),
                ArrayArg::from_raw_parts(filtered_buf.clone(), filt_len),
                ArrayArg::from_raw_parts(group_weight.clone(), refs),
                centre_slot,
                ArrayArg::from_raw_parts(sigma_buf, 1),
                ArrayArg::from_raw_parts(profile_buf, 8),
                LAMBDA_HT,
                wnorm,
                ACCUM_SCALE,
                false,
                true,
                false,
                true,
                w,
                h,
                1u32,
                k_max,
                1u32,
                refs_x,
            );
        }

        let member_pos_host =
            u32::from_bytes(&client.read_one(member_pos).expect("member_pos readback failed"))[..pos_len]
                .to_vec();
        let member_frame_host = u32::from_bytes(
            &client
                .read_one(member_frame)
                .expect("member_frame readback failed"),
        )[..pos_len]
            .to_vec();
        let member_count_host = u32::from_bytes(
            &client
                .read_one(member_count)
                .expect("member_count readback failed"),
        )[..refs]
            .to_vec();
        let group_weight_host = f32::from_bytes(
            &client
                .read_one(group_weight)
                .expect("group_weight readback failed"),
        )[..refs]
            .to_vec();
        let filtered_host =
            f32::from_bytes(&client.read_one(filtered_buf).expect("filtered readback failed"))[..filt_len]
                .to_vec();

        let mut sum = vec![0.0f64; pixels];
        let mut wsum = vec![0.0f64; pixels];
        for r in 0..refs {
            let k_use = member_count_host[r] as usize;
            let weight = group_weight_host[r] as f64;
            for m in 0..k_use {
                let idx = r * k_max as usize + m;
                if member_frame_host[idx] != centre_slot {
                    continue;
                }
                let packed = member_pos_host[idx];
                let mx = packed & 0xFFFF;
                let my = packed >> 16;
                let line_base = idx * 64;
                for row in 0..8u32 {
                    for col in 0..8u32 {
                        let px = (mx + col) as usize;
                        let py = (my + row) as usize;
                        let pix = py * w as usize + px;
                        let val = filtered_host[line_base + (row * 8 + col) as usize] as f64;
                        sum[pix] += val * weight;
                        wsum[pix] += weight;
                    }
                }
            }
        }
        let mut out = vec![0.0f32; pixels];
        for i in 0..pixels {
            assert!(
                wsum[i] > 0.0,
                "pixel {i}: centre-only arm left it with no contribution at all"
            );
            out[i] = (sum[i] / wsum[i]) as f32;
        }
        centre_only_out = Some(out);
        break;
    }

    let centre_only_out = centre_only_out.expect("the pass centred on real frame `radius` must have run");

    let cross_frame_psnr = psnr(&cross_frame_out, &base);
    let centre_only_psnr = psnr(&centre_only_out, &base);

    eprintln!(
        "cross_frame_aggregation_beats_centre_only_at_the_same_lambda: cross-frame PSNR={cross_frame_psnr:.4} \
         dB, centre-only PSNR={centre_only_psnr:.4} dB, delta={:.4} dB",
        cross_frame_psnr - centre_only_psnr
    );

    assert!(
        cross_frame_psnr > centre_only_psnr,
        "expected cross-frame aggregation to remove more noise than centre-only at the same \
         lambda_ht, got cross-frame={cross_frame_psnr:.4} dB centre-only={centre_only_psnr:.4} dB"
    );
}

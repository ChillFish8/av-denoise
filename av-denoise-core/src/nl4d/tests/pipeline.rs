use cubecl::prelude::*;

use super::helpers::{R, make_client, noisy_copy_of, psnr, textured_base};
use crate::collab::MAX_K;
use crate::collab::geometry::{fused_cubes_x, ref_count, refs_along};
use crate::collab::kernels::aggregate::{
    ACCUM_SCALE,
    collab_normalise,
    collab_zero_accum,
    cross_frame_accum_scale,
    weight_scale,
};
use crate::collab::kernels::fused::collab_fused;
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
            let frame = pending.wait().expect("readback failed");
            outputs.push(frame.into_f32().expect("f32 output"));
        }
    }
    d.flush(|frame| outputs.push(frame.as_f32().expect("f32 denoiser").to_vec()))
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

/// `spatial_radius = 16` with `temporal_radius = MAX_TEMPORAL_RADIUS` (8)
/// is the widest configuration the parameter ranges allow, and so the one
/// whose cross-frame accumulator comes closest to overflowing `i32`.
///
/// `cross_frame_accum_scale` sizes the fixed-point scale for it, so this
/// combination should denoise as cleanly as any other rather than
/// producing the non-finite or wildly out-of-range values an overflow
/// leaves behind.
#[test]
fn denoises_at_the_widest_spatial_and_temporal_radius() {
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
            let frame = pending.wait().expect("readback failed");
            outputs.push(frame.into_f32().expect("f32 output"));
        }
    }
    d.flush(|frame| outputs.push(frame.as_f32().expect("f32 denoiser").to_vec()))
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
/// every backend this project targets. Zeroing the whole cross-frame ring
/// in one dispatch would need `accum_ring_len` slots
/// (`width * height * stored_ch * (1 + 2 * temporal_radius)`) at 256
/// threads per workgroup, which a large enough resolution and
/// `temporal_radius` pushes over that limit.
///
/// A GPU that rejects an oversized dispatch leaves the ring holding
/// `client.empty`'s undefined memory instead of zero. Every later pass
/// then scatters real contributions on top of that, and
/// `collab_normalise` divides it through into wildly wrong output.
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
            let frame = pending.wait().expect("readback failed");
            outputs.push(frame.into_f32().expect("f32 output"));
        }
    }
    d.flush(|frame| outputs.push(frame.as_f32().expect("f32 denoiser").to_vec()))
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

/// Guards the `centre_slot` contract `run_collab_stage` depends on. The
/// slot the pass is centred on is what the reference patch is read from
/// and what an untouched member scatters back into, and nothing in the
/// type system pins it to the frame the caller means, so this test
/// plants content only one real frame carries and checks it survives in
/// that frame's own emitted output.
///
/// `denoises_a_static_noisy_clip` is a weak canary for this specific
/// mismatch: every ring slot there holds the same base content, so
/// feeding the filter a valid but wrong slot would barely move its PSNR
/// (a neighbour frame denoises to essentially the same clean content).
/// Here one frame alone carries a strong, low-frequency marker block no
/// other frame has. If the pass ever centred on a different physical
/// ring slot, none of the marker's own patches would be a reference at
/// all, and the marker would be attenuated or absent from its frame's
/// own completed output.
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
            let frame = pending.wait().expect("readback failed");
            outputs.push(frame.into_f32().expect("f32 output"));
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
         marker's interior; this would fail if the collaborative pass ever centred on a \
         different physical ring slot than the caller meant"
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

/// Launches the same collaborative and aggregation kernels
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
    let pixels = (w * h) as usize;
    let frame_len = pixels;

    let ring_buf = client.create_from_slice(f32::as_bytes(noisy_centre));
    let mv_dummy = client.empty(size_of::<i32>());
    let conf_dummy = client.empty(size_of::<f32>());
    let neighbour_slots_dummy = client.empty(size_of::<u32>());
    let group_weight = client.empty(refs * size_of::<f32>());
    let sigma_buf = client.create_from_slice(f32::as_bytes(&[sigma]));
    let dct_profile = dct_noise_profile(0.0);
    let dct_profile_buf = client.create_from_slice(f32::as_bytes(&dct_profile));
    let accum = client.empty(frame_len * size_of::<i32>());
    let wsum = client.empty(pixels * size_of::<i32>());
    let output = client.empty(frame_len * size_of::<f32>());

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

        collab_fused::launch_unchecked::<R>(
            client,
            CubeCount::new_2d(fused_cubes_x(w), refs_y),
            CubeDim::new_1d(64),
            stored_ch as usize,
            ArrayArg::from_raw_parts(ring_buf, noisy_centre.len()),
            ArrayArg::from_raw_parts(mv_dummy, 1),
            ArrayArg::from_raw_parts(conf_dummy, 1),
            ArrayArg::from_raw_parts(neighbour_slots_dummy, 1),
            ArrayArg::from_raw_parts(sigma_buf, stored_ch as usize),
            ArrayArg::from_raw_parts(dct_profile_buf, 8),
            ArrayArg::from_raw_parts(accum.clone(), frame_len),
            ArrayArg::from_raw_parts(wsum.clone(), pixels),
            ArrayArg::from_raw_parts(group_weight, refs),
            centre_slot,
            0.0f32,
            c_min,
            // `radius` is 0 below, so no temporal candidate is ever
            // scored and this runtime scalar is never read.
            0.0f32,
            lambda_ht,
            wnorm,
            ACCUM_SCALE,
            false,
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
            stored_ch,
            spatial_radius,
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
/// Both arms run the exact same kernel, at the same `spatial_radius`,
/// `c_min`, `lambda_ht`, and fixed `sigma`, over the identical noisy
/// centre frame. The only difference is whether the search has a
/// temporal window to look in.
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
            let frame = pending.wait().expect("readback failed");
            outputs.push(frame.into_f32().expect("f32 output"));
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
/// with the identical `lambda_ht`, and run the identical kernel. They
/// differ only in which filtered members reach the aggregate for real
/// frame `radius`'s own output.
///
/// The cross-frame arm is `Nl4dDenoiser` itself, which keeps every member
/// that ever matched into that frame, whatever pass found it.
///
/// The centre-only arm runs the one pass centred on that frame and reads
/// back only the centre slot's own region of the accumulator ring. The
/// kernel scatters each member into the region of the frame it was
/// matched in, so that region holds exactly the members whose own frame
/// is the centre, which is what a single-pass centre-only design would
/// have produced. It is built by driving the same front end
/// (`NlmDenoiser::submit_machinery`) directly, so the ring, the motion
/// field, and the noise estimate are all the ones the real denoiser
/// used.
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
            let frame = pending.wait().expect("readback failed");
            outputs.push(frame.into_f32().expect("f32 output"));
        }
    }
    let cross_frame_out = outputs
        .get(judged_frame)
        .expect("enough frames were pushed for frame `radius`'s own output to have emitted")
        .clone();

    // Centre-only arm: the same front end, driven directly rather than
    // through `Nl4dDenoiser`, so the pass centred on real frame `radius`
    // can be aggregated on its own once it runs.
    let mut nlm_params = static_clip_params(radius).nlm;
    nlm_params.temporal_radius = radius;
    let mut front = NlmDenoiser::<R>::new(&client, nlm_params, w, h);

    let pixels = (w * h) as usize;
    let refs_x = refs_along(w);
    let refs = ref_count(w, h);
    let k_max = MAX_K;
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
        let accum_scale = cross_frame_accum_scale(SPATIAL_RADIUS, radius);

        let group_weight = client.empty(refs * size_of::<f32>());
        // The whole ring, because a member matched in a neighbour frame
        // scatters into that frame's own region. Only the centre slot's
        // region is read back below, which is exactly what makes this
        // the centre-only arm.
        let accum = client.create_from_slice(i32::as_bytes(&vec![0i32; pixels * total_frames as usize]));
        let wsum = client.create_from_slice(i32::as_bytes(&vec![0i32; pixels * total_frames as usize]));
        let output = client.empty(pixels * size_of::<f32>());

        let mc = front.motion_ctx();

        unsafe {
            collab_fused::launch_unchecked::<R>(
                &client,
                CubeCount::new_2d(fused_cubes_x(w), refs_along(h)),
                CubeDim::new_1d(64),
                1usize,
                ArrayArg::from_raw_parts(view.input.clone(), ring_len),
                ArrayArg::from_raw_parts(view.mv_field.clone(), mv_len.max(1)),
                ArrayArg::from_raw_parts(view.confidence.clone(), conf_len.max(1)),
                ArrayArg::from_raw_parts(neighbour_slots_buf, view.neighbour_slots.len().max(1)),
                ArrayArg::from_raw_parts(sigma_buf, 1),
                ArrayArg::from_raw_parts(profile_buf, 8),
                ArrayArg::from_raw_parts(accum.clone(), pixels * total_frames as usize),
                ArrayArg::from_raw_parts(wsum.clone(), pixels * total_frames as usize),
                ArrayArg::from_raw_parts(group_weight, refs),
                centre_slot,
                0.0f32,
                C_MIN,
                front.thsad_value(),
                LAMBDA_HT,
                wnorm,
                accum_scale,
                false,
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
                1u32,
                SPATIAL_RADIUS,
                refs_x,
            );

            collab_normalise::launch_unchecked::<R>(
                &client,
                CubeCount::new_2d(
                    w.div_ceil(crate::nlmeans::BLOCK_X),
                    h.div_ceil(crate::nlmeans::BLOCK_Y),
                ),
                CubeDim::new_2d(crate::nlmeans::BLOCK_X, crate::nlmeans::BLOCK_Y),
                1usize,
                ArrayArg::from_raw_parts(accum, pixels * total_frames as usize),
                ArrayArg::from_raw_parts(wsum, pixels * total_frames as usize),
                ArrayArg::from_raw_parts(output.clone(), pixels),
                centre_slot * pixels as u32,
                w,
                h,
                1u32,
                1u32,
            );
        }

        let out = f32::from_bytes(&client.read_one(output).expect("readback failed"))[..pixels].to_vec();
        assert!(
            out.iter().all(|v| v.is_finite()),
            "the centre-only arm left a pixel with no contribution at all"
        );
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

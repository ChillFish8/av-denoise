use cubecl::prelude::*;

use super::helpers::{
    C_MIN,
    LAMBDA_HT,
    R,
    REFINE,
    SIGMA,
    SPATIAL_RADIUS,
    make_client,
    psnr,
    static_clip_params,
    textured_base,
};
use crate::bench_api::HostIo;
use crate::collab::geometry::{fused_cubes_x, ref_count, refs_along, strength_map_dims};
use crate::collab::kernels::aggregate::{
    ACCUM_SCALE,
    collab_normalise,
    collab_zero_accum,
    cross_frame_accum_scale,
    kaiser_window,
    weight_scale,
};
use crate::collab::kernels::fused::{STRENGTH_MAP_OFF, collab_fused};
use crate::collab::kernels::transforms::dct_noise_profile;
use crate::collab::{
    COLLAB_GROUPS,
    MAX_K,
    MAX_TEMPORAL_RADIUS,
    PATCH_SIZE,
    grid_frames,
    needs_warp_uniform_search,
};
use crate::nl4d::{Nl4dDenoiser, Nl4dParams};
use crate::nlmeans::tests::helpers::noisy_field_over;
use crate::nlmeans::{BLOCK_X, BLOCK_Y, ChannelMode, NOISE_CURVE_BINS, NlmDenoiser, NlmParams};
use crate::tune::collab::{CollabLaunch, CollabParams};
use crate::tune::zeroed;

#[test]
fn denoises_a_static_noisy_clip() {
    let client = make_client();
    let (width, height) = (64u32, 64u32);
    let radius = 2u32;
    let base = textured_base(width, height);
    let frame_count = 9usize;

    let noisy_frames: Vec<Vec<f32>> = (0..frame_count as u32)
        .map(|seed| noisy_field_over(&base, width, height, SIGMA, seed))
        .collect();

    let params = static_clip_params(radius);
    let mut denoiser = Nl4dDenoiser::<R>::new(&client, params, width, height).expect("construction failed");

    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for frame in &noisy_frames {
        denoiser.push_frame(frame);
        if let Some(frame) = denoiser.denoise().expect("denoise failed") {
            outputs.push(frame);
        }
    }

    denoiser
        .flush(|frame| {
            let values = frame.to_vec();
            outputs.push(values);
        })
        .expect("flush failed");

    assert_eq!(
        outputs.len(),
        frame_count,
        "expected one emitted frame per pushed frame"
    );

    for (i, output) in outputs.iter().enumerate() {
        let noisy_psnr = psnr(&noisy_frames[i], &base);
        let out_psnr = psnr(output, &base);
        assert!(
            out_psnr > noisy_psnr + 6.0,
            "frame {i}: expected at least a 6 dB PSNR improvement over the noisy input, got \
             noisy={noisy_psnr:.4} dB denoised={out_psnr:.4} dB"
        );
    }
}

/// The widest radii the parameter ranges allow come closest to overflowing the `i32` cross-frame
/// accumulator, so they must still denoise cleanly.
#[test]
fn denoises_at_the_widest_spatial_and_temporal_radius() {
    let client = make_client();
    let (width, height) = (64u32, 64u32);
    let radius = MAX_TEMPORAL_RADIUS;
    let base = textured_base(width, height);
    let frame_count = 3usize;

    let noisy_frames: Vec<Vec<f32>> = (0..frame_count as u32)
        .map(|seed| noisy_field_over(&base, width, height, SIGMA, seed))
        .collect();

    let clip_params = static_clip_params(radius);
    let params = Nl4dParams {
        spatial_radius: 16,
        ..clip_params
    };
    let mut denoiser = Nl4dDenoiser::<R>::new(&client, params, width, height).expect("construction failed");

    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for frame in &noisy_frames {
        denoiser.push_frame(frame);
        if let Some(frame) = denoiser.denoise().expect("denoise failed") {
            outputs.push(frame);
        }
    }

    denoiser
        .flush(|frame| {
            let values = frame.to_vec();
            outputs.push(values);
        })
        .expect("flush failed");

    assert_eq!(
        outputs.len(),
        frame_count,
        "expected one emitted frame per pushed frame"
    );

    for (i, output) in outputs.iter().enumerate() {
        // An overflow wraps the accumulator into non-finite output, so checking finiteness first
        // gives a clearer failure than the PSNR comparison.
        assert!(
            output.iter().all(|value| value.is_finite()),
            "frame {i}: output contains non-finite values, a symptom of the accumulator \
             overflow this test guards against"
        );

        let noisy_psnr = psnr(&noisy_frames[i], &base);
        let out_psnr = psnr(output, &base);
        assert!(
            out_psnr > noisy_psnr,
            "frame {i}: expected a PSNR improvement over the noisy input at spatial_radius=16, \
             temporal_radius={radius}, got noisy={noisy_psnr:.4} dB denoised={out_psnr:.4} dB"
        );
    }
}

/// Guards the whole-ring accumulator zero against the 65,535 workgroup limit on a single 1D
/// dispatch.
///
/// A rejected dispatch leaves the ring holding `client.empty` memory instead of zero, which
/// normalises into wildly wrong output. `1024 * 1024` at a 17-frame ring needs 69,632 workgroups
/// at 256 threads each. 1080p luma at `temporal_radius = 4` needs 72,900 workgroups. The frame
/// count runs the ring past a second full lap, so this also guards slot reuse.
#[test]
fn survives_a_ring_size_that_would_overflow_a_single_zero_dispatch() {
    let client = make_client();
    let (width, height) = (1024u32, 1024u32);
    let radius = MAX_TEMPORAL_RADIUS;
    let base = textured_base(width, height);
    let total_frames = 1 + 2 * radius;
    let frame_count = (2 * total_frames + 3) as usize;

    let noisy_frames: Vec<Vec<f32>> = (0..frame_count as u32)
        .map(|seed| noisy_field_over(&base, width, height, SIGMA, seed))
        .collect();

    // A narrow search keeps the test's time on the ring size under test.
    let clip_params = static_clip_params(radius);
    let params = Nl4dParams {
        spatial_radius: 2,
        refine: 1,
        ..clip_params
    };
    let mut denoiser = Nl4dDenoiser::<R>::new(&client, params, width, height).expect("construction failed");

    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for frame in &noisy_frames {
        denoiser.push_frame(frame);
        if let Some(frame) = denoiser.denoise().expect("denoise failed") {
            outputs.push(frame);
        }
    }

    denoiser
        .flush(|frame| {
            let values = frame.to_vec();
            outputs.push(values);
        })
        .expect("flush failed");

    assert_eq!(
        outputs.len(),
        frame_count,
        "expected one emitted frame per pushed frame"
    );

    for (i, output) in outputs.iter().enumerate() {
        assert!(
            output.iter().all(|value| value.is_finite()),
            "frame {i}: output contains non-finite values, a symptom of the pass-0 dispatch \
             this test guards against leaving the accumulator ring unzeroed"
        );

        let noisy_psnr = psnr(&noisy_frames[i], &base);
        let out_psnr = psnr(output, &base);
        assert!(
            out_psnr > noisy_psnr,
            "frame {i}: expected a PSNR improvement over the noisy input, got noisy={noisy_psnr:.4} dB \
             denoised={out_psnr:.4} dB; a worse-than-noisy result is what leftover garbage in the \
             accumulator ring looks like once collab_normalise divides through it"
        );
    }
}

/// Each pass must centre on the ring slot of the frame the caller means.
///
/// Only one frame carries a large flat marker block. A pass centred on a different slot would
/// have none of the marker's patches as references, so the marker would fade from its frame's
/// output. A flat block cannot be removed by ordinary shrinkage, and the check uses a wide margin
/// so filtering noise does not trip it.
#[test]
fn output_carries_its_own_frames_marker_no_other_frame_has() {
    let client = make_client();
    let (width, height) = (64u32, 64u32);
    let radius = 2u32;
    let base = textured_base(width, height);

    // `textured_base` never exceeds 0.65, so a marker at 0.92 is unambiguous against it.
    const MARKER: f32 = 0.92;
    const MARKER_X0: u32 = 24;
    const MARKER_Y0: u32 = 24;
    const MARKER_SIZE: u32 = 24;
    // Read only the block's interior, so blending at its edges cannot explain a low reading.
    const INTERIOR_MARGIN: u32 = 8;

    let mut marker_clean = base.clone();
    for y in MARKER_Y0..MARKER_Y0 + MARKER_SIZE {
        for x in MARKER_X0..MARKER_X0 + MARKER_SIZE {
            marker_clean[(y * width + x) as usize] = MARKER;
        }
    }

    // The pass centred on frame `f` runs once `f + radius + 1` frames are pushed, and frame
    // `radius` emits `radius` passes after its own, so `3 * radius + 1` pushes reach it without
    // `flush`.
    let marker_frame = radius;
    let frame_count = 3 * radius + 1;
    let frames: Vec<Vec<f32>> = (0..frame_count)
        .map(|seed| {
            let content = if seed == marker_frame {
                &marker_clean
            } else {
                &base
            };
            noisy_field_over(content, width, height, SIGMA, seed)
        })
        .collect();

    let params = static_clip_params(radius);
    let mut denoiser = Nl4dDenoiser::<R>::new(&client, params, width, height).expect("construction failed");
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for frame in &frames {
        denoiser.push_frame(frame);
        if let Some(frame) = denoiser.denoise().expect("denoise failed") {
            outputs.push(frame);
        }
    }

    // Outputs emit in frame order, so output `k` is real frame `k`'s own completed region.
    let output = outputs
        .get(marker_frame as usize)
        .expect("enough frames were pushed for frame `radius`'s own output to have emitted");

    let mut sum = 0.0f64;
    let mut count = 0usize;
    for y in (MARKER_Y0 + INTERIOR_MARGIN)..(MARKER_Y0 + MARKER_SIZE - INTERIOR_MARGIN) {
        for x in (MARKER_X0 + INTERIOR_MARGIN)..(MARKER_X0 + MARKER_SIZE - INTERIOR_MARGIN) {
            sum += output[(y * width + x) as usize] as f64;
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

#[test]
fn flush_emits_exactly_the_pushed_frame_count() {
    let client = make_client();
    let (width, height) = (64u32, 64u32);
    let radius = 2u32;
    let base = textured_base(width, height);
    let frame_count = 7u32;

    let params = static_clip_params(radius);
    let mut denoiser = Nl4dDenoiser::<R>::new(&client, params, width, height).expect("construction failed");

    let mut emitted = 0usize;
    for seed in 0..frame_count {
        let frame = noisy_field_over(&base, width, height, SIGMA, seed);
        denoiser.push_frame(&frame);
        if denoiser.denoise().expect("denoise failed").is_some() {
            emitted += 1;
        }
    }

    denoiser.flush(|_| emitted += 1).expect("flush failed");

    assert_eq!(
        emitted, frame_count as usize,
        "expected exactly {frame_count} emitted frames"
    );
}

/// Runs the denoiser's collaborative and aggregation kernels standalone on a single-frame ring at
/// `radius = 0`.
///
/// Grouping, filter, noise floor and `c_min` match the denoiser, so the only difference is that
/// no temporal candidates exist to search.
#[expect(
    clippy::too_many_arguments,
    reason = "the test helper takes the full set of parameters its cases vary"
)]
fn run_spatial_only(
    client: &ComputeClient<R>,
    noisy_centre: &[f32],
    width: u32,
    height: u32,
    spatial_radius: u32,
    refine: u32,
    c_min: f32,
    lambda_ht: f32,
    sigma: f32,
    warp_uniform: bool,
) -> Vec<f32> {
    let k_max = MAX_K;
    let stored_ch = 1u32;
    let channels_count = 1u32;
    let refs_x = refs_along(width);
    let refs_y = refs_along(height);
    let refs = ref_count(width, height);
    let pixels = (width * height) as usize;
    let frame_len = pixels;

    let centre_bytes = f32::as_bytes(noisy_centre);
    let ring_buf = client.create_from_slice(centre_bytes);
    let mv_dummy = client.empty(size_of::<i32>());
    let conf_dummy = client.empty(size_of::<f32>());
    let neighbour_slots_dummy = client.empty(size_of::<u32>());
    let group_weight = client.empty(refs * size_of::<f32>());

    let sigma_values = [sigma];
    let dct_profile = dct_noise_profile(0.0);
    let kaiser = kaiser_window(0.0);
    let zeroed_curve = [0.0f32; NOISE_CURVE_BINS];
    let sigma_bytes = f32::as_bytes(&sigma_values);
    let sigma_buf = client.create_from_slice(sigma_bytes);
    let dct_profile_bytes = f32::as_bytes(&dct_profile);
    let dct_profile_buf = client.create_from_slice(dct_profile_bytes);
    let kaiser_bytes = f32::as_bytes(&kaiser);
    let kaiser_buf = client.create_from_slice(kaiser_bytes);
    let curve_bytes = f32::as_bytes(&zeroed_curve);
    let zero_curve = client.create_from_slice(curve_bytes);

    let (map_cols, map_rows) = strength_map_dims(width, height);
    let map_len = (map_cols * map_rows) as usize;
    let unit_map = vec![1.0f32; map_len];
    let unit_map_bytes = f32::as_bytes(&unit_map);
    let unit_map_buf = client.create_from_slice(unit_map_bytes);
    let accum = client.empty(frame_len * size_of::<i32>());
    let wsum = client.empty(pixels * size_of::<i32>());
    let output = client.empty(frame_len * size_of::<f32>());

    let agg_cubes_x = width.div_ceil(BLOCK_X);
    let agg_cubes_y = height.div_ceil(BLOCK_Y);
    let agg_grid = CubeCount::new_2d(agg_cubes_x, agg_cubes_y);
    let agg_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);
    let zero_dim = 256u32;
    let zero_workgroups = (frame_len as u32).div_ceil(zero_dim);
    let zero_grid = CubeCount::new_1d(zero_workgroups);
    let zero_cube_dim = CubeDim::new_1d(zero_dim);

    let fused_x = fused_cubes_x(width);
    let fused_grid = CubeCount::new_2d(fused_x, refs_y);
    let fused_dim = CubeDim::new_1d(64);
    let weight_norm = weight_scale(sigma, &dct_profile);
    let grid_frame_count = grid_frames(0);
    let centre_slot = 0u32;

    unsafe {
        collab_zero_accum::launch_unchecked::<R>(
            client,
            zero_grid,
            zero_cube_dim,
            ArrayArg::from_raw_parts(accum.clone(), frame_len),
            ArrayArg::from_raw_parts(wsum.clone(), pixels),
            0u32,
            pixels as u32,
            stored_ch,
            zero_workgroups * zero_dim,
        );

        collab_fused::launch_unchecked::<f32, R>(
            client,
            fused_grid,
            fused_dim,
            stored_ch as usize,
            ArrayArg::from_raw_parts(ring_buf.clone(), noisy_centre.len()),
            ArrayArg::from_raw_parts(ring_buf, stored_ch as usize),
            ArrayArg::from_raw_parts(mv_dummy, 1),
            ArrayArg::from_raw_parts(conf_dummy, 1),
            ArrayArg::from_raw_parts(neighbour_slots_dummy, 1),
            ArrayArg::from_raw_parts(sigma_buf, stored_ch as usize),
            ArrayArg::from_raw_parts(zero_curve, NOISE_CURVE_BINS),
            ArrayArg::from_raw_parts(unit_map_buf, map_len),
            ArrayArg::from_raw_parts(dct_profile_buf, 8),
            ArrayArg::from_raw_parts(kaiser_buf, PATCH_SIZE as usize),
            ArrayArg::from_raw_parts(accum.clone(), frame_len),
            ArrayArg::from_raw_parts(wsum.clone(), pixels),
            ArrayArg::from_raw_parts(group_weight, refs),
            centre_slot,
            c_min,
            lambda_ht,
            0u32,
            STRENGTH_MAP_OFF,
            weight_norm,
            ACCUM_SCALE,
            warp_uniform,
            false,
            0u32,
            grid_frame_count,
            refine,
            1u32,
            1u32,
            8u32,
            8u32,
            1u32,
            1u32,
            width,
            height,
            channels_count,
            k_max,
            stored_ch,
            spatial_radius,
            refs_x,
            map_cols,
            map_rows,
            0.0f32,
            false,
            COLLAB_GROUPS,
            false,
            false,
            0,
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
            width,
            height,
            channels_count,
            stored_ch,
        );
    }

    let bytes = client.read_one(output).expect("readback failed");
    f32::from_bytes(&bytes).to_vec()
}

/// Every buffer a collab launch writes, read back from the device.
#[derive(Debug, PartialEq)]
pub(crate) struct CollabOutputs {
    pub accum: Vec<i32>,
    pub wsum: Vec<i32>,
    pub group_weight: Vec<f32>,
}

/// The spatial-only `collab_fused` pass of [run_spatial_only] as a tuner launch over a noisy
/// 64x64 frame, with a zero-filled accumulator and a closure that reads it back.
pub(crate) fn single_pass_launch(
    client: &ComputeClient<R>,
) -> (CollabLaunch<R>, impl Fn(&CollabLaunch<R>) -> CollabOutputs) {
    let (width, height) = (64u32, 64u32);
    let stored_ch = 1u32;
    let refs = ref_count(width, height);
    let pixels = (width * height) as usize;
    let frame_len = pixels;

    let base = textured_base(width, height);
    let noisy_centre = noisy_field_over(&base, width, height, SIGMA, 0);
    let centre_bytes = f32::as_bytes(&noisy_centre);
    let ring_buf = client.create_from_slice(centre_bytes);
    let mv_dummy = client.empty(size_of::<i32>());
    let conf_dummy = client.empty(size_of::<f32>());
    let neighbour_slots_dummy = client.empty(size_of::<u32>());
    let group_weight = zeroed(client, refs * size_of::<f32>());

    let sigma_values = [SIGMA];
    let dct_profile = dct_noise_profile(0.0);
    let kaiser = kaiser_window(0.0);
    let zeroed_curve = [0.0f32; NOISE_CURVE_BINS];
    let sigma_bytes = f32::as_bytes(&sigma_values);
    let sigma_buf = client.create_from_slice(sigma_bytes);
    let dct_profile_bytes = f32::as_bytes(&dct_profile);
    let dct_profile_buf = client.create_from_slice(dct_profile_bytes);
    let kaiser_bytes = f32::as_bytes(&kaiser);
    let kaiser_buf = client.create_from_slice(kaiser_bytes);
    let curve_bytes = f32::as_bytes(&zeroed_curve);
    let zero_curve = client.create_from_slice(curve_bytes);

    let (map_cols, map_rows) = strength_map_dims(width, height);
    let map_len = (map_cols * map_rows) as usize;
    let unit_map = vec![1.0f32; map_len];
    let unit_map_bytes = f32::as_bytes(&unit_map);
    let unit_map_buf = client.create_from_slice(unit_map_bytes);
    let zero_accum = vec![0i32; frame_len];
    let zero_accum_bytes = i32::as_bytes(&zero_accum);
    let accum = client.create_from_slice(zero_accum_bytes);
    let wsum = zeroed(client, pixels * size_of::<i32>());

    let params = CollabParams {
        stored_ch,
        centre_slot: 0,
        c_min: C_MIN,
        lambda_ht: LAMBDA_HT,
        curve_valid: 0,
        map_mode: STRENGTH_MAP_OFF,
        weight_scale: weight_scale(SIGMA, &dct_profile),
        accum_scale: ACCUM_SCALE,
        warp_uniform: needs_warp_uniform_search(client),
        f16_search: false,
        radius: 0,
        grid_frames: grid_frames(0),
        refine: REFINE,
        mv_stride: 1,
        conf_stride: 1,
        blk_step: 8,
        blksize: 8,
        blocks_x: 1,
        blocks_y: 1,
        width,
        height,
        channels: 1,
        k_max: MAX_K,
        spatial_radius: SPATIAL_RADIUS,
        refs_x: refs_along(width),
        refs_y: refs_along(height),
        map_cols,
        map_rows,
        pool_ratio: 0.0,
        pooled: false,
    };

    let collab = CollabLaunch {
        client: client.clone(),
        ring: ring_buf.clone(),
        ring_len: noisy_centre.len(),
        search_ring: ring_buf,
        search_len: stored_ch as usize,
        mv_field: mv_dummy,
        mv_len: 1,
        confidence: conf_dummy,
        conf_len: 1,
        neighbour_slots: neighbour_slots_dummy,
        neighbour_slots_len: 1,
        sigma: sigma_buf,
        noise_curve: zero_curve,
        strength_map: unit_map_buf,
        map_len,
        dct_profile: dct_profile_buf,
        kaiser: kaiser_buf,
        accum,
        accum_len: frame_len,
        wsum,
        wsum_len: pixels,
        group_weight,
        refs,
        params,
    };

    let read_outputs = |launch: &CollabLaunch<R>| {
        let accum_bytes = launch
            .client
            .read_one(launch.accum.clone())
            .expect("accum readback");
        let wsum_bytes = launch
            .client
            .read_one(launch.wsum.clone())
            .expect("wsum readback");
        let weight_bytes = launch
            .client
            .read_one(launch.group_weight.clone())
            .expect("group_weight readback");

        CollabOutputs {
            accum: i32::from_bytes(&accum_bytes).to_vec(),
            wsum: i32::from_bytes(&wsum_bytes).to_vec(),
            group_weight: f32::from_bytes(&weight_bytes).to_vec(),
        }
    };

    (collab, read_outputs)
}

/// On a static clip, grouping across the temporal window cancels more grain than a spatial-only
/// search on the same frame.
///
/// Both arms run the same kernel with the same `spatial_radius`, `c_min`, `lambda_ht` and fixed
/// `sigma` over the same noisy centre frame. Frame `radius` emits `radius` passes after its own,
/// so `3 * radius + 1` frames are pushed.
#[test]
fn temporal_grouping_beats_spatial_only_on_a_static_clip() {
    let client = make_client();
    let (width, height) = (64u32, 64u32);
    let radius = 2u32;
    let base = textured_base(width, height);

    let noisy_frames: Vec<Vec<f32>> = (0..(3 * radius + 1))
        .map(|seed| noisy_field_over(&base, width, height, SIGMA, seed))
        .collect();
    let centre_index = radius as usize;

    let params = static_clip_params(radius);
    let mut denoiser = Nl4dDenoiser::<R>::new(&client, params, width, height).expect("construction failed");
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for frame in &noisy_frames {
        denoiser.push_frame(frame);
        if let Some(frame) = denoiser.denoise().expect("denoise failed") {
            outputs.push(frame);
        }
    }

    let temporal_out = outputs
        .get(centre_index)
        .expect("enough frames were pushed for frame `radius`'s own output to have emitted")
        .clone();

    let warp_uniform = needs_warp_uniform_search(&client);
    let spatial_out = run_spatial_only(
        &client,
        &noisy_frames[centre_index],
        width,
        height,
        SPATIAL_RADIUS,
        REFINE,
        C_MIN,
        LAMBDA_HT,
        SIGMA,
        warp_uniform,
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

/// Isolates cross-frame aggregation's own contribution from temporal grouping's.
///
/// Both arms group across the same radius-2 window with the same `lambda_ht` and kernel. The
/// cross-frame arm is the denoiser itself, which keeps every member that matched into the judged
/// frame from any pass. The centre-only arm runs the one pass centred on that frame through the
/// same `NlmDenoiser` front end and reads back only the centre slot's region of the ring. Each
/// member scatters into the region of the frame it was matched in, so that region holds exactly
/// the members whose own frame is the centre.
#[test]
fn cross_frame_aggregation_beats_centre_only_at_the_same_lambda() {
    let client = make_client();
    let (width, height) = (64u32, 64u32);
    let radius = 2u32;
    let base = textured_base(width, height);

    let frame_count = 3 * radius + 1;
    let frames: Vec<Vec<f32>> = (0..frame_count)
        .map(|seed| noisy_field_over(&base, width, height, SIGMA, seed))
        .collect();
    let judged_frame = radius as usize;

    let params = static_clip_params(radius);
    let mut denoiser = Nl4dDenoiser::<R>::new(&client, params, width, height).expect("construction failed");
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for frame in &frames {
        denoiser.push_frame(frame);
        if let Some(frame) = denoiser.denoise().expect("denoise failed") {
            outputs.push(frame);
        }
    }

    let cross_frame_out = outputs
        .get(judged_frame)
        .expect("enough frames were pushed for frame `radius`'s own output to have emitted")
        .clone();

    let clip_params = static_clip_params(radius);
    let mut nlm_params = clip_params.nlm;
    nlm_params.temporal_radius = radius;
    let mut front = NlmDenoiser::<R>::new(&client, nlm_params, width, height);

    let pixels = (width * height) as usize;
    let refs_x = refs_along(width);
    let refs = ref_count(width, height);
    let k_max = MAX_K;
    let total_frames = 1 + 2 * radius;

    let mut centre_only_out: Option<Vec<f32>> = None;
    let mut pass_index = 0u32;
    for frame in &frames {
        front.push_frame(frame);

        let Some(view) = front.submit_machinery(radius).expect("submit_machinery failed") else {
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
        let slots_bytes = u32::as_bytes(&view.neighbour_slots);
        let neighbour_slots_buf = client.create_from_slice(slots_bytes);

        let sigmas = front.current_sigmas_temporal_only();
        let sigma = [sigmas[0]];
        let profile = dct_noise_profile(0.0);
        let kaiser = kaiser_window(0.0);
        let zeroed_curve = [0.0f32; NOISE_CURVE_BINS];
        let sigma_bytes = f32::as_bytes(&sigma);
        let sigma_buf = client.create_from_slice(sigma_bytes);
        let profile_bytes = f32::as_bytes(&profile);
        let profile_buf = client.create_from_slice(profile_bytes);
        let kaiser_bytes = f32::as_bytes(&kaiser);
        let kaiser_buf = client.create_from_slice(kaiser_bytes);
        let curve_bytes = f32::as_bytes(&zeroed_curve);
        let zero_curve = client.create_from_slice(curve_bytes);

        let (map_cols, map_rows) = strength_map_dims(width, height);
        let map_len = (map_cols * map_rows) as usize;
        let unit_map = vec![1.0f32; map_len];
        let unit_map_bytes = f32::as_bytes(&unit_map);
        let unit_map_buf = client.create_from_slice(unit_map_bytes);
        let weight_norm = weight_scale(sigmas[0], &profile);
        let accum_scale = cross_frame_accum_scale(SPATIAL_RADIUS, radius);

        let group_weight = client.empty(refs * size_of::<f32>());
        // The whole ring, because a member matched in a neighbour frame scatters into that
        // frame's region. Reading back only the centre slot is what makes this the centre-only
        // arm.
        let zeroed_accum = vec![0i32; pixels * total_frames as usize];
        let zeroed_wsum = vec![0i32; pixels * total_frames as usize];
        let accum_bytes = i32::as_bytes(&zeroed_accum);
        let accum = client.create_from_slice(accum_bytes);
        let wsum_bytes = i32::as_bytes(&zeroed_wsum);
        let wsum = client.create_from_slice(wsum_bytes);
        let output = client.empty(pixels * size_of::<f32>());

        let motion = front.motion_ctx();

        let fused_x = fused_cubes_x(width);
        let refs_y = refs_along(height);
        let fused_grid = CubeCount::new_2d(fused_x, refs_y);
        let fused_dim = CubeDim::new_1d(64);
        let warp_uniform = needs_warp_uniform_search(&client);
        let grid_frame_count = grid_frames(radius);
        let agg_cubes_x = width.div_ceil(BLOCK_X);
        let agg_cubes_y = height.div_ceil(BLOCK_Y);
        let agg_grid = CubeCount::new_2d(agg_cubes_x, agg_cubes_y);
        let agg_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);
        let stored_ch = 1usize;

        unsafe {
            collab_fused::launch_unchecked::<f32, R>(
                &client,
                fused_grid,
                fused_dim,
                stored_ch,
                ArrayArg::from_raw_parts(view.input.clone(), ring_len),
                ArrayArg::from_raw_parts(view.input.clone(), stored_ch),
                ArrayArg::from_raw_parts(view.mv_field.clone(), mv_len.max(1)),
                ArrayArg::from_raw_parts(view.confidence.clone(), conf_len.max(1)),
                ArrayArg::from_raw_parts(neighbour_slots_buf, view.neighbour_slots.len().max(1)),
                ArrayArg::from_raw_parts(sigma_buf, 1),
                ArrayArg::from_raw_parts(zero_curve, NOISE_CURVE_BINS),
                ArrayArg::from_raw_parts(unit_map_buf, map_len),
                ArrayArg::from_raw_parts(profile_buf, 8),
                ArrayArg::from_raw_parts(kaiser_buf, PATCH_SIZE as usize),
                ArrayArg::from_raw_parts(accum.clone(), pixels * total_frames as usize),
                ArrayArg::from_raw_parts(wsum.clone(), pixels * total_frames as usize),
                ArrayArg::from_raw_parts(group_weight, refs),
                centre_slot,
                C_MIN,
                LAMBDA_HT,
                0u32,
                STRENGTH_MAP_OFF,
                weight_norm,
                accum_scale,
                warp_uniform,
                false,
                radius,
                grid_frame_count,
                REFINE,
                view.mv_stride,
                view.conf_stride,
                motion.step,
                motion.blksize,
                motion.blocks_x,
                motion.blocks_y,
                width,
                height,
                1u32,
                k_max,
                1u32,
                SPATIAL_RADIUS,
                refs_x,
                map_cols,
                map_rows,
                0.0f32,
                false,
                COLLAB_GROUPS,
                false,
                false,
                0,
            );

            collab_normalise::launch_unchecked::<R>(
                &client,
                agg_grid,
                agg_dim,
                1usize,
                ArrayArg::from_raw_parts(accum, pixels * total_frames as usize),
                ArrayArg::from_raw_parts(wsum, pixels * total_frames as usize),
                ArrayArg::from_raw_parts(output.clone(), pixels),
                centre_slot * pixels as u32,
                width,
                height,
                1u32,
                1u32,
            );
        }

        let output_bytes = client.read_one(output).expect("readback failed");
        let centre_only = f32::from_bytes(&output_bytes)[..pixels].to_vec();
        assert!(
            centre_only.iter().all(|value| value.is_finite()),
            "the centre-only arm left a pixel with no contribution at all"
        );

        centre_only_out = Some(centre_only);
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

/// A clip where every frame is the previous one shifted right by 2 pixels must report
/// `[2 * k, 0]` toward the neighbour at offset `k` once the first pass has run.
#[test]
fn motion_snapshot_reports_the_field_the_pass_used() {
    let client = make_client();
    let (width, height) = (96u32, 96u32);
    let radius = 2u32;
    let base = textured_base(width, height);
    let frames: Vec<Vec<f32>> = (0..5i32)
        .map(|k| {
            let mut shifted = vec![0.0f32; (width * height) as usize];
            for y in 0..height {
                for x in 0..width {
                    let source_x = (x as i32 - 2 * (k - 2)).clamp(0, width as i32 - 1) as u32;
                    shifted[(y * width + x) as usize] = base[(y * width + source_x) as usize];
                }
            }

            shifted
        })
        .collect();

    let params = static_clip_params(radius);
    let mut denoiser = Nl4dDenoiser::<R>::new(&client, params, width, height).expect("construction failed");
    assert!(denoiser.motion_snapshot().is_none(), "no pass has run yet");

    for frame in &frames {
        denoiser.push_frame(frame);
        let _ = denoiser.denoise().expect("denoise failed");
    }

    let snapshot = denoiser.motion_snapshot().expect("a pass has run");
    assert_eq!(snapshot.offsets, vec![-2, -1, 1, 2]);
    assert_eq!(snapshot.step, 8);
    assert_eq!(snapshot.blksize, 16);

    // Block (3, 3) covers pixels 24..40, well inside the frame.
    let block = (3 * snapshot.blocks_x + 3) as usize;
    for (neighbour, &k) in snapshot.offsets.iter().enumerate() {
        assert_eq!(
            snapshot.vectors[neighbour][block],
            [2 * k, 0],
            "neighbour k={k} should be tracked as a 2*k pixel shift"
        );
        assert!(
            snapshot.confidence[neighbour][block] > 0.5,
            "a clean shift must score confidently"
        );
    }
}

/// A panning clip with one flat block, which the estimator ties on and leaves at the seed.
///
/// With `field_lambda` on, the snapshot shows the flat block pulled to its neighbours' vector.
/// With it off, the snapshot shows the estimator's field.
#[test]
fn field_regularisation_reaches_the_snapshot() {
    let client = make_client();
    let (width, height) = (128u32, 96u32);
    let radius = 1u32;
    let mut base = textured_base(width, height);

    // Flatten 52..76 x 36..60 around block (7, 5). It is large enough to tie both the fine search
    // and the coarse pyramid level for that block, but small enough that its overlapping
    // neighbours still see texture, so only the centre block ties.
    for y in 36..60u32 {
        for x in 52..76u32 {
            base[(y * width + x) as usize] = 0.5;
        }
    }

    let frames: Vec<Vec<f32>> = (0..3i32)
        .map(|k| {
            let mut shifted = vec![0.0f32; (width * height) as usize];
            for y in 0..height {
                for x in 0..width {
                    let source_x = (x as i32 - 3 * (k - 1)).clamp(0, width as i32 - 1) as u32;
                    shifted[(y * width + x) as usize] = base[(y * width + source_x) as usize];
                }
            }

            shifted
        })
        .collect();

    let run = |lambda: f32| {
        let clip_params = static_clip_params(radius);
        let params = Nl4dParams {
            field_lambda: lambda,
            ..clip_params
        };
        let mut denoiser =
            Nl4dDenoiser::<R>::new(&client, params, width, height).expect("construction failed");
        for frame in &frames {
            denoiser.push_frame(frame);
            let _ = denoiser.denoise().expect("denoise failed");
        }

        denoiser.motion_snapshot().expect("a pass ran")
    };

    let off = run(0.0);
    let on = run(1.0);

    // The block at (7, 5) spans pixels 56..72 x 40..56, inside the flat region on every frame.
    let flat_block = (5 * off.blocks_x + 7) as usize;
    let forward_neighbour = 1usize;
    assert_ne!(
        off.vectors[forward_neighbour][flat_block],
        [3, 0],
        "the flat block must not be tracked without help, or this test proves nothing"
    );
    assert_eq!(on.vectors[forward_neighbour][flat_block], [3, 0]);

    // A textured block is unchanged by the pass.
    let textured_block = (2 * off.blocks_x + 2) as usize;
    assert_eq!(off.vectors[forward_neighbour][textured_block], [3, 0]);
    assert_eq!(on.vectors[forward_neighbour][textured_block], [3, 0]);
}

/// The shipped default parameters, with `channels` switched to `Luma` because these fixtures only
/// synthesise a single plane.
///
/// The other tests here pin `field_lambda` to 0.0 so their recorded values stay stable, while the
/// shipped default of 1.0 runs the field-regularisation pass.
fn shipped_default_params() -> Nl4dParams {
    let defaults = Nl4dParams::default();
    let nlm = NlmParams {
        channels: ChannelMode::Luma,
        ..defaults.nlm
    };
    let params = Nl4dParams { nlm, ..defaults };
    assert_eq!(
        params.field_lambda, 1.0,
        "this helper exists to exercise the shipped default, not an override"
    );

    params
}

/// Runs the shipped defaults over a clean static clip, then a noisy one.
///
/// A static clip's true motion is zero everywhere, so the regularised field must read exactly zero
/// at an interior block. A dispatch that read the wrong pyramid slot, neighbour or stride would
/// pull in a mismatched vector. The clean clip has no noise, because the estimator's noise-driven
/// wobble would mask that small displacement. The noisy clip must then gain at least 6 dB, since a
/// corrupt field would feed grouping the wrong candidates.
#[test]
fn shipped_defaults_denoise_a_static_clip_and_regularise_its_field_to_zero() {
    let client = make_client();
    let (width, height) = (96u32, 96u32);
    let base = textured_base(width, height);
    let shipped_params = shipped_default_params();
    let radius = shipped_params.temporal_radius;
    let frame_count = (3 * radius + 1) as usize;

    let clean_params = shipped_default_params();
    let mut clean_denoiser = Nl4dDenoiser::<R>::new(&client, clean_params, width, height)
        .expect("construction failed for the clean phase");
    for _ in 0..frame_count {
        clean_denoiser.push_frame(&base);
        let _ = clean_denoiser.denoise().expect("denoise failed");
    }

    let snapshot = clean_denoiser.motion_snapshot().expect("a pass ran");

    // Block (2, 2) spans pixels 16..32 on both axes, away from any edge clamping.
    let interior_block = (2 * snapshot.blocks_x + 2) as usize;
    for (neighbour, &k) in snapshot.offsets.iter().enumerate() {
        assert_eq!(
            snapshot.vectors[neighbour][interior_block],
            [0, 0],
            "neighbour k={k}: a static, noiseless clip's regularised field must read exactly \
             zero at an interior block; a nonzero vector here is what a wrong pyramid slot, \
             neighbour index, or stride in the smoothing dispatch looks like"
        );
    }

    let frames: Vec<Vec<f32>> = (0..frame_count as u32)
        .map(|seed| noisy_field_over(&base, width, height, SIGMA, seed))
        .collect();
    let noisy_params = shipped_default_params();
    let mut noisy_denoiser = Nl4dDenoiser::<R>::new(&client, noisy_params, width, height)
        .expect("construction failed for the noisy phase");
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for frame in &frames {
        noisy_denoiser.push_frame(frame);
        if let Some(frame) = noisy_denoiser.denoise().expect("denoise failed") {
            outputs.push(frame);
        }
    }

    noisy_denoiser
        .flush(|frame| {
            let values = frame.to_vec();
            outputs.push(values);
        })
        .expect("flush failed");

    assert_eq!(
        outputs.len(),
        frame_count,
        "expected one emitted frame per pushed frame"
    );

    for (i, output) in outputs.iter().enumerate() {
        let noisy_psnr = psnr(&frames[i], &base);
        let out_psnr = psnr(output, &base);
        assert!(
            out_psnr > noisy_psnr + 6.0,
            "frame {i}: expected at least a 6 dB PSNR improvement over the noisy input at the \
             shipped defaults, got noisy={noisy_psnr:.4} dB denoised={out_psnr:.4} dB"
        );
    }
}

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::helpers::{R, make_client, noisy_copy_of, psnr, smooth_texture_at, textured_base};
use crate::accelerate::Accelerator;
use crate::denoiser::FrameOutput;
use crate::nl4d::subpel::{HALF_PEL_TAPS, PhasePlaneCtx, phase_planes_host, run_phase_planes};
use crate::nl4d::{Nl4dDenoiser, Nl4dParams, SubpelPrecision};
use crate::{
    Algorithm,
    ChannelIntent,
    DenoisingMode,
    Depth,
    Device,
    FrameLayout,
    Nl4dOptions,
    PlanarDenoiser,
    PlaneOptions,
    Planes,
    Subsampling,
};

/// Runs the phase-plane kernel over every slot of `ring` and reads the
/// whole phase ring back.
fn gpu_phase_ring(ring: &[f32], total_frames: u32, width: u32, height: u32, stored_ch: u32) -> Vec<f32> {
    let client = make_client();
    let ring_buf = client.create_from_slice(f32::as_bytes(ring));
    let phase_buf = client.empty(ring.len() * 4 * size_of::<f32>());
    let taps_buf = client.create_from_slice(f32::as_bytes(&HALF_PEL_TAPS));
    let ctx = PhasePlaneCtx {
        ring: &ring_buf,
        phase_ring: &phase_buf,
        taps: &taps_buf,
        total_frames,
        width,
        height,
        stored_ch,
    };

    for slot in 0..total_frames {
        run_phase_planes::<R>(&client, &ctx, slot);
    }

    let bytes = client.read_one(phase_buf).expect("phase ring readback failed");
    f32::from_bytes(&bytes).to_vec()
}

/// Interleaves `stored_ch` planes of `width * height` into the ring's
/// per-pixel channel layout.
fn interleave(planes: &[Vec<f32>], stored_ch: u32) -> Vec<f32> {
    let pixels = planes[0].len();
    let ch = stored_ch as usize;
    let mut out = vec![0.0f32; pixels * ch];
    for (c, plane) in planes.iter().enumerate() {
        for (i, &value) in plane.iter().enumerate() {
            out[i * ch + c] = value;
        }
    }
    out
}

fn assert_planes_match(ring: &[f32], total_frames: u32, width: u32, height: u32, stored_ch: u32) {
    let frame_len = (width * height * stored_ch) as usize;
    let gpu = gpu_phase_ring(ring, total_frames, width, height, stored_ch);

    for slot in 0..total_frames as usize {
        let frame = &ring[slot * frame_len..(slot + 1) * frame_len];
        let expected = phase_planes_host(frame, width, height, stored_ch);
        for (plane, want) in expected.iter().enumerate() {
            let start = (slot * 4 + plane) * frame_len;
            let got = &gpu[start..start + frame_len];
            for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (g - w).abs() < 1e-5,
                    "slot {slot} plane {plane} index {i}: gpu {g} host {w}"
                );
            }
        }
    }
}

#[test]
fn phase_planes_match_the_host_filter_for_luma() {
    let (width, height) = (40u32, 24u32);
    let base = textured_base(width, height);
    let ring: Vec<f32> = (0..3u32)
        .flat_map(|seed| noisy_copy_of(&base, width, height, 0.02, seed))
        .collect();

    assert_planes_match(&ring, 3, width, height, 1);
}

#[test]
fn phase_planes_match_the_host_filter_for_two_channel_chroma() {
    let (width, height) = (24u32, 20u32);
    let base = textured_base(width, height);
    let mut ring = Vec::new();
    for seed in 0..3u32 {
        let first = noisy_copy_of(&base, width, height, 0.02, seed);
        let second = noisy_copy_of(&base, width, height, 0.03, seed + 100);
        ring.extend(interleave(&[first, second], 2));
    }

    assert_planes_match(&ring, 3, width, height, 2);
}

/// A step at the frame's left edge and a spike in its last row exercise
/// the clamped reads on all four borders.
#[test]
fn phase_planes_clamp_at_every_border() {
    let (width, height) = (16u32, 12u32);
    let mut frame = vec![0.25f32; (width * height) as usize];
    for y in 0..height {
        frame[(y * width) as usize] = 0.9;
    }
    frame[((height - 1) * width + width - 1) as usize] = 1.0;

    assert_planes_match(&frame, 1, width, height, 1);
}

fn subpel_params(precision: SubpelPrecision) -> Nl4dParams {
    Nl4dParams {
        nlm: crate::nlmeans::NlmParams {
            channels: crate::nlmeans::ChannelMode::Luma,
            ..Nl4dParams::default().nlm
        },
        subpel: precision,
        ..Nl4dParams::default()
    }
}

#[test]
fn subpel_rejects_a_frame_without_room_for_its_margin() {
    let client = make_client();
    let error = Nl4dDenoiser::<R>::new(&client, subpel_params(SubpelPrecision::Half), 9, 9)
        .err()
        .expect("a 9x9 frame leaves no margin for sub-pixel reads");
    assert!(error.contains("subpel"), "{error}");
}

/// Checks every slot's four planes against that slot's current input, by
/// reading straight from the ring handles rather than the denoiser.
///
/// This is what lets a caller check the ring from inside a `flush` sink
/// closure, where a `&Nl4dDenoiser` borrow is unavailable because
/// `flush` itself is still holding `&mut self`. A [Handle] identifies a
/// GPU buffer, not a snapshot of it, so cloning it before `flush` runs
/// and reading it during the callback still sees that callback's own
/// up-to-date contents.
fn assert_phase_ring_in_sync_handles(
    phase: &Handle,
    input: &Handle,
    width: u32,
    height: u32,
    total_frames: u32,
) {
    let client = make_client();
    let phase_bytes = client.read_one(phase.clone()).expect("phase readback failed");
    let input_bytes = client.read_one(input.clone()).expect("input readback failed");
    let phase_values = f32::from_bytes(&phase_bytes);
    let input_values = f32::from_bytes(&input_bytes);

    let frame_len = (width * height) as usize;
    for slot in 0..total_frames as usize {
        let frame = &input_values[slot * frame_len..(slot + 1) * frame_len];
        let expected = phase_planes_host(frame, width, height, 1);
        for (plane, want) in expected.iter().enumerate() {
            let start = (slot * 4 + plane) * frame_len;
            let got = &phase_values[start..start + frame_len];
            let worst = got
                .iter()
                .zip(want.iter())
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f32, f32::max);
            assert!(worst < 1e-5, "slot {slot} plane {plane} drifted by {worst}");
        }
    }
}

/// Checks every slot's four planes against that slot's current input.
fn assert_phase_ring_in_sync(denoiser: &Nl4dDenoiser<R>, width: u32, height: u32, total_frames: u32) {
    let phase = denoiser
        .phase_ring_for_test()
        .expect("subpel allocates a phase ring");
    let input = denoiser.front_for_test().input_ring_for_test();
    assert_phase_ring_in_sync_handles(phase, input, width, height, total_frames);
}

#[test]
fn phase_ring_tracks_the_input_ring_through_flush_and_reset() {
    let client = make_client();
    let (width, height) = (48u32, 32u32);
    let params = subpel_params(SubpelPrecision::Quarter);
    let total_frames = 1 + 2 * params.temporal_radius;
    let mut denoiser = Nl4dDenoiser::<R>::new(&client, params, width, height).unwrap();
    let base = textured_base(width, height);

    let mut ran_a_pass = false;
    for seed in 0..7u32 {
        let frame = noisy_copy_of(&base, width, height, 0.02, seed);
        denoiser.push_frame(&frame);
        if denoiser.denoise_submit().unwrap().is_some() {
            ran_a_pass = true;
        }
        if ran_a_pass {
            assert_phase_ring_in_sync(&denoiser, width, height, total_frames);
        }
    }

    // `flush` holds `&mut denoiser` for its whole run and resets the
    // stream itself before returning, so there is no moment after it
    // returns where the ring still reflects its duplicate-frame passes.
    // The handles below are cloned first, so the closure can read each
    // pass's own up-to-date ring contents without borrowing `denoiser`
    // again while it is already borrowed.
    let phase_handle = denoiser
        .phase_ring_for_test()
        .expect("subpel allocates a phase ring")
        .clone();
    let input_handle = denoiser.front_for_test().input_ring_for_test().clone();
    let mut sink = |_output: &FrameOutput| {
        assert_phase_ring_in_sync_handles(&phase_handle, &input_handle, width, height, total_frames);
    };
    denoiser.flush(&mut sink).unwrap();

    for seed in 100..104u32 {
        let frame = noisy_copy_of(&base, width, height, 0.05, seed);
        denoiser.push_frame(&frame);
        let _ = denoiser.denoise_submit().unwrap();
    }
    assert_phase_ring_in_sync(&denoiser, width, height, total_frames);
}

const CLIP_FRAMES: usize = 12;
const CLIP_SIGMA: f32 = 0.03;
const TEN_BIT_MAX: f32 = 1023.0;

/// One plane of frame `frame_idx`, panning half a pixel per frame.
fn clean_plane(width: u32, height: u32, frame_idx: usize) -> Vec<f32> {
    let shift = -0.5 * frame_idx as f32;
    smooth_texture_at(width, height, shift)
}

fn to_ten_bit(plane: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(plane.len() * 2);
    for &value in plane {
        let code = (value.clamp(0.0, 1.0) * TEN_BIT_MAX).round() as u16;
        let pair = code.to_le_bytes();
        bytes.extend(pair);
    }
    bytes
}

fn from_ten_bit(bytes: &[u8]) -> Vec<f32> {
    let (pairs, _) = bytes.as_chunks::<2>();
    pairs
        .iter()
        .map(|&pair| {
            let code = u16::from_le_bytes(pair);
            code as f32 / TEN_BIT_MAX
        })
        .collect()
}

/// The clean and noisy 10-bit clips, frame by frame.
fn panning_clip(layout: &FrameLayout) -> (Vec<Vec<f32>>, Vec<Planes>) {
    let (chroma_w, chroma_h) = layout.chroma_dims();
    let mut clean_luma = Vec::with_capacity(CLIP_FRAMES);
    let mut noisy = Vec::with_capacity(CLIP_FRAMES);

    for frame_idx in 0..CLIP_FRAMES {
        let seed = frame_idx as u32 * 3;
        let luma = clean_plane(layout.width, layout.height, frame_idx);
        let chroma = clean_plane(chroma_w, chroma_h, frame_idx);
        let noisy_y = noisy_copy_of(&luma, layout.width, layout.height, CLIP_SIGMA, seed);
        let noisy_u = noisy_copy_of(&chroma, chroma_w, chroma_h, CLIP_SIGMA, seed + 1);
        let noisy_v = noisy_copy_of(&chroma, chroma_w, chroma_h, CLIP_SIGMA, seed + 2);

        noisy.push(Planes {
            y: to_ten_bit(&noisy_y),
            u: to_ten_bit(&noisy_u),
            v: to_ten_bit(&noisy_v),
        });
        clean_luma.push(luma);
    }

    (clean_luma, noisy)
}

fn subpel_plane_options(intent: ChannelIntent, precision: SubpelPrecision) -> PlaneOptions {
    let nl4d = Nl4dOptions {
        subpel: precision,
        ..Nl4dOptions::default()
    };
    PlaneOptions {
        accelerators: vec![Accelerator::Vulkan],
        device: Device::Default,
        intent,
        mode: DenoisingMode::Temporal { radius: 2 },
        algorithm: Algorithm::Nl4d(nl4d),
        luma_strength: None,
        chroma_strength: None,
        luma_lambda_ht: None,
        chroma_lambda_ht: None,
    }
}

fn stream_clip(opts: &PlaneOptions, layout: FrameLayout, frames: &[Planes]) -> Vec<Planes> {
    let mut denoiser = PlanarDenoiser::create(opts, layout).unwrap();
    let mut out = Vec::new();
    for frame in frames {
        denoiser.push(frame).unwrap();
        if let Some(planes) = denoiser.recv().unwrap() {
            out.push(planes);
        }
    }

    denoiser.flush(|planes| out.push(planes)).unwrap();
    out
}

fn luma_psnr(clean: &[Vec<f32>], frames: &[Planes]) -> f64 {
    let clean_all: Vec<f32> = clean.iter().flatten().copied().collect();
    let frames_all: Vec<f32> = frames.iter().flat_map(|planes| from_ten_bit(&planes.y)).collect();
    psnr(&clean_all, &frames_all)
}

/// Streams the panning clip through `intent` at `precision`, and checks
/// every code stays within 10 bits and luma PSNR beats the noisy input.
fn assert_subpel_denoises(intent: ChannelIntent, subsampling: Subsampling, precision: SubpelPrecision) {
    let layout = FrameLayout {
        width: 64,
        height: 64,
        subsampling,
        depth: Depth::Ten,
    };
    let (clean, noisy) = panning_clip(&layout);
    let noisy_psnr = luma_psnr(&clean, &noisy);

    let opts = subpel_plane_options(intent, precision);
    let output = stream_clip(&opts, layout, &noisy);
    assert_eq!(output.len(), CLIP_FRAMES, "{intent:?} {precision:?}");

    for planes in &output {
        for plane in [&planes.y, &planes.u, &planes.v] {
            let values = from_ten_bit(plane);
            let in_range = values.iter().all(|value| (0.0..=1.0).contains(value));
            assert!(in_range, "{intent:?} {precision:?} wrote a code above 10 bits");
        }
    }

    let denoised_psnr = luma_psnr(&clean, &output);
    assert!(
        denoised_psnr > noisy_psnr,
        "{intent:?} {precision:?}: denoised {denoised_psnr:.2} dB should beat noisy \
         {noisy_psnr:.2} dB"
    );
}

#[test]
fn half_pel_denoises_end_to_end_in_luma_chroma_mode() {
    assert_subpel_denoises(
        ChannelIntent::LumaChroma,
        Subsampling::Yuv420,
        SubpelPrecision::Half,
    );
}

#[test]
fn quarter_pel_denoises_end_to_end_in_luma_chroma_mode() {
    assert_subpel_denoises(
        ChannelIntent::LumaChroma,
        Subsampling::Yuv420,
        SubpelPrecision::Quarter,
    );
}

/// The fused mode only accepts a 4:4:4 source.
#[test]
fn half_pel_denoises_end_to_end_in_yuv_mode() {
    assert_subpel_denoises(
        ChannelIntent::YuvFused,
        Subsampling::Yuv444,
        SubpelPrecision::Half,
    );
}

#[test]
fn quarter_pel_denoises_end_to_end_in_yuv_mode() {
    assert_subpel_denoises(
        ChannelIntent::YuvFused,
        Subsampling::Yuv444,
        SubpelPrecision::Quarter,
    );
}

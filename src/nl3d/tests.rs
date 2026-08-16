use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use super::{Nl3dDenoiser, Nl3dParams};
use crate::collab::CollabParams;
use crate::nlmeans::{ChannelMode, HqParams, MotionCompensationMode, NlmDenoiser, NlmParams, PrefilterMode};

type R = WgpuRuntime;

fn make_client() -> ComputeClient<R> {
    let device = <R as Runtime>::Device::default();
    R::client(&device)
}

/// A non-flat 32x32 (or larger) luma field, built from two out-of-phase
/// sine waves rather than noise, so it carries real spatial structure a
/// denoiser can either preserve or destroy.
fn textured_base(w: u32, h: u32) -> Vec<f32> {
    let mut frame = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            let v = 0.5
                + 0.15 * (fx * 6.0 * std::f32::consts::PI).sin() * (fy * 4.0 * std::f32::consts::PI).cos();
            frame[(y * w + x) as usize] = v.clamp(0.05, 0.95);
        }
    }
    frame
}

/// A static field carrying fine, regularly repeating texture near the
/// pixel grid's own Nyquist limit, similar in spatial frequency to brick
/// mortar coursing. The Immerkær mask (a small high-pass kernel) reads
/// this kind of detail the same way it reads noise, unlike the broad,
/// low-frequency waves [`textured_base`] carries.
fn fine_textured_base(w: u32, h: u32) -> Vec<f32> {
    let mut frame = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let hx = (2.0 * std::f32::consts::PI * x as f32 / 3.0).sin();
            let hy = (2.0 * std::f32::consts::PI * y as f32 / 5.0).cos();
            let v = 0.5 + 0.10 * hx * hy;
            frame[(y * w + x) as usize] = v.clamp(0.05, 0.95);
        }
    }
    frame
}

/// Adds independent pseudo-Gaussian noise to `base`, decorrelated across
/// `seed` so different seeds over the same base give independently noisy
/// copies of the same clean content. Each sample sums four hash-derived
/// uniforms in `[-0.5, 0.5]` (Irwin-Hall) and rescales to the requested
/// standard deviation, matching the convention used throughout the
/// nlmeans and collab test trees.
fn noisy_copy_of(base: &[f32], w: u32, h: u32, sigma: f32, seed: u32) -> Vec<f32> {
    let unit_std = (1.0f32 / 3.0f32).sqrt();
    let mut frame = vec![0.0f32; base.len()];
    for idx in 0..(w * h) {
        let mut sum = 0.0f32;
        for k in 0..4u32 {
            let mut hash = (idx * 4 + k)
                .wrapping_mul(2654435761)
                .wrapping_add(seed.wrapping_mul(0x9E37_79B9).wrapping_add(k));
            hash ^= hash >> 15;
            hash = hash.wrapping_mul(0x85EB_CA6B);
            hash ^= hash >> 13;
            sum += (hash as f32 / u32::MAX as f32) - 0.5;
        }
        frame[idx as usize] = (base[idx as usize] + (sum / unit_std) * sigma).clamp(0.0, 1.0);
    }
    frame
}

/// Adds grain correlated between neighbouring pixels to a flat `base`
/// field, at a lag-1 horizontal correlation of two thirds.
///
/// An independent noise field, built the same way [`noisy_copy_of`]
/// builds its noise, is blurred horizontally with `[0.25, 0.5, 0.25]`
/// (edges clamped), then added to `base`. This is a copy of
/// `nlmeans::tests::helpers::correlated_noisy_frame`, which is
/// `pub(super)` to that module and so not reachable from here, adapted
/// to this file's own noise generator so the two stay bit-identical in
/// method even though they draw from different hashes.
fn correlated_noisy_copy_of(base: &[f32], w: u32, h: u32, sigma_pre: f32, seed: u32) -> Vec<f32> {
    let unit_std = (1.0f32 / 3.0f32).sqrt();
    let mut raw = vec![0.0f32; base.len()];
    for idx in 0..(w * h) {
        let mut sum = 0.0f32;
        for k in 0..4u32 {
            let mut hash = (idx * 4 + k)
                .wrapping_mul(2654435761)
                .wrapping_add(seed.wrapping_mul(0x9E37_79B9).wrapping_add(k));
            hash ^= hash >> 15;
            hash = hash.wrapping_mul(0x85EB_CA6B);
            hash ^= hash >> 13;
            sum += (hash as f32 / u32::MAX as f32) - 0.5;
        }
        raw[idx as usize] = (sum / unit_std) * sigma_pre;
    }

    let mut out = vec![0.0f32; raw.len()];
    for y in 0..h {
        for x in 0..w {
            let xl = x.saturating_sub(1);
            let xr = (x + 1).min(w - 1);
            let l = raw[(y * w + xl) as usize];
            let c = raw[(y * w + x) as usize];
            let r = raw[(y * w + xr) as usize];
            let blurred = 0.25 * l + 0.5 * c + 0.25 * r;
            out[(y * w + x) as usize] = (base[(y * w + x) as usize] + blurred).clamp(0.0, 1.0);
        }
    }
    out
}

fn psnr(a: &[f32], b: &[f32]) -> f64 {
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    if mse <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (1.0f64 / mse).log10()
}

/// Mean SSIM over non-overlapping 8x8 tiles of a single-channel plane,
/// against a reference frame `b` of the same size.
///
/// This is a tiled approximation of the standard SSIM formula, not the
/// full Gaussian-windowed version. Each 8x8 block gets one SSIM value
/// from its own mean, variance, and covariance against the matching
/// block in `b`, and the result is the average over every block. Tiling
/// is much simpler to get right than a sliding Gaussian window, and is
/// enough to catch the failure mode this helper exists for, a filter
/// that trades away real structure for a higher PSNR.
///
/// Frames here are normalised to `[0, 1]`, so the stabilising constants
/// use that dynamic range, `C1 = 0.01^2` and `C2 = 0.03^2`, the usual
/// choice for unit-range SSIM.
///
/// `w` and `h` do not need to be multiples of 8. A trailing partial row
/// or column of blocks is skipped, since a partial block's statistics
/// are less meaningful and every frame this helper is used on is large
/// enough that a few skipped edge pixels do not change the result.
fn ssim(a: &[f32], b: &[f32], w: u32, h: u32) -> f64 {
    const BLOCK: u32 = 8;
    const C1: f64 = 0.01 * 0.01;
    const C2: f64 = 0.03 * 0.03;

    let blocks_x = w / BLOCK;
    let blocks_y = h / BLOCK;
    assert!(blocks_x > 0 && blocks_y > 0, "frame too small for 8x8 SSIM tiles");

    let n = (BLOCK * BLOCK) as f64;
    let mut total = 0.0f64;
    let mut block_count = 0usize;

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut sum_a = 0.0f64;
            let mut sum_b = 0.0f64;
            for dy in 0..BLOCK {
                for dx in 0..BLOCK {
                    let idx = ((by * BLOCK + dy) * w + (bx * BLOCK + dx)) as usize;
                    sum_a += a[idx] as f64;
                    sum_b += b[idx] as f64;
                }
            }
            let mean_a = sum_a / n;
            let mean_b = sum_b / n;

            let mut var_a = 0.0f64;
            let mut var_b = 0.0f64;
            let mut cov = 0.0f64;
            for dy in 0..BLOCK {
                for dx in 0..BLOCK {
                    let idx = ((by * BLOCK + dy) * w + (bx * BLOCK + dx)) as usize;
                    let da = a[idx] as f64 - mean_a;
                    let db = b[idx] as f64 - mean_b;
                    var_a += da * da;
                    var_b += db * db;
                    cov += da * db;
                }
            }
            var_a /= n - 1.0;
            var_b /= n - 1.0;
            cov /= n - 1.0;

            let numerator = (2.0 * mean_a * mean_b + C1) * (2.0 * cov + C2);
            let denominator = (mean_a * mean_a + mean_b * mean_b + C1) * (var_a + var_b + C2);
            total += numerator / denominator;
            block_count += 1;
        }
    }

    total / block_count as f64
}

/// Pins the SSIM helper itself before anything else relies on it. A
/// frame against its own identical copy must score 1.0, and against an
/// obviously different frame must score well below 1.0. A helper that
/// silently returned a constant would pass every other assertion built
/// on top of it without actually measuring anything.
#[test]
fn ssim_of_identical_frames_is_one_and_of_different_frames_is_lower() {
    let (w, h) = (32u32, 32u32);
    let base = textured_base(w, h);

    let same = ssim(&base, &base, w, h);
    assert!(
        (same - 1.0).abs() < 1e-6,
        "ssim against itself should be 1.0, got {same}"
    );

    let flat_black = vec![0.0f32; (w * h) as usize];
    let different = ssim(&base, &flat_black, w, h);
    assert!(
        different < 0.5,
        "ssim against an obviously different frame should be well below 1.0, got {different}"
    );
}

/// Interleaves three separate dense planes into one `w * h * 3` frame,
/// the layout `ChannelMode::Yuv` expects from `push_frame`.
fn interleave3(y: &[f32], u: &[f32], v: &[f32]) -> Vec<f32> {
    assert_eq!(y.len(), u.len());
    assert_eq!(y.len(), v.len());
    let mut out = vec![0.0f32; y.len() * 3];
    for i in 0..y.len() {
        out[i * 3] = y[i];
        out[i * 3 + 1] = u[i];
        out[i * 3 + 2] = v[i];
    }
    out
}

/// Pulls channel `c` (0 = Y, 1 = U, 2 = V) back out of an interleaved
/// `w * h * 3` frame, the inverse of [`interleave3`].
fn deinterleave3(frame: &[f32], c: usize) -> Vec<f32> {
    frame.iter().skip(c).step_by(3).copied().collect()
}

fn yuv_params_at(temporal_radius: u32) -> Nl3dParams {
    Nl3dParams {
        nlm: NlmParams {
            temporal_radius,
            search_radius: 2,
            patch_radius: 2,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Yuv,
            prefilter: PrefilterMode::None,
            motion_compensation: MotionCompensationMode::None,
            hq: Some(HqParams::default()),
            track_weight_sq: false,
        },
        front_strength_scale: 0.8,
        collab: CollabParams {
            channels: ChannelMode::Yuv,
            ..CollabParams::default()
        },
        residual_sigma_scale: 1.0,
    }
}

/// `ChannelMode::Yuv` is the library's default channel mode and the only
/// one the end-to-end nl3d bench row exercises, yet none of the other
/// tests in this file touch it. It is the four-lane-storage,
/// three-active-channel path, which touches padding lanes, the
/// channel-zero weight reduction inside a multi-channel loop, and sigma
/// padding that `Luma`'s single active channel never exercises.
///
/// A flat frame with three distinct, unequal channel values must come
/// back flat on every channel. Distinct values matter here. A bug that
/// swapped or blended channels would still pass an all-equal-channel
/// version of this test.
#[test]
fn yuv_uniform_passthrough_stays_flat() {
    let client = make_client();
    let (w, h) = (32u32, 32u32);
    let params = yuv_params_at(0);
    let mut denoiser = Nl3dDenoiser::<R>::new(&client, params, w, h).expect("construction failed");

    let pixels = (w * h) as usize;
    let channel_values = [0.5f32, 0.3, 0.7];
    let frame = interleave3(
        &vec![channel_values[0]; pixels],
        &vec![channel_values[1]; pixels],
        &vec![channel_values[2]; pixels],
    );

    denoiser.push_frame(&frame);
    let out = denoiser
        .denoise()
        .expect("denoise failed")
        .expect("a spatial-only cascade must emit on its first push");

    for (c, &want) in channel_values.iter().enumerate() {
        let channel = deinterleave3(&out, c);
        for (px, &v) in channel.iter().enumerate() {
            assert!(
                (v - want).abs() < 1e-3,
                "channel {c} pixel {px}: expected {want}, got {v}"
            );
        }
    }
}

/// A noisy YUV frame must come out better than it went in, on every
/// channel, not only luma.
///
/// `u_base`/`v_base` are built from `y_base` through different affine
/// scales and, for `v_base`, a reversed pixel order, so the three
/// channels carry related but not identical content. A bug that
/// swapped, blended, or dropped a channel's own sigma or member weights
/// would show up as one channel failing to improve even though luma
/// still did.
#[test]
fn yuv_noisy_frame_improves_on_every_channel() {
    let client = make_client();
    let (w, h) = (64u32, 64u32);
    let sigma = 0.03f32;

    let y_base = textured_base(w, h);
    let u_base: Vec<f32> = y_base
        .iter()
        .map(|&v| (v * 0.6 + 0.2).clamp(0.05, 0.95))
        .collect();
    let v_base: Vec<f32> = y_base
        .iter()
        .rev()
        .map(|&v| (v * 0.5 + 0.25).clamp(0.05, 0.95))
        .collect();
    let clean = interleave3(&y_base, &u_base, &v_base);

    let y_noisy = noisy_copy_of(&y_base, w, h, sigma, 0);
    let u_noisy = noisy_copy_of(&u_base, w, h, sigma, 1);
    let v_noisy = noisy_copy_of(&v_base, w, h, sigma, 2);
    let noisy = interleave3(&y_noisy, &u_noisy, &v_noisy);

    let params = yuv_params_at(0);
    let mut nl3d = Nl3dDenoiser::<R>::new(&client, params, w, h).expect("construction failed");
    nl3d.push_frame(&noisy);
    let out = nl3d
        .denoise()
        .expect("denoise failed")
        .expect("a spatial-only cascade must emit on its first push");

    for (c, name) in [(0, "y"), (1, "u"), (2, "v")] {
        let clean_ch = deinterleave3(&clean, c);
        let noisy_ch = deinterleave3(&noisy, c);
        let out_ch = deinterleave3(&out, c);

        let noisy_psnr = psnr(&noisy_ch, &clean_ch);
        let out_psnr = psnr(&out_ch, &clean_ch);

        assert!(
            out_psnr >= noisy_psnr + 3.0,
            "channel {name}: expected at least a 3 dB PSNR improvement over the noisy input, \
             got noisy={noisy_psnr:.4} dB denoised={out_psnr:.4} dB"
        );
    }
}

fn params_at(temporal_radius: u32) -> Nl3dParams {
    Nl3dParams {
        nlm: NlmParams {
            temporal_radius,
            search_radius: 2,
            patch_radius: 2,
            strength: 1.2,
            self_weight: 1.0,
            channels: ChannelMode::Luma,
            prefilter: PrefilterMode::None,
            motion_compensation: MotionCompensationMode::None,
            hq: Some(HqParams::default()),
            track_weight_sq: false,
        },
        front_strength_scale: 0.8,
        collab: CollabParams {
            channels: ChannelMode::Luma,
            ..CollabParams::default()
        },
        residual_sigma_scale: 1.0,
    }
}

/// A flat frame must come back flat. This is the simplest thing a
/// two-stage cascade can get wrong, a bug in either stage that biases
/// the output away from a uniform input, or that leaves visible
/// block-grid artefacts from the collaborative filter's patch grid.
#[test]
fn uniform_passthrough_stays_flat() {
    let client = make_client();
    let (w, h) = (32u32, 32u32);
    let params = params_at(0);
    let mut denoiser = Nl3dDenoiser::<R>::new(&client, params, w, h).expect("construction failed");

    let frame = vec![0.5f32; (w * h) as usize];
    denoiser.push_frame(&frame);
    let out = denoiser
        .denoise()
        .expect("denoise failed")
        .expect("a spatial-only cascade must emit on its first push");

    for (i, &v) in out.iter().enumerate() {
        assert!((v - 0.5).abs() < 1e-3, "pixel {i}: expected 0.5, got {v}");
    }
}

/// nl3d must beat an identically configured plain `NlmDenoiser` on the
/// same noisy input. "Identically configured" means the front end's
/// strength already carries `front_strength_scale`, the same scaling
/// nl3d's own constructor applies internally, so the comparison isolates
/// the collaborative stage's contribution rather than a strength
/// difference between the two denoisers.
///
/// This is a smoke gate proving the collaborative stage contributes
/// something, not a calibration claim, so the required margin is small.
/// SSIM is measured alongside PSNR, since PSNR alone rewards
/// over-smoothing and would not catch a collaborative stage that traded
/// real structure for a smoother, higher-PSNR result.
///
/// `Nl3dDenoiser` leaves `CollabParams.rho` at its default of `0.0`,
/// plain white-noise shrinkage, rather than deriving it from
/// `rho::rho_window`. An earlier version forced that table's value on
/// unconditionally, which lowered this gate's margin from about 0.84 dB
/// to 0.28 dB on this test's content, a smooth synthetic sine-wave
/// texture. Recalibration found that shaping cost more than it gave
/// back, on both the high-frequency energy ratio and on XPSNR/SSIM, so
/// it was dropped from the default and the wider margin is back. The
/// threshold below stayed at 0.2 dB anyway, since a smoke gate should
/// stay loose enough to survive small, legitimate future changes to
/// either stage.
#[test]
fn psnr_improves_over_front_end_alone() {
    let client = make_client();
    let (w, h) = (64u32, 64u32);
    let sigma = 0.03f32;
    let base = textured_base(w, h);

    let params = params_at(1);
    let mut nl3d = Nl3dDenoiser::<R>::new(&client, params.clone(), w, h).expect("nl3d construction failed");

    let mut front_params = params.nlm.clone();
    front_params.strength *= params.front_strength_scale;
    let mut front_only = NlmDenoiser::<R>::new(&client, front_params, w, h);

    let mut nl3d_out = None;
    let mut front_out = None;
    for seed in 0..3u32 {
        let frame = noisy_copy_of(&base, w, h, sigma, seed);

        nl3d.push_frame(&frame);
        if let Some(out) = nl3d.denoise().expect("nl3d denoise failed") {
            nl3d_out = Some(out);
        }

        front_only.push_frame(&frame);
        if let Some(out) = front_only.denoise().expect("front-end denoise failed") {
            front_out = Some(out.to_vec());
        }
    }

    let nl3d_out = nl3d_out.expect("nl3d should have emitted a frame once the window filled");
    let front_out =
        front_out.expect("the front end alone should have emitted a frame once the window filled");

    let nl3d_psnr = psnr(&nl3d_out, &base);
    let front_psnr = psnr(&front_out, &base);
    let nl3d_ssim = ssim(&nl3d_out, &base, w, h);
    let front_ssim = ssim(&front_out, &base, w, h);

    println!(
        "psnr_improves_over_front_end_alone: front_psnr={front_psnr:.4} dB nl3d_psnr={nl3d_psnr:.4} dB \
         front_ssim={front_ssim:.6} nl3d_ssim={nl3d_ssim:.6}"
    );

    assert!(
        nl3d_psnr >= front_psnr + 0.2,
        "expected nl3d to beat the front end alone by at least 0.2 dB, got front={front_psnr:.4} dB \
         nl3d={nl3d_psnr:.4} dB"
    );
    assert!(
        nl3d_ssim >= front_ssim,
        "expected nl3d to be at least as good as the front end alone on SSIM, got \
         front={front_ssim:.6} nl3d={nl3d_ssim:.6}. A PSNR win alongside an SSIM loss would mean \
         the collaborative stage is trading away real structure for smoothness"
    );
}

/// Every pushed frame must produce exactly one output across the whole
/// life of a stream, whether that output comes out during pushing, once
/// the temporal window is already full, or during the final flush that
/// drains the trailing edge.
#[test]
fn flush_emits_one_output_per_push() {
    let client = make_client();
    let (w, h) = (32u32, 32u32);
    let params = params_at(2);
    let mut nl3d = Nl3dDenoiser::<R>::new(&client, params, w, h).expect("construction failed");

    let base = textured_base(w, h);
    let pushes = 5usize;
    let mut during_pushes = 0usize;
    for seed in 0..pushes as u32 {
        let frame = noisy_copy_of(&base, w, h, 0.02, seed);
        nl3d.push_frame(&frame);
        if nl3d.denoise().expect("denoise failed").is_some() {
            during_pushes += 1;
        }
    }

    let mut flushed = 0usize;
    nl3d.flush(|_frame| flushed += 1).expect("flush failed");

    assert_eq!(
        during_pushes + flushed,
        pushes,
        "every pushed frame must produce exactly one output across push-time submits and the \
         final flush, got {during_pushes} during pushing and {flushed} from flush"
    );
}

/// A flat frame whose top half carries `sigma_a` noise and whose bottom
/// half carries `sigma_b`, both built with [`noisy_copy_of`] and spliced
/// row-wise. Mirrors `nlmeans::tests::split_sigma`'s own
/// `block_heterogeneous_frame` helper, adapted to this file's noise
/// generator.
fn block_heterogeneous_frame(w: u32, h: u32, base: f32, sigma_a: f32, sigma_b: f32) -> Vec<f32> {
    let flat = vec![base; (w * h) as usize];
    let top = noisy_copy_of(&flat, w, h, sigma_a, 0);
    let bottom = noisy_copy_of(&flat, w, h, sigma_b, 1);
    let row_len = w as usize;
    let half = (h / 2) as usize;
    let mut out = top;
    for row in half..h as usize {
        let start = row * row_len;
        out[start..start + row_len].copy_from_slice(&bottom[start..start + row_len]);
    }
    out
}

/// Pins that the collaborative stage's shrinkage sigma comes from the
/// front end's low noise chain, not its median chain.
///
/// A block-heterogeneous frame (top half low noise, bottom half high
/// noise) makes the two chains read different values on the same push,
/// the same technique `nlmeans::tests::split_sigma` uses to tell them
/// apart. `residual_sigma_scale` is pinned to `1.0` by `params_at`, so
/// the collaborative-stage sigma equals `base_sigma * ratio` exactly,
/// with no further scale to account for.
///
/// If `run_collab_stage` were changed back to read `current_sigmas`
/// (the median chain) instead of `current_sigmas_low`, this test would
/// see the recorded sigma match the median-chain reconstruction instead
/// of the low-chain one, and fail.
#[test]
fn collab_stage_uses_the_low_noise_chain() {
    let client = make_client();
    let (w, h) = (256u32, 256u32);
    let sigma_a = 2.0 / 255.0;
    let sigma_b = 20.0 / 255.0;
    let frame = block_heterogeneous_frame(w, h, 0.5, sigma_a, sigma_b);

    let params = params_at(0);
    let mut nl3d = Nl3dDenoiser::<R>::new(&client, params, w, h).expect("construction failed");

    nl3d.push_frame(&frame);
    nl3d.denoise()
        .expect("denoise failed")
        .expect("a spatial-only cascade must emit on its first push");

    let median = nl3d.front.current_sigmas()[0];
    let low = nl3d.front.current_sigmas_low()[0];
    assert!(
        low < median,
        "block-heterogeneous noise: low chain {low} should read below median chain {median}, \
         or this test cannot tell the two chains apart"
    );

    let recorded = nl3d.last_collab_sigmas[0];
    let ratio = nl3d.last_collab_ratio;
    assert_eq!(
        nl3d.residual_sigma_scale, 1.0,
        "this test assumes residual_sigma_scale is 1.0 so the reconstruction below needs no \
         further scale"
    );

    let from_low = low * ratio;
    let from_median = median * ratio;
    let rel_err_low = (recorded - from_low).abs() / from_low;
    assert!(
        rel_err_low < 1e-4,
        "the collaborative stage's sigma {recorded} should reconstruct from the low chain \
         ({low} * ratio {ratio} = {from_low}), rel err {rel_err_low:.6}"
    );
    assert!(
        (recorded - from_median).abs() / from_median > 0.05,
        "the collaborative stage's sigma {recorded} should NOT match a median-chain \
         reconstruction ({median} * ratio {ratio} = {from_median}); if it does, \
         run_collab_stage has regressed back to the median chain"
    );
}

/// Pins that the front end's low chain still separates a boosted
/// reading from an unboosted one on correlated grain.
///
/// `current_sigmas_low_unboosted` is what
/// `current_sigmas_temporal_only` falls back to whenever no trustworthy
/// temporal sample exists, so its own correctness still matters even
/// though the collaborative stage no longer reads it directly on every
/// push. Correlated grain, pushed for several frames so the temporal
/// estimator has real neighbours to read a correlation figure from,
/// makes `current_sigmas_low` (with the boost) and
/// `current_sigmas_low_unboosted` (without it) read apart on the same
/// push, the same way `collab_stage_uses_the_low_noise_chain` uses
/// block-heterogeneous noise to make the low and median chains read
/// apart.
#[test]
fn low_chain_still_separates_boosted_from_unboosted() {
    let client = make_client();
    let (w, h) = (128u32, 128u32);
    let base = vec![0.5f32; (w * h) as usize];

    let params = params_at(2);
    let mut nl3d = Nl3dDenoiser::<R>::new(&client, params, w, h).expect("construction failed");

    let pushes = 8u32;
    for seed in 0..pushes {
        let frame = correlated_noisy_copy_of(&base, w, h, 0.08, seed);
        nl3d.push_frame(&frame);
        nl3d.denoise().expect("denoise failed");
    }

    let boosted = nl3d.front.current_sigmas_low()[0];
    let unboosted = nl3d.front.current_sigmas_low_unboosted()[0];
    assert!(
        boosted > unboosted * 1.05,
        "correlated grain should make the boosted low chain {boosted} read meaningfully above \
         the unboosted low chain {unboosted}, or this test cannot tell the two readings apart"
    );
}

/// Pins that the collaborative stage's shrinkage sigma comes from the
/// front end's temporal reading alone, not the maximum it takes against
/// its spatial Immerkær reading.
///
/// A static textured base with independent noise added fresh on every
/// push makes the two readings disagree: the spatial reading sees the
/// texture on a single frame and cannot tell it apart from noise, so it
/// reads high, while the temporal reading differences one frame against
/// the next, where the identical texture cancels out and only the true
/// noise remains. `current_sigmas_low_unboosted` (the maximum of both)
/// therefore reads above `current_sigmas_temporal_only` (the temporal
/// reading alone) on this content, the same way
/// `low_chain_still_separates_boosted_from_unboosted` uses correlated
/// grain to make the boosted and unboosted low chains read apart.
/// `residual_sigma_scale` is pinned to `1.0` by `params_at`, so the
/// collaborative-stage sigma equals `base_sigma * ratio` exactly.
///
/// If `run_collab_stage` were changed back to read
/// `current_sigmas_low_unboosted` instead of
/// `current_sigmas_temporal_only`, this test would see the recorded
/// sigma match the combined reconstruction instead of the temporal-only
/// one, and fail.
#[test]
fn collab_stage_uses_the_temporal_only_chain() {
    let client = make_client();
    let (w, h) = (128u32, 128u32);
    let base = fine_textured_base(w, h);

    let params = params_at(2);
    let mut nl3d = Nl3dDenoiser::<R>::new(&client, params, w, h).expect("construction failed");

    let pushes = 8u32;
    for seed in 0..pushes {
        let frame = noisy_copy_of(&base, w, h, 0.02, seed);
        nl3d.push_frame(&frame);
        nl3d.denoise().expect("denoise failed");
    }

    let low_unboosted = nl3d.front.current_sigmas_low_unboosted()[0];
    let temporal_only = nl3d.front.current_sigmas_temporal_only()[0];
    assert!(
        temporal_only < low_unboosted * 0.95,
        "texture-inflated spatial reading should make the combined low-unboosted chain \
         {low_unboosted} read meaningfully above the temporal-only chain {temporal_only}, \
         or this test cannot tell the two readings apart"
    );

    let recorded = nl3d.last_collab_sigmas[0];
    let ratio = nl3d.last_collab_ratio;
    assert_eq!(
        nl3d.residual_sigma_scale, 1.0,
        "this test assumes residual_sigma_scale is 1.0 so the reconstruction below needs no \
         further scale"
    );

    let from_temporal = temporal_only * ratio;
    let from_low_unboosted = low_unboosted * ratio;
    let rel_err_temporal = (recorded - from_temporal).abs() / from_temporal;
    assert!(
        rel_err_temporal < 1e-4,
        "the collaborative stage's sigma {recorded} should reconstruct from the temporal-only \
         chain ({temporal_only} * ratio {ratio} = {from_temporal}), rel err {rel_err_temporal:.6}"
    );
    assert!(
        (recorded - from_low_unboosted).abs() / from_low_unboosted > 0.03,
        "the collaborative stage's sigma {recorded} should NOT match a low-unboosted \
         reconstruction ({low_unboosted} * ratio {ratio} = {from_low_unboosted}); if it does, \
         run_collab_stage has regressed back to the combined maximum"
    );
}

/// Pins the fallback that keeps the collaborative stage from
/// under-filtering when no temporal reading exists at all.
///
/// At a temporal radius of zero there is no neighbouring frame to
/// difference against, so `current_sigmas_temporal_only` never has a
/// trustworthy reading to report and must fall back to
/// `current_sigmas_low_unboosted`, the combined reading this stage used
/// before this change, rather than reading zero.
#[test]
fn collab_stage_falls_back_when_no_temporal_sample_exists() {
    let client = make_client();
    let (w, h) = (128u32, 128u32);
    let base = textured_base(w, h);
    let frame = noisy_copy_of(&base, w, h, 0.02, 0);

    let params = params_at(0);
    let mut nl3d = Nl3dDenoiser::<R>::new(&client, params, w, h).expect("construction failed");

    nl3d.push_frame(&frame);
    nl3d.denoise()
        .expect("denoise failed")
        .expect("a spatial-only cascade must emit on its first push");

    let low_unboosted = nl3d.front.current_sigmas_low_unboosted()[0];
    let temporal_only = nl3d.front.current_sigmas_temporal_only()[0];
    assert!(
        low_unboosted > 0.0,
        "the low-unboosted chain should read a real sigma from this noisy frame"
    );
    assert_eq!(
        temporal_only, low_unboosted,
        "with temporal_radius=0 no temporal sample ever exists, so the temporal-only chain \
         must fall back to the low-unboosted chain rather than reading zero"
    );

    let recorded = nl3d.last_collab_sigmas[0];
    let ratio = nl3d.last_collab_ratio;
    let expected = low_unboosted * ratio;
    let rel_err = (recorded - expected).abs() / expected;
    assert!(
        rel_err < 1e-4,
        "the collaborative stage's sigma {recorded} should reconstruct from the fallback value \
         ({low_unboosted} * ratio {ratio} = {expected}), rel err {rel_err:.6}"
    );
}

/// Two identical runs of the same input through separately constructed
/// denoisers must produce bitwise-identical output. Neither stage reads
/// anything but the GPU buffers it explicitly writes, so nothing should
/// make the two runs diverge.
#[test]
fn denoise_is_deterministic() {
    let client = make_client();
    let (w, h) = (32u32, 32u32);
    let base = textured_base(w, h);
    let frames: Vec<Vec<f32>> = (0..3u32)
        .map(|seed| noisy_copy_of(&base, w, h, 0.03, seed))
        .collect();

    let params = params_at(1);
    let mut run_a = Nl3dDenoiser::<R>::new(&client, params.clone(), w, h).expect("construction failed");
    let mut run_b = Nl3dDenoiser::<R>::new(&client, params, w, h).expect("construction failed");

    let mut out_a = None;
    let mut out_b = None;
    for frame in &frames {
        run_a.push_frame(frame);
        run_b.push_frame(frame);

        if let Some(o) = run_a.denoise().expect("run a denoise failed") {
            out_a = Some(o);
        }
        if let Some(o) = run_b.denoise().expect("run b denoise failed") {
            out_b = Some(o);
        }
    }

    let out_a = out_a.expect("run a should have emitted output");
    let out_b = out_b.expect("run b should have emitted output");
    assert_eq!(
        out_a, out_b,
        "two identical runs must produce bitwise-identical output"
    );
}

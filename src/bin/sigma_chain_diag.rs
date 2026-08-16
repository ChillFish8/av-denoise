//! Measures the sigma the front end's noise estimator reports, on white
//! noise and on spatially correlated noise added at a known true sigma
//! over a clean source, and traces the full nl3d sigma chain for the
//! correlated case against the true residual noise measured from a
//! clean reference.
//!
//! This is a diagnostic, not a product binary. It builds the same
//! front-end `NlmDenoiser` nl3d's front end uses (temporal radius,
//! search radius, strength, and `HqParams` matching nl3d's defaults at
//! `front_strength_scale`), pushes synthetic frames, and reads
//! `current_sigmas()`, `current_sigmas_low()`,
//! `current_sigmas_low_unboosted()`, `current_sigmas_temporal_only()`,
//! and `residual_ratio_sqrt()` directly.
//! `current_sigmas_temporal_only()` is the value
//! `Nl3dDenoiser::run_collab_stage` actually reads to build the sigma its
//! collaborative stage shrinks against; `current_sigmas_low_unboosted()`
//! (the low chain with the correlation boost left out but still maxed
//! against the Immerkær spatial reading), `current_sigmas_low()` (the low
//! chain with the boost folded in as well), and `current_sigmas()` (the
//! median chain) are also reported for comparison, since they show what
//! the same formula produced at each earlier step of this chain.
//!
//! Two modes:
//!
//! - `--mode ground-truth` adds noise at a known sigma to a single real
//!   frame, repeated across pushes with a fresh independent draw each
//!   time, in a white and a correlated variant, and reports the
//!   estimator's reading against the true value for both, plus the full
//!   chain (front estimate, residual ratio, `residual_sigma_scale`,
//!   final assumed sigma) against the true residual sigma measured
//!   directly from the front end's own output versus the clean
//!   reference.
//! - `--mode brick` corroborates on real film grain from
//!   `data/brick_source.mkv`, which has no clean reference. It measures
//!   the standard deviation of the frame-to-frame difference in a
//!   region chosen to have the least motion of any candidate checked (a
//!   patch of sky beside the tower, one continuous shot, frames 12-23 of
//!   the clip), and compares that against what the same front end
//!   reports on the same clip.
//!
//! # Running it
//!
//! Extract raw frames first, since this binary reads raw 8-bit grey
//! `ffmpeg -pix_fmt gray -f rawvideo` dumps, matching `grouping_diag`'s
//! own input convention:
//!
//! ```sh
//! ffmpeg -y -i data/clean-1080p.mkv -vf "select=eq(n\,60)" -vframes 1 \
//!   -pix_fmt gray -f rawvideo data/sigma_diag_clean60.gray
//! ffmpeg -y -i data/brick_source.mkv -pix_fmt gray -f rawvideo \
//!   data/sigma_diag_brick_all.gray
//! ```
//!
//! Then, built with `cargo build --release --bin sigma_chain_diag
//! --features vulkan`:
//!
//! ```sh
//! ./target/release/sigma_chain_diag --mode ground-truth \
//!   --clean data/sigma_diag_clean60.gray --sigma8 8.0
//! ./target/release/sigma_chain_diag --mode brick \
//!   --brick data/sigma_diag_brick_all.gray
//! ```

use av_denoise::nl3d::Nl3dParams;
use av_denoise::nlmeans::{
    ChannelMode,
    HqParams,
    MotionCompensationMode,
    NlmDenoiser,
    NlmParams,
    PrefilterMode,
    hq_default_strength,
};
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

type R = WgpuRuntime;

fn make_client() -> ComputeClient<R> {
    let device = <R as Runtime>::Device::default();
    R::client(&device)
}

// ---------------------------------------------------------------------
// Noise generation, duplicated from `src/nlmeans/tests/helpers.rs`'s
// `noisy_field_over` and `correlated_noisy_frame` rather than imported,
// since those are `pub(super)` to the test module. Kept bit-for-bit
// identical to that logic so the noise this binary generates is the
// same shape the rest of this codebase already relies on and has
// already empirically verified (see `task-D1-residual-rho-report.md`).
//
// Both functions return the pre-clamp added noise alongside the clamped
// noisy frame, so the true sigma and correlation can be measured from
// the noise actually added rather than reverse-engineered from a
// clamped frame, which would read low wherever clipping occurred.
// ---------------------------------------------------------------------

fn hashed_unit(idx: u32, seed: u32) -> f32 {
    let unit_std = (1.0f32 / 3.0f32).sqrt();
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
    sum / unit_std
}

/// White noise: independent per pixel, matching `noisy_field_over`.
/// Returns `(clamped_frame, added_noise)`.
fn white_noise_over(clean: &[f32], w: u32, h: u32, sigma: f32, seed: u32) -> (Vec<f32>, Vec<f32>) {
    let mut frame = vec![0.0f32; (w * h) as usize];
    let mut added = vec![0.0f32; (w * h) as usize];
    for idx in 0..(w * h) {
        let n = hashed_unit(idx, seed) * sigma;
        added[idx as usize] = n;
        frame[idx as usize] = (clean[idx as usize] + n).clamp(0.0, 1.0);
    }
    (frame, added)
}

/// Correlated noise: white noise blurred horizontally with `[0.25, 0.5,
/// 0.25]`, matching `correlated_noisy_frame`, which documents a lag-1
/// horizontal correlation of two thirds and a variance scale of 0.375
/// (the sum of the taps squared). `sigma_pre` is the pre-blur sigma;
/// pass `target_sigma / sqrt(0.375)` to land close to a chosen post-blur
/// sigma. Returns `(clamped_frame, added_noise)`, where `added_noise` is
/// the post-blur, pre-clamp field.
fn correlated_noise_over(clean: &[f32], w: u32, h: u32, sigma_pre: f32, seed: u32) -> (Vec<f32>, Vec<f32>) {
    let mut raw = vec![0.0f32; (w * h) as usize];
    for idx in 0..(w * h) {
        raw[idx as usize] = hashed_unit(idx, seed) * sigma_pre;
    }

    let mut frame = vec![0.0f32; raw.len()];
    let mut added = vec![0.0f32; raw.len()];
    for y in 0..h {
        for x in 0..w {
            let xl = x.saturating_sub(1);
            let xr = (x + 1).min(w - 1);
            let l = raw[(y * w + xl) as usize];
            let c = raw[(y * w + x) as usize];
            let r = raw[(y * w + xr) as usize];
            let blurred = 0.25 * l + 0.5 * c + 0.25 * r;
            let idx = (y * w + x) as usize;
            added[idx] = blurred;
            frame[idx] = (clean[idx] + blurred).clamp(0.0, 1.0);
        }
    }
    (frame, added)
}

/// Population standard deviation and lag-1 horizontal/vertical
/// correlation of a noise field, pooled over however many frames the
/// caller has accumulated into `pooled`. `pooled` holds one `w * h`
/// noise field per call to [`NoisePool::add`].
#[derive(Default)]
struct NoisePool {
    w: u32,
    h: u32,
    sum: f64,
    sumsq: f64,
    n: u64,
    // lag-1 accumulators, horizontal and vertical
    sum_prod_h: f64,
    n_h: u64,
    sum_prod_v: f64,
    n_v: u64,
}

impl NoisePool {
    fn new(w: u32, h: u32) -> Self {
        Self {
            w,
            h,
            ..Default::default()
        }
    }

    fn add(&mut self, field: &[f32]) {
        let w = self.w;
        let h = self.h;
        for &v in field {
            self.sum += v as f64;
            self.sumsq += (v as f64) * (v as f64);
            self.n += 1;
        }
        for y in 0..h {
            for x in 0..(w - 1) {
                let a = field[(y * w + x) as usize] as f64;
                let b = field[(y * w + x + 1) as usize] as f64;
                self.sum_prod_h += a * b;
                self.n_h += 1;
            }
        }
        for y in 0..(h - 1) {
            for x in 0..w {
                let a = field[(y * w + x) as usize] as f64;
                let b = field[((y + 1) * w + x) as usize] as f64;
                self.sum_prod_v += a * b;
                self.n_v += 1;
            }
        }
    }

    /// `(sigma, rho_h, rho_v)`, sigma in the same units the pushed
    /// fields were in.
    fn stats(&self) -> (f32, f32, f32) {
        let mean = self.sum / self.n as f64;
        let var = self.sumsq / self.n as f64 - mean * mean;
        let sigma = var.max(0.0).sqrt();
        let cov_h = self.sum_prod_h / self.n_h as f64 - mean * mean;
        let cov_v = self.sum_prod_v / self.n_v as f64 - mean * mean;
        let rho_h = if var > 0.0 { cov_h / var } else { 0.0 };
        let rho_v = if var > 0.0 { cov_v / var } else { 0.0 };
        (sigma as f32, rho_h as f32, rho_v as f32)
    }
}

/// The front end's parameters, matching what `Nl3dDenoiser::new` builds
/// for its own front end at the library's default `front_strength_scale`
/// and at whichever `temporal_radius` the caller passes (the `base`
/// preset's default is 2, used throughout this project's brick
/// comparisons). `track_weight_sq` is forced on, exactly as
/// `Nl3dDenoiser::new` forces it, since `residual_ratio_sqrt` depends on
/// it.
fn front_end_params(temporal_radius: u32, channels: ChannelMode) -> NlmParams {
    let front_strength_scale = Nl3dParams::default().front_strength_scale;
    let strength = hq_default_strength(channels, temporal_radius) * front_strength_scale;
    NlmParams {
        temporal_radius,
        search_radius: 2,
        patch_radius: 4,
        strength,
        self_weight: 1.0,
        channels,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        hq: Some(HqParams::default()),
        track_weight_sq: true,
    }
}

/// One fold's readings, taken right after a push once the temporal
/// window has filled and a frame has come back out.
struct FoldMetrics {
    /// `current_sigmas()[0]`, the median chain's smoothed estimate.
    sigma_estimate: f32,
    /// `current_sigmas_low()[0]`, the low chain's smoothed estimate,
    /// correlation boost included.
    sigma_estimate_low: f32,
    /// `current_sigmas_low_unboosted()[0]`, the low chain's smoothed
    /// estimate with the correlation boost left out.
    sigma_estimate_low_unboosted: f32,
    /// `current_sigmas_temporal_only()[0]`, the temporal reading alone,
    /// with no maximum against the Immerkær spatial reading and no
    /// correlation boost. This is what the collaborative stage's sigma
    /// is actually built from.
    sigma_estimate_temporal_only: f32,
    /// `residual_ratio_sqrt()` for the same push.
    ratio: f32,
    /// True residual sigma: std of `front_output - clean` over the
    /// interior region, in the same `[0, 1]` units.
    residual_true_sigma: f32,
}

const BORDER: u32 = 16;

fn interior_residual_sigma(output: &[f32], clean: &[f32], w: u32, h: u32) -> f32 {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut n = 0u64;
    for y in BORDER..(h - BORDER) {
        for x in BORDER..(w - BORDER) {
            let idx = (y * w + x) as usize;
            let d = (output[idx] - clean[idx]) as f64;
            sum += d;
            sumsq += d * d;
            n += 1;
        }
    }
    let mean = sum / n as f64;
    let var = (sumsq / n as f64 - mean * mean).max(0.0);
    var.sqrt() as f32
}

/// Runs `total_pushes` frames of one noise variant through the real
/// front end, returning the pooled true noise stats for what was
/// actually generated and every fold's metrics once the temporal window
/// filled.
#[allow(clippy::too_many_arguments)]
fn run_variant(
    client: &ComputeClient<R>,
    clean: &[f32],
    w: u32,
    h: u32,
    temporal_radius: u32,
    correlated: bool,
    target_sigma: f32,
    total_pushes: usize,
) -> (NoisePool, Vec<FoldMetrics>) {
    let params = front_end_params(temporal_radius, ChannelMode::Luma);
    let mut denoiser = NlmDenoiser::<R>::new(client, params, w, h);
    let mut pool = NoisePool::new(w, h);
    let mut folds = Vec::new();

    // Pre-blur sigma for the correlated variant, chosen so the post-blur
    // field lands near `target_sigma`. Verified against the pooled
    // measurement below rather than trusted.
    let sigma_pre = target_sigma / 0.375f32.sqrt();

    for i in 0..total_pushes {
        let seed = i as u32 + 1;
        let (frame, added) = if correlated {
            correlated_noise_over(clean, w, h, sigma_pre, seed)
        } else {
            white_noise_over(clean, w, h, target_sigma, seed)
        };
        pool.add(&added);

        denoiser.push_frame(&frame);
        let output = denoiser.denoise().expect("denoise failed").map(<[f32]>::to_vec);
        if let Some(output) = output {
            let sigma_estimate = denoiser.current_sigmas()[0];
            let sigma_estimate_low = denoiser.current_sigmas_low()[0];
            let sigma_estimate_low_unboosted = denoiser.current_sigmas_low_unboosted()[0];
            let sigma_estimate_temporal_only = denoiser.current_sigmas_temporal_only()[0];
            let ratio = denoiser
                .residual_ratio_sqrt()
                .expect("residual_ratio_sqrt failed");
            let residual_true_sigma = interior_residual_sigma(&output, clean, w, h);
            folds.push(FoldMetrics {
                sigma_estimate,
                sigma_estimate_low,
                sigma_estimate_low_unboosted,
                sigma_estimate_temporal_only,
                ratio,
                residual_true_sigma,
            });
        }
    }

    (pool, folds)
}

fn mean(values: impl Iterator<Item = f32> + Clone) -> f32 {
    let n = values.clone().count() as f32;
    values.sum::<f32>() / n
}

fn load_gray8(path: &str, w: u32, h: u32, frame_idx: usize) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    let frame_len = (w * h) as usize;
    let start = frame_idx * frame_len;
    let end = start + frame_len;
    assert!(
        end <= bytes.len(),
        "{path} holds {} frames at {w}x{h}, frame {frame_idx} is out of range",
        bytes.len() / frame_len,
    );
    bytes[start..end].iter().map(|&b| b as f32 / 255.0).collect()
}

fn load_gray8_all(path: &str, w: u32, h: u32) -> Vec<Vec<f32>> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    let frame_len = (w * h) as usize;
    let total = bytes.len() / frame_len;
    (0..total).map(|i| load_gray8(path, w, h, i)).collect()
}

fn run_ground_truth(
    clean_path: &str,
    w: u32,
    h: u32,
    sigma8: f32,
    total_pushes: usize,
    measure: usize,
    temporal_radius: u32,
) {
    let client = make_client();
    let clean = load_gray8(clean_path, w, h, 0);
    let target_sigma = sigma8 / 255.0;
    let residual_sigma_scale = Nl3dParams::default().residual_sigma_scale;

    println!("=== ground-truth mode ===");
    println!(
        "clean={clean_path} {w}x{h}, target true sigma8={sigma8:.4} ({target_sigma:.6} in [0,1]), \
         temporal_radius={temporal_radius}, total_pushes={total_pushes}, measure_last={measure}"
    );
    println!("residual_sigma_scale (Nl3dParams default) = {residual_sigma_scale:.4}");

    for (label, correlated) in [("WHITE", false), ("CORRELATED", true)] {
        let (pool, folds) = run_variant(
            &client,
            &clean,
            w,
            h,
            temporal_radius,
            correlated,
            target_sigma,
            total_pushes,
        );
        let (true_sigma, rho_h, rho_v) = pool.stats();
        let true_sigma8 = true_sigma * 255.0;

        assert!(
            folds.len() >= measure,
            "{label}: only {} folds produced, need at least {measure}",
            folds.len()
        );
        let tail = &folds[folds.len() - measure..];

        let est_mean = mean(tail.iter().map(|f| f.sigma_estimate));
        let est_low_mean = mean(tail.iter().map(|f| f.sigma_estimate_low));
        let est_low_unboosted_mean = mean(tail.iter().map(|f| f.sigma_estimate_low_unboosted));
        let est_temporal_only_mean = mean(tail.iter().map(|f| f.sigma_estimate_temporal_only));
        let ratio_mean = mean(tail.iter().map(|f| f.ratio));
        let true_resid_mean = mean(tail.iter().map(|f| f.residual_true_sigma));

        let est_mean8 = est_mean * 255.0;
        let est_low_mean8 = est_low_mean * 255.0;
        let est_low_unboosted_mean8 = est_low_unboosted_mean * 255.0;
        let est_temporal_only_mean8 = est_temporal_only_mean * 255.0;
        let true_resid_mean8 = true_resid_mean * 255.0;
        // The collaborative stage builds its sigma from the temporal
        // reading alone, so this is the chain actually reported to
        // `run_collab_stage`.
        let final_sigma = est_temporal_only_mean * ratio_mean * residual_sigma_scale;
        let final_sigma8 = final_sigma * 255.0;
        // What the same formula produces from the low chain with the
        // boost left out but still maxed against the Immerkær spatial
        // reading, kept only for comparison.
        let final_sigma_low_unboosted8 = est_low_unboosted_mean8 * ratio_mean * residual_sigma_scale;
        // What the same formula produces from the low chain with the
        // boost still folded in, kept only for comparison.
        let final_sigma_low_boosted8 = est_low_mean8 * ratio_mean * residual_sigma_scale;
        // What the same formula would have produced from the median
        // chain, kept only for comparison against `final_sigma8`.
        let final_sigma_median8 = est_mean8 * ratio_mean * residual_sigma_scale;

        println!();
        println!("--- {label} ---");
        println!(
            "  true noise generated: sigma8={true_sigma8:.4} (target {sigma8:.4}), rho_h={rho_h:.4}, rho_v={rho_v:.4}"
        );
        println!(
            "  front-end median-chain estimate (current_sigmas, trailing {measure}-fold mean): sigma8={est_mean8:.4}"
        );
        println!(
            "  front-end low-chain estimate, boosted (current_sigmas_low, trailing {measure}-fold mean): sigma8={est_low_mean8:.4}"
        );
        println!(
            "  front-end low-chain estimate, unboosted (current_sigmas_low_unboosted, trailing {measure}-fold mean): sigma8={est_low_unboosted_mean8:.4}"
        );
        println!(
            "  front-end temporal-only estimate (current_sigmas_temporal_only, trailing {measure}-fold mean): sigma8={est_temporal_only_mean8:.4}"
        );
        println!("  median estimate / true ratio: {:.4}", est_mean8 / true_sigma8);
        println!(
            "  low (boosted) estimate / true ratio: {:.4}",
            est_low_mean8 / true_sigma8
        );
        println!(
            "  low (unboosted) estimate / true ratio: {:.4}",
            est_low_unboosted_mean8 / true_sigma8
        );
        println!(
            "  temporal-only estimate / true ratio: {:.4}",
            est_temporal_only_mean8 / true_sigma8
        );
        println!("  residual_ratio_sqrt (trailing mean): {ratio_mean:.4}");
        println!(
            "  final assumed sigma, temporal-only (what the collaborative stage actually gets): sigma8={final_sigma8:.4}"
        );
        println!(
            "  final assumed sigma, low chain unboosted (for comparison only): sigma8={final_sigma_low_unboosted8:.4}"
        );
        println!(
            "  final assumed sigma, low chain boosted (for comparison only): sigma8={final_sigma_low_boosted8:.4}"
        );
        println!(
            "  final assumed sigma, median chain (for comparison only): sigma8={final_sigma_median8:.4}"
        );
        println!(
            "  TRUE residual sigma in front-end output vs clean (trailing mean): sigma8={true_resid_mean8:.4}"
        );
        println!(
            "  assumed/true residual ratio, temporal-only: {:.4}",
            final_sigma8 / true_resid_mean8
        );
        println!(
            "  assumed/true residual ratio, low chain unboosted: {:.4}",
            final_sigma_low_unboosted8 / true_resid_mean8
        );
        println!(
            "  assumed/true residual ratio, low chain boosted: {:.4}",
            final_sigma_low_boosted8 / true_resid_mean8
        );
        println!(
            "  assumed/true residual ratio, median chain: {:.4}",
            final_sigma_median8 / true_resid_mean8
        );

        print!("  per-fold sigma_estimate8 (median):");
        for f in tail {
            print!(" {:.3}", f.sigma_estimate * 255.0);
        }
        println!();
        print!("  per-fold sigma_estimate8 (low, boosted):");
        for f in tail {
            print!(" {:.3}", f.sigma_estimate_low * 255.0);
        }
        println!();
        print!("  per-fold sigma_estimate8 (low, unboosted):");
        for f in tail {
            print!(" {:.3}", f.sigma_estimate_low_unboosted * 255.0);
        }
        println!();
        print!("  per-fold sigma_estimate8 (temporal-only):");
        for f in tail {
            print!(" {:.3}", f.sigma_estimate_temporal_only * 255.0);
        }
        println!();
        print!("  per-fold residual_true_sigma8:");
        for f in tail {
            print!(" {:.3}", f.residual_true_sigma * 255.0);
        }
        println!();
    }
}

/// Union of low-motion, low-texture 40x40 blocks found by an offline
/// sweep of `data/brick_source.mkv` (see this task's report for the
/// method): mean abs frame-to-frame difference under ~4.5/255 for every
/// within-shot consecutive pair in frames 12-23, and a per-block std
/// under ~1.7/255 on frame 12, so the region carries little texture for
/// motion to hide in either. It is a patch of sky beside the tower in
/// that shot.
const BRICK_STATIC_REGION: (u32, u32, u32, u32) = (1600, 400, 1760, 480);
const BRICK_SHOT_START: usize = 12;
const BRICK_SHOT_END_INCLUSIVE: usize = 23;

fn temporal_diff_sigma(
    frames: &[Vec<f32>],
    w: u32,
    region: (u32, u32, u32, u32),
    start: usize,
    end_inclusive: usize,
) -> f32 {
    let (x0, y0, x1, y1) = region;
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut n = 0u64;
    for t in start..end_inclusive {
        let a = &frames[t];
        let b = &frames[t + 1];
        for y in y0..y1 {
            for x in x0..x1 {
                let idx = (y * w + x) as usize;
                let d = (b[idx] - a[idx]) as f64;
                sum += d;
                sumsq += d * d;
                n += 1;
            }
        }
    }
    let mean = sum / n as f64;
    let var = (sumsq / n as f64 - mean * mean).max(0.0);
    // diff of two independent same-sigma noise draws has variance 2*sigma^2
    (var / 2.0).sqrt() as f32
}

fn run_brick(brick_path: &str, w: u32, h: u32) {
    let client = make_client();
    let frames = load_gray8_all(brick_path, w, h);
    println!("=== brick corroboration mode ===");
    println!("brick={brick_path} {w}x{h}, {} frames loaded", frames.len());
    println!(
        "static region x0={} y0={} x1={} y1={}, shot frames {}..={}",
        BRICK_STATIC_REGION.0,
        BRICK_STATIC_REGION.1,
        BRICK_STATIC_REGION.2,
        BRICK_STATIC_REGION.3,
        BRICK_SHOT_START,
        BRICK_SHOT_END_INCLUSIVE
    );

    let diff_sigma = temporal_diff_sigma(
        &frames,
        w,
        BRICK_STATIC_REGION,
        BRICK_SHOT_START,
        BRICK_SHOT_END_INCLUSIVE,
    );
    let diff_sigma8 = diff_sigma * 255.0;
    println!(
        "temporal frame-difference sigma over the static region (weaker-footing corroborating \
         estimate, not ground truth): sigma8={diff_sigma8:.4}"
    );

    // Run the front end over the *whole* clip first, purely to show why
    // that reading cannot be trusted for this comparison: a hard scene
    // cut sits right before the static region's shot (mean luma jumps
    // 181 -> 166 at frame 12, confirmed by inspection), and the noise
    // estimator's EMA (alpha 0.2) is still climbing out of the previous
    // shot's level throughout most of a 12-frame shot, never settling
    // before the next cut arrives.
    let temporal_radius = 2u32;
    {
        let params = front_end_params(temporal_radius, ChannelMode::Luma);
        let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
        let mut estimates = Vec::new();
        for frame in &frames {
            denoiser.push_frame(frame);
            if denoiser.denoise().expect("denoise failed").is_some() {
                estimates.push(denoiser.current_sigmas()[0]);
            }
        }
        print!(
            "whole-clip per-fold estimates (sigma8, for reference only, EMA never settles within a 12-frame shot):"
        );
        for &e in &estimates {
            print!(" {:.3}", e * 255.0);
        }
        println!();
    }

    // The comparable reading: a fresh denoiser fed only the static
    // region's own shot (frames 12-23), so the EMA has no cross-shot
    // history to climb out of and gets the same number of settling
    // folds the shot actually offers, exactly like a real stream
    // starting fresh on this content.
    let shot_frames = &frames[BRICK_SHOT_START..=BRICK_SHOT_END_INCLUSIVE];
    let params = front_end_params(temporal_radius, ChannelMode::Luma);
    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    let mut estimates = Vec::new();
    let mut estimates_low = Vec::new();
    let mut estimates_low_unboosted = Vec::new();
    let mut estimates_temporal_only = Vec::new();
    for frame in shot_frames {
        denoiser.push_frame(frame);
        if denoiser.denoise().expect("denoise failed").is_some() {
            estimates.push(denoiser.current_sigmas()[0]);
            estimates_low.push(denoiser.current_sigmas_low()[0]);
            estimates_low_unboosted.push(denoiser.current_sigmas_low_unboosted()[0]);
            estimates_temporal_only.push(denoiser.current_sigmas_temporal_only()[0]);
        }
    }
    let last = estimates.len().min(4);
    let tail = &estimates[estimates.len() - last..];
    let tail_low = &estimates_low[estimates_low.len() - last..];
    let tail_low_unboosted = &estimates_low_unboosted[estimates_low_unboosted.len() - last..];
    let tail_temporal_only = &estimates_temporal_only[estimates_temporal_only.len() - last..];
    let est_mean8 = mean(tail.iter().copied()) * 255.0;
    let est_low_mean8 = mean(tail_low.iter().copied()) * 255.0;
    let est_low_unboosted_mean8 = mean(tail_low_unboosted.iter().copied()) * 255.0;
    let est_temporal_only_mean8 = mean(tail_temporal_only.iter().copied()) * 255.0;
    println!(
        "front-end median-chain estimate over just the static region's own shot (trailing {last}-fold mean of current_sigmas): sigma8={est_mean8:.4}"
    );
    println!(
        "front-end low-chain estimate, boosted, over just the static region's own shot (trailing {last}-fold mean of current_sigmas_low): sigma8={est_low_mean8:.4}"
    );
    println!(
        "front-end low-chain estimate, unboosted, over just the static region's own shot (trailing {last}-fold mean of current_sigmas_low_unboosted): sigma8={est_low_unboosted_mean8:.4}"
    );
    println!(
        "front-end temporal-only estimate over just the static region's own shot (trailing {last}-fold mean of current_sigmas_temporal_only, what the collaborative stage actually gets): sigma8={est_temporal_only_mean8:.4}"
    );
    println!(
        "median estimate / temporal-diff ratio: {:.4}",
        est_mean8 / diff_sigma8
    );
    println!(
        "low (boosted) estimate / temporal-diff ratio: {:.4}",
        est_low_mean8 / diff_sigma8
    );
    println!(
        "low (unboosted) estimate / temporal-diff ratio: {:.4}",
        est_low_unboosted_mean8 / diff_sigma8
    );
    println!(
        "temporal-only estimate / temporal-diff ratio: {:.4}",
        est_temporal_only_mean8 / diff_sigma8
    );
    print!("shot-only per-fold estimates, median (sigma8):");
    for &e in &estimates {
        print!(" {:.3}", e * 255.0);
    }
    println!();
    print!("shot-only per-fold estimates, low boosted (sigma8):");
    for &e in &estimates_low {
        print!(" {:.3}", e * 255.0);
    }
    println!();
    print!("shot-only per-fold estimates, low unboosted (sigma8):");
    for &e in &estimates_low_unboosted {
        print!(" {:.3}", e * 255.0);
    }
    println!();
    print!("shot-only per-fold estimates, temporal-only (sigma8):");
    for &e in &estimates_temporal_only {
        print!(" {:.3}", e * 255.0);
    }
    println!();
}

struct Args {
    mode: String,
    clean: String,
    brick: String,
    width: u32,
    height: u32,
    sigma8: f32,
    pushes: usize,
    measure: usize,
    temporal_radius: u32,
}

fn parse_args() -> Args {
    let mut mode = "ground-truth".to_string();
    let mut clean = "data/sigma_diag_clean60.gray".to_string();
    let mut brick = "data/sigma_diag_brick_all.gray".to_string();
    let mut width = 1920u32;
    let mut height = 1080u32;
    let mut sigma8 = 8.0f32;
    let mut pushes = 40usize;
    let mut measure = 15usize;
    let mut temporal_radius = 2u32;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut next = || args.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--mode" => mode = next(),
            "--clean" => clean = next(),
            "--brick" => brick = next(),
            "--width" => width = next().parse().expect("--width must be an integer"),
            "--height" => height = next().parse().expect("--height must be an integer"),
            "--sigma8" => sigma8 = next().parse().expect("--sigma8 must be a float"),
            "--pushes" => pushes = next().parse().expect("--pushes must be an integer"),
            "--measure" => measure = next().parse().expect("--measure must be an integer"),
            "--temporal-radius" => {
                temporal_radius = next().parse().expect("--temporal-radius must be an integer")
            },
            other => panic!("unknown flag {other}"),
        }
    }

    Args {
        mode,
        clean,
        brick,
        width,
        height,
        sigma8,
        pushes,
        measure,
        temporal_radius,
    }
}

fn main() {
    let args = parse_args();
    match args.mode.as_str() {
        "ground-truth" => run_ground_truth(
            &args.clean,
            args.width,
            args.height,
            args.sigma8,
            args.pushes,
            args.measure,
            args.temporal_radius,
        ),
        "brick" => run_brick(&args.brick, args.width, args.height),
        other => panic!("unknown --mode {other}, expected ground-truth or brick"),
    }
}

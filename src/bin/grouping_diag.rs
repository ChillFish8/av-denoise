//! Measures whether stage 1 and stage 2 of the collaborative filter's
//! spatial grouping actually finds similar patches, or whether the
//! noise-floor subtraction lets grouping admit patches that are not
//! really similar.
//!
//! This is a diagnostic, not a product binary. It reads raw 8-bit grey
//! frames extracted from a clean source and a synthetically noised copy
//! of the same source, runs the real `collab_group_spatial`,
//! `collab_filter_ht`, and `collab_aggregate` kernels in the same order
//! `CollabPipeline::run_two_stage` runs them, and then compares the
//! groups each stage forms against ground truth taken from the clean
//! frame.
//!
//! # Method
//!
//! For every reference patch sampled inside a region, this measures two
//! numbers in the clean domain (the true, noise-free patch distance,
//! never the noisy or pilot distance the kernel itself used to decide
//! admission):
//!
//! - the mean clean-domain distance of the patches the kernel actually
//!   admitted into the group
//! - the mean clean-domain distance of every candidate patch the search
//!   window considered, admitted or not
//!
//! Their ratio is the headline number. A ratio near 1 means admission
//! carries no information, the group is no better matched than a random
//! draw from the search window. A ratio well under 1 means admission is
//! finding genuinely closer patches.
//!
//! This also reports the raw admission rate (the fraction of candidates
//! that pass the floor-subtracted threshold, before the top-K cap and
//! the power-of-two rounding `collab_group_spatial` applies), and the
//! distribution of `member_count` after that cap.
//!
//! # How this differs from the real nl3d cascade
//!
//! nl3d's collaborative stage never sees raw noisy pixels. It sees the
//! nlmeans front end's own output, already partly denoised, and the
//! sigma it is told is the front end's noise estimate times a measured
//! residual ratio times `residual_sigma_scale`. This diagnostic skips
//! the front end entirely and feeds the collaborative kernels a directly
//! noised frame, told its own true injected sigma times
//! `residual_sigma_scale`. That isolates exactly the mechanism under
//! suspicion, whether an inflated sigma inflates the admission floor
//! enough to break discrimination, without needing a front-end run to
//! measure a residual ratio first. It is a fair test of that mechanism
//! on its own terms, not a reproduction of nl3d's exact numbers.
//!
//! # Running it
//!
//! `scripts/grouping_discrimination.py` extracts the clean and noisy raw
//! frames this binary reads and then drives it:
//!
//! ```sh
//! uv run scripts/grouping_discrimination.py
//! ```
//!
//! It can also be built and run directly, once the raw frames it needs
//! already exist (built with `cargo build --release --bin grouping_diag
//! --features vulkan`):
//!
//! ```sh
//! ./target/release/grouping_diag --clean data/grouping_diag/clean.raw \
//!   --noisy data/grouping_diag/noisy.raw --width 1920 --height 1080
//! ```
use std::collections::HashSet;
use std::fs;

use av_denoise::collab::geometry::{member_buf_len, ref_count, ref_pos, refs_along};
use av_denoise::collab::kernels::aggregate::{ACCUM_SCALE, collab_normalise, collab_zero_accum, weight_scale};
use av_denoise::collab::kernels::filter_ht::collab_filter_ht;
use av_denoise::collab::kernels::group::collab_group_spatial;
use av_denoise::collab::kernels::transforms::dct_noise_profile;
use av_denoise::collab::{CollabParams, PATCH_AREA, PATCH_SIZE};
use av_denoise::nlmeans::{BLOCK_X, BLOCK_Y};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;

type R = WgpuRuntime;

/// Luma's channel-scale factor, `channel_scale(ChannelMode::Luma)` in
/// `collab::pipeline`, duplicated here since that helper is private to
/// its module. This diagnostic only ever runs on luma.
const CHANNEL_SCALE_LUMA: f32 = 3.0;

/// `(x0, y0, x1, y1)` in source pixel coordinates.
type Region = (u32, u32, u32, u32);

struct Args {
    clean: String,
    noisy: String,
    width: u32,
    height: u32,
    frame: usize,
    flat_region: Region,
    texture_region: Region,
    sample_target: usize,
}

fn parse_region(s: &str) -> Region {
    let parts: Vec<u32> = s
        .split(',')
        .map(|p| p.parse().expect("region must be x0,y0,x1,y1"))
        .collect();
    assert_eq!(parts.len(), 4, "region must be x0,y0,x1,y1, got {s}");
    (parts[0], parts[1], parts[2], parts[3])
}

fn parse_args() -> Args {
    let mut clean = None;
    let mut noisy = None;
    let mut width = 1920u32;
    let mut height = 1080u32;
    let mut frame = 0usize;
    let mut flat_region = (860, 500, 1140, 690);
    let mut texture_region = (1560, 300, 1680, 500);
    let mut sample_target = 60usize;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut next = || args.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        match flag.as_str() {
            "--clean" => clean = Some(next()),
            "--noisy" => noisy = Some(next()),
            "--width" => width = next().parse().expect("--width must be an integer"),
            "--height" => height = next().parse().expect("--height must be an integer"),
            "--frame" => frame = next().parse().expect("--frame must be an integer"),
            "--flat-region" => flat_region = parse_region(&next()),
            "--texture-region" => texture_region = parse_region(&next()),
            "--sample-target" => sample_target = next().parse().expect("--sample-target must be an integer"),
            other => panic!("unknown flag {other}"),
        }
    }

    Args {
        clean: clean.expect("--clean is required"),
        noisy: noisy.expect("--noisy is required"),
        width,
        height,
        frame,
        flat_region,
        texture_region,
        sample_target,
    }
}

/// Loads one frame of an 8-bit raw grey `ffmpeg -pix_fmt gray -f
/// rawvideo` dump, normalised to `[0, 1]`.
fn load_frame(path: &str, width: u32, height: u32, frame_idx: usize) -> Vec<f32> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("reading {path}: {e}"));
    let frame_len = (width * height) as usize;
    let start = frame_idx * frame_len;
    let end = start + frame_len;
    assert!(
        end <= bytes.len(),
        "{path} holds {} frames at {width}x{height}, frame {frame_idx} is out of range",
        bytes.len() / frame_len,
    );
    bytes[start..end].iter().map(|&b| b as f32 / 255.0).collect()
}

/// The channel-scaled sum of squared differences between two 8x8 patches
/// in the same plane, exactly the distance `collab_group_spatial`
/// computes on its own input, replicated on the host so it can be run
/// against the clean frame the kernel never sees.
fn patch_dist(frame: &[f32], width: u32, ax: u32, ay: u32, bx: u32, by: u32) -> f32 {
    let mut sum = 0.0f32;
    for row in 0..PATCH_SIZE {
        for col in 0..PATCH_SIZE {
            let a = frame[((ay + row) * width + (ax + col)) as usize];
            let b = frame[((by + row) * width + (bx + col)) as usize];
            let d = a - b;
            sum += d * d;
        }
    }
    sum * CHANNEL_SCALE_LUMA
}

fn clamp_top_left(v: i32, max_pos: u32) -> u32 {
    v.clamp(0, max_pos as i32) as u32
}

fn unpack_pos(packed: u32) -> (u32, u32) {
    (packed & 0xFFFF, packed >> 16)
}

/// Every distinct clamped candidate position `collab_group_spatial`'s
/// search window visits for a reference at `(rx, ry)`. Two different
/// search offsets landing on the same clamped position near an edge
/// collapse to one entry here, the same dedup the kernel itself applies
/// before a candidate can become a group member.
fn candidate_positions(rx: u32, ry: u32, width: u32, height: u32, spatial_radius: u32) -> Vec<(u32, u32)> {
    let max_x = width - PATCH_SIZE;
    let max_y = height - PATCH_SIZE;
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let r = spatial_radius as i32;
    for dy in -r..=r {
        for dx in -r..=r {
            let cx = clamp_top_left(rx as i32 + dx, max_x);
            let cy = clamp_top_left(ry as i32 + dy, max_y);
            if seen.insert((cx, cy)) {
                out.push((cx, cy));
            }
        }
    }
    out
}

/// Reference grid indices `(ix, iy)` whose 8x8 patch falls entirely
/// inside `region`, thinned to roughly `sample_target` entries by a
/// uniform stride.
fn sample_refs_in_region(
    refs_x: u32,
    refs_y: u32,
    width: u32,
    height: u32,
    region: Region,
    sample_target: usize,
) -> Vec<(u32, u32)> {
    let (x0, y0, x1, y1) = region;
    let mut all = Vec::new();
    for iy in 0..refs_y {
        let ry = ref_pos(iy, height);
        if ry < y0 || ry + PATCH_SIZE > y1 {
            continue;
        }
        for ix in 0..refs_x {
            let rx = ref_pos(ix, width);
            if rx < x0 || rx + PATCH_SIZE > x1 {
                continue;
            }
            all.push((ix, iy));
        }
    }
    let stride = (all.len() / sample_target.max(1)).max(1);
    all.into_iter().step_by(stride).collect()
}

struct GroupResult {
    member_pos: Vec<u32>,
    member_count: Vec<u32>,
}

#[allow(clippy::too_many_arguments)]
fn run_group(
    client: &ComputeClient<R>,
    input: &Handle,
    frame_len: usize,
    width: u32,
    height: u32,
    refs_x: u32,
    refs_y: u32,
    refs: usize,
    k_max: u32,
    spatial_radius: u32,
    floor: f32,
    tau: f32,
    pos_len: usize,
) -> GroupResult {
    let member_pos_buf = client.empty(pos_len * size_of::<u32>());
    let member_count_buf = client.empty(refs * size_of::<u32>());
    let grid = CubeCount::new_2d(refs_x, refs_y);
    let dim = CubeDim::new_2d(8, 8);
    unsafe {
        collab_group_spatial::launch_unchecked::<R>(
            client,
            grid,
            dim,
            1usize,
            ArrayArg::from_raw_parts(input.clone(), frame_len),
            ArrayArg::from_raw_parts(member_pos_buf.clone(), pos_len),
            ArrayArg::from_raw_parts(member_count_buf.clone(), refs),
            0u32,
            floor,
            tau,
            width,
            height,
            1u32,
            k_max,
            spatial_radius,
            refs_x,
        );
    }
    let pos_bytes = client
        .read_one(member_pos_buf)
        .expect("member_pos readback failed");
    let count_bytes = client
        .read_one(member_count_buf)
        .expect("member_count readback failed");
    GroupResult {
        member_pos: u32::from_bytes(&pos_bytes)[..pos_len].to_vec(),
        member_count: u32::from_bytes(&count_bytes)[..refs].to_vec(),
    }
}

/// One config's stage 1 group, hard-threshold, and aggregate, mirroring
/// `CollabPipeline::run_two_stage`'s first half exactly, plus a host
/// readback of the pilot it produces.
#[allow(clippy::too_many_arguments)]
fn run_stage1(
    client: &ComputeClient<R>,
    noisy_gpu: &Handle,
    frame_len: usize,
    width: u32,
    height: u32,
    refs_x: u32,
    refs_y: u32,
    refs: usize,
    params: &CollabParams,
    floor: f32,
    tau: f32,
    pos_len: usize,
    sigma: f32,
) -> (GroupResult, Handle, Vec<f32>) {
    let stage1 = run_group(
        client,
        noisy_gpu,
        frame_len,
        width,
        height,
        refs_x,
        refs_y,
        refs,
        params.k_max,
        params.spatial_radius,
        floor,
        tau,
        pos_len,
    );

    let member_pos_buf = client.create_from_slice(u32::as_bytes(&stage1.member_pos));
    let member_count_buf = client.create_from_slice(u32::as_bytes(&stage1.member_count));
    let member_frame_dummy = client.empty(size_of::<u32>());
    let member_sig2_dummy = client.empty(size_of::<f32>());
    let filtered_dummy = client.empty(size_of::<f32>());
    let group_weight = client.empty(refs * size_of::<f32>());
    let sigma_buf = client.create_from_slice(f32::as_bytes(&[sigma]));
    let profile = dct_noise_profile(params.rho);
    let dct_profile_buf = client.create_from_slice(f32::as_bytes(&profile));
    let accum = client.empty(frame_len * size_of::<i32>());
    let wsum = client.empty(frame_len * size_of::<i32>());
    let wnorm = weight_scale(sigma, &profile);

    let group_grid = CubeCount::new_2d(refs_x, refs_y);
    let group_dim = CubeDim::new_2d(8, 8);
    let zero_dim = 256u32;
    // Same 65,535-workgroups-per-dimension GPU limit `MAX_GRID_1D`
    // guards against elsewhere, clamped by literal here since that
    // constant is crate-private and this diagnostic is a separate
    // binary crate. `collab_zero_accum` is grid-strided, so the clamp
    // still reaches every slot even on frames large enough to need it.
    const MAX_GRID_1D: u32 = 65535;
    let zero_workgroups = (frame_len as u32).div_ceil(zero_dim).min(MAX_GRID_1D);
    unsafe {
        collab_zero_accum::launch_unchecked::<R>(
            client,
            CubeCount::new_1d(zero_workgroups),
            CubeDim::new_1d(zero_dim),
            ArrayArg::from_raw_parts(accum.clone(), frame_len),
            ArrayArg::from_raw_parts(wsum.clone(), frame_len),
            0u32,
            frame_len as u32,
            1u32,
            zero_workgroups * zero_dim,
        );

        collab_filter_ht::launch_unchecked::<R>(
            client,
            group_grid,
            group_dim,
            1usize,
            ArrayArg::from_raw_parts(noisy_gpu.clone(), frame_len),
            ArrayArg::from_raw_parts(member_pos_buf, pos_len),
            ArrayArg::from_raw_parts(member_frame_dummy, 1),
            ArrayArg::from_raw_parts(member_count_buf, refs),
            ArrayArg::from_raw_parts(member_sig2_dummy, 1),
            ArrayArg::from_raw_parts(accum.clone(), frame_len),
            ArrayArg::from_raw_parts(wsum.clone(), frame_len),
            ArrayArg::from_raw_parts(filtered_dummy, 1),
            ArrayArg::from_raw_parts(group_weight.clone(), refs),
            0u32,
            ArrayArg::from_raw_parts(sigma_buf, 1),
            ArrayArg::from_raw_parts(dct_profile_buf, 8),
            params.lambda_ht,
            wnorm,
            ACCUM_SCALE,
            false,
            false,
            false,
            false,
            width,
            height,
            1u32,
            params.k_max,
            1u32,
            refs_x,
        );
    }

    let pilot = client.empty(frame_len * size_of::<f32>());
    let agg_grid = CubeCount::new_2d(width.div_ceil(BLOCK_X), height.div_ceil(BLOCK_Y));
    let agg_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);
    unsafe {
        collab_normalise::launch_unchecked::<R>(
            client,
            agg_grid,
            agg_dim,
            1usize,
            ArrayArg::from_raw_parts(accum, frame_len),
            ArrayArg::from_raw_parts(wsum, frame_len),
            ArrayArg::from_raw_parts(pilot.clone(), frame_len),
            0u32,
            width,
            height,
            1u32,
            1u32,
        );
    }

    let pilot_bytes = client.read_one(pilot.clone()).expect("pilot readback failed");
    let pilot_host = f32::from_bytes(&pilot_bytes)[..frame_len].to_vec();

    (stage1, pilot, pilot_host)
}

struct StageStats {
    discrimination_ratio: f64,
    admitted_mean_clean: f64,
    candidate_pool_mean_clean: f64,
    raw_admission_rate: f64,
    sample_refs: usize,
    member_count_hist: [usize; 9],
}

/// Scores one stage's grouping against clean-domain ground truth over
/// the sampled references in `region`.
///
/// `decision_frame` is whichever frame the stage actually grouped on
/// (noisy for stage 1, the pilot for stage 2), used only for the raw,
/// uncapped admission-rate count. `member_pos`/`member_count` are that
/// stage's real, top-K-capped output, used for the discrimination ratio
/// and the member_count histogram. `clean` is ground truth, never fed to
/// any kernel.
#[allow(clippy::too_many_arguments)]
fn score_stage(
    clean: &[f32],
    decision_frame: &[f32],
    member_pos: &[u32],
    member_count: &[u32],
    width: u32,
    height: u32,
    refs_x: u32,
    k_max: u32,
    spatial_radius: u32,
    floor: f32,
    tau: f32,
    sampled_refs: &[(u32, u32)],
) -> StageStats {
    let mut admitted_clean = Vec::new();
    let mut candidate_pool_clean = Vec::new();
    let mut admit_count = 0usize;
    let mut candidate_count = 0usize;
    let mut member_count_hist = [0usize; 9];

    for &(ix, iy) in sampled_refs {
        let rx = ref_pos(ix, width);
        let ry = ref_pos(iy, height);
        let ref_idx = (iy * refs_x + ix) as usize;

        let candidates = candidate_positions(rx, ry, width, height, spatial_radius);
        for &(cx, cy) in &candidates {
            candidate_pool_clean.push(patch_dist(clean, width, rx, ry, cx, cy) as f64);
            let decision_dist = patch_dist(decision_frame, width, rx, ry, cx, cy);
            let admitted = (decision_dist - floor).max(0.0) <= tau;
            if admitted {
                admit_count += 1;
            }
        }
        candidate_count += candidates.len();

        let count = member_count[ref_idx];
        member_count_hist[count.min(8) as usize] += 1;
        for m in 0..count {
            let packed = member_pos[ref_idx * k_max as usize + m as usize];
            let (px, py) = unpack_pos(packed);
            admitted_clean.push(patch_dist(clean, width, rx, ry, px, py) as f64);
        }
    }

    let mean = |v: &[f64]| -> f64 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    let admitted_mean_clean = mean(&admitted_clean);
    let candidate_pool_mean_clean = mean(&candidate_pool_clean);

    StageStats {
        discrimination_ratio: if candidate_pool_mean_clean > 0.0 {
            admitted_mean_clean / candidate_pool_mean_clean
        } else {
            f64::NAN
        },
        admitted_mean_clean,
        candidate_pool_mean_clean,
        raw_admission_rate: admit_count as f64 / candidate_count.max(1) as f64,
        sample_refs: sampled_refs.len(),
        member_count_hist,
    }
}

fn fmt_hist(hist: &[usize; 9]) -> String {
    let total: usize = hist.iter().sum();
    let mut parts = Vec::new();
    for k in [1usize, 2, 4, 8] {
        let n = hist[k];
        let pct = if total > 0 {
            100.0 * n as f64 / total as f64
        } else {
            0.0
        };
        parts.push(format!("k={k}:{n}({pct:.0}%)"));
    }
    parts.join(" ")
}

fn main() {
    let args = parse_args();
    let client: ComputeClient<R> = {
        let device = <R as Runtime>::Device::default();
        R::client(&device)
    };

    let clean = load_frame(&args.clean, args.width, args.height, args.frame);
    let noisy = load_frame(&args.noisy, args.width, args.height, args.frame);
    assert_eq!(clean.len(), noisy.len());

    let n = clean.len() as f64;
    let mean_sq_diff: f64 = clean
        .iter()
        .zip(noisy.iter())
        .map(|(&c, &y)| ((y - c) as f64).powi(2))
        .sum::<f64>()
        / n;
    let base_sigma = mean_sq_diff.sqrt() as f32;
    println!(
        "measured true injected sigma (normalised [0,1] units, whole frame): {base_sigma:.6} \
         ({:.3} in 8-bit code units)",
        base_sigma * 255.0,
    );

    let width = args.width;
    let height = args.height;
    let frame_len = (width * height) as usize;
    let refs_x = refs_along(width);
    let refs_y = refs_along(height);
    let refs = ref_count(width, height);
    let pos_len = member_buf_len(width, height, 8);

    let regions: [(&str, Region); 2] = [("flat", args.flat_region), ("texture", args.texture_region)];
    let sample_refs: Vec<(&str, Vec<(u32, u32)>)> = regions
        .iter()
        .map(|&(name, region)| {
            (
                name,
                sample_refs_in_region(refs_x, refs_y, width, height, region, args.sample_target),
            )
        })
        .collect();

    let noisy_gpu = client.create_from_slice(f32::as_bytes(&noisy));

    #[derive(Clone, Copy)]
    struct Config {
        label: &'static str,
        residual_sigma_scale: f32,
        tau_match: f32,
    }
    let configs = [
        Config {
            label: "rss=1.0 tau=3.0 (uncalibrated sigma, default tau)",
            residual_sigma_scale: 1.0,
            tau_match: 3.0,
        },
        Config {
            label: "rss=1.9 tau=3.0 (calibrated sigma, default tau)",
            residual_sigma_scale: 1.9,
            tau_match: 3.0,
        },
        Config {
            label: "rss=1.9 tau=1.0 (calibrated sigma, stricter tau)",
            residual_sigma_scale: 1.9,
            tau_match: 1.0,
        },
        Config {
            label: "rss=1.9 tau=0.5 (calibrated sigma, strictest tau)",
            residual_sigma_scale: 1.9,
            tau_match: 0.5,
        },
    ];

    println!();
    println!(
        "{:<48} {:<8} {:>10} {:>10} {:>8} {:>10}  member_count histogram",
        "config / region / stage", "n_refs", "disc_ratio", "adm_rate", "adm_mean", "pool_mean",
    );
    println!("{}", "-".repeat(140));

    for cfg in configs {
        let sigma = base_sigma * cfg.residual_sigma_scale;
        let params = CollabParams {
            tau_match: cfg.tau_match,
            ..CollabParams::default()
        };

        let sum_sq = sigma * sigma;
        let floor = 2.0 * CHANNEL_SCALE_LUMA * sum_sq * PATCH_AREA as f32;
        let floor_epsilon = PATCH_AREA as f32 * CHANNEL_SCALE_LUMA * (1.0f32 / 255.0f32).powi(2);
        let tau = params.tau_match * f32::max(floor, floor_epsilon);

        let (stage1, _pilot_gpu, pilot_host) = run_stage1(
            &client, &noisy_gpu, frame_len, width, height, refs_x, refs_y, refs, &params, floor, tau,
            pos_len, sigma,
        );

        let stage2 = run_group(
            &client,
            &_pilot_gpu,
            frame_len,
            width,
            height,
            refs_x,
            refs_y,
            refs,
            params.k_max,
            params.spatial_radius,
            0.0,
            tau,
            pos_len,
        );

        println!(
            "== {} (sigma={sigma:.6}, floor={floor:.6}, tau={tau:.6}) ==",
            cfg.label
        );
        for (region_name, sampled) in &sample_refs {
            let s1 = score_stage(
                &clean,
                &noisy,
                &stage1.member_pos,
                &stage1.member_count,
                width,
                height,
                refs_x,
                params.k_max,
                params.spatial_radius,
                floor,
                tau,
                sampled,
            );
            let s2 = score_stage(
                &clean,
                &pilot_host,
                &stage2.member_pos,
                &stage2.member_count,
                width,
                height,
                refs_x,
                params.k_max,
                params.spatial_radius,
                0.0,
                tau,
                sampled,
            );

            for (stage_name, s) in [("stage1", &s1), ("stage2", &s2)] {
                println!(
                    "{:<48} {:<8} {:>10.4} {:>9.1}% {:>8.5} {:>10.5}  {}",
                    format!("{} / {} / {}", cfg.label, region_name, stage_name),
                    s.sample_refs,
                    s.discrimination_ratio,
                    s.raw_admission_rate * 100.0,
                    s.admitted_mean_clean,
                    s.candidate_pool_mean_clean,
                    fmt_hist(&s.member_count_hist),
                );
            }
        }
        println!();
    }
}

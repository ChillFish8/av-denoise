use cubecl::prelude::*;

use super::helpers::*;
use crate::nlmeans::noise::{
    NoiseCtx,
    TEMPORAL_NOISE_BLOCK,
    TemporalStatsCtx,
    aggregate_temporal_noise_stats,
    correlation_factor,
    partials_len,
    read_temporal_stats_slot,
    run_noise_estimate,
    run_temporal_noise_stats,
    sigma_from_abs_sum,
    temporal_stats_blocks,
    temporal_stats_buf_bytes,
    temporal_stats_record_len,
};
use crate::nlmeans::*;

/// Uploads `prev`/`new` (densely packed `pixels * stored_ch`, no
/// channel padding needed by these tests) as ring slots 0 and 1, runs
/// the temporal-residual stats kernel diffing slot 1 against slot 0,
/// and returns slot 1's stats region.
fn run_temporal_stats(w: u32, h: u32, stored_ch: u32, prev: &[f32], new: &[f32]) -> Vec<f32> {
    let client = make_client();
    let frame_count = 2u32;
    let frame_len = (w * h * stored_ch) as usize;
    assert_eq!(prev.len(), frame_len);
    assert_eq!(new.len(), frame_len);

    let mut ring = vec![0.0f32; frame_len * frame_count as usize];
    ring[..frame_len].copy_from_slice(prev);
    ring[frame_len..].copy_from_slice(new);

    let input_buf = client.create_from_slice(f32::as_bytes(&ring));
    let stats_buf = client.empty(temporal_stats_buf_bytes(w, h, stored_ch, frame_count));

    let ctx = TemporalStatsCtx {
        width: w,
        height: h,
        stored_ch,
        frame_count,
        slot_new: 1,
        slot_prev: 0,
        input_buf: &input_buf,
        stats_buf: &stats_buf,
    };
    run_temporal_noise_stats::<R>(&client, &ctx).expect("temporal noise stats dispatch failed");

    read_temporal_stats_slot::<R>(&client, &stats_buf, w, h, stored_ch, frame_count, 1)
        .expect("readback failed")
}

/// CPU oracle mirroring `nlm_temporal_noise_stats`'s block geometry
/// and summation independently of the kernel, so the kernel-unit
/// tests cross-check the GPU output against a from-scratch
/// implementation rather than a hand-derived closed form.
fn reference_temporal_stats(w: u32, h: u32, stored_ch: u32, prev: &[f32], new: &[f32]) -> Vec<f32> {
    let sch = stored_ch as usize;
    let (blocks_x, blocks_y) = temporal_stats_blocks(w, h);
    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let mut out = vec![0.0f32; (blocks_x * blocks_y) as usize * record_len];

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let ox = bx * TEMPORAL_NOISE_BLOCK;
            let oy = by * TEMPORAL_NOISE_BLOCK;
            let bw = TEMPORAL_NOISE_BLOCK.min(w - ox);
            let bh = TEMPORAL_NOISE_BLOCK.min(h - oy);

            let mut sum_d = vec![0.0f32; sch];
            let mut sum_d2 = vec![0.0f32; sch];
            let mut sum_lag = 0.0f32;

            for ly in 0..bh {
                let y = oy + ly;
                let mut d0_row = Vec::with_capacity(bw as usize);
                for lx in 0..bw {
                    let x = ox + lx;
                    let idx = (y * w + x) as usize;
                    for c in 0..sch {
                        let d = new[idx * sch + c] - prev[idx * sch + c];
                        sum_d[c] += d;
                        sum_d2[c] += d * d;
                        if c == 0 {
                            d0_row.push(d);
                        }
                    }
                }
                for lx in 0..(bw as usize).saturating_sub(1) {
                    sum_lag += d0_row[lx] * d0_row[lx + 1];
                }
            }

            let block_index = (by * blocks_x + bx) as usize;
            let base = block_index * record_len;
            out[base..base + sch].copy_from_slice(&sum_d);
            out[base + sch..base + 2 * sch].copy_from_slice(&sum_d2);
            out[base + 2 * sch] = sum_lag;
        }
    }

    out
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, msg: &str) {
    assert_eq!(actual.len(), expected.len(), "{msg}: length mismatch");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{msg}: index {i}: got {a}, expected {e} (tol {tol})"
        );
    }
}

/// A constant diff over a frame with an exact 2×2 grid of full
/// `16×16` blocks (no ragged edges): every block's sums reduce to a
/// closed form, `sum_d = n·K`, `sum_d2 = n·K²`, `sum_lag = n_pairs·K²`.
#[test]
fn kernel_uniform_diff_exact_sums() {
    let w = 32;
    let h = 32;
    let k = 0.1f32;
    let prev = vec![0.0f32; (w * h) as usize];
    let new = vec![k; (w * h) as usize];

    let n = 256.0f32;
    let n_pairs = 240.0f32;
    let expected_block = [n * k, n * k * k, n_pairs * k * k];

    let got = run_temporal_stats(w, h, 1, &prev, &new);
    let oracle = reference_temporal_stats(w, h, 1, &prev, &new);

    for block in 0..4 {
        let rec = &got[block * 3..block * 3 + 3];
        assert_close(rec, &expected_block, 1e-4, &format!("block {block}"));
    }
    assert_close(&got, &oracle, 1e-4, "kernel vs CPU oracle");
}

/// A horizontal luma ramp diff (`d0(x, y) = x`) over an exact `3×2`
/// grid of full blocks. No closed form asserted directly; this checks
/// the kernel against the independent CPU oracle instead, exercising
/// non-constant per-pixel content the uniform test can't.
#[test]
fn kernel_gradient_diff_exact_sums() {
    let w = 48;
    let h = 32;
    let mut prev = vec![0.0f32; (w * h) as usize];
    let mut new = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            new[(y * w + x) as usize] = x as f32;
        }
    }
    // Keep `prev` at zero so `d = new`.
    prev.fill(0.0);

    let got = run_temporal_stats(w, h, 1, &prev, &new);
    let oracle = reference_temporal_stats(w, h, 1, &prev, &new);
    assert_close(&got, &oracle, 1e-2, "kernel vs CPU oracle (gradient diff)");
}

/// `33×17`: ragged on both axes (`blocks_x = 3` with a 1-pixel-wide
/// last column, `blocks_y = 2` with a 1-pixel-tall last row). A
/// uniform diff still has a closed form per block, just with each
/// block's own truncated `n`/`n_pairs`, which this asserts directly
/// rather than only against the oracle.
#[test]
fn kernel_ragged_block_dims() {
    let w = 33;
    let h = 17;
    let k = 0.2f32;
    let prev = vec![0.0f32; (w * h) as usize];
    let new = vec![k; (w * h) as usize];

    let (blocks_x, blocks_y) = temporal_stats_blocks(w, h);
    assert_eq!((blocks_x, blocks_y), (3, 2));

    let got = run_temporal_stats(w, h, 1, &prev, &new);
    let oracle = reference_temporal_stats(w, h, 1, &prev, &new);
    assert_close(&got, &oracle, 1e-4, "kernel vs CPU oracle (ragged)");

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let bw = TEMPORAL_NOISE_BLOCK.min(w - bx * TEMPORAL_NOISE_BLOCK);
            let bh = TEMPORAL_NOISE_BLOCK.min(h - by * TEMPORAL_NOISE_BLOCK);
            let n = (bw * bh) as f32;
            let n_pairs = (bh * bw.saturating_sub(1)) as f32;
            let expected = [n * k, n * k * k, n_pairs * k * k];

            let block = (by * blocks_x + bx) as usize;
            let rec = &got[block * 3..block * 3 + 3];
            // A looser tolerance than the oracle comparison above: this
            // is a hand-derived closed form summing up to 256 copies of
            // a non-dyadic float (0.2), so it accumulates a little more
            // floating-point slack than comparing two summations that
            // both walk the same block in the same order.
            assert_close(
                rec,
                &expected,
                5e-3,
                &format!("ragged block ({bx},{by}), dims {bw}x{bh}"),
            );
        }
    }
}

/// White noise (no spatial correlation): the temporal estimator must
/// recover the known marginal sigma within 10%.
#[test]
fn white_noise_pair_recovers_known_sigma() {
    let size = 256;
    let true_sigma = 8.0 / 255.0;

    let prev = noisy_copy(size, 0.5, true_sigma, 1);
    let new = noisy_copy(size, 0.5, true_sigma, 2);

    let records = run_temporal_stats(size, size, 1, &prev, &new);
    let sample = aggregate_temporal_noise_stats(&records, 1, 1, size, size)
        .expect("a static white-noise pair should clear the static-block floor");

    let rel_err = (sample.sigma[0] - true_sigma).abs() / true_sigma;
    assert!(
        rel_err <= 0.10,
        "estimated sigma {} vs true {true_sigma} (rel err {rel_err:.3})",
        sample.sigma[0]
    );
}

/// Spatially-correlated grain (same white field, horizontally
/// blurred): the marginal sigma is still recoverable within 10%
/// (temporal variance ignores spatial correlation), and rho must
/// clearly register the correlation the blur introduces — the whole
/// point of this estimator versus Immerkær, which reads correlated
/// grain low (see `hq_temporal_folds_correlated_grain_above_immerkaer_alone`).
#[test]
fn correlated_noise_pair_recovers_marginal_sigma_and_rho() {
    let w = 256;
    let h = 256;
    let sigma_marginal = 8.0 / 255.0;
    // Horizontal binomial blur [0.25, 0.5, 0.25]: variance scales by
    // the sum of squared weights (0.375), so the pre-blur sigma must
    // be scaled up to land the *blurred* field at `sigma_marginal`.
    let sigma_pre = sigma_marginal / 0.375f32.sqrt();

    let prev = correlated_noisy_frame(w, h, 0.5, sigma_pre, 11);
    let new = correlated_noisy_frame(w, h, 0.5, sigma_pre, 12);

    let records = run_temporal_stats(w, h, 1, &prev, &new);
    let sample = aggregate_temporal_noise_stats(&records, 1, 1, w, h)
        .expect("a static correlated-noise pair should clear the static-block floor");

    let rel_err = (sample.sigma[0] - sigma_marginal).abs() / sigma_marginal;
    assert!(
        rel_err <= 0.10,
        "estimated sigma {} vs marginal truth {sigma_marginal} (rel err {rel_err:.3})",
        sample.sigma[0]
    );
    assert!(
        sample.rho > 0.4,
        "expected rho > 0.4 for horizontally-blurred grain, got {}",
        sample.rho
    );
}

/// A rich horizontal ramp shifted by a few pixels simulates motion: the
/// systematic per-pixel offset it introduces overwhelms the static
/// gate almost everywhere, so the aggregation should classify the
/// large majority of blocks (if not all) as non-static.
#[test]
fn moving_content_pair_mostly_non_static() {
    let w = 64;
    let h = 64;
    let shift = 4u32;

    let mut prev = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            prev[(y * w + x) as usize] = x as f32 / w as f32;
        }
    }
    let mut new = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let xs = (x + shift).min(w - 1);
            new[(y * w + x) as usize] = prev[(y * w + xs) as usize];
        }
    }

    let records = run_temporal_stats(w, h, 1, &prev, &new);
    let sample = aggregate_temporal_noise_stats(&records, 1, 1, w, h);
    let static_fraction = sample.map(|s| s.static_fraction).unwrap_or(0.0);

    assert!(
        static_fraction < 0.5,
        "expected mostly non-static blocks under a content shift, got static_fraction={static_fraction}"
    );
}

/// End-to-end: an HQ r2 denoiser fed a correlated-noise (spatially
/// blurred grain) synthetic stream must fold its estimator state to
/// within 25% of the marginal truth scaled by the correlation
/// correction, in contrast to what Immerkær alone reads on the very
/// same content (multiple times lower — this is the regression this
/// estimator exists to fix). The blur's analytic lag-1
/// autocorrelation is 2/3, from the same coefficient sums that give
/// the 0.375 variance scale.
#[test]
fn hq_temporal_folds_correlated_grain_above_immerkaer_alone() {
    let client = make_client();
    let w = 128;
    let h = 128;
    let sigma_marginal = 8.0 / 255.0;
    let sigma_pre = sigma_marginal / 0.375f32.sqrt();
    let base = 0.5f32;

    let n_frames = 14;
    let frames: Vec<Vec<f32>> = (0..n_frames)
        .map(|i| correlated_noisy_frame(w, h, base, sigma_pre, 100 + i as u32))
        .collect();

    // What the old (Immerkær-only) estimator would read on this same
    // correlated content, computed directly rather than through the
    // denoiser.
    let immerkaer_only = {
        let input_buf = client.create_from_slice(f32::as_bytes(&frames[0]));
        let partials_buf = client.empty(partials_len(w, h) * size_of::<f32>());
        let results_buf = client.empty(4 * size_of::<f32>());
        let ctx = NoiseCtx {
            width: w,
            height: h,
            channels: 1,
            stored_ch: 1,
            frame_count: 1,
            frame: 0,
            slot: 0,
            input_buf: &input_buf,
            partials_buf: &partials_buf,
            results_buf: &results_buf,
        };
        run_noise_estimate::<R>(&client, &ctx).expect("immerkaer dispatch failed");
        let bytes = client.read_one(results_buf).expect("immerkaer readback failed");
        let data = f32::from_bytes(&bytes);
        sigma_from_abs_sum(data[0], w, h)
    };
    assert!(
        immerkaer_only < sigma_marginal * 0.5,
        "expected Immerkær alone to read well below the marginal truth {sigma_marginal} \
         on correlated grain, got {immerkaer_only}"
    );

    let params = NlmParams {
        temporal_radius: 2,
        search_radius: 2,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        hq: Some(HqParams {
            auto_strength: true,
            noise_floor: true,
            sigma_override: None,
            temporal_confidence: false,
            thsad_scale: 1.0,
        }),
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    for frame in &frames {
        denoiser.push_frame(frame);
        let _ = denoiser.denoise().unwrap();
    }

    let folded = denoiser
        .noise_estimator
        .current()
        .expect("estimator should hold a value after several full-window submits")[0];

    let expected = sigma_marginal * correlation_factor(2.0 / 3.0);
    let rel_err = (folded - expected).abs() / expected;
    assert!(
        rel_err <= 0.25,
        "folded sigma {folded} vs corrected truth {expected} (rel err {rel_err:.3}), \
         versus Immerkær-alone {immerkaer_only}"
    );
}

use cubecl::prelude::*;

use super::helpers::*;
use crate::nlmeans::noise::{
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
    NoiseCtx,
    TemporalStatsCtx,
    TEMPORAL_LUMA_FLATNESS,
    TEMPORAL_LUMA_MAX,
    TEMPORAL_LUMA_MIN,
    TEMPORAL_LUMA_SUM,
    TEMPORAL_NOISE_BLOCK,
};
use crate::nlmeans::*;

/// Uploads `prev`/`new` (densely packed `pixels * stored_ch`, no
/// channel padding needed by these tests) as ring slots 0 and 1, runs
/// the temporal-residual stats kernel diffing slot 1 against slot 0,
/// and returns slot 1's stats region.
fn run_temporal_stats(
    w: u32,
    h: u32,
    stored_ch: u32,
    prev: &[f32],
    new: &[f32],
    luma_fields: bool,
) -> Vec<f32> {
    let client = make_client();
    let frame_count = 2u32;
    let frame_len = (w * h * stored_ch) as usize;
    assert_eq!(prev.len(), frame_len);
    assert_eq!(new.len(), frame_len);

    let mut ring = vec![0.0f32; frame_len * frame_count as usize];
    ring[..frame_len].copy_from_slice(prev);
    ring[frame_len..].copy_from_slice(new);

    let input_buf = client.create_from_slice(f32::as_bytes(&ring));
    let align = test_align();
    let stats_buf = client.empty(temporal_stats_buf_bytes(w, h, stored_ch, frame_count, align));

    let ctx = TemporalStatsCtx {
        width: w,
        height: h,
        stored_ch,
        frame_count,
        slot_new: 1,
        slot_prev: 0,
        input_buf: &input_buf,
        stats_buf: &stats_buf,
        align,
    };
    run_temporal_noise_stats::<R>(&client, &ctx, luma_fields).expect("temporal noise stats dispatch failed");

    read_temporal_stats_slot::<R>(&client, &stats_buf, w, h, stored_ch, frame_count, 1, align)
        .expect("readback failed")
}

/// CPU oracle mirroring `nlm_temporal_noise_stats`'s block geometry
/// and summation independently of the kernel, so the kernel-unit
/// tests cross-check the GPU output against a from-scratch
/// implementation rather than a hand-derived closed form.
///
/// Only valid for single-channel (`stored_ch == 1`) `new`/`prev` frames,
/// because the luma fields it adds always read channel 0 at a stride of
/// `stored_ch`, matching `sch` below only in that case. Every caller in
/// this file passes `stored_ch == 1`.
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

            let [luma_sum, flatness, luma_min, luma_max] = luma_fields_host(new, prev, w, h, bx, by);

            let block_index = (by * blocks_x + bx) as usize;
            let base = block_index * record_len;
            out[base..base + sch].copy_from_slice(&sum_d);
            out[base + sch..base + 2 * sch].copy_from_slice(&sum_d2);
            out[base + 2 * sch] = sum_lag;
            out[base + 2 * sch + TEMPORAL_LUMA_SUM as usize] = luma_sum;
            out[base + 2 * sch + TEMPORAL_LUMA_FLATNESS as usize] = flatness;
            out[base + 2 * sch + TEMPORAL_LUMA_MIN as usize] = luma_min;
            out[base + 2 * sch + TEMPORAL_LUMA_MAX as usize] = luma_max;
        }
    }

    out
}

/// The four luma fields one block's record carries, computed on the
/// host.
///
/// `new` and `prev` are densely packed single-channel frames, matching
/// every caller in this file.
fn luma_fields_host(new: &[f32], prev: &[f32], width: u32, height: u32, bx: u32, by: u32) -> [f32; 4] {
    let block = TEMPORAL_NOISE_BLOCK;
    let origin_x = bx * block;
    let origin_y = by * block;
    let block_w = block.min(width - origin_x);
    let block_h = block.min(height - origin_y);

    let mut sum = 0.0f32;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for y in 0..block_h {
        for x in 0..block_w {
            let value = new[((origin_y + y) * width + origin_x + x) as usize];
            sum += value;
            min = min.min(value);
            max = max.max(value);
        }
    }

    let full = block_w == block && block_h == block;
    let flatness = if full {
        let mut cells = [0.0f32; 64];
        for cy in 0..8u32 {
            for cx in 0..8u32 {
                let mut cell = 0.0f32;
                for dy in 0..2u32 {
                    for dx in 0..2u32 {
                        let idx = ((origin_y + 2 * cy + dy) * width + origin_x + 2 * cx + dx) as usize;
                        cell += 0.5 * (new[idx] + prev[idx]);
                    }
                }
                cells[(cy * 8 + cx) as usize] = cell / 4.0;
            }
        }

        let mut energy = 0.0f32;
        for cy in 0..8usize {
            for cx in 0..8usize {
                let here = cells[cy * 8 + cx];
                if cx + 1 < 8 {
                    let right = cells[cy * 8 + cx + 1];
                    energy += (here - right) * (here - right);
                }
                if cy + 1 < 8 {
                    let below = cells[(cy + 1) * 8 + cx];
                    energy += (here - below) * (here - below);
                }
            }
        }
        energy / 112.0
    } else {
        3.0e38
    };

    [sum, flatness, min, max]
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

/// A constant difference over a frame that grids into exactly four full
/// blocks, with no ragged edges.
///
/// Every block's sums then reduce to a closed form, so the expected
/// values can be written out by hand.
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

    let got = run_temporal_stats(w, h, 1, &prev, &new, true);
    let oracle = reference_temporal_stats(w, h, 1, &prev, &new);

    let record_len = temporal_stats_record_len(1) as usize;
    for block in 0..4 {
        let rec = &got[block * record_len..block * record_len + 3];
        assert_close(rec, &expected_block, 1e-4, &format!("block {block}"));
    }
    assert_close(&got, &oracle, 1e-4, "kernel vs CPU oracle");
}

/// A horizontal ramp difference over a grid of six full blocks.
///
/// Nothing is asserted in closed form here. The kernel is checked
/// against the independent CPU reference instead, which covers the
/// varying per-pixel content the uniform test cannot reach.
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

    let got = run_temporal_stats(w, h, 1, &prev, &new, true);
    let oracle = reference_temporal_stats(w, h, 1, &prev, &new);
    assert_close(&got, &oracle, 1e-2, "kernel vs CPU oracle (gradient diff)");
}

/// A 33x17 frame is ragged on both axes, leaving a last column one
/// pixel wide and a last row one pixel tall.
///
/// A uniform difference still has a closed form per block, just with
/// each block's own truncated counts.
///
/// This asserts those directly rather than only against the CPU
/// reference.
#[test]
fn kernel_ragged_block_dims() {
    let w = 33;
    let h = 17;
    let k = 0.2f32;
    let prev = vec![0.0f32; (w * h) as usize];
    let new = vec![k; (w * h) as usize];

    let (blocks_x, blocks_y) = temporal_stats_blocks(w, h);
    assert_eq!((blocks_x, blocks_y), (3, 2));

    let got = run_temporal_stats(w, h, 1, &prev, &new, true);
    let oracle = reference_temporal_stats(w, h, 1, &prev, &new);
    // `luma_sum` reduces on the device through a shared-memory tree,
    // rather than the oracle's plain sequential loop. Summing the same
    // 256 copies of a value like 0.2, which float32 cannot hold exactly,
    // in a different order picks up its own small rounding error, so
    // this needs a little more slack than the other lanes.
    assert_close(&got, &oracle, 5e-4, "kernel vs CPU oracle (ragged)");

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let bw = TEMPORAL_NOISE_BLOCK.min(w - bx * TEMPORAL_NOISE_BLOCK);
            let bh = TEMPORAL_NOISE_BLOCK.min(h - by * TEMPORAL_NOISE_BLOCK);
            let n = (bw * bh) as f32;
            let n_pairs = (bh * bw.saturating_sub(1)) as f32;
            let expected = [n * k, n * k * k, n_pairs * k * k];

            let block = (by * blocks_x + bx) as usize;
            let record_len = temporal_stats_record_len(1) as usize;
            let rec = &got[block * record_len..block * record_len + 3];
            // A looser tolerance than the reference comparison above.
            // This is a hand-derived closed form summing up to 256
            // copies of a value f32 cannot hold exactly, so it picks up
            // a little more floating-point slack than comparing two
            // sums that walk the same block in the same order.
            assert_close(
                rec,
                &expected,
                5e-3,
                &format!("ragged block ({bx},{by}), dims {bw}x{bh}"),
            );
        }
    }
}

fn assert_relative(actual: f32, expected: f32, tol: f32, msg: &str) {
    let rel_err = (actual - expected).abs() / expected.abs().max(1e-8);
    assert!(
        rel_err <= tol,
        "{msg}: got {actual}, expected {expected} (rel err {rel_err})"
    );
}

/// The four luma fields, checked block by block against the host mirror
/// on a 40x24 frame that grids into 3x2 blocks, ragged on the right
/// column and the bottom row.
#[test]
fn luma_fields_match_the_host_mirror() {
    let w = 40u32;
    let h = 24u32;

    let mut new = vec![0.0f32; (w * h) as usize];
    let mut prev = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let value = 0.2 + 0.6 * ((x * 7 + y * 13) % 17) as f32 / 17.0;
            new[idx] = value;
            let offset = 0.02 * (((x + y) % 3) as f32 - 1.0);
            prev[idx] = (value + offset).clamp(0.0, 1.0);
        }
    }

    let stored_ch = 1u32;
    let got = run_temporal_stats(w, h, stored_ch, &prev, &new, true);
    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let luma_base = 2 * stored_ch as usize;
    let (blocks_x, blocks_y) = temporal_stats_blocks(w, h);
    assert_eq!((blocks_x, blocks_y), (3, 2));

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let block_index = (by * blocks_x + bx) as usize;
            let rec = &got[block_index * record_len..(block_index + 1) * record_len];
            let [expected_sum, expected_flatness, expected_min, expected_max] =
                luma_fields_host(&new, &prev, w, h, bx, by);

            let got_sum = rec[luma_base + TEMPORAL_LUMA_SUM as usize];
            let got_flatness = rec[luma_base + TEMPORAL_LUMA_FLATNESS as usize];
            let got_min = rec[luma_base + TEMPORAL_LUMA_MIN as usize];
            let got_max = rec[luma_base + TEMPORAL_LUMA_MAX as usize];

            let label = format!("block ({bx},{by})");
            assert_relative(got_sum, expected_sum, 1e-4, &format!("{label} luma_sum"));
            if expected_flatness == 3.0e38 {
                assert_eq!(got_flatness, 3.0e38, "{label}: ragged flatness sentinel");
            } else {
                assert_relative(
                    got_flatness,
                    expected_flatness,
                    1e-4,
                    &format!("{label} flatness"),
                );
            }
            assert_eq!(got_min, expected_min, "{label}: luma_min");
            assert_eq!(got_max, expected_max, "{label}: luma_max");
        }
    }

    // The existing sum_d, sum_d^2 and lag lanes stay unaffected by the
    // new luma fields appended after them.
    let oracle = reference_temporal_stats(w, h, 1, &prev, &new);
    for block in 0..(blocks_x * blocks_y) as usize {
        let got_rec = &got[block * record_len..block * record_len + 3];
        let oracle_rec = &oracle[block * record_len..block * record_len + 3];
        assert_close(
            got_rec,
            oracle_rec,
            1e-4,
            &format!("block {block} sum_d/sum_d2/lag"),
        );
    }
}

/// A full 16x16 block of a constant base plus independent noise in each
/// frame has to read low flatness, showing the smoothed-temporal-mean
/// gate does not mistake the noise itself for texture.
#[test]
fn flat_noisy_block_reads_low_flatness() {
    let size = 16u32;
    let sigma = 0.01f32;
    let stored_ch = 1u32;
    let prev = noisy_copy(size, 0.5, sigma, 1);
    let new = noisy_copy(size, 0.5, sigma, 2);

    let got = run_temporal_stats(size, size, stored_ch, &prev, &new, true);
    let luma_base = 2 * stored_ch as usize;
    let flatness = got[luma_base + TEMPORAL_LUMA_FLATNESS as usize];

    assert!(
        flatness < 0.5 * sigma * sigma,
        "expected the smoothed-mean gate to read low flatness on pure noise, got {flatness}"
    );
}

/// With `luma_fields` off, the four luma lanes read 0, and every other
/// lane matches the `luma_fields` on run bit for bit.
#[test]
fn luma_fields_off_leaves_the_luma_lanes_zero_and_the_rest_unchanged() {
    let w = 40u32;
    let h = 24u32;
    let stored_ch = 1u32;

    let mut new = vec![0.0f32; (w * h) as usize];
    let mut prev = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let value = 0.2 + 0.6 * ((x * 7 + y * 13) % 17) as f32 / 17.0;
            new[idx] = value;
            let offset = 0.02 * (((x + y) % 3) as f32 - 1.0);
            prev[idx] = (value + offset).clamp(0.0, 1.0);
        }
    }

    let with_luma = run_temporal_stats(w, h, stored_ch, &prev, &new, true);
    let without_luma = run_temporal_stats(w, h, stored_ch, &prev, &new, false);

    let record_len = temporal_stats_record_len(stored_ch) as usize;
    let luma_base = 2 * stored_ch as usize;
    let (blocks_x, blocks_y) = temporal_stats_blocks(w, h);

    for block in 0..(blocks_x * blocks_y) as usize {
        let with_rec = &with_luma[block * record_len..block * record_len + luma_base];
        let without_rec = &without_luma[block * record_len..block * record_len + luma_base];
        assert_eq!(
            with_rec, without_rec,
            "block {block}: sum_d/sum_d2/lag must not depend on luma_fields"
        );

        for offset in [
            TEMPORAL_LUMA_SUM,
            TEMPORAL_LUMA_FLATNESS,
            TEMPORAL_LUMA_MIN,
            TEMPORAL_LUMA_MAX,
        ] {
            let value = without_luma[block * record_len + luma_base + offset as usize];
            assert_eq!(
                value, 0.0,
                "block {block}: luma lane at offset {offset} must read 0 with luma_fields off, got \
                 {value}"
            );
        }
    }
}

#[test]
fn white_noise_pair_recovers_known_sigma() {
    let size = 256;
    let true_sigma = 8.0 / 255.0;

    let prev = noisy_copy(size, 0.5, true_sigma, 1);
    let new = noisy_copy(size, 0.5, true_sigma, 2);

    let records = run_temporal_stats(size, size, 1, &prev, &new, true);
    let sample = aggregate_temporal_noise_stats(&records, 1, 1, size, size)
        .expect("a static white-noise pair should clear the static-block floor");

    let rel_err = (sample.sigma[0] - true_sigma).abs() / true_sigma;
    assert!(
        rel_err <= 0.10,
        "estimated sigma {} vs true {true_sigma} (rel err {rel_err:.3})",
        sample.sigma[0]
    );
}

/// Grain that is correlated between neighbouring pixels, made by
/// blurring a white noise field horizontally.
///
/// The sigma is still recoverable within 10%, because the temporal
/// measurement does not care about spatial correlation.
///
/// The correlation reading has to clearly register the blur. That is the
/// whole point of this estimator next to Immerkær, which reads
/// correlated grain low. See
/// `hq_temporal_folds_correlated_grain_above_immerkaer_alone`.
#[test]
fn correlated_noise_pair_recovers_marginal_sigma_and_rho() {
    let w = 256;
    let h = 256;
    let sigma_marginal = 8.0 / 255.0;
    // The blur scales the variance by 0.375, the sum of its squared
    // weights, so the sigma before the blur has to be raised to land
    // the blurred field on the target.
    let sigma_pre = sigma_marginal / 0.375f32.sqrt();

    let prev = correlated_noisy_frame(w, h, 0.5, sigma_pre, 11);
    let new = correlated_noisy_frame(w, h, 0.5, sigma_pre, 12);

    let records = run_temporal_stats(w, h, 1, &prev, &new, true);
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

/// A horizontal ramp shifted by a few pixels stands in for motion.
///
/// The steady per-pixel offset that introduces overwhelms the static
/// check almost everywhere, so most blocks, if not all of them, should
/// come out as non-static.
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

    let records = run_temporal_stats(w, h, 1, &prev, &new, true);
    let sample = aggregate_temporal_noise_stats(&records, 1, 1, w, h);
    let static_fraction = sample.map(|s| s.static_fraction).unwrap_or(0.0);

    assert!(
        static_fraction < 0.5,
        "expected mostly non-static blocks under a content shift, got static_fraction={static_fraction}"
    );
}

/// The whole pipeline, fed a synthetic stream of correlated grain.
///
/// The denoiser's estimator has to settle within 25% of the true sigma
/// once the correlation correction is applied.
///
/// Immerkær alone reads several times lower on the same content, which
/// is the exact problem this estimator exists to fix.
///
/// The blur's lag-1 correlation works out at two thirds, from the same
/// coefficients that give the 0.375 variance scale.
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
            sigma_scale: 1.0,
            windowed_noise_estimation: false,
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

/// `rho_smoothed`'s first update must seed directly from the first
/// temporal sample rather than blend it against an assumed 0. On
/// correlated grain with true rho 2/3, blending from 0 with
/// `EMA_ALPHA = 0.2` would read about 0.13 after the very first
/// sample, an 80% relative error, while seeding directly should land
/// within the same 25% tolerance
/// `hq_temporal_folds_correlated_grain_above_immerkaer_alone` uses for
/// the folded sigma on the same content.
#[test]
fn rho_smoothed_seeds_from_first_sample_not_from_zero() {
    let client = make_client();
    let w = 128;
    let h = 128;
    let sigma_marginal = 8.0 / 255.0;
    let sigma_pre = sigma_marginal / 0.375f32.sqrt();
    let base = 0.5f32;
    let true_rho = 2.0 / 3.0;

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
            sigma_scale: 1.0,
            windowed_noise_estimation: false,
        }),
    };

    let mut denoiser = NlmDenoiser::<R>::new(&client, params, w, h);
    let mut first_seeded_rho = None;
    for i in 0..(w.min(h)) {
        let frame = correlated_noisy_frame(w, h, base, sigma_pre, 200 + i);
        denoiser.push_frame(&frame);
        let _ = denoiser.denoise().unwrap();
        if let Some(rho) = denoiser.rho_smoothed {
            first_seeded_rho = Some(rho);
            break;
        }
    }

    let first_seeded_rho =
        first_seeded_rho.expect("estimator should hold a value after enough full-window submits");
    let rel_err = (first_seeded_rho - true_rho).abs() / true_rho;
    assert!(
        rel_err <= 0.25,
        "rho_smoothed after its first update was {first_seeded_rho} vs true rho {true_rho} \
         (rel err {rel_err:.3}); a from-zero blend would read close to {}",
        0.2 * true_rho
    );
}

/// A frame split into a static top half and a panning bottom half.
///
/// The top is flat content with independent measurement noise. The
/// bottom is fine texture panning a few pixels between frames, with the
/// same noise on top.
///
/// The panning residual averages close to zero over a block, so the mean
/// check alone lets nearly the whole frame through.
///
/// That is the case `moving_content_pair_mostly_non_static` does not
/// cover, because its ramp leaves a block-mean residual far above the
/// check rather than near zero.
///
/// Only the top half's variance is really noise, so the aggregation has
/// to recover close to the true sigma and reject most of the panning
/// half.
#[test]
fn moving_texture_pair_near_zero_mean_residual_recovers_true_sigma() {
    let w = 128;
    let h = 128;
    let true_sigma = 2.0 / 255.0;
    let texture_sigma_pre = 24.0 / 255.0 / 0.375f32.sqrt();
    let shift = 3u32;
    let base = 0.5f32;

    // One texture field for the bottom half, independent of the
    // measurement noise added below.
    let raw_texture = correlated_noisy_frame(w, h / 2, base, texture_sigma_pre, 1);

    let mut clean_prev = vec![base; (w * h) as usize];
    let mut clean_new = vec![base; (w * h) as usize];
    for y in 0..(h / 2) {
        for x in 0..w {
            let prev_val = raw_texture[(y * w + x) as usize];
            let xs = (x + shift).min(w - 1);
            let new_val = raw_texture[(y * w + xs) as usize];
            let out_y = h / 2 + y;
            clean_prev[(out_y * w + x) as usize] = prev_val;
            clean_new[(out_y * w + x) as usize] = new_val;
        }
    }

    let prev = noisy_field_over(&clean_prev, w, h, true_sigma, 11);
    let new = noisy_field_over(&clean_new, w, h, true_sigma, 12);

    let records = run_temporal_stats(w, h, 1, &prev, &new, true);
    let sample = aggregate_temporal_noise_stats(&records, 1, 1, w, h)
        .expect("the static top half should clear STATIC_FRACTION_MIN on its own");

    let (blocks_x, blocks_y) = temporal_stats_blocks(w, h);
    let total_blocks = blocks_x * blocks_y;
    let top_half_blocks = total_blocks / 2;
    assert!(
        (sample.static_fraction * total_blocks as f32) <= top_half_blocks as f32 * 1.5,
        "expected static_fraction to stay close to the true static top half ({} of {total_blocks} \
         blocks), got {}",
        top_half_blocks,
        sample.static_fraction
    );

    let rel_err = (sample.sigma[0] - true_sigma).abs() / true_sigma;
    assert!(
        rel_err <= 0.25,
        "estimated sigma {} vs true static-half sigma {true_sigma} (rel err {rel_err:.3})",
        sample.sigma[0]
    );
}

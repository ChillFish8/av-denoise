use super::helpers::*;
use crate::nlmeans::*;

/// Pearson correlation between a field and its neighbour one pixel to
/// the right, skipping the last column so no clamped edge pixel is ever
/// paired with itself.
fn lag1_horizontal(field: &[f32], w: u32, h: u32) -> f64 {
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_ab = 0.0f64;
    let mut sum_aa = 0.0f64;
    let mut sum_bb = 0.0f64;
    let mut n = 0.0f64;
    for y in 0..h {
        for x in 0..(w - 1) {
            let a = field[(y * w + x) as usize] as f64;
            let b = field[(y * w + x + 1) as usize] as f64;
            sum_a += a;
            sum_b += b;
            sum_ab += a * b;
            sum_aa += a * a;
            sum_bb += b * b;
            n += 1.0;
        }
    }
    let mean_a = sum_a / n;
    let mean_b = sum_b / n;
    let cov = sum_ab / n - mean_a * mean_b;
    let var_a = sum_aa / n - mean_a * mean_a;
    let var_b = sum_bb / n - mean_b * mean_b;
    cov / (var_a.sqrt() * var_b.sqrt())
}

/// Same as [`lag1_horizontal`] but along y, skipping the last row.
fn lag1_vertical(field: &[f32], w: u32, h: u32) -> f64 {
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_ab = 0.0f64;
    let mut sum_aa = 0.0f64;
    let mut sum_bb = 0.0f64;
    let mut n = 0.0f64;
    for y in 0..(h - 1) {
        for x in 0..w {
            let a = field[(y * w + x) as usize] as f64;
            let b = field[((y + 1) * w + x) as usize] as f64;
            sum_a += a;
            sum_b += b;
            sum_ab += a * b;
            sum_aa += a * a;
            sum_bb += b * b;
            n += 1.0;
        }
    }
    let mean_a = sum_a / n;
    let mean_b = sum_b / n;
    let cov = sum_ab / n - mean_a * mean_b;
    let var_a = sum_aa / n - mean_a * mean_a;
    let var_b = sum_bb / n - mean_b * mean_b;
    cov / (var_a.sqrt() * var_b.sqrt())
}

/// The actual standard deviation a `[a, 1 - 2a, a]` horizontal blur of
/// unit-variance white noise leaves behind, scaled by `sigma_pre`.
/// `a = 0` (no blur) reduces to `sigma_pre` itself, matching
/// [`noisy_field_over`]'s plain injection.
fn tap_sigma(sigma_pre: f32, a: f32) -> f32 {
    let b = 1.0 - 2.0 * a;
    sigma_pre * (2.0 * a * a + b * b).sqrt()
}

fn std_dev(field: &[f32]) -> f64 {
    let n = field.len() as f64;
    let mean: f64 = field.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var: f64 = field.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    var.sqrt()
}

/// One measured row: an input grain correlation run through the NLM
/// front end at a given search/temporal radius, reporting what comes
/// out the other side.
struct Measurement {
    rho_out_h: f64,
    rho_out_v: f64,
    sigma_ratio: f64,
}

/// Pushes `2 * temporal_radius + 1` independently-seeded noisy frames
/// (via `make_noise(seed)`) through the front end and reads back the
/// output emitted on the final push.
///
/// That final push is the one whose center frame is fully real on both
/// sides (no leading-mirror duplicate anywhere in its window), so the
/// residual it produces reflects genuine averaging across distinct
/// noise realisations, not a partially-duplicated one.
///
/// Compares the output against the flat `clean` reference to get the
/// residual, and the middle pushed frame (the one the output is
/// centered on) against the same reference to get the pre-denoise
/// input noise, so the sigma ratio and the input frame's own
/// correlation are measured on the exact same noise realisation the
/// output derives from.
#[allow(clippy::too_many_arguments)]
fn measure(
    client: &cubecl::prelude::ComputeClient<R>,
    w: u32,
    h: u32,
    base: f32,
    sigma: f32,
    make_noise: impl Fn(u32) -> Vec<f32>,
    search_radius: u32,
    temporal_radius: u32,
) -> Measurement {
    let params = NlmParams {
        temporal_radius,
        search_radius,
        patch_radius: 2,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        hq: Some(HqParams::with_sigma(sigma)),
    };
    let mut denoiser = NlmDenoiser::<R>::new(client, params, w, h);

    let n_push = 2 * temporal_radius + 1;
    let mut center_noisy: Option<Vec<f32>> = None;
    let mut output: Option<Vec<f32>> = None;
    for i in 0..n_push {
        let frame = make_noise(100 + i);
        if i == temporal_radius {
            center_noisy = Some(frame.clone());
        }
        denoiser.push_frame(&frame);
        let result = denoiser.denoise().unwrap();
        if i == n_push - 1 {
            output = result.map(|o| o.as_f32().expect("f32 denoiser").to_vec());
        }
    }

    let output = output.expect("a fully real window must emit on its final push");
    let center_noisy = center_noisy.expect("center frame must have been pushed");

    let clean = vec![base; (w * h) as usize];
    let residual: Vec<f32> = output.iter().zip(clean.iter()).map(|(&o, &c)| o - c).collect();
    let input_noise: Vec<f32> = center_noisy
        .iter()
        .zip(clean.iter())
        .map(|(&o, &c)| o - c)
        .collect();

    Measurement {
        rho_out_h: lag1_horizontal(&residual, w, h),
        rho_out_v: lag1_vertical(&residual, w, h),
        sigma_ratio: std_dev(&residual) / std_dev(&input_noise),
    }
}

/// Measures how the NLM front end's own residual noise correlation
/// compares to the correlation of the grain it was fed, across a
/// sweep of input correlations and window shapes.
///
/// The collaborative second stage that follows NLM assumes white
/// (uncorrelated) noise when it shrinks DCT coefficients. It never sees
/// the input grain, only NLM's residual, so the question a noise-shaping
/// fix needs answered is not "what is the input grain's correlation" but
/// "what correlation does NLM's own output residual actually carry."
///
/// NLM is a local weighted average, and neighbouring output pixels draw
/// on heavily overlapping windows of input pixels, so the mechanism
/// predicts the residual comes out *more* correlated than the input,
/// purely from the window overlap, independent of whether the input
/// noise carried any correlation of its own. A search radius of 0
/// removes that overlap (each pixel then only ever draws on its own
/// position across frames), so it isolates the mechanism: if residual
/// correlation collapses toward the input's own correlation at search
/// radius 0, that confirms the window overlap, not the input grain, is
/// what's inducing it.
///
/// Three input correlations are covered: uncorrelated grain
/// ([`noisy_field_over`]), an intermediate tap
/// ([`correlated_noisy_frame_with_tap`] at `a = 0.125`, predicted
/// `rho ~= 0.316`), and [`correlated_noisy_frame`]'s fixed two thirds.
/// All three are horizontal-only blurs, so the input correlation is
/// anisotropic by construction (`rho_v ~= 0` for every one of them);
/// both axes are measured on the residual regardless, since NLM's own
/// window is isotropic and could induce vertical correlation the input
/// never had.
#[test]
fn nlm_residual_correlation_exceeds_input_correlation_and_tracks_window_overlap() {
    let client = make_client();
    let w = 160;
    let h = 160;
    let base = 0.5f32;
    let sigma_pre = 0.06f32;

    struct Case {
        label: &'static str,
        rho_in_h: f64,
        rho_in_v: f64,
    }
    let cases = [
        Case {
            label: "rho_in=0.00",
            rho_in_h: 0.0,
            rho_in_v: 0.0,
        },
        Case {
            label: "rho_in=0.32",
            rho_in_h: 0.316,
            rho_in_v: 0.0,
        },
        Case {
            label: "rho_in=0.67",
            rho_in_h: 2.0 / 3.0,
            rho_in_v: 0.0,
        },
    ];

    let configs = [(2u32, 0u32), (2u32, 2u32), (0u32, 0u32), (0u32, 2u32)];

    eprintln!(
        "residual correlation sweep (w={w} h={h} sigma_pre={sigma_pre}):\n\
         {:<12} {:>6} {:>6} | {:>6} {:>6} | {:>8} {:>8} | {:>8}",
        "input", "R_s", "R_t", "rho_in_h", "rho_in_v", "rho_out_h", "rho_out_v", "sig_ratio"
    );

    // Collect every row so the assertions below can reason about the
    // whole table at once instead of one measurement in isolation.
    struct Row {
        label: &'static str,
        search_radius: u32,
        temporal_radius: u32,
        rho_in_h: f64,
        m: Measurement,
    }
    let mut rows = Vec::new();

    for case in &cases {
        for &(search_radius, temporal_radius) in &configs {
            let m = match case.label {
                "rho_in=0.00" => {
                    let clean = vec![base; (w * h) as usize];
                    measure(
                        &client,
                        w,
                        h,
                        base,
                        tap_sigma(sigma_pre, 0.0),
                        |seed| noisy_field_over(&clean, w, h, sigma_pre, seed),
                        search_radius,
                        temporal_radius,
                    )
                },
                "rho_in=0.32" => measure(
                    &client,
                    w,
                    h,
                    base,
                    tap_sigma(sigma_pre, 0.125),
                    |seed| correlated_noisy_frame_with_tap(w, h, base, sigma_pre, seed, 0.125),
                    search_radius,
                    temporal_radius,
                ),
                _ => measure(
                    &client,
                    w,
                    h,
                    base,
                    tap_sigma(sigma_pre, 0.25),
                    |seed| correlated_noisy_frame(w, h, base, sigma_pre, seed),
                    search_radius,
                    temporal_radius,
                ),
            };

            eprintln!(
                "{:<12} {:>6} {:>6} | {:>8.4} {:>8.4} | {:>8.4} {:>8.4} | {:>8.4}",
                case.label,
                search_radius,
                temporal_radius,
                case.rho_in_h,
                case.rho_in_v,
                m.rho_out_h,
                m.rho_out_v,
                m.sigma_ratio
            );

            rows.push(Row {
                label: case.label,
                search_radius,
                temporal_radius,
                rho_in_h: case.rho_in_h,
                m,
            });
        }
    }

    // The core mechanism claim: at search_radius=2, the residual's
    // horizontal correlation must exceed the input's, for every input
    // correlation tested including the uncorrelated one. Window overlap
    // alone induces correlation that was never in the input.
    for row in &rows {
        if row.search_radius == 2 {
            assert!(
                row.m.rho_out_h > row.rho_in_h,
                "{} at search_radius=2 temporal_radius={}: residual rho_h={:.4} did not exceed \
                 input rho_h={:.4}, expected window overlap to raise it",
                row.label,
                row.temporal_radius,
                row.m.rho_out_h,
                row.rho_in_h
            );
        }
    }

    // The isolating claim: at search_radius=0, no spatial window overlap
    // exists (each pixel only ever draws on its own position across
    // frames), so residual rho_h should track the input's own
    // correlation closely rather than staying inflated. The tolerance
    // (0.08 absolute) covers the small, real shift temporal averaging at
    // temporal_radius=2 introduces (measured up to ~0.025 here), not
    // measurement noise, since every generator is seeded and
    // deterministic.
    for row in &rows {
        if row.search_radius == 0 {
            assert!(
                (row.m.rho_out_h - row.rho_in_h).abs() < 0.08,
                "{} at search_radius=0 temporal_radius={}: residual rho_h={:.4} should track the \
                 input's rho_h={:.4} within 0.08 once spatial window overlap is removed",
                row.label,
                row.temporal_radius,
                row.m.rho_out_h,
                row.rho_in_h
            );
        }
    }

    // The residual's vertical correlation must stay small everywhere the
    // input's vertical correlation was zero and no spatial window is in
    // play (search_radius=0): there is no mixing mechanism at play there
    // to manufacture it.
    for row in &rows {
        if row.search_radius == 0 {
            assert!(
                row.m.rho_out_v.abs() < 0.05,
                "{} at search_radius=0 temporal_radius={}: residual rho_v={:.4} should stay near \
                 zero, the input never carried vertical correlation and there is no spatial \
                 window to manufacture it",
                row.label,
                row.temporal_radius,
                row.m.rho_out_v
            );
        }
    }

    // The surprising finding this test exists to pin down: NLM's own
    // search window is isotropic (a square), and once it is in play
    // (search_radius=2) it dominates over the input's own anisotropy.
    // Even though every input generator here is horizontal-only
    // (rho_in_v is always 0), the residual's vertical correlation still
    // comes out substantial, well above what the input ever carried,
    // because the window mixes vertical neighbours' noise together too.
    // A separable noise-shaping fix that assumes the same correlation
    // profile on both axes is closer to the truth here than one that
    // trusted the input's own anisotropy.
    for row in &rows {
        if row.search_radius == 2 {
            assert!(
                row.m.rho_out_v > 0.3,
                "{} at search_radius=2 temporal_radius={}: residual rho_v={:.4} did not rise \
                 well above the input's zero vertical correlation, expected the isotropic \
                 window to induce substantial vertical correlation regardless",
                row.label,
                row.temporal_radius,
                row.m.rho_out_v
            );
        }
    }
}

// ---------------------------------------------------------------------
// Follow-up: does the flat-content curve above hold at every search
// radius that actually ships, and does it survive on textured content
// where NLM's weights are far less uniform than on a flat field?
// ---------------------------------------------------------------------

/// Bounded variant of [`lag1_horizontal`], restricted to the half-open
/// rectangle `[x0, x1) x [y0, y1)`. Lets a single frame's correlation be
/// measured separately over sub-regions, a flat tile and a textured
/// tile side by side, rather than only over the whole field at once.
fn lag1_horizontal_rect(field: &[f32], w: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> f64 {
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_ab = 0.0f64;
    let mut sum_aa = 0.0f64;
    let mut sum_bb = 0.0f64;
    let mut n = 0.0f64;
    for y in y0..y1 {
        for x in x0..(x1 - 1) {
            let a = field[(y * w + x) as usize] as f64;
            let b = field[(y * w + x + 1) as usize] as f64;
            sum_a += a;
            sum_b += b;
            sum_ab += a * b;
            sum_aa += a * a;
            sum_bb += b * b;
            n += 1.0;
        }
    }
    let mean_a = sum_a / n;
    let mean_b = sum_b / n;
    let cov = sum_ab / n - mean_a * mean_b;
    let var_a = sum_aa / n - mean_a * mean_a;
    let var_b = sum_bb / n - mean_b * mean_b;
    cov / (var_a.sqrt() * var_b.sqrt())
}

/// Same as [`lag1_horizontal_rect`] but along y.
fn lag1_vertical_rect(field: &[f32], w: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> f64 {
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_ab = 0.0f64;
    let mut sum_aa = 0.0f64;
    let mut sum_bb = 0.0f64;
    let mut n = 0.0f64;
    for y in y0..(y1 - 1) {
        for x in x0..x1 {
            let a = field[(y * w + x) as usize] as f64;
            let b = field[((y + 1) * w + x) as usize] as f64;
            sum_a += a;
            sum_b += b;
            sum_ab += a * b;
            sum_aa += a * a;
            sum_bb += b * b;
            n += 1.0;
        }
    }
    let mean_a = sum_a / n;
    let mean_b = sum_b / n;
    let cov = sum_ab / n - mean_a * mean_b;
    let var_a = sum_aa / n - mean_a * mean_a;
    let var_b = sum_bb / n - mean_b * mean_b;
    cov / (var_a.sqrt() * var_b.sqrt())
}

/// Standard deviation restricted to the half-open rectangle
/// `[x0, x1) x [y0, y1)`.
fn std_dev_rect(field: &[f32], w: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> f64 {
    let n = ((x1 - x0) * (y1 - y0)) as f64;
    let mut sum = 0.0f64;
    for y in y0..y1 {
        for x in x0..x1 {
            sum += field[(y * w + x) as usize] as f64;
        }
    }
    let mean = sum / n;
    let mut var = 0.0f64;
    for y in y0..y1 {
        for x in x0..x1 {
            let v = field[(y * w + x) as usize] as f64;
            var += (v - mean).powi(2);
        }
    }
    (var / n).sqrt()
}

/// Runs the NLM front end once over `2 * temporal_radius + 1` pushed
/// frames of `make_noise(clean, seed)`, seeded from `seed_base`, and
/// returns the output emitted on the final push (the one whose center
/// frame has a fully real window on both sides) together with the noisy
/// frame that produced that center. A generalisation of the sweep
/// above's inlined push loop, parameterised over `clean` and
/// `patch_radius` so it can drive both a flat and a textured reference.
#[allow(clippy::too_many_arguments)]
fn run_front_end(
    client: &cubecl::prelude::ComputeClient<R>,
    w: u32,
    h: u32,
    clean: &[f32],
    sigma: f32,
    make_noise: impl Fn(&[f32], u32) -> Vec<f32>,
    search_radius: u32,
    temporal_radius: u32,
    patch_radius: u32,
    seed_base: u32,
) -> (Vec<f32>, Vec<f32>) {
    let params = NlmParams {
        temporal_radius,
        search_radius,
        patch_radius,
        strength: 1.2,
        self_weight: 1.0,
        channels: ChannelMode::Luma,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::None,
        hq: Some(HqParams::with_sigma(sigma)),
    };
    let mut denoiser = NlmDenoiser::<R>::new(client, params, w, h);

    let n_push = 2 * temporal_radius + 1;
    let mut center_noisy: Option<Vec<f32>> = None;
    let mut output: Option<Vec<f32>> = None;
    for i in 0..n_push {
        let frame = make_noise(clean, seed_base + i);
        if i == temporal_radius {
            center_noisy = Some(frame.clone());
        }
        denoiser.push_frame(&frame);
        let result = denoiser.denoise().unwrap();
        if i == n_push - 1 {
            output = result.map(|o| o.as_f32().expect("f32 denoiser").to_vec());
        }
    }

    (
        output.expect("a fully real window must emit on its final push"),
        center_noisy.expect("center frame must have been pushed"),
    )
}

/// One measured row of residual correlation, common to both the
/// flat-content and the differenced textured-content measurements
/// below.
struct Sample {
    rho_out_h: f64,
    rho_out_v: f64,
    sigma_ratio: f64,
}

/// Measures residual correlation on content whose clean reference is
/// flat, where `output - clean` is already an unbiased noise-only
/// residual: a flat field has no structure for NLM to respond to, so
/// nothing but noise-driven averaging can separate the output from the
/// clean value it started from. Same measurement `measure` above makes,
/// generalised over an arbitrary flat `clean` and `patch_radius`.
#[allow(clippy::too_many_arguments)]
fn measure_flat(
    client: &cubecl::prelude::ComputeClient<R>,
    w: u32,
    h: u32,
    clean: &[f32],
    sigma: f32,
    make_noise: impl Fn(&[f32], u32) -> Vec<f32>,
    search_radius: u32,
    temporal_radius: u32,
    patch_radius: u32,
) -> Sample {
    let (output, center_noisy) = run_front_end(
        client,
        w,
        h,
        clean,
        sigma,
        make_noise,
        search_radius,
        temporal_radius,
        patch_radius,
        100,
    );
    let residual: Vec<f32> = output.iter().zip(clean.iter()).map(|(&o, &c)| o - c).collect();
    let input_noise: Vec<f32> = center_noisy
        .iter()
        .zip(clean.iter())
        .map(|(&o, &c)| o - c)
        .collect();
    Sample {
        rho_out_h: lag1_horizontal(&residual, w, h),
        rho_out_v: lag1_vertical(&residual, w, h),
        sigma_ratio: std_dev(&residual) / std_dev(&input_noise),
    }
}

/// Measures residual correlation on content with real structure, where
/// `output - clean` is not a clean noise residual on its own: NLM's own
/// weighted average responds to the texture's edges and gradients too,
/// and that structural response is a deterministic function of the
/// clean image, not noise. Subtracting a fixed clean reference would
/// fold that deterministic bias into what is supposed to be a
/// noise-only correlation measurement.
///
/// Instead this runs the front end twice over the same clean field,
/// with two independently-seeded noise realisations (`seed_base` 100
/// and 500, far enough apart that even the widest temporal window used
/// anywhere in this file, `2 * 4 + 1 = 9` pushed frames, cannot make the
/// two seed ranges overlap), and differences the two outputs.
///
/// For two identically-distributed, independent copies `A` and `B` of
/// the same random field, `Corr(A - B) == Corr(A)` exactly: the
/// deterministic structural component the two outputs would otherwise
/// share cancels out of the difference, and both the variance and the
/// covariance of `A - B` double relative to `A` alone in the same
/// proportion (independence removes any cross term), so their ratio,
/// the correlation, is unchanged. `A - B`'s own lag-1 correlation is
/// already the answer, no separate bias correction needed. Its standard
/// deviation is `sqrt(2)` times a single realisation's, so `sigma_ratio`
/// divides that back out before comparing against the input noise's own
/// standard deviation.
#[allow(clippy::too_many_arguments)]
fn measure_diff(
    client: &cubecl::prelude::ComputeClient<R>,
    w: u32,
    h: u32,
    clean: &[f32],
    sigma: f32,
    make_noise: impl Fn(&[f32], u32) -> Vec<f32>,
    search_radius: u32,
    temporal_radius: u32,
    patch_radius: u32,
) -> Sample {
    let (output_a, noisy_a) = run_front_end(
        client,
        w,
        h,
        clean,
        sigma,
        &make_noise,
        search_radius,
        temporal_radius,
        patch_radius,
        100,
    );
    let (output_b, _noisy_b) = run_front_end(
        client,
        w,
        h,
        clean,
        sigma,
        &make_noise,
        search_radius,
        temporal_radius,
        patch_radius,
        500,
    );
    let diff: Vec<f32> = output_a
        .iter()
        .zip(output_b.iter())
        .map(|(&a, &b)| a - b)
        .collect();
    let input_noise: Vec<f32> = noisy_a.iter().zip(clean.iter()).map(|(&o, &c)| o - c).collect();
    let sqrt2 = 2.0f64.sqrt();
    Sample {
        rho_out_h: lag1_horizontal(&diff, w, h),
        rho_out_v: lag1_vertical(&diff, w, h),
        sigma_ratio: (std_dev(&diff) / sqrt2) / std_dev(&input_noise),
    }
}

/// Runs [`measure_diff`] but reports correlation separately over two
/// sub-rectangles of the frame instead of the whole field, so a single
/// frame containing both flat and textured regions can be checked for
/// whether its residual correlation actually differs between them.
#[allow(clippy::too_many_arguments)]
fn measure_diff_two_regions(
    client: &cubecl::prelude::ComputeClient<R>,
    w: u32,
    h: u32,
    clean: &[f32],
    sigma: f32,
    make_noise: impl Fn(&[f32], u32) -> Vec<f32>,
    search_radius: u32,
    temporal_radius: u32,
    patch_radius: u32,
    region_a: (u32, u32, u32, u32),
    region_b: (u32, u32, u32, u32),
) -> (Sample, Sample) {
    let (output_a, noisy_a) = run_front_end(
        client,
        w,
        h,
        clean,
        sigma,
        &make_noise,
        search_radius,
        temporal_radius,
        patch_radius,
        100,
    );
    let (output_b, _noisy_b) = run_front_end(
        client,
        w,
        h,
        clean,
        sigma,
        &make_noise,
        search_radius,
        temporal_radius,
        patch_radius,
        500,
    );
    let diff: Vec<f32> = output_a
        .iter()
        .zip(output_b.iter())
        .map(|(&a, &b)| a - b)
        .collect();
    let input_noise: Vec<f32> = noisy_a.iter().zip(clean.iter()).map(|(&o, &c)| o - c).collect();
    let sqrt2 = 2.0f64.sqrt();

    let sample_for = |(x0, y0, x1, y1): (u32, u32, u32, u32)| Sample {
        rho_out_h: lag1_horizontal_rect(&diff, w, x0, y0, x1, y1),
        rho_out_v: lag1_vertical_rect(&diff, w, x0, y0, x1, y1),
        sigma_ratio: (std_dev_rect(&diff, w, x0, y0, x1, y1) / sqrt2)
            / std_dev_rect(&input_noise, w, x0, y0, x1, y1),
    };

    (sample_for(region_a), sample_for(region_b))
}

/// Primary sweep for the search-radius-to-residual-correlation table:
/// every search radius that actually ships (`1..=4`; the presets use 2
/// and 4), measured on both flat and textured content, uncorrelated
/// input noise throughout.
///
/// `task-D1-residual-rho-report.md` flagged its flat-content-only
/// numbers as a likely upper bound: on flat content every candidate in
/// NLM's search window carries an equal weight, but on real footage the
/// Welsch weighting suppresses candidates whose patch doesn't actually
/// match, so less averaging happens and the induced correlation should
/// come out lower. This sweep checks that directly instead of trusting
/// the extrapolation, by running the identical measurement on
/// [`make_textured_frame`] alongside the flat field, at every shipping
/// search radius.
///
/// Textured content cannot reuse the flat measurement's `output - clean`
/// trick (see [`measure_diff`]'s doc comment for why), so it is measured
/// by differencing two independently-seeded denoised outputs of the same
/// clean field instead; [`Sample::sigma_ratio`] on that side is already
/// rescaled back to a single-realisation footing.
#[test]
fn nlm_residual_correlation_search_radius_sweep_flat_vs_textured() {
    let client = make_client();
    let w = 160;
    let h = 160;
    let base = 0.5f32;
    let sigma_pre = 0.06f32;
    let patch_radius = 2;

    let flat_clean = vec![base; (w * h) as usize];
    let textured_clean = make_textured_frame(w, h);

    struct Row {
        search_radius: u32,
        flat: Sample,
        textured: Sample,
    }
    let mut rows = Vec::new();

    eprintln!(
        "search radius sweep, flat vs textured (w={w} h={h} sigma_pre={sigma_pre} \
         patch_radius={patch_radius}, uncorrelated input):\n\
         {:>3} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8}",
        "R_s", "flat_h", "flat_v", "flat_sig", "tex_h", "tex_v", "tex_sig"
    );

    for &search_radius in &[1u32, 2, 3, 4] {
        let flat = measure_flat(
            &client,
            w,
            h,
            &flat_clean,
            sigma_pre,
            |clean, seed| noisy_field_over(clean, w, h, sigma_pre, seed),
            search_radius,
            0,
            patch_radius,
        );
        let textured = measure_diff(
            &client,
            w,
            h,
            &textured_clean,
            sigma_pre,
            |clean, seed| noisy_field_over(clean, w, h, sigma_pre, seed),
            search_radius,
            0,
            patch_radius,
        );

        eprintln!(
            "{:>3} | {:>8.4} {:>8.4} {:>8.4} | {:>8.4} {:>8.4} {:>8.4}",
            search_radius,
            flat.rho_out_h,
            flat.rho_out_v,
            flat.sigma_ratio,
            textured.rho_out_h,
            textured.rho_out_v,
            textured.sigma_ratio
        );

        rows.push(Row {
            search_radius,
            flat,
            textured,
        });
    }

    // Validity guard: every configuration must show real smoothing
    // before its correlation number means anything. A sigma_ratio near
    // 1.0 means the front end barely denoised at all, the failure mode
    // task-D1 hit with an uncalibrated strength.
    for row in &rows {
        assert!(
            row.flat.sigma_ratio < 0.9,
            "search_radius={}: flat sigma_ratio={:.4} too close to 1.0, no real smoothing \
             happened, this configuration's correlation number is not valid data",
            row.search_radius,
            row.flat.sigma_ratio
        );
        assert!(
            row.textured.sigma_ratio < 0.9,
            "search_radius={}: textured sigma_ratio={:.4} too close to 1.0, no real smoothing \
             happened, this configuration's correlation number is not valid data",
            row.search_radius,
            row.textured.sigma_ratio
        );
    }

    // The core regression guard: window overlap induces real
    // correlation on flat content at every shipping search radius, not
    // just the radius-2 case task-D1 covered.
    for row in &rows {
        assert!(
            row.flat.rho_out_h > 0.3,
            "search_radius={}: flat rho_out_h={:.4} did not show substantial window-induced \
             correlation",
            row.search_radius,
            row.flat.rho_out_h
        );
    }

    // The headline finding this sweep exists to check: contrary to
    // task-D1's "probably an upper bound" caveat, the sinusoidal texture
    // in make_textured_frame tracks the flat field closely at every
    // radius (measured max gap 0.0234 at search_radius=4). The 0.08
    // tolerance leaves comfortable margin above that while still
    // catching a real divergence between the two content types.
    for row in &rows {
        assert!(
            (row.flat.rho_out_h - row.textured.rho_out_h).abs() < 0.08,
            "search_radius={}: flat rho_out_h={:.4} and textured rho_out_h={:.4} diverge by more \
             than the measured tolerance, flat and textured no longer agree",
            row.search_radius,
            row.flat.rho_out_h,
            row.textured.rho_out_h
        );
    }

    // Residual correlation should not fall as the window widens: a
    // bigger search radius only ever adds more overlapping candidates,
    // never fewer, so the induced correlation should rise or plateau,
    // not reverse.
    for pair in rows.windows(2) {
        assert!(
            pair[1].flat.rho_out_h >= pair[0].flat.rho_out_h - 1e-6,
            "flat rho_out_h dropped from search_radius={} ({:.4}) to search_radius={} ({:.4})",
            pair[0].search_radius,
            pair[0].flat.rho_out_h,
            pair[1].search_radius,
            pair[1].flat.rho_out_h
        );
    }
}

/// The table this file's primary sweep produces at the shipped default
/// `patch_radius`, across every search radius the presets can select
/// (`0..=4`).
///
/// The primary sweep above measured at `patch_radius=2`, following
/// `task-D1-residual-rho-report.md`'s convention. The library default is
/// `patch_radius=4`
/// ([`crate::nlmeans::NlmParams::patch_radius`]'s doc comment), and this
/// file's own secondary-checks test found that shift is not small: at
/// `search_radius=2`, `patch_radius=4` measured `rho_out_h=0.7770`
/// against `patch_radius=2`'s `0.7006`, a 0.077 gap. A table built at
/// the wrong patch radius does not describe the configuration that
/// actually runs, so this is the sweep to bake into the codebase, not
/// the `patch_radius=2` one above.
///
/// `search_radius=0` is included here (the earlier sweep started at 1),
/// since a complete table needs every radius the presets can reach, and
/// `search_radius=0` is the isolating case: no spatial window overlap,
/// so residual correlation should collapse to near zero regardless of
/// `patch_radius` (a larger patch still only compares a single fixed
/// candidate against itself when there is no window to search).
#[test]
fn nlm_residual_correlation_search_radius_sweep_at_shipped_patch_radius() {
    let client = make_client();
    let w = 160;
    let h = 160;
    let base = 0.5f32;
    let sigma_pre = 0.06f32;
    let patch_radius = 4;

    let flat_clean = vec![base; (w * h) as usize];
    let textured_clean = make_textured_frame(w, h);

    struct Row {
        search_radius: u32,
        flat: Sample,
        textured: Sample,
    }
    let mut rows = Vec::new();

    eprintln!(
        "search radius sweep at shipped patch_radius={patch_radius}, flat vs textured \
         (w={w} h={h} sigma_pre={sigma_pre}, uncorrelated input):\n\
         {:>3} | {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8}",
        "R_s", "flat_h", "flat_v", "flat_sig", "tex_h", "tex_v", "tex_sig"
    );

    for &search_radius in &[0u32, 1, 2, 3, 4] {
        let flat = measure_flat(
            &client,
            w,
            h,
            &flat_clean,
            sigma_pre,
            |clean, seed| noisy_field_over(clean, w, h, sigma_pre, seed),
            search_radius,
            0,
            patch_radius,
        );
        let textured = measure_diff(
            &client,
            w,
            h,
            &textured_clean,
            sigma_pre,
            |clean, seed| noisy_field_over(clean, w, h, sigma_pre, seed),
            search_radius,
            0,
            patch_radius,
        );

        eprintln!(
            "{:>3} | {:>8.4} {:>8.4} {:>8.4} | {:>8.4} {:>8.4} {:>8.4}",
            search_radius,
            flat.rho_out_h,
            flat.rho_out_v,
            flat.sigma_ratio,
            textured.rho_out_h,
            textured.rho_out_v,
            textured.sigma_ratio
        );

        rows.push(Row {
            search_radius,
            flat,
            textured,
        });
    }

    // Validity guard. search_radius=0 is exempt: with no spatial window
    // at all, HQ's calibrated weighting has essentially nothing to
    // average over, so almost no smoothing happens and sigma_ratio sits
    // near 1.0 by construction, not because the measurement is broken.
    // That is exactly D1's isolating case, kept here for a complete
    // table rather than trusted as a valid correlation measurement.
    for row in &rows {
        if row.search_radius == 0 {
            continue;
        }
        assert!(
            row.flat.sigma_ratio < 0.9,
            "search_radius={}: flat sigma_ratio={:.4} too close to 1.0, no real smoothing \
             happened, this configuration's correlation number is not valid data",
            row.search_radius,
            row.flat.sigma_ratio
        );
        assert!(
            row.textured.sigma_ratio < 0.9,
            "search_radius={}: textured sigma_ratio={:.4} too close to 1.0, no real smoothing \
             happened, this configuration's correlation number is not valid data",
            row.search_radius,
            row.textured.sigma_ratio
        );
    }

    // The isolating case: search_radius=0 has no window to overlap, so
    // residual correlation should stay near the uncorrelated input's own
    // ~0, on both content types, mirroring D1's search_radius=0 finding.
    for row in &rows {
        if row.search_radius == 0 {
            assert!(
                row.flat.rho_out_h.abs() < 0.1,
                "search_radius=0: flat rho_out_h={:.4} should stay near zero, there is no \
                 spatial window to manufacture correlation",
                row.flat.rho_out_h
            );
        }
    }

    // The core regression guard at every radius with a real window: flat
    // content shows substantial window-induced correlation.
    for row in &rows {
        if row.search_radius >= 1 {
            assert!(
                row.flat.rho_out_h > 0.3,
                "search_radius={}: flat rho_out_h={:.4} did not show substantial window-induced \
                 correlation",
                row.search_radius,
                row.flat.rho_out_h
            );
        }
    }

    // Every row here should sit at or above its patch_radius=2
    // counterpart from the primary sweep: a larger patch radius makes
    // the per-candidate weight estimate less noisy (more pixels
    // averaged into the distance), which lets the window's own
    // near-uniform weighting come through more cleanly rather than
    // being scattered by measurement noise, so it raises the induced
    // correlation rather than lowering it (the secondary-checks test
    // above already pins this down at search_radius=2 alone; this
    // extends the same direction check across the radii newly measured
    // here).
    for row in &rows {
        if row.search_radius == 0 {
            continue;
        }
        assert!(
            row.flat.rho_out_h > 0.6,
            "search_radius={}: patch_radius=4 flat rho_out_h={:.4} unexpectedly low, expected it \
             to sit above the patch_radius=2 sweep's own values at this radius",
            row.search_radius,
            row.flat.rho_out_h
        );
    }

    // Residual correlation should not fall as the window widens.
    for pair in rows.windows(2) {
        assert!(
            pair[1].flat.rho_out_h >= pair[0].flat.rho_out_h - 1e-6,
            "flat rho_out_h dropped from search_radius={} ({:.4}) to search_radius={} ({:.4})",
            pair[0].search_radius,
            pair[0].flat.rho_out_h,
            pair[1].search_radius,
            pair[1].flat.rho_out_h
        );
    }
}

/// Secondary checks, fewer points than the primary sweep, on what else
/// the residual correlation depends on besides `search_radius`. All
/// three vary one parameter at a time away from a shared baseline
/// (`search_radius=2`, `patch_radius=2`, `temporal_radius=0`,
/// uncorrelated input, flat content), reusing that baseline's own row
/// from the sweep above rather than re-measuring it.
#[test]
fn nlm_residual_correlation_patch_radius_temporal_radius_and_input_correlation() {
    let client = make_client();
    let w = 160;
    let h = 160;
    let base = 0.5f32;
    let sigma_pre = 0.06f32;
    let flat_clean = vec![base; (w * h) as usize];
    let search_radius = 2;

    let baseline = measure_flat(
        &client,
        w,
        h,
        &flat_clean,
        sigma_pre,
        |clean, seed| noisy_field_over(clean, w, h, sigma_pre, seed),
        search_radius,
        0,
        2,
    );

    let patch4 = measure_flat(
        &client,
        w,
        h,
        &flat_clean,
        sigma_pre,
        |clean, seed| noisy_field_over(clean, w, h, sigma_pre, seed),
        search_radius,
        0,
        4,
    );

    let temporal2 = measure_flat(
        &client,
        w,
        h,
        &flat_clean,
        sigma_pre,
        |clean, seed| noisy_field_over(clean, w, h, sigma_pre, seed),
        search_radius,
        2,
        2,
    );

    let rho_in_h = 2.0 / 3.0;
    let corr_input = measure_flat(
        &client,
        w,
        h,
        &flat_clean,
        sigma_pre,
        |_clean, seed| correlated_noisy_frame(w, h, base, sigma_pre, seed),
        search_radius,
        0,
        2,
    );

    eprintln!(
        "secondary checks at search_radius=2, flat content (w={w} h={h} sigma_pre={sigma_pre}):\n\
         {:<32} {:>8} {:>8} {:>8}",
        "config", "rho_h", "rho_v", "sig_ratio"
    );
    for (label, s) in [
        ("baseline patch_radius=2 R_t=0 rho_in=0", &baseline),
        ("patch_radius=4", &patch4),
        ("temporal_radius=2", &temporal2),
        ("input rho_h=0.67", &corr_input),
    ] {
        eprintln!(
            "{:<32} {:>8.4} {:>8.4} {:>8.4}",
            label, s.rho_out_h, s.rho_out_v, s.sigma_ratio
        );
    }

    // Validity guard on every configuration measured here.
    for (label, s) in [
        ("baseline", &baseline),
        ("patch_radius=4", &patch4),
        ("temporal_radius=2", &temporal2),
        ("input rho_h=0.67", &corr_input),
    ] {
        assert!(
            s.sigma_ratio < 0.9,
            "{label}: sigma_ratio={:.4} too close to 1.0, no real smoothing happened, this \
             configuration's correlation number is not valid data",
            s.sigma_ratio
        );
    }

    // task-D1 measured temporal averaging diluting the window-induced
    // correlation slightly (~0.02-0.05 at search_radius=2). Confirm the
    // direction holds with patch_radius=2 held fixed here too.
    assert!(
        temporal2.rho_out_h < baseline.rho_out_h,
        "temporal_radius=2 rho_out_h={:.4} should be below temporal_radius=0's {:.4}, temporal \
         averaging is expected to dilute the spatial window's contribution",
        temporal2.rho_out_h,
        baseline.rho_out_h
    );

    // Correlated input should raise the residual's correlation further
    // above the uncorrelated baseline (task-D1's core finding, that the
    // window imposes a floor and the input's own correlation lifts it
    // further within the remaining headroom below 1), and it must not
    // exceed 1 either.
    assert!(
        corr_input.rho_out_h > baseline.rho_out_h,
        "correlated input (rho_in_h={rho_in_h:.4}) rho_out_h={:.4} should exceed the uncorrelated \
         baseline's {:.4}",
        corr_input.rho_out_h,
        baseline.rho_out_h
    );
    assert!(
        corr_input.rho_out_h < 1.0,
        "correlated input rho_out_h={:.4} must stay below 1.0",
        corr_input.rho_out_h
    );
}

/// A single clean frame with a flat region on the left and
/// [`make_textured_frame`]'s own sine formula on the right, split at
/// `split_x`. Lets one NLM run be checked for whether its residual
/// correlation actually differs between the two regions within the
/// same frame, rather than only across separately-generated frames.
fn make_flat_and_textured_frame(w: u32, h: u32, split_x: u32) -> Vec<f32> {
    let mut frame = vec![0.0f32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let v = if x < split_x {
                0.5
            } else {
                let fx = x as f32 / w as f32;
                let fy = y as f32 / h as f32;
                let raw = 0.5
                    + 0.2 * (fx * 8.0 * std::f32::consts::PI).sin() * (fy * 6.0 * std::f32::consts::PI).cos()
                    + 0.1 * (fx * 20.0 * std::f32::consts::PI).sin();
                raw.clamp(0.05, 0.95)
            };
            frame[(y * w + x) as usize] = v;
        }
    }
    frame
}

/// Whether residual correlation differs between a flat region and a
/// textured region within a *single* frame, rather than across two
/// separately-generated frames.
///
/// The sweep above already runs flat and textured content as two
/// entirely separate frames, which cannot rule out that some other
/// difference between the two setups (frame content statistics beyond
/// just flat-vs-textured, boundary handling, whatever) is responsible
/// for how closely they tracked. Building one frame with both region
/// types side by side and measuring each region's own correlation
/// removes that confound: if a spatially-varying second-stage fix ever
/// turns out to be necessary, it is because the *same* frame carries
/// two different residual-correlation profiles at once, not because two
/// different test frames happened to differ.
///
/// The two analysis tiles sit 15 pixels from the seam and from every
/// frame edge, comfortably outside `patch_radius + search_radius = 4`'s
/// reach, so neither NLM's own output near a tile boundary nor the
/// correlation measurement within a tile can be contaminated by the
/// other region.
#[test]
fn nlm_residual_correlation_within_a_single_frame_flat_vs_textured_regions() {
    let client = make_client();
    let w = 200;
    let h = 120;
    let sigma_pre = 0.06f32;
    let search_radius = 2;
    let patch_radius = 2;
    let split_x = 100;

    let clean = make_flat_and_textured_frame(w, h, split_x);
    let flat_region = (15u32, 15u32, 85u32, 105u32);
    let textured_region = (115u32, 15u32, 185u32, 105u32);

    let (flat, textured) = measure_diff_two_regions(
        &client,
        w,
        h,
        &clean,
        sigma_pre,
        |clean, seed| noisy_field_over(clean, w, h, sigma_pre, seed),
        search_radius,
        0,
        patch_radius,
        flat_region,
        textured_region,
    );

    eprintln!(
        "within-frame flat vs textured region (w={w} h={h} sigma_pre={sigma_pre} \
         search_radius={search_radius} patch_radius={patch_radius}):\n\
         {:<10} {:>8} {:>8} {:>8}",
        "region", "rho_h", "rho_v", "sig_ratio"
    );
    for (label, s) in [("flat", &flat), ("textured", &textured)] {
        eprintln!(
            "{:<10} {:>8.4} {:>8.4} {:>8.4}",
            label, s.rho_out_h, s.rho_out_v, s.sigma_ratio
        );
    }
    eprintln!(
        "within-frame gap: rho_h {:.4}, rho_v {:.4}",
        (flat.rho_out_h - textured.rho_out_h).abs(),
        (flat.rho_out_v - textured.rho_out_v).abs()
    );

    // Validity guard on both regions independently.
    assert!(
        flat.sigma_ratio < 0.9,
        "flat region sigma_ratio={:.4} too close to 1.0, no real smoothing happened",
        flat.sigma_ratio
    );
    assert!(
        textured.sigma_ratio < 0.9,
        "textured region sigma_ratio={:.4} too close to 1.0, no real smoothing happened",
        textured.sigma_ratio
    );

    // The finding this test exists to pin down: within one frame, the
    // two regions' residual correlations stay close, matching the
    // separate-frames sweep above rather than contradicting it. The
    // 0.1 tolerance is generous relative to the 0.08 used for the
    // separate-frame comparison, since each tile here covers far fewer
    // pixels (70x90) than a full 160x160 frame, so sampling noise in
    // the correlation estimate itself is larger.
    assert!(
        (flat.rho_out_h - textured.rho_out_h).abs() < 0.1,
        "within one frame, flat region rho_out_h={:.4} and textured region rho_out_h={:.4} \
         diverge by more than the tolerance; a single per-frame correlation profile would not \
         be structurally sound if this fails",
        flat.rho_out_h,
        textured.rho_out_h
    );
}

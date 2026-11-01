use super::helpers::*;
use crate::bench_api::HostIo;
use crate::nlmeans::*;
use crate::noise::tests::fold_reference::relative_gap;
use crate::noise::tests::frames::MAX_RELATIVE;
use crate::noise::{NOISE_CHANNELS, NoiseSnapshot, sigma_from_abs_sum};

fn snapshot(denoiser: &NlmDenoiser) -> NoiseSnapshot {
    denoiser.noise_snapshot().expect("the noise state reads back")
}

/// The first channel of a chain, `None` before the chain's first update.
fn first_channel(chain: Option<[f32; NOISE_CHANNELS as usize]>) -> Option<f32> {
    chain.map(|sigmas| sigmas[0])
}

/// HQ auto-estimation params at k=0.
///
/// The temporal chain needs `temporal_radius >= 1`, so at k=0 only the two spatial statistics (frame
/// mean and block p25) can tell the median and low chains apart.
fn auto_params_k0() -> NlmParams {
    NlmParams {
        temporal_radius: 0,
        search_radius: 3,
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
        }),
    }
}

/// A frame whose top half carries `sigma_a` noise and whose bottom half carries `sigma_b`, both at
/// the same base level.
fn block_heterogeneous_frame(width: u32, height: u32, base: f32, sigma_a: f32, sigma_b: f32) -> Vec<f32> {
    let top = make_noisy_gaussian_frame(width, height, 1, base, &[sigma_a]);
    let bottom = make_noisy_gaussian_frame(width, height, 1, base, &[sigma_b]);
    let row_len = width as usize;
    let half = (height / 2) as usize;
    let mut frame = top;
    for row in half..height as usize {
        let start = row * row_len;
        frame[start..start + row_len].copy_from_slice(&bottom[start..start + row_len]);
    }

    frame
}

/// Uniform noise puts the block p25 close to the frame mean, so the low chain stays close to the
/// median chain and `noise_offset` close to the median-based offset.
#[test]
fn uniform_noise_low_chain_matches_median_chain() {
    let client = make_client();
    let width = 256;
    let height = 256;
    let sigma = 8.0 / 255.0;
    let frame = make_noisy_gaussian_frame(width, height, 1, 0.5, &[sigma]);

    let params = auto_params_k0();
    let mut denoiser = NlmDenoiser::new(&client, params, width, height);
    denoiser.push_frame(&frame);
    // The flush runs the pass the noise look-ahead holds back, and leaves its fold in place.
    denoiser.flush(|_| {}).unwrap();

    let folded = snapshot(&denoiser);
    let median = first_channel(folded.median).expect("the flush pass folds");
    let low = first_channel(folded.low).expect("the flush pass folds");
    let noise_offset = folded.derived.noise_offset;

    let relative_error = (low - median).abs() / median;
    assert!(
        relative_error <= 0.15,
        "uniform noise: low chain {low} should sit close to median chain {median} (rel err {relative_error:.3})"
    );

    let expected_offset = denoiser.params.noise_offset_with(Some(&[median]));
    let offset_relative_error = (noise_offset - expected_offset).abs() / expected_offset;
    assert!(
        offset_relative_error <= 0.3,
        "uniform noise: noise_offset {noise_offset} should stay close to the pre-split median-based offset \
         {expected_offset} (rel err {offset_relative_error:.3})"
    );
}

/// Half the frame sits at a low sigma and half at a much higher one, so the low chain reads below
/// the median chain.
#[test]
fn split_noise_offset_tracks_low_chain_strength_tracks_median() {
    let client = make_client();
    let width = 256;
    let height = 256;
    let sigma_a = 2.0 / 255.0;
    let sigma_b = 20.0 / 255.0;
    let frame = block_heterogeneous_frame(width, height, 0.5, sigma_a, sigma_b);

    let params = auto_params_k0();
    let mut denoiser = NlmDenoiser::new(&client, params, width, height);
    denoiser.push_frame(&frame);
    // The flush runs the pass the noise look-ahead holds back, and leaves its fold in place.
    denoiser.flush(|_| {}).unwrap();

    let folded = snapshot(&denoiser);
    let median = first_channel(folded.median).expect("the flush pass folds");
    let low = first_channel(folded.low).expect("the flush pass folds");
    let noise_offset = folded.derived.noise_offset;
    let h2_inv_norm = folded.derived.h2_inv_norm;

    assert!(
        low < median,
        "block-heterogeneous noise: low chain {low} should read below median chain {median}"
    );

    let expected_median_offset = denoiser.params.noise_offset_with(Some(&[median]));
    assert!(
        noise_offset < expected_median_offset,
        "noise_offset {noise_offset} should track the low chain strictly below the median-based offset {expected_median_offset}"
    );

    // The fold scales a host coefficient by the folded sigma, so it matches the host formula to
    // within rounding rather than bit for bit.
    let expected_h2 = denoiser.params.h2_inv_norm_with(Some(median));
    let h2_gap = relative_gap(h2_inv_norm, expected_h2);
    assert!(
        h2_gap <= MAX_RELATIVE,
        "h2_inv_norm {h2_inv_norm} must keep tracking the median chain's {expected_h2}, {h2_gap:e} apart"
    );
}

/// Each frame's reading must survive in the history from its push until every window that
/// holds it has folded.
///
/// At `temporal_radius = 2` the leading edge is primed with copies of the low-sigma frame 0, and two
/// high-sigma frames then fill the window. A shared scratch buffer would hold the latest high-sigma
/// readings by the first fold, so frame 0's record would read high.
#[test]
fn the_history_keeps_each_frame_reading_until_it_folds() {
    let client = make_client();
    let width = 128;
    let height = 128;
    let sigma_low = 2.0 / 255.0;
    let sigma_high = 30.0 / 255.0;

    let params = NlmParams {
        temporal_radius: 2,
        ..auto_params_k0()
    };
    let mut denoiser = NlmDenoiser::new(&client, params, width, height);

    let frame_low = make_noisy_gaussian_frame(width, height, 1, 0.5, &[sigma_low]);
    let frame_high_1 = make_noisy_gaussian_frame(width, height, 1, 0.5, &[sigma_high]);
    let frame_high_2 = make_noisy_gaussian_frame(width, height, 1, 0.5, &[sigma_high]);

    for frame in [&frame_low, &frame_high_1, &frame_high_2] {
        denoiser.push_frame(frame);
        assert!(
            denoiser.denoise().unwrap().is_none(),
            "the look-ahead holds the frame"
        );
    }

    // The first output's pass folds the window holding all three frames.
    while denoiser.flush_step_gpu().unwrap().is_none() {}

    let noise = denoiser.measured_noise().expect("the noise is measured");
    let recorded = noise.stream_frame_count();
    assert_eq!(recorded, 3, "the priming copies record nothing");

    let sigmas: Vec<f32> = (0..recorded)
        .map(|frame| {
            let (_, sums) = noise.read_history_frame(frame);
            sigma_from_abs_sum(sums[0], width, height)
        })
        .collect();

    assert!(
        sigmas[0] < sigma_high * 0.15,
        "frame 0's record must still hold the low-sigma reading, got {sigmas:?}"
    );
    assert!(
        sigmas[1] > sigma_high * 0.5 && sigmas[2] > sigma_high * 0.5,
        "frames 1 and 2 hold the high-sigma readings, got {sigmas:?}"
    );
}

/// A flat mid-grey frame whose top `split_rows` rows carry `sigma_top` noise and the rest `sigma_bottom`.
fn row_split_noisy_frame(
    width: u32,
    height: u32,
    split_rows: u32,
    sigma_top: f32,
    sigma_bottom: f32,
    seed: u32,
) -> Vec<f32> {
    let clean = vec![0.5f32; (width * height) as usize];
    let top = noisy_field_over(&clean, width, height, sigma_top, seed);
    let bottom = noisy_field_over(&clean, width, height, sigma_bottom, seed);

    let split = (split_rows * width) as usize;
    let mut frame = bottom;
    frame[..split].copy_from_slice(&top[..split]);

    frame
}

/// Three eighths of the blocks carry a low sigma and the rest a higher one, so the lower quartile of
/// block sigmas lands on the low group while the median lands on the high group.
#[test]
fn temporal_only_chain_reads_the_median_block_sigma() {
    let client = make_client();
    let width = 128;
    let height = 128;
    let sigma_low = 3.0 / 255.0;
    let sigma_high = 9.0 / 255.0;

    let params = NlmParams {
        temporal_radius: 2,
        ..auto_params_k0()
    };
    let mut denoiser = NlmDenoiser::new(&client, params, width, height);

    for seed in 0..8 {
        let frame = row_split_noisy_frame(width, height, 48, sigma_low, sigma_high, 300 + seed);
        denoiser.push_frame(&frame);
        let _ = denoiser.denoise().unwrap();
    }

    denoiser.flush(|_| {}).unwrap();

    let folded = snapshot(&denoiser);
    let temporal_only =
        first_channel(folded.temporal_only).expect("a temporal sample should have folded by now");

    let relative_error = (temporal_only - sigma_high).abs() / sigma_high;
    assert!(
        relative_error <= 0.15,
        "temporal-only chain {temporal_only} should read the median block sigma near {sigma_high} \
         (rel err {relative_error:.3})"
    );
}

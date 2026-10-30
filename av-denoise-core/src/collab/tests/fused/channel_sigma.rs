use super::{Aggregated, Setup, chroma_cross_frame_setup, run_fused};

/// A distinct sigma per channel, each far enough from the others to move the threshold decisions.
const CHANNEL_SIGMAS: [f32; 2] = [0.02, 0.06];

/// Channel `channel`'s accumulator values, every region of the ring.
fn channel_accum(got: &Aggregated, channel: usize, channels: usize) -> Vec<i32> {
    got.accum
        .iter()
        .skip(channel)
        .step_by(channels)
        .copied()
        .collect()
}

/// A copy of `setup` with one sigma per channel.
fn with_sigmas(setup: &Setup, sigmas: Vec<f32>) -> Setup {
    Setup {
        channel_sigmas: Some(sigmas),
        ring: setup.ring.clone(),
        mv_field: setup.mv_field.clone(),
        confidence: setup.confidence.clone(),
        neighbour_slots: setup.neighbour_slots.clone(),
        strength_map: setup.strength_map.clone(),
        ..*setup
    }
}

/// Each channel past the first must filter exactly as a run that gives every channel past the first
/// its sigma, and differently from a run at channel 0's sigma.
fn assert_each_channel_reads_its_own_sigma(setup: Setup) {
    let channels = setup.channels() as usize;
    let mixed_sigmas = CHANNEL_SIGMAS[..channels].to_vec();
    let mixed_setup = with_sigmas(&setup, mixed_sigmas);
    let mixed = run_fused(&mixed_setup);

    let uniform_sigmas = vec![CHANNEL_SIGMAS[0]; channels];
    let uniform_setup = with_sigmas(&setup, uniform_sigmas);
    let uniform = run_fused(&uniform_setup);

    // Channel 0 sets the group weight, so it and the weights match the uniform run.
    let mixed_first = channel_accum(&mixed, 0, channels);
    let uniform_first = channel_accum(&uniform, 0, channels);
    assert_eq!(mixed_first, uniform_first, "channel 0");
    assert_eq!(mixed.wsum, uniform.wsum, "weight sums");

    let later_channels = CHANNEL_SIGMAS.iter().enumerate().take(channels).skip(1);
    for (channel, &channel_sigma) in later_channels {
        let mut reference_sigmas = vec![channel_sigma; channels];
        reference_sigmas[0] = CHANNEL_SIGMAS[0];
        let reference_setup = with_sigmas(&setup, reference_sigmas);
        let reference = run_fused(&reference_setup);

        let mixed_channel = channel_accum(&mixed, channel, channels);
        let reference_channel = channel_accum(&reference, channel, channels);
        let uniform_channel = channel_accum(&uniform, channel, channels);

        assert_eq!(
            mixed_channel, reference_channel,
            "channel {channel} against its own sigma"
        );
        assert_ne!(
            mixed_channel, uniform_channel,
            "channel {channel} against channel 0's sigma"
        );
    }
}

#[test]
fn each_chroma_channel_thresholds_with_its_own_sigma() {
    let setup = chroma_cross_frame_setup(64, 64, 2);
    assert_each_channel_reads_its_own_sigma(setup);
}

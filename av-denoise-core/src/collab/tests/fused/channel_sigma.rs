use super::{Aggregated, Setup, chroma_cross_frame_setup, run_fused, yuv_cross_frame_setup};

/// A distinct sigma per channel, each far enough from the others to move the threshold decisions.
const CHANNEL_SIGMAS: [f32; 3] = [0.02, 0.06, 0.12];

/// Channel `channel`'s accumulator values, every region of the ring.
fn channel_accum(got: &Aggregated, channel: usize, stored: usize) -> Vec<i32> {
    got.accum.iter().skip(channel).step_by(stored).copied().collect()
}

/// A copy of `setup` with one sigma per live channel.
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
    let live = setup.channel_mode.count() as usize;
    let stored = setup.stored_channels() as usize;
    let mixed_sigmas = CHANNEL_SIGMAS[..live].to_vec();
    let mixed_setup = with_sigmas(&setup, mixed_sigmas);
    let mixed = run_fused(&mixed_setup);

    let uniform_sigmas = vec![CHANNEL_SIGMAS[0]; live];
    let uniform_setup = with_sigmas(&setup, uniform_sigmas);
    let uniform = run_fused(&uniform_setup);

    // Channel 0 sets the group weight, so it and the weights match the uniform run.
    let mixed_first = channel_accum(&mixed, 0, stored);
    let uniform_first = channel_accum(&uniform, 0, stored);
    assert_eq!(mixed_first, uniform_first, "channel 0");
    assert_eq!(mixed.wsum, uniform.wsum, "weight sums");

    let later_channels = CHANNEL_SIGMAS.iter().enumerate().take(live).skip(1);
    for (channel, &channel_sigma) in later_channels {
        let mut reference_sigmas = vec![channel_sigma; live];
        reference_sigmas[0] = CHANNEL_SIGMAS[0];
        let reference_setup = with_sigmas(&setup, reference_sigmas);
        let reference = run_fused(&reference_setup);

        let mixed_channel = channel_accum(&mixed, channel, stored);
        let reference_channel = channel_accum(&reference, channel, stored);
        let uniform_channel = channel_accum(&uniform, channel, stored);

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

#[test]
fn each_yuv_channel_thresholds_with_its_own_sigma() {
    let setup = yuv_cross_frame_setup(64, 64, 2);
    assert_each_channel_reads_its_own_sigma(setup);
}

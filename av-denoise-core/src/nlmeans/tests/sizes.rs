use cubecl::server::Handle;

use super::helpers::{R, make_client};
use crate::nlmeans::denoiser::{BufferSize, front_buffer_sizes};
use crate::nlmeans::*;

// Odd dimensions leave ragged block grids and unaligned slots, so every padding rule is exercised.
const WIDTH: u32 = 101;
const HEIGHT: u32 = 67;

fn allocated_elements(handle: Option<&Handle>) -> Option<u64> {
    handle.map(|handle| handle.size_in_used() / 4)
}

/// The front end's computed count for `name`, which must appear exactly once.
fn computed(buffers: &[BufferSize], name: &str) -> Option<u64> {
    let mut matching = buffers.iter().filter(|(buffer, _)| *buffer == name);
    let (_, elements) = matching.next()?;
    assert!(matching.next().is_none(), "{name} listed twice");

    *elements
}

fn assert_sizes_match_allocations(params: NlmParams) {
    let client = make_client();
    let sizes = front_buffer_sizes(&client, &params, WIDTH, HEIGHT);
    let denoiser = NlmDenoiser::<R>::new(&client, params, WIDTH, HEIGHT);

    let pairs = [
        ("noise partials ring", denoiser.noise_partials.as_ref()),
        ("temporal stats ring", denoiser.temporal_stats_buf.as_ref()),
        ("motion field", denoiser.mv_field_buf.as_ref()),
        ("motion pyramid ring", denoiser.pyramid_input.as_ref()),
        ("motion pair ring", denoiser.pair_ring_buf.as_ref()),
        ("confidence", denoiser.confidence_buf.as_ref()),
        ("confidence pyramid ring", denoiser.confidence_pyramid.as_ref()),
        (
            "confidence vector scratch",
            denoiser.confidence_mv_scratch.as_ref(),
        ),
    ];
    for (name, handle) in pairs {
        let allocated = allocated_elements(handle);
        let expected = computed(&sizes.buffers, name);
        assert_eq!(expected, allocated, "{name}");
    }

    for (name, _) in &sizes.buffers {
        let known = pairs.iter().any(|(pair_name, _)| pair_name == name);
        assert!(known, "{name} has no allocation to compare against");
    }
}

fn hq_params(temporal_radius: u32, motion_compensation: MotionCompensationMode) -> NlmParams {
    NlmParams {
        temporal_radius,
        motion_compensation,
        hq: Some(HqParams::default()),
        ..NlmParams::default()
    }
}

#[test]
fn chained_motion_sizes_match_the_allocations() {
    let motion = MotionCompensationMode::Mvtools {
        blksize: 16,
        overlap: 8,
        search_radius: 4,
        pyramid_levels: 3,
        estimation: MotionEstimation::chained_default(),
    };
    let params = hq_params(3, motion);

    assert_sizes_match_allocations(params);
}

#[test]
fn direct_motion_sizes_match_the_allocations() {
    let motion = MotionCompensationMode::mvtools_default();
    let params = hq_params(1, motion);

    assert_sizes_match_allocations(params);
}

#[test]
fn confidence_only_sizes_match_the_allocations() {
    let params = hq_params(2, MotionCompensationMode::None);

    assert_sizes_match_allocations(params);
}

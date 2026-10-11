use super::engine::{nl4d_outputs, noisy_frames};
use super::equivalence::assert_candidates_match_baseline;
use super::fixtures::{CollabSpec, collab_site};
use super::probe::small_device_limits;
use super::runtime::test_client;
use crate::Nl4dOptions;
use crate::nlmeans::ChannelMode;
use crate::tune::gpu::read_f32;
use crate::tune::sites::{COLLAB_FUSED, CollabCandidate};
use crate::tune::{DeviceLimits, KernelPlan, SiteCandidate, VariantSite, fit_candidates};

fn luma_spec() -> CollabSpec {
    CollabSpec {
        width: 64,
        height: 48,
        channels: ChannelMode::Luma,
        radius: 2,
    }
}

#[test]
fn collab_candidate_names_round_trip() {
    for &candidate in CollabCandidate::ALL {
        let name = candidate.name();
        let parsed = CollabCandidate::from_name(name);
        assert_eq!(parsed, Some(candidate), "{name}");
    }

    assert_eq!(CollabCandidate::baseline(), CollabCandidate::Groups8);
}

/// Each 8-lane group owns a 65-float transpose region, and that is all the kernel allocates.
#[test]
fn collab_shared_bytes_count_one_transpose_region_per_group() {
    let client = test_client();
    let site = collab_site(&client, &luma_spec());

    assert_eq!(site.shared_bytes(CollabCandidate::Groups8), 8 * 65 * 4);
    assert_eq!(site.shape(CollabCandidate::Groups8).units(), 64);
}

#[test]
fn collab_keys_name_every_compiled_value() {
    let client = test_client();
    let site = collab_site(&client, &luma_spec());
    let warp = u8::from(site.shape.warp_uniform);

    let key = site.key();
    let expected = format!(
        "collab_fused|w64|h64|ch=1|stored=1|radius=2|grid=4|refine=2|blksize=16|step=8|kmax=8|spatial=9|\
         pooled=0|warp={warp}|f16=0"
    );

    assert_eq!(key.as_str(), expected);
}

#[test]
fn collab_baseline_fits_planes_narrower_than_eight_lanes() {
    let client = test_client();
    let site = collab_site(&client, &luma_spec());
    let mut limits = small_device_limits();
    limits.plane_size_min = 4;

    let fitted = fit_candidates(&site, &limits).expect("the baseline skips the plane check");

    assert_eq!(site.min_plane_size(CollabCandidate::Groups8), 8);
    assert_eq!(fitted, vec![CollabCandidate::Groups8]);
}

/// Walks the sizes, radii and warp switches with one channel mode.
fn assert_every_group_count_matches_for(channels: ChannelMode, sizes: &[(u32, u32)]) {
    let client = test_client();
    let limits = DeviceLimits::from_client(&client);

    for &(width, height) in sizes {
        for radius in [0u32, 2] {
            let spec = CollabSpec {
                width,
                height,
                channels,
                radius,
            };
            let site = collab_site(&client, &spec);

            for warp_uniform in [false, true] {
                let mut walked = site.clone();
                walked.shape.warp_uniform = warp_uniform;
                let label = format!("{spec:?} warp_uniform={warp_uniform}");

                let fitted = fit_candidates(&walked, &limits).expect("the baseline fits");
                assert!(fitted.contains(&CollabCandidate::Groups4), "{label}");
                assert!(fitted.contains(&CollabCandidate::Groups16), "{label}");

                let baseline = assert_candidates_match_baseline(&walked, &label);
                let weights =
                    read_f32(&client, &baseline.group_weight, baseline.refs).expect("weights read back");
                let has_signal = weights.iter().any(|&weight| weight > 0.0);
                assert!(has_signal, "{label}: the baseline retired no group");
            }
        }
    }
}

#[test]
fn every_group_count_matches_the_baseline_bit_for_bit_on_luma() {
    assert_every_group_count_matches_for(ChannelMode::Luma, &[(70, 45), (129, 33)]);
}

// Every walk compiles its own kernels, so this mode walks fewer sizes.
#[test]
fn every_group_count_matches_the_baseline_bit_for_bit_on_chroma() {
    assert_every_group_count_matches_for(ChannelMode::Chroma, &[(70, 45)]);
}

// Every walk compiles its own kernels, so this mode walks fewer sizes.
#[test]
fn every_group_count_matches_the_baseline_bit_for_bit_on_yuv() {
    assert_every_group_count_matches_for(ChannelMode::Yuv, &[(70, 45)]);
}

#[test]
fn sixteen_groups_still_fit_a_32kb_device() {
    let client = test_client();
    let site = collab_site(&client, &luma_spec());
    let limits = small_device_limits();

    let fitted = fit_candidates(&site, &limits).expect("the baseline fits");

    assert_eq!(site.shared_bytes(CollabCandidate::Groups16), 16 * 65 * 4);
    assert!(fitted.contains(&CollabCandidate::Groups16));
}

#[test]
fn forcing_any_group_count_leaves_nl4d_output_unchanged() {
    let options = Nl4dOptions {
        temporal_radius: 1,
        ..Nl4dOptions::default()
    };

    for channels in [ChannelMode::Luma, ChannelMode::Chroma, ChannelMode::Yuv] {
        let planes = channels.count() as usize;
        let (width, height) = (72u32, 48u32);
        let frames = noisy_frames(width, height, planes, 7);
        let baseline_plan = KernelPlan::baseline();
        let baseline = nl4d_outputs(options, channels, width, height, &frames, baseline_plan);

        for &candidate in CollabCandidate::ALL {
            let plan = KernelPlan::baseline().with_site(COLLAB_FUSED, candidate.name());
            let forced = nl4d_outputs(options, channels, width, height, &frames, plan);
            assert_eq!(forced, baseline, "{channels:?} {}", candidate.name());
        }
    }
}

use super::{
    Aggregated,
    Setup,
    chroma_cross_frame_setup,
    cross_frame_setup,
    run_fused_variant,
    unique_frame,
    yuv_cross_frame_setup,
};
use crate::collab::kernels::fused::{CollabVariant, Walk};
use crate::collab::tests::helpers::{cel_frame, constant_frame, letterboxed_frame, noisy_flat_field};
use crate::nl4d::nl4d_pool_ratio;
use crate::nlmeans::ChannelMode;
use crate::tests::golden_hash::{RADV_R9700, fnv1a_words, live_device, runs_on_recording_device};

/// One corpus entry's output, hashed buffer by buffer.
pub(super) struct Golden {
    pub(super) label: &'static str,
    pub(super) accum: u64,
    pub(super) wsum: u64,
    pub(super) group_weight: u64,
}

/// The recorded hashes, one per corpus entry and walk.
pub(super) const GOLDEN: &[Golden] = &[
    Golden {
        label: "unique/clipped",
        accum: 0xea4cd795be0e584d,
        wsum: 0x98cfd3d8120a35e3,
        group_weight: 0x5d1780954c8ba665,
    },
    Golden {
        label: "unique/uniform",
        accum: 0xea4cd795be0e584d,
        wsum: 0x98cfd3d8120a35e3,
        group_weight: 0x5d1780954c8ba665,
    },
    Golden {
        label: "ragged/clipped",
        accum: 0x7cd04982ff7fbf70,
        wsum: 0x54533c95bbb9daa4,
        group_weight: 0x6022f0fa8b9bc910,
    },
    Golden {
        label: "ragged/uniform",
        accum: 0x7cd04982ff7fbf70,
        wsum: 0x54533c95bbb9daa4,
        group_weight: 0x6022f0fa8b9bc910,
    },
    Golden {
        label: "short_search/clipped",
        accum: 0xacbf9512268be2f0,
        wsum: 0x06a062db358b9b47,
        group_weight: 0x432321e41eec2d67,
    },
    Golden {
        label: "short_search/uniform",
        accum: 0xacbf9512268be2f0,
        wsum: 0x06a062db358b9b47,
        group_weight: 0x432321e41eec2d67,
    },
    Golden {
        label: "noise/clipped",
        accum: 0x1ecd6940bb8f499a,
        wsum: 0x3f8b6c46c7ef3ca4,
        group_weight: 0xd1b05732716a75f5,
    },
    Golden {
        label: "noise/uniform",
        accum: 0x1ecd6940bb8f499a,
        wsum: 0x3f8b6c46c7ef3ca4,
        group_weight: 0xd1b05732716a75f5,
    },
    Golden {
        label: "cross_frame_r1/clipped",
        accum: 0xc9862047d34dea87,
        wsum: 0xcda56cc8e4b9d690,
        group_weight: 0xf8f80c766f1ce8aa,
    },
    Golden {
        label: "cross_frame_r1/uniform",
        accum: 0xc9862047d34dea87,
        wsum: 0xcda56cc8e4b9d690,
        group_weight: 0xf8f80c766f1ce8aa,
    },
    Golden {
        label: "cross_frame_r2/clipped",
        accum: 0x6eecec54eea7e8c9,
        wsum: 0x96f806752867b089,
        group_weight: 0x827f4e8c4002494e,
    },
    Golden {
        label: "cross_frame_r2/uniform",
        accum: 0x6eecec54eea7e8c9,
        wsum: 0x96f806752867b089,
        group_weight: 0x827f4e8c4002494e,
    },
    Golden {
        label: "all_gated/clipped",
        accum: 0x6a65329c69d6b115,
        wsum: 0x36df6825e6a8da5b,
        group_weight: 0xd84f5e932e1bdf38,
    },
    Golden {
        label: "all_gated/uniform",
        accum: 0x6a65329c69d6b115,
        wsum: 0x36df6825e6a8da5b,
        group_weight: 0xd84f5e932e1bdf38,
    },
    Golden {
        label: "f16_luma/clipped",
        accum: 0x6318627b221ebc20,
        wsum: 0xc16971af0567f0cd,
        group_weight: 0x827f4e8c4002494e,
    },
    Golden {
        label: "f16_luma/uniform",
        accum: 0x6318627b221ebc20,
        wsum: 0xc16971af0567f0cd,
        group_weight: 0x827f4e8c4002494e,
    },
    Golden {
        label: "chroma/clipped",
        accum: 0xd5d450aa588d6dfe,
        wsum: 0x6ad7951e9cf34bea,
        group_weight: 0x879a09a8188c5b11,
    },
    Golden {
        label: "chroma/uniform",
        accum: 0xd5d450aa588d6dfe,
        wsum: 0x6ad7951e9cf34bea,
        group_weight: 0x879a09a8188c5b11,
    },
    Golden {
        label: "f16_chroma/clipped",
        accum: 0x9a0235ff58477c2b,
        wsum: 0x890a2d8ae6329d5f,
        group_weight: 0xa27c494c154e204f,
    },
    Golden {
        label: "f16_chroma/uniform",
        accum: 0x9a0235ff58477c2b,
        wsum: 0x890a2d8ae6329d5f,
        group_weight: 0xa27c494c154e204f,
    },
    Golden {
        label: "yuv/clipped",
        accum: 0xd21c75a4f8ff8db3,
        wsum: 0x4f1e54286abbd738,
        group_weight: 0xc5413272e8cd48db,
    },
    Golden {
        label: "yuv/uniform",
        accum: 0xd21c75a4f8ff8db3,
        wsum: 0x4f1e54286abbd738,
        group_weight: 0xc5413272e8cd48db,
    },
    Golden {
        label: "pooled_luma/clipped",
        accum: 0x459d9cf7512aa657,
        wsum: 0x8ee5e744eb3a9caf,
        group_weight: 0x072ddf565ca67135,
    },
    Golden {
        label: "pooled_luma/uniform",
        accum: 0x459d9cf7512aa657,
        wsum: 0x8ee5e744eb3a9caf,
        group_weight: 0x072ddf565ca67135,
    },
    Golden {
        label: "pooled_chroma/clipped",
        accum: 0x3ab151e888a94e7c,
        wsum: 0x6c26e57567b95e40,
        group_weight: 0x89c06e2d9aed3b78,
    },
    Golden {
        label: "pooled_chroma/uniform",
        accum: 0x3ab151e888a94e7c,
        wsum: 0x6c26e57567b95e40,
        group_weight: 0x89c06e2d9aed3b78,
    },
    Golden {
        label: "constant_spatial/clipped",
        accum: 0xbcca41a74776af62,
        wsum: 0x74c324b51448a6e8,
        group_weight: 0xe555bbe29b7831b2,
    },
    Golden {
        label: "constant_spatial/uniform",
        accum: 0xbcca41a74776af62,
        wsum: 0x74c324b51448a6e8,
        group_weight: 0xe555bbe29b7831b2,
    },
    Golden {
        label: "constant_cross_frame/clipped",
        accum: 0x702a4136b433d081,
        wsum: 0x2a9b5c3ae52d4162,
        group_weight: 0xe555bbe29b7831b2,
    },
    Golden {
        label: "constant_cross_frame/uniform",
        accum: 0x702a4136b433d081,
        wsum: 0x2a9b5c3ae52d4162,
        group_weight: 0xe555bbe29b7831b2,
    },
    Golden {
        label: "letterbox/clipped",
        accum: 0x5dd341bbdb6dadc0,
        wsum: 0xbba9786249c5dcec,
        group_weight: 0x36e6f53b32a35f15,
    },
    Golden {
        label: "letterbox/uniform",
        accum: 0x5dd341bbdb6dadc0,
        wsum: 0xbba9786249c5dcec,
        group_weight: 0x36e6f53b32a35f15,
    },
    Golden {
        label: "cel/clipped",
        accum: 0x7fe5fe7da5781727,
        wsum: 0x2ce5bf2dcfdab060,
        group_weight: 0x9e176c887aee1afb,
    },
    Golden {
        label: "cel/uniform",
        accum: 0x7fe5fe7da5781727,
        wsum: 0x2ce5bf2dcfdab060,
        group_weight: 0x9e176c887aee1afb,
    },
];

/// A ring of `2 * radius + 1` copies of `frame`, with the cross-frame motion and full confidence.
fn repeated_ring(frame: &[f32], width: u32, height: u32, radius: u32) -> Setup {
    let mut setup = cross_frame_setup(width, height, radius);
    let frames = (2 * radius + 1) as usize;
    setup.ring = frame.repeat(frames);
    setup.confidence.fill(1.0);

    setup
}

/// Every configuration whose output the exact changes must keep.
///
/// The flat entries tie everywhere, so they decide whether a changed search breaks ties the same way.
pub(super) fn collab_corpus() -> Vec<(&'static str, Setup)> {
    let mut corpus = Vec::new();

    let unique = unique_frame(128, 96);
    let unique_setup = Setup::spatial_only(unique, 128, 96);
    corpus.push(("unique", unique_setup));

    let ragged = unique_frame(104, 96);
    let ragged_setup = Setup::spatial_only(ragged, 104, 96);
    corpus.push(("ragged", ragged_setup));

    let short = unique_frame(64, 64);
    let mut short_search = Setup::spatial_only(short, 64, 64);
    short_search.spatial_radius = 1;
    corpus.push(("short_search", short_search));

    let noise = noisy_flat_field(64, 64, 0.5, 0.05);
    let noise_setup = Setup::spatial_only(noise, 64, 64);
    corpus.push(("noise", noise_setup));

    let cross_frame_r1 = cross_frame_setup(64, 64, 1);
    corpus.push(("cross_frame_r1", cross_frame_r1));

    let cross_frame_r2 = cross_frame_setup(64, 64, 2);
    corpus.push(("cross_frame_r2", cross_frame_r2));

    let mut all_gated = cross_frame_setup(64, 64, 2);
    all_gated.c_min = 2.0;
    corpus.push(("all_gated", all_gated));

    let mut f16_luma = cross_frame_setup(64, 64, 2);
    f16_luma.f16_search = true;
    corpus.push(("f16_luma", f16_luma));

    let chroma = chroma_cross_frame_setup(64, 64, 2);
    corpus.push(("chroma", chroma));

    let mut f16_chroma = chroma_cross_frame_setup(64, 64, 2);
    f16_chroma.f16_search = true;
    corpus.push(("f16_chroma", f16_chroma));

    let yuv = yuv_cross_frame_setup(64, 64, 2);
    corpus.push(("yuv", yuv));

    let mut pooled_luma = cross_frame_setup(64, 64, 2);
    let luma_ratio = nl4d_pool_ratio(ChannelMode::Luma);
    pooled_luma.pooled = Some(luma_ratio);
    corpus.push(("pooled_luma", pooled_luma));

    let mut pooled_chroma = chroma_cross_frame_setup(64, 64, 2);
    let chroma_ratio = nl4d_pool_ratio(ChannelMode::Chroma);
    pooled_chroma.pooled = Some(chroma_ratio);
    corpus.push(("pooled_chroma", pooled_chroma));

    let constant = constant_frame(64, 64, 0.5);
    let constant_cross_frame = repeated_ring(&constant, 64, 64, 2);
    let constant_spatial = Setup::spatial_only(constant, 64, 64);
    corpus.push(("constant_spatial", constant_spatial));
    corpus.push(("constant_cross_frame", constant_cross_frame));

    let letterbox = letterboxed_frame(96, 64, 16);
    let letterbox_setup = repeated_ring(&letterbox, 96, 64, 2);
    corpus.push(("letterbox", letterbox_setup));

    let cel = cel_frame(96, 64, 12);
    let mut cel_setup = repeated_ring(&cel, 96, 64, 2);
    cel_setup.spatial_radius = 9;
    corpus.push(("cel", cel_setup));

    corpus
}

fn hash_outputs(got: &Aggregated) -> (u64, u64, u64) {
    let accum_words = got.accum.iter().map(|&value| value as u32);
    let wsum_words = got.wsum.iter().map(|&value| value as u32);
    let weight_words = got.group_weight.iter().map(|value| value.to_bits());

    let accum = fnv1a_words(accum_words);
    let wsum = fnv1a_words(wsum_words);
    let group_weight = fnv1a_words(weight_words);

    (accum, wsum, group_weight)
}

fn walk_label(uniform: bool) -> &'static str {
    if uniform { "uniform" } else { "clipped" }
}

/// The baseline variant with the clipped or the uniform walk.
fn walk_variant(uniform: bool) -> CollabVariant {
    let walk = if uniform { Walk::Uniform } else { Walk::Clipped };

    CollabVariant {
        walk,
        ..CollabVariant::BASELINE
    }
}

#[test]
fn every_collab_corpus_entry_matches_its_golden_hash() {
    let corpus = collab_corpus();
    assert_eq!(
        GOLDEN.len(),
        corpus.len() * 2,
        "every corpus entry and walk needs one golden entry"
    );

    if !runs_on_recording_device(&RADV_R9700) {
        return;
    }

    for (name, setup) in corpus {
        for uniform in [false, true] {
            let walk = walk_label(uniform);
            let label = format!("{name}/{walk}");
            let variant = walk_variant(uniform);
            let got = run_fused_variant(&setup, variant);
            let (accum, wsum, group_weight) = hash_outputs(&got);
            let golden = GOLDEN
                .iter()
                .find(|entry| entry.label == label)
                .unwrap_or_else(|| panic!("{label} has no golden entry"));

            assert_eq!(accum, golden.accum, "{label}: accum moved");
            assert_eq!(wsum, golden.wsum, "{label}: wsum moved");
            assert_eq!(group_weight, golden.group_weight, "{label}: group weights moved");
        }
    }
}

/// Prints [GOLDEN] from the current kernels.
#[test]
#[ignore = "records golden hashes, run by hand before the first exact change"]
fn record_collab_golden_hashes() {
    let live = live_device();
    println!("// Recorded on {live:?}");
    println!("pub(super) const GOLDEN: &[Golden] = &[");

    for (name, setup) in collab_corpus() {
        for uniform in [false, true] {
            let variant = walk_variant(uniform);
            let got = run_fused_variant(&setup, variant);
            let (accum, wsum, group_weight) = hash_outputs(&got);
            let walk = walk_label(uniform);
            println!(
                "    Golden {{ label: \"{name}/{walk}\", accum: {accum:#018x}, wsum: {wsum:#018x}, group_weight: {group_weight:#018x} }},"
            );
        }
    }

    println!("];");
}

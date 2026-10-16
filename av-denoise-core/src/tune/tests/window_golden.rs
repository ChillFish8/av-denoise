use super::fixtures::{WindowSpec, window_site};
use super::runtime::test_client;
use crate::nlmeans::ChannelMode;
use crate::tests::golden_hash::{RADV_R9700, fnv1a_words, live_device, runs_on_recording_device};
use crate::tune::VariantSite;
use crate::tune::gpu::read_f32;

/// One corpus entry's output, hashed buffer by buffer.
pub(super) struct WindowGolden {
    pub(super) label: &'static str,
    pub(super) accum: u64,
    pub(super) weight_sum: u64,
    pub(super) max_weight: u64,
}

/// The recorded hashes, one per corpus entry.
pub(super) const WINDOW_GOLDEN: &[WindowGolden] = &[
    WindowGolden {
        label: "128x64/luma/pair=false/ref=false/conf=false",
        accum: 0x373da6d819beca9c,
        weight_sum: 0xe713542c142ac34b,
        max_weight: 0x4f888eb115747659,
    },
    WindowGolden {
        label: "128x64/luma/pair=false/ref=true/conf=false",
        accum: 0x1eb0a0f76dfe78e1,
        weight_sum: 0x4c6c9bd2eaf546a9,
        max_weight: 0x0694e4cf6edb2e20,
    },
    WindowGolden {
        label: "128x64/luma/pair=true/ref=false/conf=false",
        accum: 0x70c4dad7a916daf7,
        weight_sum: 0x449af043c56a1a7c,
        max_weight: 0x9ebb7b31cb0dcff2,
    },
    WindowGolden {
        label: "128x64/luma/pair=true/ref=false/conf=true",
        accum: 0xc82e35220993de88,
        weight_sum: 0x809fbbbdb0d22d29,
        max_weight: 0x0246f766aaa1206d,
    },
    WindowGolden {
        label: "128x64/luma/pair=true/ref=true/conf=false",
        accum: 0xa1fdf02c75be3c04,
        weight_sum: 0x298eb87b99ef25fa,
        max_weight: 0x32d8fe84511c40e9,
    },
    WindowGolden {
        label: "128x64/luma/pair=true/ref=true/conf=true",
        accum: 0x8a640a9d4c3c6fa6,
        weight_sum: 0x46126b5359bb0f03,
        max_weight: 0x950c6913d5e242a1,
    },
    WindowGolden {
        label: "128x64/yuv/pair=false/ref=false/conf=false",
        accum: 0x6aecb3903d651c3d,
        weight_sum: 0xf4ddb44a0a2c1836,
        max_weight: 0xc0b86912f14670df,
    },
    WindowGolden {
        label: "128x64/yuv/pair=false/ref=true/conf=false",
        accum: 0x2aa2a9aeabb3ecf8,
        weight_sum: 0x25c5cb00f30451f5,
        max_weight: 0xc2177b0c1d061689,
    },
    WindowGolden {
        label: "128x64/yuv/pair=true/ref=false/conf=false",
        accum: 0xb4608441e5fc7213,
        weight_sum: 0x0f8f4f02485701ca,
        max_weight: 0x4d1a5a0c53b32727,
    },
    WindowGolden {
        label: "128x64/yuv/pair=true/ref=false/conf=true",
        accum: 0xb05b93daea411d64,
        weight_sum: 0x214d1800abc2389a,
        max_weight: 0xf93fbbe24f84846f,
    },
    WindowGolden {
        label: "128x64/yuv/pair=true/ref=true/conf=false",
        accum: 0xfe9dd66e1b9d2ed6,
        weight_sum: 0x2261bab375a6a1b1,
        max_weight: 0xbd02a03063d652e2,
    },
    WindowGolden {
        label: "128x64/yuv/pair=true/ref=true/conf=true",
        accum: 0x23abea596d694a25,
        weight_sum: 0xf22e388e10bea82c,
        max_weight: 0x8d260285ad9c013f,
    },
    WindowGolden {
        label: "100x45/luma/pair=false/ref=false/conf=false",
        accum: 0x158a83b5449674c0,
        weight_sum: 0xd7f84942280e7695,
        max_weight: 0xa2f6ec24b46e0ba3,
    },
    WindowGolden {
        label: "100x45/luma/pair=false/ref=true/conf=false",
        accum: 0x6609423b27adec85,
        weight_sum: 0xba4bd6b1886575c5,
        max_weight: 0x56f06e35d78854bc,
    },
    WindowGolden {
        label: "100x45/luma/pair=true/ref=false/conf=false",
        accum: 0x7318d3e64815c4f6,
        weight_sum: 0x8e05190945d95152,
        max_weight: 0xd04f647d6ce58480,
    },
    WindowGolden {
        label: "100x45/luma/pair=true/ref=false/conf=true",
        accum: 0x040489edb218c378,
        weight_sum: 0x770f95a96db875b7,
        max_weight: 0x5d0dde3e59f7ebad,
    },
    WindowGolden {
        label: "100x45/luma/pair=true/ref=true/conf=false",
        accum: 0x13aed2f3f52d2ae9,
        weight_sum: 0x81ed34801b3c2f9a,
        max_weight: 0x3f241da72c4d8f07,
    },
    WindowGolden {
        label: "100x45/luma/pair=true/ref=true/conf=true",
        accum: 0x008088ab8e0fbfdb,
        weight_sum: 0x9d2ba18c8460afd8,
        max_weight: 0xf562324739c50e83,
    },
    WindowGolden {
        label: "100x45/yuv/pair=false/ref=false/conf=false",
        accum: 0x81ba8b01e7b76026,
        weight_sum: 0xd5a1580fdf804436,
        max_weight: 0xf2c7ea09e0f36085,
    },
    WindowGolden {
        label: "100x45/yuv/pair=false/ref=true/conf=false",
        accum: 0x529f4e8a2cfbc31b,
        weight_sum: 0xcbc56fcd028ca69d,
        max_weight: 0x19d866fa02733708,
    },
    WindowGolden {
        label: "100x45/yuv/pair=true/ref=false/conf=false",
        accum: 0x585c5496c4a23495,
        weight_sum: 0xb432a99bee9a5d2f,
        max_weight: 0x46e99a05d78c9654,
    },
    WindowGolden {
        label: "100x45/yuv/pair=true/ref=false/conf=true",
        accum: 0xfb3e33bbcbf0a2ff,
        weight_sum: 0xcad9361157f47615,
        max_weight: 0x829d78b9c1eb71b8,
    },
    WindowGolden {
        label: "100x45/yuv/pair=true/ref=true/conf=false",
        accum: 0xe41deb25ef95b913,
        weight_sum: 0x302f649ea5ddf83f,
        max_weight: 0x02f2948ad371a01b,
    },
    WindowGolden {
        label: "100x45/yuv/pair=true/ref=true/conf=true",
        accum: 0xf836031417a2f3af,
        weight_sum: 0x4e418491403a31b0,
        max_weight: 0x5e51eafc1209017b,
    },
];

/// Single and pair windows, with and without a reference and confidence, on a frame with interior
/// blocks and on a ragged one.
pub(super) fn window_corpus() -> Vec<(String, WindowSpec)> {
    let mut corpus = Vec::new();
    let sizes = [(128u32, 64u32), (100, 45)];
    let channel_modes = [(ChannelMode::Luma, "luma"), (ChannelMode::Yuv, "yuv")];

    for (width, height) in sizes {
        for (channels, channel_name) in channel_modes {
            for pair in [false, true] {
                for reference in [false, true] {
                    for confidence in [false, true] {
                        if confidence && !pair {
                            continue;
                        }

                        let base = if pair {
                            WindowSpec::pair(width, height, channels)
                        } else {
                            WindowSpec::single(width, height, channels)
                        };
                        let spec = WindowSpec {
                            reference,
                            confidence,
                            ..base
                        };
                        let label = format!(
                            "{width}x{height}/{channel_name}/pair={pair}/ref={reference}/conf={confidence}"
                        );
                        corpus.push((label, spec));
                    }
                }
            }
        }
    }

    corpus
}

fn hash_window(spec: &WindowSpec) -> (u64, u64, u64) {
    let client = test_client();
    let site = window_site(&client, spec);
    let outputs = site.scratch_outputs();
    let baseline = site.baseline();
    site.launch(baseline, &outputs);

    let accum = read_f32(&client, &outputs.accum, outputs.accum_len).expect("accum reads back");
    let weight_sum = read_f32(&client, &outputs.weight_sum, outputs.pixels).expect("weight sum reads back");
    let max_weight = read_f32(&client, &outputs.max_weight, outputs.pixels).expect("max weight reads back");

    let accum_bits = accum.iter().map(|value| value.to_bits());
    let weight_sum_bits = weight_sum.iter().map(|value| value.to_bits());
    let max_weight_bits = max_weight.iter().map(|value| value.to_bits());

    let accum_hash = fnv1a_words(accum_bits);
    let weight_sum_hash = fnv1a_words(weight_sum_bits);
    let max_weight_hash = fnv1a_words(max_weight_bits);

    (accum_hash, weight_sum_hash, max_weight_hash)
}

#[test]
fn every_window_corpus_entry_matches_its_golden_hash() {
    let corpus = window_corpus();
    assert_eq!(
        WINDOW_GOLDEN.len(),
        corpus.len(),
        "every corpus entry needs one golden entry"
    );

    if !runs_on_recording_device(&RADV_R9700) {
        return;
    }

    for (label, spec) in corpus {
        let (accum, weight_sum, max_weight) = hash_window(&spec);
        let golden = WINDOW_GOLDEN
            .iter()
            .find(|entry| entry.label == label)
            .unwrap_or_else(|| panic!("{label} has no golden entry"));

        assert_eq!(accum, golden.accum, "{label}: accum moved");
        assert_eq!(weight_sum, golden.weight_sum, "{label}: weight sum moved");
        assert_eq!(max_weight, golden.max_weight, "{label}: max weight moved");
    }
}

/// Prints [WINDOW_GOLDEN] from the current kernels.
#[test]
#[ignore = "records golden hashes, run by hand before the first exact change"]
fn record_window_golden_hashes() {
    let live = live_device();
    println!("// Recorded on {live:?}");
    println!("pub(super) const WINDOW_GOLDEN: &[WindowGolden] = &[");

    for (label, spec) in window_corpus() {
        let (accum, weight_sum, max_weight) = hash_window(&spec);
        println!(
            "    WindowGolden {{ label: \"{label}\", accum: {accum:#018x}, weight_sum: {weight_sum:#018x}, max_weight: {max_weight:#018x} }},"
        );
    }

    println!("];");
}

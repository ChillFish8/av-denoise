mod kernels;

use av_denoise_core::bench_api::Device;
use av_denoise_core::bench_api::tune::{COLLAB_CANDIDATES, REGULARISE_CANDIDATES, WINDOW_CANDIDATES};
use clap::Parser;
use cubecl::prelude::*;
use kernels::accumulate::AccumulateBench;
use kernels::bilateral::BilateralBench;
use kernels::cast_f16::CastF16Bench;
use kernels::collab_aggregate::{CollabNormaliseBench, CollabZeroAccumBench};
use kernels::collab_fused::CollabFusedBench;
use kernels::copy::CopyBench;
use kernels::dist_2d_weight::DistWeightBench;
use kernels::dist_2d_weight_ref::DistWeightRefBench;
use kernels::distance::DistanceBench;
use kernels::distance_pair::DistancePairBench;
use kernels::distance_pair_ref::DistancePairRefBench;
use kernels::distance_ref::DistanceRefBench;
use kernels::egress::{EgressBench, EgressFormat};
use kernels::finish::FinishBench;
use kernels::fused_window::{
    FusedPairWindowBench,
    FusedPairWindowRefBench,
    FusedSingleWindowBench,
    FusedSingleWindowRefBench,
};
use kernels::grain::{GRAIN_SIZES, GrainMeasureBench, GrainReducePartialsBench, GrainSaveVectorsBench};
use kernels::horizontal_sum::HSumBench;
use kernels::horizontal_sum_pair::HSumPairBench;
use kernels::ingest::{IngestBench, IngestFormat};
use kernels::mc_block_match_coarse::BlockMatchCoarseBench;
use kernels::mc_block_match_fine::BlockMatchFineBench;
use kernels::mc_chain_compose::ChainComposeBench;
use kernels::mc_confidence::McConfidenceBench;
use kernels::mc_downscale::DownscaleBench;
use kernels::mc_warp::WarpBench;
use kernels::mv_regularise::MvRegulariseBench;
use kernels::noise_partial::NoisePartialBench;
use kernels::temporal_noise_stats::TemporalNoiseStatsBench;
use kernels::vertical_weight::VWeightBench;
use kernels::vweight_pair_accumulate::VWeightPairAccBench;
use kernels::zero::ZeroBench;
use kernels::{CHANNELS, print_header, run};

fn run_all<R: Runtime>(backend: &str, device: &R::Device) {
    let client = R::client(device);

    println!();
    println!("--- {backend} ---");
    print_header();

    for &(channels, channel_name) in CHANNELS {
        run(CopyBench {
            client: client.clone(),
            channels,
            channel_name,
        });
    }

    let cast_cases = [(1920, 1080, 1, "luma"), (960, 540, 2, "chroma")];
    for (width, height, channels, channel_name) in cast_cases {
        run(CastF16Bench {
            client: client.clone(),
            width,
            height,
            channels,
            channel_name,
        });
    }

    let ingest_cases = [
        (1920, 1080, 1, "luma", IngestFormat::U8),
        (960, 540, 2, "chroma", IngestFormat::U16Ten),
        (1920, 1080, 3, "yuv", IngestFormat::U16Ten),
        (1920, 1080, 1, "luma", IngestFormat::F32),
    ];
    for (width, height, channels, channel_name, format) in ingest_cases {
        run(IngestBench {
            client: client.clone(),
            width,
            height,
            channels,
            channel_name,
            format,
        });
    }

    let egress_cases = [
        (1920, 1080, 1, "luma", EgressFormat::U8),
        (960, 540, 2, "chroma", EgressFormat::U16Ten),
        (1920, 1080, 3, "yuv", EgressFormat::U16Ten),
        (1920, 1080, 1, "luma", EgressFormat::F32),
    ];
    for (width, height, channels, channel_name, format) in egress_cases {
        run(EgressBench {
            client: client.clone(),
            width,
            height,
            channels,
            channel_name,
            format,
        });
    }

    for &(channels, channel_name) in CHANNELS {
        run(ZeroBench {
            client: client.clone(),
            channels,
            channel_name,
        });
    }

    for &(channels, channel_name) in CHANNELS {
        run(DistWeightBench {
            client: client.clone(),
            channels,
            channel_name,
        });
    }

    for &(channels, channel_name) in CHANNELS {
        run(DistWeightRefBench {
            client: client.clone(),
            channels,
            channel_name,
        });
    }

    for &(channels, channel_name) in CHANNELS {
        for candidate in 0..WINDOW_CANDIDATES.len() {
            run(FusedSingleWindowBench {
                client: client.clone(),
                channels,
                channel_name,
                candidate,
            });
        }
    }

    for &(channels, channel_name) in CHANNELS {
        for candidate in 0..WINDOW_CANDIDATES.len() {
            run(FusedPairWindowBench {
                client: client.clone(),
                channels,
                channel_name,
                candidate,
            });
        }
    }

    for &(channels, channel_name) in CHANNELS {
        run(FusedSingleWindowRefBench {
            client: client.clone(),
            channels,
            channel_name,
        });
    }

    for &(channels, channel_name) in CHANNELS {
        run(FusedPairWindowRefBench {
            client: client.clone(),
            channels,
            channel_name,
        });
    }

    for &(channels, channel_name) in CHANNELS {
        run(DistanceBench {
            client: client.clone(),
            channels,
            channel_name,
        });
    }

    for &(channels, channel_name) in CHANNELS {
        run(DistanceRefBench {
            client: client.clone(),
            channels,
            channel_name,
        });
    }

    for &(channels, channel_name) in CHANNELS {
        run(DistancePairBench {
            client: client.clone(),
            channels,
            channel_name,
        });
    }

    for &(channels, channel_name) in CHANNELS {
        run(DistancePairRefBench {
            client: client.clone(),
            channels,
            channel_name,
        });
    }

    run(HSumBench {
        client: client.clone(),
    });
    run(HSumPairBench {
        client: client.clone(),
    });
    run(VWeightBench {
        client: client.clone(),
    });

    for &(channels, channel_name) in CHANNELS {
        run(VWeightPairAccBench {
            client: client.clone(),
            channels,
            channel_name,
        });
    }

    for &(channels, channel_name) in CHANNELS {
        run(AccumulateBench {
            client: client.clone(),
            channels,
            channel_name,
        });
    }

    for &(channels, channel_name) in CHANNELS {
        run(FinishBench {
            client: client.clone(),
            channels,
            channel_name,
        });
    }

    for &(channels, channel_name) in CHANNELS {
        run(BilateralBench {
            client: client.clone(),
            channels,
            channel_name,
        });
    }

    run(NoisePartialBench {
        client: client.clone(),
    });

    // `luma_fields` off as nlmeans runs it, and on as nl4d runs it with the noise map.
    run(TemporalNoiseStatsBench {
        client: client.clone(),
        luma_fields: false,
    });
    run(TemporalNoiseStatsBench {
        client: client.clone(),
        luma_fields: true,
    });

    // Pyramid build and block matching are luma-only since motion estimation ignores chroma. Warp
    // runs per channel mode because its memory traffic scales with `stored_ch`.
    run(DownscaleBench {
        client: client.clone(),
    });
    run(BlockMatchCoarseBench {
        client: client.clone(),
    });
    run(BlockMatchFineBench {
        client: client.clone(),
    });
    run(McConfidenceBench {
        client: client.clone(),
    });
    run(ChainComposeBench {
        client: client.clone(),
    });

    for candidate in 0..REGULARISE_CANDIDATES.len() {
        run(MvRegulariseBench {
            client: client.clone(),
            candidate,
        });
    }

    for &size in GRAIN_SIZES {
        run(GrainSaveVectorsBench {
            client: client.clone(),
            size,
        });
        run(GrainMeasureBench {
            client: client.clone(),
            size,
        });
        run(GrainReducePartialsBench {
            client: client.clone(),
            size,
        });
    }

    for &(channels, channel_name) in CHANNELS {
        for candidate in 0..COLLAB_CANDIDATES.len() {
            run(CollabFusedBench {
                client: client.clone(),
                channels,
                channel_name,
                split_mv: false,
                noise_curve: false,
                strength_map: false,
                pooled: false,
                candidate,
            });
        }
    }

    let luma_arms = [(true, false, false), (true, true, false), (true, true, true)];
    for (noise_curve, strength_map, pooled) in luma_arms {
        for candidate in 0..COLLAB_CANDIDATES.len() {
            run(CollabFusedBench {
                client: client.clone(),
                channels: 1,
                channel_name: "luma",
                split_mv: false,
                noise_curve,
                strength_map,
                pooled,
                candidate,
            });
        }
    }

    for &(channels, channel_name) in CHANNELS {
        for candidate in 0..COLLAB_CANDIDATES.len() {
            run(CollabFusedBench {
                client: client.clone(),
                channels,
                channel_name,
                split_mv: true,
                noise_curve: false,
                strength_map: false,
                pooled: false,
                candidate,
            });
        }
    }

    for &(channels, channel_name) in CHANNELS {
        run(CollabNormaliseBench {
            client: client.clone(),
            channels,
            channel_name,
        });
    }

    for &(channels, channel_name) in CHANNELS {
        run(CollabZeroAccumBench {
            client: client.clone(),
            channels,
            channel_name,
        });
    }

    for &(channels, channel_name) in CHANNELS {
        run(WarpBench {
            client: client.clone(),
            channels,
            channel_name,
        });
    }

    println!();
}

#[derive(clap::Parser, Debug)]
#[command(about = "NLMeans per-kernel benchmarks", long_about = None)]
struct Cli {
    /// GPU device to bind to, one of `default`, `discrete[:N]`, `integrated[:N]`, `virtual[:N]` or `cpu`.
    #[arg(long, default_value = "default")]
    device: Device,

    /// Swallowed, since cargo passes this when invoking the bench binary.
    #[arg(long, hide = true)]
    bench: bool,
}

fn main() {
    let cli = Cli::parse();

    println!("NLMeans Per-Kernel Benchmarks - 1920x1080 (TimingMethod::Device)");
    println!("  override sample count with BENCH_NUM_SAMPLES=N (default 15)");

    #[cfg(feature = "vulkan")]
    {
        let device = cli.device.to_wgpu().expect("wgpu device conversion failed");
        println!("  device:   {device:?}");
        run_all::<cubecl::wgpu::WgpuRuntime>("vulkan", &device);
    }

    #[cfg(feature = "cuda")]
    {
        let device = cubecl::cuda::CudaDevice::new(0);
        println!("  device:   {device:?}");
        run_all::<cubecl::cuda::CudaRuntime>("cuda", &device);
    }

    #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
    {
        let _ = cli;
        eprintln!("No GPU backend enabled. Run with --features vulkan or --features cuda");
        std::process::exit(1);
    }
}

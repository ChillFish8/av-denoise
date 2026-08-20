use cubecl::prelude::*;

mod kernels;

use kernels::accumulate::AccumulateBench;
use kernels::bilateral::BilateralBench;
use kernels::collab_aggregate::{CollabNormaliseBench, CollabZeroAccumBench};
use kernels::collab_ht::CollabHtBench;
use kernels::copy::CopyBench;
use kernels::dist_2d_weight::DistWeightBench;
use kernels::dist_2d_weight_ref::DistWeightRefBench;
use kernels::distance::DistanceBench;
use kernels::distance_pair::DistancePairBench;
use kernels::distance_pair_ref::DistancePairRefBench;
use kernels::distance_ref::DistanceRefBench;
use kernels::finish::FinishBench;
use kernels::fused_window::{
    FusedPairWindowBench,
    FusedPairWindowRefBench,
    FusedSingleWindowBench,
    FusedSingleWindowRefBench,
};
use kernels::horizontal_sum::HSumBench;
use kernels::horizontal_sum_pair::HSumPairBench;
use kernels::mc_block_match_coarse::BlockMatchCoarseBench;
use kernels::mc_block_match_fine::BlockMatchFineBench;
use kernels::mc_chain_compose::ChainComposeBench;
use kernels::mc_confidence::McConfidenceBench;
use kernels::mc_downscale::DownscaleBench;
use kernels::mc_warp::WarpBench;
use kernels::nl4d_group_temporal::CollabGroupTemporalBench;
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

    for &(ch, ch_name) in CHANNELS {
        run(CopyBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }
    for &(ch, ch_name) in CHANNELS {
        run(ZeroBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }

    for &(ch, ch_name) in CHANNELS {
        run(DistWeightBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }
    for &(ch, ch_name) in CHANNELS {
        run(DistWeightRefBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }
    for &(ch, ch_name) in CHANNELS {
        run(FusedSingleWindowBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }
    for &(ch, ch_name) in CHANNELS {
        run(FusedPairWindowBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }
    for &(ch, ch_name) in CHANNELS {
        run(FusedSingleWindowRefBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }
    for &(ch, ch_name) in CHANNELS {
        run(FusedPairWindowRefBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }

    for &(ch, ch_name) in CHANNELS {
        run(DistanceBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }
    for &(ch, ch_name) in CHANNELS {
        run(DistanceRefBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }
    for &(ch, ch_name) in CHANNELS {
        run(DistancePairBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }
    for &(ch, ch_name) in CHANNELS {
        run(DistancePairRefBench {
            client: client.clone(),
            ch,
            ch_name,
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
    for &(ch, ch_name) in CHANNELS {
        run(VWeightPairAccBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }

    for &(ch, ch_name) in CHANNELS {
        run(AccumulateBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }
    for &(ch, ch_name) in CHANNELS {
        run(FinishBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }

    for &(ch, ch_name) in CHANNELS {
        run(BilateralBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }

    // Immerkær noise estimate (stage 1 + stage 2 back-to-back).
    run(NoisePartialBench {
        client: client.clone(),
    });

    // Temporal-residual noise-stats kernel.
    run(TemporalNoiseStatsBench {
        client: client.clone(),
    });

    // Motion-compensation kernels. Pyramid build and analyse are
    // luma-only (ME doesn't look at chroma); warp runs per channel
    // mode because its memory traffic scales with `stored_ch`.
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
    for &(ch, ch_name) in CHANNELS {
        run(CollabGroupTemporalBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }
    for &(ch, ch_name) in CHANNELS {
        run(CollabHtBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }
    for &(ch, ch_name) in CHANNELS {
        run(CollabNormaliseBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }
    for &(ch, ch_name) in CHANNELS {
        run(CollabZeroAccumBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }
    for &(ch, ch_name) in CHANNELS {
        run(WarpBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }

    println!();
}

#[derive(clap::Parser, Debug)]
#[command(about = "NLMeans per-kernel benchmarks", long_about = None)]
struct Cli {
    /// GPU device to bind to. Format: `default`, `discrete[:N]`,
    /// `integrated[:N]`, `virtual[:N]`, or `cpu`.
    #[arg(long, default_value = "default")]
    device: av_denoise::Device,

    /// Swallowed: cargo passes this when invoking the bench binary.
    #[arg(long, hide = true)]
    bench: bool,
}

fn main() {
    use clap::Parser;
    let cli = Cli::parse();

    println!("NLMeans Per-Kernel Benchmarks - 1920x1080 (TimingMethod::Device)");
    println!("  override sample count with BENCH_NUM_SAMPLES=N (default 15)");

    #[cfg(feature = "vulkan")]
    {
        let device = cli.device.to_wgpu().expect("wgpu device conversion failed");
        println!("  device:   {device:?}");
        run_all::<cubecl::wgpu::WgpuRuntime>("vulkan", &device);
    }

    #[cfg(not(feature = "vulkan"))]
    {
        let _ = cli;
        eprintln!("No GPU backend enabled. Run with --features vulkan");
        std::process::exit(1);
    }
}

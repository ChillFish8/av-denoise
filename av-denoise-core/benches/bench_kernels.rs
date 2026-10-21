mod kernels;

use av_denoise_core::bench_api::Device;
use av_denoise_core::bench_api::tune_sites::{
    AccumulateCandidate,
    CollabVariant,
    RegulariseCandidate,
    WindowCandidate,
};
use av_denoise_core::tune::{DeviceLimits, IntegerCodes, SiteCandidate};
use clap::Parser;
use cubecl::prelude::*;
use kernels::accumulate::AccumulateBench;
use kernels::bilateral::BilateralBench;
use kernels::cast_codes::CastCodesBench;
use kernels::cast_f16::CastF16Bench;
use kernels::collab_aggregate::{CollabNormaliseBench, CollabZeroAccumBench};
use kernels::collab_fused::{CollabFusedBench, fitted_variants};
use kernels::collab_split::{CollabSplitBench, fitted_integer_kernels, fitted_split_kernels};
use kernels::compilation_cache::install_bench_compilation_cache;
use kernels::copy::CopyBench;
use kernels::dist_2d_weight::DistWeightBench;
use kernels::dist_2d_weight_ref::DistWeightRefBench;
use kernels::distance::DistanceBench;
use kernels::distance_pair::DistancePairBench;
use kernels::distance_pair_ref::DistancePairRefBench;
use kernels::distance_ref::DistanceRefBench;
use kernels::egress::{EgressBench, EgressFormat};
use kernels::finish::FinishBench;
use kernels::fused_window::{WindowBench, WindowKernel};
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
use kernels::{BENCH_NEIGHBOURS, CHANNELS, print_header, run};

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

    for depth in [8, 10] {
        for &(channels, channel_name) in CHANNELS {
            run(CastCodesBench {
                client: client.clone(),
                channels,
                channel_name,
                depth,
            });
        }
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

    for kernel in WindowKernel::ALL {
        let float_windows = WindowCandidate::ALL
            .iter()
            .copied()
            .filter(|candidate| !candidate.integer());
        for candidate in float_windows {
            for &(channels, channel_name) in CHANNELS {
                run(WindowBench {
                    client: client.clone(),
                    channels,
                    channel_name,
                    kernel,
                    candidate,
                    codes: None,
                });
            }
        }
    }

    // The integer windows read a u16 code ring, and run only without a reference.
    let window_limits = DeviceLimits::from_client(&client);
    let window_depths: &[u32] = if window_limits.u16_buffer { &[8, 10] } else { &[] };
    for &depth in window_depths {
        let codes = IntegerCodes { depth };

        for kernel in [WindowKernel::Single, WindowKernel::Pair] {
            for &(channels, channel_name) in CHANNELS {
                run(WindowBench {
                    client: client.clone(),
                    channels,
                    channel_name,
                    kernel,
                    candidate: WindowCandidate::ScalarInt,
                    codes: Some(codes),
                });
            }
        }
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

    for &candidate in AccumulateCandidate::ALL {
        for &(channels, channel_name) in CHANNELS {
            run(AccumulateBench {
                client: client.clone(),
                channels,
                channel_name,
                candidate,
            });
        }
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

    // Each block-match launch covers every neighbour, so they run at one neighbour and at four.
    for &neighbours in BENCH_NEIGHBOURS {
        run(BlockMatchCoarseBench {
            client: client.clone(),
            neighbours,
        });
        run(BlockMatchFineBench {
            client: client.clone(),
            neighbours,
        });
        run(McConfidenceBench {
            client: client.clone(),
            neighbours,
        });
    }

    run(ChainComposeBench {
        client: client.clone(),
    });

    // Each regularise launch covers every neighbour, so every candidate runs at one neighbour and at four.
    for &neighbours in BENCH_NEIGHBOURS {
        for &candidate in RegulariseCandidate::ALL {
            run(MvRegulariseBench {
                client: client.clone(),
                candidate,
                neighbours,
                static_field: false,
            });
        }
    }

    // A zero motion field, where six of every block's seven candidates repeat.
    for &candidate in RegulariseCandidate::ALL {
        run(MvRegulariseBench {
            client: client.clone(),
            candidate,
            neighbours: 4,
            static_field: true,
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

    // A clipped walk deadlocks on CUDA, so only the variants the device fits are launched. The
    // packed rows run only where the device adds 64-bit atomics.
    let limits = DeviceLimits::from_client(&client);
    let layouts: &[bool] = if limits.int64_atomic_add {
        &[false, true]
    } else {
        &[false]
    };

    for &packed in layouts {
        for &(channels, channel_name) in CHANNELS {
            let variants = fitted_variants(&client, channels, channel_name);

            for candidate in variants {
                run(CollabFusedBench {
                    client: client.clone(),
                    channels,
                    channel_name,
                    split_mv: false,
                    noise_curve: false,
                    strength_map: false,
                    pooled: false,
                    packed,
                    candidate,
                });
            }
        }
    }

    for &(channels, channel_name) in CHANNELS {
        let kernels = fitted_split_kernels(&client, channels, channel_name);

        for kernel in kernels {
            run(CollabSplitBench {
                client: client.clone(),
                channels,
                channel_name,
                kernel,
                codes: None,
            });
        }
    }

    let code_depths: &[u32] = if limits.u16_buffer { &[8, 10] } else { &[] };
    for &depth in code_depths {
        let codes = IntegerCodes { depth };

        for &(channels, channel_name) in CHANNELS {
            let kernels = fitted_integer_kernels(&client, channels, channel_name, codes);

            for kernel in kernels {
                run(CollabSplitBench {
                    client: client.clone(),
                    channels,
                    channel_name,
                    kernel,
                    codes: Some(codes),
                });
            }
        }
    }

    let collab_baseline = CollabVariant::for_client(&client);

    run(CollabFusedBench {
        client: client.clone(),
        channels: 1,
        channel_name: "luma",
        split_mv: false,
        noise_curve: true,
        strength_map: false,
        pooled: false,
        packed: false,
        candidate: collab_baseline,
    });
    run(CollabFusedBench {
        client: client.clone(),
        channels: 1,
        channel_name: "luma",
        split_mv: false,
        noise_curve: true,
        strength_map: true,
        pooled: false,
        packed: false,
        candidate: collab_baseline,
    });
    run(CollabFusedBench {
        client: client.clone(),
        channels: 1,
        channel_name: "luma",
        split_mv: false,
        noise_curve: true,
        strength_map: true,
        pooled: true,
        packed: false,
        candidate: collab_baseline,
    });

    for &(channels, channel_name) in CHANNELS {
        run(CollabFusedBench {
            client: client.clone(),
            channels,
            channel_name,
            split_mv: true,
            noise_curve: false,
            strength_map: false,
            pooled: false,
            packed: false,
            candidate: collab_baseline,
        });
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

    /// Runs only the benches whose name contains this text, for example `nlm_window/32x16`.
    #[arg(long)]
    filter: Option<String>,
}

fn main() {
    let cli = Cli::parse();

    println!("NLMeans Per-Kernel Benchmarks - 1920x1080 (TimingMethod::Device)");
    println!("  override sample count with BENCH_NUM_SAMPLES=N (default 15)");

    if let Some(filter) = cli.filter.clone() {
        let _ = kernels::FILTER.set(filter);
    }

    install_bench_compilation_cache();

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

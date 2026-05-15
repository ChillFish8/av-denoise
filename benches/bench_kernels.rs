//! Per-kernel benchmarks (CubeCL `Benchmark` framework, device timing).
//!
//! Run: `cargo bench --bench bench_kernels --features vulkan`
//! Override sample count: `BENCH_NUM_SAMPLES=N` (default 15).

use cubecl::prelude::*;

mod kernels;

use kernels::accumulate::AccumulateBench;
use kernels::bilateral::BilateralBench;
use kernels::copy::CopyBench;
use kernels::dist_2d_weight::DistWeightBench;
use kernels::dist_2d_weight_ref::DistWeightRefBench;
use kernels::distance::DistanceBench;
use kernels::distance_pair::DistancePairBench;
use kernels::distance_pair_ref::DistancePairRefBench;
use kernels::distance_ref::DistanceRefBench;
use kernels::finish::FinishBench;
use kernels::fused_pair_accumulate::FusedPairBench;
use kernels::fused_pair_accumulate_ref::FusedPairRefBench;
use kernels::horizontal_sum::HSumBench;
use kernels::horizontal_sum_pair::HSumPairBench;
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
        run(FusedPairBench {
            client: client.clone(),
            ch,
            ch_name,
        });
    }
    for &(ch, ch_name) in CHANNELS {
        run(FusedPairRefBench {
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

    println!();
}

#[derive(clap::Parser, Debug)]
#[command(about = "NLMeans per-kernel benchmarks", long_about = None)]
struct Cli {
    /// GPU device to bind to. Format: `default`, `discrete[:N]`,
    /// `integrated[:N]`, `virtual[:N]`, or `cpu`.
    #[arg(long, default_value = "default", value_parser = parse_device_spec)]
    device: DeviceSpec,

    /// Swallowed: cargo passes this when invoking the bench binary.
    #[arg(long, hide = true)]
    bench: bool,
}

#[derive(Clone, Debug)]
struct DeviceSpec {
    kind: DeviceKind,
    index: usize,
}

#[derive(Clone, Debug)]
enum DeviceKind {
    Default,
    Discrete,
    Integrated,
    Virtual,
    Cpu,
}

fn parse_device_spec(s: &str) -> Result<DeviceSpec, String> {
    let (kind_str, idx_str) = s.split_once(':').unwrap_or((s, "0"));
    let index = idx_str
        .parse()
        .map_err(|_| format!("invalid device index '{idx_str}' in '{s}'"))?;
    let kind = match kind_str {
        "default" => DeviceKind::Default,
        "discrete" => DeviceKind::Discrete,
        "integrated" => DeviceKind::Integrated,
        "virtual" => DeviceKind::Virtual,
        "cpu" => DeviceKind::Cpu,
        other => {
            return Err(format!(
                "unknown device kind '{other}'; expected default, discrete[:N], integrated[:N], virtual[:N], or cpu"
            ));
        },
    };
    Ok(DeviceSpec { kind, index })
}

#[cfg(feature = "vulkan")]
fn device_spec_to_wgpu(spec: &DeviceSpec) -> cubecl::wgpu::WgpuDevice {
    use cubecl::wgpu::WgpuDevice;
    match spec.kind {
        DeviceKind::Default => WgpuDevice::DefaultDevice,
        DeviceKind::Discrete => WgpuDevice::DiscreteGpu(spec.index),
        DeviceKind::Integrated => WgpuDevice::IntegratedGpu(spec.index),
        DeviceKind::Virtual => WgpuDevice::VirtualGpu(spec.index),
        DeviceKind::Cpu => WgpuDevice::Cpu,
    }
}

fn main() {
    use clap::Parser;
    let cli = Cli::parse();

    println!("NLMeans Per-Kernel Benchmarks - 1920x1080 (TimingMethod::Device)");
    println!("  override sample count with BENCH_NUM_SAMPLES=N (default 15)");

    #[cfg(feature = "vulkan")]
    {
        let device = device_spec_to_wgpu(&cli.device);
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

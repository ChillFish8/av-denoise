//! Scores nl4d's motion field against synthetic clips with known
//! motion. Prints one table per arm.
//!
//! Run with `cargo bench -p av-denoise-core --bench mc_accuracy --
//! --device discrete:1 --still brick=/path/to/brick.pgm --still
//! asterisk=/path/to/asterisk.pgm`. With no `--still` it runs on a
//! synthetic texture and says so.

use std::path::PathBuf;

use av_denoise_core::nl4d::harness::{Clip, KindScore, MotionClass, Score, Still, score, synthesise};
use av_denoise_core::nl4d::{Nl4dDenoiser, Nl4dParams};
use av_denoise_core::nlmeans::{ChannelMode, NlmParams};
use cubecl::prelude::*;

/// Grain levels on the 8-bit scale.
const GRAIN: [f32; 3] = [2.0, 6.0, 12.0];

/// A named still.
struct NamedStill {
    name: String,
    still: Still,
}

/// One configuration under test.
struct Arm {
    name: &'static str,
    params: fn() -> Nl4dParams,
}

/// `Nl4dParams::default` carries `ChannelMode::Yuv`, which expects
/// three interleaved planes per pushed frame. The harness only ever
/// synthesises a single luma plane, so every arm here switches to
/// `ChannelMode::Luma` instead.
fn baseline_params() -> Nl4dParams {
    Nl4dParams {
        nlm: NlmParams {
            channels: ChannelMode::Luma,
            ..Nl4dParams::default().nlm
        },
        ..Nl4dParams::default()
    }
}

/// `baseline_params` with `field_lambda` overridden, for the lambda
/// ladder. Building from `Nl4dParams::default()` directly would panic
/// with the three-channel default the harness cannot feed.
fn with_lambda_0_25() -> Nl4dParams {
    Nl4dParams {
        field_lambda: 0.25,
        ..baseline_params()
    }
}

fn with_lambda_0_5() -> Nl4dParams {
    Nl4dParams {
        field_lambda: 0.5,
        ..baseline_params()
    }
}

fn with_lambda_1() -> Nl4dParams {
    Nl4dParams {
        field_lambda: 1.0,
        ..baseline_params()
    }
}

fn with_lambda_2() -> Nl4dParams {
    Nl4dParams {
        field_lambda: 2.0,
        ..baseline_params()
    }
}

fn with_lambda_4() -> Nl4dParams {
    Nl4dParams {
        field_lambda: 4.0,
        ..baseline_params()
    }
}

fn arms() -> Vec<Arm> {
    vec![
        Arm {
            name: "baseline",
            params: baseline_params,
        },
        Arm {
            name: "lambda_0.25",
            params: with_lambda_0_25,
        },
        Arm {
            name: "lambda_0.5",
            params: with_lambda_0_5,
        },
        Arm {
            name: "lambda_1",
            params: with_lambda_1,
        },
        Arm {
            name: "lambda_2",
            params: with_lambda_2,
        },
        Arm {
            name: "lambda_4",
            params: with_lambda_4,
        },
    ]
}

fn parse_still(spec: &str) -> Result<NamedStill, String> {
    let (name, path) = spec
        .split_once('=')
        .ok_or_else(|| format!("--still expects name=path, got {spec}"))?;
    let bytes = std::fs::read(PathBuf::from(path)).map_err(|e| format!("{path}: {e}"))?;
    Ok(NamedStill {
        name: name.to_string(),
        still: Still::from_pgm(&bytes)?,
    })
}

fn run_clip<R: Runtime>(client: &ComputeClient<R>, params: Nl4dParams, clip: &Clip) -> Score {
    let refine = params.refine;
    let mut d = Nl4dDenoiser::<R>::new(client, params, clip.width, clip.height).expect("construction failed");
    for frame in &clip.frames {
        d.push_frame(frame);
        let _ = d.denoise_submit().expect("denoise_submit failed");
    }
    let snap = d.motion_snapshot().expect("a pass ran once the window filled");
    score(clip, &snap, refine)
}

fn print_kind(label: &str, k: &KindScore) {
    if k.patches == 0 {
        return;
    }
    println!(
        "    {label:<9} {:>6}  corner {:>5.1}%  covering {:>5.1}%  epe {:>5.2} / p95 {:>5.2}  conf {:>4.2}",
        k.patches,
        100.0 * k.in_window_rate_corner(),
        100.0 * k.in_window_rate_covering(),
        k.epe_mean(),
        k.epe_p95(),
        k.confidence_median(),
    );
}

fn run_all<R: Runtime>(device: &R::Device, stills: &[NamedStill]) {
    let client = R::client(device);
    for arm in arms() {
        println!();
        println!("=== arm: {} ===", arm.name);
        for still in stills {
            for class in MotionClass::ALL {
                for grain in GRAIN {
                    let params = (arm.params)();
                    let clip = synthesise(&still.still, class, params.temporal_radius, grain / 255.0, 7);
                    let s = run_clip::<R>(&client, params, &clip);
                    println!("  {:<10} {:<9} grain {grain:>4.0}", still.name, class.label());
                    print_kind("plain", &s.plain);
                    print_kind("boundary", &s.boundary);
                    print_kind("occluded", &s.occluded);
                }
            }
        }
    }
}

#[derive(clap::Parser, Debug)]
#[command(about = "Motion-field accuracy against synthetic known-motion clips", long_about = None)]
struct Cli {
    /// GPU device to bind to. Format: `default`, `discrete[:N]`,
    /// `integrated[:N]`, `virtual[:N]`, or `cpu`.
    #[arg(long, default_value = "default")]
    device: av_denoise_core::Device,

    /// A still to build clips from, as `name=path.pgm`. Repeatable.
    #[arg(long = "still")]
    stills: Vec<String>,

    /// Swallowed: cargo passes this when invoking the bench binary.
    #[arg(long, hide = true)]
    bench: bool,
}

fn main() {
    use clap::Parser;
    let cli = Cli::parse();

    let stills: Vec<NamedStill> = if cli.stills.is_empty() {
        println!("no --still given, running on a synthetic 256x256 texture");
        vec![NamedStill {
            name: "synthetic".to_string(),
            still: Still::synthetic(256, 256),
        }]
    } else {
        cli.stills
            .iter()
            .map(|s| parse_still(s).unwrap_or_else(|e| panic!("{e}")))
            .collect()
    };

    #[cfg(feature = "vulkan")]
    {
        let device = cli.device.to_wgpu().expect("wgpu device conversion failed");
        println!("device: {device:?}");
        run_all::<cubecl::wgpu::WgpuRuntime>(&device, &stills);
    }

    #[cfg(not(feature = "vulkan"))]
    {
        let _ = stills;
        eprintln!("No GPU backend enabled. Run with --features vulkan");
        std::process::exit(1);
    }
}

//! Scores nl4d's motion field against synthetic clips with known motion
//!
//! Prints one table per arm. With no `--still` it runs on a synthetic texture and says so.
//!
//! ```text
//! cargo bench -p av-denoise-core --bench mc_accuracy -- \
//!     --device discrete:1 --still brick=/path/to/brick.pgm --still asterisk=/path/to/asterisk.pgm
//! ```

use std::path::PathBuf;

use av_denoise_core::bench_api::harness::{Clip, KindScore, MotionClass, Score, Still, score, synthesise};
use av_denoise_core::bench_api::{Device, HostIo, Nl4dDenoiser, Nl4dParams, NlmParams};
use av_denoise_core::{ChannelMode, MotionCompensationMode};
use clap::Parser;
use cubecl::prelude::*;

/// Grain levels on the 8-bit scale.
const GRAIN: [f32; 3] = [2.0, 6.0, 12.0];

struct NamedStill {
    name: String,
    still: Still,
}

/// One configuration under test.
struct Arm {
    name: &'static str,
    params: fn() -> Nl4dParams,
}

/// The default parameters switched to `ChannelMode::Luma`.
///
/// `Nl4dParams::default` carries `ChannelMode::Yuv`, which expects three interleaved planes per
/// pushed frame, and the harness only synthesises a single luma plane.
fn baseline_params() -> Nl4dParams {
    let nlm = NlmParams {
        channels: ChannelMode::Luma,
        ..Nl4dParams::default().nlm
    };

    Nl4dParams {
        nlm,
        ..Nl4dParams::default()
    }
}

/// [baseline_params] with `field_lambda` at 0.5, for context against the shipped default.
///
/// It builds on the luma baseline because the three-channel default would panic in the harness.
fn with_lambda_0_5() -> Nl4dParams {
    Nl4dParams {
        field_lambda: 0.5,
        ..baseline_params()
    }
}

/// [baseline_params] with the motion pyramid deepened to three levels.
///
/// It tests whether the extra level earns its added kernel launch.
fn with_pyramid_3() -> Nl4dParams {
    let mut params = baseline_params();
    if let MotionCompensationMode::Mvtools { pyramid_levels, .. } = &mut params.nlm.motion_compensation {
        *pyramid_levels = 3;
    }

    params
}

fn arms() -> Vec<Arm> {
    vec![
        Arm {
            name: "baseline",
            params: baseline_params,
        },
        Arm {
            name: "lambda_0.5",
            params: with_lambda_0_5,
        },
        Arm {
            name: "pyramid_3",
            params: with_pyramid_3,
        },
    ]
}

fn parse_still(spec: &str) -> Result<NamedStill, String> {
    let (name, path) = spec
        .split_once('=')
        .ok_or_else(|| format!("--still expects name=path, got {spec}"))?;
    let path_buf = PathBuf::from(path);
    let bytes = std::fs::read(path_buf).map_err(|err| format!("{path}: {err}"))?;
    let still = Still::from_pgm(&bytes)?;

    Ok(NamedStill {
        name: name.to_string(),
        still,
    })
}

fn run_clip<R: Runtime>(client: &ComputeClient<R>, params: Nl4dParams, clip: &Clip) -> Score {
    let refine = params.refine;
    let mut denoiser =
        Nl4dDenoiser::<R>::new(client, params, clip.width, clip.height).expect("construction failed");
    for frame in &clip.frames {
        denoiser.push_frame(frame);
        let _ = denoiser.denoise().expect("denoise failed");
    }

    let snapshot = denoiser
        .motion_snapshot()
        .expect("a pass ran once the window filled");

    score(clip, &snapshot, refine)
}

fn print_kind(label: &str, kind: &KindScore) {
    if kind.patches == 0 {
        return;
    }

    let corner_rate = 100.0 * kind.in_window_rate_corner();
    let covering_rate = 100.0 * kind.in_window_rate_covering();
    let epe_mean = kind.epe_mean();
    let epe_p95 = kind.epe_p95();
    let confidence = kind.confidence_median();

    println!(
        "    {label:<9} {:>6}  corner {:>5.1}%  covering {:>5.1}%  epe {:>5.2} / p95 {:>5.2}  conf {:>4.2}",
        kind.patches, corner_rate, covering_rate, epe_mean, epe_p95, confidence,
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
                    let clip_score = run_clip::<R>(&client, params, &clip);
                    let class_label = class.label();

                    println!("  {:<10} {:<9} grain {grain:>4.0}", still.name, class_label);
                    print_kind("plain", &clip_score.plain);
                    print_kind("boundary", &clip_score.boundary);
                    print_kind("occluded", &clip_score.occluded);
                }
            }
        }
    }
}

#[derive(clap::Parser, Debug)]
#[command(about = "Motion-field accuracy against synthetic known-motion clips", long_about = None)]
struct Cli {
    /// GPU device to bind to, one of `default`, `discrete[:N]`, `integrated[:N]`, `virtual[:N]` or `cpu`.
    #[arg(long, default_value = "default")]
    device: Device,

    /// A still to build clips from, as `name=path.pgm`. Repeatable.
    #[arg(long = "still")]
    stills: Vec<String>,

    /// Swallowed, since cargo passes this when invoking the bench binary.
    #[arg(long, hide = true)]
    bench: bool,
}

fn main() {
    let cli = Cli::parse();

    let stills: Vec<NamedStill> = if cli.stills.is_empty() {
        println!("no --still given, running on a synthetic 256x256 texture");
        let still = Still::synthetic(256, 256);
        let synthetic = NamedStill {
            name: "synthetic".to_string(),
            still,
        };

        vec![synthetic]
    } else {
        cli.stills
            .iter()
            .map(|spec| parse_still(spec).unwrap_or_else(|err| panic!("{err}")))
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

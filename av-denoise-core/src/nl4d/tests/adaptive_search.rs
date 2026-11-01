use super::helpers::make_client;
use crate::bench_api::HostIo;
use crate::motion::MotionCompensationMode;
use crate::nl4d::{Nl4dDenoiser, Nl4dParams};
use crate::nlmeans::denoiser::LOOKAHEAD_FRAMES;
use crate::nlmeans::tests::models::{PanShape, pan_clip};

const WIDTH: u32 = 256;
const HEIGHT: u32 = 128;
const SIGMA: f32 = 2.0 / 255.0;
/// Twelve frames leave the noise look-ahead while the stream is pushed.
const FRAMES: usize = 12 + LOOKAHEAD_FRAMES as usize;

/// Every frame `params` puts out for a grainy pan, fast enough that some blocks escalate.
fn run_stream(params: Nl4dParams) -> Vec<Vec<f32>> {
    let client = make_client();
    let shape = PanShape {
        width: WIDTH,
        height: HEIGHT,
        chroma_scale: (2, 2),
        shift: (5, 3),
        sigma: SIGMA,
    };
    let frames = pan_clip(shape, FRAMES, 41);
    let mut denoiser = Nl4dDenoiser::new(&client, params, WIDTH, HEIGHT).expect("construction failed");

    let mut outputs = Vec::new();
    for frame in &frames {
        denoiser.push_frame(&frame.luma);
        let output = denoiser.denoise().expect("denoise failed");
        outputs.extend(output);
    }

    outputs
}

fn bits(frames: &[Vec<f32>]) -> Vec<Vec<u32>> {
    frames
        .iter()
        .map(|frame| frame.iter().map(|value| value.to_bits()).collect())
        .collect()
}

#[test]
fn escalating_defaults_are_bit_identical_run_to_run() {
    let defaults = Nl4dParams::default();
    let MotionCompensationMode::Mvtools {
        search_radius,
        max_search_escalation,
        ..
    } = defaults.nlm.motion_compensation
    else {
        panic!("nl4d always searches");
    };
    assert!(
        max_search_escalation > search_radius,
        "test premise: the defaults escalate"
    );

    let first = run_stream(defaults.clone());
    let second = run_stream(defaults);

    assert!(!first.is_empty(), "the stream puts frames out");
    assert!(
        bits(&first) == bits(&second),
        "an escalating search must give the same output on every run"
    );
}

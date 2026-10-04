use super::helpers::{R, make_client, noisy_copy_of, static_clip_params};
use crate::bench_api::HostIo;
use crate::nl4d::{Nl4dDenoiser, Nl4dParams};

const WIDTH: u32 = 96;
const HEIGHT: u32 = 64;
const FRAMES: u32 = 5;
const BACKGROUND: f32 = 0.4;
const LINE_CONTRAST: f32 = 0.025;
const GRAIN: f32 = 6.0 / 255.0;

/// Whether column `x` carries one of the faint vertical lines.
fn on_line(x: u32) -> bool {
    x % 16 == 8
}

/// A flat field with faint one-pixel vertical lines, under independent grain per frame.
fn faint_line_clip() -> Vec<Vec<f32>> {
    let mut clean = vec![BACKGROUND; (WIDTH * HEIGHT) as usize];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if on_line(x) {
                clean[(y * WIDTH + x) as usize] += LINE_CONTRAST;
            }
        }
    }

    (0..FRAMES)
        .map(|seed| noisy_copy_of(&clean, WIDTH, HEIGHT, GRAIN, seed))
        .collect()
}

fn denoise(params: Nl4dParams, frames: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let client = make_client();
    let mut denoiser = Nl4dDenoiser::<R>::new(&client, params, WIDTH, HEIGHT).expect("construction failed");

    let mut outputs = Vec::new();
    for frame in frames {
        denoiser.push_frame(frame);
        let output = denoiser.denoise().expect("denoise failed");
        if let Some(output_frame) = output {
            outputs.push(output_frame);
        }
    }

    denoiser
        .flush(|frame| {
            let output_frame = frame.to_vec();
            outputs.push(output_frame);
        })
        .expect("flush failed");
    outputs
}

/// The mean over every output of the on-line pixels minus the off-line ones.
fn line_contrast(outputs: &[Vec<f32>]) -> f64 {
    let mut on_sum = 0.0f64;
    let mut on_count = 0.0f64;
    let mut off_sum = 0.0f64;
    let mut off_count = 0.0f64;
    for output in outputs {
        for y in 8..HEIGHT - 8 {
            for x in 8..WIDTH - 8 {
                let value = output[(y * WIDTH + x) as usize] as f64;
                if on_line(x) {
                    on_sum += value;
                    on_count += 1.0;
                } else if !on_line(x + 1) && !on_line(x - 1) {
                    off_sum += value;
                    off_count += 1.0;
                }
            }
        }
    }
    on_sum / on_count - off_sum / off_count
}

fn params(pooled_threshold: bool, lambda_ht: f32) -> Nl4dParams {
    Nl4dParams {
        pooled_threshold,
        lambda_ht,
        ..static_clip_params(2)
    }
}

#[test]
fn pooling_keeps_more_of_a_faint_line() {
    let frames = faint_line_clip();

    let pooled = denoise(params(true, 3.78), &frames);
    let plain = denoise(params(false, 3.78), &frames);

    let pooled_contrast = line_contrast(&pooled);
    let plain_contrast = line_contrast(&plain);
    assert!(
        pooled_contrast > plain_contrast,
        "{pooled_contrast} vs {plain_contrast}"
    );
}

#[test]
fn lambda_still_changes_the_output_with_pooling_on() {
    let frames = faint_line_clip();

    let gentle = denoise(params(true, 3.0), &frames);
    let strong = denoise(params(true, 4.5), &frames);

    assert_ne!(gentle, strong);
}

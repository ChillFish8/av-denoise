use super::helpers::{R, SIGMA, make_client, static_clip_params, textured_base};
use crate::bench_api::HostIo;
use crate::nl4d::Nl4dDenoiser;
use crate::nlmeans::tests::helpers::noisy_field_over;

const SIZE: u32 = 64;

/// Pushes `count` grain frames and returns every output in order, plus
/// how many pushes had gone in when the first output arrived.
fn run_stream(radius: u32, count: u32) -> (Vec<Vec<f32>>, Option<u32>) {
    let client = make_client();
    let params = static_clip_params(radius);
    let mut denoiser = Nl4dDenoiser::<R>::new(&client, params, SIZE, SIZE).expect("construction failed");
    let base = textured_base(SIZE, SIZE);
    let mut outputs = Vec::new();
    let mut first_output_at = None;

    for seed in 0..count {
        let frame = noisy_field_over(&base, SIZE, SIZE, SIGMA, seed);
        denoiser.push_frame(&frame);

        let Some(values) = denoiser.denoise().expect("denoise failed") else {
            continue;
        };

        first_output_at.get_or_insert(seed + 1);
        outputs.push(values);
    }

    denoiser
        .flush(|frame| {
            let values = frame.to_vec();
            outputs.push(values);
        })
        .expect("flush failed");

    (outputs, first_output_at)
}

fn residual_std(output: &[f32], clean: &[f32]) -> f32 {
    let count = output.len() as f32;
    let mean = output
        .iter()
        .zip(clean)
        .map(|(out, reference)| out - reference)
        .sum::<f32>()
        / count;
    let variance = output
        .iter()
        .zip(clean)
        .map(|(out, reference)| (out - reference - mean).powi(2))
        .sum::<f32>()
        / count;
    variance.sqrt()
}

/// Pushes `count` frames seeded from `first_seed` and returns the push
/// count at which the first output arrived.
fn first_output_push(denoiser: &mut Nl4dDenoiser<R>, count: u32, first_seed: u32) -> Option<u32> {
    let base = textured_base(SIZE, SIZE);
    let mut first_output_at = None;

    for push in 0..count {
        let frame = noisy_field_over(&base, SIZE, SIZE, SIGMA, first_seed + push);
        denoiser.push_frame(&frame);

        if denoiser.denoise().expect("denoise failed").is_some() {
            first_output_at.get_or_insert(push + 1);
        }
    }

    first_output_at
}

#[test]
fn every_stream_length_emits_every_frame() {
    for radius in [1u32, 2] {
        for count in 1..=(4 * radius + 3) {
            let (outputs, _) = run_stream(radius, count);
            assert_eq!(outputs.len(), count as usize, "radius={radius} count={count}");
        }
    }
}

#[test]
fn the_first_output_arrives_on_the_push_that_fills_the_ring() {
    for radius in [1u32, 2] {
        let (_, first_output_at) = run_stream(radius, 4 * radius + 3);
        assert_eq!(first_output_at, Some(2 * radius + 1), "radius={radius}");
    }
}

#[test]
fn a_scene_of_exactly_the_ring_length_emits_every_frame() {
    let radius = 2;
    let count = 2 * radius + 1;
    let (outputs, _) = run_stream(radius, count);
    let clean = textured_base(SIZE, SIZE);

    assert_eq!(outputs.len(), count as usize);

    for (index, output) in outputs.iter().enumerate() {
        let residual = residual_std(output, &clean);
        assert!(
            residual < SIGMA,
            "frame {index} was not denoised, residual {residual}"
        );
    }
}

#[test]
fn a_short_scene_emits_every_frame_without_black_output() {
    let radius = 2;
    let clean = textured_base(SIZE, SIZE);

    for count in 1..=(2 * radius) {
        let (outputs, _) = run_stream(radius, count);
        assert_eq!(outputs.len(), count as usize);

        for (index, output) in outputs.iter().enumerate() {
            let mean = output.iter().sum::<f32>() / output.len() as f32;
            assert!(mean > 0.1, "count={count} frame {index} came out black");

            let residual = residual_std(output, &clean);
            assert!(
                residual < SIGMA,
                "count={count} frame {index} residual {residual}"
            );
        }
    }
}

#[test]
fn edge_frames_are_denoised_as_strongly_as_mid_scene() {
    let radius = 2;
    let count = 12;
    let (outputs, _) = run_stream(radius, count);
    let clean = textured_base(SIZE, SIZE);
    let residuals: Vec<f32> = outputs
        .iter()
        .map(|output| residual_std(output, &clean))
        .collect();
    let first = residuals[0];
    let middle = residuals[count as usize / 2];
    let last = residuals[count as usize - 1];

    assert!(first <= 1.10 * middle, "first {first} vs middle {middle}");
    assert!(last <= 1.10 * middle, "last {last} vs middle {middle}");
}

/// A ring primed without any submit reaches `flush` with `passes_run == 0` and uncleared
/// accumulators, which must not scatter stale contributions into the output.
#[test]
fn a_full_ring_primed_without_any_submit_flushes_without_black_output() {
    let radius = 2;
    let client = make_client();
    let params = static_clip_params(radius);
    let mut denoiser = Nl4dDenoiser::<R>::new(&client, params, SIZE, SIZE).expect("construction failed");
    let base = textured_base(SIZE, SIZE);
    let clean = base.clone();

    denoiser.mark_continuation();

    for seed in 0..(2 * radius + 1) {
        let frame = noisy_field_over(&base, SIZE, SIZE, SIGMA, seed);
        denoiser.push_frame(&frame);
    }

    let mut outputs = Vec::new();
    denoiser
        .flush(|frame| {
            let values = frame.to_vec();
            outputs.push(values);
        })
        .expect("flush failed");

    assert_eq!(outputs.len(), 2 * radius as usize);

    // Only two tail passes run, so coverage per frame is uneven and a couple of frames sit above
    // `SIGMA`. This pins the stale scatter, not full-strength denoising, so the bound is looser than
    // the `SIGMA` the other tests in this file use.
    for (index, output) in outputs.iter().enumerate() {
        let mean = output.iter().sum::<f32>() / output.len() as f32;
        assert!(mean > 0.1, "frame {index} came out black");

        let residual = residual_std(output, &clean);
        assert!(
            residual < 1.6 * SIGMA,
            "frame {index} residual {residual} exceeds 1.6x sigma"
        );
    }
}

#[test]
fn a_second_scene_after_a_continuation_stream_runs_head_passes() {
    let radius = 2;
    let count = 2 * radius + 3;
    let client = make_client();
    let params = static_clip_params(radius);
    let mut denoiser = Nl4dDenoiser::<R>::new(&client, params, SIZE, SIZE).expect("construction failed");

    denoiser.mark_continuation();
    first_output_push(&mut denoiser, count, 0);
    denoiser.flush(|_| {}).expect("flush failed");

    let first_output_at = first_output_push(&mut denoiser, count, 100);

    assert_eq!(first_output_at, Some(2 * radius + 1));
}

#[test]
fn a_continuation_stream_first_emits_after_the_radius_more_passes() {
    let radius = 2;
    let client = make_client();
    let params = static_clip_params(radius);
    let mut denoiser = Nl4dDenoiser::<R>::new(&client, params, SIZE, SIZE).expect("construction failed");

    denoiser.mark_continuation();
    let first_output_at = first_output_push(&mut denoiser, 4 * radius + 1, 0);

    assert_eq!(first_output_at, Some(3 * radius + 1));
}

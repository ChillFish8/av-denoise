use super::*;

// The tests name the `Vulkan` accelerator, which only exists with the `vulkan` feature.
#[cfg(feature = "vulkan")]
mod reseed {
    use super::*;
    use crate::HqParams;
    use crate::accelerate::Accelerator;

    fn layout() -> FrameLayout {
        FrameLayout {
            width: 64,
            height: 64,
            subsampling: Subsampling::Yuv420,
            depth: Depth::Eight,
        }
    }

    /// Temporal nlmeans at `radius`, denoising both planes independently.
    fn test_plane_options(radius: u32) -> PlaneOptions {
        PlaneOptions {
            accelerators: vec![Accelerator::Vulkan],
            device: Device::Default,
            intent: ChannelIntent::LumaChroma,
            mode: DenoisingMode::Temporal { radius },
            algorithm: Algorithm::default(),
            luma_strength: None,
            chroma_strength: None,
            luma_lambda_ht: None,
            chroma_lambda_ht: None,
        }
    }

    fn test_plane_options_with_intent(radius: u32, intent: ChannelIntent) -> PlaneOptions {
        PlaneOptions {
            intent,
            ..test_plane_options(radius)
        }
    }

    /// A small xorshift generator, so the test data is the same on every run.
    fn pseudo_random(mut state: u64) -> u64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    }

    /// One plane's bytes for frame `frame_index`.
    ///
    /// A spatial ramp, a per-frame offset and a deterministic dither are summed and clamped, so a
    /// temporal filter has real signal and real noise to work with.
    fn ramp_plane(pixels: usize, width: u32, frame_index: usize, plane_seed: u64) -> Vec<u8> {
        let width = width.max(1) as usize;

        (0..pixels)
            .map(|i| {
                let x = (i % width) as u32;
                let y = (i / width) as u32;
                let spatial = x.wrapping_add(y) % 120;
                let frame_offset = (frame_index as u32 * 7) % 60;
                let seed = (i as u64) ^ (frame_index as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ plane_seed;
                let dither = (pseudo_random(seed) % 16) as u32;
                let value = 20 + spatial + frame_offset + dither;
                value.min(235) as u8
            })
            .collect()
    }

    /// `count` frames whose bytes vary per frame and per pixel, so a temporal filter sees a
    /// non-degenerate signal.
    fn ramp_clip(layout: &FrameLayout, count: usize) -> Vec<Planes> {
        let (chroma_width, _) = layout.chroma_dims();

        (0..count)
            .map(|frame_index| {
                let y_plane = ramp_plane(layout.luma_pixels(), layout.width, frame_index, 1);
                let u_plane = ramp_plane(layout.chroma_pixels(), chroma_width, frame_index, 2);
                let v_plane = ramp_plane(layout.chroma_pixels(), chroma_width, frame_index, 3);

                Planes {
                    y: y_plane,
                    u: u_plane,
                    v: v_plane,
                }
            })
            .collect()
    }

    /// Renders every frame through the streaming path.
    fn stream_all(options: &PlaneOptions, frames: &[Planes]) -> Vec<Planes> {
        let frame_layout = layout();
        stream_all_with_layout(options, frame_layout, frames)
    }

    fn stream_all_with_layout(
        options: &PlaneOptions,
        frame_layout: FrameLayout,
        frames: &[Planes],
    ) -> Vec<Planes> {
        let mut denoiser = PlanarDenoiser::create(options, frame_layout).unwrap();
        let mut outputs = Vec::new();
        for frame in frames {
            denoiser.push(frame).unwrap();
            if let Some(planes) = denoiser.recv().unwrap() {
                outputs.push(planes);
            }
        }

        denoiser.flush(|planes| outputs.push(planes)).unwrap();
        outputs
    }

    fn window_of(frames: &[Planes], target: usize, radius: usize) -> Vec<Planes> {
        (0..(2 * radius + 1))
            .map(|i| {
                let index = (target + i).saturating_sub(radius).min(frames.len() - 1);
                frames[index].clone()
            })
            .collect()
    }

    #[test]
    fn reseed_matches_the_streaming_output_mid_clip() {
        let frame_layout = layout();
        let options = test_plane_options(2);
        let frames = ramp_clip(&frame_layout, 12);
        let streamed = stream_all(&options, &frames);

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let target_frame = 6;
        let window = window_of(&frames, target_frame, 2);
        let got = denoiser.reseed(&window).unwrap();

        assert_eq!(got.y, streamed[target_frame].y);
        assert_eq!(got.u, streamed[target_frame].u);
        assert_eq!(got.v, streamed[target_frame].v);
    }

    #[test]
    fn reseed_matches_the_streaming_output_at_both_clip_edges() {
        let frame_layout = layout();
        let options = test_plane_options(2);
        let frames = ramp_clip(&frame_layout, 12);
        let streamed = stream_all(&options, &frames);
        let last = frames.len() - 1;

        for target_frame in [0usize, last] {
            let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
            let window = window_of(&frames, target_frame, 2);
            let got = denoiser.reseed(&window).unwrap();

            assert_eq!(
                got.y, streamed[target_frame].y,
                "luma mismatch at k = {target_frame}"
            );
            assert_eq!(
                got.u, streamed[target_frame].u,
                "u mismatch at k = {target_frame}"
            );
            assert_eq!(
                got.v, streamed[target_frame].v,
                "v mismatch at k = {target_frame}"
            );
        }
    }

    #[test]
    fn reseed_recovers_a_half_poisoned_by_an_earlier_failure() {
        let frame_layout = layout();
        let options = test_plane_options(2);
        let frames = ramp_clip(&frame_layout, 12);

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        denoiser.luma.as_mut().unwrap().poison_for_test();
        denoiser.chroma.as_mut().unwrap().poison_for_test();

        let window = window_of(&frames, 6, 2);
        let got = denoiser.reseed(&window).unwrap();

        assert!(!got.y.is_empty());
        assert!(!got.u.is_empty());
        assert!(!got.v.is_empty());
    }

    /// Plain nlmeans carries no noise state between frames, so repeated out-of-order reseeds on one
    /// denoiser must match streaming.
    ///
    /// The radius is wider and the clip longer and more shuffled than in any other reseed test here,
    /// so a history-dependent regression has room to show.
    #[test]
    fn nlmeans_repeated_out_of_order_reseeds_match_streaming() {
        let frame_layout = layout();
        let options = test_plane_options(4);
        let frames = ramp_clip(&frame_layout, 24);
        let streamed = stream_all(&options, &frames);

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();

        // Skews late, like the VapourSynth plugin harness's shuffled order.
        let order = [
            18, 4, 23, 9, 12, 2, 20, 6, 15, 1, 22, 7, 17, 3, 11, 19, 0, 21, 8, 16, 5, 14, 10, 13,
        ];

        for &target_frame in &order {
            let window = window_of(&frames, target_frame, 4);
            let got = denoiser.reseed(&window).unwrap();

            assert_eq!(
                got.y, streamed[target_frame].y,
                "luma mismatch at k = {target_frame}"
            );
            assert_eq!(
                got.u, streamed[target_frame].u,
                "u mismatch at k = {target_frame}"
            );
            assert_eq!(
                got.v, streamed[target_frame].v,
                "v mismatch at k = {target_frame}"
            );
        }
    }

    #[test]
    fn a_reseed_leaves_the_stream_positioned_for_the_next_frame() {
        let frame_layout = layout();
        let options = test_plane_options(2);
        let frames = ramp_clip(&frame_layout, 12);
        let streamed = stream_all(&options, &frames);
        let (target_frame, radius) = (6usize, 2usize);

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let window = window_of(&frames, target_frame, radius);
        denoiser.reseed(&window).unwrap();
        denoiser.push(&frames[target_frame + 1 + radius]).unwrap();
        let got = denoiser.recv().unwrap().expect("frame k + 1");

        assert_eq!(got.y, streamed[target_frame + 1].y);
    }

    #[test]
    fn reseed_rejects_a_window_of_the_wrong_length() {
        let frame_layout = layout();
        let options = test_plane_options(2);
        let frames = ramp_clip(&frame_layout, 12);
        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();

        let error = denoiser.reseed(&frames[..3]).unwrap_err().to_string();

        assert!(
            error.contains("5"),
            "error should name the expected length, got {error}"
        );
    }

    /// `ChannelIntent::Luma` sends chroma through the passthrough queue, and a reseed queues one
    /// entry per window frame.
    ///
    /// The entry paired with the denoised centre must be the centre frame's own chroma, not a
    /// neighbour's.
    #[test]
    fn reseed_pairs_the_passthrough_plane_with_the_centre_frame() {
        let frame_layout = layout();
        let options = test_plane_options_with_intent(2, ChannelIntent::Luma);
        let frames = ramp_clip(&frame_layout, 12);
        let (target_frame, radius) = (6usize, 2usize);

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let window = window_of(&frames, target_frame, radius);
        let got = denoiser.reseed(&window).unwrap();

        assert_eq!(
            got.u, frames[target_frame].u,
            "u should pass through from the centre frame"
        );
        assert_eq!(
            got.v, frames[target_frame].v,
            "v should pass through from the centre frame"
        );
    }

    #[test]
    fn reseed_pairs_the_passthrough_luma_plane_with_the_centre_frame() {
        let frame_layout = layout();
        let options = test_plane_options_with_intent(2, ChannelIntent::Chroma);
        let frames = ramp_clip(&frame_layout, 12);
        let (target_frame, radius) = (6usize, 2usize);

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let window = window_of(&frames, target_frame, radius);
        let got = denoiser.reseed(&window).unwrap();

        assert_eq!(
            got.y, frames[target_frame].y,
            "y should pass through from the centre frame"
        );
    }

    /// A single reseed only checks the first passthrough entry `recv` pops.
    ///
    /// An extra or missing entry that still leaves the right one at the front only misaligns the
    /// plane paired with the next frame, once streaming resumes.
    #[test]
    fn reseed_then_streaming_keeps_the_passthrough_plane_aligned_on_the_next_frame() {
        let frame_layout = layout();
        let options = test_plane_options_with_intent(2, ChannelIntent::Luma);
        let frames = ramp_clip(&frame_layout, 12);
        let (target_frame, radius) = (6usize, 2usize);

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let window = window_of(&frames, target_frame, radius);
        denoiser.reseed(&window).unwrap();
        denoiser.push(&frames[target_frame + 1 + radius]).unwrap();
        let got = denoiser.recv().unwrap().expect("frame k + 1");

        assert_eq!(
            got.u,
            frames[target_frame + 1].u,
            "u should pass through from frame k + 1"
        );
        assert_eq!(
            got.v,
            frames[target_frame + 1].v,
            "v should pass through from frame k + 1"
        );
    }

    /// Temporal nl4d at `radius` with a pinned `sigma`.
    ///
    /// The automatic estimate is a moving average over every frame since the stream last reset,
    /// history a windowed `reseed` cannot supply. Pinning it keeps these tests on what the window
    /// shape and pass sequence decide.
    fn nl4d_plane_options(radius: u32) -> PlaneOptions {
        let nl4d_options = Nl4dOptions {
            sigma: Some(0.03),
            ..Nl4dOptions::default()
        };

        PlaneOptions {
            algorithm: Algorithm::Nl4d(nl4d_options),
            ..test_plane_options(radius)
        }
    }

    /// The window `span` needs for frame `target`, clamped at both clip ends like `window_of`.
    fn window_of_span(frames: &[Planes], target: usize, span: WindowSpan) -> Vec<Planes> {
        (0..span.frame_count())
            .map(|i| {
                let index = (target + i).saturating_sub(span.behind).min(frames.len() - 1);
                frames[index].clone()
            })
            .collect()
    }

    /// The shifted window around frame `target`.
    ///
    /// It stops at the clip's ends rather than repeating them, and returns the target's index in it.
    fn shifted_window_of(
        frames: &[Planes],
        target: usize,
        span: WindowSpan,
    ) -> (Vec<Planes>, ReseedWindowFlags) {
        let first = target.saturating_sub(span.behind);
        let last = (target + span.ahead).min(frames.len() - 1);
        let window = frames[first..=last].to_vec();
        let flags = ReseedWindowFlags {
            target: target - first,
            at_clip_start: first == 0,
            at_clip_end: last == frames.len() - 1,
        };
        (window, flags)
    }

    struct ReseedWindowFlags {
        target: usize,
        at_clip_start: bool,
        at_clip_end: bool,
    }

    fn reseed_shifted(denoiser: &mut PlanarDenoiser, frames: &[Planes], target: usize) -> Vec<Planes> {
        let span = denoiser.window_span();
        let (window, flags) = shifted_window_of(frames, target, span);
        let request = ReseedWindow {
            frames: &window,
            target: flags.target,
            at_clip_start: flags.at_clip_start,
            at_clip_end: flags.at_clip_end,
        };
        denoiser.reseed_window(request).unwrap()
    }

    /// How many outputs a shifted reseed at `target` returns.
    ///
    /// That is the target alone mid-clip, or the target through the clip's end.
    fn shifted_output_count(denoiser: &PlanarDenoiser, clip_len: usize, target: usize) -> usize {
        let span = denoiser.window_span();
        if target + span.ahead >= clip_len - 1 {
            clip_len - target
        } else {
            1
        }
    }

    #[test]
    fn nl4d_reseed_matches_the_streaming_output_mid_clip() {
        let frame_layout = layout();
        let options = nl4d_plane_options(2);
        let frames = ramp_clip(&frame_layout, 16);
        let streamed = stream_all(&options, &frames);

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let target_frame = 8;
        let span = denoiser.window_span();
        let window = window_of_span(&frames, target_frame, span);
        let got = denoiser.reseed(&window).unwrap();

        assert_eq!(got.y, streamed[target_frame].y);
        assert_eq!(got.u, streamed[target_frame].u);
        assert_eq!(got.v, streamed[target_frame].v);
    }

    fn max_abs_diff(left: &[u8], right: &[u8]) -> i32 {
        left.iter()
            .zip(right.iter())
            .map(|(&left_sample, &right_sample)| (left_sample as i32 - right_sample as i32).abs())
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn nl4d_reseed_window_matches_streaming_at_every_frame() {
        let frame_layout = layout();
        let option_sets = [
            nl4d_plane_options(2),
            nl4d_windowed_plane_options(2),
            nl4d_plane_options_with_intent(2, ChannelIntent::Luma),
            nl4d_plane_options_with_intent(2, ChannelIntent::Chroma),
        ];

        for options in option_sets {
            for clip_len in [3usize, 7, 12] {
                let frames = ramp_clip(&frame_layout, clip_len);
                let streamed = stream_all(&options, &frames);
                assert_eq!(streamed.len(), clip_len);

                for target_frame in 0..clip_len {
                    let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
                    let got = reseed_shifted(&mut denoiser, &frames, target_frame);

                    let expected_len = shifted_output_count(&denoiser, clip_len, target_frame);
                    assert_eq!(got.len(), expected_len, "len={clip_len} k={target_frame}");

                    for (offset, planes) in got.iter().enumerate() {
                        let index = target_frame + offset;
                        assert_eq!(
                            planes.y, streamed[index].y,
                            "len={clip_len} k={target_frame} frame {index} luma"
                        );
                        assert_eq!(
                            planes.u, streamed[index].u,
                            "len={clip_len} k={target_frame} frame {index} u"
                        );
                        assert_eq!(
                            planes.v, streamed[index].v,
                            "len={clip_len} k={target_frame} frame {index} v"
                        );
                    }
                }
            }
        }
    }

    /// `ChannelIntent::YuvFused` needs a 4:4:4 source, so it runs over its own layout.
    #[test]
    fn nl4d_reseed_window_matches_streaming_at_every_frame_in_yuv_fused_mode() {
        let fused_layout = FrameLayout {
            subsampling: Subsampling::Yuv444,
            ..layout()
        };
        let options = nl4d_plane_options_with_intent(2, ChannelIntent::YuvFused);

        for clip_len in [3usize, 7, 12] {
            let frames = ramp_clip(&fused_layout, clip_len);
            let streamed = stream_all_with_layout(&options, fused_layout, &frames);
            assert_eq!(streamed.len(), clip_len);

            for target_frame in 0..clip_len {
                let mut denoiser = PlanarDenoiser::create(&options, fused_layout).unwrap();
                let got = reseed_shifted(&mut denoiser, &frames, target_frame);

                let expected_len = shifted_output_count(&denoiser, clip_len, target_frame);
                assert_eq!(got.len(), expected_len, "len={clip_len} k={target_frame}");

                for (offset, planes) in got.iter().enumerate() {
                    let index = target_frame + offset;
                    assert_eq!(
                        planes.y, streamed[index].y,
                        "len={clip_len} k={target_frame} frame {index} luma"
                    );
                    assert_eq!(
                        planes.u, streamed[index].u,
                        "len={clip_len} k={target_frame} frame {index} u"
                    );
                    assert_eq!(
                        planes.v, streamed[index].v,
                        "len={clip_len} k={target_frame} frame {index} v"
                    );
                }
            }
        }
    }

    #[test]
    fn nl4d_reseed_window_pairs_passthrough_at_the_last_frame() {
        let frame_layout = layout();
        let options = nl4d_plane_options_with_intent(2, ChannelIntent::Luma);
        let frames = ramp_clip(&frame_layout, 16);
        let last = frames.len() - 1;
        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();

        let got = reseed_shifted(&mut denoiser, &frames, last);

        assert_eq!(got.len(), 1);
        assert_eq!(got[0].u, frames[last].u);
        assert_eq!(got[0].v, frames[last].v);
    }

    #[test]
    fn nl4d_reseed_window_pairs_passthrough_at_the_first_frame() {
        let frame_layout = layout();
        let options = nl4d_plane_options_with_intent(2, ChannelIntent::Luma);
        let frames = ramp_clip(&frame_layout, 16);
        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();

        let got = reseed_shifted(&mut denoiser, &frames, 0);

        assert_eq!(got.len(), 1);
        assert_eq!(got[0].u, frames[0].u);
        assert_eq!(got[0].v, frames[0].v);
    }

    #[test]
    fn nl4d_reseed_window_then_streaming_continues_from_the_clip_start() {
        let frame_layout = layout();
        let options = nl4d_windowed_plane_options(2);
        let frames = ramp_clip(&frame_layout, 16);
        let streamed = stream_all(&options, &frames);
        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let span = denoiser.window_span();

        reseed_shifted(&mut denoiser, &frames, 1);
        denoiser.push(&frames[2 + span.ahead]).unwrap();
        let next = denoiser.recv().unwrap().unwrap();

        assert_eq!(next.y, streamed[2].y);
    }

    #[test]
    fn nl4d_reseed_then_streaming_continues_correctly() {
        let frame_layout = layout();
        let options = nl4d_plane_options(2);
        let frames = ramp_clip(&frame_layout, 16);
        let streamed = stream_all(&options, &frames);
        let target_frame = 8usize;

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let span = denoiser.window_span();
        let window = window_of_span(&frames, target_frame, span);
        denoiser.reseed(&window).unwrap();

        // The next frame in source order after the reseed window's last frame is
        // `target_frame + 1 + span.ahead`.
        denoiser.push(&frames[target_frame + 1 + span.ahead]).unwrap();
        let got = denoiser.recv().unwrap().expect("frame k + 1");

        assert_eq!(got.y, streamed[target_frame + 1].y);
        assert_eq!(got.u, streamed[target_frame + 1].u);
        assert_eq!(got.v, streamed[target_frame + 1].v);
    }

    /// The error names nl4d's wider `4r+1` window length, not nlmeans's `2r+1`.
    #[test]
    fn nl4d_reseed_rejects_a_window_of_the_wrong_length() {
        let frame_layout = layout();
        let options = nl4d_plane_options(2);
        let frames = ramp_clip(&frame_layout, 16);
        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let expected = denoiser.window_span().frame_count();
        let expected_text = expected.to_string();

        let error = denoiser.reseed(&frames[..3]).unwrap_err().to_string();

        assert!(
            error.contains(&expected_text),
            "error should name the expected length ({expected}), got {error}"
        );
    }

    fn nl4d_plane_options_with_intent(radius: u32, intent: ChannelIntent) -> PlaneOptions {
        PlaneOptions {
            intent,
            ..nl4d_plane_options(radius)
        }
    }

    /// nl4d drains after every emission during a reseed, not only the last, and each drain pops one
    /// passthrough entry.
    ///
    /// The walk must still land on the target's own entry rather than an earlier, discarded one.
    #[test]
    fn nl4d_reseed_pairs_the_passthrough_plane_with_the_centre_frame() {
        let frame_layout = layout();
        let options = nl4d_plane_options_with_intent(2, ChannelIntent::Luma);
        let frames = ramp_clip(&frame_layout, 16);
        let target_frame = 8;

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let span = denoiser.window_span();
        let window = window_of_span(&frames, target_frame, span);
        let got = denoiser.reseed(&window).unwrap();

        assert_eq!(
            got.u, frames[target_frame].u,
            "u should pass through from the centre frame"
        );
        assert_eq!(
            got.v, frames[target_frame].v,
            "v should pass through from the centre frame"
        );
    }

    #[test]
    fn nl4d_reseed_pairs_the_passthrough_luma_plane_with_the_centre_frame() {
        let frame_layout = layout();
        let options = nl4d_plane_options_with_intent(2, ChannelIntent::Chroma);
        let frames = ramp_clip(&frame_layout, 16);
        let target_frame = 8;

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let span = denoiser.window_span();
        let window = window_of_span(&frames, target_frame, span);
        let got = denoiser.reseed(&window).unwrap();

        assert_eq!(
            got.y, frames[target_frame].y,
            "y should pass through from the centre frame"
        );
    }

    /// A single-shot pairing test only checks the first entry `recv` pops after the drop.
    ///
    /// An extra or missing entry that still leaves the right one at the front only misaligns the
    /// plane paired with the frame after the target, once streaming resumes.
    #[test]
    fn nl4d_reseed_then_streaming_keeps_the_passthrough_plane_aligned_on_the_next_frame() {
        let frame_layout = layout();
        let options = nl4d_plane_options_with_intent(2, ChannelIntent::Luma);
        let frames = ramp_clip(&frame_layout, 16);
        let target_frame = 8;

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let span = denoiser.window_span();
        let window = window_of_span(&frames, target_frame, span);
        denoiser.reseed(&window).unwrap();
        denoiser.push(&frames[target_frame + 1 + span.ahead]).unwrap();
        let got = denoiser.recv().unwrap().expect("frame k + 1");

        assert_eq!(
            got.u,
            frames[target_frame + 1].u,
            "u should pass through from frame k + 1"
        );
        assert_eq!(
            got.v,
            frames[target_frame + 1].v,
            "v should pass through from frame k + 1"
        );
    }

    /// Temporal nl4d at `radius` with window-local noise estimation and an automatic `sigma`, as
    /// `av-denoise-vs` runs it.
    ///
    /// `sigma` stays unpinned because window-local estimation exists so the automatic estimate
    /// agrees between `reseed` and streaming.
    fn nl4d_windowed_plane_options(radius: u32) -> PlaneOptions {
        let nl4d_options = Nl4dOptions {
            windowed_noise_estimation: true,
            ..Nl4dOptions::default()
        };

        PlaneOptions {
            algorithm: Algorithm::Nl4d(nl4d_options),
            ..test_plane_options(radius)
        }
    }

    /// Like `nl4d_reseed_matches_the_streaming_output_mid_clip`, with the noise estimator running
    /// instead of pinned.
    #[test]
    fn nl4d_windowed_reseed_matches_the_streaming_output_mid_clip() {
        let frame_layout = layout();
        let options = nl4d_windowed_plane_options(2);
        let frames = ramp_clip(&frame_layout, 16);
        let streamed = stream_all(&options, &frames);

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let target_frame = 8;
        let span = denoiser.window_span();
        let window = window_of_span(&frames, target_frame, span);
        let got = denoiser.reseed(&window).unwrap();

        assert_eq!(got.y, streamed[target_frame].y);
        assert_eq!(got.u, streamed[target_frame].u);
        assert_eq!(got.v, streamed[target_frame].v);
    }

    /// With window-local estimation, a `reseed` at a frame then a `push`/`recv` for the next frame must
    /// match a `reseed` at the next frame on a fresh denoiser.
    ///
    /// Without it the fast path folds history the reseed path never sees, so the two disagree on the
    /// same window of content.
    #[test]
    fn nl4d_windowed_fast_path_agrees_with_reseed_at_the_next_frame() {
        let frame_layout = layout();
        let options = nl4d_windowed_plane_options(2);
        let frames = ramp_clip(&frame_layout, 16);
        let target_frame = 8usize;

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let span = denoiser.window_span();
        let window = window_of_span(&frames, target_frame, span);
        denoiser.reseed(&window).unwrap();
        denoiser.push(&frames[target_frame + 1 + span.ahead]).unwrap();
        let via_fast_path = denoiser.recv().unwrap().expect("frame k + 1");

        let mut fresh = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let next_window = window_of_span(&frames, target_frame + 1, span);
        let via_reseed = fresh.reseed(&next_window).unwrap();

        assert_eq!(via_fast_path.y, via_reseed.y);
        assert_eq!(via_fast_path.u, via_reseed.u);
        assert_eq!(via_fast_path.v, via_reseed.v);
    }

    /// Temporal nlmeans HQ at `radius` with window-local noise estimation and an automatic `sigma`.
    fn nlmeans_hq_windowed_plane_options(radius: u32) -> PlaneOptions {
        let hq = HqParams {
            windowed_noise_estimation: true,
            ..HqParams::default()
        };
        let hq_options = NlmeansHqOptions {
            nlm: NlmeansOptions::default(),
            hq,
        };

        PlaneOptions {
            algorithm: Algorithm::NlmeansHq(hq_options),
            ..test_plane_options(radius)
        }
    }

    /// Pins a VapourSynth plugin bug where HQ with an automatic `sigma` returned different pixels for
    /// the same frame depending on request order.
    #[test]
    fn nlmeans_hq_windowed_reseed_matches_the_streaming_output_mid_clip() {
        let frame_layout = layout();
        let options = nlmeans_hq_windowed_plane_options(2);
        let frames = ramp_clip(&frame_layout, 16);
        let streamed = stream_all(&options, &frames);

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let target_frame = 8;
        let span = denoiser.window_span();
        let window = window_of_span(&frames, target_frame, span);
        let got = denoiser.reseed(&window).unwrap();

        assert_eq!(got.y, streamed[target_frame].y);
        assert_eq!(got.u, streamed[target_frame].u);
        assert_eq!(got.v, streamed[target_frame].v);
    }

    #[test]
    fn nlmeans_hq_windowed_reseed_matches_the_streaming_output_at_both_clip_edges() {
        let frame_layout = layout();
        let options = nlmeans_hq_windowed_plane_options(2);
        let frames = ramp_clip(&frame_layout, 16);
        let streamed = stream_all(&options, &frames);
        let last = frames.len() - 1;

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let span = denoiser.window_span();
        let last_window = window_of_span(&frames, last, span);
        let got = denoiser.reseed(&last_window).unwrap();

        assert_eq!(got.y, streamed[last].y, "luma mismatch at the ahead edge");
        assert_eq!(got.u, streamed[last].u, "u mismatch at the ahead edge");
        assert_eq!(got.v, streamed[last].v, "v mismatch at the ahead edge");

        const BEHIND_EDGE_TOLERANCE: i32 = 8;

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let first_window = window_of_span(&frames, 0, span);
        let got = denoiser.reseed(&first_window).unwrap();
        let luma_diff = max_abs_diff(&got.y, &streamed[0].y);

        assert!(
            luma_diff <= BEHIND_EDGE_TOLERANCE,
            "luma at the behind edge (k=0) drifted too far from streaming: max abs diff {luma_diff}"
        );
    }

    /// Drives one denoiser through the VapourSynth plugin harness's shuffled order with its hybrid
    /// fast-path and `reseed` policy, comparing every frame with a true stream.
    ///
    /// A single reseed, or a reseed then one push, passes under window-local estimation without
    /// exercising this. It takes a longer, repeatedly reseeded run to expose state that window-local
    /// estimation fails to clear.
    #[test]
    fn nlmeans_hq_windowed_repeated_out_of_order_access_matches_streaming() {
        let frame_layout = layout();
        let options = nlmeans_hq_windowed_plane_options(2);
        let frames = ramp_clip(&frame_layout, 14);
        let streamed = stream_all(&options, &frames);
        let last = frames.len() - 1;

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let span = denoiser.window_span();
        let mut previous_index: Option<usize> = None;

        // The VapourSynth plugin harness's exact shuffled order.
        let order = [9usize, 0, 13, 4, 5, 6, 1, 12, 2, 11, 3, 10, 7, 8];
        const NEAR_START_TOLERANCE: i32 = 8;

        for &frame_index in &order {
            let fast_output = if previous_index == Some(frame_index.wrapping_sub(1)) && frame_index > 0 {
                let ahead = (frame_index + span.ahead).min(last);
                denoiser.push(&frames[ahead]).unwrap();
                denoiser.recv().unwrap()
            } else {
                None
            };
            let got = match fast_output {
                Some(planes) => planes,
                None => {
                    let window = window_of_span(&frames, frame_index, span);
                    denoiser.reseed(&window).unwrap()
                },
            };
            previous_index = Some(frame_index);

            if frame_index < span.behind {
                let luma_diff = max_abs_diff(&got.y, &streamed[frame_index].y);
                let u_diff = max_abs_diff(&got.u, &streamed[frame_index].u);
                let v_diff = max_abs_diff(&got.v, &streamed[frame_index].v);
                let max_diff = luma_diff.max(u_diff).max(v_diff);

                assert!(
                    max_diff <= NEAR_START_TOLERANCE,
                    "near-start frame n = {frame_index} drifted too far from streaming: max abs diff {max_diff}"
                );
            } else {
                assert_eq!(
                    got.y, streamed[frame_index].y,
                    "luma mismatch at n = {frame_index}"
                );
                assert_eq!(got.u, streamed[frame_index].u, "u mismatch at n = {frame_index}");
                assert_eq!(got.v, streamed[frame_index].v, "v mismatch at n = {frame_index}");
            }
        }
    }

    #[test]
    fn nlmeans_hq_windowed_fast_path_agrees_with_reseed_at_the_next_frame() {
        let frame_layout = layout();
        let options = nlmeans_hq_windowed_plane_options(2);
        let frames = ramp_clip(&frame_layout, 16);
        let target_frame = 8usize;

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let span = denoiser.window_span();
        let window = window_of_span(&frames, target_frame, span);
        denoiser.reseed(&window).unwrap();
        denoiser.push(&frames[target_frame + 1 + span.ahead]).unwrap();
        let via_fast_path = denoiser.recv().unwrap().expect("frame k + 1");

        let mut fresh = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let next_window = window_of_span(&frames, target_frame + 1, span);
        let via_reseed = fresh.reseed(&next_window).unwrap();

        assert_eq!(via_fast_path.y, via_reseed.y);
        assert_eq!(via_fast_path.u, via_reseed.u);
        assert_eq!(via_fast_path.v, via_reseed.v);
    }

    /// Renders `order` with the VapourSynth plugin harness's hybrid `render` policy.
    ///
    /// A frame that directly follows the previous request takes the fast `push`/`recv` path, and any
    /// other frame, or one where `recv` yields nothing, goes through `reseed`.
    fn render_sequence(
        denoiser: &mut PlanarDenoiser,
        frames: &[Planes],
        span: WindowSpan,
        order: &[usize],
    ) -> Vec<Planes> {
        let last = frames.len() - 1;
        let mut previous_index: Option<usize> = None;
        let mut outputs = Vec::new();
        for &frame_index in order {
            let fast_output = if previous_index == Some(frame_index.wrapping_sub(1)) && frame_index > 0 {
                let ahead = (frame_index + span.ahead).min(last);
                denoiser.push(&frames[ahead]).unwrap();
                denoiser.recv().unwrap()
            } else {
                None
            };
            let got = match fast_output {
                Some(planes) => planes,
                None => {
                    let window = window_of_span(frames, frame_index, span);
                    denoiser.reseed(&window).unwrap()
                },
            };
            previous_index = Some(frame_index);
            outputs.push(got);
        }

        outputs
    }

    /// Mirrors the plugin's `a_sequential_run_after_a_seek_stays_correct_nlmeans`.
    ///
    /// After a reseed at frame 11 of a 14-frame clip, the two fast-path frames that follow are
    /// compared with the same policy run from frame 0. That is the reference the VapourSynth harness
    /// uses, rather than the true continuous stream `stream_all` produces.
    #[test]
    fn nlmeans_hq_windowed_sequential_run_after_a_seek_stays_correct() {
        let frame_layout = layout();
        let options = nlmeans_hq_windowed_plane_options(2);
        let frames = ramp_clip(&frame_layout, 14);

        let mut linear = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let span = linear.window_span();
        let linear_order: Vec<usize> = (0..frames.len()).collect();
        let linear_out = render_sequence(&mut linear, &frames, span, &linear_order);

        let mut seeked = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let seeked_out = render_sequence(&mut seeked, &frames, span, &[11, 12, 13]);

        for (i, frame_index) in [12usize, 13].into_iter().enumerate() {
            let got = &seeked_out[i + 1];
            let expected = &linear_out[frame_index];

            assert_eq!(got.y, expected.y, "luma mismatch at n = {frame_index}");
            assert_eq!(got.u, expected.u, "u mismatch at n = {frame_index}");
            assert_eq!(got.v, expected.v, "v mismatch at n = {frame_index}");
        }
    }

    /// The same check at the VapourSynth harness's clip size, 160x120.
    #[test]
    fn nlmeans_hq_windowed_sequential_run_after_a_seek_stays_correct_at_harness_size() {
        let harness_layout = FrameLayout {
            width: 160,
            height: 120,
            subsampling: Subsampling::Yuv420,
            depth: Depth::Eight,
        };
        let options = nlmeans_hq_windowed_plane_options(2);
        let frames = ramp_clip(&harness_layout, 14);

        let mut linear = PlanarDenoiser::create(&options, harness_layout).unwrap();
        let span = linear.window_span();
        let linear_order: Vec<usize> = (0..frames.len()).collect();
        let linear_out = render_sequence(&mut linear, &frames, span, &linear_order);

        let mut seeked = PlanarDenoiser::create(&options, harness_layout).unwrap();
        let seeked_out = render_sequence(&mut seeked, &frames, span, &[11, 12, 13]);

        for (i, frame_index) in [12usize, 13].into_iter().enumerate() {
            let got = &seeked_out[i + 1];
            let expected = &linear_out[frame_index];

            assert_eq!(got.y, expected.y, "luma mismatch at n = {frame_index}");
            assert_eq!(got.u, expected.u, "u mismatch at n = {frame_index}");
            assert_eq!(got.v, expected.v, "v mismatch at n = {frame_index}");
        }
    }

    /// Drives one denoiser through a shuffled order with the VapourSynth plugin's hybrid fast-path and
    /// `reseed` policy, comparing every frame with a true stream.
    ///
    /// It reproduces the plugin's `random_access_matches_sequential_access_nl4d` at the core level.
    /// It pins a defect where, under window-local estimation, the temporal-only noise estimator kept
    /// its last trustworthy reading on folds without one, unlike every other chain. A reseed starts
    /// from `reset_stream_state`, so its short run could find no reading while a true stream still
    /// coasted on one from many frames back. Targets whose window covers either clip end reseed
    /// through `reseed_window` with a shifted window.
    #[test]
    fn nl4d_windowed_repeated_out_of_order_access_matches_streaming() {
        let frame_layout = layout();
        let options = nl4d_windowed_plane_options(2);
        let frames = ramp_clip(&frame_layout, 14);
        let streamed = stream_all(&options, &frames);
        let last = frames.len() - 1;

        let mut denoiser = PlanarDenoiser::create(&options, frame_layout).unwrap();
        let span = denoiser.window_span();
        let mut previous_index: Option<usize> = None;

        // The VapourSynth plugin harness's exact shuffled order.
        let order = [9usize, 0, 13, 4, 5, 6, 1, 12, 2, 11, 3, 10, 7, 8];

        for &frame_index in &order {
            let fast_output = if previous_index == Some(frame_index.wrapping_sub(1)) && frame_index > 0 {
                let ahead = (frame_index + span.ahead).min(last);
                denoiser.push(&frames[ahead]).unwrap();
                denoiser.recv().unwrap()
            } else {
                None
            };

            let at_edge = frame_index <= span.behind || frame_index + span.ahead >= last;
            let got = match fast_output {
                Some(planes) => planes,
                None if at_edge => {
                    let outputs = reseed_shifted(&mut denoiser, &frames, frame_index);
                    outputs[0].clone()
                },
                None => {
                    let window = window_of_span(&frames, frame_index, span);
                    denoiser.reseed(&window).unwrap()
                },
            };
            previous_index = Some(frame_index);

            assert_eq!(
                got.y, streamed[frame_index].y,
                "luma mismatch at n = {frame_index}"
            );
            assert_eq!(got.u, streamed[frame_index].u, "u mismatch at n = {frame_index}");
            assert_eq!(got.v, streamed[frame_index].v, "v mismatch at n = {frame_index}");
        }
    }
}

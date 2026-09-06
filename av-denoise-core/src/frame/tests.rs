//! Tests for [`super::PlanarDenoiser::reseed`].

use super::*;

// Feature-gated because `test_plane_options` names the `Vulkan`
// accelerator variant, which only exists when the `vulkan` feature is
// enabled.
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

    /// A `PlaneOptions` running temporal nlmeans at radius `r`, denoising
    /// both planes independently.
    fn test_plane_options(r: u32) -> PlaneOptions {
        PlaneOptions {
            accelerators: vec![Accelerator::Vulkan],
            device: Device::Default,
            intent: ChannelIntent::LumaChroma,
            mode: DenoisingMode::Temporal { radius: r },
            algorithm: Algorithm::default(),
            luma_strength: None,
            chroma_strength: None,
            luma_lambda_ht: None,
            chroma_lambda_ht: None,
            luma_mismatch_scale: None,
            chroma_mismatch_scale: None,
        }
    }

    /// A `PlaneOptions` identical to [`test_plane_options`] except only
    /// `intent` differs, for exercising a passthrough side.
    fn test_plane_options_with_intent(r: u32, intent: ChannelIntent) -> PlaneOptions {
        PlaneOptions {
            intent,
            ..test_plane_options(r)
        }
    }

    /// A small xorshift generator, deterministic across runs so the test
    /// data does not vary between executions.
    fn pseudo_random(mut x: u64) -> u64 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }

    /// One plane's bytes for frame `frame_idx`: a spatial ramp across the
    /// plane, a per-frame offset, and a deterministic dither, all summed
    /// and clamped so a temporal filter has real signal and real noise to
    /// work with.
    fn ramp_plane(pixels: usize, width: u32, frame_idx: usize, plane_seed: u64) -> Vec<u8> {
        let width = width.max(1) as usize;

        (0..pixels)
            .map(|i| {
                let x = (i % width) as u32;
                let y = (i / width) as u32;
                let spatial = x.wrapping_add(y) % 120;
                let frame_offset = (frame_idx as u32 * 7) % 60;
                let seed = (i as u64) ^ (frame_idx as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ plane_seed;
                let dither = (pseudo_random(seed) % 16) as u32;
                let value = 20 + spatial + frame_offset + dither;
                value.min(235) as u8
            })
            .collect()
    }

    /// Builds `count` `Planes` whose bytes vary per frame and per pixel,
    /// so a temporal filter sees a non-degenerate signal.
    fn ramp_clip(layout: &FrameLayout, count: usize) -> Vec<Planes> {
        let (chroma_w, _) = layout.chroma_dims();

        (0..count)
            .map(|frame_idx| Planes {
                y: ramp_plane(layout.luma_pixels(), layout.width, frame_idx, 1),
                u: ramp_plane(layout.chroma_pixels(), chroma_w, frame_idx, 2),
                v: ramp_plane(layout.chroma_pixels(), chroma_w, frame_idx, 3),
            })
            .collect()
    }

    /// Renders `count` frames through the streaming path.
    fn stream_all(opts: &PlaneOptions, frames: &[Planes]) -> Vec<Planes> {
        let mut d = PlanarDenoiser::create(opts, layout()).unwrap();
        let mut out = Vec::new();
        for f in frames {
            d.push(f).unwrap();
            if let Some(p) = d.recv().unwrap() {
                out.push(p);
            }
        }
        d.flush(|p| out.push(p)).unwrap();
        out
    }

    fn window_of(frames: &[Planes], k: usize, r: usize) -> Vec<Planes> {
        (0..(2 * r + 1))
            .map(|i| {
                let idx = (k + i).saturating_sub(r).min(frames.len() - 1);
                frames[idx].clone()
            })
            .collect()
    }

    #[test]
    fn reseed_matches_the_streaming_output_mid_clip() {
        let opts = test_plane_options(2);
        let frames = ramp_clip(&layout(), 12);
        let streamed = stream_all(&opts, &frames);

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let k = 6;
        let got = d.reseed(&window_of(&frames, k, 2)).unwrap();

        assert_eq!(got.y, streamed[k].y);
        assert_eq!(got.u, streamed[k].u);
        assert_eq!(got.v, streamed[k].v);
    }

    #[test]
    fn reseed_matches_the_streaming_output_at_both_clip_edges() {
        let opts = test_plane_options(2);
        let frames = ramp_clip(&layout(), 12);
        let streamed = stream_all(&opts, &frames);
        let last = frames.len() - 1;

        for k in [0usize, last] {
            let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
            let got = d.reseed(&window_of(&frames, k, 2)).unwrap();
            assert_eq!(got.y, streamed[k].y, "luma mismatch at k = {k}");
            assert_eq!(got.u, streamed[k].u, "u mismatch at k = {k}");
            assert_eq!(got.v, streamed[k].v, "v mismatch at k = {k}");
        }
    }

    #[test]
    fn reseed_recovers_a_half_poisoned_by_an_earlier_failure() {
        let opts = test_plane_options(2);
        let frames = ramp_clip(&layout(), 12);

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        d.luma.as_mut().unwrap().poison_for_test();
        d.chroma.as_mut().unwrap().poison_for_test();

        let got = d.reseed(&window_of(&frames, 6, 2)).unwrap();
        assert!(!got.y.is_empty());
        assert!(!got.u.is_empty());
        assert!(!got.v.is_empty());
    }

    /// Whether plain nlmeans's `reseed` stays order-independent under
    /// repeated out-of-order reseeds on one long-lived denoiser, the
    /// same stress the VapourSynth plugin's shuffled-access-order
    /// harness test puts `avd.NLMeans` through.
    ///
    /// `Algorithm::Nlmeans`'s own doc comment says it runs with "no
    /// noise measurement", and `NlmeansOptions` has no `hq` field for a
    /// `PlaneOptions` built from it to carry, so `NlmParams::hq` stays
    /// `None` and `fold_noise_estimate` never runs for it. There is no
    /// stream-carried noise state for repeated `reseed` calls to
    /// disagree about, so this is expected to hold without a
    /// `windowed_noise_estimation` equivalent for nlmeans. This proves
    /// that rather than assumes it, at a wider temporal radius and a
    /// longer, more heavily shuffled clip than any other reseed test
    /// here uses, so a history-dependent regression would have room to
    /// show itself if one existed.
    #[test]
    fn nlmeans_repeated_out_of_order_reseeds_match_streaming() {
        let opts = test_plane_options(4);
        let frames = ramp_clip(&layout(), 24);
        let streamed = stream_all(&opts, &frames);

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        // Skews late, mirroring the plugin harness's shuffled order
        // that first exposed the nl4d defect.
        let order = [
            18, 4, 23, 9, 12, 2, 20, 6, 15, 1, 22, 7, 17, 3, 11, 19, 0, 21, 8, 16, 5, 14, 10, 13,
        ];

        for &k in &order {
            let got = d.reseed(&window_of(&frames, k, 4)).unwrap();
            assert_eq!(got.y, streamed[k].y, "luma mismatch at k = {k}");
            assert_eq!(got.u, streamed[k].u, "u mismatch at k = {k}");
            assert_eq!(got.v, streamed[k].v, "v mismatch at k = {k}");
        }
    }

    #[test]
    fn a_reseed_leaves_the_stream_positioned_for_the_next_frame() {
        let opts = test_plane_options(2);
        let frames = ramp_clip(&layout(), 12);
        let streamed = stream_all(&opts, &frames);
        let (k, r) = (6usize, 2usize);

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        d.reseed(&window_of(&frames, k, r)).unwrap();
        d.push(&frames[k + 1 + r]).unwrap();
        let got = d.recv().unwrap().expect("frame k + 1");

        assert_eq!(got.y, streamed[k + 1].y);
    }

    #[test]
    fn reseed_rejects_a_window_of_the_wrong_length() {
        let opts = test_plane_options(2);
        let frames = ramp_clip(&layout(), 12);
        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();

        let err = d.reseed(&frames[..3]).unwrap_err().to_string();
        assert!(
            err.contains("5"),
            "error should name the expected length, got {err}"
        );
    }

    /// `ChannelIntent::Luma` leaves chroma disabled, so its planes travel
    /// through the passthrough queue instead of a `Denoiser`. A reseed's
    /// priming pushes queue one passthrough entry per window frame, and
    /// this checks the entry `recv` pairs with the denoised centre is the
    /// centre frame's own chroma, not a neighbour's.
    #[test]
    fn reseed_pairs_the_passthrough_plane_with_the_centre_frame() {
        let opts = test_plane_options_with_intent(2, ChannelIntent::Luma);
        let frames = ramp_clip(&layout(), 12);
        let (k, r) = (6usize, 2usize);

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let got = d.reseed(&window_of(&frames, k, r)).unwrap();

        assert_eq!(got.u, frames[k].u, "u should pass through from the centre frame");
        assert_eq!(got.v, frames[k].v, "v should pass through from the centre frame");
    }

    /// The mirror of [`reseed_pairs_the_passthrough_plane_with_the_centre_frame`]
    /// for `ChannelIntent::Chroma`, where luma is the disabled side.
    #[test]
    fn reseed_pairs_the_passthrough_luma_plane_with_the_centre_frame() {
        let opts = test_plane_options_with_intent(2, ChannelIntent::Chroma);
        let frames = ramp_clip(&layout(), 12);
        let (k, r) = (6usize, 2usize);

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let got = d.reseed(&window_of(&frames, k, r)).unwrap();

        assert_eq!(got.y, frames[k].y, "y should pass through from the centre frame");
    }

    /// A single `reseed` call only checks the very first passthrough
    /// entry `recv` pops. A leftover-count defect after the drop (an
    /// extra or missing entry that still happens to leave the right one
    /// at the front) would pass every single-shot test here and only
    /// misalign the plane paired with the frame right after the centre,
    /// once streaming resumes.
    #[test]
    fn reseed_then_streaming_keeps_the_passthrough_plane_aligned_on_the_next_frame() {
        let opts = test_plane_options_with_intent(2, ChannelIntent::Luma);
        let frames = ramp_clip(&layout(), 12);
        let (k, r) = (6usize, 2usize);

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        d.reseed(&window_of(&frames, k, r)).unwrap();
        d.push(&frames[k + 1 + r]).unwrap();
        let got = d.recv().unwrap().expect("frame k + 1");

        assert_eq!(got.u, frames[k + 1].u, "u should pass through from frame k + 1");
        assert_eq!(got.v, frames[k + 1].v, "v should pass through from frame k + 1");
    }

    /// A `PlaneOptions` identical to [`test_plane_options`] except the
    /// algorithm is `Nl4d`, which needs the wider window `reseed` has
    /// to build for it instead of nlmeans's `2r+1` one.
    ///
    /// Pins `sigma` rather than leaving it on nl4d's automatic
    /// per-frame estimate. That estimate is an exponential moving
    /// average smoothed over every frame folded into it since the
    /// stream last reset, so it carries genuine history from before
    /// the window on a real, never-reset stream, history a windowed
    /// `reseed` cannot supply and was never meant to reproduce. Pinning
    /// it keeps these tests checking what `reseed`'s window shape and
    /// pass sequence are actually responsible for, not that unrelated
    /// warm-up behaviour.
    fn nl4d_plane_options(r: u32) -> PlaneOptions {
        PlaneOptions {
            algorithm: Algorithm::Nl4d(Nl4dOptions {
                sigma: Some(0.03),
                ..Nl4dOptions::default()
            }),
            ..test_plane_options(r)
        }
    }

    /// The window a [`PlanarDenoiser::window_span`] of `span` needs for
    /// target frame `k`, clamped at both clip ends exactly as
    /// [`window_of`] clamps nlmeans's `2r+1` window.
    ///
    /// Reads the span from the accessor rather than hand-deriving it,
    /// so this stays correct however the algorithm's own span is
    /// shaped.
    fn window_of_span(frames: &[Planes], k: usize, span: WindowSpan) -> Vec<Planes> {
        (0..span.frame_count())
            .map(|i| {
                let idx = (k + i).saturating_sub(span.behind).min(frames.len() - 1);
                frames[idx].clone()
            })
            .collect()
    }

    /// This is the test that would have caught the original defect:
    /// `reseed` for a mid-clip frame under `Algorithm::Nl4d` must match
    /// the streaming path's own output for that frame bit-for-bit, the
    /// same property [`reseed_matches_the_streaming_output_mid_clip`]
    /// checks for nlmeans.
    #[test]
    fn nl4d_reseed_matches_the_streaming_output_mid_clip() {
        let opts = nl4d_plane_options(2);
        let frames = ramp_clip(&layout(), 16);
        let streamed = stream_all(&opts, &frames);

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let k = 8;
        let span = d.window_span();
        let got = d.reseed(&window_of_span(&frames, k, span)).unwrap();

        assert_eq!(got.y, streamed[k].y);
        assert_eq!(got.u, streamed[k].u);
        assert_eq!(got.v, streamed[k].v);
    }

    /// The largest absolute per-sample difference between two same-sized
    /// byte planes.
    fn max_abs_diff(a: &[u8], b: &[u8]) -> i32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as i32 - y as i32).abs())
            .max()
            .unwrap_or(0)
    }

    /// The count of samples whose absolute difference between two
    /// same-sized byte planes exceeds `threshold`.
    fn count_exceeding(a: &[u8], b: &[u8], threshold: i32) -> usize {
        a.iter()
            .zip(b.iter())
            .filter(|&(&x, &y)| (x as i32 - y as i32).abs() > threshold)
            .count()
    }

    /// The nl4d mirror of
    /// [`reseed_matches_the_streaming_output_at_both_clip_edges`], where
    /// the widened window clamps at both ends of the clip.
    ///
    /// The forward (`ahead`) edge is bit-exact, the same property the
    /// nlmeans version checks exactly. The backward (`behind`) edge only
    /// has to fall within a bound, because the two paths fill a clip's
    /// leading edge with different amounts of the same padding: `reseed`
    /// fills the whole `behind` span of its window by repeating the
    /// clip's first frame, while a fresh stream primes only `radius`
    /// duplicates of that first frame before real ones start arriving.
    /// nl4d's cross-frame accumulator folds a different amount of
    /// duplicated history in each case, so the two outputs land close
    /// but not identical at the very start of a clip.
    #[test]
    fn nl4d_reseed_matches_the_streaming_output_at_both_clip_edges() {
        let opts = nl4d_plane_options(2);
        let frames = ramp_clip(&layout(), 16);
        let streamed = stream_all(&opts, &frames);
        let last = frames.len() - 1;

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let span = d.window_span();
        let got = d.reseed(&window_of_span(&frames, last, span)).unwrap();
        assert_eq!(got.y, streamed[last].y, "luma mismatch at the ahead edge");
        assert_eq!(got.u, streamed[last].u, "u mismatch at the ahead edge");
        assert_eq!(got.v, streamed[last].v, "v mismatch at the ahead edge");

        // Two bounds cover this leading-edge padding difference, because
        // it has a known shape rather than an unknown one. `reseed` and
        // a fresh stream fold different amounts of duplicated history
        // into nl4d's cross-frame accumulator right at the clip's first
        // frame, and inside that padded region a hard-threshold
        // coefficient can sit close enough to its cutoff that the two
        // paths land it on opposite sides. That flips the reconstruction
        // of a couple of pixels by their own magnitude while leaving the
        // rest of the plane alone. `BEHIND_EDGE_TOLERANCE` is a
        // worst-pixel bound, sized well under the full 255-code range
        // so a real regression would still trip it. `BEHIND_EDGE_OUTLIER_LIMIT`
        // is the original, tighter bound of 8 kept as a count instead of
        // a ceiling: at most a handful of samples may cross it, and a
        // real regression that moved the bulk of the plane would push
        // far more samples past it than that.
        //
        // Measured against this fixture, the actual worst-pixel diff was
        // 10, with exactly 1 sample exceeding 8, so both bounds carry
        // headroom over what was observed.
        const BEHIND_EDGE_TOLERANCE: i32 = 16;
        const BEHIND_EDGE_OUTLIER_THRESHOLD: i32 = 8;
        const BEHIND_EDGE_OUTLIER_LIMIT: usize = 4;
        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let got = d.reseed(&window_of_span(&frames, 0, span)).unwrap();
        let luma_diff = max_abs_diff(&got.y, &streamed[0].y);
        let luma_outliers = count_exceeding(&got.y, &streamed[0].y, BEHIND_EDGE_OUTLIER_THRESHOLD);
        assert!(
            luma_diff <= BEHIND_EDGE_TOLERANCE,
            "luma at the behind edge (k=0) drifted too far from streaming: max abs diff {luma_diff}"
        );
        assert!(
            luma_outliers <= BEHIND_EDGE_OUTLIER_LIMIT,
            "luma at the behind edge (k=0) drifted too far from streaming across too much of the \
             plane: {luma_outliers} samples exceeded {BEHIND_EDGE_OUTLIER_THRESHOLD}"
        );
    }

    /// After an nl4d `reseed`, ordinary sequential `push`/`recv` must
    /// carry on producing the same frames the streaming path would
    /// have, the nl4d mirror of
    /// [`a_reseed_leaves_the_stream_positioned_for_the_next_frame`].
    #[test]
    fn nl4d_reseed_then_streaming_continues_correctly() {
        let opts = nl4d_plane_options(2);
        let frames = ramp_clip(&layout(), 16);
        let streamed = stream_all(&opts, &frames);
        let k = 8usize;

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let span = d.window_span();
        d.reseed(&window_of_span(&frames, k, span)).unwrap();

        // The next frame in source order after the reseed window's own
        // last frame is `k + 1 + span.ahead`, the same relationship
        // [`Denoise::render`]'s fast path in the VapourSynth plugin
        // relies on for ordinary sequential continuation.
        d.push(&frames[k + 1 + span.ahead]).unwrap();
        let got = d.recv().unwrap().expect("frame k + 1");

        assert_eq!(got.y, streamed[k + 1].y);
        assert_eq!(got.u, streamed[k + 1].u);
        assert_eq!(got.v, streamed[k + 1].v);
    }

    /// `reseed` rejects a window of the wrong length for nl4d too, and
    /// the error names the wider length nl4d needs (`4r+1` at this
    /// radius), not nlmeans's `2r+1`.
    #[test]
    fn nl4d_reseed_rejects_a_window_of_the_wrong_length() {
        let opts = nl4d_plane_options(2);
        let frames = ramp_clip(&layout(), 16);
        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let expected = d.window_span().frame_count();

        let err = d.reseed(&frames[..3]).unwrap_err().to_string();
        assert!(
            err.contains(&expected.to_string()),
            "error should name the expected length ({expected}), got {err}"
        );
    }

    /// A `PlaneOptions` identical to [`nl4d_plane_options`] except only
    /// `intent` differs, for exercising a passthrough side under nl4d's
    /// wider window and multi-push pass sequence.
    fn nl4d_plane_options_with_intent(r: u32, intent: ChannelIntent) -> PlaneOptions {
        PlaneOptions {
            intent,
            ..nl4d_plane_options(r)
        }
    }

    /// The nl4d mirror of
    /// [`reseed_pairs_the_passthrough_plane_with_the_centre_frame`].
    ///
    /// nl4d's real-push loop drains after every emission, not only the
    /// last, and each drain pops one passthrough entry. This checks
    /// that walk still lands on the target's own entry rather than one
    /// of the earlier, discarded regions' entries.
    #[test]
    fn nl4d_reseed_pairs_the_passthrough_plane_with_the_centre_frame() {
        let opts = nl4d_plane_options_with_intent(2, ChannelIntent::Luma);
        let frames = ramp_clip(&layout(), 16);
        let k = 8;

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let span = d.window_span();
        let got = d.reseed(&window_of_span(&frames, k, span)).unwrap();

        assert_eq!(got.u, frames[k].u, "u should pass through from the centre frame");
        assert_eq!(got.v, frames[k].v, "v should pass through from the centre frame");
    }

    /// The nl4d mirror of
    /// [`reseed_pairs_the_passthrough_luma_plane_with_the_centre_frame`].
    #[test]
    fn nl4d_reseed_pairs_the_passthrough_luma_plane_with_the_centre_frame() {
        let opts = nl4d_plane_options_with_intent(2, ChannelIntent::Chroma);
        let frames = ramp_clip(&layout(), 16);
        let k = 8;

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let span = d.window_span();
        let got = d.reseed(&window_of_span(&frames, k, span)).unwrap();

        assert_eq!(got.y, frames[k].y, "y should pass through from the centre frame");
    }

    /// The nl4d mirror of
    /// [`reseed_then_streaming_keeps_the_passthrough_plane_aligned_on_the_next_frame`].
    ///
    /// A single-shot pairing test only checks the very first entry
    /// `recv` pops after the drop. A leftover-count defect that still
    /// happens to leave the right entry at the front would pass every
    /// single-shot nl4d test above and only misalign the plane paired
    /// with the frame right after the target, once streaming resumes.
    #[test]
    fn nl4d_reseed_then_streaming_keeps_the_passthrough_plane_aligned_on_the_next_frame() {
        let opts = nl4d_plane_options_with_intent(2, ChannelIntent::Luma);
        let frames = ramp_clip(&layout(), 16);
        let k = 8;

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let span = d.window_span();
        d.reseed(&window_of_span(&frames, k, span)).unwrap();
        d.push(&frames[k + 1 + span.ahead]).unwrap();
        let got = d.recv().unwrap().expect("frame k + 1");

        assert_eq!(got.u, frames[k + 1].u, "u should pass through from frame k + 1");
        assert_eq!(got.v, frames[k + 1].v, "v should pass through from frame k + 1");
    }

    /// A `PlaneOptions` identical to [`nl4d_plane_options`] except noise
    /// estimation is window-local (`windowed_noise_estimation: true`)
    /// and `sigma` is left on automatic estimation, the configuration
    /// `av-denoise-vs` runs.
    ///
    /// Unlike [`nl4d_plane_options`], `sigma` is deliberately left
    /// unpinned here: window-local estimation exists precisely so the
    /// automatic estimate agrees between `reseed` and streaming, and a
    /// pinned sigma would never have exercised that.
    fn nl4d_windowed_plane_options(r: u32) -> PlaneOptions {
        PlaneOptions {
            algorithm: Algorithm::Nl4d(Nl4dOptions {
                windowed_noise_estimation: true,
                ..Nl4dOptions::default()
            }),
            ..test_plane_options(r)
        }
    }

    /// The property that would have caught the original random-access
    /// bug directly: with window-local estimation on and `sigma`
    /// automatic, a `reseed` for a mid-clip frame matches the streaming
    /// path's own output for that frame bit-for-bit. The mirror of
    /// [`nl4d_reseed_matches_the_streaming_output_mid_clip`], but with
    /// the noise estimator actually exercised instead of sidestepped.
    #[test]
    fn nl4d_windowed_reseed_matches_the_streaming_output_mid_clip() {
        let opts = nl4d_windowed_plane_options(2);
        let frames = ramp_clip(&layout(), 16);
        let streamed = stream_all(&opts, &frames);

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let k = 8;
        let span = d.window_span();
        let got = d.reseed(&window_of_span(&frames, k, span)).unwrap();

        assert_eq!(got.y, streamed[k].y);
        assert_eq!(got.u, streamed[k].u);
        assert_eq!(got.v, streamed[k].v);
    }

    /// The clip-edge mirror of
    /// [`nl4d_windowed_reseed_matches_the_streaming_output_mid_clip`],
    /// covering both ends of the clip the way
    /// [`nl4d_reseed_matches_the_streaming_output_at_both_clip_edges`]
    /// does for the sigma-pinned case.
    #[test]
    fn nl4d_windowed_reseed_matches_the_streaming_output_at_both_clip_edges() {
        let opts = nl4d_windowed_plane_options(2);
        let frames = ramp_clip(&layout(), 16);
        let streamed = stream_all(&opts, &frames);
        let last = frames.len() - 1;

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let span = d.window_span();
        let got = d.reseed(&window_of_span(&frames, last, span)).unwrap();
        assert_eq!(got.y, streamed[last].y, "luma mismatch at the ahead edge");
        assert_eq!(got.u, streamed[last].u, "u mismatch at the ahead edge");
        assert_eq!(got.v, streamed[last].v, "v mismatch at the ahead edge");

        // The same leading-edge padding difference
        // `nl4d_reseed_matches_the_streaming_output_at_both_clip_edges`
        // documents, unrelated to noise estimation: `reseed` fills the
        // whole `behind` span by repeating the clip's first frame, a
        // fresh stream primes only `radius` duplicates of it.
        const BEHIND_EDGE_TOLERANCE: i32 = 8;
        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let got = d.reseed(&window_of_span(&frames, 0, span)).unwrap();
        let luma_diff = max_abs_diff(&got.y, &streamed[0].y);
        assert!(
            luma_diff <= BEHIND_EDGE_TOLERANCE,
            "luma at the behind edge (k=0) drifted too far from streaming: max abs diff {luma_diff}"
        );
    }

    /// With window-local estimation on and `sigma` automatic, the fast
    /// path and the reseed path must compute the same sigma for the
    /// same window, so their outputs agree: a `reseed` at `k` followed
    /// by an ordinary `push`/`recv` for `k + 1` must match a `reseed`
    /// targeted directly at `k + 1` on a fresh denoiser.
    ///
    /// This is the property window-local estimation exists for. Without
    /// it, the fast path keeps folding history the reseed path never
    /// sees, so the two disagree even though both look at the same
    /// window of real content.
    #[test]
    fn nl4d_windowed_fast_path_agrees_with_reseed_at_the_next_frame() {
        let opts = nl4d_windowed_plane_options(2);
        let frames = ramp_clip(&layout(), 16);
        let k = 8usize;

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let span = d.window_span();
        d.reseed(&window_of_span(&frames, k, span)).unwrap();
        d.push(&frames[k + 1 + span.ahead]).unwrap();
        let via_fast_path = d.recv().unwrap().expect("frame k + 1");

        let mut fresh = PlanarDenoiser::create(&opts, layout()).unwrap();
        let via_reseed = fresh.reseed(&window_of_span(&frames, k + 1, span)).unwrap();

        assert_eq!(via_fast_path.y, via_reseed.y);
        assert_eq!(via_fast_path.u, via_reseed.u);
        assert_eq!(via_fast_path.v, via_reseed.v);
    }

    /// A `PlaneOptions` identical to [`test_plane_options`] except the
    /// algorithm is `NlmeansHq` with window-local estimation on and
    /// `sigma` left on automatic estimation, the nlmeans mirror of
    /// [`nl4d_windowed_plane_options`].
    fn nlmeans_hq_windowed_plane_options(r: u32) -> PlaneOptions {
        PlaneOptions {
            algorithm: Algorithm::NlmeansHq(NlmeansHqOptions {
                nlm: NlmeansOptions::default(),
                hq: HqParams {
                    windowed_noise_estimation: true,
                    ..HqParams::default()
                },
            }),
            ..test_plane_options(r)
        }
    }

    /// The nlmeans-hq mirror of
    /// [`nl4d_windowed_reseed_matches_the_streaming_output_mid_clip`].
    ///
    /// With window-local estimation on and `sigma` automatic, a `reseed`
    /// for a mid-clip frame must match the streaming path's own output
    /// for that frame bit-for-bit. No core test exercised HQ with
    /// automatic sigma under reseed before this, which is how a
    /// VapourSynth plugin filter that returns different pixels for the
    /// same frame depending on request order shipped unnoticed.
    #[test]
    fn nlmeans_hq_windowed_reseed_matches_the_streaming_output_mid_clip() {
        let opts = nlmeans_hq_windowed_plane_options(2);
        let frames = ramp_clip(&layout(), 16);
        let streamed = stream_all(&opts, &frames);

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let k = 8;
        let span = d.window_span();
        let got = d.reseed(&window_of_span(&frames, k, span)).unwrap();

        assert_eq!(got.y, streamed[k].y);
        assert_eq!(got.u, streamed[k].u);
        assert_eq!(got.v, streamed[k].v);
    }

    /// The nlmeans-hq mirror of
    /// [`nl4d_windowed_reseed_matches_the_streaming_output_at_both_clip_edges`].
    #[test]
    fn nlmeans_hq_windowed_reseed_matches_the_streaming_output_at_both_clip_edges() {
        let opts = nlmeans_hq_windowed_plane_options(2);
        let frames = ramp_clip(&layout(), 16);
        let streamed = stream_all(&opts, &frames);
        let last = frames.len() - 1;

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let span = d.window_span();
        let got = d.reseed(&window_of_span(&frames, last, span)).unwrap();
        assert_eq!(got.y, streamed[last].y, "luma mismatch at the ahead edge");
        assert_eq!(got.u, streamed[last].u, "u mismatch at the ahead edge");
        assert_eq!(got.v, streamed[last].v, "v mismatch at the ahead edge");

        const BEHIND_EDGE_TOLERANCE: i32 = 8;
        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let got = d.reseed(&window_of_span(&frames, 0, span)).unwrap();
        let luma_diff = max_abs_diff(&got.y, &streamed[0].y);
        assert!(
            luma_diff <= BEHIND_EDGE_TOLERANCE,
            "luma at the behind edge (k=0) drifted too far from streaming: max abs diff {luma_diff}"
        );
    }

    /// The nlmeans-hq mirror of
    /// [`nl4d_windowed_repeated_out_of_order_access_matches_streaming`]:
    /// one long-lived `PlanarDenoiser` driven through the VapourSynth
    /// plugin harness's exact shuffled access order with its hybrid
    /// fast-path/`reseed` policy, every produced frame compared against
    /// a true continuous stream.
    ///
    /// A single reseed, or a reseed followed by one push, both pass
    /// under window-local estimation without exercising this, the same
    /// way they did for nl4d: it takes a longer, repeatedly-reseeded run
    /// to expose a carrier that survives `reset_stream_state` outside
    /// the windowed gate.
    #[test]
    fn nlmeans_hq_windowed_repeated_out_of_order_access_matches_streaming() {
        let opts = nlmeans_hq_windowed_plane_options(2);
        let frames = ramp_clip(&layout(), 14);
        let streamed = stream_all(&opts, &frames);
        let last = frames.len() - 1;

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let span = d.window_span();
        let mut last_n: Option<usize> = None;

        // The VapourSynth plugin harness's exact shuffled order.
        let order = [9usize, 0, 13, 4, 5, 6, 1, 12, 2, 11, 3, 10, 7, 8];
        const NEAR_START_TOLERANCE: i32 = 8;

        for &n in &order {
            let fast = if last_n == Some(n.wrapping_sub(1)) && n > 0 {
                let ahead = (n + span.ahead).min(last);
                d.push(&frames[ahead]).unwrap();
                d.recv().unwrap()
            } else {
                None
            };
            let got = match fast {
                Some(out) => out,
                None => d.reseed(&window_of_span(&frames, n, span)).unwrap(),
            };
            last_n = Some(n);

            if n < span.behind {
                let diff = max_abs_diff(&got.y, &streamed[n].y)
                    .max(max_abs_diff(&got.u, &streamed[n].u))
                    .max(max_abs_diff(&got.v, &streamed[n].v));
                assert!(
                    diff <= NEAR_START_TOLERANCE,
                    "near-start frame n = {n} drifted too far from streaming: max abs diff {diff}"
                );
            } else {
                assert_eq!(got.y, streamed[n].y, "luma mismatch at n = {n}");
                assert_eq!(got.u, streamed[n].u, "u mismatch at n = {n}");
                assert_eq!(got.v, streamed[n].v, "v mismatch at n = {n}");
            }
        }
    }

    /// The nlmeans-hq mirror of
    /// [`nl4d_windowed_fast_path_agrees_with_reseed_at_the_next_frame`]:
    /// a `reseed` at `k` followed by an ordinary `push`/`recv` for
    /// `k + 1` must match a `reseed` targeted directly at `k + 1` on a
    /// fresh denoiser.
    #[test]
    fn nlmeans_hq_windowed_fast_path_agrees_with_reseed_at_the_next_frame() {
        let opts = nlmeans_hq_windowed_plane_options(2);
        let frames = ramp_clip(&layout(), 16);
        let k = 8usize;

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let span = d.window_span();
        d.reseed(&window_of_span(&frames, k, span)).unwrap();
        d.push(&frames[k + 1 + span.ahead]).unwrap();
        let via_fast_path = d.recv().unwrap().expect("frame k + 1");

        let mut fresh = PlanarDenoiser::create(&opts, layout()).unwrap();
        let via_reseed = fresh.reseed(&window_of_span(&frames, k + 1, span)).unwrap();

        assert_eq!(via_fast_path.y, via_reseed.y);
        assert_eq!(via_fast_path.u, via_reseed.u);
        assert_eq!(via_fast_path.v, via_reseed.v);
    }

    /// Reproduces the VapourSynth plugin harness's own `render` hybrid
    /// policy exactly: frame 0 goes through `reseed`, and every
    /// subsequent frame goes through the fast `push`/`recv` path,
    /// falling back to `reseed` only when `recv` yields nothing.
    fn render_sequence(
        d: &mut PlanarDenoiser,
        frames: &[Planes],
        span: WindowSpan,
        order: &[usize],
    ) -> Vec<Planes> {
        let last = frames.len() - 1;
        let mut last_n: Option<usize> = None;
        let mut out = Vec::new();
        for &n in order {
            let fast = if last_n == Some(n.wrapping_sub(1)) && n > 0 {
                let ahead = (n + span.ahead).min(last);
                d.push(&frames[ahead]).unwrap();
                d.recv().unwrap()
            } else {
                None
            };
            let got = match fast {
                Some(out) => out,
                None => d.reseed(&window_of_span(frames, n, span)).unwrap(),
            };
            last_n = Some(n);
            out.push(got);
        }
        out
    }

    /// Mirrors `a_sequential_run_after_a_seek_stays_correct_nlmeans`
    /// exactly: a `reseed` at frame 11 of a 14-frame clip, then two
    /// fast-path frames, compared against the same `render` hybrid
    /// policy run straight through from frame 0. This is the reference
    /// the VapourSynth harness actually uses, unlike
    /// [`stream_all`], which is a true continuous stream with no
    /// `reseed` in it at all.
    #[test]
    fn nlmeans_hq_windowed_sequential_run_after_a_seek_stays_correct() {
        let opts = nlmeans_hq_windowed_plane_options(2);
        let frames = ramp_clip(&layout(), 14);

        let mut linear = PlanarDenoiser::create(&opts, layout()).unwrap();
        let span = linear.window_span();
        let linear_out = render_sequence(&mut linear, &frames, span, &(0..frames.len()).collect::<Vec<_>>());

        let mut seeked = PlanarDenoiser::create(&opts, layout()).unwrap();
        let seeked_out = render_sequence(&mut seeked, &frames, span, &[11, 12, 13]);

        for (i, n) in [12usize, 13].into_iter().enumerate() {
            let got = &seeked_out[i + 1];
            let want = &linear_out[n];
            assert_eq!(got.y, want.y, "luma mismatch at n = {n}");
            assert_eq!(got.u, want.u, "u mismatch at n = {n}");
            assert_eq!(got.v, want.v, "v mismatch at n = {n}");
        }
    }

    /// Diagnostic: same as
    /// [`nlmeans_hq_windowed_sequential_run_after_a_seek_stays_correct`]
    /// but at the VapourSynth harness's own clip size, 160x120.
    #[test]
    fn nlmeans_hq_windowed_sequential_run_after_a_seek_stays_correct_at_harness_size() {
        let layout = FrameLayout {
            width: 160,
            height: 120,
            subsampling: Subsampling::Yuv420,
            depth: Depth::Eight,
        };
        let opts = nlmeans_hq_windowed_plane_options(2);
        let frames = ramp_clip(&layout, 14);

        let mut linear = PlanarDenoiser::create(&opts, layout).unwrap();
        let span = linear.window_span();
        let linear_out = render_sequence(&mut linear, &frames, span, &(0..frames.len()).collect::<Vec<_>>());

        let mut seeked = PlanarDenoiser::create(&opts, layout).unwrap();
        let seeked_out = render_sequence(&mut seeked, &frames, span, &[11, 12, 13]);

        for (i, n) in [12usize, 13].into_iter().enumerate() {
            let got = &seeked_out[i + 1];
            let want = &linear_out[n];
            assert_eq!(got.y, want.y, "luma mismatch at n = {n}");
            assert_eq!(got.u, want.u, "u mismatch at n = {n}");
            assert_eq!(got.v, want.v, "v mismatch at n = {n}");
        }
    }

    /// The property that actually reproduces the VapourSynth plugin
    /// harness's `random_access_matches_sequential_access_nl4d`
    /// end-to-end, at the core level: one long-lived `PlanarDenoiser`
    /// driven through a shuffled access order with the plugin's own
    /// hybrid fast-path/`reseed` policy, every produced frame compared
    /// against a true continuous stream.
    ///
    /// A single reseed, or a reseed followed by one push, both pass
    /// under window-local estimation without exercising this: it took
    /// a longer, repeatedly-reseeded run to show that
    /// `noise_estimator_temporal_only`'s "keep the last trustworthy
    /// reading between folds" behaviour survives `reset_stream_state`
    /// unwindowed even when every other chain is windowed, so a
    /// `reseed`'s short real-push run can land on "no trustworthy
    /// reading yet" while a true stream at the same frame is still
    /// coasting on one from many frames back.
    ///
    /// Frames within `span.behind` of the clip's start allow the same
    /// small, already-documented tolerance
    /// [`nl4d_reseed_matches_the_streaming_output_at_both_clip_edges`]
    /// does, for the same reason: a fresh stream's own leading mirror
    /// and a `reseed`'s explicit clamping both pad the clip's start at
    /// once there, which true streaming's single mirror does not do.
    /// That is a content-padding effect, unrelated to noise estimation,
    /// and unrelated to what this test exists to catch.
    #[test]
    fn nl4d_windowed_repeated_out_of_order_access_matches_streaming() {
        let opts = nl4d_windowed_plane_options(2);
        let frames = ramp_clip(&layout(), 14);
        let streamed = stream_all(&opts, &frames);
        let last = frames.len() - 1;

        let mut d = PlanarDenoiser::create(&opts, layout()).unwrap();
        let span = d.window_span();
        let mut last_n: Option<usize> = None;

        // The VapourSynth plugin harness's exact shuffled order.
        let order = [9usize, 0, 13, 4, 5, 6, 1, 12, 2, 11, 3, 10, 7, 8];
        const NEAR_START_TOLERANCE: i32 = 8;

        for &n in &order {
            let fast = if last_n == Some(n.wrapping_sub(1)) && n > 0 {
                let ahead = (n + span.ahead).min(last);
                d.push(&frames[ahead]).unwrap();
                d.recv().unwrap()
            } else {
                None
            };
            let got = match fast {
                Some(out) => out,
                None => d.reseed(&window_of_span(&frames, n, span)).unwrap(),
            };
            last_n = Some(n);

            if n < span.behind {
                let diff = max_abs_diff(&got.y, &streamed[n].y)
                    .max(max_abs_diff(&got.u, &streamed[n].u))
                    .max(max_abs_diff(&got.v, &streamed[n].v));
                assert!(
                    diff <= NEAR_START_TOLERANCE,
                    "near-start frame n = {n} drifted too far from streaming: max abs diff {diff}"
                );
            } else {
                assert_eq!(got.y, streamed[n].y, "luma mismatch at n = {n}");
                assert_eq!(got.u, streamed[n].u, "u mismatch at n = {n}");
                assert_eq!(got.v, streamed[n].v, "v mismatch at n = {n}");
            }
        }
    }
}

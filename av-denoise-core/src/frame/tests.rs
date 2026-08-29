//! Tests for [`super::PlanarDenoiser::reseed`].

use super::*;

// Feature-gated because `test_plane_options` names the `Vulkan`
// accelerator variant, which only exists when the `vulkan` feature is
// enabled.
#[cfg(feature = "vulkan")]
mod reseed {
    use super::*;
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
}

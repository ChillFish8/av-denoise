use std::collections::VecDeque;

use super::{PlanarDenoiser, Planes};
use crate::EdgePadding;

/// An explicit window of source frames for [PlanarDenoiser::reseed_window].
#[derive(Debug)]
pub struct ReseedWindow<'a> {
    /// The window's frames in source order, with no repeats.
    pub frames: &'a [Planes],
    /// The target frame's index within `frames`.
    pub target: usize,
    /// Whether `frames[0]` is the clip's first frame.
    pub at_clip_start: bool,
    /// Whether the last frame is the clip's last frame.
    pub at_clip_end: bool,
}

/// Pops up to `count` entries off the front of `queue`, discarding them.
fn drop_leading<T>(queue: &mut VecDeque<T>, count: usize) {
    for _ in 0..count.min(queue.len()) {
        queue.pop_front();
    }
}

impl PlanarDenoiser {
    /// Denoises the target frame of a mid-clip window of exactly
    /// [Self::window_span] frames, oldest first.
    ///
    /// This abandons whatever stream was running and starts a new one
    /// from the window, keeping every GPU allocation. When it returns,
    /// the stream sits where it would be had the window been pushed
    /// frame by frame, so the caller can carry on with [Self::push] and
    /// [Self::recv] for the frame after the target.
    ///
    /// For the NLM algorithms, callers clamp the window's indices at a
    /// clip's ends, matching how their streams repeat the first and last
    /// frames. For nl4d, a window whose first or last frame is a clip end
    /// goes through [Self::reseed_window] instead.
    ///
    /// # Why nl4d's window is wider
    ///
    /// nl4d scatters every pass across the `2r+1` frames it reaches. A
    /// target's region is complete once the pass centred `r` frames
    /// ahead of it has run, and it first collects from the pass centred
    /// `r` frames behind it, which needs `r` more frames behind again.
    /// That doubles the span on both sides.
    pub fn reseed(&mut self, window: &[Planes]) -> Result<Planes, anyhow::Error> {
        let span = self.window_span();
        let expected = span.frame_count();
        if window.len() != expected {
            anyhow::bail!("reseed needs a window of {expected} frames, got {}", window.len());
        }

        if span.edges == EdgePadding::Shifted {
            let request = ReseedWindow {
                frames: window,
                target: span.behind,
                at_clip_start: false,
                at_clip_end: false,
            };
            let mut outputs = self.reseed_window(request)?;
            let target_output = outputs.pop();
            return target_output.ok_or_else(|| anyhow::anyhow!("reseed produced no frame, this is a bug"));
        }

        self.luma_passthrough.clear();
        self.chroma_passthrough.clear();
        self.reset_streams();

        // Prime the first `2 * radius` frames, filling the underlying
        // denoiser's window without submitting anything, as streaming
        // would have primed it. The single real push that follows emits
        // the target.
        let radius = self.temporal_radius as usize;
        let (head, tail) = window.split_at(2 * radius);
        for planes in head {
            self.push_priming(planes)?;
        }

        // Priming queues one passthrough entry per frame. Dropping the
        // leading `radius` puts the target's own entry at the front.
        drop_leading(&mut self.luma_passthrough, radius);
        drop_leading(&mut self.chroma_passthrough, radius);

        let mut result = None;
        for planes in tail {
            self.push(planes)?;
            if let Some(denoised) = self.recv()? {
                result = Some(denoised);
            }
        }

        result.ok_or_else(|| anyhow::anyhow!("a full window produced no frame, this is a bug"))
    }

    /// Rebuilds the stream from an explicit window that stops at the
    /// clip's ends.
    ///
    /// It returns the target's output, plus every later frame's output
    /// when the window ends at the clip's end. A window that starts at
    /// the clip's start runs every frame as a real push, so the head
    /// passes match a stream. A window that ends at the clip's end
    /// flushes, so the tail passes match a stream.
    pub fn reseed_window(&mut self, window: ReseedWindow<'_>) -> Result<Vec<Planes>, anyhow::Error> {
        let span = self.window_span();
        if span.edges != EdgePadding::Shifted {
            anyhow::bail!("reseed_window needs an algorithm with shifted edges, use reseed");
        }

        let radius = self.temporal_radius as usize;
        let frames = window.frames;
        if window.target >= frames.len() {
            anyhow::bail!(
                "reseed target {} is outside a window of {}",
                window.target,
                frames.len()
            );
        }

        let frames_ahead = frames.len() - 1 - window.target;
        if window.target > span.behind || frames_ahead > span.ahead {
            anyhow::bail!("reseed window is longer than the span {span:?}");
        }

        let expected_ahead = if window.at_clip_end {
            frames_ahead
        } else {
            span.ahead
        };
        let expected_behind = if window.at_clip_start {
            window.target
        } else {
            span.behind
        };
        let expected_len = expected_behind + 1 + expected_ahead;
        if window.target != expected_behind || frames.len() != expected_len {
            anyhow::bail!("reseed window does not match the span {span:?} for its edge flags");
        }

        self.luma_passthrough.clear();
        self.chroma_passthrough.clear();
        self.reset_streams();

        let real_frames = if window.at_clip_start {
            frames
        } else {
            let (priming, real) = frames.split_at(2 * radius);
            for planes in priming {
                self.push_priming(planes)?;
            }

            drop_leading(&mut self.luma_passthrough, radius);
            drop_leading(&mut self.chroma_passthrough, radius);
            real
        };

        let mut outputs = Vec::new();
        for planes in real_frames {
            self.push(planes)?;
            if let Some(denoised) = self.recv()? {
                outputs.push(denoised);
            }
        }

        if !window.at_clip_end {
            let target_output = outputs.pop();
            let target_output = target_output
                .ok_or_else(|| anyhow::anyhow!("a full window produced no frame, this is a bug"))?;
            let target_only = vec![target_output];
            return Ok(target_only);
        }

        let tail_count = frames.len().min(2 * radius);
        let tail_start = frames.len() - tail_count;
        self.replace_passthrough_with(&frames[tail_start..]);
        self.flush(|planes| outputs.push(planes))?;

        let from_end = frames.len() - window.target;
        let Some(first) = outputs.len().checked_sub(from_end) else {
            anyhow::bail!("a flushed window produced too few frames, this is a bug");
        };

        let target_onwards = outputs.split_off(first);
        Ok(target_onwards)
    }

    /// Abandons the running stream on every enabled half.
    fn reset_streams(&mut self) {
        let halves = [self.yuv.as_mut(), self.luma.as_mut(), self.chroma.as_mut()];
        for denoiser in halves.into_iter().flatten() {
            denoiser.reset_stream();
        }
    }

    /// Resets the passthrough queues to the source planes of the frames
    /// a flush is about to emit.
    fn replace_passthrough_with(&mut self, frames: &[Planes]) {
        self.luma_passthrough.clear();
        self.chroma_passthrough.clear();

        for planes in frames {
            if self.luma.is_none() {
                self.luma_passthrough.push_back(planes.y.clone());
            }

            if self.chroma.is_none() {
                let chroma_pair = (planes.u.clone(), planes.v.clone());
                self.chroma_passthrough.push_back(chroma_pair);
            }
        }
    }
}

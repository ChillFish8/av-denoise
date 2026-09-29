use std::collections::VecDeque;
use std::sync::Arc;

use av_decoders::VideoDetails;
use av_scenechange::{DetectionOptions, SceneChangeDetector};
use v_frame::frame::Frame;
use v_frame::pixel::Pixel;

/// Frames past the one being judged that the detector looks at.
pub const LOOKAHEAD_DISTANCE: usize = 5;

/// A frame whose scene has been decided.
pub struct Decided<T: Pixel> {
    pub frame: Arc<Frame<T>>,
    pub starts_scene: bool,
}

/// Decides scene cuts one frame at a time, a fixed window behind the newest frame.
///
/// Frame `n` is judged once frames `n - 1` through `n + LOOKAHEAD_DISTANCE` are held, or when
/// [SceneSplitter::finish] runs out of input. Frames are released in push order.
pub struct SceneSplitter<T: Pixel> {
    detector: SceneChangeDetector<T>,
    /// Frames from `queue_start` onwards, oldest first.
    queue: VecDeque<Arc<Frame<T>>>,
    queue_start: usize,
    next_frameno: usize,
    last_keyframe: usize,
}

impl<T: Pixel> SceneSplitter<T> {
    pub fn new(details: &VideoDetails) -> Self {
        let options = DetectionOptions::default();
        let resolution = (details.width, details.height);
        let frame_duration = details.frame_rate.recip();
        let min_scenecut_distance = options.min_scenecut_distance.unwrap_or(0);
        let max_scenecut_distance = options.max_scenecut_distance.unwrap_or(u32::MAX as usize);

        let detector = SceneChangeDetector::new(
            resolution,
            details.bit_depth,
            frame_duration,
            details.chroma_sampling,
            options.lookahead_distance,
            options.analysis_speed,
            min_scenecut_distance,
            max_scenecut_distance,
        );

        Self {
            detector,
            queue: VecDeque::with_capacity(LOOKAHEAD_DISTANCE + 2),
            queue_start: 0,
            next_frameno: 0,
            last_keyframe: 0,
        }
    }

    pub fn push(&mut self, frame: Arc<Frame<T>>) -> Vec<Decided<T>> {
        self.queue.push_back(frame);

        let mut released = Vec::new();

        while self.newest_frameno() >= self.next_frameno + LOOKAHEAD_DISTANCE {
            let decided = self.judge_next();
            released.push(decided);
        }

        released
    }

    /// Judges and releases every frame still held.
    pub fn finish(&mut self) -> Vec<Decided<T>> {
        let mut released = Vec::new();

        while !self.queue.is_empty() && self.newest_frameno() >= self.next_frameno {
            let decided = self.judge_next();
            released.push(decided);
        }

        self.queue.clear();
        released
    }

    fn newest_frameno(&self) -> usize {
        self.queue_start + self.queue.len() - 1
    }

    /// Judges `next_frameno`, then drops the frame before it, which no later judgement reads.
    fn judge_next(&mut self) -> Decided<T> {
        let frameno = self.next_frameno;
        let starts_scene = frameno == 0 || self.detect_cut(frameno);

        if starts_scene {
            self.last_keyframe = frameno;
        }

        let position = frameno - self.queue_start;
        let frame = Arc::clone(&self.queue[position]);

        if frameno > 0 {
            self.queue.pop_front();
            self.queue_start += 1;
        }

        self.next_frameno += 1;

        Decided { frame, starts_scene }
    }

    fn detect_cut(&mut self, frameno: usize) -> bool {
        let frame_set: Vec<&Arc<Frame<T>>> = self.queue.iter().take(LOOKAHEAD_DISTANCE + 2).collect();
        let (cut, _score) = self
            .detector
            .analyze_next_frame(&frame_set, frameno, self.last_keyframe);

        cut
    }
}

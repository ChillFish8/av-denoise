use std::collections::BTreeMap;
use std::ops::{Range, RangeInclusive};

use av_denoise::{EdgePadding, WindowSpan};

/// The smallest lookback a filter runs with, whatever the core's thread count.
pub const MIN_LOOKBACK: usize = 16;

/// How many outputs a request may sit ahead of the stream and still catch it up rather than rebuild it.
///
/// VapourSynth keeps about one request in flight per worker thread, so twice that covers a full burst
/// of out-of-order requests.
pub fn lookback_for(threads: usize) -> usize {
    let lookback = threads * 2;
    lookback.max(MIN_LOOKBACK)
}

/// The work that produces one output frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Push this source frame.
    Push(usize),
    /// Flush the stream, which produces every remaining output and ends it.
    Flush,
}

/// What a request has to do to obtain its output frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Plan {
    /// The output is already buffered.
    Serve,
    /// Run these steps on the live stream, then receive up to the output.
    Advance { steps: Range<usize> },
    /// Rebuild the stream so it has produced output `start`, then run these steps.
    Reseed { start: usize, steps: Range<usize> },
}

/// The fixed shape of a filter's stream.
#[derive(Debug, Clone, Copy)]
pub struct StreamGeometry {
    pub span: WindowSpan,
    pub lookback: usize,
    pub lead: usize,
    pub last_frame: usize,
}

impl StreamGeometry {
    /// The source frames a request for output `n` asks VapourSynth for.
    pub fn request_range(&self, n: usize) -> RangeInclusive<usize> {
        let first = n.saturating_sub(self.span.behind + self.lookback);
        let last = (n + self.span.ahead + self.lead).min(self.last_frame);
        first..=last
    }

    /// The work that produces output `k`.
    pub fn step(&self, k: usize) -> Step {
        let source = k + self.span.ahead;
        match self.span.edges {
            EdgePadding::Repeat => {
                let clamped = source.min(self.last_frame);
                Step::Push(clamped)
            },
            EdgePadding::Shifted if source <= self.last_frame => Step::Push(source),
            EdgePadding::Shifted => Step::Flush,
        }
    }

    /// Whether a rebuild at `start` reaches the clip's end, producing every output from `start` on.
    pub fn reseed_ends_stream(&self, start: usize) -> bool {
        let reaches_end = start + self.span.ahead >= self.last_frame;
        self.span.edges == EdgePadding::Shifted && reaches_end
    }

    /// The most finished outputs the buffer keeps once a request is served.
    pub fn buffer_capacity(&self) -> usize {
        self.lookback + self.span.frame_count() + self.lead
    }

    /// One past the last step a request for output `n` may run.
    ///
    /// Steps stop `lead` outputs past `n`, at the clip's last output, or straight after a flush.
    fn step_end(&self, n: usize) -> usize {
        let lead_end = n + self.lead;
        let end = lead_end.min(self.last_frame) + 1;

        if self.span.edges == EdgePadding::Repeat {
            return end;
        }

        let flush_step = (self.last_frame + 1).saturating_sub(self.span.ahead);
        end.min(flush_step + 1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stream {
    Dead,
    Live { next_step: usize, next_output: usize },
}

/// Decides how each request reaches its output frame, and tracks where the stream is.
///
/// It only holds indices. The caller runs each plan and reports every push, receive and flush as it
/// lands, so the planner always matches the real stream.
#[derive(Debug)]
pub struct StreamPlanner {
    geometry: StreamGeometry,
    stream: Stream,
}

impl StreamPlanner {
    pub fn new(geometry: StreamGeometry) -> Self {
        Self {
            geometry,
            stream: Stream::Dead,
        }
    }

    pub fn geometry(&self) -> StreamGeometry {
        self.geometry
    }

    /// Picks the plan for output `n`.
    ///
    /// A live stream at or before `n` is caught up when its next push is still inside `n`'s request
    /// range. Otherwise the stream is rebuilt at the lowest outstanding request whose own window fits
    /// inside that range, which is never later than `n`.
    pub fn plan(&self, n: usize, buffered: bool, outstanding: &Outstanding) -> Plan {
        if buffered {
            return Plan::Serve;
        }

        let end = self.geometry.step_end(n);

        if let Stream::Live {
            next_step,
            next_output,
        } = self.stream
            && next_output <= n
            && self.in_reach(next_step, n)
        {
            let steps = next_step..end.max(next_step);
            return Plan::Advance { steps };
        }

        let floor = n.saturating_sub(self.geometry.lookback);
        let lowest = outstanding.lowest_in(floor..=n);
        let start = lowest.unwrap_or(n);
        let first_step = start + 1;

        let steps = if self.geometry.reseed_ends_stream(start) {
            first_step..first_step
        } else {
            first_step..end.max(first_step)
        };

        Plan::Reseed { start, steps }
    }

    /// Records a rebuild that produced output `start`, and at the clip's end every later one too.
    pub fn reseeded(&mut self, start: usize) {
        if self.geometry.reseed_ends_stream(start) {
            self.stream = Stream::Dead;
            return;
        }

        let next = start + 1;
        self.stream = Stream::Live {
            next_step: next,
            next_output: next,
        };
    }

    /// Records one push.
    pub fn pushed(&mut self) {
        if let Stream::Live { next_step, .. } = &mut self.stream {
            *next_step += 1;
        }
    }

    /// Records one received output and returns its index, or `None` when nothing is in flight.
    pub fn received(&mut self) -> Option<usize> {
        let Stream::Live {
            next_step,
            next_output,
        } = &mut self.stream
        else {
            return None;
        };

        if *next_output >= *next_step {
            return None;
        }

        let index = *next_output;
        *next_output += 1;
        Some(index)
    }

    /// Records a flush and returns the outputs it produced, in order. The stream ends.
    pub fn flushed(&mut self) -> Range<usize> {
        let end = self.geometry.last_frame + 1;
        let first = match self.stream {
            Stream::Live { next_output, .. } => next_output,
            Stream::Dead => end,
        };

        self.stream = Stream::Dead;
        first..end
    }

    /// Marks the stream dead, so the next request that needs it rebuilds it.
    pub fn kill(&mut self) {
        self.stream = Stream::Dead;
    }

    fn in_reach(&self, next_step: usize, n: usize) -> bool {
        let range = self.geometry.request_range(n);
        match self.geometry.step(next_step) {
            Step::Push(source) => source >= *range.start(),
            Step::Flush => true,
        }
    }
}

/// Output frames VapourSynth has asked for and not yet been given, counted per frame.
#[derive(Debug, Default)]
pub struct Outstanding {
    requests: BTreeMap<usize, usize>,
}

impl Outstanding {
    pub fn register(&mut self, n: usize) {
        let count = self.requests.entry(n).or_insert(0);
        *count += 1;
    }

    pub fn finish(&mut self, n: usize) {
        let Some(count) = self.requests.get_mut(&n) else {
            return;
        };

        *count -= 1;
        if *count == 0 {
            self.requests.remove(&n);
        }
    }

    pub fn lowest_in(&self, range: RangeInclusive<usize>) -> Option<usize> {
        let mut requests = self.requests.range(range);
        requests.next().map(|(&n, _)| n)
    }
}

/// Finished output frames waiting for the request that asks for them.
#[derive(Debug)]
pub struct OutputBuffer<T> {
    frames: BTreeMap<usize, T>,
}

impl<T> Default for OutputBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> OutputBuffer<T> {
    pub fn new() -> Self {
        Self {
            frames: BTreeMap::new(),
        }
    }

    pub fn contains(&self, index: usize) -> bool {
        self.frames.contains_key(&index)
    }

    pub fn insert(&mut self, index: usize, frame: T) {
        self.frames.insert(index, frame);
    }

    pub fn take(&mut self, index: usize) -> Option<T> {
        self.frames.remove(&index)
    }

    /// Drops the lowest indices until at most `capacity` frames remain.
    pub fn evict_to(&mut self, capacity: usize) {
        while self.frames.len() > capacity {
            self.frames.pop_first();
        }
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

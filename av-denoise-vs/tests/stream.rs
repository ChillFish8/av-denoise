use std::ops::Range;

use av_denoise::{EdgePadding, WindowSpan};
use av_denoise_vs::stream::{
    MIN_LOOKBACK,
    OutputBuffer,
    Outstanding,
    Plan,
    Step,
    StreamGeometry,
    StreamPlanner,
    lookback_for,
};

const LOOKBACK: usize = 16;
const LEAD: usize = 2;
const LAST_FRAME: usize = 99;

fn geometry(edges: EdgePadding, behind: usize, ahead: usize, last_frame: usize) -> StreamGeometry {
    let span = WindowSpan { behind, ahead, edges };
    StreamGeometry {
        span,
        lookback: LOOKBACK,
        lead: LEAD,
        last_frame,
    }
}

fn shifted() -> StreamGeometry {
    geometry(EdgePadding::Shifted, 4, 4, LAST_FRAME)
}

fn repeat() -> StreamGeometry {
    geometry(EdgePadding::Repeat, 2, 2, LAST_FRAME)
}

/// Drives a planner the way the filter does, with each output's index standing in for its frame.
struct Harness {
    planner: StreamPlanner,
    outstanding: Outstanding,
    buffer: OutputBuffer<usize>,
    plans: Vec<Plan>,
}

impl Harness {
    fn new(geometry: StreamGeometry) -> Self {
        Self {
            planner: StreamPlanner::new(geometry),
            outstanding: Outstanding::default(),
            buffer: OutputBuffer::new(),
            plans: Vec::new(),
        }
    }

    fn register(&mut self, frames: impl IntoIterator<Item = usize>) {
        for n in frames {
            self.outstanding.register(n);
        }
    }

    fn request(&mut self, n: usize) -> Plan {
        let buffered = self.buffer.contains(n);
        let plan = self.planner.plan(n, buffered, &self.outstanding);

        match &plan {
            Plan::Serve => {},
            Plan::Advance { steps } => {
                self.run_steps(n, steps.clone());
            },
            Plan::Reseed { start, steps } => {
                self.reseed(n, *start);
                self.run_steps(n, steps.clone());
            },
        }

        while !self.buffer.contains(n) {
            let index = self.planner.received().expect("output n should be in flight");
            self.buffer.insert(index, index);
        }

        let served = self.buffer.take(n);
        assert_eq!(served, Some(n));

        let capacity = self.planner.geometry().buffer_capacity();
        self.buffer.evict_to(capacity);
        self.outstanding.finish(n);
        self.plans.push(plan.clone());
        plan
    }

    fn reseed(&mut self, n: usize, start: usize) {
        let geometry = self.planner.geometry();
        let range = geometry.request_range(n);
        let window_first = start.saturating_sub(geometry.span.behind);
        let window_last = (start + geometry.span.ahead).min(geometry.last_frame);
        assert!(
            range.contains(&window_first),
            "reseed window starts outside {range:?}"
        );
        assert!(
            range.contains(&window_last),
            "reseed window ends outside {range:?}"
        );

        self.buffer.insert(start, start);
        if geometry.reseed_ends_stream(start) {
            for index in start + 1..=geometry.last_frame {
                self.buffer.insert(index, index);
            }
        }

        self.planner.reseeded(start);
    }

    fn run_steps(&mut self, n: usize, steps: Range<usize>) {
        let geometry = self.planner.geometry();
        let range = geometry.request_range(n);

        for step in steps {
            match geometry.step(step) {
                Step::Push(source) => {
                    assert!(
                        range.contains(&source),
                        "pushed source {source} outside {range:?}"
                    );
                    self.planner.pushed();
                },
                Step::Flush => {
                    let outputs = self.planner.flushed();
                    for index in outputs {
                        self.buffer.insert(index, index);
                    }
                },
            }
        }
    }

    fn reseeds(&self) -> usize {
        self.plans
            .iter()
            .filter(|plan| matches!(plan, Plan::Reseed { .. }))
            .count()
    }
}

fn render_in_order(harness: &mut Harness, frames: impl IntoIterator<Item = usize>) {
    for n in frames {
        harness.register([n]);
        harness.request(n);
    }
}

#[test]
fn lookback_is_twice_the_thread_count_with_a_floor() {
    assert_eq!(lookback_for(4), MIN_LOOKBACK);
    assert_eq!(lookback_for(32), 64);
}

#[test]
fn request_ranges_clamp_at_both_clip_ends() {
    let geometry = shifted();

    assert_eq!(geometry.request_range(0), 0..=6);
    assert_eq!(geometry.request_range(50), 30..=56);
    assert_eq!(geometry.request_range(99), 79..=99);
}

#[test]
fn shifted_steps_flush_once_the_source_runs_past_the_clip() {
    let geometry = shifted();

    assert_eq!(geometry.step(95), Step::Push(99));
    assert_eq!(geometry.step(96), Step::Flush);
}

#[test]
fn repeat_steps_clamp_their_source_to_the_last_frame() {
    let geometry = repeat();

    assert_eq!(geometry.step(97), Step::Push(99));
    assert_eq!(geometry.step(99), Step::Push(99));
}

#[test]
fn the_first_request_reseeds_at_the_lowest_outstanding_frame() {
    let mut harness = Harness::new(shifted());
    harness.register(0..8);

    let plan = harness.request(5);

    assert_eq!(
        plan,
        Plan::Reseed {
            start: 0,
            steps: 1..8
        }
    );
}

#[test]
fn an_in_order_shifted_render_reseeds_once() {
    let mut harness = Harness::new(shifted());

    render_in_order(&mut harness, 0..=LAST_FRAME);

    assert_eq!(harness.reseeds(), 1);
}

#[test]
fn an_in_order_repeat_render_reseeds_once() {
    let mut harness = Harness::new(repeat());

    render_in_order(&mut harness, 0..=LAST_FRAME);

    assert_eq!(harness.reseeds(), 1);
}

#[test]
fn a_shuffled_burst_within_the_lookback_reseeds_once() {
    let mut harness = Harness::new(shifted());
    harness.register(0..16);

    let order = [9, 0, 13, 4, 5, 6, 1, 12, 2, 11, 3, 10, 7, 8, 15, 14];
    for n in order {
        harness.request(n);
    }

    assert_eq!(harness.reseeds(), 1);
}

#[test]
fn a_burst_after_a_seek_reseeds_once_at_its_lowest_frame() {
    let mut harness = Harness::new(shifted());
    harness.register(50..58);

    let first = harness.request(55);
    assert_eq!(
        first,
        Plan::Reseed {
            start: 50,
            steps: 51..58
        }
    );

    for n in [52, 50, 57, 51, 53, 56, 54] {
        harness.request(n);
    }

    assert_eq!(harness.reseeds(), 1);
}

#[test]
fn a_single_scrub_reseeds_at_the_requested_frame() {
    let mut harness = Harness::new(shifted());
    harness.register([70]);

    let plan = harness.request(70);

    assert_eq!(
        plan,
        Plan::Reseed {
            start: 70,
            steps: 71..73
        }
    );
}

#[test]
fn a_request_behind_the_stream_reseeds() {
    let mut harness = Harness::new(shifted());
    render_in_order(&mut harness, 0..=30);

    harness.register([10]);
    let plan = harness.request(10);

    assert_eq!(
        plan,
        Plan::Reseed {
            start: 10,
            steps: 11..13
        }
    );
}

#[test]
fn the_stream_catches_up_to_the_edge_of_the_request_range() {
    let mut harness = Harness::new(shifted());
    render_in_order(&mut harness, 0..=10);

    // The stream's next push is source 17, and 37 is the furthest output whose range reaches back to it.
    let reachable = harness.planner.plan(37, false, &harness.outstanding);
    assert_eq!(reachable, Plan::Advance { steps: 13..40 });

    let too_far = harness.planner.plan(38, false, &harness.outstanding);
    assert_eq!(
        too_far,
        Plan::Reseed {
            start: 38,
            steps: 39..41
        }
    );
}

#[test]
fn the_shifted_clip_end_flushes_and_serves_the_tail_from_the_buffer() {
    let mut harness = Harness::new(shifted());
    render_in_order(&mut harness, 0..=94);

    let last_plan = harness.plans.last();
    assert_eq!(last_plan, Some(&Plan::Advance { steps: 96..97 }));

    for n in 95..=LAST_FRAME {
        assert_eq!(harness.request(n), Plan::Serve);
    }

    assert_eq!(harness.reseeds(), 1);
}

#[test]
fn a_clip_shorter_than_two_windows_flushes_on_the_first_request() {
    let mut harness = Harness::new(geometry(EdgePadding::Shifted, 4, 4, 5));
    harness.register(0..=5);

    let first = harness.request(0);
    assert_eq!(
        first,
        Plan::Reseed {
            start: 0,
            steps: 1..3
        }
    );

    for n in 1..=5 {
        assert_eq!(harness.request(n), Plan::Serve);
    }
}

#[test]
fn a_clip_inside_one_window_reseeds_with_no_steps() {
    let mut harness = Harness::new(geometry(EdgePadding::Shifted, 4, 4, 2));
    harness.register(0..=2);

    let first = harness.request(0);
    assert_eq!(
        first,
        Plan::Reseed {
            start: 0,
            steps: 1..1
        }
    );

    for n in 1..=2 {
        assert_eq!(harness.request(n), Plan::Serve);
    }
}

#[test]
fn a_killed_stream_reseeds_on_the_next_request() {
    let mut harness = Harness::new(shifted());
    render_in_order(&mut harness, 0..=5);

    harness.planner.kill();
    harness.register([6]);
    let plan = harness.request(6);

    assert_eq!(
        plan,
        Plan::Reseed {
            start: 6,
            steps: 7..9
        }
    );
}

#[test]
fn every_plan_stays_inside_its_request_range() {
    for edges in [EdgePadding::Shifted, EdgePadding::Repeat] {
        let mut harness = Harness::new(geometry(edges, 4, 4, LAST_FRAME));

        for first in (0..=LAST_FRAME).step_by(8) {
            let last = (first + 7).min(LAST_FRAME);
            harness.register(first..=last);

            // Reversed, so every burst opens with its highest frame, the furthest from the stream.
            for n in (first..=last).rev() {
                harness.request(n);
            }
        }
    }
}

#[test]
fn a_burst_wider_than_the_lookback_still_serves_every_frame() {
    let mut harness = Harness::new(shifted());
    harness.register(0..40);

    for n in (0..40).rev() {
        harness.request(n);
    }

    assert!(harness.reseeds() > 1);
}

#[test]
fn outstanding_counts_repeated_registrations() {
    let mut outstanding = Outstanding::default();
    outstanding.register(4);
    outstanding.register(4);

    outstanding.finish(4);
    assert_eq!(outstanding.lowest_in(0..=10), Some(4));

    outstanding.finish(4);
    assert_eq!(outstanding.lowest_in(0..=10), None);
}

#[test]
fn outstanding_ignores_a_finish_without_a_registration() {
    let mut outstanding = Outstanding::default();

    outstanding.finish(3);

    assert_eq!(outstanding.lowest_in(0..=10), None);
}

#[test]
fn lowest_in_only_looks_inside_the_range() {
    let mut outstanding = Outstanding::default();
    outstanding.register(2);
    outstanding.register(8);

    assert_eq!(outstanding.lowest_in(3..=10), Some(8));
}

#[test]
fn the_buffer_evicts_the_lowest_frames_first() {
    let mut buffer = OutputBuffer::new();
    for index in 0..5 {
        buffer.insert(index, index);
    }

    buffer.evict_to(3);

    assert_eq!(buffer.len(), 3);
    assert!(!buffer.contains(1));
    assert!(buffer.contains(2));
}

#[test]
fn a_buffered_frame_is_taken_once() {
    let mut buffer = OutputBuffer::new();
    buffer.insert(7, "seven");

    assert_eq!(buffer.take(7), Some("seven"));
    assert_eq!(buffer.take(7), None);
    assert!(buffer.is_empty());
}

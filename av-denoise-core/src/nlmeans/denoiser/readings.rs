use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use cubecl::bytes::Bytes;
use cubecl::prelude::*;
use cubecl::server::{Handle, ServerError};

use crate::engine::timing::StallClock;
use crate::noise::{
    QuarterSettings,
    RESULTS_PER_SLOT,
    TemporalNoiseReading,
    sigma_block_p25_from_partials,
    temporal_noise_reading,
};

/// What a read hands back, one buffer per handle read.
type ReadFuture = Pin<Box<dyn Future<Output = Result<Vec<Bytes>, ServerError>> + Send>>;

/// A future that is pending on its first poll and ready on the next.
struct PauseOnce {
    paused: bool,
}

impl Future for PauseOnce {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<()> {
        if self.paused {
            return Poll::Ready(());
        }

        self.paused = true;
        context.waker().wake_by_ref();
        Poll::Pending
    }
}

/// A readback whose copy is already queued on the device.
///
/// The copy is queued when the read starts, right behind the kernels that wrote the data, so a
/// later wait never waits on work queued after it. Dropping a read before it lands blocks until it
/// lands, because the device holds a staging buffer for it that must be released. A read dropped
/// while the thread panics is leaked instead.
pub(in crate::nlmeans) struct InFlightRead {
    future: Option<ReadFuture>,
}

impl InFlightRead {
    /// Queues a read of `handles` behind the work already queued on this thread.
    pub(in crate::nlmeans) fn start<R: Runtime>(client: &ComputeClient<R>, handles: Vec<Handle>) -> Self {
        let owned = client.clone();
        let future = async move {
            let bytes = owned.read_async(handles);
            let pause = PauseOnce { paused: false };
            pause.await;
            bytes.await
        };
        let mut boxed: ReadFuture = Box::pin(future);

        // The first poll queues the copy and stops at the pause, so nothing here waits.
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        let first = boxed.as_mut().poll(&mut context);
        debug_assert!(
            first.is_pending(),
            "the pause keeps the first poll from finishing"
        );

        Self { future: Some(boxed) }
    }

    /// Polls once and returns the bytes if they have landed.
    ///
    /// It never blocks on wgpu. On CUDA and ROCm it waits for the copy itself.
    pub(in crate::nlmeans) fn poll_now(&mut self) -> Option<Result<Vec<Bytes>, ServerError>> {
        let future = self.future.as_mut()?;
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        let poll = future.as_mut().poll(&mut context);

        match poll {
            Poll::Ready(result) => {
                self.future = None;
                Some(result)
            },
            Poll::Pending => None,
        }
    }

    /// Blocks until the bytes land.
    ///
    /// # Panics
    ///
    /// Panics if the bytes were already taken.
    pub(in crate::nlmeans) fn wait(&mut self) -> Result<Vec<Bytes>, ServerError> {
        let future = self
            .future
            .take()
            .expect("an in-flight read hands its bytes out once");
        cubecl::future::block_on(future)
    }
}

impl Drop for InFlightRead {
    fn drop(&mut self) {
        let Some(future) = self.future.take() else {
            return;
        };

        // Leaking the read while unwinding never blocks and never frees memory the copy still uses.
        if std::thread::panicking() {
            std::mem::forget(future);
            return;
        }

        let _ = cubecl::future::block_on(future);
    }
}

/// The frame shape and settings that turn one slot's bytes into a [SlotNoiseReading].
#[derive(Debug, Clone, Copy)]
pub(in crate::nlmeans) struct ReadingShape {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub stored_ch: u32,
    /// Whether the luma noise curve and quarter classes are built.
    pub with_curve: bool,
    pub settings: QuarterSettings,
}

/// The values one ring slot's noise readback turns into.
#[derive(Debug, Clone)]
pub(in crate::nlmeans) struct SlotNoiseReading {
    /// The slot's row of Immerkær totals.
    pub immerkaer_sums: [f32; RESULTS_PER_SLOT],
    /// The lower quartile of the per-block Immerkær sigmas.
    pub immerkaer_low: [f32; 3],
    /// The temporal reading, empty when the temporal estimator is inactive.
    pub temporal: TemporalNoiseReading,
}

impl SlotNoiseReading {
    /// Builds a reading from the results row, the partials and, when present, the temporal stats.
    pub(in crate::nlmeans) fn from_bytes(buffers: &[&[u8]], shape: &ReadingShape) -> Self {
        let row = f32::from_bytes(buffers[0]);
        let mut immerkaer_sums = [0.0f32; RESULTS_PER_SLOT];
        immerkaer_sums.copy_from_slice(&row[..RESULTS_PER_SLOT]);

        let partials = f32::from_bytes(buffers[1]);
        let immerkaer_low =
            sigma_block_p25_from_partials(partials, shape.channels, shape.width, shape.height);

        let temporal = match buffers.get(2) {
            Some(stats) => {
                let records = f32::from_bytes(stats);
                temporal_noise_reading(
                    records,
                    shape.channels,
                    shape.stored_ch,
                    shape.width,
                    shape.height,
                    shape.with_curve,
                    shape.settings,
                )
            },
            None => TemporalNoiseReading::empty(),
        };

        Self {
            immerkaer_sums,
            immerkaer_low,
            temporal,
        }
    }
}

/// Where one slot's reading is.
enum EntryState {
    InFlight {
        read: InFlightRead,
        shape: ReadingShape,
    },
    Landed(Arc<SlotNoiseReading>),
    /// The read failed with this message.
    Failed(String),
}

/// One ring slot's reading, tagged with the push that filled the slot.
struct SlotEntry {
    push_index: usize,
    state: EntryState,
}

fn readback_error(message: &str) -> anyhow::Error {
    anyhow::anyhow!("noise readback failed: {message}")
}

/// Turns landed bytes into a reading.
fn land(buffers: &[Bytes], shape: &ReadingShape) -> Arc<SlotNoiseReading> {
    let slices: Vec<&[u8]> = buffers.iter().map(|buffer| &buffer[..]).collect();
    let reading = SlotNoiseReading::from_bytes(&slices, shape);

    Arc::new(reading)
}

/// Moves `state` to the reading that landed, or to the failure that stopped it.
fn settle(
    state: &mut EntryState,
    landed: Result<Vec<Bytes>, ServerError>,
    shape: &ReadingShape,
) -> Result<(), anyhow::Error> {
    match landed {
        Ok(buffers) => {
            let reading = land(&buffers, shape);
            *state = EntryState::Landed(reading);
            Ok(())
        },
        Err(error) => {
            let message = error.to_string();
            let failure = readback_error(&message);
            *state = EntryState::Failed(message);
            Err(failure)
        },
    }
}

/// The noise readings of a frame ring, read back without stalling the queue.
///
/// Each push starts a read of its slot right behind the kernels that measured it. The bytes land
/// while later work runs and become a [SlotNoiseReading] when processed. A submit takes the reading
/// of the slot it folds, waiting only if it has not landed yet. A reading belongs to the push that
/// filled its slot, and a later push into the slot drops it.
pub(in crate::nlmeans) struct NoiseReadings {
    slots: Vec<Option<SlotEntry>>,
    stalls: StallClock,
    /// Readings that had to be read when a submit asked for them.
    fallbacks: u64,
}

impl NoiseReadings {
    pub(in crate::nlmeans) fn new(slot_count: usize) -> Self {
        let slots = (0..slot_count).map(|_| None).collect();

        Self {
            slots,
            stalls: StallClock::default(),
            fallbacks: 0,
        }
    }

    /// Starts reading `handles` for the frame that push `push_index` wrote into `slot`.
    ///
    /// A reading the slot held for an earlier frame is dropped once its copy lands.
    pub(in crate::nlmeans) fn start<R: Runtime>(
        &mut self,
        client: &ComputeClient<R>,
        slot: u32,
        push_index: usize,
        handles: Vec<Handle>,
        shape: ReadingShape,
    ) {
        let read = InFlightRead::start(client, handles);
        let entry = SlotEntry {
            push_index,
            state: EntryState::InFlight { read, shape },
        };

        self.slots[slot as usize] = Some(entry);
    }

    /// Turns every landed reading started before push `first_skipped` into values.
    ///
    /// Readings from `first_skipped` on are left alone, since on CUDA and ROCm a poll waits for the
    /// copy.
    pub(in crate::nlmeans) fn process_landed(&mut self, first_skipped: usize) -> Result<(), anyhow::Error> {
        for entry in self.slots.iter_mut().flatten() {
            if entry.push_index >= first_skipped {
                continue;
            }

            let EntryState::InFlight { read, shape } = &mut entry.state else {
                continue;
            };

            let shape = *shape;
            let Some(landed) = read.poll_now() else {
                continue;
            };

            settle(&mut entry.state, landed, &shape)?;
        }

        Ok(())
    }

    /// The reading push `push_index` left in `slot`, waiting for it to land if needed.
    ///
    /// Returns `None` when the slot holds no reading from that push.
    pub(in crate::nlmeans) fn get(
        &mut self,
        slot: u32,
        push_index: usize,
    ) -> Result<Option<Arc<SlotNoiseReading>>, anyhow::Error> {
        let Some(entry) = self.slots[slot as usize].as_mut() else {
            return Ok(None);
        };

        if entry.push_index != push_index {
            return Ok(None);
        }

        if let EntryState::InFlight { read, shape } = &mut entry.state {
            let shape = *shape;
            let landed = self.stalls.time(|| read.wait());
            settle(&mut entry.state, landed, &shape)?;
        }

        let reading = match &entry.state {
            EntryState::Landed(reading) => reading,
            EntryState::Failed(message) => {
                let error = readback_error(message);
                return Err(error);
            },
            EntryState::InFlight { .. } => unreachable!("an in-flight reading lands above"),
        };

        let reading = Arc::clone(reading);
        Ok(Some(reading))
    }

    /// Drops every reading. Reads still in flight land first.
    pub(in crate::nlmeans) fn clear(&mut self) {
        for entry in &mut self.slots {
            *entry = None;
        }
    }

    pub(in crate::nlmeans) fn count_fallback(&mut self) {
        self.fallbacks += 1;
    }

    #[cfg(test)]
    pub(in crate::nlmeans) fn fallbacks(&self) -> u64 {
        self.fallbacks
    }

    pub(in crate::nlmeans) fn stalls(&self) -> &StallClock {
        &self.stalls
    }

    #[cfg(test)]
    pub(in crate::nlmeans) fn is_landed(&self, slot: u32) -> bool {
        let entry = self.slots[slot as usize].as_ref();
        entry.is_some_and(|entry| matches!(entry.state, EntryState::Landed(_)))
    }

    #[cfg(test)]
    pub(in crate::nlmeans) fn fail_slot(&mut self, slot: u32, push_index: usize, message: &str) {
        let owned_message = message.to_string();
        let entry = SlotEntry {
            push_index,
            state: EntryState::Failed(owned_message),
        };

        self.slots[slot as usize] = Some(entry);
    }
}

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::kernels::{egress_f32, egress_words, ingest_f32, ingest_words};
use super::{DevicePlane, SampleFormat};
use crate::nlmeans::{BLOCK_1D, MAX_GRID_1D};

/// Where an ingest writes, as a slot of an interleaved ring.
#[cfg_attr(not(test), expect(dead_code, reason = "no engine calls this yet"))]
pub(crate) struct IngestTarget<'a> {
    pub ring: &'a Handle,
    pub ring_len: usize,
    pub offset: u32,
    pub pixels: u32,
    pub channels: u32,
    pub stored_ch: u32,
}

/// The handle bound for plane `index`, falling back to `placeholder` for planes the kernel never reads.
#[cfg_attr(not(test), expect(dead_code, reason = "no engine calls this yet"))]
fn plane_or<'a>(planes: &[DevicePlane<'a>], index: usize, placeholder: &'a Handle) -> &'a Handle {
    match planes.get(index) {
        Some(plane) => plane.handle(),
        None => placeholder,
    }
}

#[cfg_attr(not(test), expect(dead_code, reason = "no engine calls this yet"))]
fn grid(pixels: u32) -> (CubeCount, u32) {
    let groups = pixels.div_ceil(BLOCK_1D).clamp(1, MAX_GRID_1D);
    let total_threads = groups * BLOCK_1D;
    (CubeCount::new_1d(groups), total_threads)
}

/// Queues the ingest of `planes` into one ring slot.
///
/// The caller has validated `planes` against the engine's geometry.
#[cfg_attr(not(test), expect(dead_code, reason = "no engine calls this yet"))]
pub(crate) fn ingest<R: Runtime>(
    client: &ComputeClient<R>,
    planes: &[DevicePlane<'_>],
    format: SampleFormat,
    placeholder: &Handle,
    target: IngestTarget<'_>,
) {
    let (count, total_threads) = grid(target.pixels);
    let plane_0 = plane_or(planes, 0, placeholder);
    let plane_1 = plane_or(planes, 1, placeholder);
    let plane_2 = plane_or(planes, 2, placeholder);
    let ring = unsafe { ArrayArg::from_raw_parts(target.ring.clone(), target.ring_len) };

    match format {
        SampleFormat::F32 => {
            let plane_len = target.pixels as usize;

            unsafe {
                ingest_f32::launch_unchecked::<R>(
                    client,
                    count,
                    CubeDim::new_1d(BLOCK_1D),
                    ArrayArg::from_raw_parts(plane_0.clone(), plane_len),
                    ArrayArg::from_raw_parts(plane_1.clone(), plane_len),
                    ArrayArg::from_raw_parts(plane_2.clone(), plane_len),
                    ring,
                    target.offset,
                    target.pixels,
                    target.channels,
                    target.stored_ch,
                    total_threads,
                );
            }
        },
        SampleFormat::U8 | SampleFormat::U16 { .. } => {
            let samples_per_word = format.samples_per_word();
            let words = target.pixels.div_ceil(samples_per_word) as usize;
            let max = format.max_value();

            unsafe {
                ingest_words::launch_unchecked::<R>(
                    client,
                    count,
                    CubeDim::new_1d(BLOCK_1D),
                    ArrayArg::from_raw_parts(plane_0.clone(), words),
                    ArrayArg::from_raw_parts(plane_1.clone(), words),
                    ArrayArg::from_raw_parts(plane_2.clone(), words),
                    ring,
                    max,
                    target.offset,
                    target.pixels,
                    target.channels,
                    target.stored_ch,
                    samples_per_word,
                    total_threads,
                );
            }
        },
    }
}

/// A finished interleaved f32 frame to write out.
#[cfg_attr(not(test), expect(dead_code, reason = "no engine calls this yet"))]
pub(crate) struct EgressSource<'a> {
    pub frame: &'a Handle,
    pub pixels: u32,
    pub channels: u32,
    pub stored_ch: u32,
}

/// Queues the write of `source` into `planes`.
///
/// The caller has validated `planes` against the engine's geometry.
#[cfg_attr(not(test), expect(dead_code, reason = "no engine calls this yet"))]
pub(crate) fn egress<R: Runtime>(
    client: &ComputeClient<R>,
    source: EgressSource<'_>,
    planes: &[DevicePlane<'_>],
    format: SampleFormat,
    placeholder: &Handle,
) {
    let frame_len = (source.pixels * source.stored_ch) as usize;
    let frame = unsafe { ArrayArg::from_raw_parts(source.frame.clone(), frame_len) };
    let plane_0 = plane_or(planes, 0, placeholder);
    let plane_1 = plane_or(planes, 1, placeholder);
    let plane_2 = plane_or(planes, 2, placeholder);

    match format {
        SampleFormat::F32 => {
            let (count, total_threads) = grid(source.pixels);
            let plane_len = source.pixels as usize;

            unsafe {
                egress_f32::launch_unchecked::<R>(
                    client,
                    count,
                    CubeDim::new_1d(BLOCK_1D),
                    frame,
                    ArrayArg::from_raw_parts(plane_0.clone(), plane_len),
                    ArrayArg::from_raw_parts(plane_1.clone(), plane_len),
                    ArrayArg::from_raw_parts(plane_2.clone(), plane_len),
                    source.pixels,
                    source.channels,
                    source.stored_ch,
                    total_threads,
                );
            }
        },
        SampleFormat::U8 | SampleFormat::U16 { .. } => {
            let samples_per_word = format.samples_per_word();
            let words = source.pixels.div_ceil(samples_per_word);
            let (count, total_threads) = grid(words);
            let max = format.max_value();
            let word_len = words as usize;

            unsafe {
                egress_words::launch_unchecked::<R>(
                    client,
                    count,
                    CubeDim::new_1d(BLOCK_1D),
                    frame,
                    ArrayArg::from_raw_parts(plane_0.clone(), word_len),
                    ArrayArg::from_raw_parts(plane_1.clone(), word_len),
                    ArrayArg::from_raw_parts(plane_2.clone(), word_len),
                    max,
                    source.pixels,
                    source.channels,
                    source.stored_ch,
                    samples_per_word,
                    words,
                    total_threads,
                );
            }
        },
    }
}

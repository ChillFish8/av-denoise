use cubecl::prelude::*;

use crate::nlmeans::align::StorageAlign;
use crate::nlmeans::motion::{self, MotionCtx, MotionEstimation};
use crate::nlmeans::noise::{temporal_stats_blocks, temporal_stats_record_len};
use crate::nlmeans::params::NlmParams;
use crate::nlmeans::{BLOCK_X, BLOCK_Y};

/// Bytes in one element of every buffer sized here, which all hold `f32`, `i32` or `u32`.
const ELEMENT_BYTES: u64 = 4;

/// A buffer's name and its element count, `None` when the count overflows `u64`.
pub(crate) type BufferSize = (&'static str, Option<u64>);

/// The geometry-sized buffers of an [NlmDenoiser](super::NlmDenoiser) beyond its frame rings.
pub(crate) struct FrontSizes {
    pub buffers: Vec<BufferSize>,
    /// Blocks in one motion field, when motion compensation runs.
    pub motion_blocks: Option<u64>,
}

fn times(count: Option<u64>, factor: u64) -> Option<u64> {
    count.and_then(|count| count.checked_mul(factor))
}

/// Elements in `slots` slots of `slot_bytes` bytes each.
fn slot_ring(slot_bytes: Option<u64>, slots: u64) -> Option<u64> {
    let ring_bytes = times(slot_bytes, slots);
    ring_bytes.map(|bytes| bytes / ELEMENT_BYTES)
}

fn padded_slot_bytes(elements: Option<u64>, align: StorageAlign) -> Option<u64> {
    let bytes = times(elements, ELEMENT_BYTES);
    bytes.and_then(|bytes| align.checked_pad_bytes(bytes))
}

fn block_count(geometry: &MotionCtx) -> Option<u64> {
    let blocks_x = u64::from(geometry.blocks_x);
    let blocks_y = u64::from(geometry.blocks_y);

    blocks_x.checked_mul(blocks_y)
}

/// Elements in a pyramid ring of `slots` frames, padded per level like the allocation.
fn pyramid_ring(width: u32, height: u32, levels: u32, align: StorageAlign, slots: u64) -> Option<u64> {
    let per_frame = motion::pyramid_pixels_per_frame(width, height, levels, align);
    let per_frame = u64::try_from(per_frame).ok();

    times(per_frame, slots)
}

/// The element count of every buffer [NlmDenoiser::new](super::NlmDenoiser::new) sizes from the frame,
/// other than the frame rings.
///
/// Each count repeats its allocation's formula in checked `u64`. The geometry must have passed
/// [Geometry::check_ring_fits](crate::engine::Geometry::check_ring_fits) for the frame ring, which
/// keeps the pixel count within `u32` so the shared size helpers called here cannot overflow.
pub(crate) fn front_buffer_sizes<R: Runtime>(
    client: &ComputeClient<R>,
    params: &NlmParams,
    width: u32,
    height: u32,
) -> FrontSizes {
    let align = StorageAlign::from_client(client);
    let stored_ch = params.channels.storage_count();
    let total_frames = u64::from(params.total_frames());
    let neighbours = 2 * u64::from(params.temporal_radius);
    let mut buffers = Vec::new();

    let auto_noise = params.hq.is_some_and(|hq| hq.sigma_override.is_none());
    if auto_noise {
        let cubes_x = u64::from(width.div_ceil(BLOCK_X));
        let cubes_y = u64::from(height.div_ceil(BLOCK_Y));
        let partials = times(cubes_x.checked_mul(cubes_y), 4);
        let slot_bytes = padded_slot_bytes(partials, align);
        let ring = slot_ring(slot_bytes, total_frames);
        buffers.push(("noise partials ring", ring));
    }

    if auto_noise && params.temporal_radius >= 1 {
        let (blocks_x, blocks_y) = temporal_stats_blocks(width, height);
        let blocks = u64::from(blocks_x).checked_mul(u64::from(blocks_y));
        let record_len = u64::from(temporal_stats_record_len(stored_ch));
        let slot_len = times(blocks, record_len);
        let slot_bytes = padded_slot_bytes(slot_len, align);
        let ring = slot_ring(slot_bytes, total_frames);
        buffers.push(("temporal stats ring", ring));
    }

    let motion_active = params.motion_compensation.is_active() && params.temporal_radius > 0;
    let mc_ctx = if motion_active {
        MotionCtx::new(params.motion_compensation, width, height, align)
    } else {
        None
    };

    let motion_blocks = mc_ctx.as_ref().and_then(block_count);

    if let Some(motion_ctx) = mc_ctx.as_ref() {
        let field_len = times(motion_blocks, 2);
        let field_bytes = padded_slot_bytes(field_len, align);
        let field = slot_ring(field_bytes, neighbours);
        buffers.push(("motion field", field));

        let pyramid = pyramid_ring(width, height, motion_ctx.pyramid_levels, align, total_frames);
        buffers.push(("motion pyramid ring", pyramid));
    }

    let estimation = params
        .motion_compensation
        .resolved_estimation(params.temporal_radius);
    let is_chained = matches!(estimation, Some(MotionEstimation::Chained { .. }));
    if is_chained && mc_ctx.is_some() {
        let direction_len = times(motion_blocks, 2);
        let direction_bytes = padded_slot_bytes(direction_len, align);
        let slot_bytes = times(direction_bytes, 2);
        let pair_slots = u64::from(motion::pair_ring_slot_count(params.temporal_radius));
        let ring = slot_ring(slot_bytes, pair_slots);
        buffers.push(("motion pair ring", ring));
    }

    let confidence_active = params.hq.is_some_and(|hq| hq.temporal_confidence) && params.temporal_radius > 0;
    let confidence_only_active = confidence_active && mc_ctx.is_none();
    let confidence_ctx = confidence_only_active.then(|| MotionCtx::confidence_only(width, height, align));

    let confidence_geometry = if confidence_active {
        mc_ctx.as_ref().or(confidence_ctx.as_ref())
    } else {
        None
    };
    if let Some(geometry) = confidence_geometry {
        let blocks = block_count(geometry);
        let slot_bytes = padded_slot_bytes(blocks, align);
        let confidence = slot_ring(slot_bytes, neighbours);
        buffers.push(("confidence", confidence));
    }

    if let Some(geometry) = confidence_ctx.as_ref() {
        let pyramid = pyramid_ring(width, height, geometry.pyramid_levels, align, total_frames);
        buffers.push(("confidence pyramid ring", pyramid));

        let blocks = block_count(geometry);
        let scratch = times(blocks, 2);
        buffers.push(("confidence vector scratch", scratch));
    }

    FrontSizes {
        buffers,
        motion_blocks,
    }
}

/// Rejects the first buffer that would hold more than `u32::MAX` elements, naming it.
pub(crate) fn check_u32_indexable(buffers: &[BufferSize]) -> Result<(), String> {
    for (name, elements) in buffers {
        let fits = elements.is_some_and(|elements| elements <= u64::from(u32::MAX));
        if !fits {
            return Err(format!("the {name} would hold more than u32::MAX elements"));
        }
    }

    Ok(())
}

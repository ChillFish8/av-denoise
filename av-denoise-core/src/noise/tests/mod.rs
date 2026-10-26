mod classes;
mod correlation;
mod curve;
mod estimator;
mod fold;
mod fold_frames;
pub(crate) mod fold_reference;
pub(crate) mod frames;
mod host;
mod line_ring;
mod records;
mod sample;
mod select;
mod spatial;
mod state;
mod stats;
mod tables;
mod temporal;
mod y4m;

use cubecl::prelude::*;
use cubecl::server::Handle;

#[cfg(any(feature = "vulkan", feature = "metal"))]
use super::temporal::temporal_stats_slot_view;
use super::{
    DERIVED_CURVE_VALID,
    DERIVED_MAP_MODE,
    DERIVED_NOISE_OFFSET,
    DERIVED_SIGMA,
    DERIVED_WEIGHT_SCALE,
    DERIVED_WORDS,
};
#[cfg(any(feature = "vulkan", feature = "metal"))]
use crate::engine::align::StorageAlign;

/// The step between the values a test derived buffer holds in the words its kernel must not read.
const UNREAD_WORD_STEP: f32 = -1.0e4;

/// A derived buffer whose word `field` holds `sad_floor`.
///
/// Every other word holds its own large negative value, so a kernel that reads the wrong word
/// scores every block with zero confidence.
pub(crate) fn sad_floor_buffer(client: &Client, field: u32, sad_floor: f32) -> Handle {
    let mut words = unread_words();
    words[field as usize] = sad_floor;
    let bytes = f32::as_bytes(&words);

    client.create_from_slice(bytes)
}

/// A derived buffer whose word `h2_field` holds `h2_inv_norm` and whose noise offset word holds
/// `noise_offset`.
///
/// Every other word holds its own large negative value, so an NLM kernel that reads the wrong word
/// gives infinite or zero weights.
pub(crate) fn nlm_noise_buffer(
    client: &Client,
    h2_field: u32,
    h2_inv_norm: f32,
    noise_offset: f32,
) -> Handle {
    let mut words = unread_words();
    words[h2_field as usize] = h2_inv_norm;
    words[DERIVED_NOISE_OFFSET as usize] = noise_offset;
    let bytes = f32::as_bytes(&words);

    client.create_from_slice(bytes)
}

/// The values the collab filter reads from the derived buffer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CollabNoise<'a> {
    /// One sigma per live channel.
    pub(crate) sigma: &'a [f32],
    pub(crate) weight_scale: f32,
    pub(crate) curve_valid: bool,
    /// One of the `STRENGTH_MAP_*` modes.
    pub(crate) map_mode: u32,
}

/// A derived buffer holding `noise` in the words the collab filter reads.
///
/// Every other word, the sigma of each channel past the live ones included, holds its own large
/// negative value. A kernel that reads the wrong word then thresholds with a nonsense sigma or
/// scatters a negative weight.
pub(crate) fn collab_noise_buffer(client: &Client, noise: CollabNoise) -> Handle {
    let mut words = unread_words();

    let sigma_start = DERIVED_SIGMA as usize;
    let sigma_end = sigma_start + noise.sigma.len();
    words[sigma_start..sigma_end].copy_from_slice(noise.sigma);

    words[DERIVED_WEIGHT_SCALE as usize] = noise.weight_scale;
    words[DERIVED_CURVE_VALID as usize] = if noise.curve_valid { 1.0 } else { 0.0 };
    words[DERIVED_MAP_MODE as usize] = noise.map_mode as f32;
    let bytes = f32::as_bytes(&words);

    client.create_from_slice(bytes)
}

/// Every word of a derived buffer at its own large negative value.
fn unread_words() -> Vec<f32> {
    (0..DERIVED_WORDS)
        .map(|index| UNREAD_WORD_STEP * (index + 1) as f32)
        .collect()
}

/// Reads one ring slot's temporal-stats region back.
///
/// The ring handle is sliced to the slot, so the transfer skips the rest of the ring.
#[cfg(any(feature = "vulkan", feature = "metal"))]
#[expect(
    clippy::too_many_arguments,
    reason = "the readback needs the ring handle plus every shape value that locates one slot in it"
)]
pub(crate) fn read_temporal_stats_slot(
    client: &Client,
    stats_buf: &Handle,
    width: u32,
    height: u32,
    stored_ch: u32,
    frame_count: u32,
    slot: u32,
    align: StorageAlign,
) -> Result<Vec<f32>, anyhow::Error> {
    let sliced = temporal_stats_slot_view(stats_buf, width, height, stored_ch, frame_count, slot, align);
    let bytes = client
        .read_one(sliced)
        .map_err(|error| anyhow::anyhow!("temporal noise stats readback failed: {error}"))?;
    let records = f32::from_bytes(&bytes).to_vec();

    Ok(records)
}

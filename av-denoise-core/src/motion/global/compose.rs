use cubecl::prelude::*;
use cubecl::server::Handle;

use super::ModelRing;
use crate::motion::kernels::mc_compose_models;
use crate::tune::gpu::bind;

/// The `f32` words of one composed prediction, `m00, m01, m10, m11, tx, ty`, in a pass's pixels.
pub const AFFINE_WORDS: u32 = 6;

/// A gap-table entry with no pair model, which composes as no motion.
pub const NO_PAIR: u32 = u32::MAX;

/// Builds the gap table [mc_compose_models](crate::motion::kernels::mc_compose_models()) reads.
///
/// `slot_frames` holds the stream frame of each ring slot and `centre` is the slot of the centre
/// frame. Neighbour slots wrap around the end, so physical and logical orders both work.
/// Neighbours come in motion-field order, past first, each with `radius` entries listed from the
/// centre outward. A gap is the ring slot of its newer frame's pair. A gap that does not step
/// exactly one stream frame, such as an edge duplicate, and every entry past the neighbour's
/// distance, is [NO_PAIR].
pub fn compose_gaps(slot_frames: &[u32], centre: usize, radius: usize, ring: &ModelRing) -> Vec<u32> {
    let slot_count = slot_frames.len() as isize;
    let frame_at = |offset: isize| {
        let index = (centre as isize + offset).rem_euclid(slot_count);
        slot_frames[index as usize]
    };
    let gap_between = |older: isize| {
        let older_frame = frame_at(older);
        let newer_frame = frame_at(older + 1);
        let next_frame = older_frame.wrapping_add(1);
        if newer_frame == next_frame {
            ring.slot(newer_frame)
        } else {
            NO_PAIR
        }
    };

    let mut gaps = vec![NO_PAIR; 2 * radius * radius];

    for distance in 1..=radius {
        let past_row = (radius - distance) * radius;
        let future_row = (radius + distance - 1) * radius;

        for step in 0..distance {
            let step_offset = step as isize;
            gaps[past_row + step] = gap_between(-step_offset - 1);
            gaps[future_row + step] = gap_between(step_offset);
        }
    }

    gaps
}

/// Composes one prediction per neighbour from `ring` into `out`.
///
/// `gaps` holds the table [compose_gaps] builds for `neighbours` neighbours with `max_steps`
/// entries each, and `out` takes [AFFINE_WORDS] words per neighbour.
#[expect(
    clippy::too_many_arguments,
    reason = "the dispatch threads through every buffer and shape the kernel binds"
)]
pub fn run_compose_models(
    client: &Client,
    ring: &ModelRing,
    gaps: &Handle,
    out: &Handle,
    neighbours: u32,
    max_steps: u32,
    scale_x: f32,
    scale_y: f32,
) {
    let gaps_len = (neighbours * max_steps) as usize;
    let out_len = (neighbours * AFFINE_WORDS) as usize;
    let cube_count = CubeCount::new_1d(1);
    let cube_dim = CubeDim::new_1d(neighbours);

    let models_len = ring.words();
    let models_arg = bind(ring.handle(), models_len);
    let gaps_arg = bind(gaps, gaps_len);
    let out_arg = bind(out, out_len);

    // SAFETY: each binding covers the words its buffer was allocated for, and the one cube holds a
    // thread per neighbour.
    unsafe {
        mc_compose_models::launch_unchecked(
            client, cube_count, cube_dim, models_arg, gaps_arg, out_arg, neighbours, max_steps, scale_x,
            scale_y,
        );
    }
}

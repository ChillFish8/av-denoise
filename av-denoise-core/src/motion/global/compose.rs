use cubecl::prelude::*;
use cubecl::server::Handle;

use super::ModelRing;
use crate::motion::kernels::mc_compose_models;
use crate::tune::gpu::bind;

/// The `f32` words of one composed prediction, `m00, m01, m10, m11, tx, ty`, in luma pixels.
pub const AFFINE_WORDS: u32 = 6;

/// A gap-table entry with no pair model, which composes as no motion.
pub const NO_PAIR: u32 = u32::MAX;

/// Builds the gap table [mc_compose_models](crate::motion::kernels::mc_compose_models()) reads.
///
/// `window_frames` holds the stream frame at each window position, oldest first, and `centre` is
/// the position of the centre frame. Every other position is a neighbour, in window order, with
/// `max_steps` entries listed from the centre outward. A gap is the ring slot of its newer frame's
/// pair. A gap that does not step exactly one stream frame, such as an edge duplicate, and every
/// entry past the neighbour's distance, is [NO_PAIR].
///
/// # Panics
///
/// Panics if a neighbour lies more than `max_steps` positions from the centre.
pub fn compose_gaps(window_frames: &[u32], centre: usize, max_steps: usize, ring: &ModelRing) -> Vec<u32> {
    let gap_between = |older: usize| {
        let older_frame = window_frames[older];
        let newer_frame = window_frames[older + 1];
        let next_frame = older_frame.wrapping_add(1);
        if newer_frame == next_frame {
            ring.slot(newer_frame)
        } else {
            NO_PAIR
        }
    };

    let neighbours = window_frames.len() - 1;
    let mut gaps = vec![NO_PAIR; neighbours * max_steps];

    let positions = (0..window_frames.len()).filter(|&position| position != centre);
    for (row, position) in positions.enumerate() {
        let distance = position.abs_diff(centre);
        assert!(
            distance <= max_steps,
            "a neighbour {distance} frames out needs more than {max_steps} steps"
        );

        let row_start = row * max_steps;
        for step in 0..distance {
            let older = if position < centre {
                centre - step - 1
            } else {
                centre + step
            };
            gaps[row_start + step] = gap_between(older);
        }
    }

    gaps
}

/// Composes one prediction per neighbour from `ring` into `out`.
///
/// `gaps` holds the table [compose_gaps] builds for `neighbours` neighbours with `max_steps`
/// entries each, the first `past_neighbours` of them behind the centre. `out` takes
/// [AFFINE_WORDS] words per neighbour.
pub fn run_compose_models(
    client: &Client,
    ring: &ModelRing,
    gaps: &Handle,
    out: &Handle,
    neighbours: u32,
    past_neighbours: u32,
    max_steps: u32,
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
            client,
            cube_count,
            cube_dim,
            models_arg,
            gaps_arg,
            out_arg,
            neighbours,
            max_steps,
            past_neighbours,
        );
    }
}

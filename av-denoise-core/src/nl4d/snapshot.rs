use cubecl::prelude::*;
use cubecl::server::Handle;

/// The motion field and confidence the last collaborative pass read, copied back to the host.
///
/// `vectors[t][block]` is a block's vector toward neighbour `t` in pixels, and
/// `confidence[t][block]` its confidence between 0 and 1. `offsets[t]` is neighbour `t`'s temporal
/// offset from the centre frame. Blocks run row-major, and block `(bx, by)` covers `blksize`
/// pixels starting at `bx * step` and `by * step`.
///
/// This exists for measurement tooling. It is not a stable interface.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct MotionSnapshot {
    pub blocks_x: u32,
    pub blocks_y: u32,
    pub step: u32,
    pub blksize: u32,
    pub offsets: Vec<i32>,
    pub vectors: Vec<Vec<[i32; 2]>>,
    pub confidence: Vec<Vec<f32>>,
}

/// The device buffers one pass handed the fused kernel, kept for a later snapshot readback.
pub(super) struct LastFields {
    pub mv_field: Handle,
    pub confidence: Handle,
    pub mv_stride: u32,
    pub conf_stride: u32,
    pub neighbours: u32,
}

/// Reads `fields` back and unpacks each neighbour's slice.
pub(super) fn read_snapshot<R: Runtime>(
    client: &ComputeClient<R>,
    fields: &LastFields,
    radius: u32,
    blocks_x: u32,
    blocks_y: u32,
    step: u32,
    blksize: u32,
) -> MotionSnapshot {
    let blocks = (blocks_x * blocks_y) as usize;
    let mv_bytes = client
        .read_one(fields.mv_field.clone())
        .expect("motion field readback failed");
    let mv_values = i32::from_bytes(&mv_bytes);
    let conf_bytes = client
        .read_one(fields.confidence.clone())
        .expect("confidence readback failed");
    let conf_values = f32::from_bytes(&conf_bytes);

    let mut offsets = Vec::with_capacity(fields.neighbours as usize);
    let mut vectors = Vec::with_capacity(fields.neighbours as usize);
    let mut confidence = Vec::with_capacity(fields.neighbours as usize);
    for t in 0..fields.neighbours {
        // Matches `neighbour_idx_for_k`, negative offsets first, then positive, each ascending.
        let offset = if t < radius {
            t as i32 - radius as i32
        } else {
            t as i32 - radius as i32 + 1
        };
        offsets.push(offset);

        let mv_base = (t * fields.mv_stride) as usize;
        let block_vectors = (0..blocks)
            .map(|block| [mv_values[mv_base + 2 * block], mv_values[mv_base + 2 * block + 1]])
            .collect();
        vectors.push(block_vectors);

        let conf_base = (t * fields.conf_stride) as usize;
        let block_confidence = conf_values[conf_base..conf_base + blocks].to_vec();
        confidence.push(block_confidence);
    }

    MotionSnapshot {
        blocks_x,
        blocks_y,
        step,
        blksize,
        offsets,
        vectors,
        confidence,
    }
}

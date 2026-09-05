use cubecl::prelude::*;
use cubecl::server::Handle;

/// The motion field and confidence the last collaborative pass read,
/// copied back to the host.
///
/// `vectors[t][block]` is block `block`'s vector toward neighbour `t`,
/// in pixels, and `confidence[t][block]` that block's confidence in
/// `[0, 1]`. `offsets[t]` is neighbour `t`'s temporal offset from the
/// centre frame. Blocks run row-major over `blocks_x * blocks_y`, and
/// block `(bx, by)` covers `blksize` pixels starting at `bx * step` on
/// each axis.
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

/// The device buffers one pass handed the fused kernel, kept so the
/// snapshot can read them back after the fact.
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
    let mv = i32::from_bytes(&mv_bytes);
    let conf_bytes = client
        .read_one(fields.confidence.clone())
        .expect("confidence readback failed");
    let conf = f32::from_bytes(&conf_bytes);

    let mut offsets = Vec::with_capacity(fields.neighbours as usize);
    let mut vectors = Vec::with_capacity(fields.neighbours as usize);
    let mut confidence = Vec::with_capacity(fields.neighbours as usize);
    for t in 0..fields.neighbours {
        // Mirrors `neighbour_idx_for_k`, ascending k on the negative
        // side first, then ascending k on the positive side.
        let k = if t < radius {
            t as i32 - radius as i32
        } else {
            t as i32 - radius as i32 + 1
        };
        offsets.push(k);
        let mv_base = (t * fields.mv_stride) as usize;
        vectors.push(
            (0..blocks)
                .map(|b| [mv[mv_base + 2 * b], mv[mv_base + 2 * b + 1]])
                .collect(),
        );
        let c_base = (t * fields.conf_stride) as usize;
        confidence.push(conf[c_base..c_base + blocks].to_vec());
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

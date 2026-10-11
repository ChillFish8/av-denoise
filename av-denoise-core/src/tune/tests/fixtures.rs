use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::runtime::TestRuntime;
use crate::collab::geometry::{ref_count, refs_along, strength_map_dims};
use crate::collab::kernels::aggregate::{cross_frame_accum_scale, kaiser_window, weight_scale};
use crate::collab::kernels::fused::STRENGTH_MAP_OFF;
use crate::collab::kernels::transforms::dct_noise_profile;
use crate::collab::{MAX_K, grid_frames, needs_warp_uniform_search};
use crate::nlmeans::motion::THSAD_PIXEL;
use crate::nlmeans::{ChannelMode, NOISE_CURVE_BINS};
use crate::tune::gpu::zero_words;
use crate::tune::sites::{
    AccumulateSite,
    CollabInputs,
    CollabOutputs,
    CollabShape,
    CollabSite,
    CollabValues,
    ConfidenceBinding,
    RegulariseOutputs,
    RegulariseSite,
    WindowPass,
    WindowSite,
    allocate_accum,
};

/// Frames in every fixture ring. The centre is slot 1, with a neighbour either side.
pub(crate) const RING_FRAMES: u32 = 3;

/// Gives hashed content weights spread over most of 0..1 at the fixture patch sizes.
pub(crate) const H2_INV_NORM: f32 = 0.5;

/// A value between 0.2 and 0.8, distinct for every index and seed.
pub(crate) fn hashed_unit(index: u32, seed: u32) -> f32 {
    let mut hash = index.wrapping_mul(0x9E37_79B9) ^ seed.wrapping_mul(0x85EB_CA6B);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x2C1B_3C6D);
    hash ^= hash >> 12;

    let unit = hash as f32 / u32::MAX as f32;
    0.2 + 0.6 * unit
}

/// A ring of `frames` frames of hashed values, stored with `stored_ch` lanes per pixel.
pub(crate) fn hashed_ring(width: u32, height: u32, stored_ch: u32, frames: u32, seed: u32) -> Vec<f32> {
    let len = width * height * stored_ch * frames;
    (0..len).map(|index| hashed_unit(index, seed)).collect()
}

pub(crate) fn upload(client: &ComputeClient<TestRuntime>, values: &[f32]) -> Handle {
    let bytes = f32::as_bytes(values);
    client.create_from_slice(bytes)
}

/// The geometry and switches of one windowed launch.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WindowSpec {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) channels: ChannelMode,
    pub(crate) patch_radius: u32,
    pub(crate) search_radius: u32,
    pub(crate) pair: bool,
    pub(crate) reference: bool,
    pub(crate) confidence: bool,
}

impl WindowSpec {
    pub(crate) fn single(width: u32, height: u32, channels: ChannelMode) -> Self {
        Self {
            width,
            height,
            channels,
            patch_radius: 2,
            search_radius: 3,
            pair: false,
            reference: false,
            confidence: false,
        }
    }

    pub(crate) fn pair(width: u32, height: u32, channels: ChannelMode) -> Self {
        Self {
            pair: true,
            ..Self::single(width, height, channels)
        }
    }
}

/// Confidences that vary by block, on an 8-pixel block grid.
fn confidence_binding(client: &ComputeClient<TestRuntime>, spec: &WindowSpec) -> ConfidenceBinding {
    let step = 8;
    let blocks_x = spec.width.div_ceil(step);
    let blocks_y = spec.height.div_ceil(step);
    let blocks = blocks_x * blocks_y;
    let forward_values: Vec<f32> = (0..blocks).map(|block| hashed_unit(block, 7)).collect();
    let backward_values: Vec<f32> = (0..blocks).map(|block| hashed_unit(block, 8)).collect();
    let forward = upload(client, &forward_values);
    let backward = upload(client, &backward_values);

    ConfidenceBinding {
        enabled: spec.confidence,
        forward,
        backward,
        len: blocks as usize,
        step,
        blocks_x,
        blocks_y,
    }
}

pub(crate) fn window_site(client: &ComputeClient<TestRuntime>, spec: &WindowSpec) -> WindowSite<TestRuntime> {
    let stored_ch = spec.channels.storage_count();
    let pixels = (spec.width * spec.height) as usize;
    let ring = hashed_ring(spec.width, spec.height, stored_ch, RING_FRAMES, 1);
    let input = upload(client, &ring);
    let reference = spec.reference.then(|| {
        let reference_ring = hashed_ring(spec.width, spec.height, stored_ch, RING_FRAMES, 2);
        upload(client, &reference_ring)
    });
    let outputs = allocate_accum(client, pixels * stored_ch as usize, pixels);

    let pass = if spec.pair {
        let confidence = confidence_binding(client, spec);
        WindowPass::Pair {
            frame_fwd: 2,
            frame_bwd: 0,
            noise_offset: 1.0,
            confidence,
        }
    } else {
        let side = 2 * spec.search_radius + 1;
        let offsets_len = (side * side) as usize;
        let offsets: Vec<f32> = (0..offsets_len).map(|index| 0.25 * (index % 5) as f32).collect();
        let offsets = upload(client, &offsets);
        WindowPass::Single { offsets, offsets_len }
    };

    WindowSite {
        client: client.clone(),
        pass,
        input,
        reference,
        ring_len: ring.len(),
        outputs,
        frame_t: 1,
        h2_inv_norm: H2_INV_NORM,
        width: spec.width,
        height: spec.height,
        channels: spec.channels.count(),
        stored_ch,
        patch_radius: spec.patch_radius,
        search_radius: spec.search_radius,
    }
}

/// The geometry of one accumulate launch.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AccumulateSpec {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) channels: ChannelMode,
    pub(crate) q_x: i32,
    pub(crate) q_y: i32,
}

pub(crate) fn accumulate_site(
    client: &ComputeClient<TestRuntime>,
    spec: &AccumulateSpec,
) -> AccumulateSite<TestRuntime> {
    let stored_ch = spec.channels.storage_count();
    let pixels = (spec.width * spec.height) as usize;
    let ring = hashed_ring(spec.width, spec.height, stored_ch, RING_FRAMES, 3);
    let input = upload(client, &ring);
    let weight_values: Vec<f32> = (0..pixels as u32).map(|index| hashed_unit(index, 4)).collect();
    let weights = upload(client, &weight_values);
    let outputs = allocate_accum(client, pixels * stored_ch as usize, pixels);

    AccumulateSite {
        client: client.clone(),
        input,
        ring_len: ring.len(),
        weights,
        outputs,
        frame: 1,
        q_x: spec.q_x,
        q_y: spec.q_y,
        width: spec.width,
        height: spec.height,
        stored_ch,
    }
}

/// Motion block geometry of the collab and regularise fixtures.
pub(crate) const BLK_STEP: u32 = 8;
pub(crate) const BLKSIZE: u32 = 16;

/// The geometry of one `collab_fused` launch over a cross-frame ring.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CollabSpec {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) channels: ChannelMode,
    pub(crate) radius: u32,
}

/// A cross-frame ring of hashed content with motion and confidence fields that vary by block.
///
/// The confidences run from below `c_min` to 1, so some blocks skip their window and the rest are
/// searched. The centre is the middle slot, and the neighbour slots list the frames behind it and
/// then the frames ahead of it, nearest first.
pub(crate) fn collab_site(
    client: &ComputeClient<TestRuntime>,
    spec: &CollabSpec,
) -> CollabSite<TestRuntime, f32> {
    let stored_ch = spec.channels.storage_count();
    let frames = 2 * spec.radius + 1;
    let pixels = (spec.width * spec.height) as usize;
    let ring_values = hashed_ring(spec.width, spec.height, stored_ch, frames, 5);
    let ring = upload(client, &ring_values);

    let blocks_x = spec.width.div_ceil(BLK_STEP);
    let blocks_y = spec.height.div_ceil(BLK_STEP);
    let conf_stride = blocks_x * blocks_y;
    let mv_stride = conf_stride * 2;
    let neighbours = 2 * spec.radius;
    let mut mv_values = vec![0i32; (neighbours * mv_stride).max(2) as usize];
    let mut confidence_values = vec![1.0f32; (neighbours * conf_stride).max(1) as usize];

    for t in 0..neighbours {
        for block in 0..conf_stride {
            let mv_index = (t * mv_stride + block * 2) as usize;
            let conf_index = (t * conf_stride + block) as usize;
            mv_values[mv_index] = (block % 11) as i32 - 5 + t as i32;
            mv_values[mv_index + 1] = 4 - (block % 9) as i32 - t as i32;
            confidence_values[conf_index] = ((block * 7 + t * 3) % 11) as f32 / 10.0;
        }
    }

    let mv_bytes = i32::as_bytes(&mv_values);
    let mv_field = client.create_from_slice(mv_bytes);
    let confidence = upload(client, &confidence_values);

    let centre_slot = spec.radius;
    let behind = (0..spec.radius).map(|offset| centre_slot - 1 - offset);
    let ahead = (0..spec.radius).map(|offset| centre_slot + 1 + offset);
    let mut slot_values: Vec<u32> = behind.chain(ahead).collect();
    if slot_values.is_empty() {
        slot_values.push(0);
    }

    let slot_bytes = u32::as_bytes(&slot_values);
    let neighbour_slots = client.create_from_slice(slot_bytes);

    let sigma_value = 6.0 / 255.0;
    let mut sigma_values = vec![0.0f32; stored_ch as usize];
    sigma_values[..spec.channels.count() as usize].fill(sigma_value);
    let sigma = upload(client, &sigma_values);

    let curve_values = [0.0f32; NOISE_CURVE_BINS];
    let noise_curve = upload(client, &curve_values);
    let (map_cols, map_rows) = strength_map_dims(spec.width, spec.height);
    let map_len = (map_cols * map_rows) as usize;
    let map_values = vec![1.0f32; map_len];
    let strength_map = upload(client, &map_values);
    let profile = dct_noise_profile(0.0);
    let dct_profile = upload(client, &profile);
    let kaiser_taps = kaiser_window(0.0);
    let kaiser = upload(client, &kaiser_taps);

    let spatial_radius = 9;
    let refs = ref_count(spec.width, spec.height);
    let accum_len = pixels * stored_ch as usize * frames as usize;
    let wsum_len = pixels * frames as usize;
    let accum = client.empty(accum_len * size_of::<i32>());
    let wsum = client.empty(wsum_len * size_of::<i32>());
    let group_weight = client.empty(refs * size_of::<f32>());
    zero_words(client, &accum, accum_len);
    zero_words(client, &wsum, wsum_len);
    zero_words(client, &group_weight, refs);

    let inputs = CollabInputs {
        ring: ring.clone(),
        ring_len: ring_values.len(),
        search_ring: ring,
        search_len: stored_ch as usize,
        mv_field,
        mv_len: mv_values.len(),
        confidence,
        conf_len: confidence_values.len(),
        neighbour_slots,
        neighbour_slots_len: slot_values.len(),
        sigma,
        noise_curve,
        strength_map,
        map_len,
        dct_profile,
        kaiser,
    };
    let weight_norm = weight_scale(sigma_value, &profile);
    let accum_scale = cross_frame_accum_scale(spatial_radius, spec.radius);
    let values = CollabValues {
        centre_slot,
        c_min: 0.05,
        lambda_ht: 2.7,
        curve_valid: 0,
        map_mode: STRENGTH_MAP_OFF,
        weight_scale: weight_norm,
        accum_scale,
        pool_ratio: 0.0,
    };
    let warp_uniform = needs_warp_uniform_search(client);
    let frames_per_volume = grid_frames(spec.radius);
    let refs_x = refs_along(spec.width);
    let refs_y = refs_along(spec.height);
    let shape = CollabShape {
        warp_uniform,
        f16_search: false,
        radius: spec.radius,
        grid_frames: frames_per_volume,
        refine: 2,
        mv_stride,
        conf_stride,
        blk_step: BLK_STEP,
        blksize: BLKSIZE,
        blocks_x,
        blocks_y,
        width: spec.width,
        height: spec.height,
        channels: spec.channels.count(),
        k_max: MAX_K,
        stored_ch,
        spatial_radius,
        refs_x,
        refs_y,
        map_cols,
        map_rows,
        pooled: false,
    };
    let outputs = CollabOutputs {
        accum,
        wsum,
        group_weight,
        accum_len,
        wsum_len,
        refs,
    };

    CollabSite {
        client: client.clone(),
        inputs,
        values,
        shape,
        outputs,
        search: PhantomData,
    }
}

/// The content a regularise fixture scores its candidates against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RegulariseContent {
    /// Distinct values everywhere, with the neighbour shifted by `(2, 1)` and lightly perturbed.
    Textured,
    /// Every pixel the same, so every candidate scores a SAD of zero.
    Flat,
    /// A 4x4 tile repeated across both frames, so candidates a whole tile apart score equal SADs.
    Periodic,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RegulariseSpec {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) content: RegulariseContent,
    pub(crate) field_lambda: f32,
}

fn regularise_planes(spec: &RegulariseSpec) -> (Vec<f32>, Vec<f32>) {
    let pixels = (spec.width * spec.height) as usize;

    match spec.content {
        RegulariseContent::Flat => (vec![0.5f32; pixels], vec![0.5f32; pixels]),
        RegulariseContent::Periodic => {
            let tile: Vec<f32> = (0..16).map(|index| hashed_unit(index, 11)).collect();
            let plane: Vec<f32> = (0..pixels as u32)
                .map(|index| {
                    let x = index % spec.width;
                    let y = index / spec.width;
                    tile[((y % 4) * 4 + x % 4) as usize]
                })
                .collect();
            (plane.clone(), plane)
        },
        RegulariseContent::Textured => {
            let centre: Vec<f32> = (0..pixels as u32).map(|index| hashed_unit(index, 12)).collect();
            let neighbour: Vec<f32> = (0..pixels as u32)
                .map(|index| {
                    let x = (index % spec.width) as i32;
                    let y = (index / spec.width) as i32;
                    let source_x = (x - 2).clamp(0, spec.width as i32 - 1) as u32;
                    let source_y = (y - 1).clamp(0, spec.height as i32 - 1) as u32;
                    let source = centre[(source_y * spec.width + source_x) as usize];
                    let perturbation = (hashed_unit(index, 13) - 0.5) * 0.02;
                    source + perturbation
                })
                .collect();
            (centre, neighbour)
        },
    }
}

pub(crate) fn regularise_site(
    client: &ComputeClient<TestRuntime>,
    spec: &RegulariseSpec,
) -> RegulariseSite<TestRuntime> {
    let (centre_values, neighbour_values) = regularise_planes(spec);
    let centre = upload(client, &centre_values);
    let neighbour = upload(client, &neighbour_values);

    let blocks_x = spec.width.div_ceil(BLK_STEP);
    let blocks_y = spec.height.div_ceil(BLK_STEP);
    let blocks = (blocks_x * blocks_y) as usize;
    let field_values: Vec<i32> = (0..2 * blocks as u32)
        .map(|index| (index % 7) as i32 - 3 + ((index / 14) % 3) as i32)
        .collect();
    let field_bytes = i32::as_bytes(&field_values);
    let mv_in = client.create_from_slice(field_bytes);

    let vectors = client.empty(2 * blocks * size_of::<i32>());
    let confidence = client.empty(blocks * size_of::<f32>());
    zero_words(client, &vectors, 2 * blocks);
    zero_words(client, &confidence, blocks);
    let outputs = RegulariseOutputs {
        vectors,
        confidence,
        blocks,
    };

    let block_area = (BLKSIZE * BLKSIZE) as f32;
    let thsad = block_area * THSAD_PIXEL;

    RegulariseSite {
        client: client.clone(),
        centre,
        neighbour,
        level_len: centre_values.len(),
        mv_in,
        outputs,
        lambda_pixel: spec.field_lambda * thsad,
        sad_noise_floor: 0.0,
        thsad,
        width: spec.width,
        height: spec.height,
        blksize: BLKSIZE,
        step: BLK_STEP,
        blocks_x,
        blocks_y,
    }
}

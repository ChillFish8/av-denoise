use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::Handle;

use super::runtime::TestRuntime;
use crate::collab::geometry::{ref_count, refs_along, strength_map_dims};
use crate::collab::kernels::aggregate::{cross_frame_accum_scale, kaiser_window, weight_scale};
use crate::collab::kernels::fused::STRENGTH_MAP_OFF;
use crate::collab::kernels::transforms::dct_noise_profile;
use crate::collab::{MAX_K, grid_frames, needs_warp_uniform_search};
use crate::engine::SampleFormat;
use crate::nl4d::nl4d_pool_ratio;
use crate::nlmeans::motion::THSAD_PIXEL;
use crate::nlmeans::{ChannelMode, NOISE_CURVE_BINS, StorageAlign};
use crate::tune::gpu::{read_f32, zero_words};
use crate::tune::sites::{
    AccumulateSite,
    CollabFilterSite,
    CollabInputs,
    CollabOutputs,
    CollabSearchSite,
    CollabShape,
    CollabSite,
    CollabValues,
    CollabVariant,
    ConfidenceBinding,
    RegulariseOutputs,
    RegulariseSite,
    SearchOutputs,
    WindowCodes,
    WindowPass,
    WindowSite,
    allocate_accum,
};
use crate::tune::{IntegerCodes, SiteCandidate};

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
    /// With codes the input ring holds whole codes of that depth, and the site reads their code ring.
    pub(crate) codes: Option<IntegerCodes>,
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
            codes: None,
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

/// The hashed ring rounded to `codes`, as normalised values and as the code ring of the same layout.
fn code_exact_ring(
    client: &ComputeClient<TestRuntime>,
    ring: &[f32],
    channels: u32,
    codes: IntegerCodes,
) -> (Vec<f32>, WindowCodes) {
    let code_values = quantised_codes(ring, codes);
    let max_value = codes.max_code() as f32;
    let quantised: Vec<f32> = code_values.iter().map(|&code| code as f32 / max_value).collect();
    let code_bytes = u16::as_bytes(&code_values);
    let code_ring = client.create_from_slice(code_bytes);
    let inv_scale2 = codes.inv_scale2(channels);
    let window_codes = WindowCodes {
        ring: code_ring,
        codes,
        inv_scale2,
    };

    (quantised, window_codes)
}

pub(crate) fn window_site(client: &ComputeClient<TestRuntime>, spec: &WindowSpec) -> WindowSite<TestRuntime> {
    let stored_ch = spec.channels.storage_count();
    let channels = spec.channels.count();
    let pixels = (spec.width * spec.height) as usize;
    let hashed = hashed_ring(spec.width, spec.height, stored_ch, RING_FRAMES, 1);
    let (ring, codes) = match spec.codes {
        Some(codes) => {
            let (quantised, window_codes) = code_exact_ring(client, &hashed, channels, codes);
            (quantised, Some(window_codes))
        },
        None => (hashed, None),
    };
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
        codes,
        ring_len: ring.len(),
        outputs,
        frame_t: 1,
        h2_inv_norm: H2_INV_NORM,
        width: spec.width,
        height: spec.height,
        channels,
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

/// The geometry and switches of one `collab_fused` launch over a cross-frame ring.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CollabSpec {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) channels: ChannelMode,
    pub(crate) radius: u32,
    pub(crate) spatial_radius: u32,
    pub(crate) pooled: bool,
    /// Every pixel the same, so every candidate distance ties.
    pub(crate) flat: bool,
    pub(crate) f16_search: bool,
    /// The confidence a motion block needs before its window is searched.
    pub(crate) c_min: f32,
}

impl CollabSpec {
    /// Hashed content at spatial radius 9, with the pooled threshold and the f16 search off.
    pub(crate) fn new(width: u32, height: u32, channels: ChannelMode, radius: u32) -> Self {
        Self {
            width,
            height,
            channels,
            radius,
            spatial_radius: 9,
            pooled: false,
            flat: false,
            f16_search: false,
            c_min: 0.05,
        }
    }
}

/// The ring a collab fixture searches.
fn collab_ring(spec: &CollabSpec) -> Vec<f32> {
    let stored_ch = spec.channels.storage_count();
    let frames = 2 * spec.radius + 1;

    if spec.flat {
        let len = spec.width * spec.height * stored_ch * frames;
        return vec![0.5f32; len as usize];
    }

    hashed_ring(spec.width, spec.height, stored_ch, frames, 5)
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
    let ring_values = collab_ring(spec);
    collab_site_with_ring(client, spec, &ring_values)
}

/// [collab_site] over `ring_values`, which hold `2 * spec.radius + 1` frames in slot order.
pub(crate) fn collab_site_with_ring(
    client: &ComputeClient<TestRuntime>,
    spec: &CollabSpec,
    ring_values: &[f32],
) -> CollabSite<TestRuntime, f32> {
    let stored_ch = spec.channels.storage_count();
    let frames = 2 * spec.radius + 1;
    let pixels = (spec.width * spec.height) as usize;
    let ring = upload(client, ring_values);

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

    let spatial_radius = spec.spatial_radius;
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
        search_ring: ring.clone(),
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
        codes: ring,
        codes_len: stored_ch as usize,
    };
    let weight_norm = weight_scale(sigma_value, &profile);
    let accum_scale = cross_frame_accum_scale(spatial_radius, spec.radius);
    let pool_ratio = if spec.pooled {
        nl4d_pool_ratio(spec.channels)
    } else {
        0.0
    };
    let values = CollabValues {
        centre_slot,
        c_min: spec.c_min,
        lambda_ht: 2.7,
        curve_valid: 0,
        map_mode: STRENGTH_MAP_OFF,
        weight_scale: weight_norm,
        accum_scale,
        pool_ratio,
    };
    let frames_per_volume = grid_frames(spec.radius);
    let refs_x = refs_along(spec.width);
    let refs_y = refs_along(spec.height);
    let uniform_walk_only = needs_warp_uniform_search(client);
    let shape = CollabShape {
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
        pooled: spec.pooled,
        uniform_walk_only,
        packed: false,
        codes: None,
    };
    let outputs = CollabOutputs {
        accum,
        wsum,
        group_weight,
        accum_len,
        wsum_len,
        refs,
        packed: false,
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

/// [collab_site] scoring its candidates from an f16 copy of the ring.
pub(crate) fn collab_site_f16(
    client: &ComputeClient<TestRuntime>,
    spec: &CollabSpec,
) -> CollabSite<TestRuntime, half::f16> {
    let site = collab_site(client, spec);
    let ring_values = collab_ring(spec);
    let search_values: Vec<half::f16> = ring_values
        .iter()
        .map(|&value| half::f16::from_f32(value))
        .collect();
    let search_bytes = half::f16::as_bytes(&search_values);
    let search_ring = client.create_from_slice(search_bytes);

    let inputs = CollabInputs {
        search_ring,
        search_len: search_values.len(),
        ..site.inputs
    };
    let shape = CollabShape {
        f16_search: true,
        ..site.shape
    };

    CollabSite {
        client: site.client,
        inputs,
        values: site.values,
        shape,
        outputs: site.outputs,
        search: PhantomData,
    }
}

/// `site` with the packed accumulator layout and zeroed outputs of that layout.
pub(crate) fn packed_twin(site: &CollabSite<TestRuntime, f32>) -> CollabSite<TestRuntime, f32> {
    let client = &site.client;
    let pixels = site.outputs.wsum_len;
    let stored = site.shape.stored_ch as usize;
    let rest_len = (pixels * (stored - 1)).max(1);
    let packed = client.empty(pixels * size_of::<i64>());
    let rest = client.empty(rest_len * size_of::<i32>());
    zero_words(client, &packed, 2 * pixels);
    zero_words(client, &rest, rest_len);

    let mut twin = site.clone();
    twin.shape.packed = true;
    twin.outputs = CollabOutputs {
        accum: rest,
        wsum: packed,
        accum_len: rest_len,
        packed: true,
        ..site.outputs.clone()
    };

    twin
}

/// The search and filter sites for the same pass as `site`, sharing a zeroed member buffer.
pub(crate) fn split_sites<S: Float>(
    site: &CollabSite<TestRuntime, S>,
) -> (CollabSearchSite<TestRuntime, S>, CollabFilterSite<TestRuntime>) {
    let client = &site.client;
    let search_outputs = SearchOutputs::zeroed(client, site.outputs.refs);
    let members = search_outputs.members.clone();

    let search = CollabSearchSite {
        client: client.clone(),
        inputs: site.inputs.clone(),
        values: site.values,
        shape: site.shape,
        outputs: search_outputs,
        search: PhantomData,
    };
    let filter = CollabFilterSite {
        client: client.clone(),
        inputs: site.inputs.clone(),
        members,
        values: site.values,
        shape: site.shape,
        outputs: site.outputs.clone(),
    };

    (search, filter)
}

/// The wire format a source of `codes` arrives in.
pub(crate) fn code_format(codes: IntegerCodes) -> SampleFormat {
    if codes.depth == 8 {
        return SampleFormat::U8;
    }

    SampleFormat::U16 {
        depth: codes.depth as u8,
    }
}

/// Uploads `plane` in the wire layout of `codes`, one byte per 8-bit code and two otherwise.
pub(crate) fn upload_codes(
    client: &ComputeClient<TestRuntime>,
    plane: &[u16],
    codes: IntegerCodes,
) -> Handle {
    if codes.depth == 8 {
        let bytes: Vec<u8> = plane.iter().map(|&code| code as u8).collect();
        return client.create_from_slice(&bytes);
    }

    let bytes = u16::as_bytes(plane);
    client.create_from_slice(bytes)
}

/// Each value of `values` as the code `codes` rounds it to, clamped between 0 and the top code.
pub(crate) fn quantised_codes(values: &[f32], codes: IntegerCodes) -> Vec<u16> {
    let max_value = codes.max_code() as f32;

    values
        .iter()
        .map(|&value| {
            let clamped = value.clamp(0.0, 1.0);
            let code = (clamped * max_value).round();
            code as u16
        })
        .collect()
}

/// The split sites for `fused`, reading its ring rounded to `codes` and the matching code ring.
///
/// The ring is read back and rounded on the host, so the float search and the integer search see the
/// same content.
pub(crate) fn integer_split_sites(
    fused: &CollabSite<TestRuntime, f32>,
    codes: IntegerCodes,
) -> (CollabSearchSite<TestRuntime, f32>, CollabFilterSite<TestRuntime>) {
    let client = &fused.client;
    let inputs = &fused.inputs;
    let ring_values = read_f32(client, &inputs.ring, inputs.ring_len).expect("the fixture ring reads back");
    let code_values = quantised_codes(&ring_values, codes);
    let max_value = codes.max_code() as f32;
    let quantised: Vec<f32> = code_values.iter().map(|&code| code as f32 / max_value).collect();
    let ring = upload(client, &quantised);
    let code_bytes = u16::as_bytes(&code_values);
    let code_ring = client.create_from_slice(code_bytes);

    let quantised_inputs = CollabInputs {
        ring: ring.clone(),
        search_ring: ring,
        codes: code_ring,
        codes_len: code_values.len(),
        ..inputs.clone()
    };
    let integer_shape = CollabShape {
        codes: Some(codes),
        ..fused.shape
    };
    let quantised_site = CollabSite {
        inputs: quantised_inputs,
        shape: integer_shape,
        ..fused.clone()
    };

    split_sites(&quantised_site)
}

/// The content a regularise fixture scores its candidates against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RegulariseContent {
    /// Distinct values everywhere. Neighbour `n` is the centre shifted by `(2 - n, 1 + n % 2)` and
    /// lightly perturbed.
    Textured,
    /// Every pixel the same, so every candidate scores a SAD of zero.
    Flat,
    /// A 4x4 tile repeated across every frame, so candidates a whole tile apart score equal SADs.
    Periodic,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RegulariseSpec {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) content: RegulariseContent,
    pub(crate) field_lambda: f32,
    pub(crate) neighbours: u32,
}

/// The centre frame and one frame per neighbour.
///
/// Textured neighbours are each shifted by their own amount. Flat and periodic neighbours repeat
/// the centre, so their candidates keep tying pixel for pixel.
fn regularise_frames(spec: &RegulariseSpec) -> Vec<Vec<f32>> {
    let pixels = (spec.width * spec.height) as usize;
    let tile: Vec<f32> = (0..16).map(|index| hashed_unit(index, 11)).collect();
    let centre: Vec<f32> = match spec.content {
        RegulariseContent::Flat => vec![0.5f32; pixels],
        RegulariseContent::Periodic => (0..pixels as u32)
            .map(|index| {
                let x = index % spec.width;
                let y = index / spec.width;
                tile[((y % 4) * 4 + x % 4) as usize]
            })
            .collect(),
        RegulariseContent::Textured => (0..pixels as u32).map(|index| hashed_unit(index, 12)).collect(),
    };

    let mut frames = vec![centre.clone()];

    if spec.content != RegulariseContent::Textured {
        for _ in 0..spec.neighbours {
            let duplicate = centre.clone();
            frames.push(duplicate);
        }

        return frames;
    }

    for neighbour in 0..spec.neighbours {
        let shift_x = 2 - neighbour as i32;
        let shift_y = 1 + (neighbour as i32 % 2);
        let moved: Vec<f32> = (0..pixels as u32)
            .map(|index| {
                let x = (index % spec.width) as i32;
                let y = (index / spec.width) as i32;
                let source_x = (x - shift_x).clamp(0, spec.width as i32 - 1) as u32;
                let source_y = (y - shift_y).clamp(0, spec.height as i32 - 1) as u32;
                let source = centre[(source_y * spec.width + source_x) as usize];
                let perturbation = (hashed_unit(index + neighbour * 7919, 13) - 0.5) * 0.02;
                source + perturbation
            })
            .collect();
        frames.push(moved);
    }

    frames
}

/// The pyramid slot a regularise fixture stores neighbour `neighbour` in.
///
/// The neighbours fill slots `0..neighbours` in reverse, so the table is never the identity once
/// there are two or more. The centre takes the last slot.
pub(crate) fn regularise_neighbour_slot(neighbours: u32, neighbour: u32) -> u32 {
    neighbours - 1 - neighbour
}

/// A regularise site over `spec.neighbours` neighbours laid out with the production padding.
pub(crate) fn regularise_site(
    client: &ComputeClient<TestRuntime>,
    spec: &RegulariseSpec,
) -> RegulariseSite<TestRuntime> {
    let align = StorageAlign::from_client(client);
    let pixels = (spec.width * spec.height) as usize;
    let slot_pixels = align.pad_elems::<f32>(pixels);
    let frames = regularise_frames(spec);

    let neighbour_table: Vec<u32> = (0..spec.neighbours)
        .map(|neighbour| regularise_neighbour_slot(spec.neighbours, neighbour))
        .collect();

    let mut pyramid_values = vec![0.0f32; frames.len() * slot_pixels];
    let neighbour_frames = frames.iter().skip(1);
    for (&slot, frame) in neighbour_table.iter().zip(neighbour_frames) {
        let start = slot as usize * slot_pixels;
        pyramid_values[start..start + pixels].copy_from_slice(frame);
    }

    let centre_start = spec.neighbours as usize * slot_pixels;
    pyramid_values[centre_start..centre_start + pixels].copy_from_slice(&frames[0]);

    let pyramid = upload(client, &pyramid_values);
    let table_bytes = u32::as_bytes(&neighbour_table);
    let neighbour_slots = client.create_from_slice(table_bytes);

    let blocks_x = spec.width.div_ceil(BLK_STEP);
    let blocks_y = spec.height.div_ceil(BLK_STEP);
    let blocks = (blocks_x * blocks_y) as usize;
    let mv_stride = align.pad_elems::<i32>(2 * blocks);
    let conf_stride = align.pad_elems::<f32>(blocks);
    let neighbours = spec.neighbours as usize;

    let mut field_values = vec![0i32; neighbours * mv_stride];
    for neighbour in 0..neighbours {
        for index in 0..2 * blocks {
            let value = (index % 7) as i32 - 3 + ((index / 14) % 3) as i32 + neighbour as i32;
            field_values[neighbour * mv_stride + index] = value;
        }
    }

    let field_bytes = i32::as_bytes(&field_values);
    let mv_in = client.create_from_slice(field_bytes);

    let vectors = client.empty(neighbours * mv_stride * size_of::<i32>());
    let confidence = client.empty(neighbours * conf_stride * size_of::<f32>());
    zero_words(client, &vectors, neighbours * mv_stride);
    zero_words(client, &confidence, neighbours * conf_stride);
    let outputs = RegulariseOutputs {
        vectors,
        confidence,
        blocks,
        neighbours: spec.neighbours,
        mv_stride: mv_stride as u32,
        conf_stride: conf_stride as u32,
    };

    let block_area = (BLKSIZE * BLKSIZE) as f32;
    let thsad = block_area * THSAD_PIXEL;

    RegulariseSite {
        client: client.clone(),
        pyramid,
        pyramid_len: pyramid_values.len(),
        centre_slot: spec.neighbours,
        neighbour_slots,
        slot_pixels: slot_pixels as u32,
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

/// Chunk `chunk` of the listed collab variants cut into `chunk_count` equal chunks, counting from 0.
pub(crate) fn variant_chunk(chunk: usize, chunk_count: usize) -> &'static [CollabVariant] {
    let listed = CollabVariant::ALL.len();
    assert_eq!(
        listed % chunk_count,
        0,
        "{listed} variants do not split into {chunk_count} equal chunks"
    );

    let chunk_len = listed / chunk_count;
    let start = chunk * chunk_len;
    let end = start + chunk_len;

    &CollabVariant::ALL[start..end]
}

/// Writes one test per named chunk, each passing `subject` and that chunk of the listed variants
/// to `check`.
macro_rules! variant_chunk_tests {
    (
        $check:ident($subject:expr), $chunk_count:literal chunks,
        [$($name:ident => $chunk:literal),+ $(,)?]
    ) => {
        $(
            #[test]
            fn $name() {
                let subject = $subject;
                let variants = $crate::tune::tests::fixtures::variant_chunk($chunk, $chunk_count);
                $check(subject, variants);
            }
        )+
    };
}

pub(crate) use variant_chunk_tests;

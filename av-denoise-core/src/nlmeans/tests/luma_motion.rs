use cubecl::prelude::*;
use cubecl::server::Handle;

use super::helpers::make_client;
use super::models::{ModelFitter, PanShape, matched_flags, mean_over_blocks, pan_clip, stays_inside};
use crate::bench_api::HostIo;
use crate::engine::align::StorageAlign;
use crate::motion::{LumaMotionRing, MotionCtx, RingRole, SharedMotion, Subsampling, pair_byte_offset};
use crate::nlmeans::denoiser::LOOKAHEAD_FRAMES;
use crate::nlmeans::*;
use crate::tune::tests::fixtures::noisy_field_over;

const WIDTH: u32 = 192;
const HEIGHT: u32 = 96;
const CHROMA_WIDTH: u32 = WIDTH / 2;
const CHROMA_HEIGHT: u32 = HEIGHT / 2;
/// An even luma pan, so the chroma pan is a whole number of chroma pixels.
const PAN: u32 = 4;
const SIGMA: f32 = 2.0 / 255.0;
const TEMPORAL_RADIUS: u32 = 1;
const FRAMES: usize = 12 + LOOKAHEAD_FRAMES as usize;
/// Chroma pixels a scored block keeps from the plane's edge after moving.
const MARGIN: u32 = 2;
/// The pair-field direction from the newer frame to the older one.
const NEWER_TO_OLDER: u32 = 1;
const FOUR_TWO_ZERO: Subsampling = Subsampling {
    shift_x: 1,
    shift_y: 1,
};

/// Interleaved chroma frames of flat grey plus grain, so no chroma search can find a pan.
pub(crate) fn flat_chroma(width: u32, height: u32, sigma: f32, count: usize) -> Vec<Vec<f32>> {
    let flat = vec![0.5f32; (width * height) as usize];

    (0..count as u32)
        .map(|index| {
            let u_plane = noisy_field_over(&flat, width, height, sigma, 2 * index + 1);
            let v_plane = noisy_field_over(&flat, width, height, sigma, 2 * index + 2);

            u_plane
                .iter()
                .zip(&v_plane)
                .flat_map(|(&u_value, &v_value)| [u_value, v_value])
                .collect()
        })
        .collect()
}

/// The luma planes of a 4:2:0 textured clip panning left by `pan` luma pixels a frame.
pub(crate) fn panning_luma(width: u32, height: u32, pan: u32, sigma: f32, count: usize) -> Vec<Vec<f32>> {
    let shape = PanShape {
        width,
        height,
        chroma_scale: (2, 2),
        shift: (pan, 0),
        sigma,
    };
    let frames = pan_clip(shape, count, 11);

    frames.into_iter().map(|frame| frame.luma).collect()
}

fn params(channels: ChannelMode) -> NlmParams {
    NlmParams {
        temporal_radius: TEMPORAL_RADIUS,
        search_radius: 2,
        patch_radius: 2,
        channels,
        prefilter: PrefilterMode::None,
        motion_compensation: MotionCompensationMode::mvtools_default(),
        hq: Some(HqParams::default()),
        ..NlmParams::default()
    }
}

fn chroma_ctx(client: &Client) -> MotionCtx {
    let align = StorageAlign::from_client(client);
    let mode = MotionCompensationMode::mvtools_default();

    MotionCtx::new(mode, CHROMA_WIDTH, CHROMA_HEIGHT, align).expect("a searching mode")
}

fn shared_motion(fitter: &ModelFitter, role: RingRole) -> SharedMotion {
    let motion = fitter.motion();

    SharedMotion {
        luma_motion: Some(role),
        ..motion
    }
}

/// The chroma pass's motion after the last frame, its window fields and its look-ahead pair.
struct ChromaMotion {
    window: Vec<i32>,
    lookahead: Vec<i32>,
}

fn read_words(client: &Client, handle: &Handle) -> Vec<i32> {
    let bytes = client.read_one(handle.clone()).expect("motion readback failed");

    i32::from_bytes(&bytes).to_vec()
}

/// Streams a panning luma clip and a flat chroma clip through nlmeans, with the chroma pass
/// reading luma motion when `shared`.
fn chroma_motion(shared: bool) -> ChromaMotion {
    let client = make_client();
    let mode = MotionCompensationMode::mvtools_default();
    let mut fitter = ModelFitter::new(&client, WIDTH, HEIGHT, mode);
    let chroma_ctx = chroma_ctx(&client);
    let ring = LumaMotionRing::new(&client, &chroma_ctx, TEMPORAL_RADIUS, FOUR_TWO_ZERO);
    let mut luma = NlmDenoiser::new(&client, params(ChannelMode::Luma), WIDTH, HEIGHT);
    let mut chroma = NlmDenoiser::new(&client, params(ChannelMode::Chroma), CHROMA_WIDTH, CHROMA_HEIGHT);

    if shared {
        let writer = RingRole::Writer(ring.clone());
        let reader = RingRole::Reader(ring);
        let luma_motion = shared_motion(&fitter, writer);
        let chroma_motion = shared_motion(&fitter, reader);
        luma.set_shared_motion(luma_motion);
        chroma.set_shared_motion(chroma_motion);
    }

    let luma_clip = panning_luma(WIDTH, HEIGHT, PAN, SIGMA, FRAMES);
    let chroma_clip = flat_chroma(CHROMA_WIDTH, CHROMA_HEIGHT, SIGMA, FRAMES);
    for (luma_frame, chroma_frame) in luma_clip.iter().zip(&chroma_clip) {
        fitter.fit(luma_frame);
        luma.push_frame(luma_frame);
        luma.denoise().expect("luma denoise");
        chroma.push_frame(chroma_frame);
        chroma.denoise().expect("chroma denoise");
    }

    let window_field = chroma.mv_field_buf.as_ref().expect("a motion field");
    let window = read_words(&client, window_field);

    let buffers = chroma
        .lookahead
        .as_ref()
        .expect("a look-ahead")
        .buffers_for_test();
    let pair_field = buffers
        .into_iter()
        .find_map(|(name, handle)| (name == "look-ahead pair field").then_some(handle))
        .flatten()
        .expect("a look-ahead pair field");
    let lookahead = read_words(&client, pair_field);

    ChromaMotion { window, lookahead }
}

/// The share of blocks inside both frames whose vector in `field` lies within a chroma pixel of a
/// pan of `shift_x` chroma pixels.
fn followed_share(motion_ctx: &MotionCtx, field: &[i32], shift_x: i32) -> f32 {
    let blocks = (motion_ctx.blocks_x * motion_ctx.blocks_y) as usize;
    let (vectors, _) = field[..2 * blocks].as_chunks::<2>();
    let matched = matched_flags(vectors, [shift_x, 0]);
    let inside = |col: u32, row: u32| {
        let origin_x = col * motion_ctx.step;
        let origin_y = row * motion_ctx.step;
        let inside_x = stays_inside(origin_x, motion_ctx.blksize, shift_x, CHROMA_WIDTH, MARGIN);
        let inside_y = stays_inside(origin_y, motion_ctx.blksize, 0, CHROMA_HEIGHT, MARGIN);
        inside_x && inside_y
    };

    mean_over_blocks(&matched, motion_ctx.blocks_x, motion_ctx.blocks_y, inside)
}

/// The followed share of each window neighbour, oldest first.
fn window_shares(motion_ctx: &MotionCtx, motion: &ChromaMotion) -> Vec<f32> {
    let radius = TEMPORAL_RADIUS as i32;
    let offsets = (-radius..0).chain(1..=radius);
    let stride = motion_ctx.mv_stride() as usize;

    offsets
        .enumerate()
        .map(|(neighbour, offset)| {
            let shift_x = -((PAN / 2) as i32) * offset;
            let field = &motion.window[neighbour * stride..];
            followed_share(motion_ctx, field, shift_x)
        })
        .collect()
}

/// The followed share of the look-ahead pair's newer-to-older field.
fn lookahead_share(motion_ctx: &MotionCtx, motion: &ChromaMotion) -> f32 {
    let offset_bytes = pair_byte_offset(motion_ctx, 0, NEWER_TO_OLDER);
    let offset = offset_bytes as usize / size_of::<i32>();
    let field = &motion.lookahead[offset..];
    let shift_x = (PAN / 2) as i32;

    followed_share(motion_ctx, field, shift_x)
}

#[test]
fn nlmeans_chroma_uses_luma_vectors() {
    let client = make_client();
    let motion_ctx = chroma_ctx(&client);

    let unshared = chroma_motion(false);
    let unshared_shares = window_shares(&motion_ctx, &unshared);
    let misses = unshared_shares.iter().any(|&share| share < 0.5);
    assert!(
        misses,
        "test premise: a chroma search on flat chroma misses the pan, got {unshared_shares:?}"
    );

    let shared = chroma_motion(true);
    let shares = window_shares(&motion_ctx, &shared);
    for (neighbour, &share) in shares.iter().enumerate() {
        assert_eq!(share, 1.0, "neighbour {neighbour}, all {shares:?}");
    }
}

#[test]
fn the_chroma_lookahead_follows_luma_motion() {
    let client = make_client();
    let motion_ctx = chroma_ctx(&client);

    let unshared = chroma_motion(false);
    let unshared_share = lookahead_share(&motion_ctx, &unshared);
    assert!(
        unshared_share < 0.5,
        "test premise: a chroma search on flat chroma misses the pan, got {unshared_share}"
    );

    let shared = chroma_motion(true);
    let share = lookahead_share(&motion_ctx, &shared);
    assert_eq!(share, 1.0);
}

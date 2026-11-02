use cubecl::prelude::*;

use super::helpers::make_client;
use crate::bench_api::HostIo;
use crate::engine::align::StorageAlign;
use crate::motion::{LumaMotionRing, MotionCtx, RingRole, SharedMotion, Subsampling};
use crate::nl4d::{MotionSnapshot, Nl4dDenoiser, Nl4dParams};
use crate::nlmeans::denoiser::LOOKAHEAD_FRAMES;
use crate::nlmeans::tests::luma_motion::{flat_chroma, panning_luma};
use crate::nlmeans::tests::models::{ModelFitter, matched_flags, mean_over_blocks, stays_inside};
use crate::nlmeans::{ChannelMode, NlmParams};

const WIDTH: u32 = 192;
const HEIGHT: u32 = 96;
const CHROMA_WIDTH: u32 = WIDTH / 2;
const CHROMA_HEIGHT: u32 = HEIGHT / 2;
/// An even luma pan, so the chroma pan is a whole number of chroma pixels.
const PAN: u32 = 4;
const SIGMA: f32 = 2.0 / 255.0;
const FRAMES: usize = 12 + LOOKAHEAD_FRAMES as usize;
/// Chroma pixels a scored block keeps from the plane's edge after moving.
const MARGIN: u32 = 2;
const FOUR_TWO_ZERO: Subsampling = Subsampling {
    shift_x: 1,
    shift_y: 1,
};

fn chroma_params() -> Nl4dParams {
    let defaults = Nl4dParams::default();
    let nlm = NlmParams {
        channels: ChannelMode::Chroma,
        ..defaults.nlm
    };

    Nl4dParams { nlm, ..defaults }
}

fn ring(client: &Client, params: &Nl4dParams) -> LumaMotionRing {
    let align = StorageAlign::from_client(client);
    let mode = params.nlm.motion_compensation;
    let chroma_ctx = MotionCtx::new(mode, CHROMA_WIDTH, CHROMA_HEIGHT, align).expect("a searching mode");

    LumaMotionRing::new(client, &chroma_ctx, params.temporal_radius, FOUR_TWO_ZERO)
}

/// Streams a panning luma clip and a flat chroma clip, with the chroma pass reading luma motion when
/// `shared`, and returns the chroma pass's last motion field.
fn chroma_motion(shared: bool) -> MotionSnapshot {
    let client = make_client();
    let luma_params = Nl4dParams::default();
    let chroma_params = chroma_params();
    let mode = luma_params.nlm.motion_compensation;
    let mut fitter = ModelFitter::new(&client, WIDTH, HEIGHT, mode);
    let ring = ring(&client, &chroma_params);
    let mut luma = Nl4dDenoiser::new(&client, luma_params, WIDTH, HEIGHT).expect("luma pass");
    let mut chroma =
        Nl4dDenoiser::new(&client, chroma_params, CHROMA_WIDTH, CHROMA_HEIGHT).expect("chroma pass");

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

    chroma.motion_snapshot().expect("a chroma pass ran")
}

/// The share of blocks inside both frames whose vector is within a chroma pixel of the mapped pan,
/// for each neighbour.
fn followed_shares(snapshot: &MotionSnapshot) -> Vec<f32> {
    let fields = snapshot.offsets.iter().zip(&snapshot.vectors);

    fields
        .map(|(&offset, vectors)| {
            let shift_x = -((PAN / 2) as i32) * offset;
            let matched = matched_flags(vectors, [shift_x, 0]);
            let inside = |col: u32, row: u32| {
                let origin_x = col * snapshot.step;
                let origin_y = row * snapshot.step;
                let inside_x = stays_inside(origin_x, snapshot.blksize, shift_x, CHROMA_WIDTH, MARGIN);
                let inside_y = stays_inside(origin_y, snapshot.blksize, 0, CHROMA_HEIGHT, MARGIN);
                inside_x && inside_y
            };

            mean_over_blocks(&matched, snapshot.blocks_x, snapshot.blocks_y, inside)
        })
        .collect()
}

/// The camera models of `fitter`, playing `role` in a luma motion ring.
fn shared_motion(fitter: &ModelFitter, role: RingRole) -> SharedMotion {
    let motion = fitter.motion();

    SharedMotion {
        luma_motion: Some(role),
        ..motion
    }
}

#[test]
fn the_chroma_pass_uses_luma_vectors() {
    let unshared = chroma_motion(false);
    let unshared_shares = followed_shares(&unshared);
    let misses = unshared_shares.iter().any(|&share| share < 0.5);
    assert!(
        misses,
        "test premise: a chroma search on flat chroma misses the pan, got {unshared_shares:?}"
    );

    let shared = chroma_motion(true);
    let shares = followed_shares(&shared);
    for (neighbour, &share) in shares.iter().enumerate() {
        assert_eq!(share, 1.0, "neighbour {neighbour}, all {shares:?}");
    }
}

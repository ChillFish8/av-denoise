use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::collab::geometry::{member_buf_len, ref_count, refs_along};
use crate::collab::kernels::aggregate::{collab_normalise, collab_zero_accum, weight_scale};
use crate::collab::kernels::filter_ht::collab_filter_ht;
use crate::collab::kernels::filter_wiener::collab_filter_wiener;
use crate::collab::kernels::group::collab_group_spatial;
use crate::collab::kernels::transforms::dct_noise_profile;
use crate::collab::{CollabParams, PATCH_AREA, PATCH_SIZE};
use crate::nlmeans::{BLOCK_X, BLOCK_Y, ChannelMode};

/// The per-channel distance scale collaborative grouping and the noise
/// floor use, 3 for luma, 1.5 for chroma, 1 for full YUV. Mirrors
/// `nlmeans::params::channel_scale`, which is private to that module, so
/// the collaborative pipeline keeps its own copy rather than reaching
/// across module boundaries for one small formula.
fn channel_scale(channels: ChannelMode) -> f32 {
    match channels {
        ChannelMode::Luma => 3.0,
        ChannelMode::Chroma => 1.5,
        ChannelMode::Yuv => 1.0,
    }
}

/// Drives one frame plane through both stages of the collaborative
/// filter.
///
/// BM3D-style collaborative filtering works in two passes. The first
/// groups similar patches straight out of the noisy input, shrinks their
/// transform coefficients with a hard threshold, and aggregates the
/// result into a pilot, a rough but already meaningfully cleaner
/// estimate of the frame. The second pass then regroups, this time
/// searching for similar patches in the pilot rather than the noisy
/// input.
///
/// Grouping on the pilot instead of the noisy data matters for two
/// reasons. The pilot carries far less noise, so patch similarity there
/// reflects real content rather than noise the two patches happen to
/// share, giving a better-matched group than the first pass could form.
/// And the pilot's coefficients then steer a Wiener filter applied to
/// the original noisy data. A coefficient the pilot found strongly
/// present passes through close to unchanged, and one the pilot found
/// weak or absent shrinks toward zero, softly and in proportion to its
/// expected noise, in contrast to the first pass's hard threshold. That
/// second, steered pass is what recovers detail a hard threshold alone
/// would have discarded.
///
/// `CollabPipeline` owns every GPU buffer both stages need and reuses
/// them across calls, so denoising a stream of frames at the same
/// dimensions and channel mode allocates nothing new per frame.
pub struct CollabPipeline<R: Runtime> {
    client: ComputeClient<R>,
    params: CollabParams,
    width: u32,
    height: u32,
    member_pos: Handle,
    member_count: Handle,
    member_sig2_dummy: Handle,
    filtered_dummy: Handle,
    group_weight: Handle,
    pilot: Handle,
    sigma_buf: Handle,
    dct_profile_buf: Handle,
    /// Fixed-point accumulators the filters scatter into, one weighted
    /// value per covering patch, cleared before each pass and divided
    /// out by `collab_normalise` after it.
    accum: Handle,
    wsum: Handle,
    /// The correlation profile the shrinkage uses, kept on the host too
    /// so the weight normalisation can be derived from it per frame.
    dct_profile: [f32; 8],
}

impl<R: Runtime> CollabPipeline<R> {
    /// Allocates every buffer the two-stage filter needs for frames of
    /// `width` by `height` pixels under `params`.
    ///
    /// Fails when `params` itself is invalid, or when the frame is
    /// smaller than one collaborative patch on either axis.
    pub fn new(
        client: &ComputeClient<R>,
        params: CollabParams,
        width: u32,
        height: u32,
    ) -> Result<Self, anyhow::Error> {
        params.validate()?;
        if width < PATCH_SIZE || height < PATCH_SIZE {
            anyhow::bail!(
                "frame dimensions {width}x{height} must be at least {}x{} for the \
                 collaborative filter's patch grid",
                PATCH_SIZE,
                PATCH_SIZE,
            );
        }

        let stored_ch = params.channels.storage_count() as usize;
        let k_max = params.k_max;
        let refs = ref_count(width, height);
        let pos_len = member_buf_len(width, height, k_max);
        let pixels = (width * height) as usize;
        let frame_len = pixels * stored_ch;

        let member_pos = client.empty(pos_len * size_of::<u32>());
        let member_count = client.empty(refs * size_of::<u32>());
        let member_sig2_dummy = client.empty(size_of::<f32>());
        // The filters only write their filtered patches out when
        // `emit_filtered` is set, which only tests do, so this pipeline
        // binds a one-element placeholder rather than carrying a buffer
        // large enough to hold every group.
        let filtered_dummy = client.empty(stored_ch * size_of::<f32>());
        let group_weight = client.empty(refs * size_of::<f32>());
        let pilot = client.empty(frame_len * size_of::<f32>());
        let accum = client.empty(frame_len * size_of::<i32>());
        let wsum = client.empty(pixels * size_of::<i32>());
        let sigma_buf = client.create_from_slice(f32::as_bytes(&vec![0.0f32; stored_ch]));
        // The profile is purely spatial, derived only from `params.rho`,
        // so it never changes across frames and is built once here
        // rather than every `run_two_stage` call the way `sigma_buf`
        // (which depends on the per-frame `sigmas` argument) is.
        let dct_profile = dct_noise_profile(params.rho);
        let dct_profile_buf = client.create_from_slice(f32::as_bytes(&dct_profile));

        Ok(Self {
            client: client.clone(),
            params,
            width,
            height,
            member_pos,
            member_count,
            member_sig2_dummy,
            filtered_dummy,
            group_weight,
            pilot,
            sigma_buf,
            dct_profile_buf,
            accum,
            wsum,
            dct_profile,
        })
    }

    /// Runs both stages of the collaborative filter over one frame
    /// plane.
    ///
    /// `input` and `output` are whole-plane buffers in the vectorized
    /// `Vector<f32, N>` layout this pipeline was built for, `width *
    /// height * storage_count()` scalars each. `sigmas` holds one noise
    /// standard deviation per active channel, in normalised `[0, 1]`
    /// units, `channels.count()` values long.
    ///
    /// Every kernel launch is queued in order with no readback in
    /// between, so the whole two-stage filter runs as one uninterrupted
    /// sequence of device work.
    pub fn run_two_stage(
        &mut self,
        input: &Handle,
        sigmas: &[f32],
        output: &Handle,
    ) -> Result<(), anyhow::Error> {
        let channels_count = self.params.channels.count() as usize;
        if sigmas.len() != channels_count {
            anyhow::bail!(
                "sigmas.len()={} must equal the active channel count {channels_count} for {:?}",
                sigmas.len(),
                self.params.channels,
            );
        }

        let stored_ch = self.params.channels.storage_count() as usize;
        let mut sigma_host = vec![0.0f32; stored_ch];
        sigma_host[..channels_count].copy_from_slice(sigmas);
        self.sigma_buf = self.client.create_from_slice(f32::as_bytes(&sigma_host));

        // The distance two noisy copies of the same patch are expected
        // to show by chance, mirroring `NlmParams::noise_offset_with`'s
        // own formula with `PATCH_AREA` (64 taps per 8x8 patch) in place
        // of `nlmeans`'s window tap count. `floor_epsilon` is one 8-bit
        // code level of admission slack per pixel, so grouping still
        // finds real matches at sigma 0, where `floor` itself collapses
        // to 0 and would otherwise leave `tau` at 0 too, admitting
        // nothing but a reference's own self match.
        //
        // `tau` only ever gates admission in `collab_group_spatial`, a
        // candidate patch joins a group when its distance to the
        // reference is at most `tau`. A reference's own self match, at
        // distance 0, is always seeded into the group before that gate
        // runs, so every group holds at least one member regardless of
        // `tau`. A `sigmas` entry that is not finite makes `sum_sq`, and
        // therefore `floor`, not finite too. `f32::max(floor,
        // floor_epsilon)` discards a non-finite `floor` and returns
        // `floor_epsilon`, the same small, fixed, always-positive
        // constant sigma 0 already falls back to. `tau` then admits
        // close to nothing beyond the guaranteed self match, the most
        // restrictive grouping this pipeline ever runs, not an
        // unbounded or negative one. `tau` never multiplies into a
        // pixel or coefficient value anywhere downstream, so a
        // restrictive `tau` cannot amplify anything, only shrink every
        // group toward its self-only floor.
        let scale = channel_scale(self.params.channels);
        let sum_sq: f32 = sigmas.iter().map(|&s| s * s).sum();
        let floor = 2.0 * scale * sum_sq * PATCH_AREA as f32;
        let floor_epsilon = PATCH_AREA as f32 * scale * (1.0f32 / 255.0f32).powi(2);
        let tau = self.params.tau_match * f32::max(floor, floor_epsilon);

        let (width, height) = (self.width, self.height);
        let k_max = self.params.k_max;
        let refs_x = refs_along(width);
        let refs_y = refs_along(height);
        let refs = ref_count(width, height);
        let pos_len = member_buf_len(width, height, k_max);
        let pixels = (width * height) as usize;
        let frame_len = pixels * stored_ch;

        // Every weight is divided by this before it reaches the
        // fixed-point accumulators. The first channel's sigma is the one
        // that matters, because that is the channel whose retained
        // variances the group weight is built from.
        let wnorm = weight_scale(sigmas[0], &self.dct_profile);

        let group_grid = CubeCount::new_2d(refs_x, refs_y);
        let group_dim = CubeDim::new_2d(8, 8);
        let agg_grid = CubeCount::new_2d(width.div_ceil(BLOCK_X), height.div_ceil(BLOCK_Y));
        let agg_dim = CubeDim::new_2d(BLOCK_X, BLOCK_Y);
        let zero_dim = 256u32;
        let zero_grid = CubeCount::new_1d((frame_len as u32).div_ceil(zero_dim));

        unsafe {
            // Stage one: group on the noisy input with the noise floor
            // subtracted, hard-threshold the group's coefficients, and
            // aggregate the result into the pilot.
            collab_group_spatial::launch_unchecked::<R>(
                &self.client,
                group_grid.clone(),
                group_dim,
                stored_ch,
                ArrayArg::from_raw_parts(input.clone(), frame_len),
                ArrayArg::from_raw_parts(self.member_pos.clone(), pos_len),
                ArrayArg::from_raw_parts(self.member_count.clone(), refs),
                0u32,
                floor,
                tau,
                width,
                height,
                channels_count as u32,
                k_max,
                self.params.spatial_radius,
                refs_x,
            );

            collab_zero_accum::launch_unchecked::<R>(
                &self.client,
                zero_grid.clone(),
                CubeDim::new_1d(zero_dim),
                ArrayArg::from_raw_parts(self.accum.clone(), frame_len),
                ArrayArg::from_raw_parts(self.wsum.clone(), pixels),
                pixels as u32,
                stored_ch as u32,
            );

            collab_filter_ht::launch_unchecked::<R>(
                &self.client,
                group_grid.clone(),
                group_dim,
                stored_ch,
                ArrayArg::from_raw_parts(input.clone(), frame_len),
                ArrayArg::from_raw_parts(self.member_pos.clone(), pos_len),
                ArrayArg::from_raw_parts(self.member_count.clone(), refs),
                ArrayArg::from_raw_parts(self.member_sig2_dummy.clone(), 1),
                ArrayArg::from_raw_parts(self.accum.clone(), frame_len),
                ArrayArg::from_raw_parts(self.wsum.clone(), pixels),
                ArrayArg::from_raw_parts(self.filtered_dummy.clone(), 1),
                ArrayArg::from_raw_parts(self.group_weight.clone(), refs),
                0u32,
                ArrayArg::from_raw_parts(self.sigma_buf.clone(), stored_ch),
                ArrayArg::from_raw_parts(self.dct_profile_buf.clone(), 8),
                self.params.lambda_ht,
                wnorm,
                false,
                false,
                width,
                height,
                channels_count as u32,
                k_max,
                stored_ch as u32,
                refs_x,
            );

            collab_normalise::launch_unchecked::<R>(
                &self.client,
                agg_grid.clone(),
                agg_dim,
                stored_ch,
                ArrayArg::from_raw_parts(self.accum.clone(), frame_len),
                ArrayArg::from_raw_parts(self.wsum.clone(), pixels),
                ArrayArg::from_raw_parts(self.pilot.clone(), frame_len),
                width,
                height,
                channels_count as u32,
                stored_ch as u32,
            );

            // Stage two: regroup on the pilot with no floor, since the
            // pilot is already clean and subtracting a floor there would
            // admit patches that do not really match, Wiener-filter the
            // noisy data steered by that pilot, and aggregate into the
            // caller's output.
            collab_group_spatial::launch_unchecked::<R>(
                &self.client,
                group_grid.clone(),
                group_dim,
                stored_ch,
                ArrayArg::from_raw_parts(self.pilot.clone(), frame_len),
                ArrayArg::from_raw_parts(self.member_pos.clone(), pos_len),
                ArrayArg::from_raw_parts(self.member_count.clone(), refs),
                0u32,
                0.0f32,
                tau,
                width,
                height,
                channels_count as u32,
                k_max,
                self.params.spatial_radius,
                refs_x,
            );

            collab_zero_accum::launch_unchecked::<R>(
                &self.client,
                zero_grid,
                CubeDim::new_1d(zero_dim),
                ArrayArg::from_raw_parts(self.accum.clone(), frame_len),
                ArrayArg::from_raw_parts(self.wsum.clone(), pixels),
                pixels as u32,
                stored_ch as u32,
            );

            collab_filter_wiener::launch_unchecked::<R>(
                &self.client,
                group_grid,
                group_dim,
                stored_ch,
                ArrayArg::from_raw_parts(input.clone(), frame_len),
                ArrayArg::from_raw_parts(self.pilot.clone(), frame_len),
                ArrayArg::from_raw_parts(self.member_pos.clone(), pos_len),
                ArrayArg::from_raw_parts(self.member_count.clone(), refs),
                ArrayArg::from_raw_parts(self.member_sig2_dummy.clone(), 1),
                ArrayArg::from_raw_parts(self.accum.clone(), frame_len),
                ArrayArg::from_raw_parts(self.wsum.clone(), pixels),
                ArrayArg::from_raw_parts(self.filtered_dummy.clone(), 1),
                ArrayArg::from_raw_parts(self.group_weight.clone(), refs),
                0u32,
                0u32,
                ArrayArg::from_raw_parts(self.sigma_buf.clone(), stored_ch),
                ArrayArg::from_raw_parts(self.dct_profile_buf.clone(), 8),
                wnorm,
                false,
                false,
                width,
                height,
                channels_count as u32,
                k_max,
                stored_ch as u32,
                refs_x,
            );

            collab_normalise::launch_unchecked::<R>(
                &self.client,
                agg_grid,
                agg_dim,
                stored_ch,
                ArrayArg::from_raw_parts(self.accum.clone(), frame_len),
                ArrayArg::from_raw_parts(self.wsum.clone(), pixels),
                ArrayArg::from_raw_parts(output.clone(), frame_len),
                width,
                height,
                channels_count as u32,
                stored_ch as u32,
            );
        }

        Ok(())
    }
}

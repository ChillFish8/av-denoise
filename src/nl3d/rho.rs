/// Not called by `Nl3dDenoiser` any more. Its constructor used to force
/// `CollabParams.rho` to `rho_window(search_radius)` on every build.
/// Measured with the high-frequency energy ratio against a clean
/// reference, that shaping under-shrank high-frequency coefficients
/// enough to leave visibly more noise behind on real footage, and it
/// also scored worse on XPSNR and SSIM than leaving `rho` at `0.0`, at
/// every noise level and every `residual_sigma_scale` tried. See
/// `Nl3dDenoiser::new`'s doc comment for the numbers. This table and
/// [`rho_window`] are kept, still measured and still tested, in case a
/// future change to the shrinkage kernels makes shaping earn its place
/// again, or a caller wants to opt into it directly through
/// `CollabParams.rho`.
///
/// The measured residual correlation nl3d's collaborative stage should
/// assume, indexed by the front end's `search_radius`.
///
/// Non-local means leaves a residual that is spatially correlated,
/// mainly because neighbouring output pixels draw on heavily
/// overlapping sets of source pixels through the search window, not
/// because the input grain itself was correlated. `search_radius` is
/// the dominant factor in how strong that correlation is. This table
/// converts it directly into the `rho` [`crate::collab::CollabParams`]
/// shapes its own noise variance by.
///
/// # What was swept
///
/// `search_radius` in `0..=4`, every value the shipped presets can
/// select, at `patch_radius = 4`, the library default
/// (`NlmParams::patch_radius`'s own doc comment). Both flat and
/// sine-textured synthetic content were swept, with uncorrelated input
/// noise and `temporal_radius = 0`. Each entry averages that radius's
/// horizontal and vertical correlation on both content types, which
/// stayed within 0.033 of each other at every radius measured.
///
/// The table was then checked against six consecutive frames of real
/// 1080p footage (a wide shot mixing large flat regions with fine,
/// high-frequency crowd detail), injected with two independent noise
/// realisations and denoised with the shipped front end at its measured
/// true sigma. Flat and smoothly-varying real content matched the
/// synthetic table closely, within 0.001 to 0.037 at the two radii
/// checked (2 and 4).
///
/// # Which entries are measured, and which are extrapolated
///
/// Every entry at `search_radius` 0 through 4 is a directly measured
/// value, not interpolated between neighbouring points. The curve rises
/// sharply near the low end and flattens out by radius 4, so entries at
/// `search_radius` 5 through [`crate::nlmeans::MAX_SEARCH_RADIUS`] (8),
/// which no measurement ever swept, hold flat at the radius 4 value
/// rather than extrapolating a rise that the measured curve does not
/// show continuing. This is a deliberate clamp, not a gap in the data.
///
/// # The flat-content caveat, and why it is not "fixed" here
///
/// This table is accurate for flat and smoothly-varying content. On
/// fine real texture, such as brick masonry or packed crowd detail, the
/// true residual correlation measures substantially lower, around 0.63
/// to 0.67 where this table says 0.80 to 0.86, because a wide search
/// window finds proportionally fewer well-matching candidates on fast-
/// varying content rather than more, so it induces less averaging than
/// it does on flat content.
///
/// Baking in the flat-content value anyway, rather than a lower one
/// tuned for texture, is deliberate. Overstating the correlation makes
/// the shaped profile assign *less* variance to high-frequency
/// coefficients than they truly carry, which means those coefficients
/// get shrunk *less* than a correctly-calibrated texture-aware profile
/// would shrink them. That preserves fine texture and leaves a little
/// extra residual noise behind, which is the direction this filter
/// should err in. The opposite mistake, a table tuned low for texture
/// and applied to flat content, would over-shrink flat regions instead
/// and remove less noise than it safely could, a worse trade for a
/// denoiser. A future reader who finds this table "too high" on some
/// content and lowers it is re-introducing the destroyed-texture
/// failure mode this whole profile exists to fix.
#[allow(dead_code, reason = "kept for a future caller; see the module doc comment")]
const RHO_WINDOW_TABLE: [f32; 5] = [0.00, 0.67, 0.80, 0.85, 0.86];

/// Looks up [`RHO_WINDOW_TABLE`] for `search_radius`, clamped to the
/// table's last entry above `search_radius = 4`.
#[allow(dead_code, reason = "kept for a future caller; see the module doc comment")]
pub(crate) fn rho_window(search_radius: u32) -> f32 {
    let idx = (search_radius as usize).min(RHO_WINDOW_TABLE.len() - 1);
    RHO_WINDOW_TABLE[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rho_window_matches_the_measured_table_exactly() {
        let expected = [0.00f32, 0.67, 0.80, 0.85, 0.86];
        for (radius, &want) in expected.iter().enumerate() {
            let got = rho_window(radius as u32);
            assert_eq!(got, want, "search_radius={radius}: expected {want}, got {got}");
        }
    }

    #[test]
    fn rho_window_clamps_above_the_measured_range() {
        let at_four = rho_window(4);
        for radius in 5..=crate::nlmeans::MAX_SEARCH_RADIUS {
            assert_eq!(
                rho_window(radius),
                at_four,
                "search_radius={radius} (beyond the measured range) must clamp to the \
                 radius=4 value rather than extrapolating"
            );
        }
    }

    #[test]
    fn rho_window_is_zero_at_search_radius_zero() {
        assert_eq!(
            rho_window(0),
            0.0,
            "search_radius=0 has no spatial window to induce correlation through"
        );
    }
}

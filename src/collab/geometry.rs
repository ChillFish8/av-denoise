/// Number of reference patches along one axis.
///
/// `dim` must be at least `PATCH_SIZE`, which denoiser construction
/// validates.
pub fn refs_along(dim: u32) -> u32 {
    (dim - super::PATCH_SIZE).div_ceil(super::STEP) + 1
}

/// Cubes along x for [`crate::collab::kernels::fused::collab_fused`].
///
/// That kernel gives each of its eight 8-lane groups one reference
/// patch, so a row of references needs an eighth as many cubes as
/// [`refs_along`] returns. The count rounds up, and the last cube of a
/// row runs dead groups for the references past the end.
pub fn fused_cubes_x(width: u32) -> u32 {
    refs_along(width).div_ceil(8)
}

/// Top-left pixel of reference index `i` along one axis. The last
/// reference clamps so its patch stays inside the frame.
pub fn ref_pos(i: u32, dim: u32) -> u32 {
    (i * super::STEP).min(dim - super::PATCH_SIZE)
}

/// Total reference count for a frame.
pub fn ref_count(width: u32, height: u32) -> usize {
    refs_along(width) as usize * refs_along(height) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refs_cover_1080p_exactly() {
        // (1920 - 8) / 4 + 1 = 479, (1080 - 8) / 4 + 1 = 269.
        assert_eq!(refs_along(1920), 479);
        assert_eq!(refs_along(1080), 269);
    }

    #[test]
    fn fused_cubes_cover_every_reference() {
        // 1920 gives 479 references, so the last of the 60 cubes runs
        // seven live groups and one dead one.
        assert_eq!(fused_cubes_x(1920), 60);
        for dim in [8u32, 9, 21, 64, 100, 104, 128, 1280, 1920, 3840] {
            let cubes = fused_cubes_x(dim);
            let refs = refs_along(dim);
            assert!(cubes * 8 >= refs, "dim={dim} leaves references uncovered");
            assert!((cubes - 1) * 8 < refs, "dim={dim} launches a wholly dead cube");
        }
    }

    #[test]
    fn last_ref_clamps_inside_the_frame() {
        let w = 21; // not a multiple of STEP past PATCH_SIZE
        let n = refs_along(w);
        assert_eq!(ref_pos(n - 1, w), w - 8);
        for i in 0..n {
            assert!(ref_pos(i, w) + 8 <= w);
        }
    }

    #[test]
    fn every_pixel_is_covered_by_one_to_three_refs_per_axis() {
        // Regular spacing gives 2 covering references per axis, and the
        // single clamped edge gap can add a third. It never reaches a
        // fourth, so the bound here is 1..=3, giving at most 9 (3 x 3)
        // covering references per pixel in 2D.
        for dim in [8u32, 9, 16, 21, 64] {
            for x in 0..dim {
                let n = refs_along(dim);
                let covering = (0..n)
                    .filter(|&i| {
                        let p = ref_pos(i, dim);
                        p <= x && x < p + 8
                    })
                    .count();
                assert!((1..=3).contains(&covering), "dim={dim} x={x} covering={covering}");
            }
        }
    }

    #[test]
    fn ref_count_is_the_product_of_the_per_axis_counts() {
        assert_eq!(
            ref_count(1920, 1080),
            refs_along(1920) as usize * refs_along(1080) as usize
        );
    }
}

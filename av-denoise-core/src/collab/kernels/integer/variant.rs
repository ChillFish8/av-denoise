use crate::collab::kernels::fused::{GROUP_LANES, Walk, tile_height, tile_width};

/// How the integer centre search scores its candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntCentre {
    /// Each group scores its own candidates from the code ring.
    Scalar,
    /// From a shared tile holding the union of the cube's search windows.
    Tiled,
    /// On the f16 matrix units.
    Matrix,
}

/// How the integer neighbour search scores its candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntNeighbour {
    /// Each group scores its own candidates from the code ring.
    Scalar,
    /// On the f16 matrix units.
    Matrix,
}

/// One compiled form of `collab_search_int`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IntegerVariant {
    /// References along a row per cube.
    pub cube_refs_x: u32,
    /// Reference rows per cube.
    pub ref_rows: u32,
    pub walk: Walk,
    pub centre: IntCentre,
    pub neighbour: IntNeighbour,
}

impl IntegerVariant {
    /// Threads in one cube, one 8-lane group per reference.
    pub fn lanes(self) -> u32 {
        GROUP_LANES * self.cube_refs_x * self.ref_rows
    }

    /// Whether either search runs on the matrix units.
    pub fn is_matrix(self) -> bool {
        self.centre == IntCentre::Matrix || self.neighbour == IntNeighbour::Matrix
    }

    /// Whether `collab_search_int` builds this variant.
    pub fn is_built(self) -> bool {
        let tiled_built = self.centre == IntCentre::Tiled && self.ref_rows == 1;
        let centre_built = self.centre == IntCentre::Scalar || tiled_built;
        centre_built && self.neighbour == IntNeighbour::Scalar
    }

    /// Codes in the centre tile, one per summed channel per pixel.
    ///
    /// Centres other than [IntCentre::Tiled](crate::collab::kernels::integer::IntCentre::Tiled)
    /// keep a one-element placeholder, so every variant shares one kernel body.
    pub fn tile_elements(self, spatial_radius: u32, channels: u32) -> u32 {
        if self.centre != IntCentre::Tiled {
            return 1;
        }

        let width = tile_width(self.cube_refs_x, spatial_radius);
        let height = tile_height(spatial_radius);
        width * height * channels
    }

    /// The scalar variant at the baseline's cube shape, with the uniform walk where `uniform_only` is
    /// set.
    pub fn reference_for(uniform_only: bool) -> IntegerVariant {
        let walk = if uniform_only {
            Walk::Uniform
        } else {
            Walk::Clipped
        };

        IntegerVariant {
            cube_refs_x: 8,
            ref_rows: 1,
            walk,
            centre: IntCentre::Scalar,
            neighbour: IntNeighbour::Scalar,
        }
    }

    /// Whether this variant can run on a runtime where `uniform_only` is set.
    pub fn runs_where(self, uniform_only: bool) -> bool {
        !uniform_only || self.walk == Walk::Uniform
    }

    pub fn name(self) -> &'static str {
        let position = INTEGER_VARIANTS.iter().position(|&variant| variant == self);
        let index = position.expect("every integer candidate is a listed variant");
        INTEGER_NAMES[index]
    }
}

pub const INTEGER_VARIANT_COUNT: usize = 7;

const fn integer_variant(
    cube_refs_x: u32,
    ref_rows: u32,
    walk: Walk,
    centre: IntCentre,
    neighbour: IntNeighbour,
) -> IntegerVariant {
    IntegerVariant {
        cube_refs_x,
        ref_rows,
        walk,
        centre,
        neighbour,
    }
}

/// Every integer variant, the clipped scalar reference first.
pub const INTEGER_VARIANTS: [IntegerVariant; INTEGER_VARIANT_COUNT] = [
    integer_variant(8, 1, Walk::Clipped, IntCentre::Scalar, IntNeighbour::Scalar),
    integer_variant(8, 1, Walk::Uniform, IntCentre::Scalar, IntNeighbour::Scalar),
    integer_variant(8, 1, Walk::Clipped, IntCentre::Tiled, IntNeighbour::Scalar),
    integer_variant(16, 1, Walk::Uniform, IntCentre::Matrix, IntNeighbour::Scalar),
    integer_variant(8, 2, Walk::Uniform, IntCentre::Matrix, IntNeighbour::Scalar),
    integer_variant(16, 1, Walk::Uniform, IntCentre::Matrix, IntNeighbour::Matrix),
    integer_variant(8, 2, Walk::Uniform, IntCentre::Matrix, IntNeighbour::Matrix),
];

/// Every integer variant's stored name, in [INTEGER_VARIANTS] order.
const INTEGER_NAMES: [&str; INTEGER_VARIANT_COUNT] = [
    "int_groups8_clipped",
    "int_groups8_uniform",
    "int_groups8_clipped_tiled",
    "int_refs16_matrix",
    "int_refs8x2_matrix",
    "int_refs16_matrix_nmatrix",
    "int_refs8x2_matrix_nmatrix",
];

use crate::collab::kernels::fused::{GROUP_LANES, Walk, tile_height, tile_width};

/// How the integer centre search scores its candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntCentre {
    /// Each group scores its own candidates from the code ring.
    Scalar,
    /// From a shared tile holding the union of the cube's search windows.
    Tiled,
    /// Offset by offset over a block of references, from shared sums of 4x4 blocks.
    Sliding,
}

/// How the integer neighbour search scores its candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntNeighbour {
    /// Each group scores its own candidates from the code ring.
    Scalar,
}

/// One compiled form of the integer collab search.
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
    /// Threads in one centre cube.
    ///
    /// Most centres run one 8-lane group per reference. A sliding centre runs one lane per
    /// reference, rounded up to whole 64-lane planes.
    pub const fn lanes(self) -> u32 {
        let refs = self.cube_refs_x * self.ref_rows;

        if self.is_sliding() {
            return refs.next_multiple_of(SLIDING_LANE_STEP);
        }

        GROUP_LANES * refs
    }

    /// Whether the centre search is the sliding one.
    pub const fn is_sliding(self) -> bool {
        matches!(self.centre, IntCentre::Sliding)
    }

    /// The 4x4 blocks a sliding centre cube scores per offset, one more than its references along
    /// each side.
    pub const fn sliding_blocks(self) -> u32 {
        (self.cube_refs_x + 1) * (self.ref_rows + 1)
    }

    /// Whether a kernel is built for this variant.
    pub fn is_built(self) -> bool {
        let tiled_built = self.centre == IntCentre::Tiled && self.ref_rows == 1;
        let sliding_built = self.is_sliding() && self.walk == Walk::Clipped;
        let centre_built = self.centre == IntCentre::Scalar || tiled_built || sliding_built;
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

/// The plane width a sliding centre cube's lanes round up to.
const SLIDING_LANE_STEP: u32 = 64;

pub const INTEGER_VARIANT_COUNT: usize = 5;

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
    integer_variant(7, 7, Walk::Clipped, IntCentre::Sliding, IntNeighbour::Scalar),
    integer_variant(15, 7, Walk::Clipped, IntCentre::Sliding, IntNeighbour::Scalar),
];

const fn sliding_blocks_fit_their_lanes() -> bool {
    let mut index = 0;

    while index < INTEGER_VARIANT_COUNT {
        let variant = INTEGER_VARIANTS[index];

        if variant.is_sliding() && variant.sliding_blocks() > variant.lanes() {
            return false;
        }

        index += 1;
    }

    true
}

// Each lane of a sliding centre scores one block per offset.
const _: () = assert!(
    sliding_blocks_fit_their_lanes(),
    "every sliding centre needs a lane for each of its blocks"
);

/// Every integer variant's stored name, in [INTEGER_VARIANTS] order.
const INTEGER_NAMES: [&str; INTEGER_VARIANT_COUNT] = [
    "int_groups8_clipped",
    "int_groups8_uniform",
    "int_groups8_clipped_tiled",
    "int_sliding7x7",
    "int_sliding15x7",
];

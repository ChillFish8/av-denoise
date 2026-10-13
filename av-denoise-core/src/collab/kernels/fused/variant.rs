use cubecl::prelude::*;

use crate::collab::geometry::BASELINE_GROUPS;
use crate::collab::{PATCH_SIZE, STEP, needs_warp_uniform_search};

/// Lanes in one group, one per patch column.
pub const GROUP_LANES: u32 = 8;

/// Floats in each group's transpose region, one past 64 so lanes never collide on a bank.
pub const TRANSPOSE_REGION_FLOATS: u32 = 65;

/// How the searches walk their candidate rectangles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Walk {
    /// Each group walks its own rectangle, clipped to the frame, and leaves when it is done.
    Clipped,
    /// Every group walks the full comptime span of its rectangle and masks the positions past the
    /// clip, so every group in a plane takes the same turns. A plane-wide vote skips the positions no
    /// group needs.
    Uniform,
}

impl Walk {
    /// The walk's part of a variant's stored name.
    pub fn name(self) -> &'static str {
        match self {
            Walk::Clipped => "clipped",
            Walk::Uniform => "uniform",
        }
    }
}

/// Where the centre search reads its candidates from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Centre {
    /// Straight from the frame ring in global memory.
    Global,
    /// From a shared tile holding the union of the cube's search windows, loaded once per cube.
    Tiled,
}

impl Centre {
    /// The centre's part of a variant's stored name, empty for the global baseline.
    pub fn name_suffix(self) -> &'static str {
        match self {
            Centre::Global => "",
            Centre::Tiled => "_tiled",
        }
    }
}

/// Width in pixels of the centre tile for `groups` references `STEP` apart.
pub fn tile_width(groups: u32, spatial_radius: u32) -> u32 {
    (groups - 1) * STEP + 2 * spatial_radius + PATCH_SIZE
}

/// Height in pixels of the centre tile.
pub fn tile_height(spatial_radius: u32) -> u32 {
    2 * spatial_radius + PATCH_SIZE
}

/// One compiled form of `collab_fused`.
///
/// Every variant scores the same candidates in the same order and scatters the same values, and the
/// centre tile is read in the same order as the ring, so all of them give bit identical output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CollabVariant {
    /// Reference patches per cube, one 8-lane group each.
    pub groups: u32,
    pub walk: Walk,
    /// Where the centre search reads its candidates from.
    pub centre: Centre,
    /// Whether the centre search scores 8 candidates per round of shuffles.
    pub batched: bool,
}

impl CollabVariant {
    /// The untuned variant on a runtime that runs the clipped walk.
    pub const BASELINE: CollabVariant = CollabVariant {
        groups: BASELINE_GROUPS,
        walk: Walk::Clipped,
        centre: Centre::Global,
        batched: false,
    };

    /// The untuned variant, with the uniform walk where `uniform_only` is set.
    pub fn baseline_for(uniform_only: bool) -> CollabVariant {
        if uniform_only {
            return CollabVariant {
                walk: Walk::Uniform,
                ..CollabVariant::BASELINE
            };
        }

        CollabVariant::BASELINE
    }

    /// The untuned variant on `client`'s runtime.
    pub fn for_client<R: Runtime>(client: &ComputeClient<R>) -> CollabVariant {
        let uniform_only = needs_warp_uniform_search(client);
        CollabVariant::baseline_for(uniform_only)
    }

    /// Whether this variant can run on a runtime where `uniform_only` is set.
    pub fn runs_where(self, uniform_only: bool) -> bool {
        !uniform_only || self.walk == Walk::Uniform
    }

    /// Threads in one cube.
    pub fn lanes(self) -> u32 {
        GROUP_LANES * self.groups
    }

    /// Elements of the centre tile, one per stored lane per pixel.
    ///
    /// The global centre search keeps a one-element placeholder, so every variant shares one kernel
    /// body and one set of helper signatures.
    pub fn tile_elements(self, spatial_radius: u32, stored_ch: u32) -> u32 {
        if self.centre == Centre::Global {
            return 1;
        }

        let width = tile_width(self.groups, spatial_radius);
        let height = tile_height(spatial_radius);
        width * height * stored_ch
    }

    /// Bytes of every shared-memory buffer the kernel allocates, with the tile holding values of
    /// `search_bytes` bytes.
    pub fn shared_bytes(self, spatial_radius: u32, stored_ch: u32, search_bytes: usize) -> usize {
        let transpose_floats = self.groups * TRANSPOSE_REGION_FLOATS;
        let transpose = transpose_floats as usize * size_of::<f32>();
        let tile_elements = self.tile_elements(spatial_radius, stored_ch);
        let tile = tile_elements as usize * search_bytes;

        transpose + tile
    }
}

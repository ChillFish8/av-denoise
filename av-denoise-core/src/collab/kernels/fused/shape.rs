use cubecl::prelude::*;

/// The comptime frame and patch shape the collab searches read.
///
/// `f16_search` scores candidates from the f16 search ring. `spatial_radius` is the centre
/// search's reach around the reference. `stored_ch` is the lanes each pixel stores, which can be
/// more than the `channels` the distance sums.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SearchShape {
    pub(crate) f16_search: bool,
    pub(crate) spatial_radius: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) channels: u32,
    pub(crate) stored_ch: u32,
}

/// One lane's position in its group's centre search.
///
/// `(rx, ry)` is the reference's top-left and `(tile_x0, tile_y0)` the cube's centre tile corner.
/// `sub` is the lane's column in its group and `base` the group's first lane in the plane. `scale`
/// multiplies every completed distance.
#[derive(CubeType, Clone, Copy)]
pub(crate) struct SearchCursor {
    pub(crate) rx: u32,
    pub(crate) ry: u32,
    pub(crate) tile_x0: u32,
    pub(crate) tile_y0: u32,
    pub(crate) centre_slot: u32,
    pub(crate) sub: u32,
    pub(crate) base: u32,
    pub(crate) scale: f32,
}

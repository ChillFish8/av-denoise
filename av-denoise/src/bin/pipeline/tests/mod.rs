mod convert;
mod coordinator;
mod scenes;
mod stage;
mod worker;

use av_denoise::frame::fill_plane;
use av_denoise::{Depth, FrameLayout, Planes, Subsampling};

pub fn tiny_layout() -> FrameLayout {
    // 4:2:0 chroma at this size is 4x4, clearing the denoiser's 3x3
    // minimum frame dimension.
    FrameLayout {
        width: 8,
        height: 8,
        subsampling: Subsampling::Yuv420,
        depth: Depth::Eight,
    }
}

pub fn tiny_planes(layout: FrameLayout) -> Planes {
    Planes {
        y: fill_plane(layout.luma_pixels(), layout.depth.neutral_chroma(), layout.depth),
        u: layout.neutral_chroma_plane(),
        v: layout.neutral_chroma_plane(),
    }
}

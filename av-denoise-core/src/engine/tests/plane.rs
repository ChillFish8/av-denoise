use crate::engine::{Geometry, SampleFormat};
use crate::error::Error;
use crate::nlmeans::ChannelMode;

#[test]
fn u16_depth_outside_9_to_16_is_rejected() {
    for depth in [0, 8, 17] {
        let format = SampleFormat::U16 { depth };
        let result = format.validate();
        assert!(matches!(result, Err(Error::InvalidGeometry(_))));
    }
}

#[test]
fn u16_depth_inside_9_to_16_is_accepted() {
    for depth in 9..=16 {
        let format = SampleFormat::U16 { depth };
        assert!(format.validate().is_ok());
    }
}

#[test]
fn max_value_matches_the_depth() {
    assert_eq!(SampleFormat::U8.max_value(), 255.0);
    assert_eq!(SampleFormat::U16 { depth: 10 }.max_value(), 1023.0);
    assert_eq!(SampleFormat::U16 { depth: 16 }.max_value(), 65535.0);
    assert_eq!(SampleFormat::F32.max_value(), 1.0);
}

#[test]
fn plane_bytes_round_up_to_whole_words() {
    assert_eq!(SampleFormat::U8.plane_bytes(9), 12);
    assert_eq!(SampleFormat::U16 { depth: 10 }.plane_bytes(3), 8);
    assert_eq!(SampleFormat::F32.plane_bytes(3), 12);
}

fn sized_geometry(width: u32, height: u32, channels: ChannelMode) -> Geometry {
    Geometry {
        width,
        height,
        channels,
        input: SampleFormat::F32,
        output: SampleFormat::F32,
    }
}

#[test]
fn pixels_do_not_overflow_at_the_largest_dimensions() {
    let geometry = sized_geometry(u32::MAX, u32::MAX, ChannelMode::Luma);
    let expected = u64::from(u32::MAX) * u64::from(u32::MAX);
    assert_eq!(geometry.pixels(), expected);
}

#[test]
fn a_ring_of_exactly_u32_max_elements_fits() {
    let geometry = sized_geometry(65_535, 65_537, ChannelMode::Luma);
    let result = geometry.check_ring_fits(1);
    assert!(result.is_ok());
}

#[test]
fn a_ring_one_frame_past_u32_max_elements_is_invalid_geometry() {
    let geometry = sized_geometry(65_535, 65_537, ChannelMode::Luma);
    let result = geometry.check_ring_fits(2);
    assert!(matches!(result, Err(Error::InvalidGeometry(_))));
}

#[test]
fn a_ring_whose_size_overflows_u64_is_invalid_geometry() {
    let geometry = sized_geometry(u32::MAX, u32::MAX, ChannelMode::Yuv);
    let result = geometry.check_ring_fits(u64::MAX);
    assert!(matches!(result, Err(Error::InvalidGeometry(_))));
}

#[cfg(any(feature = "vulkan", feature = "metal"))]
mod with_handles {
    use cubecl::prelude::*;
    use cubecl::wgpu::WgpuRuntime;

    use super::*;
    use crate::engine::DevicePlane;

    fn client() -> ComputeClient<WgpuRuntime> {
        let device = <WgpuRuntime as Runtime>::Device::default();
        WgpuRuntime::client(&device)
    }

    fn geometry(channels: ChannelMode) -> Geometry {
        Geometry {
            width: 3,
            height: 3,
            channels,
            input: SampleFormat::U8,
            output: SampleFormat::U8,
        }
    }

    #[test]
    fn rejects_the_wrong_plane_count() {
        let client = client();
        let handle = client.empty(12);
        let plane = DevicePlane::new(&handle, 3, 3);
        let geometry = geometry(ChannelMode::Chroma);
        let result = geometry.check_planes(&[plane], SampleFormat::U8);
        assert!(matches!(result, Err(Error::PlaneMismatch(_))));
    }

    #[test]
    fn rejects_mismatched_plane_dimensions() {
        let client = client();
        let handle = client.empty(16);
        let plane = DevicePlane::new(&handle, 4, 3);
        let geometry = geometry(ChannelMode::Luma);
        let result = geometry.check_planes(&[plane], SampleFormat::U8);
        assert!(matches!(result, Err(Error::PlaneMismatch(_))));
    }

    #[test]
    fn rejects_plane_handle_shorter_than_its_words() {
        let client = client();
        let handle = client.empty(9);
        let plane = DevicePlane::new(&handle, 3, 3);
        let geometry = geometry(ChannelMode::Luma);
        let result = geometry.check_planes(&[plane], SampleFormat::U8);
        assert!(matches!(result, Err(Error::PlaneMismatch(_))));
    }

    #[test]
    fn accepts_a_word_padded_plane() {
        let client = client();
        let handle = client.empty(12);
        let plane = DevicePlane::new(&handle, 3, 3);
        let geometry = geometry(ChannelMode::Luma);
        let result = geometry.check_planes(&[plane], SampleFormat::U8);
        assert!(result.is_ok());
    }
}

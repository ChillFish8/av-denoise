use crate::engine::{Geometry, SampleFormat};
use crate::error::Error;

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

#[cfg(any(feature = "vulkan", feature = "metal"))]
mod with_handles {
    use cubecl::prelude::*;
    use cubecl::wgpu::WgpuRuntime;

    use super::*;
    use crate::engine::DevicePlane;
    use crate::nlmeans::ChannelMode;

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

use crate::engine::{ChromaLayout, Geometry, SampleFormat};

#[test]
fn chroma_dims_round_up_for_each_layout() {
    assert_eq!(ChromaLayout::Yuv420.chroma_dims(161, 127), Some((81, 64)));
    assert_eq!(ChromaLayout::Yuv422.chroma_dims(161, 127), Some((81, 127)));
    assert_eq!(ChromaLayout::Yuv444.chroma_dims(161, 127), Some((161, 127)));
    assert_eq!(ChromaLayout::Gray.chroma_dims(161, 127), None);
}

#[test]
fn plane_count_is_one_for_gray_and_three_for_yuv() {
    assert_eq!(ChromaLayout::Gray.plane_count(), 1);
    assert_eq!(ChromaLayout::Yuv420.plane_count(), 3);
    assert_eq!(ChromaLayout::Yuv422.plane_count(), 3);
    assert_eq!(ChromaLayout::Yuv444.plane_count(), 3);
}

fn geometry(chroma: ChromaLayout) -> Geometry {
    Geometry {
        width: 13,
        height: 11,
        chroma,
        input: SampleFormat::U8,
        output: SampleFormat::U8,
    }
}

#[test]
fn a_u16_depth_outside_9_to_16_is_rejected() {
    let mut geometry = geometry(ChromaLayout::Yuv420);
    geometry.output = SampleFormat::U16 { depth: 8 };
    let result = geometry.validate();
    assert!(matches!(result, Err(crate::error::Error::InvalidGeometry(_))));
}

#[cfg(any(feature = "vulkan", feature = "metal"))]
mod with_handles {
    use cubecl::prelude::*;
    use cubecl::server::Handle;

    use super::geometry;
    use crate::engine::{ChromaLayout, DevicePlane, SampleFormat};
    use crate::error::Error;

    fn client() -> Client {
        crate::tests::compilation_cache::install_test_compilation_cache();
        crate::testing::test_client()
    }

    fn handles(client: &Client, dims: &[(u32, u32)]) -> Vec<Handle> {
        dims.iter()
            .map(|&(width, height)| {
                let pixels = u64::from(width * height);
                let bytes = SampleFormat::U8.plane_bytes(pixels);
                client.empty(bytes as usize)
            })
            .collect()
    }

    fn planes<'a>(handles: &'a [Handle], dims: &[(u32, u32)]) -> Vec<DevicePlane<'a>> {
        handles
            .iter()
            .zip(dims)
            .map(|(handle, &(width, height))| DevicePlane::new(handle, width, height))
            .collect()
    }

    /// Checks `layout` accepts planes at `dims` and rejects each plane one sample too narrow.
    fn assert_plane_sizes(layout: ChromaLayout, dims: &[(u32, u32)]) {
        let client = client();
        let geometry = geometry(layout);

        let handles = handles(&client, dims);
        let accepted = planes(&handles, dims);
        let result = geometry.check_planes(&accepted, SampleFormat::U8);
        assert!(result.is_ok(), "{layout:?} rejected {dims:?}: {result:?}");

        for index in 0..dims.len() {
            let mut wrong = dims.to_vec();
            wrong[index].0 -= 1;

            let wrong_planes = planes(&handles, &wrong);
            let result = geometry.check_planes(&wrong_planes, SampleFormat::U8);
            assert!(
                matches!(result, Err(Error::PlaneMismatch(_))),
                "{layout:?} accepted plane {index} at {:?}",
                wrong[index]
            );
        }
    }

    #[test]
    fn check_planes_wants_every_plane() {
        let client = client();

        let yuv_dims = [(13, 11), (7, 6)];
        let yuv_handles = handles(&client, &yuv_dims);
        let yuv_planes = planes(&yuv_handles, &yuv_dims);
        let yuv = geometry(ChromaLayout::Yuv420);
        let result = yuv.check_planes(&yuv_planes, SampleFormat::U8);
        assert!(matches!(result, Err(Error::PlaneMismatch(_))));

        let gray_dims = [(13, 11), (13, 11), (13, 11)];
        let gray_handles = handles(&client, &gray_dims);
        let gray_planes = planes(&gray_handles, &gray_dims);
        let gray = geometry(ChromaLayout::Gray);
        let result = gray.check_planes(&gray_planes, SampleFormat::U8);
        assert!(matches!(result, Err(Error::PlaneMismatch(_))));
    }

    #[test]
    fn check_planes_wants_each_plane_at_its_size_for_yuv420() {
        assert_plane_sizes(ChromaLayout::Yuv420, &[(13, 11), (7, 6), (7, 6)]);
    }

    #[test]
    fn check_planes_wants_each_plane_at_its_size_for_yuv422() {
        assert_plane_sizes(ChromaLayout::Yuv422, &[(13, 11), (7, 11), (7, 11)]);
    }

    #[test]
    fn check_planes_wants_each_plane_at_its_size_for_yuv444() {
        assert_plane_sizes(ChromaLayout::Yuv444, &[(13, 11), (13, 11), (13, 11)]);
    }

    #[test]
    fn check_planes_wants_each_plane_at_its_size_for_gray() {
        assert_plane_sizes(ChromaLayout::Gray, &[(13, 11)]);
    }

    #[test]
    fn check_planes_rejects_a_handle_shorter_than_its_words() {
        let client = client();
        let geometry = geometry(ChromaLayout::Yuv444);
        let full = handles(&client, &[(13, 11), (13, 11)]);
        let short = client.empty(4);

        let planes = [
            DevicePlane::new(&full[0], 13, 11),
            DevicePlane::new(&full[1], 13, 11),
            DevicePlane::new(&short, 13, 11),
        ];
        let result = geometry.check_planes(&planes, SampleFormat::U8);
        assert!(matches!(result, Err(Error::PlaneMismatch(_))));
    }
}

pub mod camera;

pub use crate::error::Error;
pub use camera::*;
pub use gausplat_importer::source::colmap::{self, ColmapSource};
pub use gausplat_renderer::{render::view::*, scene::point::*};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{fmt, io::Read};

#[derive(Clone, PartialEq)]
pub struct MultiViewDataset {
    pub cameras: Cameras,
    pub points: Points,
}

impl<S: Read + Send> TryFrom<colmap::ColmapSource<S>> for MultiViewDataset {
    type Error = Error;

    fn try_from(source: colmap::ColmapSource<S>) -> Result<Self, Self::Error> {
        let points = source
            .points
            .into_iter()
            .map(|point| Point {
                color_rgb: point.color_rgb_normalized(),
                position: point.position,
            })
            .collect();

        let images_encoded = Vec::from_iter(source.images_file.into_values())
            .into_par_iter()
            .map(|mut image_file| {
                let image_encoded = image_file.read()?;
                Ok((image_file.name, image_encoded))
            })
            .collect::<Result<dashmap::DashMap<_, _>, Self::Error>>()?;

        let cameras = Vec::from_iter(source.images.into_values())
            .into_par_iter()
            .map(|image| {
                let view_rotation = View::rotation(&image.quaternion);
                let view_position =
                    View::position(&view_rotation, &image.translation);
                let view_transform =
                    View::transform_to_view(&view_rotation, &image.translation);
                let camera_id = image.camera_id;
                let image_file_name = image.file_name;
                let id = image.image_id;

                let camera = source
                    .cameras
                    .get(&camera_id)
                    .ok_or(Error::UnknownCameraId(camera_id))?;
                let (field_of_view_x, field_of_view_y) = match camera {
                    colmap::Camera::Pinhole(camera) => (
                        2.0 * (camera.width as f64)
                            .atan2(2.0 * camera.focal_length_x),
                        2.0 * (camera.height as f64)
                            .atan2(2.0 * camera.focal_length_y),
                    ),
                };

                let image_encoded = images_encoded
                    .remove(&image_file_name)
                    .ok_or(Error::UnknownImageFileName(
                        image_file_name.to_owned(),
                    ))?
                    .1;
                let image = Image {
                    image_encoded,
                    image_file_name,
                    image_id: id,
                };
                let (image_width, image_height) = image.decode_dimensions()?;

                let view = View {
                    field_of_view_x,
                    field_of_view_y,
                    image_height,
                    image_width,
                    view_id: id,
                    view_position,
                    view_transform,
                };

                let camera = Camera {
                    camera_id: id,
                    image,
                    view,
                };

                Ok((id, camera))
            })
            .collect::<Result<_, Self::Error>>()?;

        #[cfg(debug_assertions)]
        log::debug!(
            target: "gausplat_importer::source",
            "MultiViewDataset::try_from(ColmapSource)",
        );

        Ok(Self { cameras, points })
    }
}

impl fmt::Debug for MultiViewDataset {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("MultiViewDataset")
            .field("cameras.len()", &self.cameras.len())
            .field("points.len()", &self.points.len())
            .finish()
    }
}

impl Default for MultiViewDataset {
    #[inline]
    fn default() -> Self {
        Self {
            cameras: Default::default(),
            points: vec![Default::default(); 16],
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn default() {
        use super::*;

        let dataset = MultiViewDataset::default();
        assert!(dataset.cameras.is_empty());
        assert!(!dataset.points.is_empty());
    }

    #[test]
    fn default_try_from_colmap() {
        use super::*;

        MultiViewDataset::try_from(ColmapSource::<&[u8]> {
            cameras: [Default::default()].into(),
            images: [Default::default()].into(),
            images_file: [Default::default()].into(),
            points: [Default::default()].into(),
        })
        .unwrap_err();

        MultiViewDataset::try_from(ColmapSource::<&[u8]>::default()).unwrap();
    }
}

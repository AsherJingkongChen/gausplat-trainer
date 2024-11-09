pub mod camera;

pub use crate::error::Error;
pub use camera::*;
pub use gausplat_loader::source::colmap::{self, ColmapSource};
pub use gausplat_renderer::scene::point::*;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{fmt, io::Read, ops::Mul};

#[derive(Clone, PartialEq)]
pub struct SparseViewDataset {
    pub cameras: Cameras,
    pub points: Points,
}

impl<S: Read + Send> TryFrom<ColmapSource<S>> for SparseViewDataset {
    type Error = Error;

    fn try_from(source: ColmapSource<S>) -> Result<Self, Self::Error> {
        let points = source
            .points
            .into_iter()
            .map(|point| Point {
                color_rgb: point.color_rgb_normalized(),
                position: point.position,
            })
            .collect();

        let images_encoded = source
            .images_file
            .inner
            .into_par_iter()
            .map(|(image_file_name, mut image_file)| {
                // Checking the image file name
                if image_file_name != image_file.name {
                    Err(Error::MismatchedImageFileName(
                        image_file_name.to_owned(),
                        image_file.name.to_owned(),
                    ))?;
                }
                Ok((image_file_name, image_file.read()?))
            })
            .collect::<Result<dashmap::DashMap<_, _>, Self::Error>>()?;

        let cameras = source
            .images
            .inner
            .into_par_iter()
            .map(|(id, image)| {
                // Checking the image id
                if id != image.image_id {
                    Err(Error::MismatchedImageId(id, image.image_id))?;
                }

                // Specifying the parameters
                let camera_id = image.camera_id;
                let camera = source
                    .cameras
                    .get(&camera_id)
                    .ok_or(Error::UnknownCameraId(camera_id))?;
                let field_of_view_x = (camera.width() as f64)
                    .atan2(2.0 * camera.focal_length_x())
                    .mul(2.0);
                let field_of_view_y = (camera.height() as f64)
                    .atan2(2.0 * camera.focal_length_y())
                    .mul(2.0);
                let image_file_name = image.file_name;
                let view_rotation = View::rotation(&image.quaternion);
                let view_position =
                    View::position(&view_rotation, &image.translation);
                let view_transform =
                    View::transform(&view_rotation, &image.translation);
                let image_encoded = images_encoded
                    .remove(&image_file_name)
                    .ok_or(Error::UnknownImageFileName(
                        image_file_name.to_owned(),
                    ))?
                    .1;

                // Image
                let image = Image {
                    image_encoded,
                    image_file_name,
                    image_id: id,
                };
                let (image_width, image_height) = image.decode_dimensions()?;

                // View
                let view = View {
                    field_of_view_x,
                    field_of_view_y,
                    image_height,
                    image_width,
                    view_id: id,
                    view_position,
                    view_transform,
                };

                // Camera
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
            target: "gausplat::trainer::dataset::sparse_view",
            "SparseViewDataset > try_from(ColmapSource)",
        );

        Ok(Self { cameras, points })
    }
}

impl fmt::Debug for SparseViewDataset {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("SparseViewDataset")
            .field("cameras.len()", &self.cameras.len())
            .field("points.len()", &self.points.len())
            .finish()
    }
}

impl Default for SparseViewDataset {
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

        let dataset = SparseViewDataset::default();
        assert!(dataset.cameras.is_empty(), "{:?}", dataset.cameras);
        assert!(!dataset.points.is_empty(), "{:?}", dataset.points);
    }

    #[test]
    fn default_try_from_colmap() {
        use super::*;

        SparseViewDataset::try_from(ColmapSource::<&[u8]> {
            cameras: [Default::default()].into(),
            images: [Default::default()].into(),
            images_file: [Default::default()].into(),
            points: [Default::default()].into(),
        })
        .unwrap_err();

        SparseViewDataset::try_from(ColmapSource::<&[u8]>::default()).unwrap();
    }
}

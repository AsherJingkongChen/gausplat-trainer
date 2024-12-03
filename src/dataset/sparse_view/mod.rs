pub mod camera;

pub use crate::error::Error;
pub use camera::*;
pub use gausplat_loader::source::colmap::{self, ColmapSource};
pub use gausplat_renderer::scene::point::*;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{ffi::OsStr, fmt, io::Read, ops::Mul};

#[derive(Clone, PartialEq)]
pub struct SparseViewDataset {
    pub cameras: Cameras,
    pub points: Points,
}

impl SparseViewDataset {
    pub fn init_from_colmap<S: Read + Send + Sync>(
        source: ColmapSource<S>
    ) -> Result<Self, Error> {
        let points = source.points.into_iter().map(Into::into).collect();

        let images_file = source
            .images_file
            .into_par_iter()
            .map(|(image_file_path, image_file)| {
                if image_file_path != image_file.path {
                    return Err(Error::MismatchedImageFilePath(
                        image_file_path,
                        image_file.path,
                    ));
                }

                let image_file_name = image_file_path
                    .file_name()
                    .filter(|_| image_file_path.is_file())
                    .ok_or_else(|| Error::IoIsADirectory(image_file_path.to_owned()))?
                    .to_owned();

                Ok((image_file_name, image_file))
            })
            .collect::<Result<dashmap::DashMap<_, _>, Error>>()?;

        let cameras = source
            .images
            .into_par_iter()
            .map(|(id, image)| {
                // Checking the image id
                if id != image.image_id {
                    return Err(Error::MismatchedImageId(id, image.image_id));
                }

                // Specifying the parameters
                let camera_id = image.camera_id;
                let camera = source
                    .cameras
                    .get(&camera_id)
                    .ok_or(Error::UnknownCameraId(camera_id))?;
                let field_of_view_x = (camera.width as f64)
                    .atan2(2.0 * camera.focal_length_x())
                    .mul(2.0);
                let field_of_view_y = (camera.height as f64)
                    .atan2(2.0 * camera.focal_length_y())
                    .mul(2.0);
                // NOTE: Generally, the file name encoding is UTF-8 in COLMAP model.
                let image_file_name =
                    OsStr::new(image.file_name.to_str().map_err(|_| {
                        Error::InvalidUtf8(image.file_name.to_string_lossy().into())
                    })?);
                let mut image_file = images_file
                    .remove(image_file_name)
                    .map(|p| p.1)
                    .ok_or_else(|| Error::UnknownImageFileName(image_file_name.into()))?;
                // NOTE: Reading the image file at this point is more memory efficient.
                let image_encoded = image_file.read_all()?;
                let image_file_path = image_file.path;
                let view_rotation = View::rotation(&image.quaternion);
                let view_position = View::position(&view_rotation, &image.translation);
                let view_transform = View::transform(&view_rotation, &image.translation);

                // Image
                let image = Image {
                    image_encoded,
                    image_file_path,
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
            .collect::<Result<_, Error>>()?;

        #[cfg(all(debug_assertions, not(test)))]
        log::debug!(
            target: "gausplat::trainer::dataset::sparse_view",
            "SparseViewDataset::init_from_colmap",
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
    fn default_init_from_colmap() {
        use super::*;

        SparseViewDataset::init_from_colmap(ColmapSource::<&[u8]> {
            cameras: [Default::default()].into(),
            images: [Default::default()].into(),
            images_file: [Default::default()].into(),
            points: [Default::default()].into(),
        })
        .unwrap_err();

        SparseViewDataset::init_from_colmap(ColmapSource::<&[u8]>::default()).unwrap();
    }
}

//! Sparse view camera module.

pub mod cameras;

pub use cameras::*;
pub use gausplat_loader::source::image::*;
pub use gausplat_renderer::render::view::*;

/// A camera for a sparse view.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Camera {
    /// Camera ID.
    ///
    /// This is the same as the image ID and view ID.
    pub camera_id: u32,
    /// Image.
    pub image: Image,
    /// View.
    pub view: View,
}

/// Dimension operations
impl Camera {
    /// Resizing the camera to the maximum side length of `to`.
    pub fn resize_max(
        &mut self,
        to: u32,
    ) -> Result<&mut Self, Error> {
        self.image.resize_max(to)?;
        self.view.resize_max(to);
        Ok(self)
    }
}

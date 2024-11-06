pub mod cameras;

pub use cameras::*;
pub use gausplat_loader::source::image::*;
pub use gausplat_renderer::render::view::*;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Camera {
    /// `camera_id == self.image.image_id && camera_id == self.view.view_id`
    pub camera_id: u32,
    pub image: Image,
    pub view: View,
}

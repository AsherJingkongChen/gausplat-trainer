//! A collection of cameras.

pub use super::Camera;

/// A map of [`Camera::camera_id`] to [`Camera`].
pub type Cameras = gausplat_loader::collection::IndexMap<u32, Camera>;

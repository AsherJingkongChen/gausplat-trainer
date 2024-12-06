//! Error module.

use std::path::PathBuf;

/// Error variants.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Error from I/O operations.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// Error from I/O operations (is a directory).
    #[error("IO error: is a directory: {0:?}")]
    IoIsADirectory(PathBuf),
    /// Error from invalid UTF-8 string.
    #[error("Invalid UTF-8 string: {0:?}")]
    InvalidUtf8(String),
    /// Error from [`gausplat_loader`].
    #[error("Gausplat loader error: {0}")]
    Loader(#[from] gausplat_loader::error::Error),
    /// Error from mismatched image file path.
    #[error("Mismatched image file path: {0:?}. It should be {1:?}.")]
    MismatchedImageFilePath(PathBuf, PathBuf),
    /// Error from mismatched image id.
    #[error("Mismatched image id: {0}. It should be {1}.")]
    MismatchedImageId(u32, u32),
    /// Error from mismatched tensor shape.
    #[error("Mismatched tensor shape: {0:?}. It should be {1:?}.")]
    MismatchedTensorShape(Vec<usize>, Vec<usize>),
    /// Error from [`gausplat_renderer`].
    #[error("Render error: {0}")]
    Render(#[from] gausplat_renderer::error::Error),
    /// Error from unknown camera id.
    #[error("Unknown camera id: {0}")]
    UnknownCameraId(u32),
    /// Error from unknown image file name.
    #[error("Unknown image file name: {0:?}")]
    UnknownImageFileName(PathBuf),
}

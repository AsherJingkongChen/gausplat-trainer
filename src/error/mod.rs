use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("IO error: is a directory: {0:?}")]
    IoIsADirectory(PathBuf),

    #[error("Invalid UTF-8 string: {0:?}")]
    InvalidUtf8(String),

    #[error("Gausplat loader error: {0}")]
    Loader(#[from] gausplat_loader::error::Error),

    #[error("Mismatched image file path: {0:?}. It should be {1:?}.")]
    MismatchedImageFilePath(PathBuf, PathBuf),

    #[error("Mismatched image id: {0}. It should be {1}.")]
    MismatchedImageId(u32, u32),

    #[error("Mismatched tensor shape: {0:?}. It should be {1:?}.")]
    MismatchedTensorShape(Vec<usize>, Vec<usize>),

    #[error("Render error: {0}")]
    Render(#[from] gausplat_renderer::error::Error),

    #[error("Unknown camera id: {0}")]
    UnknownCameraId(u32),

    #[error("Unknown image file name: {0:?}")]
    UnknownImageFileName(PathBuf),
}

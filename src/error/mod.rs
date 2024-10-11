#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Import error: {0}")]
    Import(#[from] gausplat_loader::error::Error),

    #[error("Unknown camera id: {0}")]
    UnknownCameraId(u32),

    #[error("Unknown image file name: {0}")]
    UnknownImageFileName(String),
}

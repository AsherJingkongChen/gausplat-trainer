#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Import error: {0}")]
    Import(#[from] gausplat_loader::error::Error),

    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Unknown camera id: {0}")]
    UnknownCameraId(u32),

    #[error("Unknown image file name: {0}")]
    UnknownImageFileName(String),
}

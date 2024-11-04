#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Import error: {0}")]
    Import(#[from] gausplat_loader::error::Error),

    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Mismatched tensor shape: {0:?}. It should be {1:?}.")]
    MismatchedTensorShape(Vec<usize>, Vec<usize>),

    #[error("Render error: {0}")]
    Render(#[from] gausplat_renderer::error::Error),

    #[error("Unknown camera id: {0}")]
    UnknownCameraId(u32),

    #[error("Unknown image file name: {0}")]
    UnknownImageFileName(String),
}

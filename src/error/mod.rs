#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Import error: {0}")]
    Import(#[from] gausplat_importer::error::Error),
}

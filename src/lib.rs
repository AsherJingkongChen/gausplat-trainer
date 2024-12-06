//! A library to train scenes for `gausplat`

#![deny(rustdoc::broken_intra_doc_links)]
#![allow(clippy::excessive_precision)]
#![deny(missing_docs)]

pub mod dataset;
pub mod error;
pub mod metric;
pub mod optimize;
pub mod range;
pub mod train;

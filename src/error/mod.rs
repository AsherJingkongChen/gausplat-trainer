use std::{error, fmt};

#[derive(Debug)]
pub enum Error {
    InvalidConfig(String),
}

impl fmt::Display for Error {
    fn fmt(
        &self,
        f: &mut fmt::Formatter,
    ) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl error::Error for Error {}

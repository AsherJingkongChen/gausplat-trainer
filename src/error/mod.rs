use std::{error, fmt};

#[derive(Debug)]
pub enum Error {
    RandomNormalDistribution(rand_distr::NormalError),
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

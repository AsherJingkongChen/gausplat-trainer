#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Random Normal Distribution Error: {0}")]
    RandomNormalDistribution(rand_distr::NormalError),
}

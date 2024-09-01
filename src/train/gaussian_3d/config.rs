pub use super::*;
pub use burn::config::Config;

#[derive(Config, Copy, Debug)]
pub struct Gaussian3dTrainerConfig {}

impl Gaussian3dTrainerConfig {
}

impl Default for Gaussian3dTrainerConfig {
    fn default() -> Self {
        Self::new()
    }
}

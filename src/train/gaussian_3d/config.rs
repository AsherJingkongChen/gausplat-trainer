pub use super::*;
pub use burn::config::Config;
pub use gausplat_importer::dataset::gaussian_3d::{Point, Points};

use burn::optim::AdamConfig;
use std::ops::Div;

#[derive(Config, Debug)]
pub struct Gaussian3dTrainerConfig {
    #[config(default = "2.5e-3")]
    pub colors_sh_learning_rate: LearningRate,

    #[config(default = "2.5e-2")]
    pub opacities_learning_rate: LearningRate,

    #[config(default = "30000")]
    pub positions_learning_count: u64,

    #[config(default = "1.6e-4")]
    pub positions_learning_rate: LearningRate,

    #[config(default = "1.6e-6")]
    pub positions_learning_rate_end: LearningRate,

    #[config(default = "Gaussian3dRendererOptions::default()")]
    pub render_options: Gaussian3dRendererOptions,

    #[config(default = "1e-3")]
    pub rotations_learning_rate: LearningRate,

    #[config(default = "5e-3")]
    pub scalings_learning_rate: LearningRate,
}

impl Gaussian3dTrainerConfig {
    pub fn init<AB: AutodiffBackend>(
        &self,
        device: &AB::Device,
        priors: Points,
    ) -> Gaussian3dTrainer<AB> {
        let param_optimizer = AdamConfig::new().with_epsilon(1e-15);
        Gaussian3dTrainer {
            config: self.to_owned(),
            metric_optimization: Default::default(),
            param_optimizer_2d: param_optimizer.init(),
            param_optimizer_3d: param_optimizer.init(),
            positions_learning_rate_decay: Self::learning_rate_decay(
                self.positions_learning_count,
                self.positions_learning_rate,
                self.positions_learning_rate_end,
            ),
            scene: Gaussian3dScene::init(device, priors),
        }
    }

    #[inline]
    fn learning_rate_decay(
        learning_count: u64,
        learning_rate: LearningRate,
        learning_rate_end: LearningRate,
    ) -> LearningRate {
        learning_rate_end
            .div(learning_rate)
            .powf((learning_count as LearningRate).recip())
    }
}

impl Default for Gaussian3dTrainerConfig {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn learning_rate_decay() {
        use super::*;

        let config = Gaussian3dTrainerConfig::default()
            .with_positions_learning_count(7000);
        let decay = Gaussian3dTrainerConfig::learning_rate_decay(
            config.positions_learning_count,
            config.positions_learning_rate,
            config.positions_learning_rate_end,
        );
        assert_eq!(decay, 0.9993423349014151);

        let config = Gaussian3dTrainerConfig::default()
            .with_positions_learning_count(30000);
        let decay = Gaussian3dTrainerConfig::learning_rate_decay(
            config.positions_learning_count,
            config.positions_learning_rate,
            config.positions_learning_rate_end,
        );
        assert_eq!(decay, 0.9998465061085267);
    }
}

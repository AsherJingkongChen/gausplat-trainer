pub use super::*;
pub use gausplat_importer::dataset::gaussian_3d::{Point, Points};

use std::ops::Div;

#[derive(Config, Debug)]
pub struct Gaussian3dTrainerConfig {
    #[config(default = "30000")]
    pub learning_count_positions: u64,

    #[config(default = "2.5e-3")]
    pub learning_rate_colors_sh: LearningRate,

    #[config(default = "2.5e-2")]
    pub learning_rate_opacities: LearningRate,

    #[config(default = "1.6e-4")]
    pub learning_rate_positions: LearningRate,

    #[config(default = "1.6e-6")]
    pub learning_rate_positions_end: LearningRate,

    #[config(default = "1e-3")]
    pub learning_rate_rotations: LearningRate,

    #[config(default = "5e-3")]
    pub learning_rate_scalings: LearningRate,

    #[config(default = "AdamConfig::new().with_epsilon(1e-15)")]
    pub optimizer_adam: AdamConfig,

    #[config(default = "Gaussian3dRendererOptions::new()")]
    pub options_renderer: Gaussian3dRendererOptions,
    // densify

    // # opt.densify_from_iter # 500
    // # opt.densify_until_iter # 15000
    // # opt.densification_interval # 100

    // # opt.densify_grad_threshold # 2e-4
    // # opt.densify_opacity_threshold = 5e-3
    // # opt.densify_size_threshold = 20

    // opt.sh_update_interval # 1000
}

impl Gaussian3dTrainerConfig {
    pub fn init<AB: AutodiffBackend>(
        &self,
        device: &AB::Device,
        priors: Points,
    ) -> Gaussian3dTrainer<AB> {
        let learning_rate_decay_positions = Self::learning_rate_decay(
            self.learning_count_positions,
            self.learning_rate_positions,
            self.learning_rate_positions_end,
        );

        Gaussian3dTrainer {
            config: self.to_owned(),
            iteration: 0,
            learning_rate_decay_positions,
            metric_optimization: MeanAbsoluteError::init(),
            optimizer_colors_sh: self.optimizer_adam.init(),
            optimizer_opacities: self.optimizer_adam.init(),
            optimizer_positions: self.optimizer_adam.init(),
            optimizer_rotations: self.optimizer_adam.init(),
            optimizer_scalings: self.optimizer_adam.init(),
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
            .with_learning_count_positions(7000);
        let decay = Gaussian3dTrainerConfig::learning_rate_decay(
            config.learning_count_positions,
            config.learning_rate_positions,
            config.learning_rate_positions_end,
        );
        assert_eq!(decay, 0.9993423349014151);

        let config = Gaussian3dTrainerConfig::default()
            .with_learning_count_positions(30000);
        let decay = Gaussian3dTrainerConfig::learning_rate_decay(
            config.learning_count_positions,
            config.learning_rate_positions,
            config.learning_rate_positions_end,
        );
        assert_eq!(decay, 0.9998465061085267);
    }
}

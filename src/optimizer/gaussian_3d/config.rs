pub use super::*;
pub use burn::{config::Config, LearningRate};

use burn::optim::AdamConfig;

#[derive(Config, Copy, Debug)]
pub struct Gaussian3dOptimizerConfig {
    #[config(default = "7000")]
    pub iteration_count: usize,

    #[config(default = "2.5e-3")]
    pub colors_sh_learning_rate: LearningRate,

    #[config(default = "2.5e-2")]
    pub opacities_learning_rate: LearningRate,

    #[config(default = "(1.6e-4, 1.6e-6)")]
    pub positions_learning_rates: (LearningRate, LearningRate),

    #[config(default = "1e-3")]
    pub rotations_learning_rate: LearningRate,

    #[config(default = "5e-3")]
    pub scalings_learning_rate: LearningRate,
}

impl Gaussian3dOptimizerConfig {
    pub fn init<AB: AutodiffBackend>(&self) -> Gaussian3dOptimizer<AB> {
        let param_updater = AdamConfig::new().with_epsilon(1e-15);

        Gaussian3dOptimizer {
            colors_sh_learning_rate: self.colors_sh_learning_rate,
            colors_sh_param_updater: param_updater.init(),
            opacities_learning_rate: self.opacities_learning_rate,
            opacities_param_updater: param_updater.init(),
            positions_learning_rate: self.positions_learning_rates.0,
            positions_learning_rate_decay: Self::learning_rate_decay(
                self.positions_learning_rates,
                self.iteration_count,
            ),
            positions_param_updater: param_updater.init(),
            rotations_learning_rate: self.rotations_learning_rate,
            rotations_param_updater: param_updater.init(),
            scalings_learning_rate: self.scalings_learning_rate,
            scalings_param_updater: param_updater.init(),
        }
    }

    #[inline]
    fn learning_rate_decay(
        learning_rates: (LearningRate, LearningRate),
        iteration_count: usize,
    ) -> LearningRate {
        (learning_rates.1 / learning_rates.0)
            .powf((iteration_count as LearningRate).recip())
    }
}

impl Default for Gaussian3dOptimizerConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn learning_rate_decay() {
        use super::*;

        let config =
            Gaussian3dOptimizerConfig::default().with_iteration_count(7000);
        let decay = Gaussian3dOptimizerConfig::learning_rate_decay(
            config.positions_learning_rates,
            config.iteration_count,
        );
        assert_eq!(decay, 0.9993423349014151);

        let config =
            Gaussian3dOptimizerConfig::default().with_iteration_count(30000);
        let decay = Gaussian3dOptimizerConfig::learning_rate_decay(
            config.positions_learning_rates,
            config.iteration_count,
        );
        assert_eq!(decay, 0.9998465061085267);
    }
}

//! 3DGS trainer configuration.

pub use super::*;
pub use crate::optimize::{AdamConfig, LearningRateConfig};

/// 3DGS trainer configuration.
#[derive(Config, Copy, Debug, PartialEq)]
pub struct Gaussian3dTrainerConfig {
    /// Learning rate for colors SH.
    #[config(default = "1e-3.into()")]
    pub learning_rate_colors_sh: LearningRateConfig,
    /// Learning rate for opacities.
    #[config(default = "3.5e-2.into()")]
    pub learning_rate_opacities: LearningRateConfig,
    /// Learning rate for positions.
    #[config(
        default = "LearningRateConfig::new(1.6e-4).with_end(1.6e-6).with_count(30000)"
    )]
    pub learning_rate_positions: LearningRateConfig,
    /// Learning rate for rotations.
    #[config(default = "1e-3.into()")]
    pub learning_rate_rotations: LearningRateConfig,
    /// Learning rate for scalings.
    #[config(default = "5e-3.into()")]
    pub learning_rate_scalings: LearningRateConfig,
    /// Adam optimizer configuration.
    #[config(default = "AdamConfig::default().with_epsilon(1e-15)")]
    pub optimizer_adam: AdamConfig,
    /// Renderer options.
    #[config(
        default = "Gaussian3dRenderOptions::default().with_colors_sh_degree_max(0)"
    )]
    pub options_renderer: Gaussian3dRenderOptions,
    /// Range for metric optimization (fine).
    #[config(default = "RangeOptions::default_with_step(2)")]
    pub range_metric_optimization_fine: RangeOptions,
    /// Refiner configuration.
    #[config(default = "Default::default()")]
    pub refiner: RefinerConfig,
}

impl Gaussian3dTrainerConfig {
    /// Initialize the trainer.
    pub fn init<AB: AutodiffBackend>(
        &self,
        device: &AB::Device,
    ) -> Gaussian3dTrainer<AB> {
        AB::seed(SEED);

        Gaussian3dTrainer {
            iteration: 0,
            learning_rate_colors_sh: self.learning_rate_colors_sh.init(),
            learning_rate_opacities: self.learning_rate_opacities.init(),
            learning_rate_positions: self.learning_rate_positions.init(),
            learning_rate_rotations: self.learning_rate_rotations.init(),
            learning_rate_scalings: self.learning_rate_scalings.init(),
            metric_optimization_coarse: metric::MeanAbsoluteError::init(),
            metric_optimization_fine: metric::MeanStructuralDissimilarity::init(device),
            optimizer_colors_sh: self.optimizer_adam.init(),
            optimizer_opacities: self.optimizer_adam.init(),
            optimizer_positions: self.optimizer_adam.init(),
            optimizer_rotations: self.optimizer_adam.init(),
            optimizer_scalings: self.optimizer_adam.init(),
            options_renderer: self.options_renderer,
            range_metric_optimization_fine: self.range_metric_optimization_fine,
            refiner: self.refiner.init(),
        }
    }
}

impl Default for Gaussian3dTrainerConfig {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

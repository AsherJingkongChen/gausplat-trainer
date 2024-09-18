pub use super::*;
pub use crate::optimize::{AdamConfig, LearningRateConfig};

use gausplat_renderer::preset::gaussian_3d::SEED;

#[derive(Config, Debug)]
pub struct Gaussian3dTrainerConfig {
    #[config(default = "2.5e-3.into()")]
    pub learning_rate_colors_sh: LearningRateConfig,

    #[config(default = "2.5e-2.into()")]
    pub learning_rate_opacities: LearningRateConfig,

    #[config(
        default = "LearningRateConfig::new(1.6e-4).with_end(1.6e-6).with_count(30000)"
    )]
    pub learning_rate_positions: LearningRateConfig,

    #[config(default = "1e-3.into()")]
    pub learning_rate_rotations: LearningRateConfig,

    #[config(default = "5e-3.into()")]
    pub learning_rate_scalings: LearningRateConfig,

    #[config(default = "AdamConfig::default().with_epsilon(1e-15)")]
    pub optimizer_adam: AdamConfig,

    #[config(
        default = "Gaussian3dRendererOptions::default().with_colors_sh_degree_max(0)"
    )]
    pub options_renderer: Gaussian3dRendererOptions,

    #[config(default = "Default::default()")]
    pub refiner: RefinerConfig,
}

impl Gaussian3dTrainerConfig {
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
            metric_optimization_fine: metric::MeanStructuralDissimilarity::init(
                device,
            ),
            optimizer_colors_sh: self.optimizer_adam.init(),
            optimizer_opacities: self.optimizer_adam.init(),
            optimizer_positions: self.optimizer_adam.init(),
            optimizer_rotations: self.optimizer_adam.init(),
            optimizer_scalings: self.optimizer_adam.init(),
            options_renderer: self.options_renderer.to_owned(),
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

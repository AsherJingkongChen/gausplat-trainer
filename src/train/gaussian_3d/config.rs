pub use super::*;
pub use gausplat_importer::dataset::gaussian_3d::{Point, Points};

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

    #[config(default = "AdamConfig::new().with_epsilon(1e-15)")]
    pub optimizer_adam: AdamConfig,

    #[config(
        default = "Gaussian3dRendererOptions::new().with_colors_sh_degree_max(0)"
    )]
    pub options_renderer: Gaussian3dRendererOptions,

    #[config(default = "Default::default()")]
    pub refiner: RefinerConfig,
}

impl Gaussian3dTrainerConfig {
    pub fn init<AB: AutodiffBackend>(
        &self,
        device: &AB::Device,
        priors: Points,
    ) -> Gaussian3dTrainer<AB> {
        Gaussian3dTrainer {
            iteration: 0,
            learning_rate_colors_sh: self.learning_rate_colors_sh.init(),
            learning_rate_opacities: self.learning_rate_opacities.init(),
            learning_rate_positions: self.learning_rate_positions.init(),
            learning_rate_rotations: self.learning_rate_rotations.init(),
            learning_rate_scalings: self.learning_rate_scalings.init(),
            metric_optimization_1: metric::MeanAbsoluteError::init(),
            metric_optimization_2: metric::MeanStructuralDissimilarity::init(
                device,
            ),
            optimizer_colors_sh: self.optimizer_adam.init(),
            optimizer_opacities: self.optimizer_adam.init(),
            optimizer_positions: self.optimizer_adam.init(),
            optimizer_rotations: self.optimizer_adam.init(),
            optimizer_scalings: self.optimizer_adam.init(),
            options_renderer: self.options_renderer.to_owned(),
            scene: Gaussian3dScene::init(device, priors),
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

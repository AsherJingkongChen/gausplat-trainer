pub mod config;
pub mod optimize;

pub use burn::{
    tensor::backend::AutodiffBackend,
    LearningRate,
};
pub use config::*;
pub use optimize::*;
pub use gausplat_importer::dataset::gaussian_3d::Camera;
pub use gausplat_renderer::scene::gaussian_3d::{
    Gaussian3dRenderer, Gaussian3dScene,
};
pub use rand::{rngs::StdRng, SeedableRng};

use burn::{
    module::Param,
    optim::{adaptor::OptimizerAdaptor, Adam},
    tensor::Tensor,
};
use std::fmt;

pub type AdamParamOptimizer<AB, const D: usize> = OptimizerAdaptor<
    Adam<<AB as AutodiffBackend>::InnerBackend>,
    Param<Tensor<AB, D>>,
    AB,
>;

pub type Cameras = indexmap::IndexMap<u32, Camera>;

#[derive(Clone)]
pub struct Gaussian3dTrainer<AB: AutodiffBackend> {
    pub cameras: Cameras,
    pub colors_sh_learning_rate: LearningRate,
    pub iteration: u64,
    pub opacities_learning_rate: LearningRate,
    pub param_optimizer_2d: AdamParamOptimizer<AB, 2>,
    pub param_optimizer_3d: AdamParamOptimizer<AB, 3>,
    pub positions_learning_rate: LearningRate,
    pub positions_learning_rate_decay: LearningRate,
    pub positions_learning_rate_end: LearningRate,
    pub rng: StdRng,
    pub rotations_learning_rate: LearningRate,
    pub scalings_learning_rate: LearningRate,
    pub scene: Gaussian3dScene<AB>,
}

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> where
    Gaussian3dScene<AB>: Gaussian3dRenderer<AB>
{
    pub fn train(&mut self) -> <Self as Iterator>::Item {
        0
    }
}

impl<AB: AutodiffBackend> Iterator for Gaussian3dTrainer<AB>
where
    Gaussian3dScene<AB>: Gaussian3dRenderer<AB>,
{
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.train())
    }
}

impl<AB: AutodiffBackend> fmt::Debug for Gaussian3dTrainer<AB> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("Gaussian3dTrainer")
            .field("cameras.len()", &self.cameras.len())
            .field("colors_sh_learning_rate", &self.colors_sh_learning_rate)
            .field("iteration", &self.iteration)
            .field("opacities_learning_rate", &self.opacities_learning_rate)
            .field("param_optimizer_2d", &format!("Adam<{}>", AB::name()))
            .field("param_optimizer_3d", &format!("Adam<{}>", AB::name()))
            .field("positions_learning_rate", &self.positions_learning_rate)
            .field(
                "positions_learning_rate_decay",
                &self.positions_learning_rate_decay,
            )
            .field(
                "positions_learning_rate_end",
                &self.positions_learning_rate_end,
            )
            .field("rng", &self.rng)
            .field("rotations_learning_rate", &self.rotations_learning_rate)
            .field("scalings_learning_rate", &self.scalings_learning_rate)
            .field("scene", &self.scene)
            .finish()
    }
}

impl<AB: AutodiffBackend> Default for Gaussian3dTrainer<AB> {
    fn default() -> Self {
        Gaussian3dTrainerConfig::default()
            .init(&Default::default(), Default::default())
    }
}

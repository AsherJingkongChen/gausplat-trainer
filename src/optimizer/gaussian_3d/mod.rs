pub mod config;

pub use burn::{optim::GradientsParams, tensor::backend::AutodiffBackend};
pub use config::*;
pub use gausplat_renderer::scene::gaussian_3d::Gaussian3dScene;

use burn::{
    module::Param,
    optim::{adaptor::OptimizerAdaptor, Adam, Optimizer},
    tensor::Tensor,
};
use std::fmt;

pub type AdamOptimizerForParam<AB, const D: usize> = OptimizerAdaptor<
    Adam<<AB as AutodiffBackend>::InnerBackend>,
    Param<Tensor<AB, D>>,
    AB,
>;

#[derive(Clone)]
pub struct Gaussian3dOptimizer<AB: AutodiffBackend> {
    colors_sh_learning_rate: LearningRate,
    colors_sh_param_updater: AdamOptimizerForParam<AB, 3>,
    opacities_learning_rate: LearningRate,
    opacities_param_updater: AdamOptimizerForParam<AB, 2>,
    positions_learning_rate: LearningRate,
    positions_learning_rate_decay: LearningRate,
    positions_param_updater: AdamOptimizerForParam<AB, 2>,
    rotations_learning_rate: LearningRate,
    rotations_param_updater: AdamOptimizerForParam<AB, 2>,
    scalings_learning_rate: LearningRate,
    scalings_param_updater: AdamOptimizerForParam<AB, 2>,
}

impl<AB: AutodiffBackend> Gaussian3dOptimizer<AB> {
    pub fn step(
        &mut self,
        mut module: Gaussian3dScene<AB>,
        mut grads: GradientsParams,
    ) -> Gaussian3dScene<AB> {
        // Update the parameters

        module.colors_sh = Self::step_param(
            &mut self.colors_sh_param_updater,
            self.colors_sh_learning_rate,
            module.colors_sh,
            &mut grads,
        );
        module.opacities = Self::step_param(
            &mut self.opacities_param_updater,
            self.opacities_learning_rate,
            module.opacities,
            &mut grads,
        );
        module.positions = Self::step_param(
            &mut self.positions_param_updater,
            self.positions_learning_rate,
            module.positions,
            &mut grads,
        );
        module.rotations = Self::step_param(
            &mut self.rotations_param_updater,
            self.rotations_learning_rate,
            module.rotations,
            &mut grads,
        );
        module.scalings = Self::step_param(
            &mut self.scalings_param_updater,
            self.scalings_learning_rate,
            module.scalings,
            &mut grads,
        );

        // Update the learning rates

        self.positions_learning_rate *= self.positions_learning_rate_decay;

        module
    }

    fn step_param<O: Optimizer<Param<Tensor<AB, D>>, AB>, const D: usize>(
        optimizer: &mut O,
        learning_rate: LearningRate,
        mut param: Param<Tensor<AB, D>>,
        grads: &mut GradientsParams,
    ) -> Param<Tensor<AB, D>> {
        let id = &param.id;
        if let Some(grad) = grads.remove::<AB, D>(id) {
            let mut grads = GradientsParams::new();
            grads.register(id.to_owned(), grad);
            param = optimizer.step(learning_rate, param, grads);
        }
        param
    }
}

impl<AB: AutodiffBackend> fmt::Debug for Gaussian3dOptimizer<AB> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("Gaussian3dOptimizer")
            .field("param_updater", &format!("Adam<{}>", AB::name()))
            .field("colors_sh_learning_rate", &self.colors_sh_learning_rate)
            .field("opacities_learning_rate", &self.opacities_learning_rate)
            .field("positions_learning_rate", &self.positions_learning_rate)
            .field(
                "positions_learning_rate_decay",
                &self.positions_learning_rate_decay,
            )
            .field("rotations_learning_rate", &self.rotations_learning_rate)
            .field("scalings_learning_rate", &self.scalings_learning_rate)
            .finish()
    }
}

impl<AB: AutodiffBackend> Default for Gaussian3dOptimizer<AB> {
    fn default() -> Self {
        Gaussian3dOptimizerConfig::default().init()
    }
}

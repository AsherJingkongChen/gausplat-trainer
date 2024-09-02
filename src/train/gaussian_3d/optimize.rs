pub use super::*;

pub use burn::optim::{GradientsParams, Optimizer};

use burn::{module::Param, tensor::Tensor};
use std::ops::Mul;

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
    pub fn optimize(
        &mut self,
        mut grads: GradientsParams,
    ) -> &mut Self {
        let scene = self.scene.to_owned();

        debug_assert_eq!(scene.colors_sh.id, self.scene.colors_sh.id);

        self.scene.colors_sh = Self::optimize_param(
            &mut self.param_optimizer_3d,
            self.colors_sh_learning_rate,
            scene.colors_sh,
            &mut grads,
        );
        self.scene.opacities = Self::optimize_param(
            &mut self.param_optimizer_2d,
            self.opacities_learning_rate,
            scene.opacities,
            &mut grads,
        );
        self.scene.positions = Self::optimize_param(
            &mut self.param_optimizer_2d,
            self.positions_learning_rate,
            scene.positions,
            &mut grads,
        );
        self.scene.rotations = Self::optimize_param(
            &mut self.param_optimizer_2d,
            self.rotations_learning_rate,
            scene.rotations,
            &mut grads,
        );
        self.scene.scalings = Self::optimize_param(
            &mut self.param_optimizer_2d,
            self.scalings_learning_rate,
            scene.scalings,
            &mut grads,
        );

        // Scheduling the learning rates

        self.positions_learning_rate = self
            .positions_learning_rate
            .mul(self.positions_learning_rate_decay)
            .max(self.positions_learning_rate_end);

        self
    }

    fn optimize_param<const D: usize>(
        optimizer: &mut impl Optimizer<Param<Tensor<AB, D>>, AB>,
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

pub use super::*;

pub use burn::optim::{GradientsParams, Optimizer};

use std::ops::Mul;

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
    pub fn optimize(
        &mut self,
        mut grads: GradientsParams,
    ) -> &mut Self {
        let scene = self.scene.to_owned();

        self.scene.colors_sh = Self::optimize_param(
            &mut self.param_optimizer_3d,
            self.config.learning_rate_colors_sh,
            scene.colors_sh,
            &mut grads,
        );
        self.scene.opacities = Self::optimize_param(
            &mut self.param_optimizer_2d,
            self.config.learning_rate_opacities,
            scene.opacities,
            &mut grads,
        );
        self.scene.positions = Self::optimize_param(
            &mut self.param_optimizer_2d,
            self.config.learning_rate_positions,
            scene.positions,
            &mut grads,
        );
        self.scene.rotations = Self::optimize_param(
            &mut self.param_optimizer_2d,
            self.config.learning_rate_rotations,
            scene.rotations,
            &mut grads,
        );
        self.scene.scalings = Self::optimize_param(
            &mut self.param_optimizer_2d,
            self.config.learning_rate_scalings,
            scene.scalings,
            &mut grads,
        );

        // Scheduling the learning rates

        self.config.learning_rate_positions = self
            .config
            .learning_rate_positions
            .mul(self.learning_rate_decay_positions)
            .max(self.config.learning_rate_positions_end);

        self
    }

    fn optimize_param<const D: usize>(
        optimizer: &mut impl Optimizer<Param<Tensor<AB, D>>, AB>,
        learning_rate: LearningRate,
        mut param: Param<Tensor<AB, D>>,
        grads: &mut GradientsParams,
    ) -> Param<Tensor<AB, D>> {
        let id = &param.id;

        if let Some(grad) = grads.remove::<AB::InnerBackend, D>(id) {
            let mut grads = GradientsParams::new();
            grads.register(id.to_owned(), grad);
            param = optimizer.step(learning_rate, param, grads);
        }

        param
    }
}

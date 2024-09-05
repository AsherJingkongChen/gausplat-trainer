pub mod adam;

pub use super::*;
pub use adam::*;

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
    pub fn optimize(
        &mut self,
        grads: &mut AB::Gradients,
    ) -> &mut Self {
        if let Some(grad) = self.scene.colors_sh.grad_remove(grads) {
            self.scene.set_colors_sh(self.optimizer_colors_sh.step(
                self.config.learning_rate_colors_sh,
                self.scene.colors_sh.val(),
                grad,
            ));
        }
        if let Some(grad) = self.scene.opacities.grad_remove(grads) {
            self.scene.set_opacities(self.optimizer_opacities.step(
                self.config.learning_rate_opacities,
                self.scene.opacities.val(),
                grad,
            ));
        }
        if let Some(grad) = self.scene.positions.grad_remove(grads) {
            self.scene.set_positions(self.optimizer_positions.step(
                self.config.learning_rate_positions,
                self.scene.positions.val(),
                grad,
            ));
        }
        if let Some(grad) = self.scene.rotations.grad_remove(grads) {
            self.scene.set_rotations(self.optimizer_rotations.step(
                self.config.learning_rate_rotations,
                self.scene.rotations.val(),
                grad,
            ));
        }
        if let Some(grad) = self.scene.scalings.grad_remove(grads) {
            self.scene.set_scalings(self.optimizer_scalings.step(
                self.config.learning_rate_scalings,
                self.scene.scalings.val(),
                grad,
            ));
        }

        // Scheduling the learning rates

        self.config.learning_rate_positions =
            Self::update_learning_rate_exponential(
                self.config.learning_rate_positions,
                self.config.learning_rate_positions_end,
                self.learning_rate_decay_positions,
            );

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::optimize");

        self
    }

    pub fn update_learning_rate_exponential(
        learning_rate: LearningRate,
        learning_rate_end: LearningRate,
        learning_rate_decay: LearningRate,
    ) -> LearningRate {
        (learning_rate * learning_rate_decay).max(learning_rate_end)
    }
}

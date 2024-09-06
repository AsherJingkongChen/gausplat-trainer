pub mod adam;
pub mod learning_rate;

pub use super::*;
pub use adam::*;
pub use learning_rate::*;

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
    pub fn optimize(
        &mut self,
        grads: &mut AB::Gradients,
    ) -> &mut Self {
        // Updating the parameters using the gradients

        if let Some(grad) = self.scene.colors_sh.grad_remove(grads) {
            self.scene
                .set_inner_colors_sh(self.optimizer_colors_sh.update(
                    *self.learning_rate_colors_sh,
                    self.scene.colors_sh.val(),
                    grad,
                ));
        }
        if let Some(grad) = self.scene.opacities.grad_remove(grads) {
            self.scene
                .set_inner_opacities(self.optimizer_opacities.update(
                    *self.learning_rate_opacities,
                    self.scene.opacities.val(),
                    grad,
                ));
        }
        if let Some(grad) = self.scene.positions.grad_remove(grads) {
            self.scene
                .set_inner_positions(self.optimizer_positions.update(
                    *self.learning_rate_positions,
                    self.scene.positions.val(),
                    grad,
                ));
        }
        if let Some(grad) = self.scene.rotations.grad_remove(grads) {
            self.scene
                .set_inner_rotations(self.optimizer_rotations.update(
                    *self.learning_rate_rotations,
                    self.scene.rotations.val(),
                    grad,
                ));
        }
        if let Some(grad) = self.scene.scalings.grad_remove(grads) {
            self.scene
                .set_inner_scalings(self.optimizer_scalings.update(
                    *self.learning_rate_scalings,
                    self.scene.scalings.val(),
                    grad,
                ));
        }

        // Updating the learning rates

        self.learning_rate_positions.update();

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::optimize");

        self
    }
}

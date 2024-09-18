pub mod adam;
pub mod learning_rate;

pub use super::*;
pub use adam::*;
pub use learning_rate::*;

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
    pub fn optimize(
        &mut self,
        scene: &mut Gaussian3dScene<AB>,
        grads: &mut AB::Gradients,
    ) -> &mut Self {
        // Updating the parameters using the gradients

        if let Some(grad) = scene.colors_sh.grad_remove(grads) {
            scene.set_inner_colors_sh(self.optimizer_colors_sh.update(
                *self.learning_rate_colors_sh,
                scene.colors_sh.val(),
                grad,
            ));
        }
        if let Some(grad) = scene.opacities.grad_remove(grads) {
            scene.set_inner_opacities(self.optimizer_opacities.update(
                *self.learning_rate_opacities,
                scene.opacities.val(),
                grad,
            ));
        }
        if let Some(grad) = scene.positions.grad_remove(grads) {
            scene.set_inner_positions(self.optimizer_positions.update(
                *self.learning_rate_positions,
                scene.positions.val(),
                grad,
            ));
        }
        if let Some(grad) = scene.rotations.grad_remove(grads) {
            scene.set_inner_rotations(self.optimizer_rotations.update(
                *self.learning_rate_rotations,
                scene.rotations.val(),
                grad,
            ));
        }
        if let Some(grad) = scene.scalings.grad_remove(grads) {
            scene.set_inner_scalings(self.optimizer_scalings.update(
                *self.learning_rate_scalings,
                scene.scalings.val(),
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

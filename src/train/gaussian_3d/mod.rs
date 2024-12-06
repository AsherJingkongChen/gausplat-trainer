//! 3DGS trainer.

pub mod config;
pub mod refine;

pub use crate::{
    dataset::{sparse_view, SparseViewDataset},
    error::Error,
    metric::{self, Metric},
    optimize::{Adam, AdamRecord, LearningRate, LearningRateRecord},
};
pub use burn::{config::Config, record::Record, tensor::Tensor};
pub use config::*;
pub use gausplat_renderer::scene::gaussian_3d::{
    backend::{self, *},
    render::{
        Gaussian3dRenderOptions, Gaussian3dRenderOutputAutodiff, Gaussian3dRenderer,
    },
    AutodiffModule, Gaussian3dScene, SEED,
};
pub use refine::*;

/// Trainer for 3DGS.
#[derive(Clone, Debug)]
pub struct Gaussian3dTrainer<AB: AutodiffBackend> {
    /// Current iteration.
    pub iteration: u64,
    /// Current learning rate for colors SH.
    pub learning_rate_colors_sh: LearningRate,
    /// Current learning rate for opacities.
    pub learning_rate_opacities: LearningRate,
    /// Current learning rate for positions.
    pub learning_rate_positions: LearningRate,
    /// Current learning rate for rotations.
    pub learning_rate_rotations: LearningRate,
    /// Current learning rate for scalings.
    pub learning_rate_scalings: LearningRate,
    /// Metric for optimization (coarse).
    pub metric_optimization_coarse: metric::MeanAbsoluteError,
    /// Metric for optimization (fine).
    pub metric_optimization_fine: metric::MeanStructuralDissimilarity<AB, 3>,
    /// Current optimizer for colors SH.
    pub optimizer_colors_sh: Adam<AB, 2>,
    /// Current optimizer for opacities.
    pub optimizer_opacities: Adam<AB, 2>,
    /// Current optimizer for positions.
    pub optimizer_positions: Adam<AB, 2>,
    /// Current optimizer for rotations.
    pub optimizer_rotations: Adam<AB, 2>,
    /// Current optimizer for scalings.
    pub optimizer_scalings: Adam<AB, 2>,
    /// Current renderer options.
    pub options_renderer: Gaussian3dRenderOptions,
    /// Current refiner.
    pub range_metric_optimization_fine: RangeOptions,
    /// Current refiner.
    pub refiner: Refiner<AB::InnerBackend>,
}

/// Trainer record for 3DGS.
#[derive(Clone, Debug, Record)]
pub struct Gaussian3dTrainerRecord<B: Backend> {
    /// Iteration.
    pub iteration: u64,
    /// Learning rate for colors SH.
    pub learning_rate_colors_sh: LearningRateRecord,
    /// Learning rate for opacities.
    pub learning_rate_opacities: LearningRateRecord,
    /// Learning rate for positions.
    pub learning_rate_positions: LearningRateRecord,
    /// Learning rate for rotations.
    pub learning_rate_rotations: LearningRateRecord,
    /// Learning rate for scalings.
    pub learning_rate_scalings: LearningRateRecord,
    /// Optimizer for colors SH.
    pub optimizer_colors_sh: AdamRecord<B, 2>,
    /// Optimizer for opacities.
    pub optimizer_opacities: AdamRecord<B, 2>,
    /// Optimizer for positions.
    pub optimizer_positions: AdamRecord<B, 2>,
    /// Optimizer for rotations.
    pub optimizer_rotations: AdamRecord<B, 2>,
    /// Optimizer for scalings.
    pub optimizer_scalings: AdamRecord<B, 2>,
    /// Renderer options.
    pub options_renderer: Gaussian3dRenderOptions,
    /// Refiner.
    pub refiner: RefinerRecord<B>,
}

impl<B: Backend> Gaussian3dTrainer<Autodiff<B>>
where
    Gaussian3dScene<Autodiff<B>>: Gaussian3dRenderer<B>,
{
    /// Train the 3DGS scene.
    pub fn train(
        &mut self,
        scene: &mut Gaussian3dScene<Autodiff<B>>,
        camera: &sparse_view::Camera,
    ) -> Result<&mut Self, Error> {
        self.iteration += 1;

        #[cfg(all(debug_assertions, not(test)))]
        log::debug!(
            target: "gausplat::trainer::gaussian_3d::train",
            "iteration ({})",
            self.iteration,
        );

        let output = scene.render(&camera.view, &self.options_renderer)?;

        let colors_rgb_2d_target = camera
            .image
            .decode_rgb_tensor(&output.colors_rgb_2d.device())?
            .set_require_grad(false);

        let loss = self.get_loss_colors_rgb_2d(
            output.colors_rgb_2d.to_owned(),
            colors_rgb_2d_target.to_owned(),
        );

        let grads = &mut loss.backward();

        Ok(self.optimize(scene, grads).refine(scene, grads, output))
    }
}

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
    /// Get the loss for colors RGB (Rendered 2D Image).
    pub fn get_loss_colors_rgb_2d(
        &self,
        value: Tensor<AB, 3>,
        target: Tensor<AB, 3>,
    ) -> Tensor<AB, 1> {
        let mut loss = self
            .metric_optimization_coarse
            .evaluate(value.to_owned(), target.to_owned());

        if self.range_metric_optimization_fine.has(self.iteration) {
            loss = loss
                .add(
                    self.metric_optimization_fine
                        .evaluate(value.movedim(2, 0), target.movedim(2, 0)),
                )
                .div_scalar(2.0);
        }

        loss
    }

    /// Optimize the 3DGS scene.
    pub fn optimize(
        &mut self,
        scene: &mut Gaussian3dScene<AB>,
        grads: &mut AB::Gradients,
    ) -> &mut Self {
        #[cfg(all(debug_assertions, not(test)))]
        log::debug!(target: "gausplat::trainer::gaussian_3d::optimize", "start");

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

        self.learning_rate_colors_sh.update();
        self.learning_rate_opacities.update();
        self.learning_rate_positions.update();
        self.learning_rate_rotations.update();
        self.learning_rate_scalings.update();

        self
    }
}

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
    /// Transfer the trainer to the device.
    pub fn to_device(
        mut self,
        device: &AB::Device,
    ) -> Self {
        self.optimizer_colors_sh = self.optimizer_colors_sh.to_device(device);
        self.optimizer_opacities = self.optimizer_opacities.to_device(device);
        self.optimizer_positions = self.optimizer_positions.to_device(device);
        self.optimizer_rotations = self.optimizer_rotations.to_device(device);
        self.optimizer_scalings = self.optimizer_scalings.to_device(device);
        self.refiner = self.refiner.to_device(device);

        self
    }

    /// Load the record.
    pub fn load_record(
        &mut self,
        record: Gaussian3dTrainerRecord<AB::InnerBackend>,
    ) -> &mut Self {
        self.iteration = record.iteration;
        self.learning_rate_colors_sh
            .load_record(record.learning_rate_colors_sh);
        self.learning_rate_opacities
            .load_record(record.learning_rate_opacities);
        self.learning_rate_positions
            .load_record(record.learning_rate_positions);
        self.learning_rate_rotations
            .load_record(record.learning_rate_rotations);
        self.learning_rate_scalings
            .load_record(record.learning_rate_scalings);
        self.optimizer_colors_sh
            .load_record(record.optimizer_colors_sh);
        self.optimizer_opacities
            .load_record(record.optimizer_opacities);
        self.optimizer_positions
            .load_record(record.optimizer_positions);
        self.optimizer_rotations
            .load_record(record.optimizer_rotations);
        self.optimizer_scalings
            .load_record(record.optimizer_scalings);
        self.options_renderer = record.options_renderer;
        self.refiner.load_record(record.refiner);

        self
    }

    /// Unload the record.
    pub fn into_record(self) -> Gaussian3dTrainerRecord<AB::InnerBackend> {
        Gaussian3dTrainerRecord {
            iteration: self.iteration,
            learning_rate_colors_sh: self.learning_rate_colors_sh.into_record(),
            learning_rate_opacities: self.learning_rate_opacities.into_record(),
            learning_rate_positions: self.learning_rate_positions.into_record(),
            learning_rate_rotations: self.learning_rate_rotations.into_record(),
            learning_rate_scalings: self.learning_rate_scalings.into_record(),
            optimizer_colors_sh: self.optimizer_colors_sh.into_record(),
            optimizer_opacities: self.optimizer_opacities.into_record(),
            optimizer_positions: self.optimizer_positions.into_record(),
            optimizer_rotations: self.optimizer_rotations.into_record(),
            optimizer_scalings: self.optimizer_scalings.into_record(),
            options_renderer: self.options_renderer,
            refiner: self.refiner.into_record(),
        }
    }
}

impl<AB: AutodiffBackend> Default for Gaussian3dTrainer<AB> {
    #[inline]
    fn default() -> Self {
        Gaussian3dTrainerConfig::default().init(&Default::default())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn default() {
        use super::*;

        Adam::<Autodiff<Wgpu>, 2>::default();
        Gaussian3dTrainer::<Autodiff<Wgpu>>::default();
        Refiner::<Autodiff<Wgpu>>::default();

        let _ = *LearningRate::from(f64::default());
        let _ = LearningRate::default();
    }
}

pub mod config;
pub mod refine;

pub use crate::{
    error::Error,
    metric::{self, Metric},
    optimize::{Adam, AdamRecord, LearningRate, LearningRateRecord},
};
pub use burn::{
    config::Config,
    record::Record,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
};
pub use config::*;
pub use gausplat_importer::dataset::gaussian_3d::{Camera, Image};
pub use gausplat_renderer::scene::gaussian_3d::{
    backend::*,
    render::{Gaussian3dRenderer, Gaussian3dRendererOptions},
    Gaussian3dScene,
};
pub use refine::*;

use crate::function::*;
use gausplat_renderer::preset::spherical_harmonics::SH_DEGREE_MAX;

#[derive(Clone, Debug)]
pub struct Gaussian3dTrainer<AB: AutodiffBackend> {
    pub iteration: u64,
    pub learning_rate_colors_sh: LearningRate,
    pub learning_rate_opacities: LearningRate,
    pub learning_rate_positions: LearningRate,
    pub learning_rate_rotations: LearningRate,
    pub learning_rate_scalings: LearningRate,
    pub metric_optimization_coarse: metric::MeanAbsoluteError,
    pub metric_optimization_fine: metric::MeanStructuralDissimilarity<AB, 3>,
    pub optimizer_colors_sh: Adam<AB, 2>,
    pub optimizer_opacities: Adam<AB, 2>,
    pub optimizer_positions: Adam<AB, 2>,
    pub optimizer_rotations: Adam<AB, 2>,
    pub optimizer_scalings: Adam<AB, 2>,
    pub options_renderer: Gaussian3dRendererOptions,
    pub refiner: Refiner<AB::InnerBackend>,
}

#[derive(Clone, Debug, Record)]
pub struct Gaussian3dTrainerRecord<B: Backend> {
    pub iteration: u64,
    pub learning_rate_colors_sh: LearningRateRecord,
    pub learning_rate_opacities: LearningRateRecord,
    pub learning_rate_positions: LearningRateRecord,
    pub learning_rate_rotations: LearningRateRecord,
    pub learning_rate_scalings: LearningRateRecord,
    pub optimizer_colors_sh: AdamRecord<B, 2>,
    pub optimizer_opacities: AdamRecord<B, 2>,
    pub optimizer_positions: AdamRecord<B, 2>,
    pub optimizer_rotations: AdamRecord<B, 2>,
    pub optimizer_scalings: AdamRecord<B, 2>,
    pub options_renderer: Gaussian3dRendererOptions,
    pub refiner: RefinerRecord<B>,
}

impl<B: Backend> Gaussian3dTrainer<Autodiff<B>>
where
    Gaussian3dScene<Autodiff<B>>: Gaussian3dRenderer<B>,
{
    pub fn train(
        &mut self,
        scene: &mut Gaussian3dScene<Autodiff<B>>,
        camera: &Camera,
    ) -> Result<&mut Self, Error> {
        // TODO: Result<_, E>
        self.iteration += 1;

        #[cfg(debug_assertions)]
        log::debug!(
            target: "gausplat_trainer::train",
            "Gaussian3dTrainer::train > iteration {}",
            self.iteration,
        );

        let output = scene.render(&camera.view, &self.options_renderer);

        let colors_rgb_2d_target = get_tensor_from_image(
            &camera.image,
            &output.colors_rgb_2d.device(),
        )?;

        let grads = &mut self
            .get_loss_colors_rgb_2d(output.colors_rgb_2d, colors_rgb_2d_target)
            .backward();

        let positions_2d_grad_norm = output
            .positions_2d_grad_norm_ref
            .grad_remove(grads)
            .expect("A gradient should exist during training");

        Ok(self.optimize(scene, grads).refine(
            scene,
            positions_2d_grad_norm,
            output.radii,
        ))
    }
}

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
    pub fn get_loss_colors_rgb_2d(
        &self,
        value: Tensor<AB, 3>,
        target: Tensor<AB, 3>,
    ) -> Tensor<AB, 1> {
        const RANGE_STEP_OPTIMIZATION_FINE: u64 = 2;

        if self.iteration % RANGE_STEP_OPTIMIZATION_FINE == 0 {
            self.metric_optimization_coarse.evaluate(value, target)
        } else {
            self.metric_optimization_coarse
                .evaluate(value.to_owned(), target.to_owned())
                .add(
                    self.metric_optimization_fine
                        .evaluate(value.movedim(2, 0), target.movedim(2, 0)),
                )
                .div_scalar(2.0)
        }
    }

    pub fn optimize(
        &mut self,
        scene: &mut Gaussian3dScene<AB>,
        grads: &mut AB::Gradients,
    ) -> &mut Self {
        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::optimize");

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

        self
    }
}

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
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
        let _ = *(&mut LearningRate::default());
    }
}

pub mod config;
pub mod optimize;
pub mod range;
pub mod refine;

pub use crate::metric::MeanAbsoluteError;
pub use burn::{
    config::Config, module::AutodiffModule, tensor::backend::AutodiffBackend,
};
pub use config::*;
pub use gausplat_importer::dataset::gaussian_3d::{Camera, Image};
pub use gausplat_renderer::scene::gaussian_3d::{
    backend::Autodiff, render::Gaussian3dRendererOptions, Backend,
    Gaussian3dRenderer, Gaussian3dScene, Module, Tensor,
};
pub use optimize::*;
pub use range::*;
pub use refine::*;

use crate::function::*;
use gausplat_renderer::preset::spherical_harmonics::SH_DEGREE_MAX;
use std::ops::Add;

#[derive(Clone, Debug)]
pub struct Gaussian3dTrainer<AB: AutodiffBackend> {
    pub iteration: u64,
    pub learning_rate_colors_sh: LearningRate,
    pub learning_rate_opacities: LearningRate,
    pub learning_rate_positions: LearningRate,
    pub learning_rate_rotations: LearningRate,
    pub learning_rate_scalings: LearningRate,
    pub metric_optimization: MeanAbsoluteError,
    pub optimizer_colors_sh: Adam<AB, 2>,
    pub optimizer_opacities: Adam<AB, 2>,
    pub optimizer_positions: Adam<AB, 2>,
    pub optimizer_rotations: Adam<AB, 2>,
    pub optimizer_scalings: Adam<AB, 2>,
    pub options_renderer: Gaussian3dRendererOptions,
    pub refiner: Refiner<AB::InnerBackend>,
    pub scene: Gaussian3dScene<AB>,
}

// #[derive(Clone, Debug, Record)]
// pub struct Gaussian3dTrainerRecord<B: Backend> {
// }

impl<B: Backend> Gaussian3dTrainer<Autodiff<B>>
where
    Gaussian3dScene<Autodiff<B>>: Gaussian3dRenderer<B>,
{
    pub fn train(
        &mut self,
        camera: &Camera,
    ) -> &mut Self {
        self.iteration += 1;

        #[cfg(debug_assertions)]
        log::debug!(
            target: "gausplat_trainer::train",
            "Gaussian3dTrainer::train > {}",
            self.iteration,
        );

        let output = self.scene.render(&camera.view, &self.options_renderer);

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::train > output");

        let colors_rgb_2d = get_tensor_from_image(
            &camera.image,
            &output.colors_rgb_2d.device(),
        )
        .expect("The image error should be handled in `gausplat-importer`");

        let loss = self
            .metric_optimization
            .forward(output.colors_rgb_2d, colors_rgb_2d);

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::train > loss");

        let grads = &mut loss.backward();

        let positions_2d_grad_norm =
            output.positions_2d_grad_norm_ref.grad_remove(grads);
        if positions_2d_grad_norm.is_none() {
            #[cfg(debug_assertions)]
            log::debug!(
                target: "gausplat_trainer::train",
                "Gaussian3dTrainer::train > Exiting if no gradient",
            );

            return self;
        }

        let positions_2d_grad_norm =
            positions_2d_grad_norm.expect("Unreachable");

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::train > grads");

        self.optimize(grads)
            .refine(positions_2d_grad_norm, output.radii)
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
        self.scene = self.scene.to_device(device);

        self
    }

    // pub fn load_record(&mut self, record: Gaussian3dTrainerRecord) -> &mut Self {}
    // pub fn into_record(self) -> Gaussian3dTrainerRecord {}
    // pub fn to_record(&self) -> Gaussian3dTrainerRecord {}
}

impl<AB: AutodiffBackend> Default for Gaussian3dTrainer<AB> {
    #[inline]
    fn default() -> Self {
        Gaussian3dTrainerConfig::default()
            .init(&Default::default(), Default::default())
    }
}

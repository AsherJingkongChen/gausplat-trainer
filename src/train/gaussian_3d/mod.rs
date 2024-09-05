pub mod config;
pub mod optimize;
pub mod refine;

pub use crate::metric::MeanAbsoluteError;
pub use burn::{
    module::AutodiffModule, tensor::backend::AutodiffBackend, LearningRate,
};
pub use config::*;
pub use gausplat_importer::dataset::gaussian_3d::{Camera, Image};
pub use gausplat_renderer::scene::gaussian_3d::{
    backend::Autodiff, render::Gaussian3dRendererOptions, Backend,
    Gaussian3dRenderer, Gaussian3dScene, Module, Tensor,
};
pub use optimize::*;
pub use refine::*;

use crate::function::*;
use gausplat_renderer::preset::spherical_harmonics::SH_DEGREE_MAX;

#[derive(Clone, Debug)]
pub struct Gaussian3dTrainer<AB: AutodiffBackend> {
    pub config: Gaussian3dTrainerConfig,
    pub iteration: u64,
    pub learning_rate_decay_positions: LearningRate,
    pub metric_optimization: MeanAbsoluteError,
    pub optimizer_colors_sh: Adam<AB, 3>,
    pub optimizer_opacities: Adam<AB, 2>,
    pub optimizer_positions: Adam<AB, 2>,
    pub optimizer_rotations: Adam<AB, 2>,
    pub optimizer_scalings: Adam<AB, 2>,
    pub scene: Gaussian3dScene<AB>,
}

impl<B: Backend> Gaussian3dTrainer<Autodiff<B>>
where
    Gaussian3dScene<Autodiff<B>>: Gaussian3dRenderer<B>,
{
    pub fn train(
        &mut self,
        camera: &Camera,
    ) -> &mut Self {
        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::train");

        self.iterate();

        let output = self
            .scene
            .render(&camera.view, &self.config.options_renderer);
        let device = &output.colors_rgb_2d.device();

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::train > output");

        let colors_rgb_2d = get_tensor_from_image(&camera.image, device)
            .expect("The image error should be handled in `gausplat-importer`");

        let loss = self
            .metric_optimization
            .forward(output.colors_rgb_2d, colors_rgb_2d);

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::train > loss");

        let grads = &mut loss.backward();

        let positions_2d_grad_norm =
            output.positions_2d_grad_norm_ref.grad_remove(grads);

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::train > grads");

        self.optimize(grads)
            .refine(positions_2d_grad_norm, output.radii)
    }
}

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
    pub fn iterate(&mut self) -> &mut Self {
        // Increase the iteration

        self.iteration += 1;

        // Update the options

        let options = &mut self.config.options_renderer;
        if self.iteration % 1000 == 0 {
            options.colors_sh_degree_max =
                (options.colors_sh_degree_max + 1).min(SH_DEGREE_MAX);
        }

        self
    }

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
}

impl<AB: AutodiffBackend> Default for Gaussian3dTrainer<AB> {
    #[inline]
    fn default() -> Self {
        Gaussian3dTrainerConfig::default()
            .init(&Default::default(), Default::default())
    }
}

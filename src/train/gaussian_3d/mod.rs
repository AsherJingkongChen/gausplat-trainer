pub mod config;
pub mod optimize;
pub mod refine;

pub use crate::metric;
pub use burn::{tensor::backend::AutodiffBackend, LearningRate};
pub use config::*;
pub use gausplat_importer::dataset::gaussian_3d::{Camera, Image};
pub use gausplat_renderer::scene::gaussian_3d::{
    backend::Autodiff, render::Gaussian3dRendererOptions, Backend,
    Gaussian3dRenderer, Gaussian3dScene, Module, Tensor,
};
pub use optimize::*;
pub use refine::*;

use gausplat_importer::function::IntoTensorData;

#[derive(Clone, Debug)]
pub struct Gaussian3dTrainer<AB: AutodiffBackend> {
    pub config: Gaussian3dTrainerConfig,
    pub iteration: u64,
    pub learning_rate_decay_positions: LearningRate,
    pub metric_optimization: metric::MeanAbsoluteError,
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

        let output = self
            .scene
            .render(&camera.view, &self.config.options_renderer);

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::train > output");

        let colors_rgb_2d = Self::get_tensor_from_image(
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

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::train > grads");

        let positions_2d_grad_norm = self
            .scene
            .positions_2d_grad_norm_ref
            .grad_remove(grads)
            .expect("positions_2d_grad_norm should exist as a gradient");

        self.optimize(grads);

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::train > optimize");

        self.refine(positions_2d_grad_norm, output.radii);

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::train > refine");

        self.iteration += 1;

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
        self.scene = self.scene.to_device(device);

        self
    }

    pub fn get_tensor_from_image(
        image: &Image,
        device: &AB::Device,
    ) -> Result<Tensor<AB, 3>, gausplat_importer::error::Error> {
        Ok(
            Tensor::from_data(image.decode_rgb()?.into_tensor_data(), device)
                .div_scalar(255.0),
        )
    }
}

impl<AB: AutodiffBackend> Default for Gaussian3dTrainer<AB> {
    #[inline]
    fn default() -> Self {
        Gaussian3dTrainerConfig::default()
            .init(&Default::default(), Default::default())
    }
}

pub mod config;
pub mod optimize;

pub use crate::metric::MeanAbsoluteError;
pub use burn::{tensor::backend::AutodiffBackend, LearningRate};
pub use config::*;
pub use gausplat_importer::dataset::gaussian_3d::Camera;
pub use gausplat_renderer::scene::gaussian_3d::{
    backend::Autodiff, render::RenderOptions, Backend, Gaussian3dRenderer,
    Gaussian3dScene,
};
pub use optimize::*;
pub use rand::{rngs::StdRng, SeedableRng};

use burn::{
    module::Param,
    optim::{adaptor::OptimizerAdaptor, Adam},
    tensor::Tensor,
};
use gausplat_importer::function::IntoTensorData;
use rand::RngCore;
use std::fmt;

pub type AdamParamOptimizer<AB, const D: usize> = OptimizerAdaptor<
    Adam<<AB as AutodiffBackend>::InnerBackend>,
    Param<Tensor<AB, D>>,
    AB,
>;

pub type Cameras = indexmap::IndexMap<u32, Camera>;

#[derive(Clone)]
pub struct Gaussian3dTrainer<AB: AutodiffBackend> {
    pub cameras: Cameras,
    pub colors_sh_learning_rate: LearningRate,
    pub iteration: u64,
    pub metric_optimization: MeanAbsoluteError,
    pub opacities_learning_rate: LearningRate,
    pub param_optimizer_2d: AdamParamOptimizer<AB, 2>,
    pub param_optimizer_3d: AdamParamOptimizer<AB, 3>,
    pub positions_learning_rate: LearningRate,
    pub positions_learning_rate_decay: LearningRate,
    pub positions_learning_rate_end: LearningRate,
    pub random_generator: StdRng,
    pub render_options: RenderOptions,
    pub rotations_learning_rate: LearningRate,
    pub scalings_learning_rate: LearningRate,
    pub scene: Gaussian3dScene<AB>,
}

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
    /// Returns a random camera (`camera_id.is_none()`)
    /// or a specific camera (`camera_id.is_some()`)
    pub fn get_camera(
        &mut self,
        camera_id: Option<u32>,
    ) -> Option<&Camera> {
        match camera_id {
            Some(camera_id) => self.cameras.get(&camera_id),
            None => {
                let camera_index_random = self.random_generator.next_u64()
                    as usize
                    % self.cameras.len();
                self.cameras.get_index(camera_index_random).map(|p| p.1)
            },
        }
    }
}

impl<B: Backend> Gaussian3dTrainer<Autodiff<B>>
where
    Gaussian3dScene<Autodiff<B>>: Gaussian3dRenderer<B>,
{
    pub fn render(
        self,
        camera: &Camera,
    ) -> Self {
        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::render");

        let output = self.scene.render(&camera.view, self.render_options);
        let colors_rgb_2d_output = output.colors_rgb_2d;

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::render > output");

        let colors_rgb_2d_reference = Tensor::from_data(
            camera.image.decode_rgb().unwrap().into_tensor_data(),
            &colors_rgb_2d_output.device(),
        );

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::render > reference");

        let loss = self
            .metric_optimization
            .forward(colors_rgb_2d_output, colors_rgb_2d_reference);

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::render > loss");

        let grads = GradientsParams::from_grads(loss.backward(), &self.scene);

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::render > grads");

        self.optimize_params(grads)
    }
}

impl<AB: AutodiffBackend> fmt::Debug for Gaussian3dTrainer<AB> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("Gaussian3dTrainer")
            .field("cameras.len()", &self.cameras.len())
            .field("colors_sh_learning_rate", &self.colors_sh_learning_rate)
            .field("iteration", &self.iteration)
            .field("metric_optimization", &self.metric_optimization)
            .field("opacities_learning_rate", &self.opacities_learning_rate)
            .field("param_optimizer_2d", &format!("Adam<{}>", AB::name()))
            .field("param_optimizer_3d", &format!("Adam<{}>", AB::name()))
            .field("positions_learning_rate", &self.positions_learning_rate)
            .field(
                "positions_learning_rate_decay",
                &self.positions_learning_rate_decay,
            )
            .field(
                "positions_learning_rate_end",
                &self.positions_learning_rate_end,
            )
            .field("random_generator", &self.random_generator)
            .field("render_options", &self.render_options)
            .field("rotations_learning_rate", &self.rotations_learning_rate)
            .field("scalings_learning_rate", &self.scalings_learning_rate)
            .field("scene", &self.scene)
            .finish()
    }
}

impl<AB: AutodiffBackend> Default for Gaussian3dTrainer<AB> {
    fn default() -> Self {
        Gaussian3dTrainerConfig::default()
            .init(&Default::default(), Default::default())
    }
}

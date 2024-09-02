pub mod config;
pub mod optimize;
pub mod refine;

pub use crate::metric::MeanAbsoluteError;
pub use burn::{tensor::backend::AutodiffBackend, LearningRate};
pub use config::*;
pub use gausplat_importer::dataset::gaussian_3d::Camera;
pub use gausplat_renderer::scene::gaussian_3d::{
    backend::Autodiff, render::Gaussian3dRendererOptions, Backend,
    Gaussian3dRenderer, Gaussian3dScene,
};
pub use optimize::*;
pub use refine::*;

use burn::{
    module::Param,
    optim::{adaptor::OptimizerAdaptor, Adam},
    tensor::Tensor,
};
use gausplat_importer::function::IntoTensorData;
use std::fmt;

pub type AdamParamOptimizer<AB, const D: usize> = OptimizerAdaptor<
    Adam<<AB as AutodiffBackend>::InnerBackend>,
    Param<Tensor<AB, D>>,
    AB,
>;

#[derive(Clone)]
pub struct Gaussian3dTrainer<AB: AutodiffBackend> {
    pub config: Gaussian3dTrainerConfig,
    pub metric_optimization: MeanAbsoluteError,
    pub param_optimizer_2d: AdamParamOptimizer<AB, 2>,
    pub param_optimizer_3d: AdamParamOptimizer<AB, 3>,
    pub positions_learning_rate_decay: LearningRate,
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

        let output = self.scene.render(&camera.view, &self.config.render_options);
        let colors_rgb_2d_output = output.colors_rgb_2d;

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::train > output");

        let colors_rgb_2d_reference = Tensor::from_data(
            camera.image.decode_rgb().unwrap().into_tensor_data(),
            &colors_rgb_2d_output.device(),
        )
        .div_scalar(255.0);

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::train > reference");

        let loss = self
            .metric_optimization
            .forward(colors_rgb_2d_output, colors_rgb_2d_reference);

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::train > loss");

        let mut grads = loss.backward();
        let _positions_2d_grad_norm = output
            .positions_2d_grad_norm_ref
            .grad(&mut grads)
            .expect("positions_2d_grad_norm should exist as a gradient");

        let grads = GradientsParams::from_grads(grads, &self.scene);

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::train > grads");

        self.optimize(grads);

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::train > optimize");

        self.refine();

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::train > refine");

        self
    }
}

impl<AB: AutodiffBackend> fmt::Debug for Gaussian3dTrainer<AB> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("Gaussian3dTrainer")
            .field("config", &self.config)
            .field("metric_optimization", &self.metric_optimization)
            .field("param_optimizer_2d", &format!("Adam<{}>", AB::name()))
            .field("param_optimizer_3d", &format!("Adam<{}>", AB::name()))
            .field(
                "positions_learning_rate_decay",
                &self.positions_learning_rate_decay,
            )
            .field("scene", &self.scene)
            .finish()
    }
}

impl<AB: AutodiffBackend> Default for Gaussian3dTrainer<AB> {
    #[inline]
    fn default() -> Self {
        Gaussian3dTrainerConfig::default()
            .init(&Default::default(), Default::default())
    }
}

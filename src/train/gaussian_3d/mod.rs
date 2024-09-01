pub mod config;

pub use burn::{
    optim::{GradientsParams, Optimizer},
    tensor::backend::AutodiffBackend,
    LearningRate,
};
pub use config::*;
pub use gausplat_importer::dataset::gaussian_3d::Camera;
pub use gausplat_renderer::scene::gaussian_3d::{
    Gaussian3dRenderer, Gaussian3dScene,
};
pub use rand::{rngs::StdRng, SeedableRng};

use burn::{
    module::Param,
    optim::{adaptor::OptimizerAdaptor, Adam},
    tensor::Tensor,
};
use std::{fmt, ops::Mul};

pub type AdamParamOptimizer<AB, const D: usize> = OptimizerAdaptor<
    Adam<<AB as AutodiffBackend>::InnerBackend>,
    Param<Tensor<AB, D>>,
    AB,
>;
pub type AdamParamOptimizerRecord<AB, const D: usize> =
    <AdamParamOptimizer<AB, D> as Optimizer<Param<Tensor<AB, D>>, AB>>::Record;
pub type Cameras = indexmap::IndexMap<u32, Camera>;

#[derive(Clone)]
pub struct Gaussian3dTrainer<AB: AutodiffBackend> {
    pub cameras: Cameras,
    pub colors_sh_learning_rate: LearningRate,
    pub iteration: u64,
    pub opacities_learning_rate: LearningRate,
    pub param_optimizer_2d: AdamParamOptimizer<AB, 2>,
    pub param_optimizer_3d: AdamParamOptimizer<AB, 3>,
    pub positions_learning_rate: LearningRate,
    pub positions_learning_rate_decay: LearningRate,
    pub positions_learning_rate_end: LearningRate,
    pub rng: StdRng,
    pub rotations_learning_rate: LearningRate,
    pub scalings_learning_rate: LearningRate,
    pub scene: Gaussian3dScene<AB>,
}

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB>
where
    Gaussian3dScene<AB>: Gaussian3dRenderer<AB>,
{
}

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
    pub fn step_params(
        &mut self,
        module: Gaussian3dScene<AB>,
        grads: GradientsParams,
    ) -> Gaussian3dScene<AB> {
        self.step(0.0.into(), module, grads)
    }

    fn step_param<const D: usize>(
        optimizer: &mut impl Optimizer<Param<Tensor<AB, D>>, AB>,
        learning_rate: LearningRate,
        mut param: Param<Tensor<AB, D>>,
        grads: &mut GradientsParams,
    ) -> Param<Tensor<AB, D>> {
        let id = &param.id;
        if let Some(grad) = grads.remove::<AB, D>(id) {
            let mut grads = GradientsParams::new();
            grads.register(id.to_owned(), grad);
            param = optimizer.step(learning_rate, param, grads);
        }
        param
    }
}

impl<AB: AutodiffBackend> Optimizer<Gaussian3dScene<AB>, AB>
    for Gaussian3dTrainer<AB>
{
    type Record = (
        AdamParamOptimizerRecord<AB, 2>,
        AdamParamOptimizerRecord<AB, 3>,
    );

    fn step(
        &mut self,
        _: LearningRate,
        mut module: Gaussian3dScene<AB>,
        mut grads: GradientsParams,
    ) -> Gaussian3dScene<AB> {
        // Update the parameters

        module.colors_sh = Self::step_param(
            &mut self.param_optimizer_3d,
            self.colors_sh_learning_rate,
            module.colors_sh,
            &mut grads,
        );
        module.opacities = Self::step_param(
            &mut self.param_optimizer_2d,
            self.opacities_learning_rate,
            module.opacities,
            &mut grads,
        );
        module.positions = Self::step_param(
            &mut self.param_optimizer_2d,
            self.positions_learning_rate,
            module.positions,
            &mut grads,
        );
        module.rotations = Self::step_param(
            &mut self.param_optimizer_2d,
            self.rotations_learning_rate,
            module.rotations,
            &mut grads,
        );
        module.scalings = Self::step_param(
            &mut self.param_optimizer_2d,
            self.scalings_learning_rate,
            module.scalings,
            &mut grads,
        );

        #[cfg(debug_assertions)]
        {
            let record = self.to_record();
            let record_keys = (
                record.0.keys().collect::<Vec<_>>(),
                record.1.keys().collect::<Vec<_>>(),
            );
            log::debug!(
                target: "gausplat_trainer::updater",
                "Gaussian3dTrainer::step > record.keys {record_keys:#?}",
            );
        }

        // Update the learning rates

        self.positions_learning_rate = self
            .positions_learning_rate
            .mul(self.positions_learning_rate_decay)
            .max(self.positions_learning_rate_end);

        module
    }

    fn to_record(&self) -> Self::Record {
        (
            self.param_optimizer_2d.to_record(),
            self.param_optimizer_3d.to_record(),
        )
    }

    fn load_record(
        mut self,
        record: Self::Record,
    ) -> Self {
        self.param_optimizer_2d = self.param_optimizer_2d.load_record(record.0);
        self.param_optimizer_3d = self.param_optimizer_3d.load_record(record.1);
        self
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
            .field("rng", &self.rng)
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

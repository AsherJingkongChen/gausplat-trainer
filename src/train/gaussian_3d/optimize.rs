pub use super::*;

pub use burn::optim::{GradientsParams, Optimizer};

use burn::{module::Param, tensor::Tensor};
use std::ops::Mul;

pub type AdamParamOptimizerRecord<AB, const D: usize> =
    <AdamParamOptimizer<AB, D> as Optimizer<Param<Tensor<AB, D>>, AB>>::Record;

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
    pub fn optimize_params(
        &mut self,
        scene: Gaussian3dScene<AB>,
        grads: GradientsParams,
    ) -> Gaussian3dScene<AB> {
        self.step(0.0.into(), scene, grads)
    }

    fn optimize_param<const D: usize>(
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
        mut scene: Gaussian3dScene<AB>,
        mut grads: GradientsParams,
    ) -> Gaussian3dScene<AB> {
        // Optimize the parameters

        scene.colors_sh = Self::optimize_param(
            &mut self.param_optimizer_3d,
            self.colors_sh_learning_rate,
            scene.colors_sh,
            &mut grads,
        );
        scene.opacities = Self::optimize_param(
            &mut self.param_optimizer_2d,
            self.opacities_learning_rate,
            scene.opacities,
            &mut grads,
        );
        scene.positions = Self::optimize_param(
            &mut self.param_optimizer_2d,
            self.positions_learning_rate,
            scene.positions,
            &mut grads,
        );
        scene.rotations = Self::optimize_param(
            &mut self.param_optimizer_2d,
            self.rotations_learning_rate,
            scene.rotations,
            &mut grads,
        );
        scene.scalings = Self::optimize_param(
            &mut self.param_optimizer_2d,
            self.scalings_learning_rate,
            scene.scalings,
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

        // Scheduling the learning rates

        self.positions_learning_rate = self
            .positions_learning_rate
            .mul(self.positions_learning_rate_decay)
            .max(self.positions_learning_rate_end);

        scene
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

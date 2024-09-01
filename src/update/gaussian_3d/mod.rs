pub mod config;

pub use burn::{
    optim::{GradientsParams, Optimizer},
    tensor::backend::AutodiffBackend,
};
pub use config::*;
pub use gausplat_renderer::scene::gaussian_3d::Gaussian3dScene;

use burn::{
    module::Param,
    optim::{adaptor::OptimizerAdaptor, Adam},
    tensor::Tensor,
};
use std::fmt;

pub type AdamParamUpdater<AB, const D: usize> = OptimizerAdaptor<
    Adam<<AB as AutodiffBackend>::InnerBackend>,
    Param<Tensor<AB, D>>,
    AB,
>;
pub type AdamParamUpdaterRecord<AB, const D: usize> =
    <AdamParamUpdater<AB, D> as Optimizer<Param<Tensor<AB, D>>, AB>>::Record;

#[derive(Clone)]
pub struct Gaussian3dUpdater<AB: AutodiffBackend> {
    pub colors_sh_updating_rate: UpdatingRate,
    pub opacities_updating_rate: UpdatingRate,
    pub positions_updating_rate: UpdatingRate,
    pub positions_updating_rate_decay: UpdatingRate,
    pub rotations_updating_rate: UpdatingRate,
    pub scalings_updating_rate: UpdatingRate,
    param_updater_2d: AdamParamUpdater<AB, 2>,
    param_updater_3d: AdamParamUpdater<AB, 3>,
}

impl<AB: AutodiffBackend> Gaussian3dUpdater<AB> {
    pub fn update(
        &mut self,
        module: Gaussian3dScene<AB>,
        grads: GradientsParams,
    ) -> Gaussian3dScene<AB> {
        self.step(0.0.into(), module, grads)
    }

    fn step_param<const D: usize>(
        optimizer: &mut impl Optimizer<Param<Tensor<AB, D>>, AB>,
        updating_rate: UpdatingRate,
        mut param: Param<Tensor<AB, D>>,
        grads: &mut GradientsParams,
    ) -> Param<Tensor<AB, D>> {
        let id = &param.id;
        if let Some(grad) = grads.remove::<AB, D>(id) {
            let mut grads = GradientsParams::new();
            grads.register(id.to_owned(), grad);
            param = optimizer.step(updating_rate, param, grads);
        }
        param
    }
}

impl<AB: AutodiffBackend> Optimizer<Gaussian3dScene<AB>, AB>
    for Gaussian3dUpdater<AB>
{
    type Record =
        (AdamParamUpdaterRecord<AB, 2>, AdamParamUpdaterRecord<AB, 3>);

    fn step(
        &mut self,
        _: UpdatingRate,
        mut module: Gaussian3dScene<AB>,
        mut grads: GradientsParams,
    ) -> Gaussian3dScene<AB> {
        // Update the parameters

        module.colors_sh = Self::step_param(
            &mut self.param_updater_3d,
            self.colors_sh_updating_rate,
            module.colors_sh,
            &mut grads,
        );
        module.opacities = Self::step_param(
            &mut self.param_updater_2d,
            self.opacities_updating_rate,
            module.opacities,
            &mut grads,
        );
        module.positions = Self::step_param(
            &mut self.param_updater_2d,
            self.positions_updating_rate,
            module.positions,
            &mut grads,
        );
        module.rotations = Self::step_param(
            &mut self.param_updater_2d,
            self.rotations_updating_rate,
            module.rotations,
            &mut grads,
        );
        module.scalings = Self::step_param(
            &mut self.param_updater_2d,
            self.scalings_updating_rate,
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
                "Gaussian3dUpdater::step > record.keys {record_keys:#?}",
            );
        }

        // Update the learning rates

        self.positions_updating_rate *= self.positions_updating_rate_decay;

        module
    }

    fn to_record(&self) -> Self::Record {
        (
            self.param_updater_2d.to_record(),
            self.param_updater_3d.to_record(),
        )
    }

    fn load_record(
        mut self,
        record: Self::Record,
    ) -> Self {
        self.param_updater_2d = self.param_updater_2d.load_record(record.0);
        self.param_updater_3d = self.param_updater_3d.load_record(record.1);
        self
    }
}

impl<AB: AutodiffBackend> fmt::Debug for Gaussian3dUpdater<AB> {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        f.debug_struct("Gaussian3dUpdater")
            .field("updater", &format!("Adam<{}>", AB::name()))
            .field("colors_sh_updating_rate", &self.colors_sh_updating_rate)
            .field("opacities_updating_rate", &self.opacities_updating_rate)
            .field("positions_updating_rate", &self.positions_updating_rate)
            .field(
                "positions_updating_rate_decay",
                &self.positions_updating_rate_decay,
            )
            .field("rotations_updating_rate", &self.rotations_updating_rate)
            .field("scalings_updating_rate", &self.scalings_updating_rate)
            .finish()
    }
}

impl<AB: AutodiffBackend> Default for Gaussian3dUpdater<AB> {
    fn default() -> Self {
        Gaussian3dUpdaterConfig::default().init()
    }
}

pub mod config;

pub use burn::{optim::GradientsParams, tensor::backend::AutodiffBackend};
pub use config::*;
pub use gausplat_renderer::scene::gaussian_3d::Gaussian3dScene;

use burn::{
    module::Param,
    optim::{adaptor::OptimizerAdaptor, Adam, Optimizer},
    tensor::Tensor,
};
use std::fmt;

pub type AdamParamUpdater<AB, const D: usize> = OptimizerAdaptor<
    Adam<<AB as AutodiffBackend>::InnerBackend>,
    Param<Tensor<AB, D>>,
    AB,
>;

#[derive(Clone)]
pub struct Gaussian3dUpdater<AB: AutodiffBackend> {
    colors_sh_updater: AdamParamUpdater<AB, 3>,
    colors_sh_updating_rate: UpdatingRate,
    opacities_updater: AdamParamUpdater<AB, 2>,
    opacities_updating_rate: UpdatingRate,
    positions_updater: AdamParamUpdater<AB, 2>,
    positions_updating_rate: UpdatingRate,
    positions_updating_rate_decay: UpdatingRate,
    rotations_updater: AdamParamUpdater<AB, 2>,
    rotations_updating_rate: UpdatingRate,
    scalings_updater: AdamParamUpdater<AB, 2>,
    scalings_updating_rate: UpdatingRate,
}

impl<AB: AutodiffBackend> Gaussian3dUpdater<AB> {
    pub fn step(
        &mut self,
        mut module: Gaussian3dScene<AB>,
        mut grads: GradientsParams,
    ) -> Gaussian3dScene<AB> {
        // Update the parameters

        module.colors_sh = Self::step_param(
            &mut self.colors_sh_updater,
            self.colors_sh_updating_rate,
            module.colors_sh,
            &mut grads,
        );
        module.opacities = Self::step_param(
            &mut self.opacities_updater,
            self.opacities_updating_rate,
            module.opacities,
            &mut grads,
        );
        module.positions = Self::step_param(
            &mut self.positions_updater,
            self.positions_updating_rate,
            module.positions,
            &mut grads,
        );
        module.rotations = Self::step_param(
            &mut self.rotations_updater,
            self.rotations_updating_rate,
            module.rotations,
            &mut grads,
        );
        module.scalings = Self::step_param(
            &mut self.scalings_updater,
            self.scalings_updating_rate,
            module.scalings,
            &mut grads,
        );

        // Update the learning rates

        self.positions_updating_rate *= self.positions_updating_rate_decay;

        module
    }

    fn step_param<O: Optimizer<Param<Tensor<AB, D>>, AB>, const D: usize>(
        optimizer: &mut O,
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

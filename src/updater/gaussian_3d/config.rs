pub use super::*;
pub use burn::{config::Config, LearningRate as UpdatingRate};

use burn::optim::AdamConfig;

#[derive(Config, Copy, Debug)]
pub struct Gaussian3dUpdaterConfig {
    #[config(default = "7000")]
    pub iteration_count: u64,

    #[config(default = "2.5e-3")]
    pub colors_sh_updating_rate: UpdatingRate,

    #[config(default = "2.5e-2")]
    pub opacities_updating_rate: UpdatingRate,

    #[config(default = "1.6e-4")]
    pub positions_updating_rate_start: UpdatingRate,

    #[config(default = "1.6e-6")]
    pub positions_updating_rate_end: UpdatingRate,

    #[config(default = "1e-3")]
    pub rotations_updating_rate: UpdatingRate,

    #[config(default = "5e-3")]
    pub scalings_updating_rate: UpdatingRate,
}

impl Gaussian3dUpdaterConfig {
    pub fn init<AB: AutodiffBackend>(&self) -> Gaussian3dUpdater<AB> {
        let updater = AdamConfig::new().with_epsilon(1e-15);
        let positions_updating_rate_decay = Self::updating_rate_decay(
            self.positions_updating_rate_start,
            self.positions_updating_rate_end,
            self.iteration_count,
        );

        Gaussian3dUpdater {
            param_updater_2d: updater.init(),
            param_updater_3d: updater.init(),
            colors_sh_updating_rate: self.colors_sh_updating_rate,
            opacities_updating_rate: self.opacities_updating_rate,
            positions_updating_rate: self.positions_updating_rate_start,
            positions_updating_rate_decay,
            rotations_updating_rate: self.rotations_updating_rate,
            scalings_updating_rate: self.scalings_updating_rate,
        }
    }

    #[inline]
    fn updating_rate_decay(
        updating_rate_start: UpdatingRate,
        updating_rate_end: UpdatingRate,
        iteration_count: u64,
    ) -> UpdatingRate {
        use std::ops::Div;

        updating_rate_end
            .div(updating_rate_start)
            .powf((iteration_count as UpdatingRate).recip())
    }
}

impl Default for Gaussian3dUpdaterConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn updating_rate_decay() {
        use super::*;

        let config =
            Gaussian3dUpdaterConfig::default().with_iteration_count(7000);
        let decay = Gaussian3dUpdaterConfig::updating_rate_decay(
            config.positions_updating_rate_start,
            config.positions_updating_rate_end,
            config.iteration_count,
        );
        assert_eq!(decay, 0.9993423349014151);

        let config =
            Gaussian3dUpdaterConfig::default().with_iteration_count(30000);
        let decay = Gaussian3dUpdaterConfig::updating_rate_decay(
            config.positions_updating_rate_start,
            config.positions_updating_rate_end,
            config.iteration_count,
        );
        assert_eq!(decay, 0.9998465061085267);
    }
}

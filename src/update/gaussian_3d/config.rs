pub use super::*;
pub use burn::{config::Config, LearningRate as UpdatingRate};

use burn::optim::AdamConfig;
use std::ops::Div;

#[derive(Config, Copy, Debug)]
pub struct Gaussian3dUpdaterConfig {
    #[config(default = "2.5e-3")]
    pub colors_sh_updating_rate: UpdatingRate,

    #[config(default = "2.5e-2")]
    pub opacities_updating_rate: UpdatingRate,

    #[config(default = "30000")]
    pub positions_updating_count: u64,

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
            self.positions_updating_count,
            self.positions_updating_rate_start,
            self.positions_updating_rate_end,
        );

        Gaussian3dUpdater {
            colors_sh_updating_rate: self.colors_sh_updating_rate,
            opacities_updating_rate: self.opacities_updating_rate,
            positions_updating_rate: self.positions_updating_rate_start,
            positions_updating_rate_decay,
            positions_updating_rate_end: self.positions_updating_rate_end,
            rotations_updating_rate: self.rotations_updating_rate,
            scalings_updating_rate: self.scalings_updating_rate,
            param_updater_2d: updater.init(),
            param_updater_3d: updater.init(),
        }
    }

    #[inline]
    fn updating_rate_decay(
        updating_count: u64,
        updating_rate_start: UpdatingRate,
        updating_rate_end: UpdatingRate,
    ) -> UpdatingRate {
        updating_rate_end
            .div(updating_rate_start)
            .powf((updating_count as UpdatingRate).recip())
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

        let config = Gaussian3dUpdaterConfig::default()
            .with_positions_updating_count(7000);
        let decay = Gaussian3dUpdaterConfig::updating_rate_decay(
            config.positions_updating_count,
            config.positions_updating_rate_start,
            config.positions_updating_rate_end,
        );
        assert_eq!(decay, 0.9993423349014151);

        let config = Gaussian3dUpdaterConfig::default()
            .with_positions_updating_count(30000);
        let decay = Gaussian3dUpdaterConfig::updating_rate_decay(
            config.positions_updating_count,
            config.positions_updating_rate_start,
            config.positions_updating_rate_end,
        );
        assert_eq!(decay, 0.9998465061085267);
    }
}

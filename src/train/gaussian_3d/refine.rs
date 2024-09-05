pub use super::*;
pub use burn::{config::Config, tensor::Int};

#[derive(Config, Debug)]
pub struct RefinementConfig {}

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
    pub fn refine(
        &mut self,
        positions_2d_grad_norm: Option<Tensor<AB::InnerBackend, 1>>,
        radii: Tensor<AB::InnerBackend, 1, Int>,
    ) -> &mut Self {
        #[cfg(debug_assertions)]
        log::debug!(
            target: "gausplat_trainer::train",
            "Gaussian3dTrainer::refine > positions_2d_grad_norm {:#?}",
            positions_2d_grad_norm
                .expect("positions_2d_grad_norm should exists as a gradient")
                .mean_dim(0).to_data().to_vec::<f32>().unwrap()
        );

        #[cfg(debug_assertions)]
        log::debug!(target: "gausplat_trainer::train", "Gaussian3dTrainer::refine");

        self
    }
}

impl Default for RefinementConfig {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

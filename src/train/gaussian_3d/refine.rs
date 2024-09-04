pub use super::*;
pub use burn::tensor::Int;

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
    pub fn refine(
        &mut self,
        positions_2d_grad_norm: Tensor<AB::InnerBackend, 1>,
        radii: Tensor<AB::InnerBackend, 1, Int>,
    ) -> &mut Self {
        #[cfg(debug_assertions)]
        log::debug!(
            target: "gausplat_trainer::train",
            "Gaussian3dTrainer::refine > positions_2d_grad_norm {:#?}",
            positions_2d_grad_norm.mean_dim(0).to_data().to_vec::<f32>().unwrap()
        );

        self
    }
}

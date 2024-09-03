pub use super::*;
pub use burn::tensor::Int;

impl<AB: AutodiffBackend> Gaussian3dTrainer<AB> {
    pub fn refine(
        &mut self,
        positions_2d_grad_norm: Tensor<AB::InnerBackend, 1>,
        radii: Tensor<AB::InnerBackend, 1, Int>,
    ) -> &mut Self {
        self
    }
}

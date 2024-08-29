pub use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

/// Computing the mean absolute error (MAE) between the inputs:
///
/// `mean(|input_0 - input_1|)`
///
#[derive(Clone, Copy, Debug, Default)]
pub struct MeanAbsoluteError;

impl MeanAbsoluteError {
    /// Computing the mean absolute error (MAE) between the inputs:
    ///
    /// `mean(|input_0 - input_1|)`
    ///
    pub fn forward<B: Backend, const D: usize>(
        &self,
        input_0: Tensor<B, D>,
        input_1: Tensor<B, D>,
    ) -> Tensor<B, 1> {
        input_0.sub(input_1).abs().mean()
    }
}

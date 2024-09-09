pub use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

/// Computing the mean square error (MSE) between the inputs:
///
/// `mean((input_0 - input_1) ^ 2)`
///
#[derive(Clone, Copy, Debug, Default)]
pub struct MeanSquareError;

impl MeanSquareError {
    /// Computing the mean square error (MSE) between the inputs:
    ///
    /// `mean((input_0 - input_1) ^ 2)`
    ///
    pub fn forward<B: Backend, const D: usize>(
        &self,
        input_0: Tensor<B, D>,
        input_1: Tensor<B, D>,
    ) -> Tensor<B, 1> {
        input_0.sub(input_1).powi_scalar(2).mean()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn forward() {
        use super::*;
        use burn::backend::NdArray;

        let device = Default::default();
        let metric = MeanSquareError;

        let input_0 = Tensor::zeros([1, 3, 256, 256], &device);
        let input_1 = Tensor::zeros([1, 3, 256, 256], &device);
        let score = metric
            .forward::<NdArray<f32>, 4>(input_0, input_1)
            .into_scalar();
        assert_eq!(score, 0.0);

        let input_0 = Tensor::ones([1, 3, 256, 256], &device);
        let input_1 = Tensor::ones([1, 3, 256, 256], &device);
        let score = metric
            .forward::<NdArray<f32>, 4>(input_0, input_1)
            .into_scalar();
        assert_eq!(score, 0.0);

        let input_0 = Tensor::zeros([1, 3, 256, 256], &device);
        let input_1 = Tensor::ones([1, 3, 256, 256], &device);
        let score = metric
            .forward::<NdArray<f32>, 4>(input_0, input_1)
            .into_scalar();
        assert_eq!(score, 1.0);
    }
}

pub use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

/// Computing the mean absolute error (MAE) between the inputs:
///
/// `mean(abs(input_0 - input_1))`
///
#[derive(Clone, Copy, Debug, Default)]
pub struct MeanAbsoluteError;

impl MeanAbsoluteError {
    #[inline]
    pub fn init() -> Self {
        Self
    }

    /// Computing the mean absolute error (MAE) between the inputs:
    ///
    /// `mean(abs(input_0 - input_1))`
    ///
    pub fn forward<B: Backend, const D: usize>(
        &self,
        input_0: Tensor<B, D>,
        input_1: Tensor<B, D>,
    ) -> Tensor<B, 1> {
        input_0.sub(input_1).abs().mean()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn forward() {
        use super::*;
        use burn::backend::NdArray;

        let device = Default::default();
        let metric = MeanAbsoluteError;

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

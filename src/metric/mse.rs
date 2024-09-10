pub use super::*;

/// Computing the mean square error (MSE) between the inputs:
///
/// `mean((input_0 - input_1) ^ 2)`
///
#[derive(Clone, Copy, Debug, Default)]
pub struct MeanSquareError;

impl MeanSquareError {
    #[inline]
    pub fn init() -> Self {
        Self
    }
}

impl<B: Backend> Metric<B> for MeanSquareError {
    /// ## Returns
    ///
    /// The mean square error (MSE) with shape `[1]`.
    #[inline]
    fn evaluate<const D: usize>(
        &self,
        value: Tensor<B, D>,
        target: Tensor<B, D>,
    ) -> Tensor<B, 1> {
        value.sub(target).powf_scalar(2.0).mean()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn evaluate() {
        use super::*;
        use burn::backend::NdArray;

        let device = Default::default();
        let metric = MeanSquareError::init();

        let input_0 = Tensor::<NdArray, 4>::zeros([1, 3, 256, 256], &device);
        let input_1 = Tensor::<NdArray, 4>::zeros([1, 3, 256, 256], &device);
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert_eq!(score, 0.0);

        let input_0 = Tensor::<NdArray, 4>::ones([1, 3, 256, 256], &device);
        let input_1 = Tensor::<NdArray, 4>::ones([1, 3, 256, 256], &device);
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert_eq!(score, 0.0);

        let input_0 = Tensor::<NdArray, 4>::zeros([1, 3, 256, 256], &device);
        let input_1 = Tensor::<NdArray, 4>::ones([1, 3, 256, 256], &device);
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert_eq!(score, 1.0);
    }
}

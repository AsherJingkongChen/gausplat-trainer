//! Mean Absolute Error (MAE) metric.

pub use super::*;

/// Computing the mean absolute error (MAE) between the inputs:
///
/// `mean(abs(input_0 - input_1))`
///
#[derive(Clone, Copy, Debug, Default)]
pub struct MeanAbsoluteError;

impl MeanAbsoluteError {
    /// Initialize the metric.
    #[inline]
    pub const fn init() -> Self {
        Self
    }
}

impl<B: Backend> Metric<B> for MeanAbsoluteError {
    /// ## Returns
    ///
    /// The mean absolute error (MAE) with shape `[1]`.
    #[inline]
    fn evaluate<const D: usize>(
        &self,
        value: Tensor<B, D>,
        target: Tensor<B, D>,
    ) -> Tensor<B, 1> {
        value.sub(target).abs().mean()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn evaluate() {
        use super::*;
        use burn::backend::NdArray;

        let device = Default::default();
        let metric = MeanAbsoluteError::init();

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

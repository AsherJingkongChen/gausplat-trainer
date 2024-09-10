pub use super::*;

/// Computing the mean of structural dissimilarity index (MDSSIM) between the inputs:
///
/// `(1 - MSSIM) / 2`
///
/// ## Details
///
/// It relies on [`MeanStructuralSimilarity`].
#[derive(Clone, Debug)]
pub struct MeanStructuralDissimilarity<B: Backend, const C: usize> {
    pub inner: MeanStructuralSimilarity<B, C>,
}

impl<B: Backend, const C: usize> MeanStructuralDissimilarity<B, C> {
    pub fn init(device: &B::Device) -> Self {
        Self {
            inner: MeanStructuralSimilarity::init(device),
        }
    }
}

impl<B: Backend, const C: usize> Metric<B>
    for MeanStructuralDissimilarity<B, C>
{
    /// ## Arguments
    ///
    /// * `value`: The input tensor with shape `[N?, C?, H, W]`.
    /// * `target`: The target tensor with shape `[N?, C?, H, W]`.
    ///
    /// ## Returns
    ///
    /// The mean of structural dissimilarity index (MDSSIM) with shape `[1]`.
    ///
    /// ## Details
    ///
    /// * The argument value should range from `0.0` to `1.0`
    /// * The result value ranges from `0.0` to `1.0`
    fn evaluate<const D: usize>(
        &self,
        value: Tensor<B, D>,
        target: Tensor<B, D>,
    ) -> Tensor<B, 1> {
        self.inner
            .evaluate(value, target)
            .neg()
            .add_scalar(1.0)
            .div_scalar(2.0)
    }
}

impl<B: Backend, const C: usize> Default for MeanStructuralDissimilarity<B, C> {
    #[inline]
    fn default() -> Self {
        Self::init(&Default::default())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn evaluate() {
        use super::*;
        use burn::{backend::NdArray, tensor::Distribution};

        let device = Default::default();
        let metric =
            MeanStructuralDissimilarity::<NdArray<f32>, 3>::init(&device);

        let input_0 = Tensor::zeros([1, 3, 256, 256], &device);
        let input_1 = Tensor::zeros([1, 3, 256, 256], &device);
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert_eq!(score, 0.0);

        let input_0 = Tensor::ones([1, 3, 256, 256], &device);
        let input_1 = Tensor::ones([1, 3, 256, 256], &device);
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert_eq!(score, 0.0);

        let input_0 = Tensor::zeros([1, 3, 256, 256], &device);
        let input_1 = Tensor::ones([1, 3, 256, 256], &device);
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert!(score < 0.5001 && score > 0.4999, "score: {:?}", score);

        let input_0 = Tensor::random(
            [1, 3, 256, 256],
            Distribution::Uniform(0.01, 0.99),
            &device,
        );
        let input_1 = input_0.to_owned().neg().add_scalar(1.0);
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert!(score > 0.5, "score: {:?}", score);
    }
}
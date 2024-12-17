//! Mean of structural dissimilarity index (MD-SSIM) metric.

pub use super::*;

/// Computing the mean of structural dissimilarity index (MD-SSIM) between the inputs:
///
/// `1 - MSSIM`
///
/// ## Details
///
/// It relies on [`MSSIM`](MeanStructuralSimilarity).
#[derive(Clone, Debug)]
pub struct MeanStructuralDissimilarity<B: Backend, const C: usize> {
    /// Inner metric.
    pub inner: MeanStructuralSimilarity<B, C>,
}

impl<B: Backend, const C: usize> MeanStructuralDissimilarity<B, C> {
    /// Initialize the metric.
    pub fn init(device: &B::Device) -> Self {
        Self {
            inner: MeanStructuralSimilarity::init(device),
        }
    }
}

impl<B: Backend, const C: usize> Metric<B> for MeanStructuralDissimilarity<B, C> {
    /// ## Arguments
    ///
    /// * `value` - The input tensor with shape `[N?, C?, H, W]`.
    /// * `target` - The target tensor with shape `[N?, C?, H, W]`.
    ///
    /// ## Returns
    ///
    /// The mean of structural dissimilarity index (MD-SSIM) with shape `[1]`.
    ///
    /// ## Details
    ///
    /// * The argument value should range from `0.0` to `1.0`
    /// * The result value ranges from `0.0` to `2.0`
    fn evaluate<const D: usize>(
        &self,
        value: Tensor<B, D>,
        target: Tensor<B, D>,
    ) -> Tensor<B, 1> {
        self.inner.evaluate(value, target).neg().add_scalar(1.0)
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
        let metric = MeanStructuralDissimilarity::<NdArray<f32>, 3>::init(&device);

        let input_0 = Tensor::<NdArray<f32>, 4>::zeros([1, 3, 36, 36], &device);
        let input_1 = Tensor::zeros([1, 3, 36, 36], &device);
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert_eq!(score, 0.0);

        let input_0 = Tensor::<NdArray<f32>, 4>::ones([1, 3, 36, 36], &device);
        let input_1 = Tensor::ones([1, 3, 36, 36], &device);
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert_eq!(score, 0.0);

        let input_0 = Tensor::<NdArray<f32>, 4>::zeros([1, 3, 36, 36], &device);
        let input_1 = Tensor::ones([1, 3, 36, 36], &device);
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert!(score <= 1.0001 && score >= 0.9999, "score: {:?}", score);

        let input_0 = Tensor::<NdArray<f32>, 4>::random(
            [1, 3, 36, 36],
            Distribution::Uniform(0.01, 0.99),
            &device,
        );
        let input_1 = input_0.to_owned().neg().add_scalar(1.0);
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert!(score > 0.5, "score: {:?}", score);
    }

    #[test]
    fn default() {
        use super::*;
        use burn::backend::NdArray;

        MeanStructuralDissimilarity::<NdArray<f32>, 3>::default();
    }
}

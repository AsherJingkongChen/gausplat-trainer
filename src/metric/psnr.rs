//! Peak signal-to-noise ratio (PSNR) metric.

pub use super::*;

/// Computing the peak signal-to-noise ratio (PSNR) between the inputs:
///
/// `10 * log10(1 / MSE) = -10 / log(10) * log(MSE)`
///
/// ## Details
///
/// It relies on [`MSE`](MeanSquareError).
#[derive(Clone, Debug)]
pub struct Psnr<B: Backend> {
    /// Coefficient for PSNR.
    pub coefficient: Tensor<B, 1>,
    /// Inner metric.
    pub mse: MeanSquareError,
}

impl<B: Backend> Psnr<B> {
    /// Initialize the metric.
    pub fn init(device: &B::Device) -> Self {
        let ten = Tensor::<B, 1>::from_floats([10.0], device);
        let coefficient = ten.clone().neg().div(ten.log());
        let mse = MeanSquareError::init();
        Self { coefficient, mse }
    }
}

impl<B: Backend> Metric<B> for Psnr<B> {
    /// ## Returns
    ///
    /// The peak signal-to-noise ratio (PSNR) with shape `[1]`.
    #[inline]
    fn evaluate<const D: usize>(
        &self,
        value: Tensor<B, D>,
        target: Tensor<B, D>,
    ) -> Tensor<B, 1> {
        let mse = self.mse.evaluate(value, target);
        self.coefficient.to_owned().mul(mse.log())
    }
}

impl<B: Backend> Default for Psnr<B> {
    fn default() -> Self {
        Self::init(&Default::default())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn default() {
        use super::*;
        use burn::backend::NdArray;

        let target = -10.0 / 10.0_f32.ln();
        let output = Psnr::<NdArray>::default().coefficient.into_scalar();
        assert_eq!(output, target);
    }

    #[test]
    fn evaluate() {
        use super::*;
        use burn::backend::NdArray;

        let device = Default::default();
        let metric = Psnr::init(&device);

        let input_0 = Tensor::<NdArray, 4>::zeros([1, 3, 256, 256], &device);
        let input_1 = Tensor::<NdArray, 4>::zeros([1, 3, 256, 256], &device);
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert_eq!(score, f32::INFINITY);

        let input_0 = Tensor::<NdArray, 4>::ones([1, 3, 256, 256], &device);
        let input_1 = Tensor::<NdArray, 4>::ones([1, 3, 256, 256], &device);
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert_eq!(score, f32::INFINITY);

        let input_0 = Tensor::<NdArray, 4>::zeros([1, 3, 256, 256], &device);
        let input_1 = Tensor::<NdArray, 4>::ones([1, 3, 256, 256], &device);
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert_eq!(score, 0.0);

        let input_0 = Tensor::<NdArray, 2>::from_floats(
            [[0.0, 0.1, 0.2], [0.5, 0.4, 0.3]],
            &device,
        );
        let input_1 = Tensor::<NdArray, 2>::from_floats(
            [[0.5, 0.6, 0.7], [0.0, 0.9, 0.8]],
            &device,
        );
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert_eq!(score, 6.0206);

        let input_0 = Tensor::<NdArray, 2>::from_floats(
            [[0.0, 0.1, 0.2], [0.5, 0.4, 0.3]],
            &device,
        );
        let input_1 = Tensor::<NdArray, 2>::from_floats(
            [[0.0, 0.6, 0.7], [0.0, 0.4, 0.3]],
            &device,
        );
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert_eq!(score, 9.030899);
    }
}

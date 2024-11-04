pub use super::*;

use crate::error::Error;
use burn::{
    module::Param,
    nn::{conv, PaddingConfig2d},
    tensor::Int,
};

/// Computing the mean of structural similarity index (MSSIM) between the inputs
/// using the approaches described in the paper:
///
/// *Wang, J., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4), 600–612.*
/// https://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf
///
/// ## Details
///
/// * `self.filter.weight` - A normalized gaussian filter
///   with shape of `[C, 1, 11, 11]` and the standard deviation of `1.5`
#[derive(Clone, Debug)]
pub struct MeanStructuralSimilarity<B: Backend, const C: usize> {
    pub filter: conv::Conv2d<B>,
}

impl<B: Backend, const C: usize> MeanStructuralSimilarity<B, C> {
    pub fn init(device: &B::Device) -> Self {
        // G
        const WEIGHT_SIZE: usize = 11;
        // G / 2
        const WEIGHT_SIZE_HALF: usize = WEIGHT_SIZE >> 1;
        // σ
        const WEIGHT_STD: f64 = 1.5;
        // σ^2
        const WEIGHT_STD2: f64 = WEIGHT_STD * WEIGHT_STD;

        let mut filter = conv::Conv2dConfig::new([C; 2], [WEIGHT_SIZE; 2])
            .with_bias(false)
            .with_groups(C)
            .with_padding(PaddingConfig2d::Valid)
            .init(device);

        // [C, 1, G, G]
        filter.weight = Param::uninitialized(
            Default::default(),
            move |device, is_required_grad| {
                // G / 2
                let size_half = WEIGHT_SIZE_HALF as i64;
                // x = [-G / 2, G / 2]
                let x = Tensor::<B, 1, Int>::arange(
                    -size_half..size_half + 1,
                    device,
                );
                // -x^2[1, G]
                let x2_n = x.powi_scalar(2).neg().float().unsqueeze::<2>();
                // -y^2[G, 1]
                let y2_n = x2_n.to_owned().transpose();
                // -(x^2 + y^2)[G, G] = -x^2[1, G] + -y^2[G, 1]
                let x2_y2_n = x2_n + y2_n;
                // w[G, G] = exp(-(x^2 + y^2) / 2σ^2)[G, G]
                let w = x2_y2_n.div_scalar(2.0 * WEIGHT_STD2).exp();
                // w'[G, G] = w[G, G] / sum(w)[1, 1]
                let w_normalized = w.to_owned().div(w.sum().unsqueeze::<2>());

                // w'[C, 1, G, G]
                w_normalized
                    .expand([C, 1, WEIGHT_SIZE, WEIGHT_SIZE])
                    .set_require_grad(is_required_grad)
            },
            device.to_owned(),
            false,
        );

        Self { filter }
    }
}

impl<B: Backend, const C: usize> Metric<B> for MeanStructuralSimilarity<B, C> {
    /// ## Arguments
    ///
    /// * `value` - The input tensor with shape `[N?, C?, H, W]`.
    /// * `target` - The target tensor with shape `[N?, C?, H, W]`.
    ///
    /// ## Returns
    ///
    /// The mean of structural similarity index (MSSIM) with shape `[1]`.
    ///
    /// ## Details
    ///
    /// * The argument value should range from `0.0` to `1.0`
    /// * The result value ranges from `-1.0` to `1.0`
    fn evaluate<const D: usize>(
        &self,
        value: Tensor<B, D>,
        target: Tensor<B, D>,
    ) -> Tensor<B, 1> {
        const K1: f64 = 0.01;
        const K2: f64 = 0.03;
        const L: f64 = 1.0;
        const C1: f64 = (K1 * L) * (K1 * L);
        const C2: f64 = (K2 * L) * (K2 * L);
        const FRAC_C1_2: f64 = C1 / 2.0;
        const FRAC_C2_2: f64 = C2 / 2.0;

        let input = (
            value.unsqueeze::<4>().expand([-1, C as i64, -1, -1]),
            target.unsqueeze::<4>().expand([-1, C as i64, -1, -1]),
        );

        let input_0_shape = input.0.shape().dims;
        let input_1_shape = input.1.shape().dims;
        if input_0_shape != input_1_shape {
            Err::<(), _>(
                Error::MismatchedTensorShape(input_0_shape, input_1_shape)
                    .to_string(),
            )
            .expect("This is an internal error");
        }

        // F(x) = sum(w[G, G] * x[H, W])
        let filter = &self.filter;
        // μ0 = F(x0)
        // μ1 = F(x1)
        let mean = (
            filter.forward(input.0.to_owned()),
            filter.forward(input.1.to_owned()),
        );
        // μ0^2 = μ0 * μ0
        // μ1^2 = μ1 * μ1
        let mean2 = (
            mean.0.to_owned().powf_scalar(2.0),
            mean.1.to_owned().powf_scalar(2.0),
        );
        // σ0^2 = F(x0^2) - μ0^2
        let std2 = (
            filter
                .forward(input.0.to_owned().powf_scalar(2.0))
                .sub(mean2.0.to_owned()),
            filter
                .forward(input.1.to_owned().powf_scalar(2.0))
                .sub(mean2.1.to_owned()),
        );
        // μ01 = μ0 * μ1
        let mean_01 = mean.0.mul(mean.1);
        // σ01 = F(x0 * x1) - μ01
        let std_01 =
            filter.forward(input.0.mul(input.1)).sub(mean_01.to_owned());
        // I(x0, x1) = (2 * μ01 + C1) * (2 * σ01 + C2) /
        //             ((μ0^2 + μ1^2 + C1) * (σ0^2 + σ1^2 + C2))
        let indexes = (mean_01 + FRAC_C1_2) * (std_01 + FRAC_C2_2) * 4.0
            / ((mean2.0 + mean2.1 + C1) * (std2.0 + std2.1 + C2));
        // MI(x0, x1) = mean(I(x0, x1))
        indexes.mean()
    }
}

impl<B: Backend, const C: usize> Default for MeanStructuralSimilarity<B, C> {
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
        let metric = MeanStructuralSimilarity::<NdArray<f32>, 3>::init(&device);

        let input_0 = Tensor::<NdArray<f32>, 4>::zeros([1, 3, 32, 32], &device);
        let input_1 = Tensor::zeros([1, 3, 32, 32], &device);
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert_eq!(score, 1.0);

        let input_0 = Tensor::<NdArray<f32>, 4>::ones([1, 3, 32, 32], &device);
        let input_1 = Tensor::ones([1, 3, 32, 32], &device);
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert_eq!(score, 1.0);

        let input_0 = Tensor::<NdArray<f32>, 4>::zeros([1, 3, 32, 32], &device);
        let input_1 = Tensor::ones([1, 3, 32, 32], &device);
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert!(score > 0.0 && score < 1e-4, "score: {:?}", score);

        let input_0 = Tensor::<NdArray<f32>, 4>::random(
            [1, 3, 32, 32],
            Distribution::Uniform(0.01, 0.99),
            &device,
        );
        let input_1 = input_0.to_owned().neg().add_scalar(1.0);
        let score = metric.evaluate(input_0, input_1).into_scalar();
        assert!(score < 0.0, "score: {:?}", score);
    }

    #[test]
    fn default() {
        use super::*;
        use burn::backend::NdArray;

        MeanStructuralSimilarity::<NdArray<f32>, 3>::default();
    }
}

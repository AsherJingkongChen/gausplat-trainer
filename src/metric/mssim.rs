pub use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

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
/// - `self.filter.weight`: `[C, 1, 11, 11]`
///   - A normalized gaussian filter
///
#[derive(Debug, Module)]
pub struct MeanStructuralSimilarity<B: Backend, const C: usize> {
    filter: conv::Conv2d<B>,
}

impl<B: Backend, const C: usize> MeanStructuralSimilarity<B, C> {
    pub fn init(device: &B::Device) -> Self {
        // G
        const WEIGHT_SIZE: usize = 11;
        // G / 2
        const WEIGHT_SIZE_HALF: usize = WEIGHT_SIZE >> 1;
        const WEIGHT_STD: f64 = 1.5;
        const WEIGHT_STD2_2: f64 = 2.0 * WEIGHT_STD * WEIGHT_STD;

        let padding = WEIGHT_SIZE_HALF;
        let mut filter = conv::Conv2dConfig::new([C; 2], [WEIGHT_SIZE; 2])
            .with_bias(false)
            .with_groups(C)
            .with_padding(PaddingConfig2d::Explicit(padding, padding))
            .init(device);

        // [C, 1, G, G]
        filter.weight = Param::uninitialized(
            Default::default(),
            move |device, is_required_grad| {
                // G / 2
                let size_half = WEIGHT_SIZE_HALF as i64;
                // 2s^2
                let s2_2 = WEIGHT_STD2_2;
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
                // w[G, G] = exp(-(x^2 + y^2) / 2s^2)[G, G]
                let w = x2_y2_n.div_scalar(s2_2).exp();
                // w'[G, G] = w[G, G] / sum(w)[1, 1]
                let w_normalized = w.to_owned().div(w.sum().unsqueeze::<2>());

                // w'[C, 1, G, G]
                w_normalized
                    .expand([C, 1, WEIGHT_SIZE, WEIGHT_SIZE])
                    .set_require_grad(is_required_grad)
            },
            device.to_owned(),
            true,
        );

        Self { filter }
    }

    /// Computing the mean of structural similarity index (MSSIM) between the inputs
    /// using the equations 13-16 and settings in the paper:
    ///
    /// *Wang, J., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4), 600–612.*
    /// https://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf
    ///
    /// ## Details
    ///
    /// - `(input_0, input_1)`: `([n, C, h, w], [n, C, h, w])`
    ///   - The values are expected to fall within the range of `0.0` to `1.0`
    /// - Return: `[1]`
    ///
    pub fn forward(
        &self,
        input_0: Tensor<B, 4>,
        input_1: Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        const K1: f64 = 0.01;
        const K2: f64 = 0.03;
        const L: f64 = 1.0;
        const C1: f64 = (K1 * L) * (K1 * L);
        const C2: f64 = (K2 * L) * (K2 * L);

        debug_assert_eq!(input_0.dims(), input_1.dims());
        debug_assert_eq!(input_0.dims()[1], C);
        debug_assert_eq!(input_1.dims()[1], C);

        let input = (input_0, input_1);
        // F(x) = sum(weight * x)
        let filter = &self.filter;
        // m0 = F(x0)
        // m1 = F(x1)
        let mean = (
            filter.forward(input.0.to_owned()),
            filter.forward(input.1.to_owned()),
        );
        // m0^2 = m0 * m0
        // m1^2 = m1 * m1
        let mean2 = (
            mean.0.to_owned().powf_scalar(2.0),
            mean.1.to_owned().powf_scalar(2.0),
        );
        // s0^2 = F(x0^2) - m0^2
        let std2 = (
            filter
                .forward(input.0.to_owned().powf_scalar(2.0))
                .sub(mean2.0.to_owned()),
            filter
                .forward(input.1.to_owned().powf_scalar(2.0))
                .sub(mean2.1.to_owned()),
        );
        // m_01 = m0 * m1
        let mean_01 = mean.0 * mean.1;
        // s_01 = F(x0 * x1) - m_01
        let std_01 = filter.forward(input.0 * input.1) - mean_01.to_owned();
        // I(x0, x1) =
        // (2 * m_01 + C1) * (2 * s_01 + C2) /
        // ((m0^2 + m1^2 + C1) * (s0^2 + s1^2 + C2))
        let indexes = (mean_01 + C1 / 2.0) * (std_01 + C2 / 2.0) * (2.0 * 2.0)
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
    fn forward() {
        use super::*;
        use burn::backend::NdArray;

        let device = Default::default();
        let metric = MeanStructuralSimilarity::<NdArray<f32>, 3>::init(&device);

        let input_0 = Tensor::zeros([1, 3, 256, 256], &device);
        let input_1 = Tensor::zeros([1, 3, 256, 256], &device);
        let score = metric.forward(input_0, input_1).into_scalar();
        assert_eq!(score, 1.0);

        let input_0 = Tensor::ones([1, 3, 256, 256], &device);
        let input_1 = Tensor::ones([1, 3, 256, 256], &device);
        let score = metric.forward(input_0, input_1).into_scalar();
        assert_eq!(score, 1.0);

        let input_0 = Tensor::zeros([1, 3, 256, 256], &device);
        let input_1 = Tensor::ones([1, 3, 256, 256], &device);
        let score = metric.forward(input_0, input_1).into_scalar();
        assert!(score < 1e-4);
        assert_ne!(score, 0.0);
    }
}
